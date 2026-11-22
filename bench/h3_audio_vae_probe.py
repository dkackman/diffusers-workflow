"""MiniMax-H3 audio-VAE context probe (#1516 stage A, #1544): does a hard-cut held
slice encode wrong at its head or tail, and does context bring it back?

dw holds a song slice cut hard at both ends (`DwH3HoldAudioStep`). The H3 audio
VAE's first tokens (causal attention) and last tokens (convolution lookahead) may
come out differently on a hard cut than inside a continuous encode. This measures
dw's own path - `h3_hold_steps.encode_audio_span` through a
`MiniMaxH3Ref2VAReferenceEncoderStep`, on a waveform normalized the way the hold
step normalizes one (`MiniMaxH3Ref2VASetupStep._normalize_audio_condition`) -
against one continuous encode of the whole song. It loads the audio VAE only.

    python bench/h3_audio_vae_probe.py SONG [--cuts 5] [--slice-seconds 5] [--at S ...]

The song is decoded at its own rate and normalized once, whole, onto the VAE's
32 kHz stereo; every slice is then cut from that waveform on a latent-hop
boundary, so its tokens line up with the continuous encode's.

Cuts: one at 0 s, `cuts - 2` inside the song, one whose slice ends at the song's
last whole token. An interior cut is the loudest slice head (mean RMS of its
first 0.5 s, a proxy for a vocal passage) in its share of the song with at least
2 s before it; `--at` names the interior cut times instead (snapped down to a
hop). Arms per cut: pre-context {0, 0.5, 1, 2} s x post-context {0, 0.2} s; the
context's tokens are dropped after the encode. Post-context past the song's end
is silence. A pre-context longer than the audio before the cut is "n/a: no audio
before", never truncated.

The bar (#1516 plan v4, "How the record reads the bar"): error is per-token
relative L2 against the continuous encode; the floor is the hard-cut arm's median
error from 1 s after the slice start to 1 s before its end. Half (a) at an end:
the hard-cut run strictly above 3x the floor is >= 0.25 s. Half (b) at the head:
>= 1 s of song before the cut and the pre 1 / post 0 arm's first-0.25 s mean
<= 2x the floor; at the tail: the pre 0 / post 0.2 arm's last-0.2 s mean <= 2x the
floor. A cut meets the bar on either end meeting both halves; the bar is met on
>= 3 cuts of 5. Token 0 is reported, never decides.

Every line the record needs starts with `RECORD`.
"""

import argparse
import dataclasses
import os
import statistics
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPO = "MiniMaxAI/MiniMax-H3"
VAE_RATE = 32000
LATENTS_PER_SECOND = 40
HOP = VAE_RATE // LATENTS_PER_SECOND  # samples per latent token
PRE_SECONDS = (0.0, 0.5, 1.0, 2.0)
POST_SECONDS = (0.0, 0.2)
MARGIN_SECONDS = 2.0  # input check: the song outlasts a slice by this much
FLOOR_EDGE_SECONDS = 1.0
RUN_SECONDS = 0.25
HEAD_SECONDS = (0.25, 0.5)
TAIL_SECONDS = 0.2
RUN_FACTOR = 3.0
CONTEXT_FACTOR = 2.0
HEAD_CONTEXT_SECONDS = 1.0
TAIL_CONTEXT_SECONDS = 0.2
CUTS_NEEDED = 3
LOUD_SECONDS = 0.5


def record(*parts):
    print("RECORD", *parts, flush=True)


# --- pure parts (tests/test_bench_h3_audio_vae_probe.py) ---


def tokens(seconds):
    """Whole latent tokens in `seconds`."""
    return int(round(seconds * LATENTS_PER_SECOND))


def token_seconds(count):
    return count / LATENTS_PER_SECOND


def check_inputs(path, length_seconds, slice_seconds):
    """The message to exit on, or None. `length_seconds` is None when the path is
    missing (it is never read)."""
    if not os.path.isfile(path):
        return f"song not found: {path}"
    needed = slice_seconds + MARGIN_SECONDS
    if length_seconds < needed:
        return (
            f"song too short: {path} is {length_seconds:.3f} s, the probe needs at "
            f"least {needed:.3f} s (slice-seconds {slice_seconds:g} + {MARGIN_SECONDS:g} s)"
        )
    return None


def choose_cuts(token_rms, slice_tokens, cuts=5, at_tokens=None):
    """Cut start tokens: 0, `cuts - 2` interior ones, and the one whose slice ends
    at the song's last token. `token_rms` is one loudness value per token of the
    song. Interior cuts keep >= 2 s before them and end >= 1 token before the end
    cut's start; `at_tokens` names them instead."""
    total = len(token_rms)
    end = total - slice_tokens
    if cuts < 2 or end <= 0:
        raise ValueError(
            f"need >= 2 cuts and a song longer than a slice ({total} tokens)"
        )
    interior = cuts - 2
    if at_tokens is not None:
        if len(at_tokens) != interior:
            raise ValueError(
                f"--at names {len(at_tokens)} cuts, the probe needs {interior}"
            )
        picked = list(at_tokens)
    else:
        low, high = tokens(MARGIN_SECONDS), end - 1
        if interior and high <= low:
            raise ValueError("the song leaves no room for an interior cut")
        loud = tokens(LOUD_SECONDS)
        picked = []
        for share in range(interior):
            first = low + (high - low) * share // interior
            last = low + (high - low) * (share + 1) // interior
            candidates = range(first, max(first + 1, last))
            picked.append(
                max(candidates, key=lambda t: sum(token_rms[t : t + loud]) / loud)
            )
    for start in picked:
        if not 0 < start < end:
            raise ValueError(f"cut at token {start} is outside (0, {end})")
    return [0, *picked, end]


@dataclasses.dataclass(frozen=True)
class Window:
    """What one arm encodes, in tokens of the song: `pre` + slice + `post`, of
    which `post_silent` tokens lie past the song's end."""

    start: int  # first token of the encode
    pre: int
    slice: int
    post: int
    post_silent: int

    @property
    def num_latents(self):
        return self.pre + self.slice + self.post

    def keep(self, latents):
        """Drop the context's tokens from `(channels, num_latents, C)` latents."""
        return latents[:, self.pre : self.pre + self.slice]


def arm_window(cut, slice_tokens, pre, post, total):
    """The arm's Window, or None ("n/a: no audio before") when the pre-context is
    longer than the song before `cut`."""
    if pre > cut:
        return None
    silent = max(0, cut + slice_tokens + post - total)
    return Window(cut - pre, pre, slice_tokens, post, min(post, silent))


def window_samples(waveform, window):
    """The `(channels, samples)` audio a Window encodes; past the song's end it is
    short, and `encode_audio_span` pads it with silence."""
    begin = window.start * HOP
    return waveform[..., begin : begin + window.num_latents * HOP]


def token_errors(arm, reference):
    """Per-token relative L2 and cosine of `(channels, n, C)` latents against the
    same tokens of the continuous encode, both channels taken together."""
    errors, cosines = [], []
    for index in range(reference.shape[1]):
        a = arm[:, index].reshape(-1).tolist()
        b = reference[:, index].reshape(-1).tolist()
        diff = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        errors.append(diff / norm_b if norm_b else float("inf"))
        dot = sum(x * y for x, y in zip(a, b))
        cosines.append(dot / (norm_a * norm_b) if norm_a and norm_b else 0.0)
    return errors, cosines


def floor_of(errors):
    """The hard-cut arm's median error, 1 s in from either end of the slice."""
    edge = tokens(FLOOR_EDGE_SECONDS)
    middle = errors[edge : len(errors) - edge]
    if not middle:
        raise ValueError("the slice is too short for a floor (needs > 2 s)")
    return statistics.median(middle)


def run_tokens(errors, threshold, from_end=False):
    """Consecutive tokens from the first (or last) whose error is strictly above
    `threshold`."""
    count = 0
    for error in reversed(errors) if from_end else errors:
        if error <= threshold:
            break
        count += 1
    return count


def lasts_at_least(count, seconds):
    """Whether `count` tokens span >= `seconds`, in whole samples so that exactly
    0.25 s counts."""
    return count * HOP >= round(seconds * VAE_RATE)


def head_mean(errors, seconds):
    return statistics.fmean(errors[: tokens(seconds)])


def tail_mean(errors, seconds):
    return statistics.fmean(errors[-tokens(seconds) :])


def arm_metrics(errors, cosines, floor):
    threshold = RUN_FACTOR * floor
    return {
        "tok0_rel": errors[0],
        "tok0_cos": cosines[0],
        "head_0.25": head_mean(errors, HEAD_SECONDS[0]),
        "head_0.5": head_mean(errors, HEAD_SECONDS[1]),
        "last": errors[-1],
        "tail_0.2": tail_mean(errors, TAIL_SECONDS),
        "head_run": token_seconds(run_tokens(errors, threshold)),
        "tail_run": token_seconds(run_tokens(errors, threshold, from_end=True)),
        "head_run_tokens": run_tokens(errors, threshold),
        "tail_run_tokens": run_tokens(errors, threshold, from_end=True),
    }


@dataclasses.dataclass
class CutVerdict:
    cut: int
    head_a: bool
    head_b: bool | None  # None: not judged (half (a) failed)
    head_why: str
    tail_a: bool
    tail_b: bool | None
    meets: bool


def judge_cut(cut, floor, hard, head_arm, tail_arm):
    """One cut against the bar. `hard` is the hard-cut arm's metrics; `head_arm`
    the pre 1 s / post 0 arm's, None when there is no 1 s of song before the cut;
    `tail_arm` the pre 0 / post 0.2 s arm's."""
    limit = CONTEXT_FACTOR * floor
    head_a = lasts_at_least(hard["head_run_tokens"], RUN_SECONDS)
    head_b, why = None, ""
    if head_a:
        if head_arm is None:
            head_b, why = False, "no 1 s of audio before"
        else:
            head_b = head_arm["head_0.25"] <= limit
    tail_a = lasts_at_least(hard["tail_run_tokens"], RUN_SECONDS)
    tail_b = tail_arm["tail_0.2"] <= limit if tail_a else None
    meets = bool(head_a and head_b) or bool(tail_a and tail_b)
    return CutVerdict(cut, head_a, head_b, why, tail_a, tail_b, meets)


def bar_met(verdicts):
    return sum(v.meets for v in verdicts) >= CUTS_NEEDED


def _yes(value):
    return "-" if value is None else ("yes" if value else "no")


def verdict_line(verdicts, starts_seconds):
    parts = []
    for number, (v, start) in enumerate(zip(verdicts, starts_seconds), 1):
        head_b = _yes(v.head_b) + (f" ({v.head_why})" if v.head_why else "")
        parts.append(
            f"cut {number} @ {start:.3f}s: head a={_yes(v.head_a)} b={head_b}, "
            f"tail a={_yes(v.tail_a)} b={_yes(v.tail_b)}, "
            f"{'meets' if v.meets else 'does not meet'}"
        )
    met = sum(v.meets for v in verdicts)
    return "; ".join(parts) + (
        f"; {met} of {len(verdicts)} cuts meet it: bar "
        f"{'MET' if bar_met(verdicts) else 'NOT MET'}"
    )


def sig(value, figures=4):
    return f"{value:.{figures}g}"


# --- the run ---


def load_song(path):
    """`(channels, samples)` float32 at the file's own rate, and that rate."""
    import av
    import numpy as np

    with av.open(path) as container:
        stream = container.streams.audio[0]
        rate = stream.rate
        layout = "stereo" if stream.channels >= 2 else "mono"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
        chunks = [
            out.to_ndarray()
            for frame in container.decode(stream)
            for out in resampler.resample(frame)
        ]
        chunks += [out.to_ndarray() for out in resampler.resample(None)]
    return np.concatenate(chunks, axis=1).astype("float32"), rate


def commit():
    result = subprocess.run(
        ["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def build():
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(REPO)
    # dw's H3 templates load every component bf16; the audio VAE is never quantized
    pipe.load_components(names=["audio_vae"], dtype=torch.bfloat16)
    pipe.audio_vae.to("cuda")
    return pipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("song")
    parser.add_argument("--cuts", type=int, default=5)
    parser.add_argument("--slice-seconds", type=float, default=5.0)
    parser.add_argument("--at", type=float, nargs="*", default=None)
    args = parser.parse_args(argv)

    # Input checks, before any model loads
    if not os.path.isfile(args.song):
        print(check_inputs(args.song, None, args.slice_seconds), file=sys.stderr)
        return 2
    song, rate = load_song(args.song)
    length = song.shape[-1] / rate
    problem = check_inputs(args.song, length, args.slice_seconds)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the cached weights, no download
    sys.path.insert(0, REPO_ROOT)
    import diffusers
    import torch
    from diffusers.modular_pipelines.minimax_h3.before_encoder import (
        MiniMaxH3Ref2VASetupStep,
    )
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VAReferenceEncoderStep,
    )

    from dw.pipeline_processors.h3_hold_steps import encode_audio_span

    record(
        "commit",
        commit(),
        "diffusers",
        diffusers.__version__,
        "torch",
        torch.__version__,
    )
    record("gpu", torch.cuda.get_device_name(0))
    record(
        "song",
        os.path.abspath(args.song),
        "length",
        f"{length:.3f}s",
        "rate",
        rate,
        "channels",
        song.shape[0],
    )
    record(
        "arguments",
        f"--cuts {args.cuts} --slice-seconds {args.slice_seconds:g}",
        f"--at {args.at}" if args.at else "",
    )

    pipe = build()
    assert pipe.audio_sampling_rate == VAE_RATE, pipe.audio_sampling_rate
    encoder = MiniMaxH3Ref2VAReferenceEncoderStep()
    waveform = MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        torch.from_numpy(song), rate, VAE_RATE, max_duration=length
    )
    total = waveform.shape[-1] // HOP
    slice_tokens = tokens(args.slice_seconds)

    def encode(samples, num_latents):
        return encode_audio_span(encoder, pipe, samples, num_latents).float()

    reference = encode(waveform, total)
    record(
        "continuous encode",
        tuple(reference.shape),
        f"{total} tokens",
        f"hop {HOP} samples = {1000 / LATENTS_PER_SECOND:g} ms",
    )

    rms = (
        waveform[:, : total * HOP]
        .reshape(2, total, HOP)
        .pow(2)
        .mean(dim=(0, 2))
        .sqrt()
        .tolist()
    )
    at = [int(s * LATENTS_PER_SECOND) for s in args.at] if args.at else None
    try:
        cuts = choose_cuts(rms, slice_tokens, args.cuts, at)
    except ValueError as error:
        print(f"cannot place the cuts: {error}", file=sys.stderr)
        return 2
    starts = [token_seconds(c) for c in cuts]
    roles = (
        ["0 s"]
        + ["interior (loudest head)" if at is None else "interior (--at)"]
        * (len(cuts) - 2)
        + ["ends at song end"]
    )

    record(
        "| cut | start s | role | pre s | post s | tok0 rel | tok0 cos | head 0.25 | head 0.5 | last | tail 0.2 | head run s | tail run s |"
    )
    record("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    verdicts = []
    for number, (cut, role) in enumerate(zip(cuts, roles), 1):
        truth = reference[:, cut : cut + slice_tokens]
        results = {}
        hard_window = arm_window(cut, slice_tokens, 0, 0, total)
        hard_errors, _ = token_errors(
            hard_window.keep(
                encode(window_samples(waveform, hard_window), hard_window.num_latents)
            ),
            truth,
        )
        floor = floor_of(hard_errors)
        record(
            f"floor cut {number} @ {starts[number - 1]:.3f}s: {sig(floor)} (3x {sig(RUN_FACTOR * floor)}, 2x {sig(CONTEXT_FACTOR * floor)})"
        )
        for pre_s in PRE_SECONDS:
            for post_s in POST_SECONDS:
                window = arm_window(
                    cut, slice_tokens, tokens(pre_s), tokens(post_s), total
                )
                cells = f"| {number} | {starts[number - 1]:.3f} | {role} | {pre_s:g} | "
                if window is None:
                    record(
                        cells
                        + f"{post_s:g} | n/a: no audio before |  |  |  |  |  |  |  |"
                    )
                    continue
                post_cell = f"{post_s:g}" + (
                    " (silence past song end)" if window.post_silent else ""
                )
                latents = window.keep(
                    encode(window_samples(waveform, window), window.num_latents)
                )
                errors, cosines = token_errors(latents, truth)
                m = arm_metrics(errors, cosines, floor)
                results[(pre_s, post_s)] = m
                record(
                    cells
                    + f"{post_cell} | {sig(m['tok0_rel'])} | {sig(m['tok0_cos'])} | {sig(m['head_0.25'])} | "
                    f"{sig(m['head_0.5'])} | {sig(m['last'])} | {sig(m['tail_0.2'])} | {m['head_run']:.3f} | {m['tail_run']:.3f} |"
                )
        verdicts.append(
            judge_cut(
                cut,
                floor,
                results[(0.0, 0.0)],
                results.get((HEAD_CONTEXT_SECONDS, 0.0)),
                results[(0.0, TAIL_CONTEXT_SECONDS)],
            )
        )

    # Determinism: the second interior cut's pre 1 / post 0 arm, encoded twice more
    cut = cuts[min(2, len(cuts) - 1)]
    window = arm_window(cut, slice_tokens, tokens(1.0), 0, total) or arm_window(
        cut, slice_tokens, 0, 0, total
    )
    samples = window_samples(waveform, window)
    first, second = (window.keep(encode(samples, window.num_latents)) for _ in range(2))
    truth = reference[:, cut : cut + slice_tokens]
    e1, _ = token_errors(first, truth)
    e2, _ = token_errors(second, truth)
    record(
        f"repeat encode cut @ {token_seconds(cut):.3f}s pre {token_seconds(window.pre):g} post 0: max |diff| "
        f"{sig(float((first - second).abs().max()))}; head 0.25 {sig(head_mean(e1, 0.25))} vs {sig(head_mean(e2, 0.25))}; "
        f"tail 0.2 {sig(tail_mean(e1, 0.2))} vs {sig(tail_mean(e2, 0.2))}"
    )
    record("verdict:", verdict_line(verdicts, starts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
