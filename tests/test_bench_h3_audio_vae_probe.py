"""bench/h3_audio_vae_probe.py: cut selection, token-drop arithmetic, the error
metrics, run length, the bar, and the input-check exits - with a fake
context-sensitive encoder, so no model loads."""

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import h3_audio_vae_probe as probe  # noqa: E402

HOP = probe.HOP
SLICE = 200  # 5 s
TOTAL = 600  # 15 s


# --- cut selection ---


def flat_rms(total=TOTAL, loud_at=()):
    rms = [0.1] * total
    for start in loud_at:
        rms[start : start + 20] = [1.0] * 20
    return rms


def test_tokens_are_whole_hops():
    assert HOP == 800
    assert probe.tokens(2.0) == 80
    assert probe.token_seconds(10) == 0.25


def test_cuts_are_first_zero_and_last_ends_at_song_end():
    cuts = probe.choose_cuts(flat_rms(), SLICE)
    assert all(isinstance(c, int) for c in cuts)
    assert len(cuts) == 5
    assert cuts[0] == 0
    assert cuts[-1] == TOTAL - SLICE


def test_interior_cuts_pick_the_loudest_head_in_their_share():
    cuts = probe.choose_cuts(flat_rms(loud_at=(100, 200, 350)), SLICE)
    assert cuts[1:4] == [100, 200, 350]


def test_interior_cuts_keep_two_seconds_before_them():
    # a loud spot at token 10 is inside the first 2 s: it must not be chosen
    cuts = probe.choose_cuts(flat_rms(loud_at=(10,)), SLICE)
    assert all(c >= probe.tokens(2.0) for c in cuts[1:-1])


def test_at_tokens_override_the_interior_cuts():
    assert probe.choose_cuts(flat_rms(), SLICE, 5, [90, 150, 300]) == [
        0,
        90,
        150,
        300,
        400,
    ]


def test_wrong_at_count_raises():
    with pytest.raises(ValueError, match="--at"):
        probe.choose_cuts(flat_rms(), SLICE, 5, [90, 150])


def test_song_no_longer_than_a_slice_raises():
    with pytest.raises(ValueError):
        probe.choose_cuts(flat_rms(total=SLICE), SLICE)


# --- token-drop arithmetic ---


def test_arm_window_counts():
    w = probe.arm_window(100, SLICE, 40, 8, TOTAL)
    assert (w.start, w.pre, w.slice, w.post) == (60, 40, SLICE, 8)
    assert w.num_latents == 40 + SLICE + 8
    assert w.post_silent == 0


def test_keep_drops_exactly_the_context_tokens():
    w = probe.arm_window(100, SLICE, 40, 8, TOTAL)
    latents = np.arange(2 * w.num_latents * 3).reshape(2, w.num_latents, 3)
    kept = w.keep(latents)
    assert kept.shape == (2, SLICE, 3)
    assert np.array_equal(kept, latents[:, 40 : 40 + SLICE])


def test_pre_longer_than_the_cut_is_not_available():
    assert probe.arm_window(0, SLICE, 20, 0, TOTAL) is None
    assert probe.arm_window(39, SLICE, 40, 0, TOTAL) is None
    assert probe.arm_window(40, SLICE, 40, 0, TOTAL) is not None


def test_post_silent_counts_tokens_past_the_song_end():
    assert probe.arm_window(400, SLICE, 0, 8, TOTAL).post_silent == 8
    assert probe.arm_window(395, SLICE, 0, 8, TOTAL).post_silent == 3
    assert probe.arm_window(300, SLICE, 0, 8, TOTAL).post_silent == 0


def test_window_samples_slices_on_hop_boundaries():
    wave_ = np.arange(2 * TOTAL * HOP).reshape(2, TOTAL * HOP)
    w = probe.arm_window(100, SLICE, 40, 8, TOTAL)
    got = probe.window_samples(wave_, w)
    assert got.shape == (2, w.num_latents * HOP)
    assert np.array_equal(got, wave_[:, 60 * HOP : (60 + w.num_latents) * HOP])


def test_window_samples_is_short_past_the_song_end():
    wave_ = np.zeros((2, TOTAL * HOP))
    w = probe.arm_window(400, SLICE, 0, 8, TOTAL)
    assert probe.window_samples(wave_, w).shape == (2, SLICE * HOP)


# --- a fake context-sensitive encoder, end to end ---

CAUSAL = 4  # hops of look-back; the first CAUSAL - 1 tokens of an encode are off
LOOKAHEAD = 2  # hops of look-ahead; the last LOOKAHEAD tokens are off


def fake_encode(samples, num_latents):
    """Pad with zeros to num_latents hops (as encode_audio_span does), then a
    causal running mean plus a look-ahead mean per hop: `(channels, n, 2)`."""
    padded = np.zeros((samples.shape[0], num_latents * HOP))
    padded[:, : samples.shape[1]] = samples
    f = padded.reshape(samples.shape[0], num_latents, HOP).mean(axis=2)
    ext = np.pad(f, ((0, 0), (CAUSAL - 1, LOOKAHEAD)))
    causal = sum(
        ext[:, CAUSAL - 1 - j : CAUSAL - 1 - j + num_latents] for j in range(CAUSAL)
    )
    look = sum(
        ext[:, CAUSAL - 1 + j : CAUSAL - 1 + j + num_latents]
        for j in range(1, LOOKAHEAD + 1)
    )
    return np.stack([causal / CAUSAL, look / LOOKAHEAD], axis=-1)


@pytest.fixture(scope="module")
def flow():
    rng = np.random.default_rng(0)
    per_hop = 1.0 + rng.random(TOTAL)
    song = np.repeat(per_hop, HOP)[None].repeat(2, axis=0)
    reference = fake_encode(song, TOTAL)
    cut = 200
    truth = reference[:, cut : cut + SLICE]

    def run(pre, post):
        w = probe.arm_window(cut, SLICE, pre, post, TOTAL)
        latents = w.keep(fake_encode(probe.window_samples(song, w), w.num_latents))
        return probe.token_errors(latents, truth)

    hard_errors, hard_cos = run(0, 0)
    floor = probe.floor_of(hard_errors)
    return {
        "floor": floor,
        "hard": probe.arm_metrics(hard_errors, hard_cos, floor),
        "pre": probe.arm_metrics(*run(40, 0), floor),
        "post": probe.arm_metrics(*run(0, 8), floor),
    }


def test_fake_hard_cut_has_a_clean_middle(flow):
    assert flow["floor"] < 1e-9


def test_hard_cut_head_error_is_high_and_context_restores_it(flow):
    assert flow["hard"]["head_0.25"] > 0.05
    assert flow["pre"]["head_0.25"] < 1e-9
    assert flow["hard"]["head_run_tokens"] == CAUSAL - 1


def test_hard_cut_tail_error_is_high_and_post_context_restores_it(flow):
    assert flow["hard"]["tail_0.2"] > 0.05
    assert flow["post"]["tail_0.2"] < 1e-9
    assert flow["hard"]["tail_run_tokens"] == LOOKAHEAD


# --- error metrics and the floor ---


def test_identical_latents_have_zero_error_and_unit_cosine():
    a = np.random.default_rng(1).random((2, 5, 3)) + 0.5
    errors, cosines = probe.token_errors(a, a)
    assert errors == pytest.approx([0.0] * 5)
    assert cosines == pytest.approx([1.0] * 5)


def test_token_errors_known_values():
    arm = np.array([[[3.0, 4.0]]])
    ref = np.array([[[3.0, 0.0]]])
    errors, cosines = probe.token_errors(arm, ref)
    assert errors == pytest.approx([4 / 3])
    assert cosines == pytest.approx([9 / 15])


def test_token_errors_take_both_channels_together():
    arm = np.array([[[1.0]], [[1.0]]])
    ref = np.array([[[1.0]], [[0.0]]])
    errors, _ = probe.token_errors(arm, ref)
    assert errors == pytest.approx([1.0])


def test_floor_is_the_median_of_the_middle():
    errors = [100.0] * 40 + [float(i) for i in range(20)] + [100.0] * 40
    assert probe.floor_of(errors) == 9.5


def test_floor_of_a_too_short_slice_raises():
    with pytest.raises(ValueError, match="too short"):
        probe.floor_of([0.1] * 80)


# --- run length ---


def test_run_from_head_and_tail():
    errors = [0.5, 0.5, 0.3, 0.1, 0.4]
    assert probe.run_tokens(errors, 0.2) == 3
    assert probe.run_tokens(errors, 0.2, from_end=True) == 1


def test_a_token_equal_to_the_threshold_ends_the_run():
    errors = [0.5, 0.5, 0.3, 0.5]
    assert probe.run_tokens(errors, 0.3) == 2
    assert probe.run_tokens(errors, 0.3, from_end=True) == 1


def test_quarter_second_counts_exactly():
    assert probe.lasts_at_least(10, 0.25)
    assert not probe.lasts_at_least(9, 0.25)


# --- the bar ---

FLOOR = 0.5  # 2x floor = 1.0


def hard(head=0, tail=0):
    return {"head_run_tokens": head, "tail_run_tokens": tail}


def head_arm(value):
    return {"head_0.25": value}


def tail_arm(value):
    return {"tail_0.2": value}


def test_head_only_cut_meets():
    v = probe.judge_cut(1, FLOOR, hard(head=12), head_arm(0.9), tail_arm(9.0))
    assert (v.head_a, v.head_b, v.tail_a, v.tail_b) == (True, True, False, None)
    assert v.meets


def test_tail_only_cut_meets_through_the_post_arm():
    v = probe.judge_cut(1, FLOOR, hard(tail=10), head_arm(9.0), tail_arm(0.9))
    assert (v.head_a, v.head_b, v.tail_a, v.tail_b) == (False, None, True, True)
    assert v.meets


def test_tail_failing_half_b_does_not_meet():
    v = probe.judge_cut(1, FLOOR, hard(tail=10), head_arm(0.0), tail_arm(1.5))
    assert v.tail_a and v.tail_b is False
    assert not v.meets


def test_zero_second_cut_cannot_meet_through_the_head():
    v = probe.judge_cut(0, FLOOR, hard(head=12), None, tail_arm(9.0))
    assert v.head_a and v.head_b is False
    assert "no 1 s" in v.head_why
    assert not v.meets


def test_two_times_the_floor_is_inclusive():
    assert probe.judge_cut(1, FLOOR, hard(head=12), head_arm(1.0), tail_arm(9.0)).meets
    assert not probe.judge_cut(
        1, FLOOR, hard(head=12), head_arm(1.01), tail_arm(9.0)
    ).meets


def test_short_runs_fail_half_a():
    v = probe.judge_cut(1, FLOOR, hard(head=9, tail=9), head_arm(0.0), tail_arm(0.0))
    assert not v.head_a and not v.tail_a and not v.meets


def verdicts(meeting, total=5):
    good = probe.judge_cut(1, FLOOR, hard(head=12), head_arm(0.0), tail_arm(0.0))
    bad = probe.judge_cut(1, FLOOR, hard(), head_arm(0.0), tail_arm(0.0))
    return [good] * meeting + [bad] * (total - meeting)


def test_three_of_five_meets_the_bar():
    assert probe.bar_met(verdicts(3))


def test_two_of_five_does_not_meet_the_bar():
    assert not probe.bar_met(verdicts(2))


def test_verdict_line_ends_with_the_bar_state():
    starts = [0.0, 5.0, 6.0, 7.0, 10.0]
    assert probe.verdict_line(verdicts(3), starts).endswith("bar MET")
    assert probe.verdict_line(verdicts(2), starts).endswith("bar NOT MET")


# --- input checks ---


def write_wav(path, seconds, rate=32000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def test_check_inputs_boundary(tmp_path):
    song = tmp_path / "song.wav"
    write_wav(song, 0.1)
    assert probe.check_inputs(str(song), 7.0, 5.0) is None
    assert "too short" in probe.check_inputs(str(song), 6.999, 5.0)


def test_check_inputs_names_a_missing_file(tmp_path):
    missing = str(tmp_path / "nope.wav")
    assert missing in probe.check_inputs(missing, None, 5.0)


def test_main_exits_non_zero_on_a_missing_song(tmp_path, capsys):
    missing = str(tmp_path / "nope.wav")
    assert probe.main([missing]) != 0
    assert missing in capsys.readouterr().err


def test_main_exits_non_zero_on_a_short_song(tmp_path, capsys):
    pytest.importorskip("av")
    song = tmp_path / "short.wav"
    write_wav(song, 3.0)
    assert probe.main([str(song), "--slice-seconds", "5"]) != 0
    err = capsys.readouterr().err
    assert "3.000" in err and "7.000" in err
