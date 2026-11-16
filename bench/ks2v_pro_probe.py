"""Kandinsky S2V 1.0 Pro probe (#1339): one still and the narration -> a talking take.

The plan's fallback path. S2V's text/audio encoder step takes the speech track as a
16 kHz mono float waveform (`encoders.py`, the `audio` type check), and dw's argument
realization only decodes image and video media (`dw/argument_media.py`), so an
`asset:` wav reaches the pipeline as a path and fails there. Onboarding needs an
`audio` media branch; until then the probe runs here, in lem's `~/spikes/ks2v` venv,
and `bench/arms/ks2v-pro-4bit.json` is the shape it takes in dw once that lands.

    python ks2v_pro_probe.py REPO SECONDS SEED OUT.mp4 [--backend sdnq|bnb] [--group-offload]

The transformer loads 4-bit: SDNQ int4 first and bnb nf4 if SDNQ fails to load. The
text/audio encoder is offloaded by the components manager and the VAE decodes tiled.
On a CUDA OOM the probe re-runs itself once with the transformer group-offloaded,
then records the failure and stops. The resolution is never lowered.

Every line it prints that the record needs starts with `RECORD`.
"""

import argparse
import math
import os
import sys
import time

import av
import numpy as np
import torch
from PIL import Image

WORKSPACE = "/home/don/diffusers-workspace/constitution-walk-3/assets"
KEYFRAME = f"{WORKSPACE}/uploads/bench/keyframes-v2/kf00.jpg"
NARRATION = f"{WORKSPACE}/bench/constitution-narration-retimed.wav"
WIDTH, HEIGHT, FPS, CHUNK_SECONDS = 768, 512, 25, 5

# Embeddings and modulation stay bf16/fp32: they are small, and the checkpoint keeps
# the time embedding and every modulation in fp32 (`_keep_in_fp32_modules`)
SDNQ_SKIP = [
    "time_embeddings",
    "text_embeddings",
    "pooled_text_embeddings",
    "audio_embeddings",
    "visual_embeddings",
    "mask_embeddings",
    "modulation",
]


def record(*parts):
    print("RECORD", *parts, flush=True)


def load_track(seconds, rate):
    container = av.open(NARRATION)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
    samples = np.concatenate(
        [
            out.to_ndarray()[0]
            for frame in container.decode(audio=0)
            for out in resampler.resample(frame)
        ]
    )
    return samples[: int(seconds * rate)]


def quantization_config(backend):
    if backend == "sdnq":
        from sdnq import SDNQConfig

        return SDNQConfig(
            weights_dtype="int4",
            quantization_device="cuda",
            return_device="cpu",
            use_quantized_matmul=True,
            dequantize_fp32=False,
            modules_to_not_convert=SDNQ_SKIP,
        )
    from diffusers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=SDNQ_SKIP,
    )


def build(repo, backend, group_offload):
    from diffusers import ComponentsManager, ModularPipeline

    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(repo, components_manager=manager)
    pipe.load_components(
        dtype=torch.bfloat16,
        quantization_config={"transformer": quantization_config(backend)},
    )
    if group_offload:
        # The transformer streams its blocks from CPU and the encoders stream their
        # leaves; the components manager's whole-model offload stays off
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        for name in ("text_encoder", "text_encoder_2"):
            component = getattr(pipe, name, None)
            if component is not None:
                component.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                )
        pipe.vae.to("cuda")
    else:
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="6GiB")
    pipe.vae.enable_tiling()
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("seconds", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("out")
    parser.add_argument("--backend", choices=("sdnq", "bnb"), default="sdnq")
    parser.add_argument("--group-offload", action="store_true")
    args = parser.parse_args()

    import diffusers
    from diffusers.utils import encode_video

    record("diffusers", diffusers.__version__, "torch", torch.__version__)
    record("repo", args.repo, "seconds", args.seconds, "seed", args.seed)
    record("size", f"{WIDTH}x{HEIGHT}", "fps", FPS, "group_offload", args.group_offload)

    backend = args.backend
    try:
        pipe = build(args.repo, backend, args.group_offload)
    except torch.OutOfMemoryError:
        raise
    except Exception as error:  # noqa: BLE001 - any SDNQ load failure falls to bnb
        if backend != "sdnq":
            raise
        record("sdnq failed to load:", repr(error)[:400])
        backend = "bnb"
        pipe = build(args.repo, backend, args.group_offload)
    record("backend loaded", backend)
    record("components", list(pipe.components))

    image = Image.open(KEYFRAME).convert("RGB")
    audio, speech = load_track(args.seconds, 16000), load_track(args.seconds, 48000)
    num_chunks = math.ceil(args.seconds / CHUNK_SECONDS)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    video = pipe(
        image=image,
        audio=audio,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        num_chunks=num_chunks,
        generator=torch.Generator().manual_seed(args.seed),
        output="videos",
    )[0]
    minutes = (time.time() - started) / 60
    record("frames", len(video), "size", video[0].size, "chunks", num_chunks)
    record("minutes", round(minutes, 1), "per chunk", round(minutes / num_chunks, 2))
    record(
        "peak GiB allocated",
        round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "reserved",
        round(torch.cuda.max_memory_reserved() / 2**30, 1),
    )
    # Muxes the 48 kHz narration under the take - pair_audio's job in the dw arm
    encode_video(
        video,
        fps=FPS,
        output_path=args.out,
        audio=torch.from_numpy(speech)[None],
        audio_sample_rate=48000,
    )
    record("wrote", args.out)


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as error:
        record("OOM:", str(error).splitlines()[0])
        if "--group-offload" in sys.argv:
            record("OOM again with the transformer group-offloaded; stopping")
            sys.exit(2)
        record("retrying once with the transformer group-offloaded")
        os.execv(sys.executable, [sys.executable, *sys.argv, "--group-offload"])
