"""LongCat-Video-Avatar 1.6 INT8 fit/take driver for #1465 (outside dw).

Mirrors LongCat-Video/run_demo_avatar_single_audio_to_video.py (commit 58eb336e) with
the vendor modules unmodified, plus sequential placement so it fits one 24 GB card:
UMT5 runs on the GPU once to encode the prompt, then stays on the CPU; Whisper runs on
the GPU for the audio embedding, then is freed; then the INT8 DiT, both mix LoRAs and
the VAE go to the GPU for the windows. With --offload blocks (what fitted: the record,
`bench/records/longcat-avatar.md`) dit.blocks stay pinned on the CPU and each block is
swapped onto the GPU only for its own forward. Prints RECORD lines for the bench record.

usage: python -I probe.py --image kf00.jpg --audio narration.wav --seconds 30 --out take
           --offload blocks --prompt "a woman speaking to camera"
"""

import argparse
import datetime
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")

import numpy as np
import PIL.Image
import psutil
import soundfile as sf
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "LongCat-Video"))

from audio_separator.separator import Separator  # noqa: E402
from diffusers.utils import load_image  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

from longcat_video.audio_process import (  # noqa: E402
    get_audio_encoder,
    get_audio_feature_extractor,
)
from longcat_video.audio_process.vocal_separator import separate_vocals  # noqa: E402
from longcat_video.context_parallel import context_parallel_util  # noqa: E402
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan  # noqa: E402
from longcat_video.modules.quantization import load_quantized_dit  # noqa: E402
from longcat_video.modules.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.pipeline_longcat_video_avatar import (  # noqa: E402
    LongCatVideoAvatarPipeline,
)

FFMPEG = str(
    next(
        (HERE / "venv/lib/python3.10/site-packages/imageio_ffmpeg/binaries").glob(
            "ffmpeg-*"
        )
    )
)
NEG = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)
GB = 1024**3


def record(**kv):
    print("RECORD " + json.dumps(kv), flush=True)


class Monitor(threading.Thread):
    """Peak host RSS (this process and children) and nvidia-smi used on GPU 0; enforces
    the RSS and extra-window stop lines."""

    def __init__(self, rss_stop_gb, window_stop_s):
        super().__init__(daemon=True)
        self.rss_stop_gb, self.window_stop_s = rss_stop_gb, window_stop_s
        self.peak_rss = 0
        self.peak_smi = 0
        self.window = None  # (index, start) of the running extra window
        self.proc = psutil.Process()

    def rss(self):
        total = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total

    def mem(self):
        """This process's RSS split into anonymous and file-backed (mmap'd weights)."""
        kv = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "RssAnon", "RssFile"):
                    kv[k] = round(int(v.split()[0]) / 1024**2, 1)
        kids = [
            (c.name(), round(c.memory_info().rss / GB, 1))
            for c in self.proc.children(recursive=True)
            if c.is_running()
        ]
        return {"rss_gb": kv, "children": kids}

    def smi(self):
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return int(out) if out.isdigit() else 0

    def run(self):
        while True:
            r = self.rss()
            self.peak_rss = max(self.peak_rss, r)
            self.peak_smi = max(self.peak_smi, self.smi())
            if r / GB > self.rss_stop_gb:
                record(
                    stop="host RSS over limit", rss_gb=round(r / GB, 1), **self.mem()
                )
                os._exit(3)
            if self.window is not None:
                idx, start = self.window
                if time.time() - start > self.window_stop_s:
                    record(
                        stop="extra window over limit",
                        window=idx,
                        seconds=round(time.time() - start),
                    )
                    os._exit(4)
            time.sleep(1.0)


def stream_blocks(dit, dev, resident):
    """The most aggressive offload: everything but dit.blocks on the GPU, the first
    `resident` blocks too, and every other block's tensors kept pinned on the CPU and
    swapped onto the GPU only for its own forward."""
    for name, child in dit.named_children():
        if name != "blocks":
            child.to(dev)
    for p in dit.parameters(recurse=False):
        p.data = p.data.to(dev)
    for b in dit.buffers(recurse=False):
        b.data = b.data.to(dev)
    for i, block in enumerate(dit.blocks):
        if i < resident:
            block.to(dev)
            continue
        tensors = list(block.parameters()) + list(block.buffers())
        pinned = [t.data.pin_memory() for t in tensors]
        for t, c in zip(tensors, pinned):
            t.data = c

        def pre(_m, _a, _kw=None, tensors=tensors, pinned=pinned):
            for t, c in zip(tensors, pinned):
                t.data = c.to(dev, non_blocking=True)

        def post(_m, _a, _o, tensors=tensors, pinned=pinned):
            for t, c in zip(tensors, pinned):
                t.data = c

        block.register_forward_pre_hook(pre)
        block.register_forward_hook(post)


def write_video(frames, fps, wav, seconds, path):
    """Encode RGB frames with the narration slice muxed on (libx264 crf 17, aac)."""
    n, h, w, _ = frames.shape
    cmd = [
        FFMPEG,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(wav),
        "-t",
        f"{seconds:.3f}",
        "-c:v",
        "libx264",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a woman speaking to camera")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resolution", default="480p", choices=["480p"])
    ap.add_argument("--mix_steps", type=int, default=4)
    ap.add_argument("--mix_alpha", type=float, default=1.0)
    ap.add_argument("--offload", default="none", choices=["none", "blocks"])
    ap.add_argument("--resident_blocks", type=int, default=0)
    ap.add_argument("--rss_stop_gb", type=float, default=90)
    ap.add_argument("--window_stop_s", type=float, default=360)
    args = ap.parse_args()

    ckpt = HERE / "weights/LongCat-Video-Avatar-1.6"
    base = HERE / "weights/LongCat-Video"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_type, fps, audio_stride = "avatar-v1.5", 25, 1
    num_frames, num_cond = 93, 13
    steps, tg, ag = 8, 1.0, 1.0  # use_distill for avatar-v1.5
    target = int(round(args.seconds * fps))
    segments = 1 + max(0, math.ceil((target - num_frames) / (num_frames - num_cond)))
    record(
        prompt=args.prompt,
        seed=args.seed,
        seconds=args.seconds,
        segments=segments,
        target_frames=target,
        steps=steps,
        mix_steps=args.mix_steps,
        mix_alpha=args.mix_alpha,
        offload=args.offload,
        resident_blocks=args.resident_blocks,
    )

    mon = Monitor(args.rss_stop_gb, args.window_stop_s)
    mon.start()

    # narration slice (the model's audio and the muxed track)
    audio, sr = sf.read(args.audio, always_2d=True)
    sl = out / "narration-slice.wav"
    sf.write(sl, audio[: int(round(args.seconds * sr))], sr)

    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
    context_parallel_util.init_context_parallel(
        context_parallel_size=1, global_rank=0, world_size=1
    )
    cp_split_hw = context_parallel_util.get_optimal_split(1)
    dev = torch.device("cuda:0")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        base, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        ckpt, subfolder="scheduler", torch_dtype=torch.bfloat16
    )
    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=None,
        audio_encoder=None,
        audio_feature_extractor=None,
        model_type=model_type,
    )
    pipe.device = dev
    record(phase="umt5 loaded on cpu", seconds=round(time.time() - t0), **mon.mem())

    # 1. UMT5 on the GPU once; the embeddings are cached for every window. Then only its
    # config stays (generate_* read text_encoder.config.d_model): sequential loading keeps
    # host RSS to one model at a time.
    text_encoder.to(dev)
    with torch.no_grad():
        pe, pm = pipe._get_t5_prompt_embeds(
            prompt=[args.prompt],
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=dev,
            dtype=torch.bfloat16,
        )
        ne, nm = pipe._get_t5_prompt_embeds(
            prompt=[NEG],
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=dev,
            dtype=torch.bfloat16,
        )
    pipe.text_encoder = types.SimpleNamespace(
        config=text_encoder.config, dtype=torch.bfloat16
    )
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    audio_encoder = get_audio_encoder(str(ckpt / "whisper-large-v3"), model_type)
    pipe.audio_encoder = audio_encoder
    pipe.audio_feature_extractor = get_audio_feature_extractor(
        str(ckpt / "whisper-large-v3"), model_type
    )

    def encode_prompt(
        prompt, negative_prompt=None, do_classifier_free_guidance=True, **_
    ):
        if do_classifier_free_guidance:
            return pe, pm, ne, nm
        return pe, pm, None, None

    pipe.encode_prompt = encode_prompt

    # 2. vocal separation and Whisper on the GPU, then free them
    tmp = out / "audio_temp"
    sep = Separator(
        output_dir=tmp / "vocals",
        output_single_stem="vocals",
        model_file_dir=str(ckpt / "vocal_separator"),
    )
    if sep.onnx_execution_provider == ["CUDAExecutionProvider"]:
        sep.onnx_execution_provider = [
            (
                "CUDAExecutionProvider",
                {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"},
            )
        ]
    sep.load_model("Kim_Vocal_2.onnx")
    gen_dur = num_frames / fps + (segments - 1) * (num_frames - num_cond) / fps
    speech, asr = separate_vocals(sep, str(sl), duration=gen_dur + 1.0)
    del sep
    pad = math.ceil((gen_dur + 1.0 - len(speech) / asr) * asr)
    if pad > 0:
        speech = np.append(speech, [0.0] * pad)
    audio_encoder.to(dev)
    with torch.no_grad():
        full_emb = pipe.get_audio_embedding(
            speech,
            fps=fps * audio_stride,
            device=dev,
            sample_rate=asr,
            model_type=model_type,
            use_loudness_norm=False,
        )
    if torch.isnan(full_emb).any():
        raise ValueError("broken audio embedding with nan values")
    audio_encoder.to("cpu")
    pipe.audio_encoder = None
    del audio_encoder
    gc.collect()
    torch.cuda.empty_cache()
    record(
        phase="prompt and audio encoded",
        seconds=round(time.time() - t0),
        smi_peak_mib=mon.peak_smi,
        **mon.mem(),
    )

    # 3. DiT + LoRAs + VAE on the GPU
    dit = load_quantized_dit(
        str(ckpt), subfolder="base_model_int8", cp_split_hw=cp_split_hw
    )
    dit.load_lora(
        str(ckpt / "loraA/lora.safetensors"),
        "dmd",
        multiplier=1.0,
        lora_network_dim=128,
        lora_network_alpha=64,
    )
    dit.load_lora(
        str(ckpt / "loraB/lora.safetensors"),
        "opsd",
        multiplier=1.0,
        lora_network_dim=128,
        lora_network_alpha=64,
    )
    dit.enable_lora_mix("dmd", "opsd", args.mix_alpha, args.mix_steps)
    record(
        phase="dit loaded on cpu",
        seconds=round(time.time() - t0),
        dit_dtype=str(dit.dtype),
        **mon.mem(),
    )
    if dit.dtype != torch.bfloat16:
        pe, ne = pe.to(dit.dtype), ne.to(dit.dtype)
    pipe.dit = dit
    if args.offload == "blocks":
        stream_blocks(dit, dev, args.resident_blocks)
    else:
        dit.to(dev)
    for lora_network in getattr(dit, "lora_dict", {}).values():
        for lora in lora_network.loras:
            lora.to(dev)
    vae.to(dev)
    torch.cuda.synchronize()
    record(
        phase="dit on gpu",
        allocated_gb=round(torch.cuda.memory_allocated() / GB, 2),
        smi_mib=mon.smi(),
    )

    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    idx = torch.arange(5) - 2

    def emb_at(start):
        c = torch.arange(start, start + audio_stride * num_frames, audio_stride)
        c = torch.clamp(c.unsqueeze(1) + idx.unsqueeze(0), 0, full_emb.shape[0] - 1)
        return full_emb[c][None, ...].to(dev)

    image = load_image(args.image)
    times = []
    try:
        torch.cuda.reset_peak_memory_stats()
        ts = time.time()
        output, latent = pipe.generate_ai2v(
            image=image,
            prompt=args.prompt,
            negative_prompt=NEG,
            resolution=args.resolution,
            num_frames=num_frames,
            num_inference_steps=steps,
            text_guidance_scale=tg,
            audio_guidance_scale=ag,
            output_type="both",
            generator=gen,
            audio_emb=emb_at(0),
            use_distill=True,
        )
        torch.cuda.synchronize()
        times.append(time.time() - ts)
        frames = (output[0] * 255).astype(np.uint8)
        h, w = frames.shape[1:3]
        record(
            window=1,
            seconds=round(times[-1], 1),
            max_allocated_gb=round(torch.cuda.max_memory_allocated() / GB, 2),
            smi_peak_mib=mon.peak_smi,
            rss_peak_gb=round(mon.peak_rss / GB, 1),
            frame_size=f"{w}x{h}",
            fps=fps,
        )
        total = len(frames) + (segments - 1) * (num_frames - num_cond)
        allf = np.empty((total, h, w, 3), np.uint8)
        allf[: len(frames)] = frames
        count = len(frames)
        ref_latent = latent[:, :, :1].clone()
        current = [PIL.Image.fromarray(f) for f in frames]
        start = 0
        for seg in range(1, segments):
            start += audio_stride * (num_frames - num_cond)
            torch.cuda.reset_peak_memory_stats()
            ts = time.time()
            mon.window = (seg + 1, ts)
            output, latent = pipe.generate_avc(
                video=current,
                video_latent=latent,
                prompt=args.prompt,
                negative_prompt=NEG,
                height=h,
                width=w,
                num_frames=num_frames,
                num_cond_frames=num_cond,
                num_inference_steps=steps,
                text_guidance_scale=tg,
                audio_guidance_scale=ag,
                generator=gen,
                output_type="both",
                use_kv_cache=args.mix_steps == 0,
                offload_kv_cache=False,
                enhance_hf=False,
                audio_emb=emb_at(start),
                ref_latent=ref_latent,
                ref_img_index=10,
                mask_frame_range=0,
                use_distill=True,
            )
            torch.cuda.synchronize()
            mon.window = None
            times.append(time.time() - ts)
            new = (output[0] * 255).astype(np.uint8)
            add = new[num_cond:]
            allf[count : count + len(add)] = add
            count += len(add)
            current = [PIL.Image.fromarray(f) for f in new]
            record(
                window=seg + 1,
                seconds=round(times[-1], 1),
                max_allocated_gb=round(torch.cuda.max_memory_allocated() / GB, 2),
                smi_peak_mib=mon.peak_smi,
                rss_peak_gb=round(mon.peak_rss / GB, 1),
            )
    except torch.cuda.OutOfMemoryError as e:
        record(
            stop="OOM at 480p",
            windows_done=len(times),
            max_allocated_gb=round(torch.cuda.max_memory_allocated() / GB, 2),
            smi_peak_mib=mon.peak_smi,
            error=str(e)[:300],
        )
        sys.exit(2)

    allf = allf[: min(count, target)]
    take = out / "take.mp4"
    write_video(allf, fps, sl, args.seconds, take)

    PIL.Image.fromarray(allf[int(round(0.5 * fps))]).save(out / "ref-0.5s.png")
    record(
        done=str(take),
        frames=len(allf),
        seconds_out=round(len(allf) / fps, 2),
        first_window_s=round(times[0], 1),
        extra_window_mean_s=round(float(np.mean(times[1:])), 1)
        if len(times) > 1
        else None,
        extra_window_max_s=round(max(times[1:]), 1) if len(times) > 1 else None,
        smi_peak_mib=mon.peak_smi,
        rss_peak_gb=round(mon.peak_rss / GB, 1),
        wall_s=round(time.time() - t0),
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
