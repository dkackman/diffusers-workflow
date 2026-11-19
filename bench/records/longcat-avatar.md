# LongCat-Video-Avatar 1.6 INT8 probe (2026-11-19, #1465 of #1443) - 30 s gate, seed 1

**Result: it fits one 24 GB 3090 only with the DiT's blocks streamed from pinned host
memory. At that offload, an extra window takes 255 s. The 30 s take fails the gate on
two of four thresholds: SFace to kf00, and 256 px Laplacian sharpness. The 86 s take
was not run.**

Recorded by claude-opus-5-5 (anthropic).

## Setup

All of this ran outside dw, in `lem:~/spikes/longcat-avatar/`, on cuda:0 alone
(`CUDA_VISIBLE_DEVICES=0`), and only while dw-serve had no queued or running job.

- **Preflight:** 626G free on `/` at 2026-11-19 12:00:46 UTC (`preflight-df.txt`), above
  the 60 GB floor. After the download, 572G was free. Nothing was deleted.
- **Vendor code:** `meituan-longcat/LongCat-Video` at
  `58eb336edc5c8c63389838727e40b8c18434e54e` (`LongCat-Video/`), unmodified.
- **Venv:** `venv/`, a fresh uv venv.
  - Python 3.10.22, torch 2.6.0+cu124, diffusers 0.35.1, transformers 4.41.0.
  - flash_attn 2.7.4.post1 came from the prebuilt wheel. No CUDA extension was built from
    source, so the build-failure stop never came into play.
  - setuptools 84.2.1 was added after the first demo run. triton imports it, and a uv
    venv doesn't ship it.
- **Weights:** about 46 GB, downloaded with `hf download --include` (`dl.sh`), with HF_HOME
  under the spike dir.

  | repo @ revision | part | size |
  |---|---|---|
  | `meituan-longcat/LongCat-Video-Avatar-1.6` @ `acfc6282…` | `base_model_int8` | 15G |
  | | `loraA` (dmd) | 2.4G |
  | | `loraB` (opsd) | 2.4G |
  | | `whisper-large-v3` | 2.9G |
  | | `vocal_separator` | 64M |
  | | config, scheduler | <1M |
  | `meituan-longcat/LongCat-Video` @ `03b55529…` | `text_encoder` (UMT5, fp32 on disk) | 22G |
  | | `vae` | 485M |
  | | `tokenizer` | 21M |

  The plan estimated about 36 GB. The difference is the fp32 UMT5.
- **Driver:** `bench/longcat_avatar_probe.py`, run as `probe.sh <log> <args>`, which
  calls `venv/bin/python -I` under `/usr/bin/time -v` with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  - It mirrors the vendor's `run_demo_avatar_single_audio_to_video.py` using the vendor
    modules: avatar-v1.6, INT8 DiT, distill, 8 steps, guidance 1.0/1.0, mix_steps 4 with
    mix_alpha 1.0, 480p bucket, 93-frame windows, 13 conditioning frames, 25 fps.
  - Its stops are: OOM, host RSS over 90 GB (summed over the process and its children),
    and an extra window over 360 s.

## Fit

| run | placement | outcome |
|---|---|---|
| vendor demo (`demo.sh`, torchrun, vendor sample `single_example_1.json`) | the vendor's own: whole pipe `.to(cuda)` | **OOM at load**, in `pipe.to(cuda)`: 22.81 GiB allocated, nvidia-smi peak 24107 MiB; max RSS 85.6 GiB; 2:02 wall |
| probe 1, vendor sample 7 s | every model built on the CPU at once | **host-RSS stop** at 90.9 GB summed (time -v max RSS 85.7 GB); exit 3 before any window |
| probe 2, vendor sample 7 s | sequential: UMT5 on GPU to encode then freed; Whisper on GPU to embed then freed; INT8 DiT + LoRAs + VAE resident | **OOM in window 1**: the DiT on the GPU took 17.57 GB allocated (smi 18661 MiB); max allocated 22.4 GB when it tried to allocate 576 MiB more |
| **probe 3, vendor sample 7 s** | sequential, plus `--offload blocks`: `dit.blocks` pinned on the CPU, each block swapped onto the GPU for its own forward; the rest of the DiT, the LoRAs and the VAE resident | **fits.** 175 frames, 768x512, 25 fps |
| **30 s take on kf00** | same as probe 3 | **fits.** 750 frames, **832x480, 25 fps** (native) |

Probe 3 is the most aggressive offload the plan's OOM stop refers to. It is the first that
fitted, so the stop did not fire.

Measured on the kf00 30 s take (`probe-kf00-30s.log`):

| | |
|---|---|
| peak VRAM | `max_memory_allocated` 10.25 GB; nvidia-smi peak 13473 MiB |
| host RSS | 80.2 GB max (time -v), 83.8 GB summed over the process tree. It sits close to the 90 GB stop because the pinned DiT and the CPU copies sit side by side during placement |
| prompt + audio phase | nvidia-smi peak 11.7 GB |
| load to first window | about 2 min: UMT5 loaded 18 s, encoded 35-45 s, DiT loaded 117-131 s |
| first window (93 frames, 3.72 s) | 255.9 s |
| each extra window (80 frames, 3.2 s) | 255.4 s mean, 255.6 s max (stop: 360 s) |
| 30 s take | 10 windows, 45:08 wall |

On the vendor sample the windows ran 249.5 s, then 247.4 and 247.2 s.

That rate is about 80 s of GPU per second of video. An 86 s take would be 27 windows,
about 1 h 55 min.

The audio separator's onnxruntime CUDA provider failed to load (`libcudnn_adv.so.9`
missing) and ran on the CPU. It is not in the timed windows.

## 30 s take

- **Inputs:**
  - the still: kf00 (`asset:uploads/bench/keyframes-v2/kf00.jpg`);
  - the narration: the first 30.0 s of `asset:bench/constitution-narration-retimed.wav`;
  - seed 1, 480p;
  - prompt **"a woman speaking to camera"**.
- The narration slice is muxed onto the take as AAC.

Measured with `bench/ks2v_gate.py --ref kf00 --crop256 --windows 0-10,20-30`, with both
takes in one run (`out/kf00-30s/gate.txt`). The H3 keeper is job 514b463a855f, run
`20261110-230854-abc647e9`.

| at 0.5/5/10/15/20/25/29.5 s | H3 moving keystone s1 (keeper) | **LongCat-Avatar 1.6 INT8** |
|---|---|---|
| SFace to its own 0.5 s face | 1/.87/.93/.95/.92/.88/.94 | **1/.84/.89/.81/.88/.91/.90** |
| SFace to kf00 | .93/.86/.92/.95/.93/.87/.91 | **.93/.83/.89/.80/.90/.86/.89** |
| saturation | 105.3/104.8/107.1/100.1/110.0/109.3/107.3 | **94.8/93.0/91.5/87.6/98.7/89.9/95.2** |
| face-crop Laplacian, native scale | 312/208/210/308/343/304/260 | **560/174/379/372/700/438/251** |
| face-crop Laplacian, 256 px | 487/303/367/504/545/415/505 | **265/102/209/199/367/249/141** |
| still-mouth bins, 0-30 s | 0 of 60 | **0 of 62** |

| 10 s windows (box crop: 256 px Laplacian / saturation / SFace to the 0.5 s face) | H3 keeper | **LongCat-Avatar** |
|---|---|---|
| 0-10 s | 448 / 101.6 / .94 | **273 / 93.9 / .89** |
| 20-30 s | 537 / 106.1 / .93 | **254 / 94.6 / .90** |

The native-scale Laplacian runs high, but the 256 px figure runs low. At 832x480 the face
crop is small, so its native variance counts pixel-level edges. Scaled to the common
256 px, the face carries about half the H3 keeper's detail.

## Gate (edges inclusive)

| threshold | LongCat-Avatar at 30 s | |
|---|---|---|
| SFace to kf00 ≥ 0.88 at every sample | .83 at 5 s, .80 at 15 s, .86 at 25 s | **fails** |
| saturation rise from 0.5 s ≤ 5 | 94.8 → 98.7 at most, +3.9 | passes |
| 256 px Laplacian at 20-30 s ≥ 410 | 254 | **fails** |
| still-mouth bins over 0-30 s ≤ 3 | 0 of 62 | passes |

**The gate fails, so the 86 s take was not run, as the plan says.**

The H3 keeper itself reads .86 at 5 s and .87 at 25 s to kf00 on these samples, so the
SFace bar is tight. LongCat's 0.80 at 15 s is below anything the keeper reads, and its
sharpness miss is wide: 254 against 410.

## Contact sheet

`out/kf00-30s/kf00-30s-faces.jpg`: LongCat (row 1) beside the H3 keeper (row 2) at
0.5/15/29.5 s. `gate-faces.jpg` has the same at all seven samples.

Reading the seven-sample sheet:
- **Identity:** the woman is recognisably the same throughout. The hair, earrings, scarf,
  coat and plate hold.
- **Skin:** smoother and flatter than H3. The forehead and cheek folds the keeper keeps
  are softened.
- **Colour:** the face sits darker and less saturated than H3's from the first frame
  (94.8 against 105.3).
- **Eyes:** the lids get heavier at 5 s and 15 s, and these are the SFace dips.
- **Mouth:** very active and articulate, opening wide at 29.5 s. No still-mouth bins.
- **Camera:** static and frontal, since the still is all it gets. There is no moving
  keystone.

## Deviations from plan v2

- **setuptools** was added to the venv, because triton needs it.
- **Models loaded in sequence**, with UMT5 and Whisper freed after use, and **the DiT's
  blocks streamed** (`--offload blocks`). The vendor's all-resident placement OOMs on 24 GB.
- **No separate 7 s fit on kf00.** The fit and timing ran on the vendor sample. The 30 s
  kf00 take measures the same numbers on kf00 (above).
- **The audio separator ran on the CPU**, because onnxruntime-gpu lacks cuDNN 9 adv.
- **Downloads came to about 46 GB, against the planned 36 GB**, because of the fp32 UMT5.
- **Uploads were staged through the workspace directory.** `upload_asset` reads only
  inside the workspace, so each file was copied into a temporary folder there, uploaded,
  and the folder removed.
- **The RSS stop sums the process tree.** `time -v` max RSS reads about 4 GB lower.

## Files on the server (workspace `constitution-walk-3`)

- the take, with sound: `asset:uploads/bench/longcat-avatar/kf00-30s-s1.mp4`;
- the contact sheet: `asset:uploads/bench/longcat-avatar/kf00-30s-faces.jpg`;
- the seven-sample sheet: `asset:uploads/bench/longcat-avatar/kf00-30s-gate-faces.jpg`;
- the 0.5 s reference frame: `asset:uploads/bench/longcat-avatar/kf00-30s-ref-0.5s.png`.

The vendor-sample fit output is `lem:~/spikes/longcat-avatar/out/vendor-7s/take.mp4`.
The venv, the weights (about 46 GB) and the logs stay in `lem:~/spikes/longcat-avatar/`.
Deleting them is Don's call.
