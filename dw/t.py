import torch
from diffusers import LTX2Pipeline
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video

MODEL_ID = "Lightricks/LTX-2.5-Diffusers"

pipe = LTX2Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload(device="cuda")
pipe.vae.enable_tiling()

video, audio = pipe(
    prompt="A cinematic shot of a marmot walking through a snowy forest at dawn, "
           "the camera tracking alongside, snow crunching underfoot.",
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=960,
    height=544,
    num_frames=121,
    frame_rate=24.0,
    sigmas=DISTILLED_SIGMA_VALUES,
    guidance_scale=1.0,
    audio_guidance_scale=1.0,
    stg_scale=0.0,
    audio_stg_scale=0.0,
    modality_scale=1.0,
    audio_modality_scale=1.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=24,
    output_path="ltx25.mp4",
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
)
