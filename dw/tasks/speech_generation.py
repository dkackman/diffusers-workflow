"""
Speech generation via HuggingFace transformers.

Takes a line of text and speaks it with a local text-to-speech model, returning
the waveform and the rate it was generated at so the rest of the audio plumbing -
slice_audio, fade_audio and pair_audio - composes with it directly (concat_videos
and dissolve_videos join videos; pair the track onto a video first).

The role this is built for is voice *timbre reference*, not the track a mouth
follows. MiniMax H3 lip-syncs well when it generates the speech itself and poorly
when it must follow supplied audio, so its audio reference takes a few seconds of
a voice to fix timbre, pitch and delivery while the model still generates the
line. A generated clip referenced in every shot makes voice consistency an actual
conditioning signal rather than a prose description that has to land identically
a dozen times. The other honest uses are a voice that must be matched, and
narration over shots where nothing has to lip-sync to it.
"""

import logging

import torch
from transformers import pipeline as hf_pipeline

from .. import preferred_task_dtype
from ..result import AudioTrack
from .audio_utils import as_channels_samples, load_audio, resample_waveform
from .model_cache import cached_model, hf_pipeline_placement

logger = logging.getLogger("dw")

# Small, loads without a separate speaker-embedding dataset, and its voice presets
# give distinct speakers - which is the point when two characters have to sound
# like two people. A single-voice model like facebook/mms-tts-eng (which takes no
# voice_preset) is a quarter the size and a reasonable override where only one
# voice is needed
_DEFAULT_MODEL = "suno/bark-small"

# What SpeechT5's voice-cloning recipes are built around - trained on the same
# corpus (VoxCeleb) the CMU ARCTIC x-vectors that ship with the model card come
# from, so a reference clip reduces to a vector in the space the model expects
_SPEAKER_ENCODER_MODEL = "speechbrain/spkrec-xvect-voxceleb"
# What the encoder was trained on - resampling anything else to this rate is
# part of extracting a voiceprint, not an approximation of one
_SPEAKER_ENCODER_SAMPLE_RATE = 16000


def _speaker_embedding_tensor(location, device):
    """The x-vector speechbrain's spkrec-xvect-voxceleb extracts from a
    reference audio file - what SpeechT5 conditions its voice on.

    Args:
        location: Path to a reference audio file (an 'asset:' reference has
            already resolved to this by the time a task sees it).
        device: Where to run the encoder.
    """
    from speechbrain.inference.speaker import EncoderClassifier

    def load_encoder():
        return EncoderClassifier.from_hparams(
            source=_SPEAKER_ENCODER_MODEL,
            run_opts={"device": str(device)},
        )

    encoder = cached_model(
        ("speaker_encoder", _SPEAKER_ENCODER_MODEL, str(device)), load_encoder
    )

    waveform, sample_rate = load_audio(location)
    waveform = resample_waveform(waveform, sample_rate, _SPEAKER_ENCODER_SAMPLE_RATE)
    mono = waveform.mean(axis=0)

    with torch.no_grad():
        embedding = encoder.encode_batch(torch.as_tensor(mono).unsqueeze(0))
        embedding = torch.nn.functional.normalize(embedding, dim=2)
    # SpeechT5's generate() wants (batch, 512); the encoder's raw output is
    # (1, 1, 512), so squeeze collapses it back to (512,) before restoring
    # the batch dimension the model actually requires
    return embedding.squeeze().unsqueeze(0).to(device=device, dtype=torch.float32)


def generate_speech(text, device="cpu", **kwargs):
    """Speak a line of text with a local text-to-speech model.

    Args:
        text: The line to speak.
        device: Target device ("cuda", "mps", "cpu").
        **kwargs:
            model_name: HuggingFace model ID. Defaults to suno/bark-small.
            voice_preset: The speaker to use, for a model that has presets -
                "v2/en_speaker_6" and friends for Bark. This is a preprocessing
                argument: it selects the speaker before generation rather than
                parameterizing it, which is why it is named here rather than left
                to forward_params, where it would be silently dropped.
            speaker_embedding: Path to a reference audio file (typically an
                'asset:' reference) whose voice a SpeechT5 model should speak
                in. Reduced to an x-vector with speechbrain's
                spkrec-xvect-voxceleb and injected into forward_params as
                'speaker_embeddings' - SpeechT5 is the only pipeline here that
                conditions on one. A VITS model's speaker instead takes a plain
                'speaker_id' int, which already reaches the model unchanged
                through forward_params and needs no argument of its own.
            forward_params: Passed to the model's forward/generate call.
            generate_kwargs: Ad-hoc generation settings for a generative model -
                temperature, do_sample and so on.

    Returns:
        An AudioTrack holding the waveform, shaped (channels, samples), and the
        sample rate the model generated it at.

    Raises:
        ValueError: If the model reports no sample rate for what it generated.
    """
    model_name = kwargs.get("model_name", _DEFAULT_MODEL)
    voice_preset = kwargs.get("voice_preset", None)
    speaker_embedding = kwargs.get("speaker_embedding", None)
    dtype = preferred_task_dtype(device)

    def load_pipe():
        logger.info(f"Generating speech with {model_name} on {device}")
        placement = hf_pipeline_placement(device)
        return hf_pipeline(
            "text-to-speech",
            model=model_name,
            torch_dtype=dtype,
            **placement,
        )

    pipe = cached_model(
        ("speech_generation", model_name, str(device), str(dtype)),
        load_pipe,
    )

    if voice_preset and getattr(pipe, "processor", None) is None:
        # A single-voice model has no processor to hand the preset to;
        # transformers logs the kwarg as unrecognised and speaks anyway
        raise ValueError(
            f"{model_name} takes no 'voice_preset' - it has one voice. Drop the "
            "preset, or use a model with speaker presets such as suno/bark-small"
        )

    forward_params = kwargs.get("forward_params") or {}
    if speaker_embedding:
        model_type = getattr(getattr(pipe.model, "config", None), "model_type", None)
        if model_type != "speecht5":
            # Only SpeechT5 conditions on an x-vector; a model that does not
            # would drop 'speaker_embeddings' as an unrecognised forward kwarg
            # and generate in its own voice, same failure mode as voice_preset
            raise ValueError(
                f"{model_name} takes no 'speaker_embedding' - only a SpeechT5 "
                "model conditions on an x-vector. Drop it, or use a SpeechT5 "
                "model such as microsoft/speecht5_tts"
            )
        forward_params = {
            **forward_params,
            "speaker_embeddings": _speaker_embedding_tensor(speaker_embedding, device),
        }

    logger.info(f"Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")
    output = pipe(
        text,
        preprocess_params={"voice_preset": voice_preset} if voice_preset else {},
        forward_params=forward_params,
        generate_kwargs=kwargs.get("generate_kwargs") or {},
    )

    sample_rate = output.get("sampling_rate")
    if sample_rate is None:
        # Saving at the 44100 default instead would be quietly wrong rather than
        # loud, and speech at the wrong rate is wrong in pitch as well as length
        raise ValueError(
            f"{model_name} reported no sample rate for the speech it generated"
        )

    return AudioTrack(as_channels_samples(output["audio"]), int(sample_rate))
