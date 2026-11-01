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
import math

import torch
from transformers import pipeline as hf_pipeline

from .. import get_device_type, preferred_task_dtype
from ..dsp import as_channels_samples
from ..media_types import AudioTrack
from ..dsp import resample_waveform
from .audio_utils import load_audio
from ..security import UntrustedWorkflowError
from ..trust import (
    GENERATE_REMOTE_CODE_ARGUMENTS,
    REMOTE_CODE_ARGUMENTS,
    require_trusted_generate_arguments,
    workflows_are_trusted,
)
from . import audio8_tts_model as audio8
from .model_cache import cached_model, hf_pipeline_placement
from .weights import argument_or_default, file_entries, repo_entries

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

# Audio8-TTS (#1112) is a vendored architecture (audio8_tts_model), not a
# transformers pipeline, so it gets its own branch; its identity is the
# vendored module's constant. Its 2,048-position context holds the prompt
# text, the reference's codec frames and the speech it generates, at the
# codec's 44100 / 2048 = ~21.5 frames a second
_AUDIO8_REFERENCE_ARGUMENTS = ("reference_audio", "reference_text")
_AUDIO8_REFUSED_ARGUMENTS = ("speaker_embedding", "voice_preset", "messages")
# A longer reference leaves too little of the context for the line (#1112 Q2)
_AUDIO8_MAX_REFERENCE_SECONDS = 20.0
# The estimate a line is refused on before anything loads: speech at ~14
# characters a second, ~2.5 characters a prompt token (Cyrillic tokenizes
# denser than English, so this leans long), and the prompt's fixed chat
# scaffolding. A line that still runs out of context is refused after
# generation rather than returned cut off
_AUDIO8_CHARS_PER_SECOND = 14.0
_AUDIO8_CHARS_PER_TOKEN = 2.5
_AUDIO8_PROMPT_OVERHEAD = 48
# What the vendored generate() takes from generate_kwargs / forward_params
_AUDIO8_GENERATION_KEYS = frozenset(
    {"temperature", "top_p", "top_k", "do_sample", "max_new_tokens"}
)
# The architecture's defaults, which the pinned config.json matches - the fit
# estimate runs before that file is fetched
_AUDIO8_CONFIG = audio8.ArkttsConfig()


def model_source(arguments):
    """The text-to-speech repo a generate_speech step loads."""
    return argument_or_default(arguments, "model_name", _DEFAULT_MODEL)


def required_weights(arguments):
    """The speech model's repo, plus speechbrain's speaker encoder when the
    step clones a voice from a `speaker_embedding` clip, for the plan
    (dw/tasks/weights.py)."""
    if model_source(arguments) == audio8.REPO_ID:
        # Every file at the pinned commit the vendored module was reviewed
        # at - a repo entry cannot pin, and codec.pth is a pickle-format file
        return [
            entry
            for filename in audio8.MODEL_FILES
            for entry in file_entries(audio8.REPO_ID, filename, audio8.REVISION)
        ]
    encoder = (
        _SPEAKER_ENCODER_MODEL if (arguments or {}).get("speaker_embedding") else None
    )
    return repo_entries(model_source(arguments), encoder)


def _speaker_embedding_tensor(location, device, dtype):
    """The x-vector speechbrain's spkrec-xvect-voxceleb extracts from a
    reference audio file - what SpeechT5 conditions its voice on.

    Args:
        location: Path to a reference audio file (an 'asset:' reference has
            already resolved to this by the time a task sees it).
        device: Where to run the encoder.
        dtype: The speech pipeline's own dtype - the embedding is concatenated
            with the pipe's hidden states inside the decoder prenet, so it must
            match the pipe's precision (fp16 on CUDA) or the linear layer there
            refuses a Float x Half product.
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
    return embedding.squeeze().unsqueeze(0).to(device=device, dtype=dtype)


def _check_messages(messages):
    """Chat-templated input: a non-empty list of {role, content} string dicts."""
    valid = isinstance(messages, list) and len(messages) > 0
    if valid:
        for message in messages:
            if (
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
            ):
                valid = False
                break
    if not valid:
        raise ValueError(
            "generate_speech's 'messages' needs a non-empty list of "
            "{'role': ..., 'content': ...} dicts, both strings - a bare "
            "string or a wrong-keyed dict is not chat-templated input"
        )


def _generation_settings(value, name):
    """A workflow's forward_params or generate_kwargs: an object, with the
    Hub-code arguments refused unless workflows are trusted."""
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"generate_speech: {name} must be an object of settings, got "
            f"{type(value).__name__}"
        )
    require_trusted_generate_arguments(value, f"speech_generation ({name})")
    return value


# The Hub-code switches as top-level task arguments. generate_speech reads
# none of them there, and a kwargs task drops what it does not read, so one
# would otherwise pass silently - looking as if it took effect (#1121 bounce)
_TOP_LEVEL_REMOTE_CODE_ARGUMENTS = tuple(
    dict.fromkeys(REMOTE_CODE_ARGUMENTS + GENERATE_REMOTE_CODE_ARGUMENTS)
)


def _refuse_top_level_remote_code(kwargs):
    """Refuse trust_remote_code / custom_pipeline / custom_generate given as
    step arguments, for every model, before anything loads: the trust gate's
    refusal for an untrusted workflow, and a plain error for a trusted one
    (the task still would not use it)."""
    given = [name for name in _TOP_LEVEL_REMOTE_CODE_ARGUMENTS if kwargs.get(name)]
    if not given:
        return
    names = " or ".join(map(repr, given))
    if not workflows_are_trusted():
        raise UntrustedWorkflowError(
            f"Refusing to run speech_generation: its arguments set {names}, "
            "which downloads and executes Python from a Hub repository, so it "
            "needs trusted workflows - see docs/SECURITY.md. generate_speech "
            "loads no Hub code from its arguments in any case; drop it"
        )
    raise ValueError(
        f"generate_speech takes no {names} argument - it loads no Hub code "
        "from its arguments, so the setting would have no effect. Drop it"
    )


def _audio8_dtype(device):
    # The checkpoint is bf16, which a 0.6B Qwen-style decoder keeps finite
    # where fp16 may not; MPS and CPU run it in fp32 (fp16/bf16 NaN or crawl)
    return torch.bfloat16 if get_device_type(device) == "cuda" else torch.float32


def _check_audio8_arguments(model_name, kwargs):
    """The argument refusals around the Audio8 branch, all raised before
    anything loads - a kwargs task silently drops an argument it does not
    read, so a misplaced one would otherwise run in the wrong voice."""
    given = {name for name, value in kwargs.items() if value is not None}
    if model_name != audio8.REPO_ID:
        misplaced = [name for name in _AUDIO8_REFERENCE_ARGUMENTS if name in given]
        if misplaced:
            raise ValueError(
                f"{model_name} takes no {' or '.join(map(repr, misplaced))} - "
                f"only {audio8.REPO_ID} clones a voice from a reference clip "
                "and its transcript. Drop it, or set model_name to that model"
            )
        return
    refused = [name for name in _AUDIO8_REFUSED_ARGUMENTS if name in given]
    if refused:
        raise ValueError(
            f"{audio8.REPO_ID} takes no {' or '.join(map(repr, refused))} - its "
            "voice comes from 'reference_audio' with 'reference_text' (or its "
            "default voice when neither is given), and it speaks plain 'text'"
        )
    has_audio = "reference_audio" in given
    has_text = bool(kwargs.get("reference_text"))
    if has_audio != has_text:
        missing, present = (
            ("reference_text", "reference_audio")
            if has_audio
            else ("reference_audio", "reference_text")
        )
        raise ValueError(
            f"generate_speech: '{present}' needs '{missing}' - {audio8.REPO_ID} "
            "clones a voice from a reference clip together with the words "
            "spoken in it. Give both, or neither for the default voice"
        )
    if has_text and not isinstance(kwargs["reference_text"], str):
        raise ValueError(
            "generate_speech: 'reference_text' must be the transcript of the "
            f"reference clip as a string, got {type(kwargs['reference_text']).__name__}"
        )


def _audio8_generation_kwargs(forward_params, generate_kwargs):
    """The sampling settings the vendored generate() takes, merged; anything
    else is refused by name rather than raised from inside a loaded model."""
    settings = {**forward_params, **generate_kwargs}
    unknown = sorted(set(settings) - _AUDIO8_GENERATION_KEYS)
    if unknown:
        raise ValueError(
            f"{audio8.REPO_ID} takes no generation setting(s) {unknown} - it "
            f"takes {sorted(_AUDIO8_GENERATION_KEYS)} in generate_kwargs"
        )
    return settings


def _audio8_estimated_positions(text, reference_text, reference_seconds):
    """How many of the context's positions a line is expected to take: the
    prompt (scaffolding, both texts, the reference's frames) plus the frames
    of the speech it generates."""
    frame_rate = _AUDIO8_CONFIG.codec_sample_rate / _AUDIO8_CONFIG.codec_frame_size
    text_chars = len(" ".join(text.split()))
    reference_chars = len(" ".join((reference_text or "").split()))
    prompt = (
        _AUDIO8_PROMPT_OVERHEAD
        + (text_chars + reference_chars) / _AUDIO8_CHARS_PER_TOKEN
        + reference_seconds * frame_rate
    )
    speech = text_chars / _AUDIO8_CHARS_PER_SECOND * frame_rate
    return int(math.ceil(prompt + speech))


def _load_audio8(device, dtype):
    """(model, processor) from the pinned files, fetched to the Hugging Face
    cache here - the vendored module reads local paths only."""
    from huggingface_hub import hf_hub_download

    def load():
        logger.info(f"Loading {audio8.REPO_ID} at {audio8.REVISION[:8]} on {device}")
        paths = {
            filename: hf_hub_download(
                repo_id=audio8.REPO_ID, filename=filename, revision=audio8.REVISION
            )
            for filename in audio8.MODEL_FILES
        }
        model = audio8.load_model(
            paths[audio8.CONFIG_FILE],
            paths[audio8.WEIGHTS_FILE],
            paths[audio8.CODEC_FILE],
            generation_config_path=paths[audio8.GENERATION_CONFIG_FILE],
            device=device,
            dtype=dtype,
        )
        tokenizer = audio8.load_tokenizer(paths[audio8.TOKENIZER_FILE])
        return model, audio8.ArkttsProcessor.for_model(tokenizer, model.config)

    return cached_model(
        ("speech_generation", audio8.REPO_ID, str(device), str(dtype), audio8.REVISION),
        load,
    )


def _generate_audio8(text, device, seed, kwargs, settings):
    """Speak `text` with Audio8-TTS, in the voice of reference_audio when one
    is given; every refusal has been raised by the time this loads."""
    if text is None:
        raise ValueError(f"generate_speech: {audio8.REPO_ID} needs 'text'")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("generate_speech: 'text' must be a non-empty string")
    reference_audio = kwargs.get("reference_audio")
    reference_text = kwargs.get("reference_text")

    reference = None
    reference_seconds = 0.0
    if reference_audio is not None:
        waveform, sample_rate = load_audio(reference_audio)
        reference_seconds = waveform.shape[-1] / float(sample_rate)
        if reference_seconds > _AUDIO8_MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"generate_speech: 'reference_audio' is {reference_seconds:.1f} s; "
                f"{audio8.REPO_ID} takes a reference of at most "
                f"{_AUDIO8_MAX_REFERENCE_SECONDS:.0f} s (its context holds the "
                "reference and the line together). Trim it with slice_audio - "
                "5 to 10 s of clean speech is plenty"
            )
        reference = (waveform.mean(axis=0), int(sample_rate))

    context = _AUDIO8_CONFIG.max_seq_len
    estimated = _audio8_estimated_positions(text, reference_text, reference_seconds)
    if estimated > context:
        raise ValueError(
            f"generate_speech: 'text' is too long for {audio8.REPO_ID} - it is "
            f"estimated to need {estimated} of the model's {context} context "
            "positions (prompt, reference and speech together, about 95 s in "
            "all). Split the line into shorter generate_speech steps"
            + (" or use a shorter reference" if reference is not None else "")
        )

    dtype = _audio8_dtype(device)
    model, processor = _load_audio8(device, dtype)
    if reference is None:
        inputs = processor(text)
    else:
        inputs = processor(
            text,
            reference_text=reference_text,
            reference_audio=reference[0],
            sampling_rate=reference[1],
        )
    # Fill the context by default - the checkpoint's own 512-frame cap would
    # end a line at ~24 s without saying so
    settings = {"max_new_tokens": context, **settings}
    logger.info(f"Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")
    # An explicit generator, not the global seed the pipelines here use: the
    # codec builds lazily inside the first generate() that encodes a
    # reference, and its init draws from the global RNG, so a globally seeded
    # first take would differ from every later one
    generator = (
        torch.Generator(device=model.device).manual_seed(int(seed))
        if seed is not None
        else None
    )
    output = model.generate(
        **inputs, return_dict_in_generate=True, generator=generator, **settings
    )
    if not bool(output.finished.all()):
        raise ValueError(
            f"generate_speech: {audio8.REPO_ID} ran out of room before the line "
            "ended, so it is refused rather than returned cut off. Split the "
            "line into shorter generate_speech steps"
            + (" or use a shorter reference" if reference is not None else "")
            + " (or raise generate_kwargs.max_new_tokens if you lowered it)"
        )
    waveforms, lengths = model.decode_audio(output.codes)
    samples = waveforms[0, : int(lengths[0])].cpu().numpy()
    return AudioTrack(as_channels_samples(samples), int(model.config.codec_sample_rate))


def generate_speech(text=None, device="cpu", seed=None, **kwargs):
    """Speak a line of text with a local text-to-speech model - any
    transformers text-to-speech checkpoint named by model_name. Three
    families are wired for their voice controls: Bark (suno/bark-small, the
    default; distinct speakers by voice_preset), MMS/VITS
    (facebook/mms-tts-eng and the other per-language mms-tts-* checkpoints;
    one voice, small and fast; a multi-speaker VITS takes speaker_id through
    forward_params) and SpeechT5 (microsoft/speecht5_tts; speaks in the
    voice of a reference clip given as speaker_embedding). Audio8-TTS
    (Edge0/Audio8-TTS-Realistic-0.6b, pinned to one revision) is not a
    transformers pipeline and runs through its own vendored loader: it speaks
    44.1 kHz mono in the voice of a short reference clip given as
    reference_audio with its transcript as reference_text, or in its default
    voice when neither is given.

    Args:
        text: The line to speak. Mutually exclusive with messages - exactly
            one of the two is required.
        device: Target device ("cuda", "mps", "cpu").
        seed: The workflow/step-resolved seed, when one was set. A pipeline
            step gets an explicit torch.Generator; transformers' generate()
            takes no such argument, so reproducing Bark (and any other
            sampling model here) means seeding the global RNG immediately
            before the call - unseeded when None, matching every other
            unseeded task (#261). Audio8 takes it as an explicit
            torch.Generator instead, so the same seed repeats the same take.
        **kwargs:
            messages: Chat-templated input for a model such as VibeVoice that
                takes a conversation rather than a bare string - a list of
                {"role": ..., "content": ...} dicts. Passed straight through
                as the pipeline's text_inputs, which applies the model's own
                chat template; a model with no chat template configured (Bark
                and friends) raises when handed this instead of text.
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
                conditions on one, and it is not optional for SpeechT5: the
                model refuses to generate without a speaker embedding, so this
                is required whenever model_name is a SpeechT5 checkpoint. A
                VITS model's speaker instead takes a plain 'speaker_id' int,
                which already reaches the model unchanged through
                forward_params and needs no argument of its own.
            reference_audio: Edge0/Audio8-TTS-Realistic-0.6b only - a clip of
                the voice to clone (typically an 'asset:' reference), at most
                20 s; 5-10 s of clean speech is plenty. Requires
                reference_text. Refused on any other model, and refused
                together with speaker_embedding, voice_preset or messages,
                which Audio8 does not take. Clone only a voice you have
                permission to use.
            reference_text: Edge0/Audio8-TTS-Realistic-0.6b only - the exact
                words spoken in reference_audio. Requires reference_audio
                and is refused on any other model. Text plus reference must
                fit the model's 2,048-position context (about 95 s of
                prompt, reference and speech together); a line estimated not
                to fit is refused before generation, never truncated.
            forward_params: Passed to the model's forward/generate call.
            generate_kwargs: Ad-hoc generation settings for a generative model -
                temperature, do_sample and so on. Audio8 takes temperature,
                top_p, top_k, do_sample and max_new_tokens only (here or in
                forward_params); anything else is refused by name.

    Returns:
        An AudioTrack holding the waveform, shaped (channels, samples), and the
        sample rate the model generated it at.

    Raises:
        ValueError: If the model reports no sample rate for what it generated,
            or if text/messages are both given or neither is, or for any of
            the Audio8 refusals above - each raised before a model loads.
        UntrustedWorkflowError: If an untrusted workflow sets
            trust_remote_code, custom_pipeline or custom_generate as a step
            argument (a trusted one gets a ValueError: the task never reads
            them there), or Hub-code settings in forward_params or
            generate_kwargs.
    """
    messages = kwargs.get("messages", None)
    if (text is None) == (messages is None):
        raise ValueError(
            "generate_speech needs exactly one of 'text' or 'messages' - "
            "a plain line to speak, or chat-templated input for a model "
            "such as VibeVoice"
        )
    if messages is not None:
        _check_messages(messages)
    text_inputs = messages if messages is not None else text
    _refuse_top_level_remote_code(kwargs)
    # Both reach model.generate() for a generative model, where
    # custom_generate / trust_remote_code download and run Hub code: refused
    # for an untrusted workflow before anything loads, as text_generation does
    forward_params = _generation_settings(
        kwargs.get("forward_params"), "forward_params"
    )
    generate_kwargs = _generation_settings(
        kwargs.get("generate_kwargs"), "generate_kwargs"
    )

    model_name = model_source(kwargs)
    _check_audio8_arguments(model_name, kwargs)
    if model_name == audio8.REPO_ID:
        settings = _audio8_generation_kwargs(forward_params, generate_kwargs)
        return _generate_audio8(text, device, seed, kwargs, settings)

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
            "speaker_embeddings": _speaker_embedding_tensor(
                speaker_embedding, device, dtype
            ),
        }

    if messages is not None:
        logger.info(f"Speaking {len(messages)} chat-templated message(s)")
    else:
        logger.info(f"Speaking: {text[:100]}{'...' if len(text) > 100 else ''}")
    if seed is not None:
        torch.manual_seed(seed)
    output = pipe(
        text_inputs,
        preprocess_params={"voice_preset": voice_preset} if voice_preset else {},
        forward_params=forward_params,
        generate_kwargs=generate_kwargs,
    )

    sample_rate = output.get("sampling_rate")
    if sample_rate is None:
        # Saving at the 44100 default instead would be quietly wrong rather than
        # loud, and speech at the wrong rate is wrong in pitch as well as length
        raise ValueError(
            f"{model_name} reported no sample rate for the speech it generated"
        )

    return AudioTrack(as_channels_samples(output["audio"]), int(sample_rate))
