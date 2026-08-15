import os
import numpy
import torch
import soundfile
import json
import mimetypes
import logging
from diffusers.utils import (
    export_to_video,
    export_to_gif,
    encode_video,
    is_av_available,
)
from collections.abc import Iterable
from .security import validate_output_path, validate_string_input, SecurityError

logger = logging.getLogger("dw")

# Result saving constants
MAX_BASE_NAME_LENGTH = 200
DEFAULT_AUDIO_SAMPLE_RATE = 44100

# Audio content types soundfile can write, mapped to their file extension and to any
# write arguments the extension alone does not imply. Opus has no extension of its own
# in libsndfile - it is a subtype of the ogg container.
AUDIO_FORMATS = {
    "audio/wav": (".wav", {}),
    "audio/x-wav": (".wav", {}),
    "audio/aiff": (".aiff", {}),
    "audio/flac": (".flac", {}),
    "audio/x-flac": (".flac", {}),
    "audio/mpeg": (".mp3", {}),
    "audio/mp3": (".mp3", {}),
    "audio/ogg": (".ogg", {}),
    "audio/vorbis": (".ogg", {}),
    "audio/opus": (".ogg", {"format": "OGG", "subtype": "OPUS"}),
}

# Result definition keys passed through to soundfile - encoding quality controls
AUDIO_WRITE_ARGUMENTS = ["subtype", "format", "compression_level", "bitrate_mode"]

# Audio is written in chunks of this many frames - see write_audio
AUDIO_WRITE_CHUNK_FRAMES = 1 << 20

# The only container encode_video writes - it always encodes h264 video
MUXED_VIDEO_CONTENT_TYPE = "video/mp4"

# The names a modular pipeline's outputs go by. Asked for more than one output it returns
# them in a dict rather than on a pipeline output object, so its videos and the soundtrack
# generated alongside them arrive keyed instead of as attributes.
MODULAR_VIDEO_KEYS = ("videos", "frames")
MODULAR_AUDIO_KEYS = ("audio", "audios")
MODULAR_SAMPLE_RATE_KEYS = ("sampling_rate", "audio_sample_rate")


class AudioVideo:
    """A generated video together with the audio track generated alongside it.

    Pipelines like LTX-2 return audio next to their frames. Keeping the two paired lets
    the result mux them into one file instead of dropping the audio on the floor.
    """

    def __init__(self, frames, audio, sample_rate):
        """
        Args:
            frames: The video, as PIL images or an array of frames
            audio: Waveform for this video, shaped (channels, samples)
            sample_rate: Sample rate of the waveform, or None if the pipeline did not report one
        """
        self.frames = frames
        self.audio = audio
        self.sample_rate = sample_rate


class Result:
    """Manages and stores results from workflow steps.

    Handles result storage, artifact management, and file saving with support
    for multiple content types including images, video, audio, and JSON.
    """

    def __init__(self, result_definition):
        """Initialize Result with configuration for how to handle/save results.

        Args:
            result_definition: Dict containing result configuration including:
                - content_type: MIME type of the result
                - save: Boolean indicating if result should be saved
                - file_base_name: Base name for saved files
        """
        self.result_definition = result_definition
        self.result_list = []
        self.metadata = None
        logger.debug(f"Initialized Result with definition: {result_definition}")

    def set_metadata(self, metadata):
        """Set metadata to embed in saved image artifacts.

        Args:
            metadata: Dict of generation parameters to embed
        """
        self.metadata = metadata

    def add_result(self, result):
        """Add one or more results to the result list.

        Args:
            result: Single result or list of results to store
        """
        if isinstance(result, list):
            logger.debug(f"Adding {len(result)} results to result list")
            self.result_list.extend(result)
        else:
            if isinstance(result, str):
                # Clean up string results by removing extra quotes and whitespace
                result = result.strip().strip('"').strip()
            logger.debug("Adding single result to result list")
            self.result_list.append(result)

    def get_artifacts(self):
        """Retrieve all artifacts from stored results.

        Returns:
            List of all artifacts from all results
        """
        artifacts = []
        for result in self.result_list:
            artifacts.extend(get_artifact_list(result))

        logger.debug(f"Retrieved {len(artifacts)} artifacts from results")
        return artifacts

    def get_artifact_properties(self, property_name):
        """Extract specific properties from results.

        Args:
            property_name: Name of property to extract from results

        Returns:
            List of property values from results where property exists
        """
        values = []
        for result in self.result_list:
            if isinstance(result, Iterable):
                if property_name in result:
                    values.append(result[property_name])
            else:
                logger.warning(
                    f"Skipping non-dict result when getting property {property_name}: {type(result)}"
                )

        logger.debug(f"Retrieved {len(values)} values for property: {property_name}")
        return values

    def save(self, output_dir, default_base_name):
        """Save results to files based on content type.

        Args:
            output_dir: Directory to save files in
            default_base_name: Default name to use for files
        """
        try:
            # Validate output directory
            validated_output_dir = validate_output_path(output_dir, None)
            validated_base_name = validate_string_input(
                default_base_name, max_length=MAX_BASE_NAME_LENGTH
            )

            # Add directory check/creation
            if not os.path.exists(validated_output_dir):
                logger.debug(f"Creating output directory: {validated_output_dir}")
                os.makedirs(validated_output_dir, exist_ok=True)
            elif not os.path.isdir(validated_output_dir):
                raise ValueError(
                    f"Output path exists but is not a directory: {validated_output_dir}"
                )
        except SecurityError as e:
            logger.error(f"Security validation failed for output: {e}")
            raise
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create output directory: {e}")
            raise

        # Check if saving is enabled and content type is specified
        content_type = self.result_definition.get("content_type", None)
        if not self.result_definition.get("save", True) or content_type is None:
            logger.debug("Skipping save - disabled or no content type specified")
            return

        # Determine base filename with validation
        file_base_name = validated_base_name
        if "file_base_name" in self.result_definition:
            custom_base = validate_string_input(
                self.result_definition["file_base_name"],
                max_length=MAX_BASE_NAME_LENGTH,
            )
            file_base_name = custom_base + validated_base_name

        # Get file extension for content type
        extension = guess_extension(content_type)
        logger.debug(
            f"Saving with content type: {content_type}, extension: {extension}"
        )

        # Save each result
        for i, result in enumerate(self.result_list):
            if content_type.endswith("json"):
                # Handle JSON content type
                output_path = os.path.join(
                    validated_output_dir, f"{file_base_name}-{i}{extension}"
                )
                logger.info(f"Saving JSON result to {output_path}")
                with open(output_path, "w") as file:
                    file.write(json.dumps(result, indent=4))
            else:
                # Handle other content types
                for j, artifact in enumerate(get_artifact_list(result)):
                    self.save_artifact(
                        validated_output_dir,
                        artifact,
                        f"{file_base_name}-{i}.{j}",
                        content_type,
                        extension,
                    )

    def save_artifact(
        self, output_dir, artifact, file_base_name, content_type, extension
    ):
        """Save individual artifact to file based on its type.

        Args:
            output_dir: Directory to save file in
            artifact: The artifact to save
            file_base_name: Base name for the file
            content_type: MIME type of the content
            extension: File extension to use
        """
        if artifact is None:
            logger.warning(f"Skipping None artifact for {file_base_name}")
            return

        try:
            # Validate inputs
            validated_output_dir = validate_output_path(output_dir, None)
            validated_base_name = validate_string_input(
                file_base_name, max_length=MAX_BASE_NAME_LENGTH
            )
        except SecurityError as e:
            logger.error(f"Security validation failed for artifact save: {e}")
            raise

        if isinstance(artifact, dict):
            # Recursively save dictionary items
            logger.debug(
                f"Saving dictionary artifact with keys: {list(artifact.keys())}"
            )
            for k, v in artifact.items():
                self.save_artifact(
                    validated_output_dir,
                    v,
                    f"{validated_base_name}-{k}",
                    content_type,
                    extension,
                )
            return

        output_path = os.path.join(
            validated_output_dir, f"{validated_base_name}{extension}"
        )
        logger.info(f"Saving artifact to {output_path}")

        try:
            if content_type.startswith("video"):
                if isinstance(artifact, AudioVideo):
                    self.save_audio_video(artifact, output_path, content_type)
                else:
                    export_to_video(
                        artifact, output_path, fps=self.result_definition.get("fps", 8)
                    )
            elif content_type == "image/gif":
                export_to_gif(
                    artifact, output_path, fps=self.result_definition.get("fps", 8)
                )
            elif content_type.startswith("audio"):
                waveforms = normalize_audio(artifact)
                sample_rate = self.result_definition.get(
                    "sample_rate",
                    self.result_definition.get("samplerate", DEFAULT_AUDIO_SAMPLE_RATE),
                )
                # A batched waveform holds several songs - save each one separately
                if len(waveforms) > 1:
                    for k, waveform in enumerate(waveforms):
                        self.save_artifact(
                            validated_output_dir,
                            waveform,
                            f"{validated_base_name}-{k}",
                            content_type,
                            extension,
                        )
                    return
                write_audio(
                    output_path,
                    waveforms[0],
                    sample_rate,
                    **self.get_audio_write_arguments(content_type),
                )
            elif content_type.endswith("json"):
                with open(output_path, "w") as file:
                    file.write(json.dumps(artifact, indent=4))
            elif content_type.startswith("text"):
                with open(output_path, "w") as file:
                    file.write(artifact)
            elif hasattr(artifact, "save"):
                if (
                    self.metadata is not None
                    and self.result_definition.get("embed_metadata", False)
                    and content_type.startswith("image/")
                ):
                    self._save_image_with_metadata(artifact, output_path, content_type)
                else:
                    artifact.save(output_path)
            else:
                raise ValueError(
                    f"Content type {content_type} does not match result type {type(artifact)}"
                )
        except Exception as e:
            logger.error(
                f"Error saving artifact to {output_path}: {str(e)}", exc_info=True
            )
            raise

    def save_audio_video(self, artifact, output_path, content_type):
        """Write a video and the audio generated with it into a single file.

        encode_video muxes the two into an h264/mp4 file with PyAV. When PyAV is missing,
        the container is not mp4, or nothing told us the sample rate, the video is written
        on its own and the audio is dropped.

        Args:
            artifact: AudioVideo holding the frames and their waveform
            output_path: Path of the file to write
            content_type: MIME type of the video being written
        """
        fps = self.result_definition.get("fps", 8)
        # The pipeline reports the sample rate of what it generated - the result
        # definition can still override it
        sample_rate = self.result_definition.get(
            "audio_sample_rate", artifact.sample_rate
        )

        reason = None
        if artifact.audio is None:
            reason = "the pipeline returned no audio"
        elif sample_rate is None:
            reason = "the audio sample rate is unknown"
        elif content_type != MUXED_VIDEO_CONTENT_TYPE:
            reason = f"audio can only be muxed into {MUXED_VIDEO_CONTENT_TYPE}"
        elif not is_av_available():
            reason = "PyAV is not installed - install it with: pip install av"

        if reason is not None:
            logger.warning(f"Saving {output_path} without its audio because {reason}")
            export_to_video(artifact.frames, output_path, fps=fps)
            return

        logger.debug(f"Muxing audio at {sample_rate}Hz into {output_path}")
        encode_video(
            artifact.frames,
            fps=fps,
            output_path=output_path,
            audio=as_audio_track(artifact.audio),
            audio_sample_rate=sample_rate,
        )

    def get_audio_write_arguments(self, content_type):
        """Collect the soundfile arguments for an audio content type.

        The container comes from the content type and any encoding quality settings
        from the result definition.

        Args:
            content_type: MIME type of the audio being written

        Returns:
            Dict of keyword arguments for soundfile.write
        """
        _, write_arguments = AUDIO_FORMATS.get(content_type, (None, {}))
        write_arguments = dict(write_arguments)

        for argument_name in AUDIO_WRITE_ARGUMENTS:
            value = self.result_definition.get(argument_name, None)
            if value is not None:
                write_arguments[argument_name] = value

        return write_arguments

    def _save_image_with_metadata(self, image, output_path, content_type):
        """Save an image with embedded generation metadata.

        Args:
            image: PIL Image to save
            output_path: Path to save the image to
            content_type: MIME type of the image
        """
        metadata_json = json.dumps(self.metadata, default=str)

        if content_type == "image/png":
            from PIL.PngImagePlugin import PngInfo

            png_info = PngInfo()
            png_info.add_text("parameters", metadata_json)
            image.save(output_path, pnginfo=png_info)
            logger.debug(f"Embedded PNG metadata in {output_path}")
        elif content_type in ("image/jpeg", "image/webp"):
            try:
                import piexif

                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}
                if hasattr(image, "info") and "exif" in image.info:
                    exif_dict = piexif.load(image.info["exif"])
                exif_dict["Exif"][piexif.ExifIFD.UserComment] = (
                    piexif.helper.UserComment.dump(metadata_json)
                )
                exif_bytes = piexif.dump(exif_dict)
                image.save(output_path, exif=exif_bytes)
                logger.debug(f"Embedded EXIF metadata in {output_path}")
            except ImportError:
                logger.warning(
                    "piexif not installed - saving without metadata. "
                    "Install with: pip install piexif"
                )
                image.save(output_path)
        else:
            image.save(output_path)


def get_artifact_list(result):
    """Extract list of artifacts from a result object.

    Handles various result types including images, embeddings, frames, and audio.

    Args:
        result: Result object to extract artifacts from

    Returns:
        List of artifacts
    """
    # Already a paired video and audio track - it has frames, but they are one artifact
    if isinstance(result, AudioVideo):
        return [result]
    if hasattr(result, "images"):
        return result.images
    if hasattr(result, "image_embeds"):
        return result.image_embeds
    if hasattr(result, "image_embeddings"):
        return result.image_embeddings
    if hasattr(result, "frames"):
        # Some video pipelines generate an audio track along with the frames
        if getattr(result, "audio", None) is not None:
            return pair_audio_with_frames(
                result.frames, result.audio, getattr(result, "audio_sample_rate", None)
            )
        return result.frames
    if hasattr(result, "audios"):
        return [audio.T.float().cpu().numpy() for audio in result.audios]
    if isinstance(result, dict):
        # A modular pipeline asked for several outputs returns them keyed
        artifacts = modular_artifacts(result)
        if artifacts is not None:
            return artifacts
    if isinstance(result, list):
        return result
    return [result]


def modular_artifacts(result):
    """Extract the artifacts from the outputs a modular pipeline returns together.

    Asked for several outputs - `"output": ["videos", "audio", "sampling_rate"]` - a
    modular pipeline returns them in a dict instead of on one output object. Pairing the
    videos with the audio generated alongside them here saves them the same way a video
    pipeline's own output is saved, muxed into a single file. Any other requested output
    - "images" or "latents", say - is not part of that pairing, so it is carried along as
    one extra dict artifact, saved key by key the same way any other dictionary result is.

    Args:
        result: Dict of outputs returned by a modular pipeline

    Returns:
        List of artifacts, or None when the outputs hold no video - those are saved one
        output at a time instead
    """
    video_key, videos = first_item(result, MODULAR_VIDEO_KEYS)
    if videos is None:
        return None

    consumed_keys = {video_key}

    audio_key, audio = first_item(result, MODULAR_AUDIO_KEYS)
    if audio is None:
        artifacts = videos
    else:
        consumed_keys.add(audio_key)
        rate_key, sample_rate = first_item(result, MODULAR_SAMPLE_RATE_KEYS)
        consumed_keys.add(rate_key)
        artifacts = pair_audio_with_frames(videos, audio, sample_rate)

    # Keys the video/audio pairing above did not consume still need to be saved, not
    # dropped - carry them along as one extra artifact, saved key by key like any other
    # dictionary result
    leftovers = {
        key: value
        for key, value in result.items()
        if key not in consumed_keys and value is not None
    }
    if leftovers:
        artifacts = list(artifacts) + [leftovers]

    return artifacts


def first_item(values, keys):
    """The key and value of the first of `keys` present in `values`.

    Returns (None, None) when none of them are.
    """
    for key in keys:
        value = values.get(key, None)
        if value is not None:
            return key, value

    return None, None


def pair_audio_with_frames(videos, audio, sample_rate):
    """Pair each generated video with its own audio track.

    Both are batched - videos[i] and audio[i] belong to the same generation. Only the
    pipeline knows the sample rate its vocoder produced the audio at, so it comes along
    rather than being guessed at here.

    Args:
        videos: The generated videos, one list of frames per generation
        audio: The generated waveforms, one per generation
        sample_rate: Sample rate of the waveforms, or None if the pipeline did not report one

    Returns:
        List of AudioVideo artifacts, one per generated video
    """
    return [
        AudioVideo(frames, audio[i] if i < len(audio) else None, sample_rate)
        for i, frames in enumerate(videos)
    ]


def as_audio_track(audio):
    """Convert a generated waveform into the tensor encode_video expects.

    encode_video wants a float torch tensor on the CPU shaped (channels, samples) -
    pipelines hand back bfloat16 tensors that are still on the GPU, or numpy arrays.

    Args:
        audio: Waveform as a torch tensor or numpy array

    Returns:
        Float CPU torch tensor holding the waveform
    """
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(numpy.asarray(audio))

    return audio.detach().float().cpu()


def write_audio(output_path, waveform, sample_rate, **write_arguments):
    """Write a single waveform to disk.

    soundfile.write() hands the whole waveform to libsndfile in one call, whose vorbis
    encoder segfaults past 2**21 frames - about 48 seconds of 44.1kHz audio. Writing in
    chunks avoids that and bounds the encoder's working set for long audio.

    Args:
        output_path: Path of the file to write
        waveform: Numpy array shaped (samples,) or (samples, channels)
        sample_rate: Sample rate to record in the file
        write_arguments: Container and encoding arguments for soundfile
    """
    channels = 1 if waveform.ndim == 1 else waveform.shape[1]

    with soundfile.SoundFile(
        output_path,
        "w",
        samplerate=sample_rate,
        channels=channels,
        **write_arguments,
    ) as audio_file:
        for start in range(0, len(waveform), AUDIO_WRITE_CHUNK_FRAMES):
            audio_file.write(waveform[start : start + AUDIO_WRITE_CHUNK_FRAMES])


def normalize_audio(artifact):
    """Convert an audio artifact into waveforms soundfile can write.

    Pipelines return audio as torch tensors or numpy arrays, channels first and
    optionally batched. soundfile wants samples first, one waveform at a time.

    Args:
        artifact: Audio waveform(s) as a torch tensor or numpy array

    Returns:
        List of numpy arrays shaped (samples,) or (samples, channels)
    """
    # Torch tensors may be on the GPU and in a dtype numpy does not understand
    if hasattr(artifact, "detach"):
        artifact = artifact.detach().float().cpu().numpy()

    waveform = numpy.asarray(artifact)

    if waveform.ndim == 1:
        return [waveform]

    if waveform.ndim == 2:
        # Channels first - a waveform always has far more samples than channels
        if waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T
        return [waveform]

    if waveform.ndim == 3:
        # (batch, channels, samples) - one waveform per batch item
        return [item.T for item in waveform]

    raise ValueError(f"Cannot save audio with shape {waveform.shape}")


def guess_extension(content_type):
    """Determine file extension from MIME type.

    Args:
        content_type: MIME type string

    Returns:
        String containing file extension with leading dot
    """
    if not content_type:
        logger.warning("No content type provided for extension guess")
        return ""

    # Audio is looked up first - soundfile picks the container from the extension and
    # does not recognize every extension mimetypes suggests, such as '.oga' for ogg
    if content_type in AUDIO_FORMATS:
        return AUDIO_FORMATS[content_type][0]

    ext = mimetypes.guess_extension(content_type)
    if ext is not None:
        return ext

    return ""
