import os
import time
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
from collections.abc import Mapping
from .events import emit_log, emit_phase, emit_warning
from .security import (
    SecurityError,
    validate_file_base_name,
    validate_output_path,
    validate_string_input,
)

logger = logging.getLogger("dw")

# Result saving constants
MAX_BASE_NAME_LENGTH = 200
DEFAULT_AUDIO_SAMPLE_RATE = 44100
# The rate a video is written at when neither the workflow nor the artifact
# says - a diffusers convention old enough that changing it would restate
# every existing workflow's output
DEFAULT_VIDEO_FPS = 8


def _artifact_size(artifact):
    """How much there is to write, said the way the thing itself counts -
    frames for a video, samples for a waveform. Best effort: it is narration
    beside a file name, so anything it cannot measure it does not mention."""
    try:
        frames = getattr(artifact, "frames", None)
        if frames is not None:
            return f"{len(frames)} frames"
        if hasattr(artifact, "__len__") and not isinstance(artifact, (str, bytes)):
            return f"{len(artifact)} frames"
    except Exception:
        pass
    return ""


# A deliverable this close to full scale has no headroom left: an mp3 or AAC
# encode of it decodes above 0 dBFS and clips, which is why a track measured
# at +1.3 dBFS in the gallery can have been written from samples that never
# exceeded 1.0. Same ceiling `match_levels` holds a gain to
# (MATCH_CEILING_DBFS in dw/tasks/audio_utils.py)
HEADROOM_WARN_DBFS = -0.5


def _peak_dbfs(waveform):
    """The loudest sample of anything saveable as audio, in dBFS, or None
    when it cannot be measured cheaply (no samples, a lazily-decoded
    reader, a shape nothing here recognises)."""
    try:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        samples = numpy.asarray(waveform)
        if samples.size == 0 or not numpy.issubdtype(samples.dtype, numpy.number):
            return None
        peak = float(numpy.abs(samples).max())
    except Exception:
        logger.debug("Could not measure the peak of a saved track", exc_info=True)
        return None
    if peak <= 0.0:
        return None
    return 20.0 * float(numpy.log10(peak))


def warn_without_headroom(waveform, file_name):
    """Say when the soundtrack about to be written is at or over full scale.

    A clipped deliverable is invisible to the consumer this server is built
    for: the job succeeds, and an agent that cannot listen has `peak_dbfs`
    and no rule to read it against - `get_gallery_metadata` teaches the
    near-silent end of the range and said nothing about the other one (#158).
    A warning rather than a change to the mix: what level a deliverable
    should sit at is the workflow's to decide, and `normalize_audio` is the
    step that decides it.
    """
    peak = _peak_dbfs(waveform)
    if peak is None or peak < HEADROOM_WARN_DBFS:
        return None
    emit_warning(
        f"The soundtrack written to {file_name} peaks at {peak:+.1f} dBFS, "
        f"which leaves no headroom below full scale - an mp3 or AAC encode "
        f"of it decodes above 0 dBFS and clips. Add a 'normalize_audio' "
        f"step (peak_dbfs: -1) before the step that saves it, or "
        f"'match_levels' on the join that made it.",
        kind="audio_no_headroom",
        file=file_name,
        peak_dbfs=round(peak, 2),
    )
    return peak


# The written file, decoded, is the only measurement that is the consumer's
# own. `warn_without_headroom` measures the waveform handed to the writer,
# and the encoder is downstream of that: a track normalized to exactly
# -1.0 dBFS came back out of an AAC mux at +0.94, so the deliverable of a
# clean default run of `music-video` was above full scale and nothing
# warned, because the number the check read was -1.0 (#158, #159, #161)
CLIPPED_WARN_DBFS = 0.0


def warn_if_written_above_full_scale(output_path, already_warned=False):
    """Say when the file just written decodes above full scale.

    The overshoot a lossy encode adds is material-dependent - about 0.1 dB
    on the mp3s measured for #159 and about 1.9 dB on the AAC mux of the
    same song - so no amount of headroom chosen up front can be known to be
    enough. Reading the file back is what closes that: whatever the encoder
    did, this is the number a consumer's decoder will see.

    Silent when `warn_without_headroom` has already spoken for this file -
    but only for a plain audio save. That suppression assumed the encoder
    only ever adds overshoot, which held for the mp3s #159/#161 measured but
    is backwards for an H3 video mux: #174 measured that family's AAC mux
    landing *under* full scale after starting over it, so the pre-encode
    warning was right and the caller's suppression hid the post-encode
    check that would have said so. The caller decides which case applies
    (`content_type`), not this function.

    Best effort. A file that will not probe is not a level problem, and a
    deliverable that is already written is not worth failing a finished run
    over.
    """
    if already_warned:
        return None
    try:
        from .media_info import probe_media

        info = probe_media(output_path) or {}
    except Exception:
        logger.debug(
            f"Could not measure the written level of {output_path}", exc_info=True
        )
        return None
    peak = info.get("peak_dbfs")
    if peak is None or peak < CLIPPED_WARN_DBFS:
        return peak
    name = os.path.basename(output_path)
    emit_warning(
        f"{name} decodes at {peak:+.2f} dBFS - above full scale, so it "
        f"clips on playback. The encode adds its own overshoot on top of "
        f"the level it was handed, so the fix is more headroom before the "
        f"file is written: a 'normalize_audio' step at 'peak_dbfs: -3' "
        f"ahead of the step that saves it. A mux into a video needs more "
        f"of it than an audio file does.",
        kind="audio_clipped",
        file=name,
        peak_dbfs=round(peak, 2),
    )
    return peak


def _file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def frames_for_encoding(frames):
    """Generated frames in the form `encode_video` encodes without first
    inspecting them.

    A pipeline that returns `output_type="np"` hands back float frames in
    [0, 1], and diffusers' `encode_video` establishes that range with three
    full-size temporaries - `np.zeros_like`, `np.ones_like` and the bool
    mask - before converting. On a 121-frame 960x544 clip that is ~3 GB of
    allocation and 16 s of wall clock on an idle box, against 2.4 s for the
    encode itself, and it is the bulk of a 'saving' phase that ran for 53 s
    with nothing else in it (#97, measured on lem 2026-09-14).

    Converting here is a pass and a half and hands back a torch tensor,
    which `encode_video` takes as given - so the check never runs. Frames
    outside [0, 1] are left exactly as they were: that is the branch where
    diffusers warns and treats them as pixel values already, and it is not
    a path any pipeline here produces or that this can be tested against.

    The source array is never written to - a later step may still read this
    result through a `previous_result:` reference, and the step cache
    retains it.
    """
    if not isinstance(frames, numpy.ndarray) or frames.size == 0:
        return frames
    if not numpy.issubdtype(frames.dtype, numpy.floating):
        return frames
    if float(frames.min()) < 0.0 or float(frames.max()) > 1.0:
        return frames
    denormalized = numpy.empty(frames.shape, dtype=numpy.uint8)
    # Frame by frame: the whole-array form allocates another copy the size
    # of the video, which is the cost this exists to avoid
    for index in range(frames.shape[0]):
        denormalized[index] = numpy.round(frames[index] * 255.0)
    return torch.from_numpy(denormalized)


def output_file_path(output_dir, file_name):
    """The path a result file is written to, confined to the output directory.

    Every part of the name is workflow-supplied - the workflow id, the step
    name, a result's file_base_name, and the keys of a dict artifact - and
    they are concatenated into a file name. Without this a name carrying a
    path separator would write outside the output directory, so the joined
    path goes through the same validator every other path in the engine does.

    A rerun that would otherwise produce a name already on disk gets a
    '-2', '-3', ... counter instead of silently overwriting it - the same
    guarantee ComfyUI's SaveImage node makes by scanning its output
    directory before every write.
    """
    candidate = validate_output_path(os.path.join(output_dir, file_name), output_dir)
    return _dedupe_existing_path(candidate)


def _dedupe_existing_path(path):
    """Append an incrementing counter before the extension until `path` is free."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    counter = 2
    while True:
        candidate = f"{base}-{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


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

# Distinguishes "the artifact has no such attribute" from "it has one holding None" -
# an AudioVideo whose pipeline reported no sample rate carries exactly that
_NO_PROPERTY = object()

# The names a modular pipeline's outputs go by. Asked for more than one output it returns
# them in a dict rather than on a pipeline output object, so its videos and the soundtrack
# generated alongside them arrive keyed instead of as attributes. Every diffusers modular
# pipeline (minimax_h3, ltx2, ...) names these "videos"/"audio"/"sampling_rate" - kept as
# tuples, rather than plain strings, only so the lookup goes through first_item like the
# other key sets. "audio_sample_rate" is real too: it is the name dw's own
# attach_audio_sample_rate (pipeline_processors/pipeline.py) gives the rate when it
# attaches it to a non-modular output - a modular result carrying it under that name is
# tested (TestModularOutputs.test_audio_sample_rate_names_the_rate_too) and kept for it.
MODULAR_VIDEO_KEYS = ("videos",)
MODULAR_AUDIO_KEYS = ("audio",)
MODULAR_SAMPLE_RATE_KEYS = ("sampling_rate", "audio_sample_rate")


class AudioVideo:
    """A generated video together with the audio track generated alongside it.

    Pipelines like LTX-2 return audio next to their frames. Keeping the two paired lets
    the result mux them into one file instead of dropping the audio on the floor.
    """

    def __init__(self, frames, audio, sample_rate, fps=None):
        """
        Args:
            frames: The video, as PIL images or an array of frames
            audio: Waveform for this video, shaped (channels, samples)
            sample_rate: Sample rate of the waveform, or None if the pipeline did not report one
            fps: Frame rate these frames are meant to play at, when something
                knows it - a joined video's own rate, or the rate of the file
                a task read. Carried for the same reason AudioTrack carries
                its sample rate: `result.fps` defaults to 8, and a step that
                joins 24 fps shots writing them at 8 is three times slow with
                its audio still the right length (#84). A declared
                `result.fps` still wins over this
        """
        self.frames = frames
        self.audio = audio
        self.sample_rate = sample_rate
        self.fps = fps


class AudioTrack:
    """A generated waveform together with the rate it was generated at.

    A step that produces audio alone usually returns the waveform by itself, and
    the workflow declares the rate - which is fine where the rate is a property of
    the workflow (a slice of a file it named) rather than of the model. It is not
    fine for a generated track: every text-to-speech model has its own rate, and a
    declared 44100 against a 24 kHz model plays the speech fast without failing.

    Carrying the rate with the waveform is what lets a workflow say nothing about
    it. Everything downstream of audio already reads '.audio' and '.sample_rate'
    off whatever it is handed - slice_audio, fade_audio, pair_audio and the H3
    audio references all accept one of these - and a rate the workflow does declare
    still wins over the one carried here.
    """

    def __init__(self, audio, sample_rate):
        """
        Args:
            audio: The waveform, shaped (channels, samples)
            sample_rate: Sample rate the waveform was generated at
        """
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
        self.saved_files = []
        # Set when a select step's Selected wrapper flows through
        # add_result - the winning position/score, replayable in the
        # manifest and step_end alongside the unwrapped value (#119)
        self.selected = None
        # Whether the file currently being written already drew a headroom
        # warning from the waveform it was handed, so the written-level
        # check does not say the same thing twice (#161)
        self._no_headroom_warned = False
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
        from .tasks.select import Selected

        if isinstance(result, Selected):
            self.selected = {"position": result.position, "score": result.score}
            result = result.value

        if isinstance(result, list):
            logger.debug(f"Adding {len(result)} results to result list")
            self.result_list.extend(result)
        else:
            if isinstance(result, str):
                # Clean up string results by removing extra quotes and whitespace
                result = result.strip().strip('"').strip()
            logger.debug("Adding single result to result list")
            self.result_list.append(result)

    @property
    def retainable(self):
        """Whether the step cache may keep this Result's result_list.

        A chain step's save_segments spills each segment to disk and
        replays it lazily through a SegmentedFrames; save() removes those
        files once the video is written (SegmentedFrames.cleanup(), called
        from save_audio_video below). A Result cached after that point would
        point a later cache hit at files that are already gone, so a
        result_list holding such an artifact is not retainable - checked by
        duck-typed attribute rather than isinstance, so this module need not
        import pipeline_processors.chain.
        """
        for result in self.result_list:
            frames = getattr(result, "frames", None)
            if getattr(frames, "cleaned", False):
                return False
        return True

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

        A dict result is looked up by key; anything else by attribute, which is how
        a step reaches into an artifact that is an object rather than a mapping -
        the frames or the soundtrack of the AudioVideo a video-with-audio pipeline
        produces, say, where the next step takes one of them on its own. Methods are
        not properties: 'previous_result:step.index' on a list result names nothing
        the workflow meant, so it fails rather than passing a bound method along.

        Args:
            property_name: Name of property to extract from results

        Returns:
            List of property values from results where property exists

        Raises:
            ValueError: If a result is neither a dict-like (Mapping) object with that
                key nor an object carrying it as a data attribute - a plain string or
                other scalar result has no properties to look up, and staying quiet
                about that (or doing a membership/substring test instead of a key
                lookup) would silently drop data or raise a confusing TypeError.
        """
        values = []
        for result in self.result_list:
            if isinstance(result, Mapping):
                if property_name in result:
                    values.append(result[property_name])
                continue

            value = getattr(result, property_name, _NO_PROPERTY)
            # A string's every 'property' is a method, and so is most of a list's -
            # the original loud failure for those is the useful answer
            if value is _NO_PROPERTY or callable(value):
                raise ValueError(
                    f"result has no property '{property_name}' "
                    f"(it is a {type(result).__name__}, not a dict)"
                )
            values.append(value)

        logger.debug(f"Retrieved {len(values)} values for property: {property_name}")
        return values

    def save(self, output_dir, default_base_name):
        """Save results to files based on content type.

        Args:
            output_dir: Directory to save files in
            default_base_name: Default name to use for files

        Returns:
            List of file paths written, in the order they were written - the
            step's manifest. Empty when saving is disabled or nothing saved.
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
            self.saved_files = []
            return self.saved_files

        # Determine base filename with validation. A file_base_name *replaces*
        # the derived name - it is set to get a name the caller can predict, and
        # gluing it onto the name it was meant to replace made it neither (#100).
        # What the derived name guaranteed - a distinct name per step - is then
        # the caller's to keep; output_file_path's counter catches a collision.
        file_base_name = validated_base_name
        if "file_base_name" in self.result_definition:
            file_base_name = validate_file_base_name(
                validate_string_input(
                    self.result_definition["file_base_name"],
                    max_length=MAX_BASE_NAME_LENGTH,
                )
            )

        # Get file extension for content type
        extension = guess_extension(content_type)
        logger.debug(
            f"Saving with content type: {content_type}, extension: {extension}"
        )

        # Encoding a video here is minutes of work after the last denoise
        # step, with the step's bar sitting full
        emit_phase("saving", detail=content_type)

        # Save each result, collecting the paths written as the step's manifest
        saved_files = []
        for i, result in enumerate(self.result_list):
            if content_type.endswith("json"):
                # Handle JSON content type
                output_path = output_file_path(
                    validated_output_dir, f"{file_base_name}-{i}{extension}"
                )
                logger.info(f"Saving JSON result to {output_path}")
                with open(output_path, "w") as file:
                    file.write(json.dumps(result, indent=4))
                saved_files.append(output_path)
            else:
                # Handle other content types
                for j, artifact in enumerate(get_artifact_list(result)):
                    saved_files.extend(
                        self.save_artifact(
                            validated_output_dir,
                            artifact,
                            f"{file_base_name}-{i}.{j}",
                            content_type,
                            extension,
                        )
                    )
        self.saved_files = saved_files
        return saved_files

    def save_artifact(
        self, output_dir, artifact, file_base_name, content_type, extension
    ):
        """Save individual artifact to file based on its type.

        Args:
            output_dir: Directory to save file in, already validated by save()
            artifact: The artifact to save
            file_base_name: Base name for the file, derived from names save()
                validated
            content_type: MIME type of the content
            extension: File extension to use

        Returns:
            List of file paths written.
        """
        if artifact is None:
            logger.warning(f"Skipping None artifact for {file_base_name}")
            return []

        if isinstance(artifact, (int, float, bool)):
            # Static validation (dw/scalar_result_validation.py, #212) refuses
            # a 'result' block on a command declared to return a scalar, but
            # a command name reached only through a 'variable:' is literal
            # only at run time and so invisible to that check - this is the
            # same refusal for the one path that can still get here, naming
            # the value rather than failing inside soundfile/PIL/open() with
            # a bare TypeError after the step's own work is already done
            raise ValueError(
                f"'{file_base_name}' is a {type(artifact).__name__} ({artifact!r}), "
                "not an artifact - a 'result' block cannot save it"
            )

        if isinstance(artifact, dict):
            # Recursively save dictionary items
            logger.debug(
                f"Saving dictionary artifact with keys: {list(artifact.keys())}"
            )
            saved_files = []
            for k, v in artifact.items():
                saved_files.extend(
                    self.save_artifact(
                        output_dir,
                        v,
                        f"{file_base_name}-{k}",
                        content_type,
                        extension,
                    )
                )
            return saved_files

        output_path = output_file_path(output_dir, f"{file_base_name}{extension}")
        logger.info(f"Saving artifact to {output_path}")
        # Writing one file is the whole of the 'saving' phase's wall clock,
        # and on a video it is minutes of it with nothing else to report -
        # the denoise counter is frozen at its last step and there is no
        # further event until step_end, so a healthy run is indistinguishable
        # from a hung one (#97). Name the file as it starts and report what
        # it cost as it finishes, the same shape the modular block lead-in
        # got in #95
        emit_log(
            f"writing {os.path.basename(output_path)}"
            + (f" ({_artifact_size(artifact)})" if _artifact_size(artifact) else ""),
            file=os.path.basename(output_path),
            content_type=content_type,
        )
        started = time.monotonic()
        self._no_headroom_warned = False

        try:
            if content_type.startswith("video"):
                if isinstance(artifact, AudioVideo):
                    self.save_audio_video(artifact, output_path, content_type)
                else:
                    export_to_video(artifact, output_path, fps=self.video_fps(artifact))
            elif content_type == "image/gif":
                export_to_gif(artifact, output_path, fps=self.video_fps(artifact))
            elif content_type.startswith("audio"):
                waveforms = normalize_audio(artifact)
                # Declared rate > the rate a generated track carries > default
                declared_rate = self.result_definition.get("sample_rate")
                carried_rate = getattr(artifact, "sample_rate", None)
                # A template's 'result.sample_rate' relabels the file at save
                # time exactly the way a task argument's 'sample_rate' does -
                # and #180's guard only caught the argument, not this. A
                # caller who passed the source's own correct rate as the
                # argument (so the argument-level check is clean) still got
                # the wrong file with `warnings: []` when the *result* block
                # hardcoded a different rate (#205). Same warning either way.
                if (
                    declared_rate is not None
                    and carried_rate is not None
                    and declared_rate != carried_rate
                ):
                    from .tasks.audio_utils import _warn_on_rate_override

                    _warn_on_rate_override("save_artifact", carried_rate, declared_rate)
                sample_rate = declared_rate or carried_rate or DEFAULT_AUDIO_SAMPLE_RATE
                # A batched waveform holds several songs - save each one separately
                if len(waveforms) > 1:
                    saved_files = []
                    for k, waveform in enumerate(waveforms):
                        saved_files.extend(
                            self.save_artifact(
                                output_dir,
                                # Keep the rate the track carries across the
                                # recursion - a bare waveform would fall back
                                # to the default
                                (
                                    AudioTrack(waveform.T, sample_rate)
                                    if getattr(artifact, "sample_rate", None)
                                    is not None
                                    else waveform
                                ),
                                f"{file_base_name}-{k}",
                                content_type,
                                extension,
                            )
                        )
                    return saved_files
                self._no_headroom_warned = (
                    warn_without_headroom(waveforms[0], os.path.basename(output_path))
                    is not None
                )
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

        # The level of what was actually written, which is the only one the
        # consumer will hear: the encode's own overshoot sits between the
        # waveform `warn_without_headroom` measured and this (#161). Only a
        # file that can carry a soundtrack, so an image never pays a probe.
        # The pre-encode warning only suppresses this for a plain audio
        # save - a video mux's overshoot is not reliably positive (#174),
        # so a video always gets the ground-truth post-encode check
        if content_type.startswith("audio") or content_type.startswith("video"):
            warn_if_written_above_full_scale(
                output_path,
                already_warned=(
                    self._no_headroom_warned
                    if content_type.startswith("audio")
                    else False
                ),
            )

        emit_log(
            f"wrote {os.path.basename(output_path)} in "
            f"{time.monotonic() - started:.1f}s ({_file_size_mb(output_path):.1f} MB)",
            file=os.path.basename(output_path),
            seconds=round(time.monotonic() - started, 1),
        )
        return [output_path]

    def video_fps(self, artifact):
        """The frame rate this video is written at.

        Declared `result.fps` first, then the rate the artifact carries (a
        join's own rate, or the rate of the files it read), then 8.

        The order matters more than it looks: `result.fps` and a task's own
        `fps` argument are separate knobs, and the one an author thinks to
        set is the task's. A step that told `concat_videos` its shots are 24
        fps and said nothing on `result` used to write them at 8 - the
        picture three times long against an audio track still the right
        length, with nothing said about it (#84). A workflow that does
        declare `result.fps` still wins, so writing at a rate other than the
        source's - a deliberate slow motion - stays available, and says so.
        """
        declared = self.result_definition.get("fps")
        carried = getattr(artifact, "fps", None)
        if declared is None:
            return carried or DEFAULT_VIDEO_FPS
        if carried and abs(declared - carried) > 0.01:
            emit_warning(
                f"Writing video at {declared} fps, but the frames it was "
                f"given run at {carried} fps - the file will play "
                f"{declared / carried:.2g}x speed "
                f"({carried / declared:.2g} times as long). Drop 'fps' from "
                f"the step's result to keep the source rate",
                kind="fps_mismatch",
                declared_fps=declared,
                source_fps=carried,
            )
        return declared

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
        fps = self.video_fps(artifact)
        # The pipeline reports the sample rate of what it generated - the result
        # definition can still override it
        sample_rate = self.result_definition.get(
            "audio_sample_rate", artifact.sample_rate
        )
        audio = artifact.audio
        # Frames generated in memory (a previous_result-chained shot, a modular
        # pipeline's own output) never went through _decode_audio_video, so a
        # codec-padding mismatch between the audio and the frame count survives
        # here instead of being trimmed there (#197). Segment-backed frames are
        # written by the chain pipeline processor, which already carries its own
        # fps and does its own segment-length accounting - left alone.
        #
        # Written back onto the artifact, not just the local used for muxing:
        # this same AudioVideo instance is what a later previous_result:
        # consumer (concat_videos, dissolve_videos) reads directly out of the
        # result store, so a local-only fit left the artifact's own audio
        # unfitted and a chain built from in-memory shots still drifted even
        # though each shot's own saved file was correct (#197 reopened).
        if audio is not None and sample_rate is not None and fps and not hasattr(
            artifact.frames, "cleanup"
        ):
            from .tasks.video_utils import _fit_audio_to_frames

            audio = _fit_audio_to_frames(audio, len(artifact.frames), fps, sample_rate)
            artifact.audio = audio

        # Segment-backed frames (a chained step with save_segments) replay from
        # disk one segment at a time, so the final video is streamed instead of
        # materialized - and the segment files are removed once it is written
        if hasattr(artifact.frames, "cleanup"):
            if content_type != MUXED_VIDEO_CONTENT_TYPE:
                raise ValueError(
                    f"Segment-backed video can only be written as "
                    f"{MUXED_VIDEO_CONTENT_TYPE}, not {content_type}"
                )
            if not is_av_available():
                raise ValueError(
                    "Writing segment-backed video needs PyAV - install it "
                    "with: pip install av"
                )

            audio = None
            if artifact.audio is not None and sample_rate is not None:
                audio = as_audio_track(artifact.audio)
                self._no_headroom_warned = (
                    warn_without_headroom(artifact.audio, os.path.basename(output_path))
                    is not None
                )
            logger.debug(
                f"Streaming {len(artifact.frames)} segments into {output_path}"
            )
            encode_video(
                iter(artifact.frames),
                fps=fps,
                output_path=output_path,
                audio=audio,
                audio_sample_rate=sample_rate if audio is not None else None,
                video_chunks_number=len(artifact.frames),
            )
            artifact.frames.cleanup()
            return

        reason = None
        if audio is None:
            reason = "the pipeline returned no audio"
        elif sample_rate is None:
            reason = "the audio sample rate is unknown"
        elif content_type != MUXED_VIDEO_CONTENT_TYPE:
            reason = f"audio can only be muxed into {MUXED_VIDEO_CONTENT_TYPE}"
        elif not is_av_available():
            reason = "PyAV is not installed - install it with: pip install av"

        if reason is not None:
            # No audio at all is an expected shape - video-only chains and
            # concatenations - so it logs quietly; losing audio we do have warns
            log = logger.debug if audio is None else logger.warning
            log(f"Saving {output_path} without its audio because {reason}")
            export_to_video(artifact.frames, output_path, fps=fps)
            return

        logger.debug(f"Muxing audio at {sample_rate}Hz into {output_path}")
        self._no_headroom_warned = (
            warn_without_headroom(audio, os.path.basename(output_path))
            is not None
        )
        encode_video(
            frames_for_encoding(artifact.frames),
            fps=fps,
            output_path=output_path,
            audio=as_audio_track(audio),
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
                import piexif.helper

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


def read_embedded_metadata(path):
    """The generation metadata a saved image carries, or None.

    The read-side mirror of _save_image_with_metadata: the 'parameters' PNG
    text chunk, or the EXIF UserComment for JPEG/WebP. Returns the parsed
    dict, or None when the file has no metadata this writer produced.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            text = getattr(image, "text", {}).get("parameters")
            if text is None and "exif" in getattr(image, "info", {}):
                import piexif
                import piexif.helper

                exif = piexif.load(image.info["exif"])
                comment = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
                if comment:
                    text = piexif.helper.UserComment.load(comment)
        if text is None:
            return None
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logger.debug(f"No readable metadata in {path}: {e}")
        return None


def _frames_from_attributes(result):
    """The frames extractor for a pipeline output that carries `.frames` directly.

    Some video pipelines (LTX-2) generate an audio track along with the frames, exposed
    as `.audio` and `.audio_sample_rate` attributes alongside `.frames`.
    """
    return frames_with_audio(
        result.frames,
        getattr(result, "audio", None),
        getattr(result, "audio_sample_rate", None),
    )


def _audios_from_attribute(result):
    """Each `.audios` item as an artifact.

    With the rate attach_audio_sample_rate recorded, each item becomes an
    AudioTrack carrying it, shaped (channels, samples); without one, the bare
    (samples, channels) array it always was, and the workflow's 'sample_rate'
    (or the default) applies at save.
    """
    sample_rate = getattr(result, "audio_sample_rate", None)
    if sample_rate is None:
        return [as_waveform_array(audio) for audio in result.audios]
    return [
        AudioTrack(as_waveform_array(audio).T, int(sample_rate))
        for audio in result.audios
    ]


# Diffusers output fields get_artifact_list knows how to turn into artifacts, tried in
# this order. A result is dispatched to the first field it has - images wins over frames
# if a result somehow has both, matching the fixed hasattr chain this replaced. Supporting
# a new diffusers output field (e.g. a standalone "depth" attribute) is one more entry
# here, instead of another branch threaded through the chain.
OUTPUT_FIELD_EXTRACTORS = [
    ("images", lambda result: result.images),
    ("image_embeds", lambda result: result.image_embeds),
    ("image_embeddings", lambda result: result.image_embeddings),
    ("frames", _frames_from_attributes),
    ("audios", _audios_from_attribute),
]


def get_artifact_list(result):
    """Extract list of artifacts from a result object.

    Handles various result types including images, embeddings, frames, and audio.

    Args:
        result: Result object to extract artifacts from

    Returns:
        List of artifacts
    """
    # Already a paired video and audio track - it has a .frames attribute of its own,
    # but the pair is one artifact, not something to run back through frame extraction
    if isinstance(result, AudioVideo):
        return [result]

    for field_name, extract in OUTPUT_FIELD_EXTRACTORS:
        if hasattr(result, field_name):
            return extract(result)

    if isinstance(result, dict):
        # A modular pipeline asked for several outputs returns them keyed
        artifacts = modular_artifacts(result)
        if artifacts is not None:
            return artifacts

    if isinstance(result, list):
        return result

    if hasattr(result, "to_tuple") or hasattr(result, "__dataclass_fields__"):
        # A diffusers output (BaseOutput subclasses have to_tuple; plain dataclasses
        # have __dataclass_fields__) whose fields matched none of the extractors above -
        # log what it actually looks like so the resulting content-type mismatch on save
        # is diagnosable instead of a bare "does not match result type" surprise.
        logger.warning(
            f"Don't know how to extract artifacts from a {type(result).__name__} - "
            f"treating it as a single artifact. Its fields are: {output_field_names(result)}"
        )

    return [result]


def output_field_names(result):
    """Best-effort list of field names on a diffusers-style output object, for logging."""
    if hasattr(result, "keys"):
        return list(result.keys())
    if hasattr(result, "__dataclass_fields__"):
        return list(result.__dataclass_fields__.keys())
    return []


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
    sample_rate = None
    if audio is not None:
        consumed_keys.add(audio_key)
        rate_key, sample_rate = first_item(result, MODULAR_SAMPLE_RATE_KEYS)
        consumed_keys.add(rate_key)

    artifacts = frames_with_audio(videos, audio, sample_rate)

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


def frames_with_audio(frames, audio, sample_rate):
    """Pair frames with the audio track generated alongside them, if there is one.

    The one place that decides whether frames need pairing with audio at all - used by
    both routes a pipeline's frames-plus-audio output can take: attributes on a pipeline
    output object (`_frames_from_attributes`), and keys in the dict a modular pipeline
    returns (`modular_artifacts`). Frames without audio are returned unchanged; actually
    pairing them is `pair_audio_with_frames`'s job.

    Args:
        frames: The generated video(s), one list of frames per generation
        audio: The generated waveform(s), or None if the pipeline produced no audio
        sample_rate: Sample rate of the waveform(s), or None if unknown

    Returns:
        `frames` unchanged if `audio` is None, otherwise the list of AudioVideo pairs
        `pair_audio_with_frames` produces
    """
    if audio is None:
        return frames
    return pair_audio_with_frames(frames, audio, sample_rate)


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


def as_waveform_array(audio):
    """Transpose one batch item of a pipeline's `.audios` output to (samples, channels).

    `.audios` is shaped (batch, channels, samples). Under the diffusers default
    `output_type='np'`, pipelines such as AudioLDM2 and StableAudio already call
    `.numpy()` before returning, so each item here is a numpy ndarray rather than a
    torch tensor - it has no `.float()`/`.cpu()` methods, only `.T`/`.astype()`.

    Args:
        audio: One batch item, shaped (channels, samples), as a torch tensor or numpy array

    Returns:
        Numpy float32 array shaped (samples, channels)
    """
    if isinstance(audio, torch.Tensor):
        return audio.T.float().cpu().numpy()

    return numpy.asarray(audio).T.astype(numpy.float32, copy=False)


def as_audio_track(audio):
    """Convert a generated waveform into the tensor encode_video expects.

    encode_video wants a float torch tensor on the CPU shaped (channels, samples) -
    pipelines hand back bfloat16 tensors that are still on the GPU, or numpy arrays.

    A mono track is duplicated into two channels, because the mp4 audio stream
    takes nothing else: diffusers' _write_audio refuses any other channel count
    with a raw tensor shape, and a mono voice track paired onto a picture is the
    ordinary case, not an edge one (#106). Duplicating one channel is lossless,
    but it is a change to what was handed in, so it warns.

    Args:
        audio: Waveform as a torch tensor or numpy array

    Returns:
        Float CPU torch tensor holding the waveform, with at least 2 channels
    """
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(numpy.asarray(audio))

    audio = audio.detach().float().cpu()
    return _as_stereo(audio)


def _as_stereo(audio):
    """Duplicate a mono waveform into two channels, leaving anything else alone.

    Orientation is read the way encode_video reads it: (samples,) and (1, samples)
    are mono, and so is (samples, 1) - a lone channel laid out samples-first.
    A 2-channel waveform in either orientation is already what the encoder takes
    and is handed back untouched, so a short stereo track is never mistaken for
    many channels of one sample.
    """
    if audio.ndim == 2:
        if audio.shape[0] == 2 or audio.shape[1] == 2:
            return audio
        if audio.shape[0] == 1:
            audio = audio[0]
        elif audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            return audio
    elif audio.ndim != 1:
        return audio

    emit_warning(
        f"Duplicating a mono audio track of {audio.shape[0]} samples into two "
        f"channels - an mp4 audio stream takes stereo and nothing else"
    )
    return audio.unsqueeze(0).repeat(2, 1)


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
        artifact: Audio waveform(s) as a torch tensor or numpy array, or anything
            carrying one as '.audio' - an AudioTrack, or a video whose soundtrack
            is what is being saved

    Returns:
        List of numpy arrays shaped (samples,) or (samples, channels)
    """
    if hasattr(artifact, "audio"):
        if artifact.audio is None:
            raise ValueError(
                f"Cannot save a {type(artifact).__name__} as audio - it carries no audio track"
            )
        artifact = artifact.audio

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
