import os
import time
import torch
import json
import logging
from diffusers.utils import (
    export_to_video,
    export_to_gif,
    encode_video,
    is_av_available,
)
from collections.abc import Mapping
from .content_types import (
    AUDIO_FORMATS,
    LOSSY_AUDIO_CONTENT_TYPES,
    MUXED_VIDEO_CONTENT_TYPE,
    guess_extension,
    refuse_active_content_type,
)
from .audio_qc import check_written_media, warn_without_headroom
from .events import emit_log, emit_phase, emit_warning
from .media_types import (
    AudioTrack,
    AudioVideo,
    Selected,
    fit_codec_padding,
    sample_axis,
    warn_on_rate_override,
)
from .output_extraction import get_artifact_list
from .writers import (
    _artifact_size,
    _file_size_mb,
    as_audio_track,
    embed_image_metadata,
    flatten_alpha_for,
    frames_for_encoding,
    normalize_audio,
    output_file_path,
    write_audio,
    write_json_file,
    write_text_file,
)
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


# Result definition keys passed through to soundfile - encoding quality controls
AUDIO_WRITE_ARGUMENTS = ["subtype", "format", "compression_level", "bitrate_mode"]


# Distinguishes "the artifact has no such attribute" from "it has one holding None" -
# an AudioVideo whose pipeline reported no sample rate carries exactly that
_NO_PROPERTY = object()


def _refuse_scalar_artifact(artifact, file_base_name):
    """Refuse a number or bool: it is not an artifact."""
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


class Result:
    """Manages and stores results from workflow steps.

    Handles result storage, artifact management, and file saving with support
    for multiple content types including images, video, audio, and JSON.
    """

    def __init__(self, result_definition, consumed_by_normalizer=False):
        """Initialize Result with configuration for how to handle/save results.

        Args:
            result_definition: Dict containing result configuration including:
                - content_type: MIME type of the result
                - save: Boolean indicating if result should be saved
                - file_base_name: Base name for saved files
            consumed_by_normalizer: Whether a later step resets this result's
                level (normalize_audio/match_levels) before anything ships
                it - when true, this save's own headroom is not a deliverable
                concern (#286)
        """
        self.result_definition = result_definition
        self._consumed_by_normalizer = consumed_by_normalizer
        self.result_list = []
        self.metadata = None
        self.saved_files = []
        # The shots (dw/shots.py) of each saved file whose artifact carried
        # any, keyed by the path in saved_files - plain data, so a step cache
        # hit's stripped copy still reports them (#378)
        self.saved_shots = {}
        # Set when a select step's Selected wrapper flows through
        # add_result - the winning position/score, replayable in the
        # manifest and step_end alongside the unwrapped value (#119)
        self.selected = None
        # Whether the file currently being written already drew a headroom
        # warning from the waveform it was handed, so the written-level
        # check does not say the same thing twice (#161)
        self._no_headroom_warned = False
        # The predicted peak (dBFS) from the pre-encode headroom check, paired
        # with _no_headroom_warned above; only ever set inside save_artifact,
        # initialized here so a caller reading it before that runs gets None
        # rather than an AttributeError.
        self._predicted_peak_dbfs = None
        # get_artifact_list(result) memoized by id(result) - see
        # _artifacts_for. Keeps result_list itself untouched, so
        # Result.retainable's attribute walk over result_list is unaffected.
        self._artifact_cache = {}
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
            artifacts.extend(self._artifacts_for(result))

        logger.debug(f"Retrieved {len(artifacts)} artifacts from results")
        return artifacts

    def _artifacts_for(self, result):
        """get_artifact_list(result), memoized by identity of `result`.

        save() extracts an in-memory pipeline output's artifacts and mutates
        one in place to fit generated audio to the frame count (#197). A
        later get_artifacts() call - a previous_result: consumer reading a
        chained shot directly, bypassing any output: round trip - must see
        that same fitted object rather than a fresh, unfitted extraction
        from the same raw pipeline output, which get_artifact_list would
        otherwise rebuild every time it runs.
        """
        key = id(result)
        if key not in self._artifact_cache:
            self._artifact_cache[key] = get_artifact_list(result)
        return self._artifact_cache[key]

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
        missing = None
        for result in self.result_list:
            if isinstance(result, Mapping):
                if property_name in result:
                    values.append(result[property_name])
                else:
                    missing = missing or result
                continue

            value = getattr(result, property_name, _NO_PROPERTY)
            # A string's every 'property' is a method, and so is most of a list's -
            # the original loud failure for those is the useful answer
            if value is _NO_PROPERTY or callable(value):
                missing = missing or result
                continue
            values.append(value)

        # Only when no result carries the property is the missing one an error
        # (#499). An empty list is not "nothing to do": the step reading it gets
        # zero iterations and succeeds having written nothing - a modular step's
        # raw output dict holds 'sampling_rate', and asking it for 'sample_rate'
        # silently skipped the mux that depended on it
        if missing is not None and not values:
            if isinstance(missing, Mapping):
                keys = ", ".join(sorted(str(key) for key in missing)) or "none"
                raise ValueError(
                    f"result has no property '{property_name}' "
                    f"(it is a dict with keys: {keys})"
                )
            raise ValueError(
                f"result has no property '{property_name}' "
                f"(it is a {type(missing).__name__}, not a dict)"
            )

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
            self.saved_shots = {}
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

        # The same refusal validation makes, for a definition that reached
        # the writer without it
        refuse_active_content_type(content_type)

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
        saved_shots = {}
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
                for j, artifact in enumerate(self._artifacts_for(result)):
                    paths = self.save_artifact(
                        validated_output_dir,
                        artifact,
                        f"{file_base_name}-{i}.{j}",
                        content_type,
                        extension,
                    )
                    shots = getattr(artifact, "shots", None)
                    if shots:
                        saved_shots.update((path, shots) for path in paths)
                    saved_files.extend(paths)
        self.saved_files = saved_files
        self.saved_shots = saved_shots
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

        _refuse_scalar_artifact(artifact, file_base_name)

        if isinstance(artifact, dict):
            return self._save_mapping_artifact(
                output_dir, artifact, file_base_name, content_type, extension
            )

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
        self._predicted_peak_dbfs = None

        try:
            batched = self._write_artifact_file(
                output_dir,
                artifact,
                file_base_name,
                content_type,
                extension,
                output_path,
            )
            if batched is not None:
                return batched
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
            check_written_media(
                output_path,
                artifact,
                content_type,
                video_fps=self.video_fps,
                consumed_by_normalizer=self._consumed_by_normalizer,
                headroom_warned=self._no_headroom_warned,
                predicted_peak_dbfs=self._predicted_peak_dbfs,
            )

        emit_log(
            f"wrote {os.path.basename(output_path)} in "
            f"{time.monotonic() - started:.1f}s ({_file_size_mb(output_path):.1f} MB)",
            file=os.path.basename(output_path),
            seconds=round(time.monotonic() - started, 1),
        )
        return [output_path]

    def _save_mapping_artifact(
        self, output_dir, artifact, file_base_name, content_type, extension
    ):
        """Save a dict artifact one entry per file, named `<base>-<key>`."""
        # Recursively save dictionary items
        logger.debug(f"Saving dictionary artifact with keys: {list(artifact.keys())}")
        saved_files = []
        for k, v in artifact.items():
            if isinstance(v, torch.Tensor) and not content_type.startswith("audio"):
                # A modular pipeline's leftover output not part of the
                # video/audio pairing (dw's own 'latents', from an H3
                # upscale step's output: [..., "latents"]) is raw model
                # state, not media - it has no video/image/json rendering
                # under the step's declared content_type, and trying one
                # crashed the exporter deep inside its own error (#507).
                # It stays reachable as previous_result:<step>.<key>
                # straight off the in-memory result; only the file write
                # here is skipped.
                emit_warning(
                    f"'{k}' in '{file_base_name}' is a raw tensor, not "
                    f"media - skipped saving it under content_type "
                    f"{content_type!r}. It is still available as "
                    f"previous_result:<step>.{k}.",
                    kind="non_media_artifact_skipped",
                    key=k,
                    content_type=content_type,
                )
                continue
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

    def _write_artifact_file(
        self, output_dir, artifact, file_base_name, content_type, extension, output_path
    ):
        """Write one artifact by content type.

        Returns the saved paths for a batched audio waveform (each song went
        through `save_artifact` itself), else None.
        """
        if content_type.startswith("video"):
            self._write_video_file(artifact, output_path, content_type)
        elif content_type == "image/gif":
            export_to_gif(artifact, output_path, fps=self.video_fps(artifact))
        elif content_type.startswith("audio"):
            return self._write_audio_file(
                output_dir,
                artifact,
                file_base_name,
                content_type,
                extension,
                output_path,
            )
        elif content_type.endswith("json"):
            write_json_file(artifact, output_path)
        elif content_type.startswith("text"):
            write_text_file(artifact, output_path, file_base_name, content_type)
        elif hasattr(artifact, "save"):
            self._write_saveable(artifact, output_path, content_type)
        else:
            raise ValueError(
                f"Content type {content_type} does not match result type {type(artifact)}"
            )
        return None

    def _write_video_file(self, artifact, output_path, content_type):
        if isinstance(artifact, AudioVideo):
            self.save_audio_video(artifact, output_path, content_type)
        else:
            export_to_video(artifact, output_path, fps=self.video_fps(artifact))

    def _write_audio_file(
        self, output_dir, artifact, file_base_name, content_type, extension, output_path
    ):
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
            warn_on_rate_override("save_artifact", carried_rate, declared_rate)
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
                            if getattr(artifact, "sample_rate", None) is not None
                            else waveform
                        ),
                        f"{file_base_name}-{k}",
                        content_type,
                        extension,
                    )
                )
            return saved_files
        self._no_headroom_warned = (
            False
            if self._consumed_by_normalizer
            else warn_without_headroom(
                waveforms[0],
                os.path.basename(output_path),
                lossless=content_type not in LOSSY_AUDIO_CONTENT_TYPES,
            )
            is not None
        )
        write_audio(
            output_path,
            waveforms[0],
            sample_rate,
            **self.get_audio_write_arguments(content_type),
        )
        return None

    def _write_saveable(self, artifact, output_path, content_type):
        """Save anything with a `.save`: flatten alpha, embed metadata.

        The flattened image stays local - the caller's `artifact` is never
        rebound, and an image skips the post-write checks that read it.
        """
        if content_type.startswith("image/"):
            artifact = flatten_alpha_for(
                artifact, content_type, os.path.basename(output_path)
            )
        if (
            self.metadata is not None
            and self.result_definition.get("embed_metadata", False)
            and content_type.startswith("image/")
        ):
            embed_image_metadata(artifact, output_path, content_type, self.metadata)
        else:
            artifact.save(output_path)

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

    def conform_artifact(self, artifact):
        """Stamp this result's declared fps onto artifact and fit its audio
        to its own frame count. Returns the resolved fps.

        save_audio_video applies this right before writing. A composed
        child's own last step never reaches that write when its parent owns
        saving (#92, parent_saves_this in workflow.py) - so without this
        method a child's declared fps, and its own audio-to-frames fit,
        would never reach the artifact a later previous_result: consumer
        (concat_videos, dissolve_videos, or the parent's own save) reads
        (#561). Called from both places so the two cannot drift.
        """
        fps = self.video_fps(artifact)
        # A declared result.fps is the rate this video now plays at, so it is
        # written back onto the artifact the way the fitted audio below is: a
        # later previous_result: consumer reads this same instance, and a 24
        # fps shot written at 12 still told join_into_song it was 24 - its
        # frames silently re-timed rather than refused (#513). A cache hit or
        # an output: reload already reads the rate off the written file
        if self.result_definition.get("fps") is not None:
            artifact.fps = fps
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
        if (
            audio is not None
            and sample_rate is not None
            and fps
            and not hasattr(artifact.frames, "cleanup")
        ):
            fitted = fit_codec_padding(audio, len(artifact.frames), fps, sample_rate)
            axis = sample_axis(audio)
            if (
                axis is not None
                and fitted.shape[axis] != audio.shape[axis]
                and artifact.shots
            ):
                # The fit trims or pads to the frames' own duration - a shot
                # map measured against the pre-fit track (#426) now overruns
                # or falls short of what actually gets written, so it is
                # re-measured against the same length the mux will see
                from .shots import measured_num_samples

                measured_num_samples(artifact.shots, fitted.shape[axis])
            artifact.audio = fitted
        return fps

    def conform_artifacts(self):
        """conform_artifact for every AudioVideo this result holds, and
        replace result_list with the extracted, conformed artifacts.

        Called in place of a skipped save (parent_saves_this, workflow.py)
        so a composed child that never writes its own file still leaves its
        declared fps and frame-fitted audio on the artifacts it hands up.
        Workflow.run returns this Result's result_list to the parent step's
        Step.run, which folds it into its own, separate Result via
        add_result - a fresh _artifact_cache, keyed by id() of the raw
        pipeline output. Stamping only the artifact get_artifacts() returns
        (as before) leaves result_list holding the original raw output
        (frames/audio attributes, no fps of its own); the parent's own
        get_artifact_list() then rebuilds a brand new, unstamped AudioVideo
        from it and falls back to DEFAULT_VIDEO_FPS (#561, the deployed
        #561 fix that didn't hold). Replacing result_list's entries with the
        conformed artifacts themselves means the parent receives these exact
        instances, and get_artifact_list's AudioVideo passthrough
        (isinstance(result, AudioVideo): return [result]) hands them back
        unchanged instead of re-extracting.
        """
        conformed = []
        for result in self.result_list:
            artifacts = self._artifacts_for(result)
            for artifact in artifacts:
                if isinstance(artifact, AudioVideo):
                    self.conform_artifact(artifact)
            conformed.extend(artifacts)
        self.result_list = conformed

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
        fps = self.conform_artifact(artifact)
        sample_rate = self.result_definition.get(
            "audio_sample_rate", artifact.sample_rate
        )
        audio = artifact.audio

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
                self._predicted_peak_dbfs = warn_without_headroom(
                    artifact.audio, os.path.basename(output_path), emit=False
                )
                self._no_headroom_warned = self._predicted_peak_dbfs is not None
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
        self._predicted_peak_dbfs = warn_without_headroom(
            audio, os.path.basename(output_path), emit=False
        )
        self._no_headroom_warned = self._predicted_peak_dbfs is not None
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
