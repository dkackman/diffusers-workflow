"""A chain's segments on disk (`save_segments`, dw/pipeline_processors/chain.py):
`SegmentSpill` writes each finished segment as a playable mp4 beside the
workflow's outputs, `SegmentedFrames` replays those files one segment at a time
when the final video is written, and `SavedFrames` is what a later step reads
once the segment files are gone - the finished video, decoded on first use.
`chain.py` re-exports the three; `dw/result.py`, `dw/previous_results.py` and
`dw/tasks/assess.py` import them from there.
"""

import logging
import os
from collections.abc import Sequence

import numpy
import torch
from PIL import Image
from diffusers.utils import is_av_available

from ..writers import encode_video, frames_for_encoding, output_file_path

logger = logging.getLogger("dw")


class SegmentedFrames:
    """Chained segments spilled to disk, replayed at save time.

    Iterating yields one uint8 (frames, height, width, 3) torch tensor per
    segment file - the chunk shape encode_video streams from - so the final
    video is written holding only one segment in memory at a time.
    """

    def __init__(self, paths, total_frames=None, keep_files=False, stored_frames=None):
        """
        Args:
            paths: The segment files, in output order
            total_frames: Frames to yield in total - the match_audio tail trim.
                None yields every stored frame
            keep_files: Leave the segment files in place after cleanup()
            stored_frames: Frames across all the files, when known - what
                frame_count reports, since len() counts files
        """
        self.paths = list(paths)
        self.stored_frames = stored_frames
        self.total_frames = total_frames
        self.keep_files = keep_files
        # True once cleanup() has actually removed the files (not when
        # keep_files skipped that) - a cached Result pointing at a cleaned-up
        # SegmentedFrames must not be served to a later miss, since the files
        # a replay would open are gone. See Result.retainable.
        self.cleaned = False

    def __len__(self):
        """Chunk count - one per segment file."""
        return len(self.paths)

    @property
    def frame_count(self):
        """Frames the replay yields, or None when the files were not counted."""
        if self.stored_frames is None:
            return None
        if self.total_frames is None:
            return self.stored_frames
        return min(self.stored_frames, self.total_frames)

    def __iter__(self):
        remaining = self.total_frames
        for path in self.paths:
            frames = _decode_segment(path)
            if remaining is not None:
                frames = frames[:remaining]
                remaining -= len(frames)
            if len(frames):
                yield frames
            if remaining == 0:
                return

    def cleanup(self):
        """Remove the segment files once the final video is safely written."""
        if self.keep_files:
            return
        for path in self.paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        self.cleaned = True


class SavedFrames(Sequence):
    """The frames of a chain's finished video, read back from the file it saved.

    Once SegmentedFrames.cleanup() has removed the segment files, a later step
    that names the chain's result (previous_result:) gets these instead: a
    frame sequence - length is the frame count - decoded from the final file
    the first time a frame is asked for, so a chain nobody consumes never pays
    for the decode.
    """

    def __init__(self, path, count=None, cleaned=True):
        self.path = path
        self._count = count
        self._frames = None
        # Result.retainable: the segment files this replaced are gone, and a
        # cache entry must not outlive the file this one reads
        self.cleaned = cleaned

    def _load(self):
        if self._frames is None:
            from ..media import decode_rgb_frames

            self._frames = [Image.fromarray(f) for f in decode_rgb_frames(self.path)]
            self._count = len(self._frames)
        return self._frames

    def __len__(self):
        if self._count is None:
            self._load()
        return self._count

    def __getitem__(self, index):
        return self._load()[index]


class SegmentSpill:
    """Writes each completed segment to disk as a playable mp4.

    Files are named {prefix}.{iteration}.segment-{index:03d}.mp4 in the
    workflow's output directory - a crashed chain leaves them behind, ready to
    salvage with gather_videos + concat_videos.
    """

    def __init__(self, pipeline, config):
        output_dir = getattr(pipeline, "output_dir", None)
        file_prefix = getattr(pipeline, "file_prefix", None)
        if not output_dir or not file_prefix:
            raise ValueError(
                "save_segments needs the workflow's output directory - it is "
                "only available when the chain runs through a workflow"
            )
        if not is_av_available():
            raise ValueError(
                "save_segments writes mp4 segment files with PyAV - install "
                "it with: pip install av"
            )
        if config.fps is None:
            raise ValueError(
                "save_segments needs the frame rate to encode segment files - "
                "set 'fps' on the chain or a 'frame_rate' pipeline argument"
            )

        # The same step can chain more than once (previous_result fan-out) -
        # a per-wrapper counter keeps each iteration's files apart
        iteration = getattr(pipeline, "_chain_iteration", -1) + 1
        pipeline._chain_iteration = iteration

        self.output_dir = output_dir
        self.iteration = iteration
        self.base_name = f"{file_prefix}.{iteration}"
        self.fps = config.fps
        self.paths = []
        self.frame_count = 0

    def write(self, frames, audio, sample_rate):
        """Encode one trimmed segment to disk and record its path.

        Args:
            frames: The segment's on-timeline PIL frames
            audio: The segment's on-timeline generated audio as
                (channels, samples) numpy, or None - muxed in so a crashed
                chain leaves fully playable segments
            sample_rate: Sample rate of that audio
        """
        # A crashed or completed chain leaves segment files behind on disk,
        # ready to salvage - and the per-wrapper iteration counter (base_name)
        # restarts at 0 in a fresh process, so a rerun must dedupe through the
        # same '-N' convention every other output goes through rather than
        # silently overwriting them.
        path = output_file_path(
            self.output_dir,
            f"{self.base_name}.segment-{len(self.paths):03d}.mp4",
        )

        audio_track = None
        if audio is not None and audio.shape[1] and sample_rate is not None:
            audio_track = torch.from_numpy(numpy.ascontiguousarray(audio))

        encode_video(
            frames_for_encoding(frames),
            fps=self.fps,
            output_path=path,
            audio=audio_track,
            audio_sample_rate=sample_rate if audio_track is not None else None,
        )
        self.paths.append(path)
        self.frame_count += len(frames)
        logger.info(f"Saved chain segment to {path}")


def _decode_segment(path):
    """Read a segment file back as a uint8 (frames, height, width, 3) tensor."""
    from ..media import decode_rgb_frames

    return torch.from_numpy(numpy.stack(decode_rgb_frames(path), axis=0))


def spilled_frames(config, spill):
    """A spilled chain's frames, replayed lazily from its segment files.

    match_audio and duration_seconds overshoot by design - the tail trim happens
    as the lazy frames replay, so the files themselves stay whole."""
    return SegmentedFrames(
        spill.paths,
        config.total_frames,
        config.keep_segments,
        spill.frame_count,
    )
