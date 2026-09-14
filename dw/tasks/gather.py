import glob as glob_lib
import logging
from diffusers.utils import load_image
from ..arguments import fetch_image
from ..security import SecurityError
from ..locations import contained_matches, validate_media_glob, validate_media_url
from .video_utils import load_audio_video

logger = logging.getLogger("dw")


def gather_images(glob=None, urls=None):
    """
    Gather images from local files and/or URLs.

    Args:
        glob: Pattern for matching local image files (e.g., "images/*.png")
        urls: List of URLs to download images from

    Returns:
        List of loaded images

    Raises:
        ValueError: If no images are found
        SecurityError: If validation fails
    """
    if urls is None:
        urls = []
    images = []

    # Load local images matching glob pattern
    if glob is not None:
        logger.debug(f"Searching for images matching pattern: {glob}")
        # The pattern is a location like any other, and goes through the same
        # policy: it may only expand inside the directories this workflow may
        # read, and each match is re-checked because a wildcard can leave the
        # tree through a symlink (dw/locations.py)
        pattern = validate_media_glob(glob, what="the images glob")
        # Sorted, because glob returns filesystem order: a numbered sequence
        # of images gathered for concatenation has to come back in its own
        # order, not in whatever order the directory happens to hold
        image_paths = contained_matches(
            sorted(glob_lib.glob(pattern)), what="the images glob"
        )
        logger.info(f"Found {len(image_paths)} local images")

        for path in image_paths:
            try:
                logger.debug(f"Loading image from: {path}")
                images.append(fetch_image(path))
            except SecurityError:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to load image from {path}: {str(e)}", exc_info=True
                )
                raise

    # Load images from URLs
    for url in urls:
        try:
            logger.debug(f"Loading image from URL: {url}")
            validated_url = validate_media_url(url, "a gathered image url")
            images.append(load_image(validated_url))
        except SecurityError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to load image from URL {url}: {str(e)}", exc_info=True
            )
            raise

    # Validate that we found at least one image
    if len(images) == 0:
        error_msg = "No images found"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Successfully gathered {len(images)} images")
    return images


def gather_videos(glob=None, urls=None):
    """
    Gather videos from local files and/or URLs, audio included. To join
    videos rather than iterate over them, give their paths to concat_videos
    directly instead of gathering them first.

    Each video comes back as one artifact holding its frames and whatever
    audio was muxed alongside them, so a step that references this one
    iterates over videos rather than over frames - and a step meant to
    consume all of them at once, as concat_videos is, would be fanned out
    over them one at a time.

    Args:
        glob: Pattern for matching local video files (e.g., "videos/*.mp4")
        urls: List of URLs to download videos from

    Returns:
        List of AudioVideo artifacts, one per gathered video

    Raises:
        ValueError: If no videos are found
        SecurityError: If validation fails
    """
    if urls is None:
        urls = []
    videos = []

    # Load local videos matching glob pattern
    if glob is not None:
        logger.debug(f"Searching for videos matching pattern: {glob}")
        # Same containment as gather_images - one policy, two tasks
        pattern = validate_media_glob(glob, what="the videos glob")
        # Sorted, because glob returns filesystem order: a numbered sequence
        # of videos gathered for concatenation has to come back in its own
        # order, not in whatever order the directory happens to hold
        video_paths = contained_matches(
            sorted(glob_lib.glob(pattern)), what="the videos glob"
        )
        logger.info(f"Found {len(video_paths)} local videos")

        for path in video_paths:
            try:
                logger.debug(f"Loading video from: {path}")
                videos.append(load_audio_video(path))
            except SecurityError:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to load video from {path}: {str(e)}", exc_info=True
                )
                raise

    # Load videos from URLs
    for url in urls:
        try:
            logger.debug(f"Loading video from URL: {url}")
            videos.append(load_audio_video(url))
        except SecurityError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to load video from URL {url}: {str(e)}", exc_info=True
            )
            raise

    # Validate that we found at least one video
    if len(videos) == 0:
        error_msg = "No videos found"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Successfully gathered {len(videos)} videos")
    return videos


def gather_inputs(kwargs):
    """
    Gather input arguments for passing to next task.

    Args:
        kwargs: Dictionary of input arguments

    Returns:
        Input arguments unchanged
    """
    logger.debug(f"Gathering input arguments: {kwargs}")
    return kwargs
