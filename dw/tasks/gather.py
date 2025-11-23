import glob as glob_lib
import logging
from diffusers.utils import load_image, load_video
from ..security import validate_path, validate_url, SecurityError, ALLOWED_IMAGE_EXTENSIONS

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
        image_paths = glob_lib.glob(glob)
        logger.info(f"Found {len(image_paths)} local images")

        for path in image_paths:
            try:
                logger.debug(f"Loading image from: {path}")
                images.append(load_image(path))
            except Exception as e:
                logger.error(
                    f"Failed to load image from {path}: {str(e)}", exc_info=True
                )
                raise

    # Load images from URLs
    for url in urls:
        try:
            logger.debug(f"Loading image from URL: {url}")
            validated_url = validate_url(url)
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
    Gather videos from local files and/or URLs.

    Args:
        glob: Pattern for matching local video files (e.g., "videos/*.mp4")
        urls: List of URLs to download videos from

    Returns:
        List of loaded videos
    """
    if urls is None:
        urls = []
    videos = []

    # Load local videos matching glob pattern
    if glob is not None:
        logger.debug(f"Searching for videos matching pattern: {glob}")
        video_paths = glob_lib.glob(glob)
        logger.info(f"Found {len(video_paths)} local videos")

        for path in video_paths:
            try:
                logger.debug(f"Loading video from: {path}")
                videos.append(load_video(path))
            except Exception as e:
                logger.error(
                    f"Failed to load video from {path}: {str(e)}", exc_info=True
                )
                raise

    # Load videos from URLs
    for url in urls:
        try:
            logger.debug(f"Loading video from URL: {url}")
            validated_url = validate_url(url)
            videos.append(load_video(validated_url))
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
