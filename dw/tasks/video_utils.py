from PIL import Image
from typing import List


def process_video(video: List[Image.Image], processor, device, kwargs):
    processor = processor.lower()

    if processor == "get_frame":
        return get_frame(video, kwargs.get("frame_index", 0))

    if processor == "get_last_frame":
        return get_frame(video, len(video) - 1)

    if processor == "get_first_frame":
        return get_frame(video, 0)

    raise Exception(f"Unknown video processor type: {processor}")


def get_frame(video: List[Image.Image], frame_index=0):
    return video[frame_index]
