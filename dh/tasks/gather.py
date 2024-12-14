import glob as glob_lib
from diffusers.utils import load_image, load_video


def gather_images(glob = None, urls = []):
    images = []
    if glob is not None:
        image_paths = glob_lib.glob(glob)
        for path in image_paths:            
            images.append(load_image(path))
    
    for url in urls:
        images.append(load_image(url))

    if len(images) == 0:
        raise ValueError("No images found")
    
    return images

def gather_videos(glob = None, urls = []):
    videos = []
    if glob is not None:
        video_paths = glob_lib.glob(glob)
        for path in video_paths:            
            videos.append(load_video(path))
    
    for url in urls:
        videos.append(load_video(url))

    return videos

def gather_inputs(kwargs):
    # gather inputs returns the input array so it can be put into a result
    # and passed to the next task
    return kwargs