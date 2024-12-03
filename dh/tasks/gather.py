import glob as glob_lib
from diffusers.utils import load_image

def gather(glob = None, urls = []):
    """Gather and process images matching glob pattern.
    
    Args:
        glob_pattern: Pattern to match image files
        
    Returns:
        List of loaded and resized PIL Images
    """
    
    images = []
    if glob is not None:
        image_paths = glob_lib.glob(glob)
        for path in image_paths:            
            images.append(load_image(path))
    
    for url in urls:
        images.append(load_image(url))
        
    return images