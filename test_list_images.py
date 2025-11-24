#!/usr/bin/env python3
"""
Standalone test for list image/video handling without importing torch
"""
import sys
import os
import tempfile
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the functions we need (avoiding torch import)
from dw.arguments import fetch_image, fetch_video

def test_fetch_image_list():
    """Test that fetch_image can handle lists of images"""
    print("Testing fetch_image with list of paths...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        img1 = Image.new('RGB', (50, 50), color='red')
        img2 = Image.new('RGB', (50, 50), color='blue')
        path1 = os.path.join(temp_dir, 'test1.png')
        path2 = os.path.join(temp_dir, 'test2.png')
        img1.save(path1)
        img2.save(path2)
        
        # Test list of paths
        result = fetch_image([path1, path2])
        assert isinstance(result, list), 'Result should be a list'
        assert len(result) == 2, 'Should have 2 images'
        assert all(hasattr(img, 'mode') for img in result), 'All should be PIL Images'
        print('✓ List of paths works')
        
        # Test list of dicts
        result = fetch_image([{'location': path1}, {'location': path2}])
        assert isinstance(result, list), 'Result should be a list'
        assert len(result) == 2, 'Should have 2 images'
        print('✓ List of dicts works')
        
        # Test list of already loaded images
        loaded = [img1, img2]
        result = fetch_image(loaded)
        assert isinstance(result, list), 'Result should be a list'
        assert len(result) == 2, 'Should have 2 images'
        assert result[0] is img1, 'First image should be same object'
        assert result[1] is img2, 'Second image should be same object'
        print('✓ List of already loaded images works')

def test_fetch_video_list():
    """Test that fetch_video can handle lists"""
    print("\nTesting fetch_video with list of frames...")
    # Test list of already loaded frames (PIL Images)
    frames = [Image.new('RGB', (100, 100)), Image.new('RGB', (100, 100))]
    result = fetch_video(frames)
    assert isinstance(result, list), 'Result should be a list'
    assert len(result) == 2, 'Should have 2 frames'
    print('✓ List of already loaded frames works')

if __name__ == '__main__':
    try:
        test_fetch_image_list()
        test_fetch_video_list()
        print('\n✅ All tests passed!')
        sys.exit(0)
    except AssertionError as e:
        print(f'\n❌ Test failed: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
