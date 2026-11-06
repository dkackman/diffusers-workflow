"""Marigold's pipelines return their result as `prediction`, a numpy array -
depth (N, H, W, 1) in [0, 1], normals (N, H, W, 3) in [-1, 1] - which no
output extractor recognized, so the whole output object reached the save as
one artifact and was refused for image/png (#1201)."""

import numpy
from PIL import Image

from diffusers.pipelines.marigold.pipeline_marigold_depth import MarigoldDepthOutput
from diffusers.pipelines.marigold.pipeline_marigold_intrinsics import (
    MarigoldIntrinsicsOutput,
)
from diffusers.pipelines.marigold.pipeline_marigold_normals import (
    MarigoldNormalsOutput,
)

from dw.output_extraction import get_artifact_list


def test_a_depth_prediction_saves_as_one_image_per_input():
    prediction = numpy.linspace(0, 1, 2 * 6 * 4, dtype=numpy.float32).reshape(
        2, 6, 4, 1
    )
    output = MarigoldDepthOutput(prediction=prediction, uncertainty=None, latent=None)
    images = get_artifact_list(output)
    assert len(images) == 2
    assert all(isinstance(image, Image.Image) for image in images)
    assert images[0].size == (4, 6)
    assert images[0].mode == "RGB"


def test_a_normals_prediction_saves_as_one_image_per_input():
    prediction = numpy.zeros((1, 6, 4, 3), dtype=numpy.float32)
    prediction[..., 2] = 1.0  # facing the viewer
    output = MarigoldNormalsOutput(prediction=prediction, uncertainty=None, latent=None)
    [image] = get_artifact_list(output)
    assert image.size == (4, 6)
    # A normal facing the viewer renders as diffusers' own visualiser draws
    # it: (0, 0, 1) -> (127, 127, 255), not a raw [-1, 1] clip
    assert image.getpixel((0, 0)) == (127, 127, 255)


def test_an_intrinsics_prediction_is_not_drawn_as_normals():
    """Also three channels, but in [0, 1] and several targets per image -
    its visualiser needs the pipeline's target properties, which the output
    does not carry. Left as it was rather than rendered wrong."""
    prediction = numpy.zeros((2, 6, 4, 3), dtype=numpy.float32)
    output = MarigoldIntrinsicsOutput(
        prediction=prediction, uncertainty=None, latent=None
    )
    assert get_artifact_list(output) == [output]
