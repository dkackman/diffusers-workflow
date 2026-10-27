"""remove_object (#956): prompt-free LaMa removal.

The mask is binarized, resized and dilated here, an all-black mask returns the
input without loading a model, and only the masked region comes from the
network. The weights are a Hub repo and a safetensors file, refused otherwise
at validation and again in the loader, and they must load as a masked image
model. No test downloads the real weights: a stub descriptor stands in for
LaMa, and the wrong-architecture case uses a tiny real spandrel network."""

from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from dw.locations import location_errors
from dw.security import InvalidInputError, PathTraversalError
from dw.task_domains import task_argument_errors
from dw.tasks import model_cache, remove_object as remover
from dw.tasks.remove_object import (
    DEFAULT_REMOVER_REPO,
    DEFAULT_REMOVER_WEIGHTS,
    EMPTY_MASK_WARNING,
    REMOVER_REVISION,
    prepare_mask,
    remove_object,
)
from dw.tasks.task import Task
from dw.media_types import AudioVideo

PICKLES = ["big-lama.pt", "big-lama.ckpt", "big-lama.bin", "big-lama.pth"]
LOCAL_PATHS = ["/tmp/lama", "./weights", "../lama", "~/lama"]


def photo(size=(64, 48)):
    """A two-tone image with a red square in the middle - the 'object'."""
    image = Image.new("RGB", size, (40, 120, 200))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, size[1] * 3 // 4, size[0], size[1]), fill=(200, 180, 90))
    w, h = size
    draw.rectangle((w // 2 - 4, h // 2 - 4, w // 2 + 4, h // 2 + 4), fill=(255, 0, 0))
    return image


def square_mask(size=(64, 48), box=None, value=255):
    w, h = size
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(
        box or (w // 2 - 5, h // 2 - 5, w // 2 + 5, h // 2 + 5), fill=value
    )
    return mask


class StubDescriptor:
    """Stands in for spandrel's LaMa descriptor: records what it is handed and
    returns solid green, so any changed pixel is visibly the model's."""

    def __init__(self):
        self.calls = []

    def __call__(self, image, mask):
        self.calls.append((image, mask))
        out = torch.zeros_like(image)
        out[:, 1] = 1.0
        return out


@pytest.fixture
def stub():
    descriptor = StubDescriptor()
    with mock.patch.object(remover, "_load_remover", return_value=descriptor) as load:
        descriptor.load = load
        yield descriptor


def step(arguments, command="remove_object"):
    return {
        "name": "erase",
        "task": {
            "command": command,
            "arguments": {
                "image": "previous_result:photo",
                "mask_image": "previous_result:mask",
                **arguments,
            },
        },
    }


class TestTheMask:
    def test_binarized_at_128(self):
        mask = Image.new("L", (4, 1))
        mask.putdata([0, 127, 128, 255])
        assert list(prepare_mask(mask, (4, 1), dilate=0).get_flattened_data()) == [
            0,
            0,
            255,
            255,
        ]

    def test_an_rgb_mask_is_read_as_grayscale(self):
        mask = Image.new("RGB", (4, 4), (255, 255, 255))
        assert prepare_mask(mask, (4, 4), dilate=0).mode == "L"
        assert set(prepare_mask(mask, (4, 4), dilate=0).get_flattened_data()) == {255}

    def test_resized_to_the_image_nearest(self):
        mask = square_mask((32, 24), box=(16, 0, 31, 23))
        out = prepare_mask(mask, (64, 48), dilate=0)
        assert out.size == (64, 48)
        assert set(out.get_flattened_data()) <= {0, 255}
        arr = np.array(out)
        assert arr[:, :32].max() == 0 and arr[:, 32:].min() == 255

    def test_dilated_by_n_pixels(self):
        mask = Image.new("L", (21, 21), 0)
        mask.putpixel((10, 10), 255)
        arr = np.array(prepare_mask(mask, (21, 21), dilate=3)) > 0
        ys, xs = np.nonzero(arr)
        assert (xs.min(), xs.max(), ys.min(), ys.max()) == (7, 13, 7, 13)

    def test_dilate_zero_leaves_it_alone(self):
        mask = square_mask()
        assert np.array_equal(
            np.array(prepare_mask(mask, mask.size, dilate=0)), np.array(mask)
        )


class TestAllBlack:
    def test_returns_the_input_with_no_model_load(self):
        image = photo()
        with (
            mock.patch.object(remover, "_load_remover") as load,
            mock.patch.object(remover, "emit_warning") as warn,
        ):
            assert remove_object(image, Image.new("L", image.size, 0)) is image
        load.assert_not_called()
        warn.assert_called_once_with(EMPTY_MASK_WARNING)

    def test_a_mask_below_the_threshold_is_all_black(self):
        image = photo()
        with mock.patch.object(remover, "_load_remover") as load:
            assert remove_object(image, square_mask(value=127)) is image
        load.assert_not_called()

    def test_the_command_warns_once_for_a_video(self):
        frames = AudioVideo([photo() for _ in range(3)], None, None, fps=12)
        task = Task({"command": "remove_object", "arguments": {}}, "cpu")
        with (
            mock.patch.object(remover, "_load_remover") as load,
            mock.patch("dw.events.emit_warning") as warn,
        ):
            out = task.run({"image": frames, "mask_image": Image.new("L", (64, 48))})
        load.assert_not_called()
        warn.assert_called_once_with(EMPTY_MASK_WARNING)
        assert len(out.frames) == 3 and out.fps == 12


class TestTheFill:
    def test_outside_the_dilated_mask_is_unchanged(self, stub):
        image = photo()
        mask = square_mask()
        out = remove_object(image, mask, dilate=2)
        assert out.size == image.size
        grown = np.array(prepare_mask(mask, image.size, dilate=2)) > 0
        before, after = np.array(image), np.array(out)
        assert np.array_equal(before[~grown], after[~grown])
        # and inside it is the model's
        assert (after[grown] == [0, 255, 0]).all()

    def test_a_mismatched_mask_still_gives_the_images_size(self, stub):
        image = photo((64, 48))
        out = remove_object(image, square_mask((128, 96)))
        assert out.size == (64, 48)
        image_tensor, mask_tensor = stub.calls[0]
        assert mask_tensor.shape == (1, 1, 48, 64)
        assert image_tensor.shape == (1, 3, 48, 64)

    def test_the_mask_sent_is_zero_or_one(self, stub):
        remove_object(photo(), square_mask())
        _, mask_tensor = stub.calls[0]
        assert set(mask_tensor.unique().tolist()) == {0.0, 1.0}

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_tensors_sent_are_fp32_on_every_device(self, stub, device):
        moved = []

        def fake_to_tensor(image, target, dtype=None):
            moved.append((target, dtype))
            arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dtype)

        with mock.patch.object(remover, "pil_to_float_tensor", fake_to_tensor):
            remove_object(photo(), square_mask(), device=device)
        assert moved == [(device, torch.float32), (device, torch.float32)]
        image_tensor, mask_tensor = stub.calls[0]
        assert image_tensor.dtype == torch.float32
        assert mask_tensor.dtype == torch.float32
        assert stub.load.call_args.args[0] == device

    def test_the_loaded_descriptor_is_made_fp32(self):
        descriptor = mock.MagicMock()
        model_cache.clear_model_cache()
        with (
            mock.patch("huggingface_hub.hf_hub_download", return_value="w"),
            mock.patch("safetensors.torch.load_file", return_value={}),
            mock.patch.object(
                remover, "_descriptor_from_state_dict", return_value=descriptor
            ),
        ):
            remover._load_remover("cpu", DEFAULT_REMOVER_REPO, DEFAULT_REMOVER_WEIGHTS)
        model_cache.clear_model_cache()
        descriptor.float.assert_called_once()
        descriptor.half.assert_not_called()

    def test_video_frames_keep_count_and_fps(self, stub):
        frames = AudioVideo([photo() for _ in range(4)], None, None, fps=8)
        task = Task({"command": "remove_object", "arguments": {}}, "cpu")
        out = task.run({"image": frames, "mask_image": square_mask()})
        assert len(out.frames) == 4 and out.fps == 8
        assert len(stub.calls) == 4

    def test_the_command_takes_the_tasks_device(self, stub, all_backends_available):
        task = Task({"command": "remove_object", "arguments": {}}, "mps")
        task.run({"image": photo(), "mask_image": square_mask()})
        assert stub.load.call_args.args[0] == "mps"


class TestTheWeights:
    def test_the_pin_is_for_the_default_file_alone(self):
        assert remover.remover_source({}) == (
            DEFAULT_REMOVER_REPO,
            DEFAULT_REMOVER_WEIGHTS,
            REMOVER_REVISION,
        )
        assert remover.remover_source({"model_name": "biali/big-lama"}) == (
            "biali/big-lama",
            DEFAULT_REMOVER_WEIGHTS,
            None,
        )

    def test_the_default_downloads_at_the_pin_and_is_cached(self):
        descriptor = mock.MagicMock()
        model_cache.clear_model_cache()
        with (
            mock.patch("huggingface_hub.hf_hub_download", return_value="w") as download,
            mock.patch("safetensors.torch.load_file", return_value={}),
            mock.patch.object(
                remover, "_descriptor_from_state_dict", return_value=descriptor
            ),
        ):
            for _ in range(2):
                remover._load_remover(
                    "cpu", DEFAULT_REMOVER_REPO, DEFAULT_REMOVER_WEIGHTS
                )
        assert download.call_count == 1
        assert download.call_args.kwargs == {
            "repo_id": DEFAULT_REMOVER_REPO,
            "filename": DEFAULT_REMOVER_WEIGHTS,
            "revision": REMOVER_REVISION,
        }
        assert model_cache._cache
        model_cache.clear_model_cache()
        assert not model_cache._cache

    def test_wrong_architecture_is_refused(self):
        from spandrel.architectures.Compact.__arch.SRVGG import SRVGGNetCompact

        net = SRVGGNetCompact(num_feat=8, num_conv=2, upscale=2)
        with pytest.raises(ValueError, match="inpainting model"):
            remover._descriptor_from_state_dict(net.state_dict())

    @pytest.mark.parametrize("name", PICKLES)
    def test_a_pickle_weight_name_is_refused_in_the_loader(self, name):
        with (
            mock.patch("huggingface_hub.hf_hub_download") as download,
            mock.patch.object(remover, "_load_remover") as load,
        ):
            with pytest.raises(InvalidInputError, match="safetensors"):
                remove_object(photo(), square_mask(), weight_name=name)
        download.assert_not_called()
        load.assert_not_called()

    @pytest.mark.parametrize("repo", LOCAL_PATHS)
    def test_a_local_model_name_is_refused_in_the_loader(self, repo):
        with mock.patch.object(remover, "_load_remover") as load:
            with pytest.raises((InvalidInputError, PathTraversalError)):
                remove_object(photo(), square_mask(), model_name=repo)
        load.assert_not_called()


class TestAtValidation:
    def test_the_defaults_are_clean(self):
        definition = {"steps": [step({})]}
        assert location_errors(definition) == []
        assert task_argument_errors(definition) == []

    def test_named_defaults_are_clean(self):
        definition = {
            "steps": [
                step(
                    {
                        "model_name": DEFAULT_REMOVER_REPO,
                        "weight_name": DEFAULT_REMOVER_WEIGHTS,
                        "dilate": 0,
                    }
                )
            ]
        }
        assert location_errors(definition) == []
        assert task_argument_errors(definition) == []

    @pytest.mark.parametrize("name", PICKLES)
    def test_a_pickle_weight_name_is_refused(self, name):
        errors = location_errors({"steps": [step({"weight_name": name})]})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.weight_name"]

    @pytest.mark.parametrize("repo", LOCAL_PATHS)
    def test_a_local_model_name_is_refused(self, repo):
        errors = location_errors({"steps": [step({"model_name": repo})]})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.model_name"]

    @pytest.mark.parametrize("dilate", [-1, 2.5])
    def test_a_bad_dilate_is_refused(self, dilate):
        errors = task_argument_errors({"steps": [step({"dilate": dilate})]})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.dilate"]

    def test_a_traversing_mask_is_refused(self):
        errors = location_errors(
            {"steps": [step({"mask_image": "../../etc/mask.png"})]}
        )
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.mask_image"]


class TestRegistration:
    def test_get_task_lists_the_arguments_and_defaults(self):
        from dw.introspection import describe_task

        params = {p["name"]: p for p in describe_task("remove_object")["parameters"]}
        assert {"image", "mask_image", "dilate", "model_name", "weight_name"} <= set(
            params
        )
        assert params["dilate"]["default"] == 8
        assert params["model_name"]["default"] == DEFAULT_REMOVER_REPO
        assert params["weight_name"]["default"] == DEFAULT_REMOVER_WEIGHTS
