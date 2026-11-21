"""The MiniMax-H3-Image catalog (#1317, stage #1424): the templates' rules are
read off the diffusers pipeline that enforces them, the recipe is the bench's
(so a host's quantized cache keeps hitting), and the family stays apart from
MiniMax-H3's, whose engine-side rules key on its classes by name."""

import json
import os
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILY = os.path.join(ROOT, "workflows", "templates", "minimax-h3-image")
T2I = os.path.join(FAMILY, "text-to-image.json")
EDIT = os.path.join(FAMILY, "multi-reference.json")
MODEL = os.path.join(ROOT, "workflows", "models", "minimax-h3-image.json")
BENCH = os.path.join(ROOT, "bench", "workflows", "h3-image-t2i.json")
TEMPLATES = [T2I, EDIT]
DEFINITIONS = [T2I, EDIT, MODEL]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pipeline_step(definition):
    (step,) = definition["steps"]
    return step["pipeline"]


def test_the_pipeline_is_not_minimax_h3_s():
    """dw's H3-only behaviour (sparse attention's class match, the H3
    catalog rules) must not reach this family by inheritance."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MiniMaxH3ModularPipeline,
    )
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline,
    )

    assert not issubclass(MiniMaxH3ImageModularPipeline, MiniMaxH3ModularPipeline)


def test_the_transformer_is_not_the_class_sparse_attention_matches():
    from diffusers.models.transformers.transformer_minimax_h3_image import (
        MiniMaxH3ImageTransformer2DModel,
    )

    from dw import sparse_attention

    assert MiniMaxH3ImageTransformer2DModel.__name__ != sparse_attention._H3_CLASS
    assert MiniMaxH3ImageTransformer2DModel.__name__ != "MiniMaxH3Transformer3DModel"


def test_the_32_pixel_grid_is_the_pipeline_s_canvas_multiple():
    from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import (
        MiniMaxH3ImageModularPipeline as Pipeline,
    )

    # The properties fall back to the released checkpoint's values with no
    # components loaded: 16 (VAE) * 2 (patch) = 32.
    bare = types.SimpleNamespace()
    bare.vae_spatial_compression_ratio = Pipeline.vae_spatial_compression_ratio.fget(
        bare
    )
    bare.patch_size = Pipeline.patch_size.fget(bare)
    multiple = Pipeline.canvas_multiple.fget(bare)
    assert multiple == 32
    for path in DEFINITIONS:
        constraints = load(path)["variable_constraints"]
        for axis in ("width", "height"):
            assert constraints[axis]["modulus"] == multiple, (path, axis)
            assert constraints[axis]["remainder"] == 0, (path, axis)
    for path in TEMPLATES:
        for name, bundle in load(path)["presets"]["size"].items():
            assert bundle["width"] % multiple == 0, (path, name)
            assert bundle["height"] % multiple == 0, (path, name)


def test_references_are_named_one_based_as_the_pipeline_documents():
    import inspect

    from diffusers.modular_pipelines.minimax_h3_image import (
        before_encoder,
        encoders,
    )

    assert "`<Picture 1>`, `<Picture 2>`" in inspect.getsource(before_encoder)
    assert 'f"Picture {index + 1}: "' in inspect.getsource(encoders)
    edit = load(EDIT)
    assert "<Picture 1>" in edit["description"]
    assert "<Picture 1>" in edit["variables"]["prompt"]
    assert pipeline_step(edit)["arguments"]["image"] == "variable:images"


def test_the_pipeline_takes_no_step_count_guidance_or_negative_prompt():
    """The sampler is fixed, so the templates declare none of these and an
    argument naming one is refused as undeclared."""
    for path in DEFINITIONS:
        definition = load(path)
        names = set(definition["variables"]) | set(
            pipeline_step(definition)["arguments"]
        )
        for name in (
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "min_generation_area",
        ):
            assert name not in names, (path, name)


@pytest.mark.parametrize("path", DEFINITIONS)
def test_the_recipe_is_the_bench_s_so_the_quantized_cache_hits(path):
    bench = pipeline_step(load(BENCH))
    ours = pipeline_step(load(path))
    assert ours["configuration"] == bench["configuration"]
    assert ours["from_pretrained_arguments"] == bench["from_pretrained_arguments"]


@pytest.mark.parametrize("path", DEFINITIONS)
def test_host_ram_is_the_warm_floor_and_the_cold_need_is_stated(path):
    definition = load(path)
    assert definition["host_ram_gb"] == 64
    assert "120 GB" in definition["description"]


@pytest.mark.parametrize("path", TEMPLATES)
def test_the_templates_do_not_reach_for_the_prompt_enhancer(path):
    text = json.dumps(load(path))
    assert "trust_remote_code" not in text
    assert "-PE" not in text
    assert " PE" not in text


def _edit_errors(arguments):
    from dw.workflow import Workflow

    return Workflow(load(EDIT), None, EDIT).validation_errors(arguments)


def test_an_empty_images_list_is_refused_at_validate_with_the_reason():
    # M-F407 step 4: `images: []` validated clean; an edit with no reference
    # has nothing to edit, and the pipeline would only say so after loading
    rule = load(EDIT)["variable_constraints"]["images"]
    errors = [
        e for e in _edit_errors({"images": []}) if e["path"] == "arguments.images"
    ]
    assert len(errors) == 1, errors
    assert "must not be empty" in errors[0]["message"]
    assert rule["reason"] in errors[0]["message"]


def test_one_reference_still_validates():
    errors = _edit_errors({"images": [{"location": "asset:cast/estela.jpg"}]})
    assert [e for e in errors if e["path"].startswith("arguments.images")] == []
