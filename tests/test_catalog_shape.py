"""Shape, trait and summary derivation over raw workflow definitions.

Every rule in spec §1.2 gets a minimal definition that exercises only it.
"""

import pytest

from dw.server.catalog_shape import (
    GENERATIVE_TASKS,
    SHAPES,
    SUMMARY_LIMIT,
    TRAITS,
    derive_catalog_metadata,
)


def pipeline_step(name, content_type, component_type="{Fake}", arguments=None, chain=None):
    pipeline = {
        "configuration": {"component_type": component_type},
        "from_pretrained_arguments": {"model_name": "m"},
        "arguments": arguments or {"prompt": "variable:prompt"},
    }
    if chain is not None:
        pipeline["chain"] = chain
    return {"name": name, "pipeline": pipeline, "result": {"content_type": content_type}}


def task_step(name, command, arguments, content_type=None):
    step = {"name": name, "task": {"command": command, "arguments": arguments}}
    if content_type:
        step["result"] = {"content_type": content_type}
    return step


def definition(*steps, **top):
    return {"id": "t", "steps": list(steps), **top}


def test_vocabularies_are_closed_and_stable():
    assert SHAPES == ("image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility")
    assert TRAITS == (
        "speech", "chained", "image-conditioned", "identity-referenced",
        "needs-input-media", "composes-workflows",
    )
    assert GENERATIVE_TASKS == frozenset(
        {"generate_speech", "text_generation", "image_to_text", "diffusion_upscale", "interpolate_frames"}
    )


def test_a_single_still_is_image():
    meta = derive_catalog_metadata(definition(pipeline_step("gen", "image/jpeg")))
    assert meta["shape"] == "image"
    assert meta["traits"] == []


def test_two_image_steps_are_an_image_set():
    meta = derive_catalog_metadata(
        definition(pipeline_step("a", "image/jpeg"), pipeline_step("b", "image/jpeg"))
    )
    assert meta["shape"] == "image-set"


def test_a_workflow_step_emitting_images_is_an_image_set():
    step = {
        "name": "sub",
        "workflow": {"path": "x.json", "arguments": {"prompt": "variable:prompt"}},
        "result": {"content_type": "image/jpeg"},
    }
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "image-set"
    assert "composes-workflows" in meta["traits"]


@pytest.mark.parametrize(
    "component_type", ["FluxImg2ImgPipeline", "StableDiffusionInpaintPipeline", "QwenImageEditPipeline", "FluxKontextPipeline", "StableDiffusionUpscalePipeline"]
)
def test_an_editing_pipeline_is_image_edit(component_type):
    meta = derive_catalog_metadata(definition(pipeline_step("gen", "image/jpeg", component_type)))
    assert meta["shape"] == "image-edit"


def test_an_image_argument_on_an_image_pipeline_is_image_edit():
    step = pipeline_step("gen", "image/jpeg", arguments={"prompt": "p", "image": "variable:image"})
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "image-edit"
    assert "needs-input-media" in meta["traits"]


def test_one_clip_is_a_shot():
    assert derive_catalog_metadata(definition(pipeline_step("v", "video/mp4")))["shape"] == "shot"


def test_a_concat_fed_by_two_steps_is_a_sequence():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            pipeline_step("b", "video/mp4"),
            task_step("cut", "concat_videos", {"videos": ["previous_result:a", "previous_result:b"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"


def test_a_dissolve_fed_by_two_steps_is_a_sequence():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            pipeline_step("b", "video/mp4"),
            task_step("cut", "dissolve_videos", {"videos": ["previous_result:a", "previous_result:b"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"


def test_a_concat_fed_by_one_step_is_still_a_shot():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("a", "video/mp4"),
            task_step("cut", "concat_videos", {"videos": ["previous_result:a", "previous_result:a"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "shot"


def test_a_chain_is_a_chained_shot_not_a_sequence():
    step = pipeline_step("v", "video/mp4", chain={"segments": 3})
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "shot"
    assert "chained" in meta["traits"]


@pytest.mark.parametrize("name", ["last_frame", "last_segment", "last_image", "match_audio"])
def test_continuation_arguments_are_chained(name):
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", name: "previous_result:x"})
    assert "chained" in derive_catalog_metadata(definition(step))["traits"]


def test_video_outranks_image_when_both_are_produced():
    meta = derive_catalog_metadata(
        definition(pipeline_step("board", "image/jpeg"), pipeline_step("v", "video/mp4"))
    )
    assert meta["shape"] == "shot"


def test_audio_only_is_audio():
    step = task_step("speak", "generate_speech", {"text": "variable:text"}, "audio/wav")
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "audio"
    assert "speech" in meta["traits"]


def test_text_only_is_text():
    step = task_step("expand", "text_generation", {"prompt": "variable:prompt"}, "text/plain")
    assert derive_catalog_metadata(definition(step))["shape"] == "text"


def test_processing_tasks_alone_are_utility():
    step = task_step("crop", "crop_square", {"image": "asset:a.png"}, "image/jpeg")
    meta = derive_catalog_metadata(definition(step))
    assert meta["shape"] == "utility"
    assert "needs-input-media" in meta["traits"]


def test_a_generative_task_is_not_utility():
    step = task_step("up", "diffusion_upscale", {"image": "asset:a.png"}, "image/jpeg")
    assert derive_catalog_metadata(definition(step))["shape"] != "utility"


def test_a_video_pipeline_emitting_audio_speaks():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "output": ["videos", "audio"]})
    assert "speech" in derive_catalog_metadata(definition(step))["traits"]


def test_a_video_pipeline_with_an_image_argument_is_image_conditioned():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "image": "previous_result:still"})
    meta = derive_catalog_metadata(definition(pipeline_step("still", "image/jpeg"), step))
    assert "image-conditioned" in meta["traits"]
    # previous_result is not supplied media
    assert "needs-input-media" not in meta["traits"]


def test_an_image_to_video_component_is_image_conditioned():
    step = pipeline_step("v", "video/mp4", "LTX2ImageToVideoPipeline")
    assert "image-conditioned" in derive_catalog_metadata(definition(step))["traits"]


def test_references_are_identity_referenced():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "references": ["previous_result:face"]})
    assert "identity-referenced" in derive_catalog_metadata(definition(step))["traits"]


def test_a_location_argument_needs_input_media():
    step = pipeline_step("v", "video/mp4", arguments={"prompt": "p", "image": {"location": "https://x/y.png"}})
    assert "needs-input-media" in derive_catalog_metadata(definition(step))["traits"]


def test_a_pipeline_reference_step_counts_as_generation():
    ref = {
        "name": "shot_2",
        "pipeline_reference": {"reference_name": "shot_1", "arguments": {"prompt": "p", "references": ["asset:face.png"]}},
        "result": {"content_type": "video/mp4"},
    }
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("shot_1", "video/mp4"),
            ref,
            task_step("cut", "concat_videos", {"videos": ["previous_result:shot_1", "previous_result:shot_2"]}, "video/mp4"),
        )
    )
    assert meta["shape"] == "sequence"
    assert "identity-referenced" in meta["traits"]
    assert "needs-input-media" in meta["traits"]


def test_summary_is_the_first_sentence():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description="Makes a cat. Then more."))
    assert meta["summary"] == "Makes a cat."
    assert meta["summary_truncated"] is False


def test_summary_splits_on_newline_too():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description="Line one\nLine two."))
    assert meta["summary"] == "Line one"


def test_a_long_first_sentence_is_truncated_at_a_word_boundary():
    words = " ".join(["word"] * 40) + "."
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg"), description=words))
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert meta["summary"].endswith("…")
    assert not meta["summary"][:-1].endswith(" ")
    assert meta["summary_truncated"] is True


def test_no_description_means_empty_summary():
    meta = derive_catalog_metadata(definition(pipeline_step("g", "image/jpeg")))
    assert meta["summary"] == ""


def test_declarations_override_and_are_reported():
    meta = derive_catalog_metadata(
        definition(
            pipeline_step("g", "image/jpeg"),
            description="A still.",
            shape="image-set",
            traits=["chained"],
            summary="Declared.",
        )
    )
    assert meta["shape"] == "image-set"
    assert meta["traits"] == ["chained"]
    assert meta["summary"] == "Declared."
    assert meta["declared"] == {"shape", "traits", "summary"}


def test_an_empty_or_malformed_definition_derives_something():
    assert derive_catalog_metadata({})["shape"] == "utility"
    assert derive_catalog_metadata({"steps": "nope"})["shape"] == "utility"
