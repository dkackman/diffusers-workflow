"""The catalog's two trees, and the invariant each carries.

templates/ teaches a pattern and models/ records a hardware fact. The
distinction is only useful if it is legible from outside the file, so a
template must describe itself and a model config must name the template it
configures.
"""

import json
import os
import re

import pytest

from dw.server.app import workflow_details
from dw.server.catalog_shape import (
    SUMMARY_LIMIT,
    derive_catalog_metadata,
    project_listing,
)
from dw.workflow_sources import WorkflowSource, listing
from tests.test_examples import BUILTIN_DIR, REPO_ROOT, get_example_files

TEMPLATES = [f for f in get_example_files() if f.startswith("workflows/templates/")]
MODEL_CONFIGS = [f for f in get_example_files() if f.startswith("workflows/models/")]


def test_there_are_templates():
    assert TEMPLATES


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_describes_itself(path):
    """A template is read before it is run - by a person choosing one and by an
    agent matching a request's shape. An undescribed template is invisible to
    both, whatever its filename says."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))

    assert definition.get("description", "").strip(), f"{path} has no description"


def test_there_are_model_configs():
    assert MODEL_CONFIGS


@pytest.mark.parametrize("path", MODEL_CONFIGS)
def test_every_model_config_names_the_template_it_configures(path):
    """A model config is a tuned instance of a pattern. Without the pointer it
    is just another entry in the list, which is the problem this restructure
    exists to fix."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))
    configures = definition.get("configures", "")

    assert configures, f"{path} has no 'configures'"
    target = os.path.join(REPO_ROOT, "workflows", f"{configures}.json")
    assert os.path.isfile(
        target
    ), f"{path} configures '{configures}', which is not a workflow ({target})"


def load(path):
    return json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))


# The regression net for the derivation rules: real templates whose shape
# is subtle enough that a rule change should have to answer to them.
EXPECTED_SHAPES = {
    "workflows/templates/text-to-image.json": ("image", []),
    "workflows/templates/minimax/storyboard.json": (
        "shot",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/dialogue-short.json": (
        "sequence",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/music-video.json": (
        "sequence",
        ["has-audio", "identity-referenced"],
    ),
    "workflows/templates/minimax/chained-segments.json": (
        "shot",
        ["chained", "has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/ltx2/chained-segments.json": (
        "shot",
        ["chained", "has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/ltx2/keyframes.json": (
        "shot",
        ["has-audio", "image-conditioned", "needs-input-media"],
    ),
    "workflows/templates/image-variation.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/segment-and-inpaint.json": (
        "image-edit",
        ["needs-input-media"],
    ),
    "workflows/templates/describe-and-regenerate.json": (
        "image-set",
        ["composes-workflows", "needs-input-media"],
    ),
    "workflows/templates/compose-workflows.json": (
        "shot",
        ["composes-workflows", "image-conditioned"],
    ),
    "workflows/templates/generate-speech.json": ("audio", ["has-audio"]),
    "workflows/templates/assemble-and-score.json": ("sequence", ["needs-input-media"]),
    "workflows/templates/image-processors.json": ("utility", []),
}

# Templates that genuinely are utilities - processing with no generation.
# Anything else that derives 'utility' is a rule that missed.
UTILITIES = {
    "workflows/templates/audio-trim-fade.json",
    "workflows/templates/image-processors.json",
    "workflows/templates/recenter-crop.json",
    "workflows/templates/segment.json",
    "workflows/templates/upscale-spandrel.json",
}


@pytest.mark.parametrize("path, expected", sorted(EXPECTED_SHAPES.items()))
def test_the_rules_read_these_templates_as_expected(path, expected):
    meta = derive_catalog_metadata(load(path))
    assert (meta["shape"], meta["traits"]) == expected


@pytest.mark.parametrize("path", TEMPLATES)
def test_no_template_falls_through_to_utility(path):
    meta = derive_catalog_metadata(load(path))
    if meta["shape"] == "utility":
        assert (
            path in UTILITIES
        ), f"{path} derived 'utility' - a rule missed it, or add it to UTILITIES"
    else:
        assert (
            path not in UTILITIES
        ), f"{path} is listed as a utility but derives {meta['shape']}"


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declaration_must_differ_from_the_derivation(path):
    """An override that repeats the rules is noise that will rot when the
    rules or the file change. Declare only what derivation gets wrong."""
    definition = load(path)
    meta = derive_catalog_metadata(definition)
    stripped = {
        k: v for k, v in definition.items() if k not in ("shape", "traits", "summary")
    }
    derived = derive_catalog_metadata(stripped)
    for key in meta["declared"]:
        assert (
            meta[key] != derived[key]
        ), f"{path} declares {key}={meta[key]!r}, which derivation already produces"


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_has_a_summary_that_fits(path):
    meta = derive_catalog_metadata(load(path))
    assert meta["summary"], f"{path}: description has no first sentence"
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert not meta[
        "summary_truncated"
    ], f"{path}: first sentence runs past {SUMMARY_LIMIT} chars - shorten it or declare 'summary': {meta['summary']!r}"


BUILTINS = sorted(
    os.path.relpath(os.path.join(BUILTIN_DIR, name), REPO_ROOT)
    for name in os.listdir(BUILTIN_DIR)
    if name.endswith(".json")
)


def test_workflow_ids_are_unique_across_the_catalog():
    """A duplicate id is a step-cache collision waiting to happen, and it
    makes job history ambiguous about which workflow ran."""
    seen = {}
    for path in TEMPLATES + MODEL_CONFIGS + BUILTINS:
        identity = load(path).get("id")
        assert identity, f"{path} has no id"
        assert (
            identity not in seen
        ), f"{path} and {seen[identity]} share id {identity!r}"
        seen[identity] = path


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declared_cost_is_well_formed(path):
    cost = load(path).get("cost")
    if cost is None:
        return
    assert (
        isinstance(cost, list) and cost
    ), f"{path}: cost must be a non-empty list or absent"
    for entry in cost:
        assert entry["device"] in ("cuda", "mps", "cpu"), path
        assert (
            isinstance(entry["vram_gb"], (int, float)) and entry["vram_gb"] >= 0
        ), path
        assert (
            isinstance(entry["minutes"], (int, float)) and entry["minutes"] >= 0
        ), path


# The catalog's descriptions quote identifiers in single quotes, not
# backticks - JSON strings, where a backtick reads as a stray character.
QUOTED = re.compile(r"'([a-z_][a-z0-9_]*)'")

# Mentions that are deliberately not this workflow's variables: a
# sub-workflow's argument, a step argument, a chain field or a result field.
# One line per entry saying what the name actually is.
LEGITIMATE_MENTIONS = {
    # the argument the composed image-to-video workflow receives
    "workflows/templates/compose-workflows.json": {"image"},
    # a 'result' field, and the point is that this workflow omits it
    "workflows/templates/generate-speech.json": {"sample_rate"},
    # arguments of the image_to_text step, edited into the file rather than passed
    "workflows/templates/image-to-text.json": {"model_name", "prompt"},
    # a field of the step's 'chain' block
    "workflows/templates/minimax/chained-segments.json": {"trim_frames"},
    # the fl2va sub-workflow's argument, named to say this one leaves it unset
    "workflows/templates/minimax/last-frame-only.json": {"image"},
    # a 'result' field the modular pipeline needs declared
    "workflows/templates/minimax/music.json": {"sample_rate"},
}


def _variable_names_in_catalog():
    names = set()
    for path in TEMPLATES + MODEL_CONFIGS:
        names |= set((load(path).get("variables") or {}).keys())
    return names


def _mentioned(definition):
    return set(QUOTED.findall(definition.get("description", "")))


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_description_names_only_variables_the_workflow_declares(path):
    """A description that says 'num_frames' for a workflow with no such
    variable sends an agent to pass an argument nothing reads. Restricted
    to identifiers that are variable names somewhere in the catalog, so a
    quoted task or type name is not a false positive, and to the mentions
    LEGITIMATE_MENTIONS does not account for."""
    definition = load(path)
    declared = set((definition.get("variables") or {}).keys())
    catalog_variables = _variable_names_in_catalog()
    allowed = LEGITIMATE_MENTIONS.get(path, set())
    undeclared = (_mentioned(definition) & catalog_variables) - declared - allowed
    assert (
        not undeclared
    ), f"{path} describes {sorted(undeclared)} but declares no such variable"


def test_the_drift_check_actually_matches_something():
    """The quoting convention is the whole test: match backticks instead and
    every set is empty and every assertion passes for nothing."""
    matched = [
        path
        for path in TEMPLATES
        if _mentioned(load(path)) & _variable_names_in_catalog()
    ]
    assert (
        matched
    ), "no template description quotes a catalog variable name - the pattern is wrong"


def test_no_stale_entry_in_the_allowlist():
    for path, names in LEGITIMATE_MENTIONS.items():
        assert (
            path in TEMPLATES + MODEL_CONFIGS
        ), f"{path} is allowlisted but not in the catalog"
        assert names <= _mentioned(
            load(path)
        ), f"{path} no longer mentions {sorted(names - _mentioned(load(path)))}"


# Spec targets, as chars / 4. The listing is the first thing an agent reads;
# these are the ceilings that keep it readable rather than skimmed.
COMPACT_BUDGET = 6_000  # was 5_500; raised with the informative MiniMax/LTX-2 summaries, measured 5_552
FILTERED_BUDGET = 1_500


def _tokens(payload):
    return len(json.dumps(payload)) / 4


def test_the_compact_listing_fits_the_budget():
    found = listing(
        [WorkflowSource(os.path.join(REPO_ROOT, "workflows"), "workspace", True)]
    )
    details = workflow_details(found)

    compact = project_listing(details, view="compact")
    assert (
        _tokens(compact) <= COMPACT_BUDGET
    ), f"compact listing is {_tokens(compact):.0f} tokens"

    sequences = project_listing(details, view="compact", shape="sequence")
    assert sequences, "no template derives 'sequence'"
    assert (
        _tokens(sequences) <= FILTERED_BUDGET
    ), f"shape=sequence is {_tokens(sequences):.0f} tokens"


def _walk(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key, inner
            yield from _walk(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk(inner)


REMOTE_CODE_KEYS = ("trust_remote_code", "custom_pipeline")


BUILTINS = sorted(
    os.path.relpath(os.path.join(BUILTIN_DIR, name), REPO_ROOT)
    for name in os.listdir(BUILTIN_DIR)
    if name.endswith(".json")
)


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS + BUILTINS)
def test_no_catalog_entry_needs_trust_workflows(path):
    """A server started without --trust-workflows refuses either key at load,
    after validation has passed. An entry that carries one runs only where an
    operator lowered that guard, and an agent has no way to find out whether
    this server did - so the catalog carries neither, nor do the packaged
    builtin: sub-workflows a template can compose, and a workflow that needs
    remote code is written elsewhere."""
    definition = load(path)

    found = [key for key, _ in _walk(definition) if key in REMOTE_CODE_KEYS]

    assert not found, f"{path} sets {', '.join(found)}, which needs --trust-workflows"


def test_the_remote_code_check_sees_nested_keys():
    definition = {
        "steps": [{"pipeline": {"from_pretrained_arguments": {"custom_pipeline": "x"}}}]
    }

    assert [k for k, _ in _walk(definition) if k in REMOTE_CODE_KEYS] == [
        "custom_pipeline"
    ]


def _step(definition, name):
    return next(step for step in definition["steps"] if step["name"] == name)


class TestLtxTwoStage:
    """LTX-2.5's distilled two-stage flow is three moves: eight sigmas at half
    size, a 2x latent upsample, then renoise and three more sigmas at full size.
    The renoise scale is the first stage-two sigma, which no reference syntax can
    name, so the template carries the literal and this test ties it to the library."""

    def _definition(self):
        path = os.path.join(REPO_ROOT, "workflows", "templates", "ltx2", "two-stage.json")
        return json.load(open(path, encoding="utf-8"))

    def test_the_renoise_scale_is_the_first_stage_two_sigma(self):
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        refine = _step(self._definition(), "refine")

        assert refine["pipeline"]["arguments"]["noise_scale"] == STAGE_2_DISTILLED_SIGMA_VALUES[0]

    def test_the_refine_pass_runs_the_stage_two_schedule_on_the_upsampled_latents(self):
        refine = _step(self._definition(), "refine")
        arguments = refine["pipeline"]["arguments"]

        assert arguments["sigmas"] == "constant:diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES"
        assert arguments["latents"] == "previous_result:upscale.frames"
        assert arguments["audio_latents"] == "previous_result:base.audio"

    def test_the_base_and_refine_passes_share_one_pipeline(self):
        definition = self._definition()
        base = _step(definition, "base")["pipeline"]
        refine = _step(definition, "refine")["pipeline"]

        base_cache_key = {k: v for k, v in base.items() if k != "arguments"}
        refine_cache_key = {k: v for k, v in refine.items() if k != "arguments"}

        assert base_cache_key == refine_cache_key
        assert not _step(definition, "base").get("release_pipeline", False)
