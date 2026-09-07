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

from dw.server.catalog_shape import SUMMARY_LIMIT, derive_catalog_metadata
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
    "workflows/templates/minimax/storyboard.json": ("shot", ["has-audio", "identity-referenced"]),
    "workflows/templates/minimax/dialogue-short.json": ("sequence", ["has-audio", "identity-referenced"]),
    "workflows/templates/minimax/music-video.json": ("sequence", ["has-audio", "identity-referenced"]),
    "workflows/templates/minimax/chained-segments.json": ("shot", ["chained", "has-audio", "image-conditioned", "needs-input-media"]),
    "workflows/templates/ltx2/chained-segments.json": ("shot", ["chained", "has-audio", "image-conditioned", "needs-input-media"]),
    "workflows/templates/image-variation.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/segment-and-inpaint.json": ("image-edit", ["needs-input-media"]),
    "workflows/templates/describe-and-regenerate.json": ("image-set", ["composes-workflows", "needs-input-media"]),
    "workflows/templates/compose-workflows.json": ("shot", ["composes-workflows", "image-conditioned"]),
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
        assert path in UTILITIES, f"{path} derived 'utility' - a rule missed it, or add it to UTILITIES"
    else:
        assert path not in UTILITIES, f"{path} is listed as a utility but derives {meta['shape']}"


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declaration_must_differ_from_the_derivation(path):
    """An override that repeats the rules is noise that will rot when the
    rules or the file change. Declare only what derivation gets wrong."""
    definition = load(path)
    meta = derive_catalog_metadata(definition)
    stripped = {k: v for k, v in definition.items() if k not in ("shape", "traits", "summary")}
    derived = derive_catalog_metadata(stripped)
    for key in meta["declared"]:
        assert meta[key] != derived[key], f"{path} declares {key}={meta[key]!r}, which derivation already produces"


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_has_a_summary_that_fits(path):
    meta = derive_catalog_metadata(load(path))
    assert meta["summary"], f"{path}: description has no first sentence"
    assert len(meta["summary"]) <= SUMMARY_LIMIT
    assert not meta["summary_truncated"], (
        f"{path}: first sentence runs past {SUMMARY_LIMIT} chars - shorten it or declare 'summary': {meta['summary']!r}"
    )


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
        assert identity not in seen, f"{path} and {seen[identity]} share id {identity!r}"
        seen[identity] = path


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_declared_cost_is_well_formed(path):
    cost = load(path).get("cost")
    if cost is None:
        return
    assert isinstance(cost, list) and cost, f"{path}: cost must be a non-empty list or absent"
    for entry in cost:
        assert entry["device"] in ("cuda", "mps", "cpu"), path
        assert isinstance(entry["vram_gb"], (int, float)) and entry["vram_gb"] >= 0, path
        assert isinstance(entry["minutes"], (int, float)) and entry["minutes"] >= 0, path


BACKTICKED = re.compile(r"`([a-z_][a-z0-9_]*)`")


def _variable_names_in_catalog():
    names = set()
    for path in TEMPLATES + MODEL_CONFIGS:
        names |= set((load(path).get("variables") or {}).keys())
    return names


@pytest.mark.parametrize("path", TEMPLATES + MODEL_CONFIGS)
def test_a_description_names_only_variables_the_workflow_declares(path):
    """A description that says `num_frames` for a workflow with no such
    variable sends an agent to pass an argument nothing reads. Restricted
    to identifiers that are variable names somewhere in the catalog, so a
    backticked task or type name is not a false positive."""
    definition = load(path)
    declared = set((definition.get("variables") or {}).keys())
    catalog_variables = _variable_names_in_catalog()
    mentioned = set(BACKTICKED.findall(definition.get("description", "")))
    undeclared = (mentioned & catalog_variables) - declared
    assert not undeclared, f"{path} describes {sorted(undeclared)} but declares no such variable"


from dw.server.app import workflow_details
from dw.server.catalog_shape import project_listing
from dw.workflow_sources import WorkflowSource, listing

# Spec targets, as chars / 4. The listing is the first thing an agent reads;
# these are the ceilings that keep it readable rather than skimmed.
COMPACT_BUDGET = 5_500
FILTERED_BUDGET = 1_500


def _tokens(payload):
    return len(json.dumps(payload)) / 4


def test_the_compact_listing_fits_the_budget():
    found = listing([WorkflowSource(os.path.join(REPO_ROOT, "workflows"), "workspace", True)])
    details = workflow_details(found)

    compact = project_listing(details, view="compact")
    assert _tokens(compact) <= COMPACT_BUDGET, f"compact listing is {_tokens(compact):.0f} tokens"

    sequences = project_listing(details, view="compact", shape="sequence")
    assert sequences, "no template derives 'sequence'"
    assert _tokens(sequences) <= FILTERED_BUDGET, f"shape=sequence is {_tokens(sequences):.0f} tokens"
