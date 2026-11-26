"""The shipped default preferences (`dw/preference_defaults.json`, #1625).

The file maps a task key (`text_to_image`) to tiers of catalog targets that
#1626's resolver ranks. Swapping one target for another is only safe if every
target in a key takes the same variables, so the file carries a per-key
variable contract and these pins hold the catalog to it:

- every target resolves through `dw/library.py` to a catalog workflow;
- every target declares each required contract variable;
- an optional variable a target lacks is declined, with a reason;
- every `configures: templates/text-to-image` entry is placed in the ladder
  or left out, with a reason - never both, never neither;
- keys are variable names and tiers are draft/standard/best;
- a `{workflow, arguments}` target's arguments are variables it declares;
- the file ships in the wheel.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from dw.library import library_path
from dw.security import validate_variable_name
from tests.test_examples import REPO_ROOT

DEFAULTS_PATH = os.path.join(REPO_ROOT, "dw", "preference_defaults.json")
TIERS = {"draft", "standard", "best"}

# The template a key's catalog entries configure: every model entry that
# configures it must be placed in that key's ladder or in its left_out
CONFIGURED_TEMPLATE = {"text_to_image": "templates/text-to-image"}


def load_defaults():
    with open(DEFAULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def catalog():
    return library_path("workflows", None, primary=os.path.join(REPO_ROOT, "workflows"))


def resolve(target):
    """A target's workflow name and its arguments. A target is a catalog
    name, or `{"workflow": name, "arguments": {...}}`."""
    if isinstance(target, str):
        return target, {}
    assert isinstance(target, dict), f"target {target!r} is neither a name nor a dict"
    assert set(target) <= {"workflow", "arguments"}, target
    return target["workflow"], target.get("arguments", {})


def definition_of(name, library=None):
    found = (library or catalog()).find(name)
    assert found, f"{name} does not resolve to a catalog workflow"
    path, _root = found
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def declared_variables(name, library=None):
    return set(definition_of(name, library).get("variables", {}))


def every_target(defaults):
    for key, tiers in defaults["preferences"].items():
        for tier, targets in tiers.items():
            for target in targets:
                yield key, tier, target


def ladder_names(defaults, key):
    return {
        resolve(target)[0]
        for tier_targets in defaults["preferences"][key].values()
        for target in tier_targets
    }


DEFAULTS = load_defaults()
TARGETS = list(every_target(DEFAULTS))


def target_id(item):
    key, tier, target = item
    return f"{key}-{tier}-{resolve(target)[0]}"


def test_the_file_has_its_sections():
    assert DEFAULTS["version"] == 1
    assert set(DEFAULTS) == {
        "version",
        "contracts",
        "preferences",
        "declines",
        "left_out",
    }
    # Every ladder has a contract, and every contract a ladder
    assert set(DEFAULTS["preferences"]) == set(DEFAULTS["contracts"])
    assert set(DEFAULTS["left_out"]) <= set(DEFAULTS["preferences"])


def test_the_shipped_ladder_is_text_to_image_only():
    """#1625 Q1: image_to_image, image_edit and the video keys were cut."""
    assert set(DEFAULTS["preferences"]) == {"text_to_image"}


def test_keys_are_variable_names_and_tiers_are_known():
    for key, tiers in DEFAULTS["preferences"].items():
        validate_variable_name(key)
        assert tiers, f"{key} has no tiers"
        assert set(tiers) <= TIERS, f"{key} has unknown tiers {set(tiers) - TIERS}"
        for tier, targets in tiers.items():
            assert targets, f"{key}.{tier} is empty"
    for key in DEFAULTS["contracts"]:
        validate_variable_name(key)
    for key in DEFAULTS["left_out"]:
        validate_variable_name(key)


def test_a_contract_names_variables_once():
    for key, contract in DEFAULTS["contracts"].items():
        assert set(contract) == {"required", "optional"}, key
        required, optional = contract["required"], contract["optional"]
        for name in required + optional:
            validate_variable_name(name)
        assert len(set(required)) == len(required), key
        assert len(set(optional)) == len(optional), key
        assert not set(required) & set(optional), (
            f"{key}: {set(required) & set(optional)} is both required and optional"
        )


@pytest.mark.parametrize("item", TARGETS, ids=target_id)
def test_every_target_resolves_to_a_catalog_workflow(item):
    _key, _tier, target = item
    name, _arguments = resolve(target)
    definition = definition_of(name)
    assert definition.get("steps"), f"{name} has no steps"


@pytest.mark.parametrize("item", TARGETS, ids=target_id)
def test_every_target_declares_the_required_variables(item):
    key, _tier, target = item
    name, _arguments = resolve(target)
    missing = set(DEFAULTS["contracts"][key]["required"]) - declared_variables(name)
    assert not missing, f"{name} lacks the {key} contract's required {missing}"


@pytest.mark.parametrize("item", TARGETS, ids=target_id)
def test_every_optional_variable_a_target_lacks_is_declined(item):
    key, _tier, target = item
    name, _arguments = resolve(target)
    lacks = set(DEFAULTS["contracts"][key]["optional"]) - declared_variables(name)
    declined = DEFAULTS["declines"].get(name, {})
    assert lacks <= set(declined), (
        f"{name} lacks {lacks - set(declined)} and does not decline it"
    )
    for variable in lacks:
        assert isinstance(declined[variable], str) and declined[variable].strip(), (
            f"{name} declines {variable} with no reason"
        )


def test_a_decline_is_for_a_ladder_target_and_a_variable_it_lacks():
    """A stale decline - a target that gained the variable, or left the
    ladder - would tell the resolver to skip something that works."""
    in_ladder = set().union(
        *(ladder_names(DEFAULTS, k) for k in DEFAULTS["preferences"])
    )
    optional = set().union(*(c["optional"] for c in DEFAULTS["contracts"].values()))
    for name, declined in DEFAULTS["declines"].items():
        assert name in in_ladder, f"{name} declines variables but is in no ladder"
        assert declined, f"{name} has an empty declines entry"
        for variable, reason in declined.items():
            assert variable in optional, f"{name} declines {variable}, not optional"
            assert variable not in declared_variables(name), (
                f"{name} declines {variable}, which it declares"
            )
            assert isinstance(reason, str) and reason.strip(), (name, variable)


@pytest.mark.parametrize("key", sorted(CONFIGURED_TEMPLATE))
def test_every_configuring_entry_is_placed_or_left_out_never_both(key):
    """A new `configures: templates/text-to-image` entry has to be put in
    the ladder or listed as left out, with a reason - that friction is the
    point (#1625)."""
    template = CONFIGURED_TEMPLATE[key]
    found, _shadowed = catalog().entries()
    configuring = set()
    for name in found:
        if not name.startswith("models/"):
            continue
        if definition_of(name).get("configures") == template:
            configuring.add(name)
    assert configuring, f"nothing configures {template}"

    placed = ladder_names(DEFAULTS, key)
    left_out = DEFAULTS["left_out"].get(key, {})
    both = placed & set(left_out)
    assert not both, f"{both} are both in the {key} ladder and left out"
    unplaced = configuring - placed - set(left_out)
    assert not unplaced, (
        f"{sorted(unplaced)} configure {template} but are neither in the "
        f"{key} ladder nor in left_out.{key}"
    )
    for name, reason in left_out.items():
        assert name in configuring, (
            f"left_out.{key} names {name}, which does not configure {template}"
        )
        assert isinstance(reason, str) and reason.strip(), f"{name} left out, no reason"


def test_flux3_klein_is_left_out_for_the_upsampler_input():
    """#1625 Q3: Klein's only ladder route takes `request`, not `prompt`."""
    reason = DEFAULTS["left_out"]["text_to_image"]["models/flux3-klein"]
    assert "request" in reason and "prompt" in reason
    upsampler = declared_variables("templates/flux3/upsample-then-generate")
    assert "request" in upsampler and "prompt" not in upsampler


def _argument_problems(target, library=None):
    name, arguments = resolve(target)
    declared = declared_variables(name, library)
    return sorted(set(arguments) - declared)


def test_a_workflow_arguments_target_names_declared_variables(tmp_path):
    """A `{workflow, arguments}` target (the shape video keys will need, a
    tier preset passed as an argument) may only pass variables the workflow
    declares - pinned against a fixture, since the shipped ladder has none."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "tiered.json").write_text(
        json.dumps(
            {
                "id": "tiered",
                "description": "A fixture with a tier preset.",
                "variables": {"prompt": "x", "tier": "draft"},
                "steps": [],
            }
        )
    )
    fixture = library_path("workflows", None, primary=str(tmp_path))

    good = {"workflow": "templates/tiered", "arguments": {"tier": "final"}}
    assert _argument_problems(good, fixture) == []
    bad = {"workflow": "templates/tiered", "arguments": {"tier": "x", "quality": 1}}
    assert _argument_problems(bad, fixture) == ["quality"]


@pytest.mark.parametrize("item", TARGETS, ids=target_id)
def test_every_shipped_target_passes_only_declared_arguments(item):
    _key, _tier, target = item
    assert _argument_problems(target) == []


def test_the_file_is_not_under_a_workflow_library():
    """`dw/library.py` lists every *.json under workflows/ as a workflow,
    and dw/workflows/ is the builtin library: the file must be beside them,
    not in them."""
    assert os.path.dirname(DEFAULTS_PATH) == os.path.join(REPO_ROOT, "dw")


def test_the_file_ships_in_the_wheel(tmp_path):
    """Built from a copy of the source tree, so the build's own build/ and
    egg-info never touch the checkout other tests are reading."""
    source = tmp_path / "src"
    source.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy(os.path.join(REPO_ROOT, name), source / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "ui", "docs")
    for package in ("dw", "dw_mcp"):
        shutil.copytree(
            os.path.join(REPO_ROOT, package), source / package, ignore=ignore
        )

    out = tmp_path / "wheel"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(source),
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "-q",
            "-w",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    (wheel,) = list(out.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "dw/preference_defaults.json" in archive.namelist()
        shipped = json.loads(archive.read("dw/preference_defaults.json"))
    assert shipped == DEFAULTS
