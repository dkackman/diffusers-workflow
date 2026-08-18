"""The schema's pipeline configuration matches the keys the code actually reads.

A configuration key the code reads and the schema does not declare used to be
invisible: additionalProperties defaulted to true, so a workflow naming it
validated, and a workflow misspelling one validated too and then silently did
nothing - a missing 'offload' surfacing as an out-of-memory error rather than as
a typo. The schema is closed now, which makes the drift the other way fatal: a
key added to the code without being declared here fails validation for every
workflow that uses it. This test is what keeps the two in step.
"""

import ast
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The modules that read a pipeline's configuration
SOURCES = (
    os.path.join("dw", "pipeline_processors", "pipeline.py"),
    os.path.join("dw", "pipeline_processors", "config_objects.py"),
)

# Keys read from a variable named 'configuration' that belong to another block
# entirely - a component definition rather than a pipeline configuration. They
# are declared by their own schemas, which this test does not cover
OTHER_BLOCKS = {
    # from a component definition: { "quantization_config": {...} }
    "quantization_config",
    # from a named component block: { "vae": { "torch_dtype": ... } }
    "torch_dtype",
}


def schema():
    with open(os.path.join(REPO_ROOT, "dw", "workflow_schema.json")) as file:
        return json.load(file)


def configuration_keys_read_by_the_code():
    """Every literal key the pipeline processors read from a configuration dict.

    Found by walking the source rather than by keeping a list here - a list would
    be one more thing to drift.
    """
    found = {}
    for source in SOURCES:
        tree = ast.parse(open(os.path.join(REPO_ROOT, source), encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute) or function.attr != "get":
                continue
            if not node.args:
                continue

            target = function.value
            name = (
                target.id
                if isinstance(target, ast.Name)
                else target.attr if isinstance(target, ast.Attribute) else None
            )
            if name not in ("configuration", "component_configuration"):
                continue

            key = node.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.setdefault(key.value, set()).add(source)

    return found


def declared_keys():
    """Every key a configuration may hold, at either level.

    Component entries are included because the helpers are shared: the same
    function reads 'group_offload' from a pipeline configuration and from one
    component's, so the keys of both are legitimate reads.
    """
    configuration = schema()["$defs"]["pipeline_configuration"]["properties"]
    components = configuration["components"]["additionalProperties"]["properties"]
    return set(configuration) | set(components)


def test_the_schema_is_closed():
    """An unknown key must fail validation rather than being silently ignored"""
    assert schema()["$defs"]["pipeline_configuration"]["additionalProperties"] is False


@pytest.mark.parametrize("key", sorted(configuration_keys_read_by_the_code()))
def test_every_key_the_code_reads_is_declared(key):
    if key in OTHER_BLOCKS:
        pytest.skip(f"'{key}' belongs to a component definition, not a configuration")

    assert key in declared_keys(), (
        f"the code reads configuration['{key}'] but the schema does not declare "
        f"it - with the configuration closed, every workflow using it now fails "
        f"validation. Declare it in $defs/pipeline_configuration"
    )


def test_the_scan_finds_the_keys_it_is_meant_to():
    """A scan that quietly matched nothing would pass every assertion above"""
    found = configuration_keys_read_by_the_code()

    assert {"offload", "components", "load_components"} <= set(found)
