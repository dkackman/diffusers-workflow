"""templates/ltx2/extend-clip's default path loads two LTX-2.5 stacks.

'opening' generates the first clip and 'extended' conditions on it with a
second, differently-keyed pipeline (LTX2ConditionPipeline vs LTX2Pipeline),
so the engine's step-redefinition sharing does not apply and only
`release_pipeline` frees the first stack before the second loads (#523).
"""

import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PATH = os.path.join(REPO_ROOT, "workflows", "templates", "ltx2", "extend-clip.json")


@pytest.fixture
def definition():
    with open(PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_opening_releases_its_pipeline_before_extended_loads(definition):
    steps = {step["name"]: step for step in definition["steps"]}
    assert steps["opening"]["release_pipeline"] is True


def test_release_pipeline_does_not_disturb_the_elision_path(definition):
    """`opening` must stay unreferenced when the caller supplies `clip`, so
    `release_pipeline` costs nothing on the elided path (#446)."""
    assert definition["variables"]["clip"] == "previous_result:opening"
    opening = {s["name"]: s for s in definition["steps"]}["opening"]
    assert "shared_components" not in opening
