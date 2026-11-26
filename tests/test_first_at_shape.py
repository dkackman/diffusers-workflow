"""A cost row's `first_at_shape` term (#1615, stage #1763): a one-time cost
the first run at a shape pays on a card, added once per unseen shape."""

import copy
import json
import math

import jsonschema
import pytest

from dw.cost_scaling import fixed_minutes
from dw.plan import build_plan
from dw.plan_pricing import (
    _own_price,
    first_at_shape,
    first_at_shape_errors,
    first_at_shape_term,
    run_shapes,
)
from dw.server.catalog_shape import _compact_entry
from dw.server.observed_cost import seen_shapes
from dw.workflow import workflow_from_definition

from .test_observed_cost import run
from .test_plan import (
    CATALOG_CHAINED,
    _repo_json,
    observed,
    output_root,  # noqa: F401 - fixture
    plan,  # noqa: F401 - fixture
    prompt_library,  # noqa: F401 - fixture
)

SHAPE = {"minutes": 4.0, "keys": ["width", "height"]}


def definition(first=SHAPE, **cost):
    entry = {"device": "cuda", "name": "card", "vram_gb": 8, "minutes": 10, **cost}
    if first is not None:
        entry["first_at_shape"] = copy.deepcopy(first)
    return {
        "id": "shaped",
        "cost": [entry],
        "cost_drivers": ["width", "height", "shots"],
        "variables": {
            "width": 1344,
            "height": 768,
            "tier": "standard",
            "shots": [{"prompt": "a", "num_frames": 121}],
        },
        "presets": {
            "tier": {
                "standard": {"width": 1344, "height": 768},
                "draft": {"width": 896, "height": 512},
            }
        },
        "steps": [
            {
                "name": "still",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "org/model"},
                    "arguments": {
                        "width": "variable:width",
                        "height": "variable:height",
                    },
                },
            }
        ],
    }


def cost_item_schema():
    with open("dw/workflow_schema.json") as file:
        return json.load(file)["properties"]["cost"]["items"]


def row(first):
    return {"device": "cuda", "vram_gb": 8, "minutes": 10, "first_at_shape": first}


class TestSchema:
    def test_a_well_formed_block_is_accepted(self):
        jsonschema.validate(row(SHAPE), cost_item_schema())
        assert first_at_shape_errors(definition()) == []

    @pytest.mark.parametrize(
        "first",
        [
            {"minutes": -1, "keys": ["width"]},
            {"minutes": True, "keys": ["width"]},
            {"minutes": 1, "keys": []},
            {"minutes": 1},
            {"keys": ["width"]},
            {"minutes": 1, "keys": ["width"], "extra": 1},
            {"minutes": 1, "keys": ["width", "width"]},
        ],
    )
    def test_the_schema_refuses(self, first):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(row(first), cost_item_schema())

    @pytest.mark.parametrize("minutes", [math.inf, math.nan, -2, True])
    def test_validation_refuses_an_unusable_minutes(self, minutes):
        errors = first_at_shape_errors(
            definition({"minutes": minutes, "keys": ["width"]})
        )
        assert [error["path"] for error in errors] == ["cost[0].first_at_shape.minutes"]

    def test_validation_refuses_an_undeclared_key(self):
        errors = first_at_shape_errors(
            definition({"minutes": 1, "keys": ["width", "weights_dtype"]})
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "cost[0].first_at_shape.keys"
        assert "'weights_dtype'" in errors[0]["message"]

    def test_the_validate_pipeline_runs_the_check(self, tmp_path):
        from dw.validation import ERROR_CHECKS

        assert "first_at_shape" in [check.name for check in ERROR_CHECKS]

    def test_the_reader(self):
        assert first_at_shape(row(SHAPE)) == (4.0, ["width", "height"])
        assert first_at_shape(row({"minutes": True, "keys": ["w"]})) is None
        assert first_at_shape(row({"minutes": 1, "keys": []})) is None
        assert first_at_shape({"minutes": 1}) is None
        assert first_at_shape(None) is None


def term(spec, variables, seen):
    entry = spec["cost"][0]
    return first_at_shape_term(entry, spec, {**spec["variables"], **variables}, seen)


class TestTerm:
    def test_every_shape_seen_adds_nothing(self):
        spec = definition()
        seen = run_shapes(spec, spec["variables"], SHAPE["keys"])
        assert term(spec, {}, seen) == (0.0, None)

    def test_no_history_answer_adds_nothing(self):
        assert term(definition(), {}, None) == (0.0, None)

    def test_one_unseen_shape_adds_once(self):
        minutes, clause = term(definition(), {}, set())
        assert minutes == 4.0
        assert "1 shape" in clause and "121 frames, 1344x768" in clause

    def test_two_distinct_entry_counts_add_twice(self):
        shots = [{"num_frames": 121}, {"num_frames": 226}]
        assert term(definition(), {"shots": shots}, set())[0] == 8.0

    def test_a_repeated_entry_count_adds_once(self):
        shots = [{"num_frames": 121}, {"num_frames": 121}, {"num_frames": 121}]
        assert term(definition(), {"shots": shots}, set())[0] == 4.0

    def test_only_the_unseen_counts(self):
        spec = definition()
        seen = run_shapes(spec, spec["variables"], SHAPE["keys"])
        shots = [{"num_frames": 121}, {"num_frames": 226}]
        minutes, clause = term(spec, {"shots": shots}, seen)
        assert minutes == 4.0 and "226 frames" in clause

    def test_more_than_three_shapes_are_summarised(self):
        shots = [{"num_frames": n} for n in (25, 49, 73, 97, 121)]
        minutes, clause = term(definition(), {"shots": shots}, set())
        assert minutes == 20.0
        assert clause.endswith("and 2 more)")

    def test_presets_resolve_before_comparing(self):
        spec = definition()
        # a seen run that asked for tier=draft rendered at 896x512
        rows = [run(60, {"tier": "draft"})]
        seen = seen_shapes(spec, rows, SHAPE["keys"])
        assert seen == {(896.0, 512.0, 121.0)}
        assert term(spec, {"width": 896, "height": 512}, seen) == (0.0, None)

    def test_partial_runs_are_not_seen(self, monkeypatch):
        import dw.server.observed_cost as observed_cost

        spec = definition()
        monkeypatch.setattr(
            observed_cost, "_was_partial", lambda d, arguments: "partial" in arguments
        )
        rows = [run(60, {"partial": 1, "tier": "draft"}), run(60, {})]
        assert seen_shapes(spec, rows, SHAPE["keys"]) == {(1344.0, 768.0, 121.0)}


class TestEstimate:
    def quote(self, plan, spec, **overrides):  # noqa: F811
        return plan(spec, **overrides)["estimate"]

    def test_no_callable_no_term(self, plan):  # noqa: F811
        bare = self.quote(plan, definition(first=None))
        assert self.quote(plan, definition())["minutes"] == bare["minutes"]

    def test_an_unseen_shape_is_added_after_the_price(self, plan):  # noqa: F811
        bare = self.quote(plan, definition(first=None))
        shaped = self.quote(plan, definition(), seen_shapes=lambda keys: set())
        assert shaped["minutes"] == pytest.approx(bare["minutes"] + 4.0)
        assert "first pass at 1 shape" in shaped["scaled"]
        for key in ("basis", "warm_minutes", "cached_minutes", "low_confidence"):
            assert shaped.get(key) == bare.get(key)

    def test_a_seen_shape_adds_nothing(self, plan):  # noqa: F811
        spec = definition()
        seen = run_shapes(spec, spec["variables"], SHAPE["keys"])
        bare = self.quote(plan, definition(first=None))
        assert self.quote(plan, spec, seen_shapes=lambda keys: seen) == bare

    def test_the_observed_rescale_is_untouched(self, plan):  # noqa: F811
        history = lambda arguments: observed(minutes=7.0, runs=5)  # noqa: E731
        bare = self.quote(plan, definition(first=None), observed=history)
        shaped = self.quote(
            plan, definition(), observed=history, seen_shapes=lambda keys: set()
        )
        assert bare["basis"] == "observed"
        assert shaped["minutes"] == pytest.approx(bare["minutes"] + 4.0)
        assert shaped["basis"] == bare["basis"]

    def test_the_fixed_share_is_untouched(self):
        with_key = definition(fixed_minutes=6.0)
        without = definition(first=None, fixed_minutes=6.0)
        assert fixed_minutes(with_key["cost"][0]) == 6.0
        moved = {**with_key["variables"], "width": 2688}
        price = _own_price(with_key, {"variables": moved}, {}, "cuda", {})
        plain = _own_price(without, {"variables": moved}, {}, "cuda", {})
        assert price == plain

    def test_the_term_lands_after_the_checked_floor(self, plan):  # noqa: F811
        spec = _repo_json(CATALOG_CHAINED)
        common = {
            "shot": "dialogue",
            "segments": 2,
            "num_frames": 124,
            "seed": 1262,
            "prompts": [
                "A man speaks. <d>We the people of the United States.</d>",
                "A man speaks. <d>In order to form a more perfect union.</d>",
            ],
        }
        checked = {**common, "retries": 1, "check": {"words_similarity_min": 0.8}}

        def history(arguments):
            return None if arguments.get("check") is False else observed(5.5, 1)

        bare = plan(spec, arguments=checked, observed=history)["estimate"]
        assert "floored at the same chain unchecked" in bare["scaled"]
        for entry in spec["cost"]:
            entry["first_at_shape"] = {"minutes": 3.0, "keys": ["num_frames"]}
        shaped = plan(
            spec,
            arguments=checked,
            observed=history,
            seen_shapes=lambda keys: set(),
        )["estimate"]
        assert shaped["minutes"] == pytest.approx(bare["minutes"] + 3.0)
        assert shaped["scaled"].index("floored") < shaped["scaled"].index("first pass")


def test_the_compact_listing_strips_the_key():
    entry = {
        "name": "shaped",
        "cost": [row(SHAPE)],
    }
    slim = _compact_entry(entry, None)
    assert all("first_at_shape" not in item for item in slim["cost"])
    assert entry["cost"][0]["first_at_shape"] == SHAPE


def test_no_catalog_workflow_carries_the_key():
    import glob

    for path in glob.glob("workflows/**/*.json", recursive=True) + glob.glob(
        "dw/workflows/**/*.json", recursive=True
    ):
        assert "first_at_shape" not in open(path).read(), path


def test_build_plan_with_no_shape_key_never_asks(tmp_path):  # noqa: F811
    asked = []
    candidate = workflow_from_definition(
        definition(first=None), str(tmp_path), str(tmp_path), str(tmp_path)
    )
    from unittest.mock import patch

    import dw.plan_downloads

    with patch.object(dw.plan_downloads, "scan_models", return_value={"repos": []}):
        build_plan(
            candidate,
            {},
            device="cuda",
            lookup_sizes=False,
            seen_shapes=lambda keys: asked.append(keys) or set(),
        )
    assert asked == []


class TestServerWiring:
    def test_no_history_store_answers_none(self):
        from types import SimpleNamespace

        from dw.server.deps import seen_shapes_for_name

        state = SimpleNamespace()
        assert seen_shapes_for_name(state, "w", definition(), ["width"]) is None

    def test_the_rows_come_from_this_card(self):
        from types import SimpleNamespace

        from dw.server.deps import seen_shapes_for_name

        asked = {}

        class Costs:
            def rows_on_card(self, name, *, workspace=None, card=None):
                asked.update(name=name, workspace=workspace, card=card)
                return [run(60, {"tier": "draft"})]

        state = SimpleNamespace(observed_costs=Costs())
        shapes = seen_shapes_for_name(
            state, "w", definition(), SHAPE["keys"], workspace="ws", card=1
        )
        assert shapes == {(896.0, 512.0, 121.0)}
        assert asked == {"name": "w", "workspace": "ws", "card": 1}

    def test_admission_passes_the_callable(self):
        import inspect

        from dw.server import admission

        assert "seen_shapes=" in inspect.getsource(admission.validation_plan)
