"""The best-of-n-to-video template wires select/judge the way
docs/proposals/score-and-select-complete.md designed: a for_each fan-out,
a scalar judge with no result block, and select's argmax over the paired
gather: lists. This is a structural check - the reducer engine itself is
tested in tests/test_select.py / tests/test_judge.py, and schema/reference
validity is already swept by the parametrized tests in test_examples.py."""

import json
import os

TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "workflows", "templates", "best-of-n-to-video.json"
)


def load():
    with open(TEMPLATE_PATH) as f:
        return json.load(f)


def steps_by_name(definition):
    return {step["name"]: step for step in definition["steps"]}


def test_the_judge_step_carries_no_result_block():
    """judge returns a scalar - a result block on it is refused by
    dw/scalar_result_validation.py, so the template must never carry one."""
    steps = steps_by_name(load())

    assert "result" not in steps["judge"]


def test_pick_selects_by_argmax_over_the_paired_gather_lists():
    steps = steps_by_name(load())
    pick_task = steps["pick"]["task"]

    assert pick_task["command"] == "select"
    assert pick_task["arguments"]["rule"] == "argmax"
    assert pick_task["arguments"]["candidates"] == ["gather:still"]
    assert pick_task["arguments"]["scores"] == ["gather:judge"]


def test_still_and_judge_share_the_same_for_each_list():
    """judge's previous_result:still must resolve to the matching member by
    position, which only holds if both steps for_each the same variable."""
    steps = steps_by_name(load())

    assert steps["still"]["for_each"] == steps["judge"]["for_each"]


def test_the_template_declares_no_cost_yet():
    """A curated cost figure has to be measured on real GPU hardware - see
    the Global Constraints note in docs/superpowers/plans/
    2026-09-20-tier1-proposals.md. Once someone measures a real run, this
    test should be updated (or removed) alongside adding the real number."""
    definition = load()

    assert "cost" not in definition


def test_saving_steps_are_marked_final_or_intermediate():
    steps = load()["steps"]
    saving_steps = [s for s in steps if "result" in s]

    for step in saving_steps:
        assert step["result"].get("subfolder") in ("final", "intermediate"), (
            f"step '{step['name']}' saves without a subfolder marking"
        )
