"""A step nothing references, and which saves nothing, does not run.

`dialogue-short` cast from portraits that already exist still ran its two
Z-Image steps and threw their output away - roughly 55 s and two model loads
per episode on pictures nothing looked at (#109, #122). Elision drops them,
with four guardrails: a step that saves is kept, the last step is kept, a
release moves onto the step that runs in its place, and every elision is a
warning, because a misspelled reference would otherwise make the step feeding
it vanish silently.
"""

import copy
import json
import pathlib

import pytest

from dw.elision import elide_definition, elide_unreferenced_steps
from dw.workflow import Workflow


def step(name, **extra):
    return {"name": name, **extra}


def task(name, reads=None, **extra):
    arguments = {"audio": f"previous_result:{reads}"} if reads else {"audio": "x"}
    return step(name, task={"command": "fade_audio", "arguments": arguments}, **extra)


def names(steps):
    return [s["name"] for s in steps]


class TestWhatIsDropped:
    def test_a_step_nothing_reads_and_which_saves_nothing(self):
        kept, elided = elide_unreferenced_steps(
            [task("orphan"), task("deliverable", result={"content_type": "audio/wav"})]
        )
        assert names(kept) == ["deliverable"]
        assert elided[0]["step"] == "orphan"
        assert "nothing after it reads" in elided[0]["reason"]

    def test_elision_is_transitive(self):
        """Dropping a step can leave the one it read unreferenced in turn."""
        kept, elided = elide_unreferenced_steps(
            [
                task("first"),
                task("second", reads="first"),
                task("kept", result={"content_type": "audio/wav"}),
            ]
        )
        assert names(kept) == ["kept"]
        assert [e["step"] for e in elided] == ["first", "second"]

    def test_the_records_are_in_written_order(self):
        _, elided = elide_unreferenced_steps(
            [
                task("a"),
                task("b", reads="a"),
                task("last", result={"content_type": "audio/wav"}),
            ]
        )
        assert [e["step"] for e in elided] == ["a", "b"]


class TestTheGuardrails:
    def test_a_step_that_saves_is_kept(self):
        """`save` defaults to true - a workflow whose whole point is writing
        three images references nothing."""
        steps = [
            task("picture", result={"content_type": "image/png"}),
            task("last", result={"content_type": "audio/wav"}),
        ]
        kept, elided = elide_unreferenced_steps(steps)
        assert names(kept) == ["picture", "last"]
        assert elided == []

    def test_a_sub_workflow_step_is_kept_even_without_a_result(self):
        """A composing step's child writes its own files; the parent's
        missing `result` block says nothing about that. Dropping it would
        make the job succeed without the child's output."""
        kept, elided = elide_unreferenced_steps(
            [
                step("score", workflow={"path": "templates/minimax/music3"}),
                task("deliverable", result={"content_type": "audio/wav"}),
            ]
        )
        assert names(kept) == ["score", "deliverable"]
        assert elided == []

    def test_save_false_is_what_makes_it_droppable(self):
        steps = [
            task("picture", result={"content_type": "image/png", "save": False}),
            task("last", result={"content_type": "audio/wav"}),
        ]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["last"]

    def test_the_last_step_is_always_kept(self):
        kept, elided = elide_unreferenced_steps([task("a", result={"save": False})])
        assert names(kept) == ["a"]
        assert elided == []

    def test_a_referenced_step_is_kept(self):
        steps = [task("source"), task("sink", reads="source")]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["source", "sink"]

    def test_a_property_reference_counts(self):
        """`previous_result:segment.mask` reads `segment`."""
        steps = [
            task("segment"),
            step(
                "use",
                task={
                    "command": "fade_audio",
                    "arguments": {"audio": "previous_result:segment.mask"},
                },
            ),
        ]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["segment", "use"]

    def test_a_from_previous_result_object_counts(self):
        steps = [
            task("draw"),
            step(
                "shot",
                pipeline={
                    "arguments": {
                        "references": [
                            {"reference_type": "X", "from_previous_result": "draw"}
                        ]
                    }
                },
            ),
        ]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["draw", "shot"]

    def test_a_pipeline_reference_counts(self):
        """The step is read for its loaded pipeline rather than its result."""
        steps = [
            step("load", pipeline={"configuration": {"component_type": "X"}}),
            step("reuse", pipeline_reference={"reference_name": "load"}),
        ]
        kept, elided = elide_unreferenced_steps(steps)
        assert names(kept) == ["load", "reuse"]
        assert elided == []

    def test_a_shared_component_counts(self):
        """Sharing is keyed on the component's name, so the step that shares
        is referenced without ever being named."""
        steps = [
            step("loader", pipeline={"shared_components": ["text_encoder"]}),
            step("user", pipeline={"reused_components": ["text_encoder"]}),
        ]
        kept, elided = elide_unreferenced_steps(steps)
        assert names(kept) == ["loader", "user"]
        assert elided == []


class TestReleasesMoveRatherThanDisappearing:
    PIPELINE = {
        "configuration": {"component_type": "ZImagePipeline"},
        "from_pretrained_arguments": {"model_name": "Tongyi-MAI/Z-Image-Turbo"},
    }

    def draw(self, name, prompt, **extra):
        return step(
            name,
            pipeline={**copy.deepcopy(self.PIPELINE), "arguments": {"prompt": prompt}},
            **extra,
        )

    def test_a_release_carries_onto_the_step_that_loaded_the_same_pipeline(self):
        """Two identical pipeline definitions share one loaded model, so the
        release belongs on whichever of them still runs."""
        steps = [
            self.draw("a", "one"),
            self.draw("b", "two", release_pipeline=True),
            task("uses_a", reads="a", result={"content_type": "audio/wav"}),
        ]
        kept, elided = elide_unreferenced_steps(steps)
        assert names(kept) == ["a", "uses_a"]
        assert kept[0]["release_pipeline"] is True
        assert "release was carried" in elided[0]["reason"]

    def test_a_release_with_nothing_before_it_is_dropped(self):
        """Nothing loaded, so nothing leaks."""
        steps = [
            self.draw("a", "one", release_pipeline=True),
            task("last", result={"content_type": "audio/wav"}),
        ]
        kept, elided = elide_unreferenced_steps(steps)
        assert names(kept) == ["last"]
        assert "release was carried" not in elided[0]["reason"]

    def test_a_release_is_not_moved_onto_a_different_pipeline(self):
        """Moving it would unload something the workflow never asked to
        unload."""
        other = step(
            "other",
            pipeline={"configuration": {"component_type": "FluxPipeline"}},
            result={"content_type": "image/png"},
        )
        steps = [
            other,
            self.draw("a", "one", release_pipeline=True),
            task("last", result={"content_type": "audio/wav"}),
        ]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["other", "last"]
        assert "release_pipeline" not in kept[0]

    def test_release_models_always_carries(self):
        """It frees the process-wide task-model cache rather than one
        pipeline, so the step that ran before is exactly where it belongs."""
        steps = [
            task("earlier", result={"content_type": "audio/wav"}),
            task("orphan", release_models=True),
            task("last", result={"content_type": "audio/wav"}),
        ]
        kept, _ = elide_unreferenced_steps(steps)
        assert names(kept) == ["earlier", "last"]
        assert kept[0]["release_models"] is True


class TestTheWarning:
    """Without it a misspelled reference makes the step feeding it vanish and
    the failure moves from "previous result not found" to "the picture is
    wrong". A log line is not enough - a consumer over the API or MCP reads
    the job's warnings and nothing else (#82)."""

    def test_every_elided_step_is_named(self):
        from dw.elision import warn_elided
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            warn_elided([{"step": "draw_character_a", "reason": "nothing reads it"}])
        finally:
            deactivate_context(token)

        warnings = [e for e in events if e["event"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["kind"] == "step_elided"
        assert warnings[0]["step"] == "draw_character_a"
        assert "did not run" in warnings[0]["message"]


class TestASuppliedReferenceIsNotAMisspelling:
    """#157: `music-video`'s singer portrait is elided *because the caller
    supplied one*, and the warning said their reference was probably
    misspelled. The two cases are indistinguishable from `job.warnings`
    otherwise, and one of them is the documented happy path."""

    def written(self):
        return {
            "id": "w",
            "variables": {"singer_reference": {"from_previous_result": "draw_singer"}},
            "steps": [
                task("draw_singer"),
                step(
                    "shot",
                    task={
                        "command": "fade_audio",
                        "arguments": {"audio": "variable:singer_reference"},
                    },
                    result={"content_type": "video/mp4"},
                ),
            ],
        }

    def test_the_overriding_variable_is_named(self):
        from dw.elision import overriding_variables

        supplied = [task("draw_singer"), task("shot")]
        assert overriding_variables(self.written(), supplied) == {
            "draw_singer": "singer_reference"
        }

    def test_a_variable_still_reading_the_step_is_not_an_override(self):
        from dw.elision import overriding_variables

        still = [task("draw_singer"), task("shot", reads="draw_singer")]
        assert overriding_variables(self.written(), still) == {}

    def test_a_variable_no_step_reads_is_not_how_the_step_was_reached(self):
        """A dead variable naming a step says nothing about why the step went
        unread, so the general diagnosis stands."""
        from dw.elision import overriding_variables

        written = self.written()
        written["steps"][1]["task"]["arguments"]["audio"] = "x"
        assert overriding_variables(written, [task("draw_singer"), task("shot")]) == {}

    def test_the_reason_says_what_was_supplied(self):
        written = self.written()
        definition = copy.deepcopy(written)
        # As substitution leaves it once the caller passes a portrait
        elided = elide_definition(definition, written)
        assert elided == [
            {
                "step": "draw_singer",
                "reason": "'singer_reference' was supplied, so nothing reads "
                "its result",
                "overridden_by": "singer_reference",
            }
        ]

    def test_without_the_written_definition_it_is_the_old_diagnosis(self):
        """Every other caller of elide_definition is unchanged."""
        elided = elide_definition(copy.deepcopy(self.written()))
        assert "nothing after it reads" in elided[0]["reason"]
        assert "overridden_by" not in elided[0]

    def test_the_warning_does_not_suggest_a_typo(self):
        from dw.elision import warn_elided
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            warn_elided(
                [
                    {
                        "step": "draw_singer",
                        "reason": "'singer_reference' was supplied, so nothing "
                        "reads its result",
                        "overridden_by": "singer_reference",
                    }
                ]
            )
        finally:
            deactivate_context(token)

        warning = [e for e in events if e["event"] == "warning"][0]
        assert "misspelled" not in warning["message"]
        assert "singer_reference" in warning["message"]
        assert warning["overridden_by"] == "singer_reference"

    def test_an_orphan_still_gets_the_diagnosis(self):
        """A step nothing ever read is the case the advice exists for."""
        from dw.elision import warn_elided
        from dw.events import RunContext, activate_context, deactivate_context

        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            warn_elided([{"step": "orphan", "reason": "nothing reads it"}])
        finally:
            deactivate_context(token)

        assert (
            "misspelled" in [e for e in events if e["event"] == "warning"][0]["message"]
        )


class TestTheMusicVideoSinger:
    """#146's happy path, end to end through the template itself."""

    PATH = "workflows/templates/minimax/music-video.json"

    def definition(self):
        return json.loads(pathlib.Path(self.PATH).read_text())

    def test_a_supplied_portrait_elides_without_suggesting_a_mistake(self):
        written = self.definition()
        definition = copy.deepcopy(written)
        definition["variables"]["singer_reference"] = {
            "reference_type": "variable:image_reference_type",
            "from_file": "asset:cast/priya.jpg",
        }
        expanded = Workflow(definition, "outputs", self.PATH).expanded_definition()

        elided = elide_definition(expanded, written)

        assert [e["step"] for e in elided] == ["draw_singer"]
        assert elided[0]["overridden_by"] == "singer_reference"


class TestDialogueShort:
    """The case that raised it."""

    PATH = "workflows/templates/minimax/dialogue-short.json"

    def definition(self):
        return json.loads(pathlib.Path(self.PATH).read_text())

    def expanded(self, definition):
        return Workflow(definition, "outputs", self.PATH).expanded_definition()

    def cast_from_files(self, definition):
        for entry in definition["variables"]["shots"]:
            for reference in entry.get("references", []):
                if "from_previous_result" in reference:
                    name = reference.pop("from_previous_result")
                    reference["from_file"] = f"asset:cast/{name}.jpg"
        return definition

    def test_the_draw_steps_save_nothing(self):
        """Don's condition on the engine change: without `save: false` the
        deliverable guardrail keeps them and the motivating case saves
        nothing."""
        steps = {s["name"]: s for s in self.definition()["steps"]}
        for name in ("draw_character_a", "draw_character_b"):
            assert steps[name]["result"]["save"] is False

    def test_the_default_run_is_unchanged(self):
        expanded = self.expanded(self.definition())
        elided = elide_definition(expanded)
        assert elided == []
        assert "draw_character_a" in names(expanded["steps"])

    def test_a_cast_episode_draws_nothing(self):
        expanded = self.expanded(self.cast_from_files(self.definition()))
        elided = elide_definition(expanded)
        assert [e["step"] for e in elided] == [
            "draw_character_a",
            "draw_character_b",
        ]
        assert not [n for n in names(expanded["steps"]) if n.startswith("draw_")]

    def test_a_half_cast_episode_still_draws(self):
        """Something still reads them, so they still run."""
        definition = self.definition()
        entry = definition["variables"]["shots"][0]
        for reference in entry.get("references", []):
            reference.pop("from_previous_result", None)
        expanded = self.expanded(definition)
        assert elide_definition(expanded) == []


class TestTheCatalogIsUnchanged:
    """Elision is a property every workflow inherits - no template may lose a
    step to it on its stored defaults."""

    @pytest.mark.parametrize(
        "path", sorted(str(p) for p in pathlib.Path("workflows").rglob("*.json"))
    )
    def test_no_step_is_elided_on_the_defaults(self, path):
        definition = json.loads(pathlib.Path(path).read_text())
        if not isinstance(definition, dict) or "steps" not in definition:
            pytest.skip("not a workflow")
        expanded = Workflow(definition, "outputs", path).expanded_definition()
        assert elide_definition(expanded) == []


class TestMusicVideo:
    """#146: a standing cast member sings, without copying the template."""

    PATH = "workflows/templates/minimax/music-video.json"

    def definition(self):
        return json.loads(pathlib.Path(self.PATH).read_text())

    def expanded(self, definition):
        return Workflow(definition, "outputs", self.PATH).expanded_definition()

    def test_the_singer_reference_is_one_argument(self):
        """The shot step reads it from a variable, so `arguments` reaches it.

        It was written into steps[].pipeline.arguments.references, where no
        argument could reach it, and reusing a portrait meant resubmitting the
        whole template inline.
        """
        definition = self.definition()
        assert definition["variables"]["singer_reference"] == {
            "reference_type": "variable:image_reference_type",
            "from_previous_result": "draw_singer",
        }
        shot = next(s for s in definition["steps"] if s["name"] == "shot")
        assert shot["pipeline"]["arguments"]["references"][0] == (
            "variable:singer_reference"
        )

    def test_the_portrait_saves_nothing(self):
        """Same role as dialogue-short's draw steps, so the same rule: a
        conditioning image is not a deliverable, and the guardrail would
        otherwise keep the step a cast episode has no use for."""
        steps = {s["name"]: s for s in self.definition()["steps"]}
        assert steps["draw_singer"]["result"]["save"] is False

    def test_a_cast_singer_draws_nothing(self):
        definition = self.definition()
        definition["variables"]["singer_reference"] = {
            "reference_type": "variable:image_reference_type",
            "from_file": "asset:qa-cast/priya-portrait.jpg",
        }
        expanded = self.expanded(definition)
        assert [e["step"] for e in elide_definition(expanded)] == ["draw_singer"]

    def test_the_default_run_still_draws(self):
        expanded = self.expanded(self.definition())
        assert elide_definition(expanded) == []
        assert "draw_singer" in names(expanded["steps"])
