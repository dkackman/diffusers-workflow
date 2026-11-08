"""The dw plugin: skills that teach an agent to compose a model family's
workflows, kept true by tests rather than by memory.

A skill quotes catalog names and states numeric rules. Each name has to
resolve to a file, each number has to come from the library that enforces
it, and the plugin's version has to be the engine's - an installed skill is
matched to the server it was written against by that number.
"""

import copy
import glob
import inspect
import json
import os
import re

import pytest

from tests.test_examples import REPO_ROOT

PLUGIN_DIR = os.path.join(REPO_ROOT, "plugins", "dw")
SKILLS = sorted(glob.glob(os.path.join(PLUGIN_DIR, "skills", "*", "SKILL.md")))
SKILL_SIZE_LIMIT = 12 * 1024

CATALOG_NAME = re.compile(r"`((?:templates|models)/[A-Za-z0-9_./-]+)`")
REFERENCE_LINK = re.compile(r"`(references/[A-Za-z0-9_.-]+\.md)`")


def skill_body(path):
    """SKILL.md alone - what loads whenever the skill triggers."""
    return open(path, encoding="utf-8").read()


def skill_references(path):
    """The files beside a skill that it sends an agent to read on demand."""
    folder = os.path.join(os.path.dirname(path), "references")
    return sorted(glob.glob(os.path.join(folder, "*.md")))


def skill_text(path):
    """A skill and its references, so a rule moved out of SKILL.md is still
    held to the library that enforces it."""
    parts = [skill_body(path)]
    parts += [open(ref, encoding="utf-8").read() for ref in skill_references(path)]
    return "\n".join(parts)


def frontmatter(text):
    """The YAML block between the leading '---' lines, as a dict of the
    top-level 'key: value' pairs (the two the plugin format needs)."""
    assert text.startswith("---\n"), "a skill begins with frontmatter"
    end = text.index("\n---", 4)
    fields = {}
    for line in text[4:end].splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
    return fields


def _pyproject_version():
    text = open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8").read()
    return re.search(r'^version = "(.*)"$', text, re.M).group(1)


def test_the_marketplace_names_the_plugin():
    import json

    manifest = json.load(
        open(
            os.path.join(REPO_ROOT, ".claude-plugin", "marketplace.json"),
            encoding="utf-8",
        )
    )

    assert manifest["name"] == "diffusers-workflow"
    (plugin,) = manifest["plugins"]
    assert plugin["name"] == "dw"
    assert plugin["source"] == "./plugins/dw"


def test_the_plugin_version_is_the_engine_version():
    """An installed plugin is matched to the engine it was written against by
    this number, so the release script bumps both in one commit."""
    import json

    plugin = json.load(
        open(
            os.path.join(PLUGIN_DIR, ".claude-plugin", "plugin.json"), encoding="utf-8"
        )
    )

    assert plugin["name"] == "dw"
    assert plugin["version"] == _pyproject_version()


def test_there_are_skills():
    assert SKILLS, "the plugin ships at least one skill"


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_has_a_triggering_description_under_the_size_cap(path):
    """The cap is on SKILL.md, which loads every time the skill triggers;
    detail only some requests need goes in `references/`, read on demand."""
    text = skill_body(path)
    fields = frontmatter(text)

    assert fields["name"] == os.path.basename(os.path.dirname(path))
    assert "description" in fields and len(fields["description"]) > 40
    assert len(text.encode("utf-8")) <= SKILL_SIZE_LIMIT, (
        f"{path} is over {SKILL_SIZE_LIMIT} bytes"
    )


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_links_each_reference_and_each_link_resolves(path):
    """A reference no SKILL.md names is never read, and a link to a file that
    is not there sends the agent nowhere."""
    linked = set(REFERENCE_LINK.findall(skill_body(path)))
    shipped = {"references/" + os.path.basename(ref) for ref in skill_references(path)}

    assert linked == shipped, (
        f"{path}: linked but missing {sorted(linked - shipped)}, "
        f"shipped but never linked {sorted(shipped - linked)}"
    )


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_every_catalog_name_a_skill_quotes_resolves(path):
    """A renamed template fails here rather than in a cold session."""
    names = CATALOG_NAME.findall(skill_text(path))

    assert names, f"{path} quotes no catalog names"
    for name in names:
        target = os.path.join(
            REPO_ROOT, "workflows", name.removesuffix(".json") + ".json"
        )
        assert os.path.isfile(target), (
            f"{path} quotes {name}, which is not a workflow ({target})"
        )


# The task commands a skill sends an agent to by name. A skill cannot be
# swept for these the way catalog names are - a backticked identifier may be
# a tool, a field or a task - so each one is listed here.
SKILL_TASKS = {
    "minimax-music3": ["attribute_voices"],
    "minimax-h3": ["join_into_song", "slice_audio", "normalize_audio", "pair_audio"],
    "script-to-video": ["concat_videos", "get_first_frame", "get_last_frame"],
    "series-episodes": ["apply_lut", "film_grain", "concat_videos", "loop_audio"],
    "long-form": ["pair_audio", "export_timeline", "split_takes", "find_loop_bed"],
}


@pytest.mark.parametrize(
    "skill,task",
    [(skill, task) for skill, tasks in SKILL_TASKS.items() for task in tasks],
)
def test_every_task_a_skill_names_is_in_list_tasks(skill, task):
    """A renamed task fails here rather than in a cold session."""
    from dw.introspection import list_tasks

    path = os.path.join(PLUGIN_DIR, "skills", skill, "SKILL.md")
    assert f"`{task}`" in skill_text(path)
    tasks = list_tasks()
    # get_first_frame and its kin are listed as video processors, not commands
    assert task in tasks["commands"] + tasks["video_processors"]


H3_SKILL = os.path.join(PLUGIN_DIR, "skills", "minimax-h3", "SKILL.md")


class TestMiniMaxH3Skill:
    """The numbers the H3 skill states come from the diffusers modular
    pipeline that enforces them, so a library change fails here."""

    def test_the_frame_rule_and_bounds_are_the_pipeline_s(self):
        import inspect

        from diffusers.modular_pipelines.minimax_h3 import (
            before_encoder,
            modular_pipeline,
        )

        text = skill_text(H3_SKILL)
        assert "17n + 5" in text or "17 * n + 5" in text
        assert "17 * n + 5" in inspect.getsource(before_encoder)
        assert modular_pipeline.MINIMAX_H3_FPS == 24 and "24 fps" in text
        # 124 and 345 are the smallest and largest 17n + 5 inside 5 to 15 seconds at 24 fps
        assert "124" in text and "345" in text
        # min_duration/max_duration are instance properties on MiniMaxH3ModularPipeline, so
        # the window is pinned by the check that reads them rather than by their values
        assert (
            "min_duration <= duration <= components.max_duration"
            in inspect.getsource(before_encoder)
        )
        assert (
            124 / modular_pipeline.MINIMAX_H3_FPS >= 5
            and 345 / modular_pipeline.MINIMAX_H3_FPS <= 15
        )

    def test_the_denoise_step_count_is_the_scheduler_s(self):
        """#110: `denoise_total_steps` comes back one less than the
        `num_inference_steps` asked for, on every H3 run. Not a dropped step
        and not an off-by-one in our progress reporting - MiniMaxH3Scheduler
        counts sigma grid points with the terminal zero among them, so the
        schedule it builds evaluates the model N-1 times, and the bar we
        report is `len(scheduler.timesteps)`. Pinned here because from
        outside the two are indistinguishable, which is what got it filed.
        """
        from diffusers import MiniMaxH3Scheduler

        scheduler = MiniMaxH3Scheduler(shift=12.0)
        for requested, evaluations in ((5, 4), (9, 8), (20, 19)):
            scheduler.set_timesteps(requested)
            assert len(scheduler.timesteps) == evaluations

        text = skill_text(H3_SKILL)
        # The 4-step default since 2026-10-10
        assert "denoise_total_steps" in text and "reports 4" in text

    def test_the_canvas_rules_are_the_pipeline_s(self):
        import inspect

        from diffusers.modular_pipelines.minimax_h3 import (
            before_encoder,
            modular_pipeline,
        )

        text = skill_text(H3_SKILL)
        source = inspect.getsource(before_encoder)
        assert 'ConfigSpec("canvas_short_edge", 768)' in source and "768" in text
        assert "768 * 1344" in source and "1344" in text
        assert modular_pipeline.MINIMAX_H3_MIN_ASPECT_RATIO == 1 / 4
        assert modular_pipeline.MINIMAX_H3_MAX_ASPECT_RATIO == 4
        assert "1:4" in text and "4:1" in text
        assert "multiples of 32" in text

    def test_the_reference_and_audio_limits_are_the_pipeline_s(self):
        import inspect

        from diffusers.modular_pipelines.minimax_h3 import modular_pipeline
        from diffusers.modular_pipelines.minimax_h3.before_encoder import (
            MiniMaxH3Ref2VASetupStep,
        )

        text = skill_text(H3_SKILL)
        limits = inspect.signature(MiniMaxH3Ref2VASetupStep.__init__).parameters
        assert limits["max_images"].default == 9 and "9 images" in text
        assert limits["max_videos"].default == 3 and "3 videos" in text
        assert limits["max_audios"].default == 3 and "3 audio clips" in text
        assert limits["max_references"].default == 12 and "12 files" in text
        # the 32 kHz rate is the audio VAE's, with this literal as the fallback
        assert "return 32000" in inspect.getsource(modular_pipeline)
        assert (
            modular_pipeline.MINIMAX_H3_AUDIO_CHANNELS == 2 and "32 kHz stereo" in text
        )
        assert "audio can" in text and "never be the only reference" in text

    def test_the_checkpoint_coupling_is_stated(self):
        """A checkpoint comes with its canvas, its shift and its alpha.

        Six reference-only templates used to carry no LoRA and run 20 steps,
        because the only turbo LoRA was distilled against the base transformer
        while they load the reference one. A Ref2VA turbo LoRA ended that
        (#149), and the skill has to say what now goes with what - the
        2026-09-08 drill caught it saying 20 for storyboard, and a wrong shift
        or alpha is the same class of error with no error message.
        """
        import json

        text = skill_text(H3_SKILL)
        assert "Never put an FL2VA LoRA on a reference template" in text
        assert "video_shift" in text and "lora_alpha" in text
        assert "alpha 128" not in text
        assert "alpha unset" in text

        for name in (
            "storyboard",
            "dialogue-short",
            "music-video",
            "chain-matched-and-aligned",
            "reference-to-video",
            "composable-references",
            "voice-timbre-reference",
            "generated-subject-reference",
            "chain-matched-to-audio",
            "chain-video-continuity",
        ):
            path = os.path.join(
                REPO_ROOT, "workflows", "templates", "minimax", name + ".json"
            )
            spec = open(path, encoding="utf-8").read()
            definition = json.loads(spec)
            assert "ref2v" in definition["variables"]["lora_weight_name"], name
            assert definition["variables"]["num_inference_steps"] == 9, name
            assert definition["variables"]["video_shift"] == 12.0, name

    def test_the_768p_path_is_offered_and_its_numbers_are_right(self):
        """The three that move together, on the one template that differs."""
        import json

        text = skill_text(H3_SKILL)
        assert "video-with-audio-768p" in text

        path = os.path.join(
            REPO_ROOT,
            "workflows",
            "templates",
            "minimax",
            "video-with-audio-768p.json",
        )
        variables = json.load(open(path, encoding="utf-8"))["variables"]
        assert variables["width"] == 1344 and variables["height"] == 768
        assert variables["video_shift"] == 6.0
        assert variables["audio_shift"] == 3.0
        assert variables["lora_alpha"] is None
        # The 4-step v1.2 file, trained at 768p (2026-10-10)
        assert variables["num_inference_steps"] == 5
        assert "768p" in variables["lora_weight_name"]

    def test_the_skill_defers_prompt_format_to_minimax(self):
        text = skill_text(H3_SKILL)
        assert "h3-prompt-writing" in text
        assert "npx skills add MiniMax-AI/MiniMax-H3 --skill h3-prompt-writing" in text
        assert "VIDEO_PROMPT_WRITING_GUIDE_base_en.md" in text
        assert "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md" in text
        assert "`templates/minimax/enhance-prompt`" in text
        # no transcription of the format: none of its section labels appear as instructions
        assert "retention_analysis:" not in text

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(H3_SKILL)
        assert text.index("get_server_info") < text.index("templates/minimax/")

    def test_the_cuts_templates_are_described_as_list_driven(self):
        """Both cut templates take one 'shots' list; the skill says what an
        entry carries, so an agent writes entries rather than the shot_N_*
        arguments T005 and this rewrite removed."""
        import json

        text = skill_text(H3_SKILL)
        assert "`shots`" in text
        assert "shot_1_" not in text and "shot_2_" not in text
        for name, fields in (
            ("dialogue-short", {"name", "prompt", "references", "num_frames"}),
            (
                "music-video",
                {
                    "name",
                    "prompt",
                    "start_frame",
                    "num_frames",
                    "lead_frames",
                    "cut_frames",
                },
            ),
        ):
            path = os.path.join(
                REPO_ROOT, "workflows", "templates", "minimax", name + ".json"
            )
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            entries = spec["variables"]["shots"]
            assert all(set(entry) == fields for entry in entries), name
            for field in fields:
                assert f"`{field}`" in text, f"the skill does not name {field}"
        # cost: the listing's per_entry block when present, the honest
        # fallback when it is not
        assert "`per_entry`" in text
        assert "`lists`" in text

    def test_the_768p_ref2va_tradeoff_points_at_the_recipes_guide(self):
        """Ref2VA reaches 1344x768 too (field report #484 item 6), but the
        skill's byte cap can't hold the long-form trade-off - it has to
        point at the recipes guide, and the guide has to actually say it.

        No diffusers symbol enforces these figures (they're a measured
        field report, not a library constant), so this pins the literal
        numbers directly rather than deriving them.
        """
        text = skill_text(H3_SKILL)
        assert "1344x768" in text
        assert "`recipes` guide" in text and "MiniMax-H3" in text

        from dw.server.guides import get_guide

        guide = get_guide("recipes", section="MiniMax-H3")["content"]
        assert "1344x768" in guide
        assert "175" in guide and "17n+5" in guide
        assert "31 minutes" in guide

    def test_the_guides_argument_is_offered_and_its_numbers_are_the_engine_s(self):
        """#648: an agent working from the skill alone has to find `guides`.
        The limit and the frame step come from the guide layout that
        enforces them, and the section the skill points at has to exist."""
        from dw.pipeline_processors.h3_rules import GUIDE_FRAMES_PER_CHUNK, GUIDE_LIMIT
        from dw.server.guides import get_guide

        body = skill_body(H3_SKILL)
        section = "H3: holding a clip with `guides`"
        assert "`guides`" in body
        assert f'section="{section}"' in body
        assert f"up to {GUIDE_LIMIT} clips" in body
        # the chain adds a guide at frame 0, so one fewer of the caller's own
        assert f"{GUIDE_LIMIT - 1} on a guide chain" in " ".join(body.split())
        assert f"multiple of {GUIDE_FRAMES_PER_CHUNK}" in body
        assert "`guides`" in get_guide("workflows", section=section)["content"]
        # C-F301: a full-length guide copies the take, so the skill must not
        # offer guides as a restyle
        assert "restyle a take" not in body
        assert "does not restyle" in " ".join(body.split())

    def test_the_guide_chain_lengths_the_skill_offers_are_the_engines(self):
        """Review 2026-10-07: `guide_frames` 22 or 39 is hand-written in the
        skill; the engine's GUIDE_CHAIN_FRAMES is the rule it must match."""
        from dw.pipeline_processors.h3_rules import GUIDE_CHAIN_FRAMES

        body = " ".join(skill_body(H3_SKILL).split())
        offered = " or ".join(str(frames) for frames in GUIDE_CHAIN_FRAMES)
        assert f"`guide_frames` {offered}" in body

    def test_the_dialogue_into_a_song_recipe_is_the_task_s_and_the_guide_s(self):
        """#514: the skill states the recipe in brief and points at the
        `workflows` guide for the worked tail. The parameter names are
        join_into_song's own, the -3 dBFS is music-video's `balanced` step,
        and the guide's tail validates as a workflow, so a rename or a
        re-level fails here rather than in a session following the text."""
        from dw.server.guides import get_guide
        from dw.tasks.join_into_song import join_into_song
        from dw.workflow import Workflow

        text = skill_text(H3_SKILL)
        parameters = inspect.signature(join_into_song).parameters
        for name in ("cue_seconds", "song_shots", "song"):
            assert name in parameters
            assert f"`{name}`" in text

        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "music-video.json"
        )
        steps = json.load(open(path, encoding="utf-8"))["steps"]
        (balanced,) = [step for step in steps if step["name"] == "balanced"]
        assert balanced["task"]["arguments"]["peak_dbfs"] == -3.0
        assert "-3 dBFS" in text

        section = "A spoken scene breaking into a song"
        assert f'"{section}"' in text
        guide = get_guide("workflows", section=section)["content"]
        for word in (
            "`join_into_song`",
            "`cue_seconds`",
            "`slice_audio`",
            "`normalize_audio`",
            "`pair_audio`",
            "-3 dBFS",
        ):
            assert word in guide

        (tail,) = re.findall(r"```json\n(.*?)```", guide, re.DOTALL)
        definition = {"id": "number", "steps": json.loads(f"[{tail}]")}
        (normalize,) = [
            step
            for step in definition["steps"]
            if step["task"]["command"] == "normalize_audio"
        ]
        assert normalize["task"]["arguments"]["peak_dbfs"] == -3.0
        workflow = Workflow(definition, REPO_ROOT, os.path.join(REPO_ROOT, "n.json"))
        assert workflow.validation_errors() == []

    def test_the_locked_frame_lip_sync_recipe_names_real_task_types(self):
        """#985: a locked frame that lip-syncs is FL2VA with the song held,
        which keeps the pixels; #852's Ref2VA with a first-frame keyframe and
        the song held is the fallback. "FL2VA takes no audio" was false as
        stated, and stays out. The task types are MiniMax's, which the
        built-in Context-IR enhancer teaches, and `hold_audio` is the argument
        music-video's ref2va step takes, so a rename of either fails here."""
        body = " ".join(skill_body(H3_SKILL).split())
        assert "FL2VA takes no audio" not in body
        assert (
            "FL2VA with the song held by `hold_audio` keeps the first frame's pixels"
            in body
        )
        task = "[keyframe completion + audio reuse]"
        assert f"Fallback: Ref2VA `{task}`" in body
        assert "`hold_audio`" in body

        enhancer = os.path.join(REPO_ROOT, "dw", "workflows", "h3_context_ir.json")
        system_prompt = open(enhancer, encoding="utf-8").read()
        for name in ("keyframe completion", "audio reuse"):
            assert name in system_prompt

        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "music-video.json"
        )
        steps = json.load(open(path, encoding="utf-8"))["steps"]
        assert any(
            step.get("pipeline", {})
            .get("from_pretrained_arguments", {})
            .get("workflow")
            == "ref2va"
            and "hold_audio" in step["pipeline"].get("arguments", {})
            for step in steps
        )

    def test_the_1344x768_reference_ceilings_are_the_templates_vram_estimate(self):
        """#479: `gb_per_reference` and the per-step projection moved the
        Ref2VA ceiling from one number (175 frames, unstated reference count)
        to one per reference count. Derived here from
        reference-to-video.json's own `vram_estimate`/`cost`/
        `variable_constraints` rather than hardcoded, so a recalibration of
        the template fails this rather than leaving the skill stale.
        """
        import json

        from dw.vram_estimate import required_gb

        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "reference-to-video.json"
        )
        definition = json.load(open(path, encoding="utf-8"))
        estimate = definition["vram_estimate"]
        (cost_entry,) = definition["cost"]
        capacity = cost_entry["vram_gb"]
        constraint = definition["variable_constraints"]["num_frames"]
        width, height = 1344, 768

        def max_frames_for(references):
            best = None
            n = 0
            while True:
                frames = constraint["modulus"] * n + constraint["remainder"]
                if frames > constraint["max_frames"]:
                    return best
                if frames >= constraint["min_frames"]:
                    values = {"width": width, "height": height, "num_frames": frames}
                    if required_gb(estimate, values, references) <= capacity:
                        best = frames
                n += 1

        ceilings = {refs: max_frames_for(refs) for refs in (1, 2, 3, 4)}
        assert ceilings == {1: 243, 2: 209, 3: 175, 4: 141}

        text = skill_text(H3_SKILL)
        for frames in ceilings.values():
            assert str(frames) in text

    def test_the_regional_edit_branch_is_the_template_s_and_the_guide_s(self):
        """#1078: "fix one region of a good take" sends an agent to a template
        whose edit step the engine's own checks pass, at the strength and grain
        the guide measured."""
        from dw.hold_audio import (
            hold_audio_errors,
            mask_errors,
            refine_strength_errors,
        )
        from dw.pipeline_processors import h3_mask_cells

        text = skill_text(H3_SKILL)
        assert "`templates/minimax/regional-edit`" in text
        assert "`references/regional-edit.md`" in text
        with open(
            os.path.join(os.path.dirname(H3_SKILL), "references", "regional-edit.md")
        ) as f:
            reference = f.read()

        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "regional-edit.json"
        )
        with open(path) as f:
            template = json.load(f)
        variables = template["variables"]
        assert variables["refine_strength"] == 0.9
        # 5: the 4-step turbo file plus the terminal zero, the bands' step count
        assert variables["num_inference_steps"] == 5
        (step,) = template["steps"]
        pipeline = step["pipeline"]
        assert pipeline["from_pretrained_arguments"]["workflow"] == "t2va"
        arguments = pipeline["arguments"]
        assert arguments["source_video"] == "variable:source_video"
        assert arguments["mask_image"] == "variable:mask_image"
        assert arguments["hold_audio"] == "variable:source_video"
        assert arguments["refine_strength"] == "variable:refine_strength"

        # The engine's static checks on the substituted step: the arguments
        # rules pass (the placeholder assets are not on disk, so no probe)
        substituted = copy.deepcopy(template)
        for name, value in substituted["steps"][0]["pipeline"]["arguments"].items():
            if isinstance(value, str) and value.startswith("variable:"):
                substituted["steps"][0]["pipeline"]["arguments"][name] = variables[
                    value.removeprefix("variable:")
                ]
        assert mask_errors(substituted) == []
        assert refine_strength_errors(substituted) == []
        assert hold_audio_errors(substituted) == []

        # The bands and the grain are the guide's section, word for word
        guide = open(os.path.join(REPO_ROOT, "docs", "WORKFLOW_GUIDE.md")).read()
        section = guide.split("#### Choosing refine_strength for a regional edit")[1]
        section = section.split("\n#")[0]
        assert "seed 1062, 5 turbo steps" in section and "seed 1062" in reference
        assert "`video_shift` 6" in section and "`video_shift` 6" in reference
        for strength in ("0.85 or less", "| 0.9 |", "| 0.95 |", "| 1.0 |"):
            assert strength in section and strength in reference
        assert (
            "did not land below 1.0" in section
            and "did not land below 1.0" in reference
        )
        # Both wrap their prose, so the grain is matched with the lines joined
        assert "16 px times the patch size" in " ".join(guide.split())
        assert "16 px times the patch size" in " ".join(reference.split())
        assert "`16 * patch`" in inspect.getsource(h3_mask_cells)


LTX_SKILL = os.path.join(PLUGIN_DIR, "skills", "ltx-2.5", "SKILL.md")


def _fenced_block_after(text, heading):
    """The first fenced code block after a markdown heading."""
    start = text.index(heading)
    open_fence = text.index("\n```", start)
    open_end = text.index("\n", open_fence + 1)
    close_fence = text.index("\n```", open_end)
    return text[open_end + 1 : close_fence]


class TestLtx25Skill:
    """The LTX-2.5 skill's numbers come from the pipeline that enforces them,
    and the one vendor text it quotes is the library's own constant."""

    def test_the_schedule_is_the_library_s(self):
        from diffusers.pipelines.ltx2.utils import (
            DISTILLED_SIGMA_VALUES,
            STAGE_2_DISTILLED_SIGMA_VALUES,
        )

        text = skill_text(LTX_SKILL)
        assert len(DISTILLED_SIGMA_VALUES) == 8 and "eight" in text
        assert len(STAGE_2_DISTILLED_SIGMA_VALUES) == 3 and "three" in text
        assert str(STAGE_2_DISTILLED_SIGMA_VALUES[0]) in text

    def test_the_size_and_frame_rules_are_the_pipeline_s(self):
        import inspect

        from diffusers.pipelines.ltx2 import pipeline_ltx2
        from diffusers.pipelines.ltx2.utils import MAX_CONDITIONING_FPS

        text = skill_text(LTX_SKILL)
        source = inspect.getsource(pipeline_ltx2)
        assert (
            "height % 32 != 0 or width % 32 != 0" in source
            and "multiples of 32" in text
        )
        assert "(num_frames - 1) // self.vae_temporal_compression_ratio + 1" in source
        assert "8k + 1" in text or "8n + 1" in text
        assert MAX_CONDITIONING_FPS == 60.0 and "60" in text

    def test_the_image_condition_crf_is_the_library_s(self):
        from diffusers.pipelines.ltx2.utils import LTX2_5_IMAGE_CRF

        assert LTX2_5_IMAGE_CRF == 18
        assert "CRF 18" in skill_text(LTX_SKILL)

    def test_the_quoted_caption_spec_is_the_library_constant(self):
        """The one vendor text the plugin carries, tied to the library that
        ships it so it cannot drift."""
        from diffusers.pipelines.ltx2.utils import LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

        quoted = _fenced_block_after(
            skill_text(LTX_SKILL), "## The trained caption spec"
        )

        assert " ".join(quoted.split()) == " ".join(
            LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT.split()
        )

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(LTX_SKILL)
        assert text.index("get_server_info") < text.index("templates/ltx2/")


KANDINSKY6_SKILL = os.path.join(PLUGIN_DIR, "skills", "kandinsky-6", "SKILL.md")


class TestKandinsky6Skill:
    """The Kandinsky 6.0 skill's numbers come from the pipeline that enforces
    them; the schedule, which the vendor states and no library constant
    holds, is the one the templates run."""

    def _call_defaults(self):
        from diffusers.pipelines.kandinsky6.pipeline_kandinsky6_ti2va import (
            Kandinsky6TI2VAPipeline,
        )

        signature = inspect.signature(Kandinsky6TI2VAPipeline.__call__)
        return {
            name: parameter.default
            for name, parameter in signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }

    def test_the_pipeline_defaults_the_skill_warns_about_are_the_pipeline_s(self):
        defaults = self._call_defaults()
        text = skill_text(KANDINSKY6_SKILL)
        assert (defaults["height"], defaults["width"]) == (512, 768)
        assert defaults["guidance_scale"] == 5.0
        assert "512x768 at guidance 5.0" in text
        assert defaults["num_frames"] == 121 and defaults["frame_rate"] == 24.0
        assert "121 frames at 24 fps" in text

    def test_the_size_and_frame_rules_are_the_pipeline_s(self):
        from diffusers.models.transformers import transformer_kandinsky6
        from diffusers.pipelines.kandinsky6 import pipeline_kandinsky6_ti2va

        text = skill_text(KANDINSKY6_SKILL)
        source = inspect.getsource(pipeline_kandinsky6_ti2va)
        # 8x spatial VAE times the transformer's 2x patch, refused outright
        assert (
            "spatial_multiple = self.vae_scale_factor_spatial * max(self.transformer_patch_size[1:])"
            in source
        )
        assert "else 8" in source and "else (1, 2, 2)" in source
        assert "multiples of 16" in text
        # 4x temporal VAE, floored rather than refused
        assert (
            "num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1"
            in source
        )
        assert "`4k + 1`" in text and "floors" in text
        # 128 latent frames of rotary table at 4x is 509 frames
        rope = inspect.signature(transformer_kandinsky6.Kandinsky6RoPE3D.__init__)
        assert rope.parameters["max_pos"].default[0] == 128
        assert (128 - 1) * 4 + 1 == 509 and "509 frames" in text

    def test_the_audio_rate_is_the_audio_vae_s(self):
        from diffusers import MMAudioVAE

        signature = inspect.signature(MMAudioVAE.__init__)
        assert signature.parameters["sample_rate"].default == 44_100
        assert "44.1 kHz" in skill_text(KANDINSKY6_SKILL)

    def test_the_scheduler_refuses_custom_sigmas(self):
        from diffusers.schedulers import scheduling_piflow

        source = inspect.getsource(scheduling_piflow)
        assert (
            "if sigmas is not None or mu is not None or timesteps is not None:"
            in source
        )
        assert "refuses custom sigmas" in skill_text(KANDINSKY6_SKILL)

    def test_the_schedule_the_skill_states_is_the_one_the_templates_run(self):
        text = skill_text(KANDINSKY6_SKILL)
        assert "10 steps at `guidance_scale` 1.0" in text
        for name in ("text-to-video", "image-to-video"):
            path = os.path.join(
                REPO_ROOT, "workflows", "templates", "kandinsky6", name + ".json"
            )
            with open(path, encoding="utf-8") as f:
                spec = json.load(f)
            arguments = spec["steps"][0]["pipeline"]["arguments"]
            assert arguments["num_inference_steps"] == 10, name
            assert arguments["guidance_scale"] == 1.0, name

    def test_the_upscale_limits_are_the_template_s_and_the_pipeline_s(self):
        """The 24 GB fit rests on two template values the skill tells an
        agent not to raise; the scale is one the pipeline accepts."""
        from diffusers.pipelines.kandinsky6 import pipeline_kandinsky6_sr

        source = inspect.getsource(pipeline_kandinsky6_sr)
        assert "if resolution_scale not in (2, 2.25, 4):" in source
        path = os.path.join(
            REPO_ROOT,
            "workflows",
            "templates",
            "kandinsky6",
            "generate-and-upscale.json",
        )
        with open(path, encoding="utf-8") as f:
            upscale = json.load(f)["steps"][1]["pipeline"]
        assert upscale["arguments"]["resolution_scale"] == 2
        assert upscale["transformer"]["from_pretrained_arguments"]["tile_sizes"] == [
            [512, 512]
        ]
        text = skill_text(KANDINSKY6_SKILL)
        assert "tiles are held to 512x512 and the scale to 2" in text

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(KANDINSKY6_SKILL)
        assert text.index("get_server_info") < text.index("templates/kandinsky6/")


FLUX3_SKILL = os.path.join(PLUGIN_DIR, "skills", "flux3-klein", "SKILL.md")


def flux3_text():
    """The Klein skill with its line wraps collapsed, so a phrase the tests
    pin does not break when a paragraph is rewrapped."""
    return " ".join(skill_text(FLUX3_SKILL).split())


class TestFlux3KleinSkill:
    """The Klein skill's numbers come from the diffusers modular blocks that
    enforce them; the 10-reference cap is the vendor's training limit, which
    the pipeline does not enforce, and the skill says so."""

    def test_the_step_count_is_the_distilled_pipeline_s(self):
        # The distilled denoise step, not Flux3SetTimestepsStep (which
        # defaults to 50), holds the 16 the templates run.
        from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import (
            Flux3DistilledCoreDenoiseStep,
        )

        defaults = {p.name: p.default for p in Flux3DistilledCoreDenoiseStep().inputs}
        assert defaults["num_inference_steps"] == 16
        assert "16 steps" in flux3_text()

    def test_the_grid_is_the_vae_scale_times_the_2x2_packing(self):
        from diffusers.modular_pipelines.flux3 import before_denoise

        # The multiple is read off the loaded VAE, so only the source shows it.
        assert "vae_scale_factor * 2" in inspect.getsource(before_denoise)
        assert "multiples of 16" in flux3_text()

    def test_the_distilled_model_takes_no_guidance(self):
        assert "no guidance" in flux3_text()

    def test_the_reference_cap_is_stated_as_the_vendor_s(self):
        text = flux3_text()
        assert "10 references" in text and "<ref_image_0>" in text
        assert "does not enforce" in text

    def test_the_skill_defers_prompt_format_to_bfl(self):
        text = flux3_text()
        assert "https://docs.bfl.ml/guides/prompting_summary" in text
        assert "`flux3_klein_upsample`" in text
        assert "`flux3_klein_upsample_edit`" in text

    def test_the_skill_starts_with_the_server(self):
        text = flux3_text()
        assert text.index("get_server_info") < text.index("list_workflows")
        assert text.index("get_server_info") < text.index("templates/flux3/")


SCRIPT_SKILL = os.path.join(PLUGIN_DIR, "skills", "script-to-video", "SKILL.md")
SCRIPT_JOINS = os.path.join(os.path.dirname(SCRIPT_SKILL), "references", "joins.md")


class TestScriptToVideoJoins:
    """The cutaway and lead-in joins (#1093): composed from separate runs,
    with the order the lead-in needs stated, since nothing enforces it."""

    def test_the_shot_list_names_every_join_and_links_the_recipes(self):
        body = skill_body(SCRIPT_SKILL)
        for join in ("`continue`", "`cut`", "`cutaway`", "`lead_in`"):
            assert join in body
        assert "`ending_state` of its\n  **source**" in body
        assert "`references/joins.md`" in body

    def test_the_recipes_name_both_joins_and_render_the_target_first(self):
        text = open(SCRIPT_JOINS, encoding="utf-8").read()
        assert "## Cutaway" in text and "## Lead-in" in text
        assert "Render the target first" in text
        assert "Re-rolling C invalidates B" in text
        # the cutaway's return continues from A, the lead-in pins C's opening
        assert "`get_last_frame` of `output:<A>`" in text
        assert "`get_first_frame` of `output:<C>`" in text

    def test_the_frame_step_is_a_real_task_step(self, tmp_path):
        from dw.workflow import workflow_from_file

        workflow = json.loads(
            _fenced_block_after(
                open(SCRIPT_JOINS, encoding="utf-8").read(),
                "## Taking a frame off a finished shot",
            )
        )
        step = workflow["steps"][0]
        assert step["task"]["command"] == "get_last_frame"
        step["task"]["arguments"]["video"] = "variable:video"
        workflow["variables"] = {"video": "shot.mp4"}
        (tmp_path / "shot.mp4").write_bytes(b"")
        path = tmp_path / "shot-frame.json"
        path.write_text(json.dumps(workflow), encoding="utf-8")
        workflow_from_file(str(path), str(tmp_path)).validate()


SCRIPT_BEATS = os.path.join(
    os.path.dirname(SCRIPT_SKILL), "references", "beat-review.md"
)
# What SKILL.md keeps free under SKILL_SIZE_LIMIT after the beat review moved
# to a reference (#1125): an addition that eats it belongs in `references/`.
SCRIPT_SKILL_MARGIN = 1024


class TestScriptToVideoBeatReview:
    """Beat authoring and the beat review (#1122 stage A): a beat a frame can
    prove, judged blind on a majority of frames, one reason per rejection."""

    def test_the_shot_sheet_names_beat_and_must_not(self):
        body = skill_body(SCRIPT_SKILL)
        assert "`beat`" in body and "`must_not`" in body
        assert body.index("`ending_state`") < body.index("`beat`")
        assert body.index("`beat`") < body.index("## 3.")

    def test_run_and_judge_keeps_its_tools_and_points_to_the_review(self):
        body = skill_body(SCRIPT_SKILL)
        judge = body[body.index("## Run and judge") : body.index("## 6.")]
        assert "get_output_frames" in judge
        assert "`references/beat-review.md`" in judge
        assert "adds no new mechanism" in body

    def test_the_review_names_its_rules(self):
        text = open(SCRIPT_BEATS, encoding="utf-8").read()
        assert "**majority**" in text
        for reason in (
            "`beat_not_shown`",
            "`must_not_present`",
            "`identity_drift`",
            "`text_in_frame`",
            "`reference_leak`",
        ):
            assert reason in text
        for banned in (
            "Motion over time",
            "Intent",
            "Off-screen facts",
            "Emotion words",
        ):
            assert f"**{banned}**" in text
        assert "**3 evenly spaced**" in text and "**5 only when" in text
        assert text.index("Describe blind, before the verdict") < text.index(
            "A verdict per frame"
        )
        assert "three-attempt budget" in text
        assert "Do not render while an\nedit-list shot has no beat" in text
        assert "## Audit the auditor" in text and "workspace note" in text

    def test_series_episodes_points_to_the_review(self):
        body = skill_body(SERIES_SKILL)
        assert "`script-to-video/references/beat-review.md`" in body
        assert "beat_not_shown" not in body  # a pointer, not a restatement

    def test_script_to_video_keeps_headroom_under_the_cap(self):
        size = len(skill_body(SCRIPT_SKILL).encode("utf-8"))
        assert size <= SKILL_SIZE_LIMIT - SCRIPT_SKILL_MARGIN, (
            f"{SCRIPT_SKILL} is {size} bytes, within {SCRIPT_SKILL_MARGIN} of "
            f"the {SKILL_SIZE_LIMIT}-byte cap: move detail to references/"
        )


SERIES_SKILL = os.path.join(PLUGIN_DIR, "skills", "series-episodes", "SKILL.md")
SERIES_LOOK = os.path.join(os.path.dirname(SERIES_SKILL), "references", "look.md")


class TestSeriesEpisodesLook:
    """The optional look step: a shared palette through apply_lut, then
    film_grain, as a workflow the engine's own validator accepts."""

    def _look_workflow(self):
        text = open(SERIES_LOOK, encoding="utf-8").read()
        return json.loads(
            _fenced_block_after(text, "## The workflow, one run per episode")
        )

    def _validate(self, workflow, tmp_path):
        from dw.workflow import workflow_from_file

        path = tmp_path / "series-look.json"
        path.write_text(json.dumps(workflow), encoding="utf-8")
        workflow_from_file(str(path), str(tmp_path)).validate()

    def test_the_look_is_optional_and_not_a_sixth_beat(self):
        body = skill_body(SERIES_SKILL)
        assert "five-beat" in body and "Optional look" in body
        assert "Not a beat" in body
        assert "`references/look.md`" in body

    def test_the_recipe_is_a_shared_palette_then_grain(self):
        steps = self._look_workflow()["steps"]
        assert [s["task"]["command"] for s in steps] == ["apply_lut", "film_grain"]
        assert steps[0]["task"]["arguments"]["palette"] == "variable:look"
        assert steps[1]["task"]["arguments"]["media"] == "previous_result:graded"

    def test_the_recipe_validates(self, tmp_path):
        self._validate(self._look_workflow(), tmp_path)

    def test_the_validation_is_real(self, tmp_path):
        workflow = self._look_workflow()
        workflow["variables"]["look"] = ["#12345", "#ffffff"]
        with pytest.raises(Exception, match="#12345"):
            self._validate(workflow, tmp_path)


CINE_DIR = os.path.join(PLUGIN_DIR, "skills", "cinematography")
CINE_SKILL = os.path.join(CINE_DIR, "SKILL.md")
CINE_REFS = os.path.join(CINE_DIR, "references")

# The closed vocabulary. Later stages (the shot document, the storyboard
# skill, the A/B probes) cite these ids verbatim, so a rename fails here.
CINE_AXES = {
    "shot_size": [
        "extreme-wide",
        "wide",
        "full",
        "medium",
        "medium-close",
        "close-up",
        "extreme-close-up",
    ],
    "move": [
        "static",
        "dolly-in",
        "dolly-out",
        "truck-left",
        "truck-right",
        "crane-up",
        "crane-down",
        "orbit",
        "tracking",
        "whip-pan",
        "push-in",
        "handheld",
        "crash-zoom",
    ],
    "lens": ["wide", "normal", "long", "anamorphic", "vintage"],
    "light": ["high-key", "low-key", "golden-hour", "practical", "noir"],
    "era": ["8mm", "16mm", "35mm", "vhs-dv", "digital"],
    "tempo": ["still", "measured", "brisk", "frantic"],
}
CINE_GRADES = ["noir", "teal-orange", "70s-stock", "vhs", "bleach-bypass"]
CINE_COMPILES = {
    "minimax-h3": "compile-h3.md",
    "ltx-2.5": "compile-ltx-2.5.md",
    "kandinsky-6": "compile-kandinsky-6.md",
    "flux3-klein": "compile-flux3-klein.md",
}


def _table_ids(text, section):
    """The backticked ids in the first column of a markdown table, read
    from `section` (a heading) to the next heading of any level."""
    start = text.index(section)
    end = text.find("\n#", start + len(section))
    block = text[start : end if end != -1 else len(text)]
    return re.findall(r"^\| `([^`]+)` \|", block, re.M)


class TestCinematographySkill:
    """A closed director vocabulary compiled to one sentence per family's
    caption format (#1142). The words are hints the models may ignore, so
    the skill says so; the grades are real apply_lut palettes and real
    film_grain arguments."""

    def _ref(self, name):
        return open(os.path.join(CINE_REFS, name), encoding="utf-8").read()

    def _grades(self):
        return json.loads(_fenced_block_after(self._ref("grades.md"), "## The grades"))

    def test_cinematography_skill_is_registered(self):
        for document in (
            os.path.join(PLUGIN_DIR, "README.md"),
            os.path.join(REPO_ROOT, "CLAUDE.md"),
        ):
            assert "`cinematography`" in open(document, encoding="utf-8").read()
        assert CINE_SKILL in SKILLS

    def test_cinematography_names_no_camera_brands(self):
        """Higgsfield retreated from camera-body presets; a brand only works
        if the model learned it, and nobody has measured that."""
        for path in glob.glob(os.path.join(CINE_DIR, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            text = open(path, encoding="utf-8").read()
            for brand in ("ARRI", "RED ", "Cooke", "Venice", "Alexa"):
                assert brand not in text, f"{brand!r} in {path}"

    def test_cinematography_vocabulary_defines_every_axis(self):
        text = self._ref("vocabulary.md")
        for axis, ids in CINE_AXES.items():
            assert _table_ids(text, f"## `{axis}`") == ids, axis

    def _move_rows(self):
        """{id: [class, means, note]} for the vocabulary's `move` table."""
        text = self._ref("vocabulary.md")
        start = text.index("## `move`")
        end = text.find("\n#", start + len("## `move`"))
        rows = {}
        for line in text[start:end].splitlines():
            if not line.startswith("| `"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows[cells[0].strip("`")] = cells[1:]
        return rows

    def test_cinematography_every_move_has_a_static_or_moving_class(self):
        """#1255: the long-form playbook picks an H3 `shot` preset from this
        column; only `static` is static (push-in and handheld are moving)."""
        rows = self._move_rows()
        assert list(rows) == CINE_AXES["move"]
        classes = {move: cells[0] for move, cells in rows.items()}
        assert set(classes.values()) == {"`static`", "`moving`"}
        assert [m for m, c in classes.items() if c == "`static`"] == ["static"]
        assert classes["push-in"] == classes["handheld"] == "`moving`"
        text = _flat(self._ref("vocabulary.md"))
        assert "position and orientation don't change across the take" in text

    def test_cinematography_every_moving_id_carries_the_h3_last_frame_rule(self):
        for move, (cls, _means, note) in self._move_rows().items():
            if cls == "`moving`":
                assert "H3: never `last_frame` with `look_hold`/`carry_pick`" in note
                assert "measured on the arc only" in note, move
            else:
                assert "last_frame" not in note, move

    @pytest.mark.parametrize("family", sorted(CINE_COMPILES))
    def test_cinematography_vocabulary_covers_every_family(self, family):
        text = self._ref(CINE_COMPILES[family])
        assert _table_ids(text, "## `shot_size`") == CINE_AXES["shot_size"]
        assert _table_ids(text, "## `move`") == CINE_AXES["move"]
        assert f"`{family}`" in text

    def test_cinematography_says_presets_are_hints(self):
        body = " ".join(skill_body(CINE_SKILL).split())
        assert "a hint, not a promise" in body
        assert "get_output_frames" in body

    def test_cinematography_scopes_out_audio_families(self):
        body = skill_body(CINE_SKILL)
        section = body[body.index("## Not in scope") :]
        assert "`minimax-music3`" in section
        flat = " ".join(section.split())
        assert (
            "covers H3, LTX-2.5 and Kandinsky 6 video plus FLUX 3 Klein stills only"
            in flat
        )
        assert "do not borrow a table" in flat

    def test_cinematography_ltx_every_move_row_carries_a_viewpoint(self):
        text = self._ref("compile-ltx-2.5.md")
        section = text[text.index("## `move`") :]
        section = section[: section.index("Example:")]
        rows = [r for r in section.splitlines() if r.startswith("| `")]
        assert len(rows) == len(CINE_AXES["move"])
        for row in rows:
            assert "<viewpoint>" in row, row

    def test_cinematography_whip_pan_direction_is_the_shots(self):
        for name in CINE_COMPILES.values():
            for row in self._ref(name).splitlines():
                if row.startswith("| `whip-pan`"):
                    assert "to the right" not in row and "pans right" not in row

    def test_cinematography_compiles_to_each_family_s_own_spec(self):
        """Each table follows its family's caption rules, not a shared one."""
        ltx = " ".join(self._ref("compile-ltx-2.5.md").split())
        assert "remains static" in ltx and "viewpoint" in ltx
        assert "comes into view" in ltx
        k6 = " ".join(self._ref("compile-kandinsky-6.md").split())
        assert "exactly one camera sentence" in k6
        assert "The camera is static." in k6
        h3 = " ".join(self._ref("compile-h3.md").split())
        assert "integrated_multimodal_description" in h3
        assert "amplitude" in h3 and "speed" in h3
        klein = " ".join(self._ref("compile-flux3-klein.md").split())
        assert "still" in klein and "no camera move" in klein

    def test_grades_are_valid_palettes(self):
        from dw.task_problems import check_lut_source

        grades = self._grades()
        assert sorted(grades) == sorted(CINE_GRADES)
        for name, grade in grades.items():
            check_lut_source(palette=grade["palette"])
            assert 0 < grade["strength"] <= 1, name

    def test_grades_name_real_film_grain_arguments(self):
        from dw.introspection import list_tasks
        from dw.task_domains import check_arguments
        from dw.tasks.finish import film_grain

        list_tasks()  # registers the command domains
        accepted = set(inspect.signature(film_grain).parameters) - {"media", "seed"}
        for name, grade in self._grades().items():
            grain = grade["film_grain"]
            assert set(grain) <= accepted, name
            check_arguments("film_grain", **grain)

    def test_the_grade_recipe_validates(self, tmp_path):
        from dw.workflow import workflow_from_file

        workflow = json.loads(
            _fenced_block_after(self._ref("grades.md"), "## Applying one")
        )
        steps = workflow["steps"]
        assert [s["task"]["command"] for s in steps] == ["apply_lut", "film_grain"]
        path = tmp_path / "grade.json"
        path.write_text(json.dumps(workflow), encoding="utf-8")
        workflow_from_file(str(path), str(tmp_path)).validate()

    @pytest.mark.parametrize("family", sorted(CINE_COMPILES))
    def test_family_skills_point_at_cinematography(self, family):
        path = os.path.join(PLUGIN_DIR, "skills", family, "SKILL.md")
        assert "`cinematography`" in skill_body(path)


MUSIC_SKILL = os.path.join(PLUGIN_DIR, "skills", "minimax-music3", "SKILL.md")


class TestMiniMaxMusic3Skill:
    """The Music 3 skill's numbers come from the diffusers modular pipeline
    that enforces them, and the caption format is deferred to MiniMax."""

    def test_the_ceiling_and_the_caps_are_the_encoder_s(self):
        from diffusers.modular_pipelines.minimax_music3 import encoders

        text = skill_text(MUSIC_SKILL)
        source = inspect.getsource(encoders)
        assert encoders._MAX_AUDIO_FRAMES == 9000 and "9000 frames" in text
        assert encoders._MAX_PROMPT_TOKENS == 5000 and "5,000 tokens" in text
        # the ceiling semantics: the language model may stop earlier
        assert re.search(r'"audio_duration",\s*default=60.0', source)
        flat = " ".join(text.split())
        assert "pipeline's default is 60 seconds" in flat
        # each template's own ceiling, as the skill states it
        for template, stated in (
            ("music", "120 s (`music`)"),
            ("music-video", "30 s (`music-video`)"),
        ):
            with open(
                os.path.join(
                    REPO_ROOT, "workflows", "templates", "minimax", f"{template}.json"
                ),
                encoding="utf-8",
            ) as handle:
                default = json.load(handle)["variables"]["audio_duration"]
            assert stated.startswith(f"{default} s") and stated in flat, template
        with open(
            os.path.join(
                REPO_ROOT, "workflows", "templates", "minimax", "music-long.json"
            ),
            encoding="utf-8",
        ) as handle:
            sections = json.load(handle)["variables"]["sections"]
        assert {section["audio_duration"] for section in sections} == {45}
        assert "45 s a section (`music-long`)" in flat
        assert "ceiling, not a target" in text
        # a tag line keeps only its leading bracketed tags, lower-cased
        assert "_LEADING_TAGS_RE" in source and hasattr(encoders, "_normalize_lyrics")
        assert "dropped" in text and "lower-cased" in text

    def test_the_frame_rate_and_the_output_rate_are_the_pipeline_s(self):
        from diffusers.modular_pipelines.minimax_music3 import modular_pipeline
        from diffusers.models.autoencoders import minimax_music3_vocoder

        text = skill_text(MUSIC_SKILL)
        assert "frame_rate = 25.0" in inspect.getsource(modular_pipeline)
        assert "25 frames per second" in text and "360 seconds" in text
        assert "sampling_rate: int = 44100" in inspect.getsource(minimax_music3_vocoder)
        assert "44.1 kHz stereo" in text and "sample_rate: 44100" in text

    def test_the_window_steps_and_guidance_are_the_denoiser_s(self):
        from diffusers.modular_pipelines.minimax_music3 import before_denoise, denoise

        text = skill_text(MUSIC_SKILL)
        assert before_denoise._CHUNK_FRAMES == 200 and "200-frame windows" in text
        source = inspect.getsource(denoise)
        assert re.search(r'"num_inference_steps",\s*default=30', source)
        assert "30 steps" in text
        assert '"guidance_scale": 1.7' in source and "fixed at 1.7" in text

    def test_the_skill_defers_caption_format_to_minimax(self):
        text = skill_text(MUSIC_SKILL)
        assert "music-caption-rewriter" in text
        assert (
            "npx skills add MiniMax-AI/MiniMax-Music3 --skill music-caption-rewriter"
            in text
        )
        assert "https://github.com/MiniMax-AI/MiniMax-Music3" in text
        assert "https://huggingface.co/MiniMaxAI/MiniMax-Music3" in text
        for tag in ("[Intro]", "[Pre-Chorus]", "[Bridge]", "[Instrumental]", "[Outro]"):
            assert tag in text, tag
        assert "2026-09-08-minimax-music3-audit.md" in text

    def test_the_music_video_ceiling_clears_its_slices(self):
        """The sliced total is 496 frames at 24 fps; the ceiling must clear it
        with margin, since the model may stop early (audit 2026-09-08)."""
        import json

        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", "music-video.json"
        )
        spec = json.load(open(path, encoding="utf-8"))
        assert spec["variables"]["audio_duration"] >= 496 / 24 + 5


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_states_the_subfolder_convention(path):
    """The templates put the deliverable in `final` and the scratch in
    `intermediate`; an agent reading get_job needs to know that, and one
    composing a new workflow needs to keep it."""
    text = skill_text(path)
    assert "`subfolder`" in text, f"{path} does not name the subfolder field"
    assert "`final`" in text and "`intermediate`" in text, (
        f"{path} does not state the final/intermediate convention"
    )
    # the convention is stated where the manifest is read
    assert text.index("`subfolder`") > text.index("## Run and judge")


@pytest.mark.parametrize("path", [H3_SKILL, LTX_SKILL], ids=["minimax-h3", "ltx-2.5"])
def test_a_video_skill_names_the_tools_that_see_and_hear_a_clip(path):
    """Both video skills said "no tool returns a frame from one" and told the
    agent to ask the user to look; `get_output_frames` and `get_output_audio`
    exist now (#193), and the skill has to send the agent there while still
    handing the user the gallery url."""
    text = skill_text(path)
    assert "no tool returns a frame" not in text
    assert "`get_output_frames(count=12)`" in text
    assert "`get_output_audio`" in text
    assert "gallery `url`" in text
    judge = text[text.index("## Run and judge") :]
    assert "get_output_frames" in judge


def test_the_h3_skill_names_each_cut_templates_final_step():
    """The skill names the one `final` step of each cut template - episode,
    music_video, voyage; if a template's roles change the skill must change
    with it."""
    text = skill_text(H3_SKILL)
    for name in ("dialogue-short", "music-video", "storyboard"):
        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", name + ".json"
        )
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        finals = [
            step["name"]
            for step in spec["steps"]
            if (step.get("result") or {}).get("subfolder") == "final"
        ]
        assert len(finals) == 1, (name, finals)
        assert f"`{finals[0]}`" in text, (
            f"the skill does not name {name}'s final step {finals[0]}"
        )


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_points_at_the_stored_exemplars(path):
    """Every family's hardest input is its prompt format, and the library
    already holds captions written to it. A skill that names the vendor's
    spec and not the worked example leaves an agent inventing prose it
    could have read - and `prompts/ltx2/` as a directory is unreachable
    from a client that has the MCP and no checkout."""
    text = skill_text(path)
    assert "`list_prompts" in text or "`get_prompt`" in text, (
        f"{path} never names the prompt library; a filesystem path is not a "
        f"call an MCP client can make"
    )


@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_is_enumerated_where_the_plugin_describes_itself(path):
    """A skill nobody lists is a capability nobody installs for. The four
    documents that enumerate them drifted the moment a fourth skill shipped,
    and the size and catalog tests glob the directory, so nothing noticed."""
    name = os.path.basename(os.path.dirname(path))
    for document in (
        os.path.join(PLUGIN_DIR, "README.md"),
        os.path.join(REPO_ROOT, "CLAUDE.md"),
    ):
        with open(document) as file:
            content = file.read()
            assert f"`{name}`" in content, (
                f"skill {name!r} is not named in {os.path.relpath(document, REPO_ROOT)}"
            )


@pytest.mark.parametrize("path", [H3_SKILL, LTX_SKILL], ids=["minimax-h3", "ltx-2.5"])
def test_a_video_skill_maps_its_failure_modes_to_the_tool_that_shows_them(path):
    """Naming the tools and naming the failure modes in separate sentences
    leaves the agent to guess which shows which; the step has to say
    `seams=true` is for a join and `at` is for a moment."""
    text = skill_text(path)
    judge = text[text.index("## Run and judge") :]
    assert "`seams=true`" in judge
    assert "`at`" in judge


STORY_FORGE_FINDINGS = "docs/proposals/audits/2026-11-story-forge-lessons.md"


class TestStoryForgeLessons:
    """Story-forge's production lessons (#1123): only what held on dw's
    pipeline lands as a rule. The probes ruled lessons 1-4 and 6 not
    reproduced, so no skill states them; lesson 5 lands unmeasured, worded
    as guidance and citing the findings doc."""

    def test_the_findings_doc_rules_what_landed(self):
        text = open(
            os.path.join(REPO_ROOT, STORY_FORGE_FINDINGS), encoding="utf-8"
        ).read()
        assert "## Lesson 5" in text and "(not probed)" in text
        assert "## What landed" in text

    @pytest.mark.parametrize(
        "path", [SCRIPT_SKILL, SERIES_SKILL], ids=["script-to-video", "series-episodes"]
    )
    def test_lesson_5_is_framing_guidance_citing_the_findings(self, path):
        body = " ".join(skill_body(path).split())
        assert "not a measured rule" in body
        assert "frame a recurring character large" in body
        assert "keep a wide master too" in body
        assert STORY_FORGE_FINDINGS in body

    def test_lesson_5_sits_in_the_casting_steps(self):
        script = skill_body(SCRIPT_SKILL)
        step3 = script[script.index("## 3.") : script.index("## 4.")]
        assert "frame a recurring character large" in " ".join(step3.split())
        series = skill_body(SERIES_SKILL)
        step0 = series[series.index("## 0.") : series.index("## 1-5.")]
        assert "frame a recurring character large" in " ".join(step0.split())

    def test_no_rule_lands_for_a_lesson_not_reproduced(self):
        ltx_dir = os.path.dirname(LTX_SKILL)
        assert not os.path.exists(os.path.join(ltx_dir, "references", "staging.md"))
        for path in (LTX_SKILL, H3_SKILL, FLUX3_SKILL, SCRIPT_SKILL, SERIES_SKILL):
            text = skill_text(path).lower()
            assert "locked-off on a tripod" not in text  # L2's remedy wording
            assert "all-caps" not in text and "all caps" not in text  # L4
            assert "just before contact" not in text  # L3


STB_REFS = os.path.join(os.path.dirname(SCRIPT_SKILL), "references")
SHOT_DOCUMENT = os.path.join(STB_REFS, "shot-document.md")
FIX_LADDER = os.path.join(STB_REFS, "fix-ladder.md")
H3_LINT = os.path.join(
    PLUGIN_DIR, "skills", "minimax-h3", "references", "prompt-lint.md"
)
SHOT_KEYS = (
    "id cast location opening_state ending_state beat must_not join "
    "shot_size move lens light duration"
).split()


def _read(path):
    return open(path, encoding="utf-8").read()


def _registered_mcp_tools():
    import asyncio

    import httpx

    pytest.importorskip("mcp", reason="the mcp extra is not installed")
    from dw_mcp.client import DwClient
    from dw_mcp.server import build_server

    client = DwClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    server = build_server(client)
    return {tool.name for tool in asyncio.run(server.list_tools())}


class TestShotDocumentAndFixLadder:
    """Stage B (#1143): the shot list persisted in the prompt library, the
    fix ladder with its caps, and the H3 prompt lint."""

    def test_shot_document_names_its_prompt_keys(self):
        text = _read(SHOT_DOCUMENT)
        assert "<project>/shots" in text and "<project>/shot-NN" in text
        for key in SHOT_KEYS:
            assert f"`{key}`" in text, key

    def test_shot_document_points_at_the_vocabulary_not_a_copy(self):
        text = _read(SHOT_DOCUMENT)
        assert "cinematography/references/vocabulary.md" in text
        assert "extreme-close-up" not in text  # ids live in one place

    def test_shot_document_uses_only_existing_tools(self):
        tools = _registered_mcp_tools()
        for path in (SHOT_DOCUMENT, FIX_LADDER, H3_LINT):
            for name in re.findall(r"`([a-z][a-z_]+)\(", _read(path)):
                assert name in tools, f"{os.path.basename(path)}: {name}"

    def test_shot_document_templates_resolve(self):
        for path in (SHOT_DOCUMENT, FIX_LADDER, H3_LINT):
            for name in CATALOG_NAME.findall(_read(path)):
                assert os.path.exists(
                    os.path.join(REPO_ROOT, "workflows", name + ".json")
                ) or os.path.exists(os.path.join(REPO_ROOT, "workflows", name)), name

    def test_fix_ladder_order(self):
        text = _read(FIX_LADDER).lower()
        at = [
            text.index(marker)
            for marker in ("regional-edit", "face-repair", "re-prompt", "reseed", "ask")
        ]
        assert at == sorted(at)

    def test_fix_ladder_has_caps(self):
        text = " ".join(_read(FIX_LADDER).lower().split())
        assert "three failed attempts" in text and "gpu-minute" in text

    def test_h3_lint_checks(self):
        text = _read(H3_LINT)
        for check in ("overall_soundscape", "overfull", "no new information"):
            assert check in text

    def test_credits(self):
        text = _read(SHOT_DOCUMENT) + _read(FIX_LADDER) + _read(H3_LINT)
        for name in ("seriesmith", "OpenReel", "open-higgsfield-skills", "Picky"):
            assert name in text
        assert "GPL" in text

    def test_script_to_video_points_at_the_three_references(self):
        body = skill_body(SCRIPT_SKILL)
        for ref in ("references/shot-document.md", "references/fix-ladder.md"):
            assert f"`{ref}`" in body
        assert "minimax-h3/references/prompt-lint.md" in body
        for key in ("shot_size", "move", "lens", "light", "duration"):
            assert f"`{key}`" in body


STORY_DIR = os.path.join(PLUGIN_DIR, "skills", "storyboard")
STORY_SKILL = os.path.join(STORY_DIR, "SKILL.md")


class TestStoryboardSkill:
    """Boards on FLUX 3 Klein, a six-cut coverage set, boards as start
    frames, and an animatic of them (#1145, #1146)."""

    def _ref(self, name):
        return open(
            os.path.join(STORY_DIR, "references", name), encoding="utf-8"
        ).read()

    def test_storyboard_skill_registered(self):
        for document in (
            os.path.join(PLUGIN_DIR, "README.md"),
            os.path.join(REPO_ROOT, "CLAUDE.md"),
        ):
            assert "`storyboard`" in open(document, encoding="utf-8").read()
        assert STORY_SKILL in SKILLS

    def test_storyboard_names_six_coverage_cuts(self):
        text = self._ref("coverage-set.md").lower()
        for cut in (
            "establishing",
            "medium",
            "close-up",
            "over-the-shoulder",
            "insert",
            "low angle",
        ):
            assert cut in text, cut

    def test_storyboard_caps_boards_at_eight(self):
        flat = " ".join(skill_body(STORY_SKILL).split())
        assert "up to eight boards" in flat
        assert "image one" in flat

    def test_storyboard_credits_its_sources(self):
        text = skill_text(STORY_SKILL)
        assert "Higgsfield" in text and "Popcorn" in text
        assert "Shot Board" in text and "MIT" in text

    def test_storyboard_animatic_recipe_names_the_template(self):
        text = self._ref("animatic.md")
        assert "`templates/animatic`" in text
        assert "`templates/generate-speech`" in text
        assert "`templates/minimax/music`" in text

    def test_storyboard_maps_boards_to_each_family_s_start_frame(self):
        text = skill_text(STORY_SKILL)
        for template in (
            "templates/minimax/image-to-video",
            "templates/ltx2/image-to-video",
            "templates/kandinsky6/image-to-video",
        ):
            assert f"`{template}`" in text

    @pytest.mark.parametrize("template", ["animatic", "face-drift-check"])
    def test_new_templates_validate(self, template, tmp_path):
        from dw.workflow import workflow_from_file

        path = os.path.join(REPO_ROOT, "workflows", "templates", f"{template}.json")
        workflow_from_file(path, str(tmp_path)).validate()

    def test_face_drift_check_is_pointed_at_from_the_series_skills(self):
        for skill in ("script-to-video", "series-episodes"):
            body = skill_body(os.path.join(PLUGIN_DIR, "skills", skill, "SKILL.md"))
            assert "`templates/face-drift-check`" in body


SITCOM = os.path.join(os.path.dirname(SERIES_SKILL), "references", "sitcom-multicam.md")


class TestSitcomMulticam:
    """Stage F (#1147): the multicam grammar as a series-episodes reference."""

    def test_sitcom_reference_names_its_tools(self):
        from dw.introspection import list_tasks

        text = _read(SITCOM)
        tasks = list_tasks()
        for name in ("transcribe_audio", "plan_cuts", "analyze_face_drift"):
            assert f"`{name}`" in text, name
            assert name in tasks["commands"], name
        assert "`templates/minimax/dialogue-short`" in text
        assert "`templates/face-drift-check`" in text
        assert "`templates/assemble-and-score`" in text
        for name in CATALOG_NAME.findall(text):
            assert os.path.isfile(
                os.path.join(REPO_ROOT, "workflows", name + ".json")
            ), name
        assert "One-shot Multicam" in text and "MIT" in text
        assert "cinematography/references/vocabulary.md" in text
        assert "`templates/attribute-lines`" in text and "`attribute_voices`" in text
        assert '`timestamps` set to `"segment"`' in text
        assert '`timestamps` set to `"word"`' not in text
        for field in ("start_frame", "cut_frames", "num_frames", "modulus: 17"):
            assert field in text, field
        assert "script-to-video/references/shot-document.md" in text

    def test_series_points_at_sitcom(self):
        assert "`references/sitcom-multicam.md`" in skill_body(SERIES_SKILL)


def _fenced_json(text):
    """Every ```json block in `text`, parsed."""
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", text, re.S)]


CINEMATOGRAPHY_KEYS = ("shot_size", "move", "lens", "light", "duration")
SCENE_KEYS = ("id", "shots", "length_seconds", "continuity")


def _is_scene_list(block):
    """`<project>/scenes` (#1199 Task 25) is the shot document's other array."""
    return all(isinstance(item, dict) and "length_seconds" in item for item in block)


class TestShotDocumentShape:
    """Final review of #1141: every shot carries the five cinematography keys,
    with null for the model's choice, a worked example, and the project name
    pattern the prompt library accepts."""

    def test_example_shot_and_two_shot_array_carry_every_key(self):
        blocks = _fenced_json(_read(SHOT_DOCUMENT))
        objects = [block for block in blocks if isinstance(block, dict)]
        arrays = [
            block
            for block in blocks
            if isinstance(block, list) and not _is_scene_list(block)
        ]
        assert objects and any(len(array) == 2 for array in arrays)
        for shot in objects + [shot for array in arrays for shot in array]:
            for key in SHOT_KEYS:
                assert key in shot, (shot.get("id"), key)
        nulls = [
            key
            for shot in objects + [s for a in arrays for s in a]
            for key in CINEMATOGRAPHY_KEYS
            if shot[key] is None
        ]
        assert nulls, "the example must show an axis left to the model as null"

    def test_null_and_seconds_are_stated(self):
        flat = " ".join(_read(SHOT_DOCUMENT).split())
        assert "all five cinematography keys" in flat
        assert "`null`" in flat and "model's own choice" in flat
        assert "`duration` is a number of seconds, not a vocabulary id" in flat
        body = " ".join(skill_body(SCRIPT_SKILL).split())
        assert "`null`" in body and "`duration` (seconds)" in body
        assert "ids from `cinematography`, never invented) and `duration`" in body

    def test_project_name_pattern_matches_the_prompt_library(self):
        from dw.security import PROMPT_REFERENCE_PATTERN

        segment = PROMPT_REFERENCE_PATTERN[1:].split("(")[0]
        assert segment == r"[\w][\w.-]*"
        text = _read(SHOT_DOCUMENT)
        assert f"`{segment}`" in text
        assert "no spaces and no further `/`" in " ".join(text.split())
        pattern = re.compile(PROMPT_REFERENCE_PATTERN)
        assert pattern.match("gilded_echoes/shots")
        assert not pattern.match("gilded echoes/shots")
        assert not pattern.match("series/ep1/shots")


class TestStoryboardFinalReview:
    """Final review of #1141: names keep_output accepts, one duration per
    board in the animatic, and one plugin description."""

    def test_no_storyboard_keep_output_name_ends_in_png(self):
        names = [
            (path, name)
            for path in glob.glob(os.path.join(STORY_DIR, "**", "*.md"), recursive=True)
            for name in re.findall(r'asset_name="([^"]*)"', _read(path))
        ]
        assert len(names) >= 2
        for path, name in names:
            assert not name.endswith(".png"), (path, name)
        assert "asset:boards/<scene>-NN.jpg" in skill_body(STORY_SKILL)

    def test_animatic_takes_one_float_duration_per_board(self, tmp_path):
        from dw.variables import argument_errors

        path = os.path.join(REPO_ROOT, "workflows", "templates", "animatic.json")
        with open(path, encoding="utf-8") as handle:
            definition = json.load(handle)
        variables = definition["variables"]
        assert variables["durations"] == [2.0] * len(variables["boards"])
        assert isinstance(variables["fps"], float)
        assert argument_errors(definition, {"durations": [1, 2, 0.5]}) == []
        assert argument_errors(definition, {"fps": 30}) == []
        assert "one duration per board" in definition["description"]
        assert "one number holds every board" not in definition["description"]

    def test_animatic_docs_say_one_duration_per_board(self):
        recipe = " ".join(
            open(os.path.join(STORY_DIR, "references", "animatic.md"), encoding="utf-8")
            .read()
            .split()
        )
        assert "one duration per board" in recipe
        assert "one number for every board" not in recipe
        assert "one duration per board" in " ".join(skill_body(STORY_SKILL).split())
        assert "`width`" not in skill_body(STORY_SKILL)

    def test_plugin_and_marketplace_describe_the_plugin_alike(self):
        with open(
            os.path.join(PLUGIN_DIR, ".claude-plugin", "plugin.json"), encoding="utf-8"
        ) as handle:
            plugin = json.load(handle)
        with open(
            os.path.join(REPO_ROOT, ".claude-plugin", "marketplace.json"),
            encoding="utf-8",
        ) as handle:
            marketplace = json.load(handle)
        entry = next(p for p in marketplace["plugins"] if p["name"] == "dw")
        assert plugin["description"] == entry["description"]
        assert "storyboard" in plugin["description"]


class TestSitcomAssembly:
    """Final review of #1141: the master's dialogue is the score with the
    shots' own sound muted, and master spans are cut with trim_video."""

    def test_master_dialogue_is_the_score_with_world_gain_zero(self):
        text = " ".join(_read(SITCOM).split())
        assert "Pass the master as `score`" in text
        assert "`world_gain: 0`" in text
        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "assemble-and-score.json"
        )
        with open(path, encoding="utf-8") as handle:
            variables = json.load(handle)["variables"]
        for name in ("score", "world_gain", "score_start_frame", "total_frames"):
            assert name in variables, name
            assert f"`{name}" in text, name

    def test_master_spans_are_cut_with_trim_video(self):
        text = _read(SITCOM)
        steps = [b for b in _fenced_json(text) if isinstance(b, dict)]
        trims = [s for s in steps if s.get("task", {}).get("command") == "trim_video"]
        assert trims
        tasks_doc = _read(os.path.join(REPO_ROOT, "docs", "TASKS.md"))
        section = tasks_doc[tasks_doc.index("### trim_video") :]
        section = section[: section.index("\n### ", 4)]
        for name in trims[0]["task"]["arguments"]:
            assert f"| `{name}` |" in section, name
        flat = " ".join(text.split())
        assert "planned `start_frame` and `cut_frames`" in flat


LONG_FORM_DIR = os.path.join(PLUGIN_DIR, "skills", "long-form")
LONG_FORM_SKILL = os.path.join(LONG_FORM_DIR, "SKILL.md")
CHAINED_SEGMENTS = os.path.join(
    REPO_ROOT, "workflows", "templates", "minimax", "chained-segments.json"
)


def test_long_form_skill_is_named_in_the_readme_and_claude_md():
    """The long-form playbook (#1199 Task 24) is a cross-cutting skill like
    storyboard: it ships, and the two documents that enumerate skills name it."""
    assert LONG_FORM_SKILL in SKILLS
    for document in (
        os.path.join(PLUGIN_DIR, "README.md"),
        os.path.join(REPO_ROOT, "CLAUDE.md"),
    ):
        assert "`long-form`" in open(document, encoding="utf-8").read(), document


class TestLongFormSkill:
    """The playbook's settled rules are the template's and the bench's; the
    open questions stay marked until a record settles them."""

    def _text(self):
        return " ".join(skill_text(LONG_FORM_SKILL).split())

    def test_the_shot_presets_are_the_template_s(self):
        with open(CHAINED_SEGMENTS, encoding="utf-8") as handle:
            spec = json.load(handle)
        presets = spec["presets"]["shot"]
        assert presets["static"] == {
            "continuity": "guide",
            "anchor": "segment_1",
            "anchor_strength": 0.25,
        }
        assert presets["static_long"] == {
            "continuity": "guide",
            "keystone": "segment_1",
            "keystone_slot": "last",
        }
        assert presets["moving"] == {"continuity": "guide"}
        assert presets["dialogue"] == {"continuity": "guide"}
        assert spec["variables"]["continuity"] == "guide"
        text = self._text()
        for shot in ("static", "static_long", "moving", "dialogue"):
            assert f"`shot: {shot}`" in text, shot
        assert "`anchor_strength` 0.25" in text

    def test_the_cost_row_is_the_template_s(self):
        with open(CHAINED_SEGMENTS, encoding="utf-8") as handle:
            spec = json.load(handle)
        (row,) = spec["cost"]
        assert f"{row['minutes']} min" in self._text()

    def test_the_check_ceilings_are_the_engine_s(self):
        from dw.pipeline_processors import chain_check

        text = self._text()
        assert '`"check": {"cut_ratio_max": null}`' in text
        assert "warn-only" in text
        for name, ceiling in (
            ("jump_ratio", chain_check.JUMP_RATIO_MAX),
            ("look_ratio", chain_check.LOOK_RATIO_MAX),
            ("cut_ratio", chain_check.CUT_RATIO_MAX),
            ("texture_ratio", chain_check.TEXTURE_RATIO_MAX),
        ):
            assert f"`{name}` (ceiling {ceiling:g})" in text, name
        # The audio floor re-rolls (#1221), so it is quoted as a floor, not
        # a warn-only ceiling
        assert f"`audio_dbfs` (floor {chain_check.AUDIO_DBFS_MIN:g} dBFS)" in text
        assert '`"audio_dbfs_min": null`' in text

    def test_the_audio_re_roll_is_not_promised_to_fix_silence(self):
        # Plan v2 of #1221: on the S5 repro both seeds read about -89 dBFS,
        # so the skill and the guide say a re-roll may not bring sound back
        guide = " ".join(
            open(os.path.join(REPO_ROOT, "docs", "WORKFLOW_GUIDE.md"), encoding="utf-8")
            .read()
            .split()
        )
        for text in (self._text(), guide):
            assert "not guaranteed to bring sound back" in text
            assert "`chain_segment_silent`" in text

    def test_the_quoted_worst_case_is_the_estimate_s(self):
        # Both branches of cost_scaling._worst_case_retries (#1221)
        text = self._text()
        assert "N(1 + `retries`)" in text
        assert "1 + (N - 1)(1 + `retries`)" in text

    def test_every_rule_cites_the_bench(self):
        text = self._text()
        for record in (
            "counters.md",
            "check.md",
            "repair.md",
            "structural.md",
            "noise-floor.md",
            "determinism.md",
        ):
            assert f"bench/records/{record}" in text, record

    def test_the_open_questions_are_marked_pending(self):
        text = self._text()
        assert "pending measurement" in text
        for topic in ("windowed refine", "latent carry", "capstone", "tier", "ltx"):
            assert topic in text.lower(), topic

    def test_the_measured_levers_carry_their_verdicts(self):
        """The structural levers, the keystone and LTX parity were measured
        (bench/records/structural.md, repair.md, ltx.md): each names its
        verdict rather than reading as pending."""
        text = " ".join(self._text().split())
        for verdict in (
            "not a burn fix",  # windowed refine
            "not a default",  # latent carry
            "does not win",  # late capstone
            "Don ruled slot `last` the static long-take default",  # keystone
            "stays the `templates/ltx2/chained-segments` default",
            "Cap an LTX-2.5 chain at about 15-20 s",
        ):
            assert verdict in text, verdict
        assert "experimental_keystone_slot" not in text

    def test_the_move_class_comes_from_the_vocabulary(self):
        """#1255: the "until #1200" stand-in is replaced by a pointer to the
        cinematography vocabulary's `class` column, in both files."""
        text = self._text()
        assert "#1200" not in text and "until #1200" not in text
        assert "`class` column of `cinematography/references/vocabulary.md`" in text
        assert "`push-in` and `handheld` included, is `shot: moving`" in text
        levers = _flat(_read(LONG_FORM_LEVERS))
        assert "#1200" not in levers
        assert "`class` column of `cinematography/references/vocabulary.md`" in levers

    def test_the_cut_ideas_are_named_so_they_are_not_re_proposed(self):
        text = self._text()
        for idea in ("keystone", "backoff", "sharpest", "guide `look_hold`"):
            assert idea in text, idea


LONG_FORM_LEVERS = os.path.join(LONG_FORM_DIR, "references", "levers.md")
H3_CHAINS = os.path.join(os.path.dirname(H3_SKILL), "references", "chains.md")
STORYBOARD_SKILL = os.path.join(PLUGIN_DIR, "skills", "storyboard", "SKILL.md")
MUSIC3_SKILL = os.path.join(PLUGIN_DIR, "skills", "minimax-music3", "SKILL.md")


def _flat(text):
    return " ".join(text.split())


class TestLongFormPointers:
    """#1199 Task 25: the planning skills carry a scene's length and
    continuity, and the family skills hand anything longer than one clip to
    the `long-form` playbook instead of restating it."""

    def test_the_shot_document_plans_length_and_continuity_per_scene(self):
        text = _flat(_read(SHOT_DOCUMENT))
        assert "<project>/scenes" in text
        assert "`length_seconds`" in text and "`continuity`" in text
        for shape in ("`one_clip`", "`cuts`", "`chain`", "`bridge`"):
            assert shape in text, shape
        assert "`long-form`" in text
        scenes = [
            block
            for block in _fenced_json(_read(SHOT_DOCUMENT))
            if isinstance(block, list) and _is_scene_list(block)
        ]
        assert scenes, "the shot document shows a worked `<project>/scenes`"
        shapes = {"one_clip", "cuts", "chain", "bridge"}
        for scene in scenes[0]:
            assert set(scene) == set(SCENE_KEYS), scene
            assert isinstance(scene["length_seconds"], (int, float))
            assert scene["continuity"] in shapes, scene

    @pytest.mark.parametrize(
        "path", [H3_SKILL, LTX_SKILL], ids=["minimax-h3", "ltx-2.5"]
    )
    def test_a_video_family_hands_longer_than_one_clip_to_long_form(self, path):
        body = _flat(skill_body(path))
        assert "**Longer than one clip" in body
        assert "the `long-form` skill first" in body

    @pytest.mark.parametrize(
        "path",
        [STORYBOARD_SKILL, SCRIPT_SKILL, SERIES_SKILL],
        ids=["storyboard", "script-to-video", "series-episodes"],
    )
    def test_a_planning_skill_points_a_long_scene_to_long_form(self, path):
        assert "`long-form`" in skill_body(path)

    def test_music_over_one_track_points_to_the_long_form_audio_recipes(self):
        body = _flat(skill_body(MUSIC3_SKILL))
        assert "`long-form/references/audio.md`" in body

    def test_the_community_drift_notes_moved_into_the_playbook(self):
        chains = _read(H3_CHAINS)
        assert "NerdNotes" not in chains
        assert "`long-form`" in chains
        levers = _flat(_read(LONG_FORM_LEVERS))
        assert "NerdNotes" in levers
        for claim in (
            "half-static",
            "segments 3-6",
            "masked area",
            "Whole-frame averages",
            "#1033",
        ):
            assert claim in levers, claim

    @pytest.mark.parametrize(
        "path",
        [
            H3_SKILL,
            LTX_SKILL,
            STORYBOARD_SKILL,
            SCRIPT_SKILL,
            SERIES_SKILL,
            MUSIC3_SKILL,
        ],
        ids=lambda p: os.path.basename(os.path.dirname(p)),
    )
    def test_every_touched_skill_holds_its_budget(self, path):
        size = len(skill_body(path).encode("utf-8"))
        assert size <= SKILL_SIZE_LIMIT, f"{path} is {size} bytes"
