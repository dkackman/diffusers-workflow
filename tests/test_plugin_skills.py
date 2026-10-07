"""The dw plugin: skills that teach an agent to compose a model family's
workflows, kept true by tests rather than by memory.

A skill quotes catalog names and states numeric rules. Each name has to
resolve to a file, each number has to come from the library that enforces
it, and the plugin's version has to be the engine's - an installed skill is
matched to the server it was written against by that number.
"""

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
    assert task in list_tasks()["commands"]


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
        for requested, evaluations in ((9, 8), (20, 19)):
            scheduler.set_timesteps(requested)
            assert len(scheduler.timesteps) == evaluations

        text = skill_text(H3_SKILL)
        assert "denoise_total_steps" in text and "reports 8" in text

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
        assert variables["num_inference_steps"] == 9
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
        assert "if sigmas is not None or mu is not None or timesteps is not None:" in source
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

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(KANDINSKY6_SKILL)
        assert text.index("get_server_info") < text.index("templates/kandinsky6/")


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
        assert "Default\n  60 seconds" in text or "Default 60 seconds" in text
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
