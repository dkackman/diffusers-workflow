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


def skill_text(path):
    return open(path, encoding="utf-8").read()


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
    text = skill_text(path)
    fields = frontmatter(text)

    assert fields["name"] == os.path.basename(os.path.dirname(path))
    assert "description" in fields and len(fields["description"]) > 40
    assert len(text.encode("utf-8")) <= SKILL_SIZE_LIMIT, (
        f"{path} is over {SKILL_SIZE_LIMIT} bytes"
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
        assert 124 == 17 * 7 + 5 and 345 == 17 * 20 + 5
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

        from diffusers.modular_pipelines.minimax_h3 import (
            before_encoder,
            modular_pipeline,
        )
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
        assert before_encoder is not None

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
        assert "alpha 128" in text

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
        assert variables["lora_alpha"] == 128
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
            ("music-video", {"name", "prompt", "start_frame"}),
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
                f"skill {name!r} is not named in "
                f"{os.path.relpath(document, REPO_ROOT)}"
            )
