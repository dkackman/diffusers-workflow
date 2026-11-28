"""The guides the engine serves to an agent that has to pick a capability.

They live on the server rather than in the MCP client's install so the guides
an agent reads are the guides for the engine it is about to drive: an MCP at
one version against a server at another would otherwise index sections the
server does not have. What matters: the index is complete enough to route a
request by, a guide can be fetched a section at a time, and a wrong name says
what the right ones are.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dw.server import guides
from dw.server.app import create_app
from dw.server.guides import GuideError
from dw.server.jobs import JobManager
from tests.test_server import ScriptedWorkerManager, success_script

TASKS_TEXT = (
    "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n\n"
    "## Frame Interpolation\n\n"
    "interpolate_frames - see [it](../workflows/templates/interpolate-frames.json)\n"
)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout-shaped tree holding two small guides, with the module
    resolving files against it and the table trimmed to those two."""
    root = tmp_path / "checkout"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "TASKS.md").write_text(TASKS_TEXT)
    (root / "docs" / "WORKFLOW_GUIDE.md").write_text(
        "## Structure\n\nsteps - as in workflows/templates/text-to-image.json\n\n"
        "## Cross-Step Data Flow\n\n"
        "### One step per entry: for_each\n\n"
        "for_each runs a step once per entry.\n\n"
        "### previous_result\n\n"
        "previous_result chains steps.\n\n"
        "#### Steps in detail\n\n"
        "a minor heading under previous_result.\n\n"
        "## Comparing\n\n"
        "### Comparing outputs\n\n"
        "the grid.\n\n"
        "#### Texture bands\n\n"
        "bands by wavelength.\n\n"
        "### Remediation\n\n"
        "recut or regenerate.\n"
    )
    monkeypatch.setattr(guides, "__file__", str(root / "dw" / "server" / "guides.py"))
    monkeypatch.setattr(
        guides,
        "GUIDES",
        {"tasks": guides.GUIDES["tasks"], "workflows": guides.GUIDES["workflows"]},
    )
    return root


class TestListing:
    def test_every_curated_guide_is_listed(self, checkout):
        listed = {guide["name"] for guide in guides.list_guides()["guides"]}

        assert listed == {"tasks", "workflows"}

    def test_each_guide_carries_a_summary_and_its_sections(self, checkout):
        # The summary is what an agent matches a request against; the
        # section headings are the routing table
        tasks = next(g for g in guides.list_guides()["guides"] if g["name"] == "tasks")

        assert tasks["summary"].strip()
        assert tasks["file"] == "TASKS.md"
        assert tasks["sections"] == ["Speech Generation", "Frame Interpolation"]


class TestWhereTheyComeFrom:
    def test_a_checkout_is_preferred_over_a_stale_packaged_copy(
        self, tmp_path, monkeypatch
    ):
        """build_dist.sh leaves dw/docs/ behind (gitignored). If that copy won,
        every later edit to docs/ would be invisible to the server - so the
        repo's docs/ wins whenever it is there, and the packaged copy is only
        for an install, which has no docs/ beside the package."""
        root = tmp_path / "checkout"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        (root / "docs" / "TASKS.md").write_text("## Checkout\n")
        monkeypatch.setattr(
            guides, "__file__", str(root / "dw" / "server" / "guides.py")
        )

        assert guides.read_guide("tasks") == "## Checkout\n"

    def test_an_install_reads_the_packaged_copy(self, tmp_path, monkeypatch):
        root = tmp_path / "site-packages"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        monkeypatch.setattr(
            guides, "__file__", str(root / "dw" / "server" / "guides.py")
        )

        assert guides.read_guide("tasks") == "## Packaged\n"

    def test_a_guide_missing_from_the_install_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            guides, "__file__", str(tmp_path / "dw" / "server" / "guides.py")
        )

        with pytest.raises(FileNotFoundError, match="missing from this install"):
            guides.read_guide("tasks")


class TestFetching:
    def test_no_section_answers_the_index_not_the_whole_file(self, checkout):
        """A whole guide is ~19.6k tokens in one call - more than the entire
        tool surface costs to connect - and the call that spends it is the
        one an agent makes by accident (#101). The opening and the first
        section come back; the rest is named, not sent."""
        guide = guides.get_guide("tasks")

        assert guide["section"] is None
        assert (
            guide["content"] == "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n"
        )
        assert guide["sections"] == ["Speech Generation", "Frame Interpolation"]
        assert guide["withheld"] == ["Frame Interpolation"]
        assert "Frame Interpolation" in guide["sections"]
        assert "section=" in guide["note"]

    def test_a_guide_with_no_sections_comes_back_whole(self, checkout, tmp_path):
        """Nothing to index, so there is nothing to withhold."""
        (tmp_path / "checkout" / "docs" / "TASKS.md").write_text(
            "# Tasks\n\nall of it\n"
        )

        guide = guides.get_guide("tasks")

        assert guide["content"] == "# Tasks\n\nall of it\n"
        assert guide["withheld"] == [] and "note" not in guide

    def test_one_section_comes_back_alone_with_its_heading(self, checkout):
        guide = guides.get_guide("tasks", section="Speech Generation")

        assert guide["section"] == "Speech Generation"
        assert guide["content"] == "## Speech Generation\n\ngenerate_speech\n"

    def test_the_last_section_runs_to_the_end_of_the_file(self, checkout):
        guide = guides.get_guide("tasks", section="Frame Interpolation")

        assert guide["content"] == (
            "## Frame Interpolation\n\n"
            "interpolate_frames - see [it](../workflows/templates/interpolate-frames.json)\n"
        )

    def test_a_section_name_need_not_match_case_or_spacing(self, checkout):
        # An agent reproduces a heading from the listing loosely -
        # "speech-generation" for "Speech Generation" - and being strict
        # about it just costs a round trip
        guide = guides.get_guide("tasks", section="speech-generation")

        assert guide["section"] == "Speech Generation"

    def test_a_subsection_is_reachable_by_its_own_heading(self, checkout):
        # #503: a `###` subsection isn't in the top-level index, but a
        # caller who copied its exact heading still reaches it, not a
        # refusal that only lists `##` headings.
        guide = guides.get_guide("workflows", section="One step per entry: for_each")

        assert guide["section"] == "One step per entry: for_each"
        assert guide["parent_section"] == "Cross-Step Data Flow"
        assert "for_each runs a step once per entry." in guide["content"]
        # Just the subsection - not its sibling, not the whole ## section
        assert "previous_result chains steps." not in guide["content"]

    def test_a_subsection_is_reachable_by_a_term_inside_its_heading(self, checkout):
        # The concept a caller has in hand ("for_each") rather than the
        # full sentence the heading is written as.
        guide = guides.get_guide("workflows", section="for_each")

        assert guide["section"] == "One step per entry: for_each"
        assert guide["parent_section"] == "Cross-Step Data Flow"

    def test_a_section_match_wins_over_a_subsection_match(self, checkout):
        # An exact `##` match still takes priority - subsections are a
        # fallback for a term the top level does not otherwise resolve.
        guide = guides.get_guide("workflows", section="Cross-Step Data Flow")

        assert guide["section"] == "Cross-Step Data Flow"
        assert "parent_section" not in guide

    def test_a_minor_heading_is_reachable_by_name(self, checkout):
        # #1160: a `####` heading is named content a skill points at; it
        # runs to the next `###`/`####` and names its `##` section.
        guide = guides.get_guide("workflows", section="Texture bands")

        assert guide["section"] == "Texture bands"
        assert guide["parent_section"] == "Comparing"
        assert "bands by wavelength." in guide["content"]
        assert "recut or regenerate." not in guide["content"]
        assert "the grid." not in guide["content"]

    def test_a_subsection_still_carries_its_minor_headings(self, checkout):
        guide = guides.get_guide("workflows", section="Comparing outputs")

        assert guide["section"] == "Comparing outputs"
        assert "the grid." in guide["content"]
        assert "bands by wavelength." in guide["content"]
        assert "recut or regenerate." not in guide["content"]

    def test_a_subsection_match_wins_over_a_minor_heading_match(self, checkout):
        # The `####` pass runs only when no `###` matched, so a term a `###`
        # heading already resolved still lands there.
        assert (
            guides.get_guide("workflows", section="previous_result")["section"]
            == "previous_result"
        )
        assert (
            guides.get_guide("workflows", section="Steps in detail")["section"]
            == "Steps in detail"
        )

    def test_a_section_linking_an_example_says_how_to_reach_it(self, checkout):
        """A guide's link to an example is a repo path. The reader these
        guides exist for has no checkout - it has the MCP and nothing else -
        so a payload holding such a path carries the translation rule, rather
        than 114 links being rewritten in prose people read too."""
        body = guides.get_guide("tasks", "Frame Interpolation")

        # The note is one fixed sentence - it states the rule, it does not
        # name the paths it saw
        assert body["catalog_paths"] == guides.CATALOG_PATH_NOTE
        assert "get_workflow" in body["catalog_paths"]

    def test_a_section_linking_nothing_carries_no_note(self, checkout):
        body = guides.get_guide("tasks", "Speech Generation")

        assert "catalog_paths" not in body

    def test_an_index_holding_a_path_is_noted_too(self, checkout):
        """The index carries the guide's opening and its whole first section,
        which is as able to link an example as any other."""
        assert "catalog_paths" not in guides.get_guide("tasks")
        assert "catalog_paths" in guides.get_guide("workflows")


class TestErrors:
    def test_an_unknown_guide_names_the_ones_that_exist(self, checkout):
        with pytest.raises(GuideError, match="tasks, workflows"):
            guides.get_guide("nonexistent")

    def test_an_unknown_section_names_the_sections_that_exist(self, checkout):
        with pytest.raises(GuideError, match="Speech Generation, Frame Interpolation"):
            guides.get_guide("tasks", section="nonexistent")


class TestTheRealDocs:
    """Over the repo's own docs/, so a renamed file or heading fails here."""

    def test_install_is_a_served_guide(self):
        assert guides.GUIDES["install"][0] == "INSTALL.md"

    def test_references_names_the_subsection_that_lists_every_prefix(self):
        # #1721: "References" is contained in "Default preferences", which
        # comes first; the exact `### References` heading must win, and it
        # must list `preference:` beside the other prefixes, as the MCP
        # instructions promise.
        guide = guides.get_guide("workflows", section="References")

        assert guide["section"] == "References"
        for prefix in ("variable:", "previous_result:", "asset:", "preference:"):
            assert f"`{prefix}`" in guide["content"]

    def test_the_workflows_index_stays_small(self):
        # #1787: every session reads the index, so the first section must not
        # absorb maintainer material; the preference ladder is its own section.
        index = guides.get_guide("workflows")

        assert len(index["content"].encode()) < 3000
        assert "Default preferences" in index["withheld"]
        section = guides.get_guide("workflows", section="Default preferences")
        assert "`text_to_image`" in section["content"]
        assert "must be placed" in section["content"]

    def test_composing_points_at_the_preference_override(self):
        composing = guides.get_guide(
            "workflows", section="Composing a stored workflow"
        )["content"]
        assert "preference:<key>" in composing
        preferences = guides.get_guide("workflows", section="Model preferences")
        assert "Per-run override" in preferences["content"]

    def test_every_curated_guide_file_exists(self):
        for name in guides.GUIDES:
            assert guides.read_guide(name).strip()

    def test_a_served_guide_links_only_docs_that_are_served(self):
        # #1237: a link a served guide follows must land in a guide the MCP
        # serves, or the consumer holding only the MCP dead-ends.
        import re

        served = {meta[0] for meta in guides.GUIDES.values()}
        # every relative link to a markdown doc, wherever it sits (examples are
        # linked as repo paths and carry their own note)
        link = re.compile(r"(?<!!)\]\((?!https?:|mailto:|#)([^)#\s]+\.md)")
        unserved = {}
        for name in guides.GUIDES:
            for target in link.findall(guides.read_guide(name)):
                # credits for third-party models; not guidance for a consumer
                if target not in served and target != "ATTRIBUTIONS.md":
                    unserved.setdefault(name, set()).add(target)
        assert not unserved

    def test_the_server_guide_reaches_the_sections_other_guides_link(self):
        for section in ("Choosing the card", "Progress events"):
            assert guides.get_guide("server", section=section)["content"].strip()

    def test_the_tasks_guide_indexes_speech_generation(self):
        tasks = next(g for g in guides.list_guides()["guides"] if g["name"] == "tasks")

        assert "Speech Generation" in tasks["sections"]

    def test_the_assessment_section_states_the_procedure_and_authority(self):
        """#388: assess_output's guide section - procedure, findings shape,
        the authority rule and the remediation table - is reachable by name,
        and names every rule the table in dw/assessment_rules.py holds."""
        from dw.assessment_rules import RULES

        guide = guides.get_guide("workflows", section="Assessing a run's output")
        content = guide["content"]

        assert guide["section"] == "Assessing a run's output"
        for word in ("`assess_output`", "not a verdict", "recut", "regenerate"):
            assert word in content, word
        for field in ("rule", "severity", "at", "value", "threshold", "says"):
            assert field in content, field
        for rule in RULES:
            assert f"`{rule['name']}`" in content, rule["name"]

    def test_the_authoring_section_is_reachable_by_name(self):
        guide = guides.get_guide(
            "workflows", section="authoring-a-workflow-from-an-agent"
        )

        assert guide["section"] == "Authoring a workflow from an agent"

    def test_the_texture_bands_heading_is_reachable(self):
        """#1160: the minimax-h3 skill points at *Comparing outputs*,
        *Texture bands*; both names resolve, and the first carries the second."""
        bands = guides.get_guide("workflows", section="Texture bands")
        assert bands["section"] == "Texture bands"
        assert "texture_bands=true" in bands["content"]
        assert "not a score" in bands["content"]

        comparing = guides.get_guide("workflows", section="Comparing outputs")
        assert comparing["section"] == "Comparing outputs"
        assert "#### Texture bands" in comparing["content"]

    def test_the_for_each_subsection_is_reachable_without_its_parent(self):
        """#503: both terms the issue's repro used reach the `###` subsection
        directly, rather than only through its `##` parent."""
        for section in ("for_each", "One step per entry: for_each"):
            guide = guides.get_guide("workflows", section=section)

            assert "for_each" in guide["section"]
            assert guide["parent_section"] == "Authoring a workflow from an agent"
            assert "for_each" in guide["content"]

    def test_the_h3_promoting_workflow_is_reachable(self):
        """#499: the inline upscale workflow is a `###` subsection, so every
        term an agent holds - the heading, a word in it, either task name -
        reaches it, and it carries the workflow JSON."""
        for section in (
            "Promoting an H3 take to 768p in latent space",
            "Promoting",
            "upscale_h3_latents",
            "decode_h3_latents",
        ):
            guide = guides.get_guide("workflows", section=section)

            assert guide["section"].startswith("Promoting an H3 take")
            assert guide["parent_section"] == "Pipeline Configuration"
            assert '"command": "upscale_h3_latents"' in guide["content"]
            assert '"command": "pair_audio"' in guide["content"]

    def test_the_authoring_section_names_every_reference_prefix(self):
        """The prefixes the engine reserves are the ones the section has to
        explain; a new prefix added to the engine fails here until it is
        written up."""
        from dw.references import RESERVED_TEXT

        content = guides.get_guide(
            "workflows", section="Authoring a workflow from an agent"
        )["content"]

        for prefix in RESERVED_TEXT:
            assert f"`{prefix}`" in content, prefix

    def test_the_authoring_section_states_the_cartesian_rule_and_the_loop(self):
        content = guides.get_guide(
            "workflows", section="Authoring a workflow from an agent"
        )["content"]

        assert "cartesian" in content.lower()
        for tool in (
            "validate_workflow",
            "save_workflow",
            "run_workflow",
            "wait_for_job",
            "get_output_image",
        ):
            assert f"`{tool}`" in content, tool
        for shape in (
            "image",
            "image-set",
            "image-edit",
            "shot",
            "sequence",
            "audio",
            "text",
            "utility",
        ):
            assert f"`{shape}`" in content, shape

    def test_the_sweep_section_names_the_loop_and_only_real_tools(self):
        """#890: a sweep is the existing tools called once per variant, so
        the section has to name each one, and every tool-shaped name it
        tells an agent to call has to exist on the MCP server."""
        import asyncio
        import re

        from dw_mcp.client import DwClient
        from dw_mcp.server import build_server

        guide = guides.get_guide("workspaces", section="Running a sweep")
        content = guide["content"]

        assert guide["section"] == "Running a sweep"
        loop = (
            "create_workspace",
            "use_workspace",
            "set_note",
            "get_workflow",
            "save_workflow",
            "validate_workflow",
            "run_workflow",
            "wait_for_job",
            "list_jobs",
            "get_job",
            "get_gallery_metadata",
            "list_gallery",
            "keep_output",
        )
        for tool in loop:
            assert f"`{tool}" in content, tool
        for convention in (
            "plan.estimate",
            "reused",
            "latest run",
            "score ",
            "judged_by",
        ):
            assert convention in content, convention

        tools = {t.name for t in asyncio.run(build_server(DwClient()).list_tools())}
        named = set(re.findall(r"`([a-z]+(?:_[a-z]+)+)\b", content))
        not_tools = {"num_inference_steps", "judged_by"}
        assert named - not_tools <= tools, sorted(named - not_tools - tools)

    def test_the_tasks_guide_carries_the_catalog_path_note(self):
        """TASKS.md links 49 example workflows by repo path - the most of any
        served guide."""
        body = guides.get_guide("tasks", "Examples")

        assert "catalog_paths" in body


@pytest.fixture
def client(tmp_path, checkout):
    """An app over the two-guide checkout; nothing here queues a job."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir()
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(prompt_dir),
        job_manager=manager,
    )
    with TestClient(app, base_url="http://localhost") as c:
        yield c


class TestRoutes:
    def test_the_listing_is_the_index(self, client):
        body = client.get("/api/guides").json()

        names = [g["name"] for g in body["guides"]]
        assert names == ["tasks", "workflows"]
        tasks = body["guides"][0]
        assert tasks["sections"] == ["Speech Generation", "Frame Interpolation"]
        assert tasks["summary"].strip()

    def test_no_section_answers_the_index(self, client):
        body = client.get("/api/guides/tasks").json()

        assert body["section"] is None
        assert body["withheld"] == ["Frame Interpolation"]
        assert "Frame Interpolation" not in body["content"]

    def test_one_section_matched_loosely(self, client):
        body = client.get(
            "/api/guides/tasks", params={"section": "speech-generation"}
        ).json()

        assert body["section"] == "Speech Generation"
        assert body["content"] == "## Speech Generation\n\ngenerate_speech\n"

    def test_an_unknown_guide_is_a_404_naming_the_guides(self, client):
        response = client.get("/api/guides/nonexistent")

        assert response.status_code == 404
        assert "tasks, workflows" in response.json()["detail"]

    def test_an_unknown_section_is_a_404_naming_the_sections(self, client):
        response = client.get("/api/guides/tasks", params={"section": "nope"})

        assert response.status_code == 404
        assert "Speech Generation, Frame Interpolation" in response.json()["detail"]


class TestParenthesisedHeadingSuffix:
    """#1257: a heading's trailing "(`path`)" is optional in a section name."""

    @pytest.mark.parametrize(
        "guide, asked, heading",
        [
            ("security-quickref", "Trust gate", "Trust gate (`dw/trust.py`)"),
            ("security-quickref", "HTTP", "HTTP (`dw/server/http_security.py`)"),
        ],
    )
    def test_the_real_guide_resolves_without_the_suffix(self, guide, asked, heading):
        body = guides.get_guide(guide, section=asked)

        assert body["section"] == heading

    def test_the_full_heading_still_resolves(self):
        body = guides.get_guide(
            "security-quickref", section="Trust gate (`dw/trust.py`)"
        )

        assert body["section"] == "Trust gate (`dw/trust.py`)"


class TestAgentRules:
    """#1547: the hard rules a client without the dw plugin reads, served as a
    guide and named by the MCP instructions."""

    def test_it_is_listed_with_its_summary_and_sections(self):
        listed = next(
            g for g in guides.list_guides()["guides"] if g["name"] == "agent-rules"
        )
        assert listed["summary"] == (
            "Hard rules per model family, for pieces longer than one clip and "
            "for choosing a model when the user names none, for a client "
            "without the dw plugin."
        )
        assert listed["sections"] == [
            "FLUX 3 Klein",
            "MiniMax H3",
            "Longer than one clip",
            "Choosing a model",
        ]

    def test_a_section_matches_loosely(self):
        guide = guides.get_guide("agent-rules", section="minimax-h3")
        assert guide["section"] == "MiniMax H3"
        assert "`audio_shift: 3.0`" in guide["content"]

    def test_an_unknown_section_names_its_sections(self):
        with pytest.raises(
            GuideError, match="FLUX 3 Klein, MiniMax H3, Longer than one clip"
        ):
            guides.get_guide("agent-rules", section="ltx")

    def test_its_pointers_resolve(self):
        import re

        text = guides.read_guide("agent-rules")
        pointers = re.findall(r'get_guide\("([\w-]+)", section="([^"]+)"\)', text)
        assert pointers
        for name, section in pointers:
            assert guides.get_guide(name, section=section)["content"].strip()

    def test_the_mcp_instructions_name_it(self):
        from dw_mcp.server import INSTRUCTIONS

        assert 'get_guide("agent-rules")' in INSTRUCTIONS

    def test_the_mcp_guide_states_the_client_claim(self):
        content = " ".join(
            guides.get_guide("mcp", section="Clients")["content"].split()
        )
        assert "Claude Code with the `dw` plugin is the tested path" in content
        assert "no-plugin Claude Sonnet session" in content
        assert "Other MCP clients and models are untested" in content


def test_no_served_guide_puts_the_updates_on_the_models_page():
    """#1799: the dw and diffusers updates are on Server -> System. Matched over
    the whole text, since the mentions that survived a first pass wrapped."""
    docs = Path(__file__).resolve().parents[1] / "docs"
    for file_name, _ in guides.GUIDES.values():
        text = " ".join((docs / file_name).read_text().split())
        assert "Models page" not in text, file_name

    server = (docs / "SERVER.md").read_text()
    models = server.split("- **Models** (Server)")[1].split("- **Preferences**")[0]
    assert "Update from" not in models and "Update dw" not in models
