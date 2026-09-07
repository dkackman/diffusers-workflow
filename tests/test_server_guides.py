"""The guides the engine serves to an agent that has to pick a capability.

They live on the server rather than in the MCP client's install so the guides
an agent reads are the guides for the engine it is about to drive: an MCP at
one version against a server at another would otherwise index sections the
server does not have. What matters: the index is complete enough to route a
request by, a guide can be fetched a section at a time, and a wrong name says
what the right ones are.
"""

import pytest
from fastapi.testclient import TestClient

from dw.server import guides
from dw.server.app import create_app
from dw.server.guides import GuideError
from dw.server.jobs import JobManager
from tests.test_server import ScriptedWorkerManager, success_script

TASKS_TEXT = (
    "# Tasks\n\n## Speech Generation\n\ngenerate_speech\n\n"
    "## Frame Interpolation\n\ninterpolate_frames\n"
)


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A checkout-shaped tree holding two small guides, with the module
    resolving files against it and the table trimmed to those two."""
    root = tmp_path / "checkout"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "TASKS.md").write_text(TASKS_TEXT)
    (root / "docs" / "WORKFLOW_GUIDE.md").write_text("## Structure\n\nsteps\n")
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
    def test_a_guide_comes_back_whole_by_default(self, checkout):
        guide = guides.get_guide("tasks")

        assert guide == {"name": "tasks", "section": None, "content": TASKS_TEXT}

    def test_one_section_comes_back_alone_with_its_heading(self, checkout):
        guide = guides.get_guide("tasks", section="Speech Generation")

        assert guide["section"] == "Speech Generation"
        assert guide["content"] == "## Speech Generation\n\ngenerate_speech\n"

    def test_the_last_section_runs_to_the_end_of_the_file(self, checkout):
        guide = guides.get_guide("tasks", section="Frame Interpolation")

        assert guide["content"] == "## Frame Interpolation\n\ninterpolate_frames\n"

    def test_a_section_name_need_not_match_case_or_spacing(self, checkout):
        # An agent reproduces a heading from the listing loosely -
        # "speech-generation" for "Speech Generation" - and being strict
        # about it just costs a round trip
        guide = guides.get_guide("tasks", section="speech-generation")

        assert guide["section"] == "Speech Generation"


class TestErrors:
    def test_an_unknown_guide_names_the_ones_that_exist(self, checkout):
        with pytest.raises(GuideError, match="tasks, workflows"):
            guides.get_guide("nonexistent")

    def test_an_unknown_section_names_the_sections_that_exist(self, checkout):
        with pytest.raises(GuideError, match="Speech Generation, Frame Interpolation"):
            guides.get_guide("tasks", section="nonexistent")


class TestTheRealDocs:
    """Over the repo's own docs/, so a renamed file or heading fails here."""

    def test_every_curated_guide_file_exists(self):
        for name in guides.GUIDES:
            assert guides.read_guide(name).strip()

    def test_the_tasks_guide_indexes_speech_generation(self):
        tasks = next(g for g in guides.list_guides()["guides"] if g["name"] == "tasks")

        assert "Speech Generation" in tasks["sections"]

    def test_the_authoring_section_is_reachable_by_name(self):
        guide = guides.get_guide(
            "workflows", section="authoring-a-workflow-from-an-agent"
        )

        assert guide["section"] == "Authoring a workflow from an agent"

    def test_the_authoring_section_names_every_reference_prefix(self):
        """The prefixes the engine reserves are the ones the section has to
        explain; a new prefix added to the engine fails here until it is
        written up."""
        from dw.prompts import RESERVED_TEXT_PREFIXES

        content = guides.get_guide(
            "workflows", section="Authoring a workflow from an agent"
        )["content"]

        for prefix in RESERVED_TEXT_PREFIXES:
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

    def test_a_whole_guide(self, client):
        body = client.get("/api/guides/tasks").json()

        assert body == {"name": "tasks", "section": None, "content": TASKS_TEXT}

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
