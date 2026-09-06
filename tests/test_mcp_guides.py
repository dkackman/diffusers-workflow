"""The guides an agent reads before choosing a capability.

These are packaged documentation rather than server state, so unlike the rest
of the MCP surface they are read locally and there is no route to pin. What
matters instead: the index is complete enough to route a request by, a guide
can be fetched a section at a time rather than whole, and a wrong name says
what the right ones are.
"""

import pytest

from dw_mcp import guides
from dw_mcp.client import DwApiError


class TestListing:
    def test_every_curated_guide_is_listed(self):
        listed = {guide["name"] for guide in guides.list_guides()["guides"]}

        assert listed == set(guides.GUIDES)

    def test_each_guide_carries_a_summary(self):
        # The summary is what an agent matches a request against before
        # spending context on the guide itself
        for guide in guides.list_guides()["guides"]:
            assert guide["summary"].strip()

    def test_each_guide_lists_its_sections(self):
        # The index doubles as a routing table: section headings are what a
        # request shape is matched to, so they have to be in the listing
        listed = guides.list_guides()["guides"]
        tasks = next(g for g in listed if g["name"] == "tasks")

        assert "Speech Generation" in tasks["sections"]

    def test_every_curated_guide_file_exists(self):
        # Guards the set against a doc being renamed or moved out from under it
        for name in guides.GUIDES:
            assert guides.read_guide(name).strip()


class TestWhereTheyComeFrom:
    def test_a_checkout_is_preferred_over_a_stale_packaged_copy(self, tmp_path, monkeypatch):
        """build_dist.sh leaves dw/docs/ behind (gitignored). If that copy won,
        every later edit to docs/ would be invisible to the MCP and to these
        tests - so the repo's docs/ wins whenever it is there, and the packaged
        copy is only for an install, which has no docs/ beside the package."""
        root = tmp_path / "checkout"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "docs").mkdir()
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        (root / "docs" / "TASKS.md").write_text("## Checkout\n")
        monkeypatch.setattr(guides, "__file__", str(root / "dw_mcp" / "guides.py"))

        assert guides.read_guide("tasks") == "## Checkout\n"

    def test_an_install_reads_the_packaged_copy(self, tmp_path, monkeypatch):
        root = tmp_path / "site-packages"
        (root / "dw" / "docs").mkdir(parents=True)
        (root / "dw" / "docs" / "TASKS.md").write_text("## Packaged\n")
        monkeypatch.setattr(guides, "__file__", str(root / "dw_mcp" / "guides.py"))

        assert guides.read_guide("tasks") == "## Packaged\n"

    def test_a_checkout_with_no_packaged_copy_reads_the_repo_docs(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        (root / "docs").mkdir(parents=True)
        (root / "docs" / "TASKS.md").write_text("## Checkout\n")
        monkeypatch.setattr(guides, "__file__", str(root / "dw_mcp" / "guides.py"))

        assert guides.read_guide("tasks") == "## Checkout\n"

    def test_a_guide_missing_from_the_install_says_so(self, tmp_path, monkeypatch):
        monkeypatch.setattr(guides, "__file__", str(tmp_path / "dw_mcp" / "guides.py"))

        with pytest.raises(DwApiError, match="missing from this install"):
            guides.read_guide("tasks")


class TestFetching:
    def test_a_guide_comes_back_whole_by_default(self):
        guide = guides.get_guide("tasks")

        assert guide["name"] == "tasks"
        assert guide["section"] is None
        assert "## Speech Generation" in guide["content"]

    def test_one_section_comes_back_alone(self):
        # A whole guide is thousands of lines; handing all of it over is how an
        # agent ends up reading none of it carefully
        guide = guides.get_guide("tasks", section="Speech Generation")

        assert guide["section"] == "Speech Generation"
        assert "generate_speech" in guide["content"]
        assert "## Frame Interpolation" not in guide["content"]

    def test_a_section_keeps_its_own_heading(self):
        guide = guides.get_guide("tasks", section="Speech Generation")

        assert guide["content"].lstrip().startswith("## Speech Generation")

    def test_a_section_name_need_not_match_case_or_spacing(self):
        # An agent reproduces a heading from the listing loosely - "speech-generation"
        # for "Speech Generation" - and being strict about it just costs a round trip
        guide = guides.get_guide("tasks", section="speech-generation")

        assert guide["section"] == "Speech Generation"


class TestErrors:
    def test_an_unknown_guide_names_the_ones_that_exist(self):
        with pytest.raises(DwApiError, match="tasks"):
            guides.get_guide("nonexistent")

    def test_an_unknown_section_names_the_sections_that_exist(self):
        with pytest.raises(DwApiError, match="Speech Generation"):
            guides.get_guide("tasks", section="nonexistent")
