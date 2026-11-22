"""A skill's intents index: one row per result someone wants, pointing at the
pattern that writes it and the record that settled it.

`plugins/dw/skills/<skill>/references/intents.md` holds a five-column table,
`intent | template | pattern | evidence | status`. The index is only worth
reading while its pointers resolve, so this pins them: the status is one of
three, a confirmed or refuted row cites a record in the repo, every cited
record exists, and the pattern's file and heading exist.
"""

import glob
import os
import re

import pytest

from tests.test_examples import REPO_ROOT

SKILLS_DIR = os.path.join(REPO_ROOT, "plugins", "dw", "skills")
INDEXES = sorted(glob.glob(os.path.join(SKILLS_DIR, "*", "references", "intents.md")))

COLUMNS = ["intent", "template", "pattern", "evidence", "status"]
STATUSES = {"confirmed", "refuted", "untested"}
RECORD = re.compile(r"`((?:bench/records|docs/proposals)/[^`]+)`")
PATTERN = re.compile(r"^`([^`]+\.md)`\s*§\s*(\S.*)$")
TEMPLATE = re.compile(r"^`((?:templates|models)/[A-Za-z0-9_./-]+)`$")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_rows(path):
    """(line number, cells) for each body row of the file's table."""
    lines = open(path, encoding="utf-8").read().splitlines()
    rows, header_seen = [], False
    for number, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            continue
        cells = _cells(line)
        if not header_seen:
            assert cells == COLUMNS, (
                f"{path}:{number}: header is {cells}, expected {COLUMNS}"
            )
            header_seen = True
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append((number, cells))
    assert header_seen, f"{path}: no intents table"
    return rows


def headings(path):
    with open(path, encoding="utf-8") as handle:
        return {match.group(1) for match in map(HEADING.match, handle) if match}


def row_name(path, number, cells):
    skill = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return f"{skill} intents.md:{number} ({cells[0][:60]!r})"


ROWS = [(path, number, cells) for path in INDEXES for number, cells in table_rows(path)]
ROW_IDS = [row_name(*row) for row in ROWS]


def test_an_index_exists():
    assert os.path.join(SKILLS_DIR, "minimax-h3", "references", "intents.md") in INDEXES
    assert ROWS, "the minimax-h3 intents index has no rows"


@pytest.mark.parametrize("path,number,cells", ROWS, ids=ROW_IDS)
def test_row_shape_and_status(path, number, cells):
    name = row_name(path, number, cells)
    assert len(cells) == len(COLUMNS), (
        f"{name}: {len(cells)} columns, expected {len(COLUMNS)}"
    )
    assert cells[0], f"{name}: no intent"
    assert cells[4] in STATUSES, (
        f"{name}: status {cells[4]!r} is not one of {sorted(STATUSES)}"
    )
    template = cells[1]
    if template and template != "any":
        match = TEMPLATE.match(template)
        assert match, (
            f"{name}: template {template!r} is not one backticked catalog name"
        )
        target = os.path.join(REPO_ROOT, "workflows", match.group(1) + ".json")
        assert os.path.isfile(target), (
            f"{name}: template {match.group(1)} is not in workflows/"
        )


@pytest.mark.parametrize("path,number,cells", ROWS, ids=ROW_IDS)
def test_row_evidence_resolves(path, number, cells):
    name = row_name(path, number, cells)
    records = RECORD.findall(cells[3])
    if cells[4] in ("confirmed", "refuted"):
        assert records, (
            f"{name}: a {cells[4]} row cites a `bench/records/` or `docs/proposals/` path; "
            "an issue-only result is untested"
        )
    for record in records:
        assert os.path.isfile(os.path.join(REPO_ROOT, record)), (
            f"{name}: {record} does not exist"
        )


@pytest.mark.parametrize("path,number,cells", ROWS, ids=ROW_IDS)
def test_row_pattern_heading_exists(path, number, cells):
    name = row_name(path, number, cells)
    match = PATTERN.match(cells[2])
    assert match, f"{name}: pattern {cells[2]!r} is not `file.md` § heading"
    relative, heading = match.groups()
    skill_dir = os.path.dirname(os.path.dirname(path))
    candidates = [os.path.join(skill_dir, relative), os.path.join(SKILLS_DIR, relative)]
    found = next((c for c in candidates if os.path.isfile(c)), None)
    assert found, (
        f"{name}: pattern file {relative} is in neither the skill nor plugins/dw/skills/"
    )
    assert heading in headings(found), f"{name}: {relative} has no heading {heading!r}"
