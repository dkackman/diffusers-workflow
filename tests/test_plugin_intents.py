"""A skill's intents index: one row per result someone wants, pointing at the
pattern that writes it and the record that settled it.

`plugins/dw/skills/<skill>/references/intents.md` holds a five-column table,
`intent | template | pattern | evidence | status`. The index is only worth
reading while its pointers resolve, so this pins them: the status is one of
three, a confirmed or refuted row cites a record in the repo, every cited
record exists, and the pattern's file and heading exist. Every
`bench/records/*.md` is cited by some index's row, or sits in `EXEMPT` with
the reason it serves no family's index: a record that lands owes its row in
the same change (`bench/README.md`, *Records*).
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

BENCH_RECORDS = sorted(glob.glob(os.path.join(REPO_ROOT, "bench", "records", "*.md")))

# A bench record no index row cites, and why. Each serves neither an H3 nor an
# LTX-2.5 result, or is the bench's own infrastructure.
EXEMPT = {
    "h3-image.md": "MiniMax H3-Image stills: a family with no intents index",
    "longcat-avatar.md": "LongCat avatar: a family with no intents index",
    "face-encode-floor.md": "infrastructure: the face-score encode floor every face margin must clear",
    "noise-floor.md": "infrastructure: the bench's seed and rerun floors that verdicts are read against",
    "determinism.md": "infrastructure: the server's cuDNN and TF32 settings for reproducible reruns",
    "for-don.md": "a digest of Don's open checks and answers; each result is cited at its own record",
    "h3-hybrid.md": "an engine experiment with no template; its look verdict is Don's and pending (#1441)",
    "h3-audio-encode-context.md": "an engine-question probe with no template; its verdict (bar not met) settles #1516 stage A's gate (#1544)",
    "held-frames.md": "a corpus probe with no template; its verdict settles #1527 stage A's gate (#1551)",
    "sol-h3-vae.md": "an engine-question probe with no template; its verdict settles #1574's build-or-not (#1583)",
    "h3-dmad.md": "a LoRA probe stopped at its load check (0 tensors mapped); it settles no intent row, only that DMAD isn't adopted (#1584)",
}


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


CITED = {
    os.path.basename(record)
    for _path, _number, cells in ROWS
    for record in RECORD.findall(cells[3])
    if record.startswith("bench/records/")
}


@pytest.mark.parametrize("skill", ["minimax-h3", "ltx-2.5"])
def test_an_index_exists(skill):
    path = os.path.join(SKILLS_DIR, skill, "references", "intents.md")
    assert path in INDEXES, f"{skill} has no references/intents.md"
    assert any(row[0] == path for row in ROWS), f"the {skill} intents index has no rows"


@pytest.mark.parametrize(
    "record", BENCH_RECORDS, ids=[os.path.basename(r) for r in BENCH_RECORDS]
)
def test_every_bench_record_is_cited(record):
    name = os.path.basename(record)
    if name in EXEMPT:
        return
    assert name in CITED, (
        f"bench/records/{name} is cited by no intents.md row: add the row its "
        "verdict settles to a family's references/intents.md, or exempt it in "
        "EXEMPT with a one-line reason"
    )


@pytest.mark.parametrize("name,reason", sorted(EXEMPT.items()))
def test_exempt_record_exists_with_a_reason(name, reason):
    assert reason.strip(), f"EXEMPT entry {name} has no reason"
    assert os.path.isfile(os.path.join(REPO_ROOT, "bench", "records", name)), (
        f"EXEMPT names bench/records/{name}, which does not exist"
    )
    assert name not in CITED, (
        f"bench/records/{name} is cited by an intents.md row; drop it from EXEMPT"
    )


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
