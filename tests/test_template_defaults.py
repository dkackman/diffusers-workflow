"""A catalog default must run: no placeholder hosts, and one spelling per repo.

upscale-* defaulted to https://example.com/image.jpg (not an image), and
flux-torchao spelled FLUX.1-dev two ways - the Hub cache is keyed on the
spelling, so the transformer downloaded twice (~23 GB)."""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted([*ROOT.glob("workflows/**/*.json"), *ROOT.glob("dw/workflows/*.json")])
PLACEHOLDER = re.compile(r"https?://(www\.)?example\.(com|org|net)/")
WITHDRAWN = ("runwayml/",)


def strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings(value)
    elif isinstance(node, str):
        yield node


def test_no_default_points_at_a_placeholder_host():
    offenders = [
        f"{path.relative_to(ROOT)}: {s}"
        for path in FILES
        for s in strings(json.loads(path.read_text()))
        if PLACEHOLDER.search(s)
    ]
    assert offenders == []


def test_no_withdrawn_repos():
    offenders = [
        f"{path.relative_to(ROOT)}: {s}"
        for path in FILES
        for s in strings(json.loads(path.read_text()))
        if s.startswith(WITHDRAWN)
    ]
    assert offenders == []


def test_one_spelling_per_repo_within_a_file():
    for path in FILES:
        repos = {
            s
            for s in strings(json.loads(path.read_text()))
            if re.fullmatch(r"[\w.-]+/[\w.-]+", s)
        }
        lowered = {}
        for repo in repos:
            lowered.setdefault(repo.lower(), set()).add(repo)
        clashes = [names for names in lowered.values() if len(names) > 1]
        assert clashes == [], f"{path.relative_to(ROOT)}: {clashes}"
