"""The prose guides, served by the engine they describe.

The gap they close is that a catalog entry describes what a workflow *is*,
and an open-ended request ("a lego movie trailer set in the marvel
universe") names nothing that appears in any of them. What has to be
matched is shape - multi-shot video, cuts, a consistent cast, narration -
and that is what the guides are written in. An agent that reads one before
choosing composes from what exists instead of authoring a fresh workflow
badly.

They are served from here, not read from the MCP client's install, because
the guides an agent reads have to be the guides for the engine it is about
to drive: an MCP at one version against a server at another would
otherwise index sections the server does not have.

Two things keep this cheap. The listing carries each guide's section
headings, so the index is itself the routing table and is small enough to
read every time. And a guide can be fetched one section at a time, because
handing over three thousand lines of TASKS.md is how an agent ends up
reading none of it carefully.
"""

import re
from pathlib import Path

# The docs that bear on choosing a capability. Deliberately not all of them -
# testing, releasing, security and dependency notes are for people working on
# dw, and listing them would dilute an index whose whole value is that it is
# short enough to read in full.
GUIDES = {
    "workflows": (
        "WORKFLOW_GUIDE.md",
        "How the catalog is organised - templates/ teaching a pattern, models/ "
        "recording what makes a checkpoint fit - and how a workflow is put "
        "together: steps, variables, the reference conventions that carry data "
        "between steps, sub-workflows, and releasing models mid-run. Read "
        "before authoring one.",
    ),
    "tasks": (
        "TASKS.md",
        "The utility task commands - image and video processing, audio, "
        "captioning, text and speech generation, gathering. Compose these "
        "before writing anything new.",
    ),
    "recipes": (
        "RECIPES_24GB.md",
        "What actually fits and runs on a 24 GB accelerator, model by model. "
        "Read before proposing something large.",
    ),
    "acceleration": (
        "ACCELERATION.md",
        "Offloading, attention slicing, caching and compilation - how to make "
        "a pipeline fit or run faster, and what each costs.",
    ),
    "quantization": (
        "QUANTIZATION.md",
        "Running a model in fewer bits: the supported frameworks and how a "
        "quantization config is written per component.",
    ),
    "loras": (
        "LORAS.md",
        "Loading, weighting and combining LoRA adapters.",
    ),
    "prompt-weighting": (
        "PROMPT_WEIGHTING.md",
        "Emphasis and de-emphasis syntax in prompts, and which pipelines honour it.",
    ),
    "ip-adapter": (
        "IP_ADAPTER.md",
        "Conditioning generation on a reference image with IP-Adapter.",
    ),
    "workspaces": (
        "WORKSPACES.md",
        "How a run's workflows, prompts, assets and outputs are rooted, and "
        "what changes when a workspace is named.",
    ),
}

# Only the top level. Sub-headings would triple the index for detail that is
# better reached by reading the section they sit in
SECTION_PATTERN = re.compile(r"^## (.+)$", re.M)

# A doc's link to an example is a repo-relative path, which resolves for a
# reader with a checkout and dead-ends for the one these guides are served
# to: an agent holding the MCP and nothing else. Said once, on a payload
# that contains such a path, rather than rewritten into 114 links in prose
# that people read too.
CATALOG_PATH_NOTE = (
    "Paths like `workflows/templates/minimax/image-to-video.json` in this "
    "text are repo-relative and are not served by this machine. The catalog "
    "name is the part after `workflows/` with `.json` dropped - "
    "`templates/minimax/image-to-video` - which is what `get_workflow`, "
    "`validate_workflow`, `run_workflow` and a sub-workflow step's `path` "
    "all take. Paths under `dw/workflows/` are the packaged built-ins a "
    "sub-workflow names as `builtin:<file>.json`."
)

# The same shape tests/test_docs_links.py checks the targets of: the
# lookbehind keeps `workflows/` as the start of the path, so `dw/workflows/`
# and `tests/test_data/workflows/` do not match.
_EXAMPLE_PATH = re.compile(r"(?<![\w/-])(?:\.\./)?workflows/[A-Za-z0-9_.\-/]+\.json")


class GuideError(LookupError):
    """A guide or section that does not exist. The message names what does,
    so the route can hand it straight back as a 404 detail."""


def _guide_file(file_name):
    """Where a guide lives: the repo's docs/ in a checkout, else the copy
    build_dist.sh puts under dw/docs/ for an install.

    The checkout wins because the build leaves dw/docs/ behind; if that copy
    took precedence, editing docs/ would change nothing an agent reads. The
    same rule default_ui_dir applies to the SPA.
    """
    root = Path(__file__).resolve().parent.parent.parent
    checkout = root / "docs" / file_name
    if checkout.is_file():
        return checkout
    return root / "dw" / "docs" / file_name


def read_guide(name):
    """One guide's full markdown text.

    Raises:
        GuideError: If the name is not a guide.
        FileNotFoundError: If the guide's file is missing from this install.
    """
    if name not in GUIDES:
        raise GuideError(
            f"No guide named '{name}'. The guides are: {', '.join(sorted(GUIDES))}."
        )
    path = _guide_file(GUIDES[name][0])
    if not path.is_file():
        raise FileNotFoundError(
            f"The '{name}' guide is missing from this install "
            f"({GUIDES[name][0]} was not found)."
        )
    return path.read_text(encoding="utf-8")


def _sections(text):
    """The top-level section headings of a guide, in the order they appear."""
    return SECTION_PATTERN.findall(text)


def _normalized(heading):
    """A heading reduced to what a loose match compares.

    An agent reproduces a heading from the listing approximately -
    "speech-generation" for "Speech Generation" - and refusing that costs a
    round trip to say something it already knew.
    """
    return re.sub(r"[^a-z0-9]+", "", heading.lower())


def _extract_section(text, section):
    """One section's text, heading included, or None if there is no such one."""
    wanted = _normalized(section)
    matches = list(SECTION_PATTERN.finditer(text))
    for index, match in enumerate(matches):
        if _normalized(match.group(1)) != wanted:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return match.group(1), text[match.start() : end].rstrip() + "\n"
    return None


def _noted(body):
    """The payload, with the repo-path rule attached when it holds one.

    Attached to the body rather than woven into the guides because the same
    markdown is read on GitHub and in an editor, where the links resolve.
    """
    if _EXAMPLE_PATH.search(body.get("content") or ""):
        body["catalog_paths"] = CATALOG_PATH_NOTE
    return body


def list_guides():
    """Every guide, with what it covers and the sections it holds."""
    listed = []
    for name, (file_name, summary) in GUIDES.items():
        listed.append(
            {
                "name": name,
                "file": file_name,
                "summary": summary,
                "sections": _sections(read_guide(name)),
            }
        )
    return {"guides": listed}


def get_guide(name, section=None):
    """One guide, or one section of it.

    Without a `section` the answer is the *index*: the guide's preamble,
    its first section, and the headings of the rest - not the whole file.
    A full WORKFLOW_GUIDE.md is ~19.6k tokens and TASKS.md ~16k, which is
    more in one call than the entire 58-tool MCP surface costs to connect,
    and an agent can make that call twice before noticing (#101). Every
    withheld section is named in `sections` and fetched by name, so
    nothing is unreachable - only unspent by accident.
    """
    text = read_guide(name)
    if section is None:
        return _noted(_index(name, text))

    found = _extract_section(text, section)
    if found is None:
        raise GuideError(
            f"The '{name}' guide has no section '{section}'. Its sections are: "
            f"{', '.join(_sections(text))}."
        )
    heading, content = found
    return _noted({"name": name, "section": heading, "content": content})


def _index(name, text):
    """The guide's opening plus the headings of what was not sent.

    `content` is everything before the first heading followed by the first
    section - the part that says what the guide is for - so an agent that
    reads only this still knows which section it wants.
    """
    matches = list(SECTION_PATTERN.finditer(text))
    headings = [match.group(1) for match in matches]
    if not matches:
        return {
            "name": name,
            "section": None,
            "content": text,
            "sections": headings,
            "withheld": [],
        }
    end = matches[1].start() if len(matches) > 1 else len(text)
    content = text[:end].rstrip() + "\n"
    withheld = headings[1:]
    answer = {
        "name": name,
        "section": None,
        "content": content,
        "sections": headings,
        "withheld": withheld,
    }
    if withheld:
        answer["note"] = (
            f"This is the '{name}' guide's opening and its first section. "
            f"{len(withheld)} further section(s) - {len(text) - len(content)} "
            f"characters - were not sent: read one with "
            f"get_guide('{name}', section='<heading>'). `sections` lists "
            f"every heading in order."
        )
    return answer
