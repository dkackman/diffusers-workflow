"""The prose guides, served to an agent that has to pick a capability.

Everything else in this package proxies the server. These do not: they are
documentation shipped with the package, so they are read locally and work the
same against a remote engine as a local one.

The gap they close is that a catalog entry describes what a workflow *is*, and
an open-ended request ("a lego movie trailer set in the marvel universe")
names nothing that appears in any of them. What has to be matched is shape -
multi-shot video, cuts, a consistent cast, narration - and that is what the
guides are written in. An agent that reads one before choosing composes from
what exists instead of authoring a fresh workflow badly.

Two things keep this cheap. The listing carries each guide's section headings,
so the index is itself the routing table and is small enough to read every
time. And a guide can be fetched one section at a time, because handing over
three thousand lines of TASKS.md is how an agent ends up reading none of it
carefully.
"""

import re
from pathlib import Path

from dw_mcp.client import DwApiError

# The docs that bear on choosing a capability. Deliberately not all of them -
# testing, releasing, security and dependency notes are for people working on
# dw, and listing them would dilute an index whose whole value is that it is
# short enough to read in full.
GUIDES = {
    "workflows": (
        "WORKFLOW_GUIDE.md",
        "How a workflow is put together: steps, variables, the reference "
        "conventions that carry data between steps, sub-workflows, and "
        "releasing models mid-run. Read before authoring one.",
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
        "Emphasis and de-emphasis syntax in prompts, and which pipelines "
        "honour it.",
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


def _guide_file(file_name):
    """Where a guide's markdown actually is.

    Packaged into dw/docs/ at build time, the way the SPA is copied into
    dw/server/ui - so an install carries them. A checkout has no such copy
    until something builds one, and reads the repo's own docs/ instead.
    """
    packaged = Path(__file__).resolve().parent.parent / "dw" / "docs" / file_name
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parent.parent / "docs" / file_name


def read_guide(name):
    """One guide's full markdown text.

    Raises:
        DwApiError: If the name is not a guide, or its file is missing - as a
            DwApiError rather than anything else so the message reaches the
            model instead of being replaced by a traceback.
    """
    if name not in GUIDES:
        raise DwApiError(
            f"No guide named '{name}'. The guides are: "
            f"{', '.join(sorted(GUIDES))}."
        )

    path = _guide_file(GUIDES[name][0])
    if not path.is_file():
        raise DwApiError(
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


def list_guides():
    """Every guide, with what it covers and the sections it holds."""
    listed = []
    for name, (file_name, summary) in GUIDES.items():
        text = read_guide(name)
        listed.append(
            {
                "name": name,
                "file": file_name,
                "summary": summary,
                "sections": _sections(text),
            }
        )
    return {"guides": listed}


def get_guide(name, section=None):
    """One guide, whole or one section of it."""
    text = read_guide(name)
    if section is None:
        return {"name": name, "section": None, "content": text}

    found = _extract_section(text, section)
    if found is None:
        raise DwApiError(
            f"The '{name}' guide has no section '{section}'. Its sections are: "
            f"{', '.join(_sections(text))}."
        )
    heading, content = found
    return {"name": name, "section": heading, "content": content}
