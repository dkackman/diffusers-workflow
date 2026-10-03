"""The LoRA catalog: which adapters fit a base model, and in what order to
offer them.

An entry (`dw/lora_catalog_schema.json`) records a LoRA that was tried on a
base - proven, still a trial, or rejected with the reason. Matching is exact
on the base repo id and, for MiniMax-H3, on the partition a step denoises
against (the `workflow=` value `dw/adapter_compatibility.py` reads): a LoRA
on the wrong base usually loads without complaint and is quietly worse, so a
family prefix or an alias would list exactly the entries that fail silently.

No network here - `dw/lora_hub.py` is the only module that searches the Hub.
"""

import re

from . import references
from .schema import load_schema, validate_data

SCHEMA_NAME = "lora_catalog"
# A catalog entry by this name would be shadowed by GET /api/loras/recommend
RESERVED_NAMES = frozenset({"recommend"})
STATUS_ORDER = {"proven": 0, "trial": 1, "rejected": 2}

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# Words that say nothing about which LoRA: every style LoRA is a "style lora"
STOP_WORDS = frozenset({"style", "lora", "loras", "the", "and", "with", "for", "image", "images"})
MIN_TERM_LENGTH = 3


def entry_errors(entry):
    """The schema's complaint about an entry, or None when it is valid."""
    valid, message = validate_data(entry, load_schema(SCHEMA_NAME))
    return None if valid else message


def is_repo_id(value):
    """Whether `value` is a Hub repo id - `owner/name`, no traversal."""
    return isinstance(value, str) and bool(REPO_ID.match(value)) and ".." not in value


def _resolved(value, variables):
    """A `variable:` reference replaced by the variable's default; anything
    else unchanged."""
    if isinstance(value, str) and value.startswith(references.VARIABLE):
        return variables.get(value[len(references.VARIABLE):])
    return value


def workflow_bases(definition):
    """The `(repo, partition)` each pipeline step loads, in step order, once
    each. A base that is not a repo id once its variable is substituted (a
    null default, a local path) is skipped: nothing on the Hub is keyed by it."""
    variables = definition.get("variables")
    if not isinstance(variables, dict):
        variables = {}
    steps = definition.get("steps")
    bases = []
    for step in steps if isinstance(steps, list) else []:
        pipeline = step.get("pipeline") if isinstance(step, dict) else None
        if not isinstance(pipeline, dict):
            continue
        arguments = pipeline.get("from_pretrained_arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        repo = _resolved(arguments.get("model_name"), variables)
        if not is_repo_id(repo):
            continue
        partition = _resolved(arguments.get("workflow"), variables)
        pair = (repo, partition if isinstance(partition, str) else None)
        if pair not in bases:
            bases.append(pair)
    return bases


def matches(entry, bases):
    """Whether an entry fits any of `bases`. A partitioned entry fits a step
    of that partition, or a bare repo (no partition asked for)."""
    constraint = entry.get("workflow")
    for repo, partition in bases:
        if repo not in entry.get("base_models", []):
            continue
        if constraint is None or partition is None or constraint == partition:
            return True
    return False


def query_terms(query):
    """The words of a request worth matching on, lower-cased, in order."""
    words = re.findall(r"[a-z0-9]+", (query or "").lower())
    terms = []
    for word in words:
        if len(word) >= MIN_TERM_LENGTH and word not in STOP_WORDS and word not in terms:
            terms.append(word)
    return terms


def _score(entry, terms):
    text = " ".join(
        [entry.get("use_when", ""), entry.get("description", "")] + list(entry.get("tags", []))
    ).lower()
    return sum(1 for term in terms if term in text)


def ranked(entries, terms):
    """Entries as rows (`{"name", **entry}`): most query terms first, then
    proven before trial before rejected, then by name."""
    rows = [{"name": name, **entry} for name, entry in entries.items()]
    rows.sort(
        key=lambda row: (
            -_score(row, terms),
            STATUS_ORDER.get(row.get("status"), len(STATUS_ORDER)),
            row["name"],
        )
    )
    return rows


def rejection_reasons(entries):
    """`model_name -> reason` for every rejected entry; the reason is its
    first evidence note, which the schema requires."""
    return {
        entry["model_name"]: entry["evidence"][0]["note"]
        for entry in entries.values()
        if entry.get("status") == "rejected" and entry.get("evidence")
    }
