"""The workflow reference prefixes, spelled once.

A string value in a workflow definition that starts with one of these is a
reference, not a literal: something to substitute (`variable:`, `item:`),
an earlier step's result (`previous_result:`, `gather:`), a file or text to
fetch (`asset:`, `output:`, `prompt:`), a python value (`constant:`), a
packaged sub-workflow (`builtin:`) or a declared rule (`constraint:`).
Every module that tests for, strips or builds one does it through this
module, and no other module spells a prefix
(`scripts/arch_metrics.py` counts any that do).

It also holds how a reference's location is written: the separator in a
for_each member's name (`<group>@<entry>`) and a JSON path to a value
(`steps[3].task.arguments.videos[1]`).

This module imports nothing from dw, so every module can import it.
"""

ASSET = "asset:"
OUTPUT = "output:"
PROMPT = "prompt:"
VARIABLE = "variable:"
PREVIOUS_RESULT = "previous_result:"
CONSTANT = "constant:"
ITEM = "item:"
GATHER = "gather:"
BUILTIN = "builtin:"
CONSTRAINT = "constraint:"

# The key naming the file an argument object is constructed from
FROM_FILE_KEY = "from_file"

# The key naming the step whose output an argument object is constructed from
FROM_PREVIOUS_RESULT_KEY = "from_previous_result"

# The key holding the arguments an argument object is constructed from, for a type
# that takes its contents as plain fields rather than opening media itself
FROM_ARGUMENTS_KEY = "from_arguments"

# Still to be substituted: what the variable and for_each passes replace.
# A check that runs on the substituted definition skips a value still
# spelled this way - it is an entry field or a variable those passes left
SUBSTITUTED = (VARIABLE, ITEM)
# Not a literal until the run reaches the step: substitution plus the
# results of earlier steps
UNRESOLVED = (VARIABLE, ITEM, PREVIOUS_RESULT, GATHER)
# Anything a value may still hold before realization: the unresolved
# prefixes plus the ones realize_args fetches or looks up
DEFERRED = UNRESOLVED + (ASSET, OUTPUT, PROMPT, CONSTANT, BUILTIN)


def is_ref(kind, value):
    """Whether `value` is a reference of `kind` - one prefix, or a tuple of
    them (any of). False for anything that is not a string."""
    return isinstance(value, str) and value.startswith(kind)


def _single(kind):
    # A prefix set has no one name to strip or build with - is_ref is the
    # only helper that takes one
    if not isinstance(kind, str):
        raise TypeError(f"expected one reference prefix, got {kind!r}")


def ref_name(kind, value):
    """What follows the `kind` prefix in `value`, or None when `value` is
    not a reference of that kind. `kind` is a single prefix."""
    _single(kind)
    if not is_ref(kind, value):
        return None
    return value[len(kind) :]


def make_ref(kind, name):
    """The reference of `kind` naming `name`: make_ref(ASSET, "iris.png")
    is "asset:iris.png"."""
    _single(kind)
    return f"{kind}{name}"


def author_index(source_indices, index):
    """The index, in the steps the author wrote, of expanded step `index`.

    `expand_for_each` records one source index per expanded step, so an
    error in a for_each member is reported at the template step it came
    from. Without a record (no expansion, or a list too short) the step is
    its own source."""
    if source_indices is not None and index < len(source_indices):
        return source_indices[index]
    return index


# A for_each member is named `<group>@<entry>`
MEMBER_SEPARATOR = "@"


def render_path(path):
    """'steps[3].task.arguments.videos[1]' - the same shape schema errors use."""
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = str(part)
    return rendered


def iter_previous_result_references(value, *, descend_into_from, path=()):
    """Every previous-result reference under `value`, as `(path, name, via)`.

    `path` is the tuple of keys and list indices that reaches it. `via` is
    "prefix" for a string spelled `previous_result:<name>` (`name` is what
    follows the prefix) and "key" for a `from_previous_result` string, which
    names its step bare (`name` is the string as written).

    A `from_previous_result` value that is not a string is walked like any
    other. One that is a string is yielded as "key" and, only when
    `descend_into_from` is true, walked too - so a prefixed spelling written
    there yields a second, "prefix" reference at the same path. Callers
    filter what they want: a `variable:` value, say, is yielded as written.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            here = path + (key,)
            if key == FROM_PREVIOUS_RESULT_KEY and isinstance(item, str):
                yield here, item, "key"
                if not descend_into_from:
                    continue
            yield from iter_previous_result_references(
                item, descend_into_from=descend_into_from, path=here
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_previous_result_references(
                item, descend_into_from=descend_into_from, path=path + (index,)
            )
    elif is_ref(PREVIOUS_RESULT, value):
        yield path, ref_name(PREVIOUS_RESULT, value), "prefix"
