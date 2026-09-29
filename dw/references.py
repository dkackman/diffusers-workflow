"""The workflow reference prefixes, spelled once.

A string value in a workflow definition that starts with one of these is a
reference, not a literal: something to substitute (`variable:`, `item:`),
an earlier step's result (`previous_result:`, `gather:`), a file or text to
fetch (`asset:`, `output:`, `prompt:`), a python value (`constant:`), a
packaged sub-workflow (`builtin:`) or a declared rule (`constraint:`).
Every module that tests for, strips or builds one does it through this
module, and no other module spells a prefix
(`scripts/arch_metrics.py` counts any that do).

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
