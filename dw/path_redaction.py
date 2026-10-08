"""A failed run's message and its warnings, with the server's library paths
named back as references.

A task that fails on an input reports the file it was handed, and by then an
'asset:' reference has been resolved to an absolute path - so a job error
read over MCP or the HTTP API carried the server's home directory and
workspace layout (GHSA-fwg5-jfjg-fxpf). A warning naming the track it
resampled did the same (GHSA-cr8g-q9j9-j68g). The text is rewritten where the
worker builds the reply - the failure, and each free-text run event - once,
rather than in every task that might name a path: a file under an asset root
becomes 'asset:<name>', one under the job's output directory 'output:<name>' - the reference the caller would
write to address it - and a bare root becomes the library's name.

It imports only dw.references - which imports nothing from dw - so the
worker's failure path cannot fail for want of it, and the reference is built
by the module that owns its spelling.
"""

import os
import re

from . import references

# What follows a root in a message, up to the end of the path: anything but
# whitespace or the quotes a message wraps a path in. A name with a space in
# it is cut at the space, so the reference reads short - the root is gone
# either way
_REST_OF_PATH = r"([^\s'\"`]*)"


def _spellings(root):
    """The ways a root may appear in a message: as configured, and resolved
    (a path validate_path returned has had its symlinks followed - /tmp is
    /private/tmp on macOS)."""
    spellings = {os.path.abspath(root), os.path.realpath(root)}
    return {spelling.rstrip(os.sep) for spelling in spellings if spelling}


def redact_paths(text, asset_roots=(), output_dir=None):
    """`text` with every path under the given roots written as a reference.

    Args:
        text: A message or traceback; None or "" is returned as it came
        asset_roots: The directories on the job's asset search path
        output_dir: The job's output directory
    """
    if not text:
        return text
    rewrites = []
    for root in asset_roots or ():
        if root:
            rewrites += [
                (s, references.ASSET, "the asset library") for s in _spellings(root)
            ]
    if output_dir:
        rewrites += [
            (s, references.OUTPUT, "the output folder") for s in _spellings(output_dir)
        ]
    # Longest first, so a root nested in another is rewritten by its own
    # prefix rather than as a subfolder of the outer one
    for root, kind, label in sorted(rewrites, key=lambda r: -len(r[0])):
        text = re.sub(
            re.escape(root + os.sep) + _REST_OF_PATH,
            # A reference's separator is '/', whatever the server's is
            lambda match, kind=kind: references.make_ref(
                kind, match.group(1).replace(os.sep, "/")
            ),
            text,
        )
        # Only the root itself: '/ws/assets' must not turn '/ws/assets2'
        # into 'the asset library2'
        text = re.sub(re.escape(root) + r"(?![\w-]|\.\w)", label, text)
    return text


# The run events whose text a step writes freely - a warning naming the
# track it resampled, a log line naming the file it wrote, a phase detail
# naming the model it loads. The others carry structured fields: 'files'
# and 'manifest' the server relativises itself (dw/server/jobs.py), and
# they must reach it absolute to be
FREE_TEXT_EVENTS = ("warning", "log", "phase")


def redact_event(event, redact):
    """A run event with every string in it - keys too, as a warning's
    per-track `sample_rates` is keyed by track name - passed through
    `redact`, when it is one of FREE_TEXT_EVENTS; any other event as it came.

    Args:
        event: The event dict a RunContext emits
        redact: One string to its redacted form (what redact_paths does,
            bound to a job's roots)
    """
    if not isinstance(event, dict) or event.get("event") not in FREE_TEXT_EVENTS:
        return event

    def walk(value):
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, dict):
            return {walk(key): walk(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(item) for item in value]
        return value

    return walk(event)
