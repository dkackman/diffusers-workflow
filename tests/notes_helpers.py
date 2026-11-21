"""A stand-in for saving a user's note, until the notes routes exist: the
note is put into the stored ledger file by hand."""

import json
import os

from dw.workspace import REQUESTS_FILE_NAME


def save_note(root, request_id, text, item=None, at=None, refs=None):
    """Replace the notes on an item (or on the request itself when `item` is
    None) with one note carrying `text` (and `refs`, stored as given), or with none when `text` is None.
    The note is stamped `at`, else with the request's latest history entry, so
    call this right after the write that stands for the user saving it. Returns the
    note, or None when cleared."""
    path = os.path.join(str(root), REQUESTS_FILE_NAME)
    with open(path) as f:
        document = json.load(f)
    request = document["requests"][request_id]
    owner = (
        request
        if item is None
        else next(i for i in request["items"] if i["key"] == item)
    )
    saved = None
    if text is not None:
        at = at or request["history"][-1]["at"]
        saved = {
            "id": f"n{request['next_note']}",
            "text": text,
            "created": at,
            "at": at,
            **({"refs": refs} if refs else {}),
        }
        request["next_note"] += 1
    owner["notes"] = [saved] if saved else []
    with open(path, "w") as f:
        json.dump(document, f)
    return saved
