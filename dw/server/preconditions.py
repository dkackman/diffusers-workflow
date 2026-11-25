"""Opt-in conditional writes: ETags and If-Match / If-None-Match.

A request without either header is not checked at all. The ETag is a
strong validator over the file's bytes, so it changes exactly when the
file does, whoever wrote it (another tab, an agent, an editor).

Nothing here resolves a path: callers pass the path their own validators
already returned, and the checks read only that one file.
"""

import hashlib
import os
from typing import Optional

from fastapi import HTTPException, Request


def content_etag_of(data: bytes) -> str:
    """The ETag of exactly these bytes - what a reader that already holds
    the bytes it serves should send, so body and tag never disagree."""
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


def content_etag(path: str) -> Optional[str]:
    """`"` + the first 32 hex characters of the SHA-256 of the file's
    bytes + `"`, or None when the file does not exist."""
    try:
        with open(path, "rb") as file:
            data = file.read()
    except FileNotFoundError:
        return None
    return content_etag_of(data)


def _tags(header: str):
    """The entity tags in an If-Match value, weak marks dropped."""
    return [
        tag.strip().removeprefix("W/").strip()
        for tag in header.split(",")
        if tag.strip()
    ]


def check_write_preconditions(request: Request, path: str, label: str) -> None:
    """Refuse a write whose request says what it expects of `path`.

    `If-None-Match: *` means "create only": a file already there is a 409
    `exists`, answered from its existence alone - never by reading it, so
    a large upload's name collision does not hash it on the event loop. `If-Match: <etag>` means "replace only what I read": a file
    that differs, or is gone, is a 409 `stale_write` carrying the current
    etag (None when gone). Call it inside the write lock, against the
    destination the write will land on.
    """
    if_none_match = request.headers.get("if-none-match")
    if if_none_match is not None and "*" in _tags(if_none_match):
        if os.path.exists(path):
            raise HTTPException(
                status_code=409,
                detail={"code": "exists", "message": f"{label} already exists"},
            )
    if request.headers.get("if-match") is not None:
        _match(request, content_etag(path), label)


def check_if_match(request: Request, data: bytes, label: str) -> None:
    """`If-Match` against bytes the caller already holds - a value stored
    inside a file, such as one note in notes.json - rather than a whole
    file: a 409 `stale_write` carrying their etag unless the header names
    it. No header, no check. Call it under the lock its write takes."""
    if request.headers.get("if-match") is not None:
        _match(request, content_etag_of(data), label)


def _match(request: Request, current: Optional[str], label: str) -> None:
    tags = _tags(request.headers["if-match"])
    if current is None or not ("*" in tags or current in tags):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stale_write",
                "message": f"{label} changed since it was read",
                "etag": current,
            },
        )
