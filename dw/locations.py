"""The one policy for a location a workflow's arguments supply.

A workflow JSON is untrusted input under the default posture (see
docs/SECURITY.md's Trust model section). Its media arguments name *where* to
read from, and until this module existed each loader answered that question
for itself: `fetch_image` accepted any absolute path the JSON wrote,
`gather_images` handed its `glob` straight to the filesystem, and any
`http(s)` URL was fetched whatever host it named. That is arbitrary file read
and SSRF from a document the server treats as data (#114, #115, #116, #112).

Two rules, applied wherever a caller-supplied location is resolved:

- A path must land inside one of the roots this installation already works
  in - the workflow's own directory, the asset libraries on the search path,
  the output root. `..` never appears in a legitimate one and `validate_path`
  already refuses it, so in practice this closes the *absolute* path that
  pointed somewhere else entirely. The remedy is an `asset:` reference, which
  is what the roots exist for.
- An `http(s)` URL must not name a host inside the deployment - loopback,
  link-local (the cloud metadata address), or a private range. The check runs
  on the resolved address, not on the literal string, so a hostname that
  answers 127.0.0.1 is caught too.

Both yield to `--trust-workflows`, exactly as the import and remote-code
gates do: an operator who has vouched for a workflow's source may point it at
a scratch directory or an internal endpoint. Neither yields to anything else -
there is no per-argument opt-out, because the argument is the untrusted part.

Enforcement is in two places on purpose. `location_errors` runs at validation
time, so `validate_workflow` refuses the workflow before a model load is
spent on it; the loaders call the same functions at run time, because a
location that arrives through a variable or a previous result was never in
the document to check.
"""

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse

from .security import (
    InvalidInputError,
    PathTraversalError,
    validate_path,
    validate_url,
    workflows_are_trusted,
)

logger = logging.getLogger("dw")

# Media argument names follow the same conventions realize_args dispatches on
# (dw/arguments.py): a key named like its media, or an explicit
# {"media_type", "location"} reference, or an object's "from_file"
MEDIA_KEY_SUFFIXES = ("_image", "_video", "_audio")
MEDIA_KEY_NAMES = ("image", "video", "audio", "location", "from_file")

# The tasks whose arguments name a filesystem pattern rather than one file
GLOB_ARGUMENT = "glob"


def is_http_url(value):
    """Whether a value is a string the loaders would fetch over HTTP."""
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def media_roots(base_dir=None):
    """Every directory a workflow's own locations may point inside, resolved.

    The workflow's directory, each asset library on the search path, and the
    output root - the three places this installation keeps the media a
    workflow works with. A root that cannot be resolved (no workspace, no
    active run) is dropped rather than failing the check open.

    Args:
        base_dir: The workflow file's directory, when one anchors the search
    """
    candidates = []
    if base_dir:
        candidates.append(base_dir)

    from .assets import asset_search_path

    try:
        candidates.extend(asset_search_path(base_dir=base_dir))
    except Exception:
        logger.debug("Could not resolve the asset search path", exc_info=True)

    from .runs import output_root

    try:
        candidates.append(output_root())
    except Exception:
        logger.debug("Could not resolve the output root", exc_info=True)

    roots = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = os.path.normpath(
                os.path.realpath(os.path.abspath(os.path.expanduser(str(candidate))))
            )
        except (OSError, ValueError):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _within(path, root):
    return path == root or path.startswith(root + os.sep)


def validate_media_path(
    location, base_dir=None, what="a media argument", require_exists=True
):
    """The validated absolute path a media location names, confined.

    Args:
        location: The path the workflow supplied, relative or absolute
        base_dir: Directory a relative path is resolved against - the
            workflow file's directory
        what: Short phrase naming the argument, for the error message
        require_exists: Whether a contained path that does not exist is an
            error. False for the validation-time pass, which is about policy
            rather than about what happens to be on disk right now

    Returns:
        The absolute, resolved, contained path

    Raises:
        PathTraversalError: If the path resolves outside every root
        InvalidInputError, PathTraversalError: Whatever validate_path raises
    """
    # base_dir is the first root, so a relative path keeps resolving against
    # the workflow file exactly as it did before this check existed.
    # allow_create here, with the existence check moved below the containment
    # one: refusing an out-of-root path only once it turned out to exist made
    # the refusal itself a file-existence oracle for the whole filesystem
    # (#114)
    resolved = validate_path(
        location if os.path.isabs(str(location)) else _joined(location, base_dir),
        allow_create=True,
    )
    if not workflows_are_trusted():
        roots = media_roots(base_dir)
        if not any(_within(resolved, root) for root in roots):
            raise PathTraversalError(
                f"Refusing to read {what} at '{location}': it resolves "
                f"outside every directory this workflow may read "
                f"({', '.join(roots) or 'none configured'}). Put the file in "
                f"the asset library and name it with an 'asset:' reference, "
                f"or pass --trust-workflows if you trust this workflow's "
                f"source."
            )

    if require_exists and not os.path.exists(resolved):
        raise InvalidInputError(f"Path does not exist: {resolved}")
    return resolved


def _joined(location, base_dir):
    return os.path.join(base_dir, str(location)) if base_dir else str(location)


def validate_media_glob(pattern, base_dir=None, what="a glob argument"):
    """The validated glob pattern, confined to one of the media roots.

    A glob is a location with wildcards in it, and it is checked the same
    way - but on the pattern's fixed leading directory, since the pattern
    itself does not exist as a path. Each *match* is checked individually
    too by the loader that opens it, which is what catches a wildcard
    escaping through a symlink.

    Returns:
        The pattern, absolute, for glob to expand
    """
    absolute = pattern if os.path.isabs(str(pattern)) else _joined(pattern, base_dir)
    if workflows_are_trusted():
        return absolute

    # The part of the pattern before the first wildcard: the directory the
    # expansion starts from, which is the thing containment is about
    fixed = str(absolute)
    for wildcard in ("*", "?", "["):
        cut = fixed.find(wildcard)
        if cut >= 0:
            fixed = fixed[:cut]
    fixed = os.path.dirname(fixed) if not fixed.endswith(os.sep) else fixed
    if ".." in fixed.replace("\\", "/").split("/"):
        raise PathTraversalError(
            f"Refusing {what} '{pattern}': it contains a '..' path segment."
        )
    try:
        resolved = os.path.normpath(os.path.realpath(os.path.abspath(fixed or ".")))
    except (OSError, ValueError) as e:
        raise InvalidInputError(f"Invalid glob pattern {pattern!r}: {e}")

    roots = media_roots(base_dir)
    if not any(_within(resolved, root) for root in roots):
        raise PathTraversalError(
            f"Refusing {what} '{pattern}': it expands under {resolved}, "
            f"outside every directory this workflow may read "
            f"({', '.join(roots) or 'none configured'}). Glob inside the "
            f"asset library, or pass --trust-workflows if you trust this "
            f"workflow's source."
        )
    return absolute


def contained_matches(paths, base_dir=None, what="a glob argument"):
    """The matches of an allowed glob that are themselves inside a root.

    A pattern can be contained and still match outside its own tree through
    a symlink, so every match is re-checked on its real path. A match that
    escapes is dropped with a warning rather than failing the run: the
    pattern was legitimate, one entry under it was not.
    """
    if workflows_are_trusted():
        return list(paths)

    roots = media_roots(base_dir)
    kept = []
    for path in paths:
        try:
            resolved = os.path.normpath(os.path.realpath(os.path.abspath(path)))
        except (OSError, ValueError):
            continue
        if any(_within(resolved, root) for root in roots):
            kept.append(path)
        else:
            logger.warning(
                f"Skipping {what} match '{path}': it resolves to {resolved}, "
                f"outside every directory this workflow may read"
            )
    return kept


# Hosts a workflow may not send the server to: its own loopback, the
# link-local range cloud metadata services answer on, and the private ranges
# that make up whatever network the box sits in. This is the SSRF boundary -
# an internal address is exactly the thing a caller cannot otherwise reach,
# which is why naming one is the attack rather than a mistake
def _is_internal(address):
    return (
        address.is_loopback
        or address.is_link_local
        or address.is_private
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _resolved_addresses(host):
    """Every IP a hostname answers on, as ip_address objects.

    A literal is returned as itself without a lookup. A name that does not
    resolve yields nothing - the fetch will fail on its own, and refusing it
    here would turn a typo into a security error.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        logger.debug(f"Could not resolve {host!r} for the host policy")
        return []
    addresses = []
    for info in infos:
        try:
            addresses.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    return addresses


def validate_media_url(url, what="a media argument"):
    """The validated URL, refused if it names a host inside the deployment.

    Args:
        url: The http(s) URL the workflow supplied
        what: Short phrase naming the argument, for the error message

    Returns:
        The URL, unchanged

    Raises:
        InvalidInputError: If the scheme is not http(s), or the host is
            internal to the deployment
    """
    validated = validate_url(url)
    if workflows_are_trusted():
        return validated

    host = (urlparse(validated).hostname or "").strip("[]")
    internal = [
        address for address in _resolved_addresses(host) if _is_internal(address)
    ]
    if internal:
        raise InvalidInputError(
            f"Refusing to fetch {what} from '{url}': {host} resolves to "
            f"{internal[0]}, an address inside this deployment (loopback, "
            f"link-local or private). A workflow may not use the server to "
            f"reach its own network. Pass --trust-workflows if you trust "
            f"this workflow's source."
        )
    return validated


# Hosts this machine's HuggingFace token belongs to. The token is the
# credential the box holds for the Hub; a workflow chooses
# `remote_text_encoder.url`, so attaching the token to whatever it named
# would let an untrusted document exfiltrate it with one POST (#112). An
# endpoint outside these is still reachable - it just does not get the
# credential, and an endpoint that needs one is by definition a HuggingFace
# endpoint
HF_TOKEN_HOST_SUFFIXES = (
    "huggingface.co",
    "huggingface.cloud",
    "hf.space",
)


def token_host_allowed(host):
    """Whether the HuggingFace token may be attached to a request to `host`."""
    host = (host or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in HF_TOKEN_HOST_SUFFIXES
    )


def validate_remote_encoder_url(url):
    """The validated remote text-encoder URL, or a refusal saying why.

    https only, and no address inside this deployment: the workflow file is
    untrusted input, and this field sends a request - with a credential - to
    an address it chooses. `--trust-workflows` lifts both, for an operator
    running their own endpoint on the box or over plain http on a LAN.

    Raises:
        InvalidInputError: On a non-https scheme or an internal host
    """
    if not workflows_are_trusted():
        scheme = urlparse(str(url)).scheme
        if scheme != "https":
            raise InvalidInputError(
                f"Refusing the remote text encoder at '{url}': its scheme is "
                f"'{scheme or 'none'}', and an untrusted workflow may only "
                f"reach an https endpoint - the request carries this "
                f"machine's HuggingFace token. Pass --trust-workflows if you "
                f"trust this workflow's source."
            )
    return validate_media_url(url, "the remote text encoder url")


def validate_model_name(name, base_dir=None):
    """A model identifier: a Hub repo id, or a path inside the media roots.

    `download_model` has always refused a path-shaped `repo_id`; the same
    name reached `from_pretrained_arguments.model_name` unchecked, so a
    workflow could name an absolute path and let diffusers decide (#117).
    A local model directory stays supported - inside a root, like any other
    location.

    Raises:
        PathTraversalError, InvalidInputError: If it is neither
    """
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    try:
        validate_repo_id(str(name))
        return str(name)
    except HFValidationError:
        pass
    return validate_media_path(
        str(name), base_dir, "a model_name", require_exists=False
    )


# ------------------------------------------------------------- validation


def _is_media_key(key):
    return isinstance(key, str) and (
        key in MEDIA_KEY_NAMES or key.endswith(MEDIA_KEY_SUFFIXES)
    )


def _deferred(value):
    """Whether a location is resolved later rather than being one now."""
    return value.startswith(
        (
            "variable:",
            "previous_result:",
            "item:",
            "gather:",
            "asset:",
            "output:",
            "prompt:",
            "constant:",
            "builtin:",
        )
    )


def _check(value, base_dir, what):
    """The policy message for one literal location, or None if it is fine."""
    if not isinstance(value, str) or not value or _deferred(value):
        return None
    try:
        if is_http_url(value):
            validate_media_url(value, what)
        elif os.path.isabs(value):
            # Only an absolute path can escape: validate_path already
            # refuses '..', so a relative one is under base_dir by
            # construction and costs a stat nobody asked for here
            validate_media_path(value, base_dir, what, require_exists=False)
    except (PathTraversalError, InvalidInputError) as e:
        return str(e)
    except Exception:
        # A path that simply does not exist is not this check's business -
        # the loader will say so, with the run's own error
        logger.debug(f"Location check skipped for {value!r}", exc_info=True)
    return None


def location_errors(definition, source_indices=None, base_dir=None):
    """Every media location in the definition that policy refuses.

    Reported as [{path, message}] like the other validation passes, so a
    caller learns before `run_workflow` that the workflow will not be
    allowed to read what it names - rather than after a pipeline load.

    Args:
        definition: The expanded, substituted workflow definition
        source_indices: Step index -> index in the file the author wrote
        base_dir: The workflow file's directory
    """
    errors = []
    steps = definition.get("steps") or []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        source = (
            source_indices[index]
            if source_indices and index < len(source_indices)
            else index
        )
        _walk(step, f"steps[{source}]", base_dir, errors)
    return errors


def _walk(node, path, base_dir, errors):
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key == "model_name" and isinstance(value, str):
                message = _model_name_message(value, base_dir)
                if message:
                    errors.append({"path": here, "message": message})
                continue
            if key == "remote_text_encoder" and isinstance(value, dict):
                url = value.get("url")
                if isinstance(url, str) and not _deferred(url):
                    message = _refusal(validate_remote_encoder_url, url)
                    if message:
                        errors.append({"path": f"{here}.url", "message": message})
                continue
            if key == GLOB_ARGUMENT and isinstance(value, str):
                message = _glob_message(value, base_dir)
                if message:
                    errors.append({"path": here, "message": message})
                continue
            if _is_media_key(key):
                for sub_path, item in _each(value, here):
                    message = _check(item, base_dir, f"'{key}'")
                    if message:
                        errors.append({"path": sub_path, "message": message})
            if key == "urls" and isinstance(value, list):
                for sub_path, item in _each(value, here):
                    message = _check(item, base_dir, f"'{key}'")
                    if message:
                        errors.append({"path": sub_path, "message": message})
            _walk(value, here, base_dir, errors)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _walk(item, f"{path}[{index}]", base_dir, errors)


def _each(value, path):
    if isinstance(value, list):
        return [(f"{path}[{i}]", item) for i, item in enumerate(value)]
    return [(path, value)]


def _glob_message(pattern, base_dir):
    if _deferred(pattern):
        return None
    return _refusal(validate_media_glob, pattern, base_dir)


def _model_name_message(name, base_dir):
    if _deferred(name):
        return None
    return _refusal(validate_model_name, name, base_dir)


def _refusal(check, *args):
    """The message `check` refused its argument with, or None if it allowed it."""
    try:
        check(*args)
    except (PathTraversalError, InvalidInputError) as e:
        return str(e)
    except Exception:
        logger.debug(f"Location check skipped for {args[0]!r}", exc_info=True)
    return None
