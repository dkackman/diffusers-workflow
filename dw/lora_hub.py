"""LoRA candidates from the Hugging Face Hub - the one place dw searches the
Hub on its own, and only when `recommend_loras` asks.

Exact base: the search is `list_models(filter="base_model:adapter:<repo>")`,
so an adapter declared for another revision of a family never appears.
Nothing is downloaded: the file list and card come with the listing
(`expand=`), and the weight file's safetensors header is a range read, which
is enough to tell a `load_lora_weights` layout from a full-weight diff that
diffusers will refuse (the FastH3 failure, #585). A repo holding only pickle
`.bin` weights is dropped - dw does not offer to load pickles.

What comes back is a candidate to trial, never a recommendation: download
counts are the only quality signal the Hub has.
"""

import logging
import threading

from .lora_catalog import is_repo_id

logger = logging.getLogger("dw")

HUB_TIMEOUT = 20.0
HEADER_TIMEOUT = 5.0
SEARCH_EXPAND = [
    "downloads",
    "likes",
    "lastModified",
    "gated",
    "cardData",
    "sha",
    "siblings",
]
SAFETENSORS = ".safetensors"
KOHYA_PREFIXES = ("lora_unet_", "lora_te")
FULL_WEIGHT_SUFFIXES = (".diff", ".diff_b")
DIFFUSERS_MARKERS = ("lora_A", "lora_B")
KOHYA_MARKERS = ("lora_down", "lora_up")
CARD_TEXT_LIMIT = 200
MAX_SEARCH_TERMS = 4
BUSY_ERROR = "A Hub search is already running on this server; try again shortly"
# one Hub search per server, released when the worker finishes (not at the timeout),
# so a hung search cannot stack threads.
_IN_FLIGHT = threading.Lock()


def classify_format(keys):
    """Which layout a LoRA's tensor names are in. A full-weight diff is
    checked first: a file carrying any is refused by diffusers' converters
    whatever else it holds."""
    keys = list(keys)
    if any(key.endswith(FULL_WEIGHT_SUFFIXES) for key in keys):
        return "full_weight"
    if any(key.startswith(KOHYA_PREFIXES) for key in keys):
        return "kohya"
    if any(marker in key for key in keys for marker in KOHYA_MARKERS):
        return "kohya"
    if any(marker in key for key in keys for marker in DIFFUSERS_MARKERS):
        return "diffusers"
    return "unknown"


def _card_value(card, key):
    if card is None:
        return None
    value = card.get(key) if hasattr(card, "get") else getattr(card, key, None)
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        value = value[:CARD_TEXT_LIMIT]
    return value


def _searches(query, terms):
    """The search strings to run: the request as typed, then each term (up to
    MAX_SEARCH_TERMS to prevent unbounded amplification). No terms (a stop-word
    or empty query) is one unfiltered search."""
    if not terms:
        return [None]
    searches = []
    for search in [(query or "").strip()] + terms[:MAX_SEARCH_TERMS]:
        if search and search not in searches:
            searches.append(search)
    return searches


def _inspect(api, info, base_modified):
    """One listing row as a candidate, or None for a repo with no
    safetensors weights."""
    weights = sorted(
        s.rfilename for s in info.siblings or [] if s.rfilename.endswith(SAFETENSORS)
    )
    if not weights:
        return None
    warnings = []
    weight = weights[0] if len(weights) == 1 else None
    if weight is None:
        warnings.append("multiple_weights")
    form = "unknown"
    if weight is not None:
        try:
            header = api.parse_safetensors_file_metadata(
                info.id, weight, revision=info.sha, timeout=HEADER_TIMEOUT
            )
            form = classify_format(header.tensors.keys())
        except Exception as error:  # one candidate's failure is its warning
            logger.info(f"LoRA header for {info.id} unreadable: {error}")
            warnings.append("header_unreadable")
    if form == "full_weight":
        warnings.append("will_not_load")
    elif form == "unknown" and "header_unreadable" not in warnings and weight:
        warnings.append("unknown_format")
    card = info.card_data
    license_id = _card_value(card, "license")
    if not license_id:
        warnings.append("no_license")
    if info.gated:
        warnings.append("gated")
    if base_modified and info.last_modified and info.last_modified < base_modified:
        warnings.append("stale")
    candidate = {
        "source": "hub",
        "status": "candidate",
        "model_name": info.id,
        "trigger": _card_value(card, "instance_prompt"),
        "license": license_id,
        "gated": bool(info.gated),
        "downloads": info.downloads or 0,
        "likes": getattr(info, "likes", 0) or 0,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "format": form,
        "warnings": warnings,
    }
    if weight is None:
        candidate["weights"] = weights
    else:
        candidate["as_lora"] = {
            "model_name": info.id,
            "weight_name": weight,
            "revision": info.sha,
            "scale": 1.0,
        }
    return candidate


def hub_candidates(bases, query, terms, limit, rejected, api):
    """Up to `limit` candidates for `bases`, most downloaded first, plus a
    `rejected` row for each catalog-rejected repo the search turned up."""
    found, base_of = {}, {}
    for base in bases:
        for search in _searches(query, terms):
            for info in api.list_models(
                filter=f"base_model:adapter:{base}",
                search=search,
                sort="downloads",
                limit=limit * 3,
                expand=SEARCH_EXPAND,
            ):
                if not is_repo_id(info.id) or info.id in found:
                    continue
                found[info.id] = info
                base_of[info.id] = base
    base_modified = {}
    for base in bases:
        try:
            base_modified[base] = api.model_info(base).last_modified
        except Exception as error:
            logger.info(f"Base {base} last-modified unreadable: {error}")
    results, offered = [], 0
    for info in sorted(found.values(), key=lambda i: -(i.downloads or 0)):
        if offered >= limit:
            break
        if info.id in rejected:
            results.append(
                {
                    "source": "hub",
                    "status": "rejected",
                    "model_name": info.id,
                    "reason": rejected[info.id],
                }
            )
            continue
        candidate = _inspect(api, info, base_modified.get(base_of[info.id]))
        if candidate is not None:
            results.append(candidate)
            offered += 1
    return results


def search_hub(bases, query, terms, limit, rejected, api=None, timeout=HUB_TIMEOUT):
    """`(results, hub_error)`. Never raises: an unreachable, rate-limited or
    slow Hub is reported, and the caller still answers with the catalog. A
    second concurrent call answers BUSY_ERROR."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if not _IN_FLIGHT.acquire(blocking=False):
        return [], BUSY_ERROR
    outcome = {}

    def run():
        try:
            outcome["results"] = hub_candidates(
                bases, query, terms, limit, rejected, api
            )
        except Exception as error:
            outcome["error"] = f"Hub search failed: {type(error).__name__}: {error}"
        finally:
            _IN_FLIGHT.release()

    # A daemon thread: a hung Hub request must not hold up server shutdown
    try:
        worker = threading.Thread(target=run, name="lora-hub-search", daemon=True)
        worker.start()
    except Exception as error:
        _IN_FLIGHT.release()
        return [], f"Hub search failed: {type(error).__name__}: {error}"
    except BaseException:
        _IN_FLIGHT.release()
        raise
    worker.join(timeout)
    if worker.is_alive():
        return [], f"Hub search timed out after {timeout:g} s"
    if "error" in outcome:
        return [], outcome["error"]
    return outcome["results"], None
