"""check_narration: whether the speech matched the script, take by take.

A long narration is spoken in takes and joined (`templates/narration-long`),
and a take can loop, skip or stop early without anything saying so. This
transcribes the joined audio once, with the same cached ASR pipeline
`transcribe_audio` uses, aligns the words heard to the words of the takes in
order (`difflib.SequenceMatcher`, the alignment `check_script` uses for
lines) and attributes every dropped or inserted word to the take it falls in,
so the takes to re-speak are named.

Two rules place an inserted word. One that replaces expected words (a word
misheard) belongs to the take of the words it replaces, so a misheard last
word is blamed on its own take. A pure insertion belongs to the take whose
expected words it sits in front of (the last take when it comes after them
all), so a filler between two takes counts against the later one; a looped
take therefore shows as insertions in the take that follows the loop's start,
and the takes after it still align.
"""

import difflib
import re

from ..events import emit_warning
from .audio_transcription import (
    model_source,
    required_weights,  # noqa: F401 - re-exported for the registry hook
    transcribe_audio,
)
from .audio_utils import waveform_and_rate
from .text_takes import split_takes

DEFAULT_MAX_WORD_ERROR_RATE = 0.1
# How many dropped words a warning quotes
_QUOTED_WORDS = 5


def normalize(text):
    """The words of a text: lowercase, punctuation stripped."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def align(expected_tokens, heard_tokens):
    """Pair expected and heard tokens in order.

    Returns (kind, expected index or None, heard index or None) per token:
    "equal" for a match, "dropped" for an expected word not heard, "inserted"
    for a heard word not expected. A replaced stretch is its drops then its
    insertions.
    """
    pairs = []
    matcher = difflib.SequenceMatcher(
        None, expected_tokens, heard_tokens, autojunk=False
    )
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            pairs.extend(("equal", i1 + k, j1 + k) for k in range(i2 - i1))
            continue
        pairs.extend(("dropped", i, None) for i in range(i1, i2))
        # an insertion keeps the expected position it sits at, in `i1`
        pairs.extend(("inserted", None, j) for j in range(j1, j2))
    return pairs


def _owners(pairs, expected_tokens, heard_tokens, starts):
    """Per take, the dropped and inserted words the alignment attributes."""
    dropped = [[] for _ in starts]
    inserted = [[] for _ in starts]
    count = len(expected_tokens)

    def take_at(position):
        position = min(position, count - 1) if count else 0
        owner = 0
        for index, start in enumerate(starts):
            if start <= position:
                owner = index
        return owner

    cursor = 0  # the expected position the next inserted word sits in front of
    replaced = None  # the take of the words the current replace dropped
    for kind, i, j in pairs:
        if kind == "inserted":
            if replaced is not None:
                owner = replaced
            elif cursor >= count:
                owner = len(starts) - 1
            else:
                owner = take_at(cursor)
            inserted[owner].append(heard_tokens[j])
            continue
        replaced = None
        if kind == "dropped":
            replaced = take_at(i)
            dropped[replaced].append(expected_tokens[i])
        cursor = i + 1
    return dropped, inserted


def _asr(waveform, rate, device, model_name):
    """The production transcriber: `transcribe_audio` on the shared cache."""
    return transcribe_audio(waveform, device, rate, model_name=model_name)


def check_narration(
    takes=None,
    audio=None,
    device="cpu",
    sample_rate=None,
    model_name=None,
    max_word_error_rate=DEFAULT_MAX_WORD_ERROR_RATE,
    transcribe=None,
    text=None,
):
    """Task command: check that a narration speaks its script, take by take.

    Args:
        takes: The takes the narration was spoken from, in order: a list of
            strings, or one string, split as `split_takes` splits (give this
            or `text`)
        audio: Path or URL of the narration (or of a video whose soundtrack
            is taken), or a waveform (which needs sample_rate)
        device: Target device ("cuda", "mps", "cpu")
        sample_rate: Sample rate of a waveform passed directly
        model_name: A Whisper-class ASR model (default: the one
            `transcribe_audio` uses)
        max_word_error_rate: A take whose dropped plus inserted words, over
            its expected words, exceed this fails (0 to 1)
        transcribe: A callable (waveform, rate, device, model_name) -> text
            standing in for the ASR; for tests
        text: The script as one string, split into takes; the alternative
            to `takes`

    Returns:
        {takes, checked, failed, worst}: per take {index, text, words,
        dropped, inserted, word_error_rate, ok}; how many were checked; the
        failed takes' indices; and the worst take's index and rate (None
        with no takes)

    Raises:
        ValueError: If neither or both of `takes` and `text` are given, or the
            audio is missing
    """
    if (takes is None) == (text is None):
        raise ValueError(
            "check_narration needs exactly one of 'text' or 'takes' - the "
            "script as one string, or as the takes it was spoken in"
        )
    if audio is None:
        raise ValueError("check_narration needs 'audio', the narration to check")
    if takes is None:
        takes = text
    if isinstance(takes, str):
        takes = split_takes(takes)
    takes = [str(take) for take in takes]

    waveform, rate = waveform_and_rate(audio, sample_rate, "check_narration")
    model = model_source({"model_name": model_name})
    heard = (transcribe or _asr)(waveform, rate, device, model)

    per_take = [normalize(take) for take in takes]
    expected = [word for words in per_take for word in words]
    starts, position = [], 0
    for words in per_take:
        starts.append(position)
        position += len(words)
    heard_tokens = normalize(heard or "")
    dropped, inserted = _owners(
        align(expected, heard_tokens), expected, heard_tokens, starts
    )

    results = []
    for index, words in enumerate(per_take):
        error_rate = (len(dropped[index]) + len(inserted[index])) / max(1, len(words))
        results.append(
            {
                "index": index,
                "text": takes[index],
                "words": len(words),
                "dropped": dropped[index],
                "inserted": inserted[index],
                "word_error_rate": round(error_rate, 4),
                "ok": error_rate <= max_word_error_rate,
            }
        )
    failed = [r["index"] for r in results if not r["ok"]]
    for r in (results[i] for i in failed):
        shown = ", ".join(r["dropped"][:_QUOTED_WORDS]) or "none"
        emit_warning(
            f"check_narration: take {r['index']} does not match the audio "
            f"(word error rate {r['word_error_rate']:.2f}, {len(r['dropped'])} "
            f"dropped - first: {shown} - and {len(r['inserted'])} inserted); "
            f"re-speak it",
            kind="narration_take_mismatch",
            command="check_narration",
            take=r["index"],
            word_error_rate=r["word_error_rate"],
        )
    worst = max(results, key=lambda r: r["word_error_rate"], default=None)
    return {
        "takes": results,
        "checked": len(results),
        "failed": failed,
        "worst": (
            {"index": worst["index"], "word_error_rate": worst["word_error_rate"]}
            if worst
            else None
        ),
    }


def _register():
    from .registry import register_command
    from ..task_domains import UNIT

    @register_command(
        "check_narration",
        implementation="dw.tasks.narration_check.check_narration",
        returns="json",
        required_weights="dw.tasks.narration_check.required_weights",
        consumes_device=True,
        domains={"max_word_error_rate": UNIT},
    )
    def _handle_check_narration(task, arguments, previous_pipelines):
        """Check that a narration speaks its script, take by take"""
        return check_narration(device=task.device_for(arguments), **arguments)


_register()
