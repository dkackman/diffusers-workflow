"""Tests for #609: check_script - whether a take speaks its script.

Pure-function tests on synthetic word chunks and synthetic waveforms; no ASR
model is loaded. `check_script()` itself runs with `transcribe_audio` patched.
"""

import unittest
from unittest.mock import patch

import numpy

from dw.introspection import list_tasks
from dw.task_domains import script_lines_errors, task_argument_errors
from dw.tasks import script_check
from dw.tasks.script_check import (
    align,
    check,
    check_script,
    guard_words,
    line_similarity,
    normalize_words,
    parse_lines,
    strip_markup,
    tail_level,
    word_level,
)
from dw.tasks.task import _COMMAND_INFO, _COMMAND_REGISTRY
from dw.workflow import Workflow

RATE = 16000


def voiced(seconds, amplitude=0.1):
    t = numpy.arange(int(round(seconds * RATE)), dtype=numpy.float32) / RATE
    return (amplitude * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32)


def silent(seconds):
    return numpy.zeros(int(round(seconds * RATE)), dtype=numpy.float32)


def words(text, start=0.0, step=0.5, length=0.4):
    """Word chunks for `text`, one per whitespace-separated word."""
    return [
        {
            "start": start + i * step,
            "end": start + i * step + length,
            "text": f" {word}",
        }
        for i, word in enumerate(text.split())
    ]


def run(lines, chunks, mono=None, similarity=0.85):
    if mono is None:
        end = max((c["end"] for c in chunks), default=1.0)
        mono = voiced(end + 1.0)
    return check(parse_lines(lines), chunks, mono, RATE, similarity)


def rules(answer, name):
    return [f for f in answer["findings"] if f["rule"] == name]


class TestNormalize(unittest.TestCase):
    def test_lowercase_punctuation_and_apostrophes(self):
        self.assertEqual(normalize_words("Hello, World!"), ["hello", "world"])
        self.assertEqual(normalize_words("Don’t 'quote' it"), ["don't", "quote", "it"])
        self.assertEqual(normalize_words("well-known"), ["well", "known"])
        self.assertEqual(normalize_words("♪ ..."), [])


class TestStripMarkup(unittest.TestCase):
    def test_d_tag_with_language(self):
        text, tokens = strip_markup("<d>[English] Hello there</d>")
        self.assertEqual(text, "Hello there")
        self.assertEqual(tokens, ["d", "english", "d"])

    def test_each_token(self):
        cases = {
            "Hi <scenetrans> there": ("Hi there", ["scenetrans"]),
            "Hi <cutoff>": ("Hi", ["cutoff"]),
            "Hi [unclear] there": ("Hi there", ["unclear"]),
            "(S1) Hi": ("Hi", ["s1"]),
            "(S1,S2) Hi": ("Hi", ["s1", "s2"]),
            "(S1, S2) Hi": ("Hi", ["s1", "s2"]),
        }
        for line, expected in cases.items():
            with self.subTest(line=line):
                self.assertEqual(strip_markup(line), expected)

    def test_plain_parenthetical_stays_text(self):
        self.assertEqual(
            strip_markup("Hello (quietly) there"), ("Hello (quietly) there", [])
        )


class TestParseLines(unittest.TestCase):
    def test_accepts_strings_and_text_objects(self):
        parsed = parse_lines(["Hello <cutoff>", {"text": "(S1) World"}])
        self.assertEqual(
            parsed,
            [
                {"text": "Hello", "tokens": ["cutoff"]},
                {"text": "World", "tokens": ["s1"]},
            ],
        )

    def test_empty_list_is_valid(self):
        self.assertEqual(parse_lines([]), [])

    def test_refusals_name_lines(self):
        bad = [
            "oops",
            {"text": "x"},
            None,
            [""],
            ["   "],
            [3],
            [{"other": "x"}],
            [{}],
            [{"text": "x", "speaker": 1}],
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    parse_lines(value)
                self.assertIn("lines", str(caught.exception))


class TestLinesErrors(unittest.TestCase):
    """The validate-time `lines` check, through task_domains' one walk."""

    @staticmethod
    def errors(lines):
        return task_argument_errors(
            {
                "steps": [
                    {
                        "name": "c",
                        "task": {
                            "command": "check_script",
                            "arguments": {"audio": "x.wav", "lines": lines},
                        },
                    }
                ]
            }
        )

    def test_malformed_lines_is_one_error_at_lines(self):
        errors = self.errors("oops")
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0]["path"].endswith(".lines"), errors[0]["path"])
        self.assertTrue(errors[0]["message"].endswith("."), errors[0]["message"])

    def test_valid_list_has_no_errors(self):
        self.assertEqual(self.errors(["a b", {"text": "c"}]), [])

    def test_reference_string_is_skipped(self):
        self.assertEqual(self.errors("variable:lines"), [])

    def test_literal_with_a_colon_is_not_a_reference(self):
        self.assertEqual(len(self.errors("Hello: world")), 1)

    def test_lines_omitted_is_left_to_the_signature(self):
        self.assertEqual(script_lines_errors({"audio": "x.wav"}), [])


class TestWorkflowValidation(unittest.TestCase):
    def errors(self, **arguments):
        definition = {
            "id": "check",
            "steps": [
                {
                    "name": "c",
                    "task": {
                        "command": "check_script",
                        "arguments": {"audio": "asset:x.wav", **arguments},
                    },
                }
            ],
        }
        return Workflow(definition, "outputs", None).validation_errors()

    def test_malformed_lines_is_refused(self):
        errors = self.errors(lines="oops")
        self.assertTrue(
            any(e["path"].endswith("arguments.lines") for e in errors), errors
        )

    def test_similarity_outside_unit_range_is_refused(self):
        errors = self.errors(lines=["a b"], similarity=1.5)
        self.assertTrue(any("similarity" in e["path"] for e in errors), errors)


class TestLevels(unittest.TestCase):
    def test_word_level_silence_is_none_and_signal_is_a_level(self):
        mono = numpy.concatenate([silent(1.0), voiced(1.0)])
        self.assertIsNone(word_level(mono, RATE, 0.2, 0.6))
        level = word_level(mono, RATE, 1.2, 1.6)
        self.assertGreater(level, -30)
        self.assertLess(level, 0)

    def test_zero_length_span_is_measured_as_one_window(self):
        mono = voiced(1.0)
        self.assertIsNotNone(word_level(mono, RATE, 0.5, 0.5))

    def test_tail_level(self):
        self.assertIsNone(tail_level(silent(1.0), RATE))
        self.assertGreater(tail_level(voiced(1.0), RATE), -30)


class TestGuardWords(unittest.TestCase):
    def test_silent_word_is_discarded_with_none_level(self):
        mono = numpy.concatenate([voiced(1.0), silent(1.0)])
        chunks = words("real", 0.0) + words("ghost", 1.2)
        heard, discarded = guard_words(chunks, mono, RATE)
        self.assertEqual([h["word"] for h in heard], ["real"])
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0]["text"], "ghost")
        self.assertIsNone(discarded[0]["level_dbfs"])

    def test_quiet_but_above_floor_stays_heard(self):
        # 0.002 amplitude sine is about -57 dBFS rms, above the -65 floor
        mono = voiced(2.0, amplitude=0.002)
        heard, discarded = guard_words(words("soft words"), mono, RATE)
        self.assertEqual(len(heard), 2)
        self.assertEqual(discarded, [])

    def test_very_quiet_signal_is_discarded_with_a_level(self):
        mono = voiced(2.0, amplitude=0.0002)  # about -77 dBFS
        heard, discarded = guard_words(words("faint"), mono, RATE)
        self.assertEqual(heard, [])
        self.assertLess(discarded[0]["level_dbfs"], -65)

    def test_chunks_that_normalize_to_nothing_are_dropped(self):
        chunks = [
            {"start": 0.0, "end": 0.4, "text": " ♪"},
            {"start": 0.5, "end": 0.9, "text": " ..."},
        ]
        heard, discarded = guard_words(chunks, voiced(2.0), RATE)
        self.assertEqual((heard, discarded), ([], []))

    def test_silent_word_is_not_a_finding_or_aligned(self):
        mono = numpy.concatenate([voiced(1.0), silent(1.0)])
        chunks = words("hello", 0.0) + words("ghost", 1.2)
        answer = run(["hello"], chunks, mono)
        self.assertEqual(answer["findings"], [])
        self.assertEqual(answer["lines"][0]["heard"], "hello")
        self.assertEqual(answer["unmatched"], [])
        self.assertEqual(len(answer["discarded"]), 1)


class TestRepetitionGuard(unittest.TestCase):
    def test_loud_loop_is_discarded_as_repetition(self):
        chunks = [
            {"start": i * 0.1, "end": i * 0.1, "text": " Pre" if i == 0 else "-pre"}
            for i in range(40)
        ]
        heard, discarded = guard_words(chunks, voiced(5.0), RATE)
        self.assertEqual(heard, [])
        self.assertEqual(len(discarded), 40)
        self.assertEqual({d["reason"] for d in discarded}, {"repetition"})

    def test_loop_over_music_is_not_speech_where_silent(self):
        chunks = words(" ".join(["pre"] * 30), step=0.1, length=0.0)
        answer = run([], chunks, voiced(5.0))
        self.assertEqual(answer["findings"], [])
        self.assertEqual(len(answer["discarded"]), 30)

    def test_short_real_repetition_stays_heard(self):
        chunks = words("no no no no no stop")
        heard, discarded = guard_words(chunks, voiced(4.0), RATE)
        self.assertEqual(len(heard), 6)
        self.assertEqual(discarded, [])

    def test_period_two_loop_is_discarded(self):
        chunks = words(" ".join(["thank you"] * 6), step=0.3, length=0.2)
        heard, discarded = guard_words(chunks, voiced(5.0), RATE)
        self.assertEqual(heard, [])
        self.assertEqual(len(discarded), 12)

    def test_loop_does_not_take_the_speech_around_it(self):
        chunks = (
            words("hello there", 0.0)
            + words(" ".join(["pre"] * 10), 1.0, step=0.1, length=0.05)
            + words("good night", 2.5)
        )
        heard, discarded = guard_words(chunks, voiced(4.0), RATE)
        self.assertEqual(
            [h["word"] for h in heard], ["hello", "there", "good", "night"]
        )
        self.assertEqual(len(discarded), 10)

    def test_loop_inside_one_chunk_is_discarded(self):
        chunks = [{"start": 0.0, "end": 3.0, "text": " Pre-pre-pre-pre-pre-pre-pre"}]
        heard, discarded = guard_words(chunks, voiced(4.0), RATE)
        self.assertEqual(heard, [])
        self.assertEqual(discarded[0]["reason"], "repetition")

    def test_below_floor_reason(self):
        mono = numpy.concatenate([voiced(1.0), silent(1.0)])
        _, discarded = guard_words(words("real", 0.0) + words("ghost", 1.2), mono, RATE)
        self.assertEqual(discarded[0]["reason"], "below_floor")

    def test_repeat_thresholds_reported(self):
        self.assertEqual(
            script_check.THRESHOLDS["repeat_run_min"], script_check.REPEAT_RUN_MIN
        )
        self.assertEqual(
            script_check.THRESHOLDS["repeat_max_period"],
            script_check.REPEAT_MAX_PERIOD,
        )


class TestAlignAndSimilarity(unittest.TestCase):
    @staticmethod
    def heard(text):
        return [{"word": w} for w in text.split()]

    def test_align_exact(self):
        assigned, unassigned = align([["a", "b"], ["c"]], self.heard("a b c"))
        self.assertEqual(assigned, [[0, 1], [2]])
        self.assertEqual(unassigned, [])

    def test_align_insert_between_lines_is_unassigned_with_neighbours(self):
        assigned, unassigned = align([["a", "b"], ["c"]], self.heard("a b x c"))
        self.assertEqual(assigned, [[0, 1], [2 + 1]])
        self.assertEqual(unassigned, [(2, (0, 1))])

    def test_align_insert_inside_a_line_belongs_to_it(self):
        assigned, unassigned = align([["a", "b", "c"]], self.heard("a x b c"))
        self.assertEqual(assigned, [[0, 1, 2, 3]])
        self.assertEqual(unassigned, [])

    def test_align_insert_at_the_edges_has_one_neighbour(self):
        _, unassigned = align([["a"]], self.heard("x a y"))
        self.assertEqual(unassigned, [(0, (0,)), (2, (0,))])

    def test_line_similarity(self):
        self.assertEqual(line_similarity([], []), 1.0)
        self.assertEqual(line_similarity(["a"], []), 0.0)
        self.assertEqual(line_similarity(["a", "b"], ["a", "b"]), 1.0)
        self.assertAlmostEqual(line_similarity(["a", "b"], ["a", "c"]), 0.5)


class TestCheckAlignment(unittest.TestCase):
    LINES = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india"]

    def test_exact_lines_have_no_findings(self):
        chunks = words("alpha bravo charlie delta echo foxtrot golf hotel india")
        answer = run(self.LINES, chunks)
        self.assertEqual(answer["findings"], [])
        self.assertEqual([x["similarity"] for x in answer["lines"]], [1.0] * 3)
        self.assertEqual(answer["unmatched"], [])
        self.assertEqual(answer["rules_applied"], list(script_check.LINE_RULES))
        self.assertEqual(
            [s["rule"] for s in answer["rules_skipped"]], ["speech_where_silent"]
        )

    def test_dropped_line_is_a_mismatch_at_previous_line_end(self):
        chunks = words("alpha bravo charlie", 0.0) + words("golf hotel india", 5.0)
        answer = run(self.LINES, chunks)
        dropped = answer["lines"][1]
        self.assertEqual(dropped["heard"], "")
        self.assertEqual(dropped["similarity"], 0)
        found = rules(answer, "line_mismatch")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["line"], 1)
        self.assertAlmostEqual(found[0]["at"]["seconds"], 1.4, places=2)
        self.assertEqual(answer["lines"][0]["similarity"], 1.0)
        self.assertEqual(answer["lines"][2]["similarity"], 1.0)

    def test_added_line_hurts_no_neighbour(self):
        chunks = (
            words("alpha bravo charlie", 0.0)
            + words("zulu yankee", 3.0)
            + words("delta echo foxtrot", 5.0)
            + words("golf hotel india", 8.0)
        )
        answer = run(self.LINES, chunks)
        self.assertEqual(rules(answer, "line_mismatch"), [])
        self.assertEqual([x["similarity"] for x in answer["lines"]], [1.0] * 3)
        self.assertEqual([u["text"] for u in answer["unmatched"]], ["zulu", "yankee"])

    def test_reordered_line_is_exactly_one_mismatch(self):
        chunks = words("alpha bravo charlie golf hotel india delta echo foxtrot")
        answer = run(self.LINES, chunks)
        self.assertEqual(len(rules(answer, "line_mismatch")), 1)

    def test_swapped_line_is_a_mismatch_at_its_first_heard_word(self):
        chunks = (
            words("alpha bravo charlie", 0.0)
            + words("kilo lima mike", 3.0)
            + words("golf hotel india", 6.0)
        )
        answer = run(self.LINES, chunks)
        found = rules(answer, "line_mismatch")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["line"], 1)
        self.assertEqual(found[0]["at"]["seconds"], 3.0)
        self.assertEqual(found[0]["at"]["word"], " kilo".strip())


class TestCheckMarkup(unittest.TestCase):
    def test_tag_not_counted_against_similarity(self):
        for line in ("Hello there <cutoff>", "Hello [unclear] there"):
            with self.subTest(line=line):
                answer = run([line], words("hello there"))
                self.assertEqual(answer["lines"][0]["similarity"], 1.0)
                self.assertEqual(answer["findings"], [])

    def test_spoken_tag_does_not_lower_similarity(self):
        plain = "We should and call the landlord before dark"
        tagged = "We should <pause> and call the landlord before dark."
        heard = words("we should pause and call the landlord before dark")
        answer = run([tagged], heard)
        self.assertEqual(answer["lines"][0]["similarity"], 1.0)
        self.assertEqual(rules(answer, "line_mismatch"), [])
        self.assertEqual(len(rules(answer, "tag_spoken")), 1)
        self.assertLess(run([plain], heard)["lines"][0]["similarity"], 1.0)

    def test_tag_spoken_inside_a_line(self):
        answer = run(["Hello [unclear] there"], words("hello unclear there"))
        found = rules(answer, "tag_spoken")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["value"], "unclear")
        self.assertEqual(found[0]["at"]["line"], 0)

    def test_tag_spoken_at_a_lines_end(self):
        chunks = words("hello there cutoff", 0.0) + words("next line", 3.0)
        answer = run(["Hello there <cutoff>", "next line"], chunks)
        found = rules(answer, "tag_spoken")
        self.assertEqual([f["value"] for f in found], ["cutoff"])
        self.assertEqual(found[0]["at"]["line"], 0)

    def test_tag_spoken_at_the_very_end(self):
        answer = run(["Hello there <cutoff>"], words("hello there cutoff"))
        self.assertEqual(len(rules(answer, "tag_spoken")), 1)

    def test_tag_spoken_at_the_very_start(self):
        answer = run(["(S1) Hello there"], words("s1 hello there"))
        self.assertEqual([f["value"] for f in rules(answer, "tag_spoken")], ["s1"])

    def test_token_that_is_a_real_word_of_the_line_is_not_flagged(self):
        answer = run(["[English] I speak english well"], words("i speak english well"))
        self.assertEqual(rules(answer, "tag_spoken"), [])

    def test_h3_markup_words(self):
        line = "<d>[English] Hello there</d> <scenetrans> (S1,S2) fine"
        parsed = parse_lines([line])[0]
        self.assertEqual(parsed["text"], "Hello there fine")
        self.assertEqual(
            set(parsed["tokens"]), {"d", "english", "scenetrans", "s1", "s2"}
        )


class TestClippedTail(unittest.TestCase):
    def test_voiced_tail_with_last_word_in_it_is_clipped(self):
        mono = voiced(2.0)
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 1.6, "end": 1.95, "text": " world"},
        ]
        answer = run(["hello world"], chunks, mono)
        found = rules(answer, "line_clipped_at_end")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["line"], 0)
        self.assertEqual(found[0]["at"]["seconds"], 1.95)

    def test_silent_final_quarter_second_is_not_clipped(self):
        mono = numpy.concatenate([voiced(1.7), silent(0.3)])
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 1.3, "end": 1.9, "text": " world"},
        ]
        answer = run(["hello world"], chunks, mono)
        self.assertEqual(rules(answer, "line_clipped_at_end"), [])

    def test_last_word_ending_well_before_the_tail_is_not_clipped(self):
        mono = voiced(3.0)
        answer = run(["hello world"], words("hello world"), mono)
        self.assertEqual(rules(answer, "line_clipped_at_end"), [])


class TestEmptyLines(unittest.TestCase):
    def test_voiced_words_are_one_speech_where_silent(self):
        answer = run([], words("one two three"))
        self.assertEqual(len(answer["findings"]), 1)
        finding = answer["findings"][0]
        self.assertEqual(finding["rule"], "speech_where_silent")
        self.assertEqual(finding["value"], 3)
        self.assertEqual(answer["rules_applied"], ["speech_where_silent"])
        self.assertEqual(
            [s["rule"] for s in answer["rules_skipped"]],
            list(script_check.LINE_RULES),
        )
        self.assertEqual(answer["lines"], [])

    def test_words_only_over_silence_are_discarded_not_found(self):
        answer = run([], words("one two"), silent(3.0))
        self.assertEqual(answer["findings"], [])
        self.assertEqual(len(answer["discarded"]), 2)


class TestRegistration(unittest.TestCase):
    def test_registered_as_json_command_without_assessment_flag(self):
        self.assertIn("check_script", _COMMAND_REGISTRY)
        self.assertIn("check_script", list_tasks()["commands"])
        self.assertNotIn("check_script", list_tasks()["assessment"])
        info = _COMMAND_INFO["check_script"]
        self.assertEqual(info["returns"], "json")
        self.assertNotIn("assessment", info)


class TestCheckScriptTask(unittest.TestCase):
    def test_transcribes_with_word_timestamps_and_returns_the_answer(self):
        canned = {
            "text": " hello there",
            "chunks": words("hello there"),
        }
        waveform = voiced(2.0)[numpy.newaxis, :]
        with patch(
            "dw.tasks.audio_transcription.transcribe_audio", return_value=canned
        ) as transcribe:
            answer = check_script(
                waveform,
                ["Hello there"],
                model_name="openai/whisper-tiny",
                sample_rate=RATE,
            )
        kwargs = transcribe.call_args.kwargs
        self.assertEqual(kwargs["timestamps"], "word")
        self.assertEqual(kwargs["model_name"], "openai/whisper-tiny")
        for key in (
            "findings",
            "lines",
            "discarded",
            "transcript",
            "rules_applied",
            "rules_skipped",
        ):
            self.assertIn(key, answer)
        self.assertEqual(answer["transcript"], " hello there")
        self.assertEqual(answer["findings"], [])
        self.assertEqual(answer["lines"][0]["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
