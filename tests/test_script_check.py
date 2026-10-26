"""Tests for #609: check_script - whether a take speaks its script.

Pure-function tests on synthetic word chunks and synthetic waveforms; no ASR
model is loaded. `check_script()` itself runs with `transcribe_audio` patched.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from dw.introspection import list_tasks
from dw.task_domains import task_argument_errors
from dw.task_problems import script_lines_errors
from dw.script_lines import (
    normalize_words,
    disambiguated,
    parse_lines,
    parse_shots,
    shot_names_error,
    with_full_shot_names,
    strip_markup,
)
from dw.tasks import script_check
from dw.tasks.script_check import (
    align,
    check,
    check_script,
    guard_words,
    line_similarity,
    overlapping_shot,
    shot_spans,
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


def run(lines, chunks, mono=None, similarity=0.85, spans=None, shots_source="none"):
    if mono is None:
        end = max((c["end"] for c in chunks), default=1.0)
        mono = voiced(end + 0.5)
    return check(
        parse_lines(lines),
        chunks,
        mono,
        RATE,
        similarity,
        spans=spans,
        shots_source=shots_source,
    )


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
                {
                    "text": "Hello",
                    "tokens": ["cutoff"],
                    "shot": None,
                    "start_frame": None,
                },
                {"text": "World", "tokens": ["s1"], "shot": None, "start_frame": None},
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
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0]["reason"], "repetition")
        self.assertEqual(discarded[0]["count"], 40)
        self.assertEqual(discarded[0]["start"], 0.0)
        self.assertEqual(discarded[0]["end"], 3.9)

    def test_loop_over_music_is_not_speech_where_silent(self):
        chunks = words(" ".join(["pre"] * 30), step=0.1, length=0.0)
        answer = run([], chunks, voiced(5.0))
        self.assertEqual(answer["findings"], [])
        self.assertEqual(len(answer["discarded"]), 1)
        self.assertEqual(answer["discarded"][0]["count"], 30)

    def test_short_real_repetition_stays_heard(self):
        chunks = words("no no no no no stop")
        heard, discarded = guard_words(chunks, voiced(4.0), RATE)
        self.assertEqual(len(heard), 6)
        self.assertEqual(discarded, [])

    def test_period_two_loop_is_discarded(self):
        chunks = words(" ".join(["thank you"] * 6), step=0.3, length=0.2)
        heard, discarded = guard_words(chunks, voiced(5.0), RATE)
        self.assertEqual(heard, [])
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0]["count"], 12)

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
        self.assertEqual(len(discarded), 1)
        self.assertEqual(discarded[0]["count"], 10)

    def test_loop_inside_one_chunk_is_discarded(self):
        chunks = [{"start": 0.0, "end": 3.0, "text": " Pre-pre-pre-pre-pre-pre-pre"}]
        heard, discarded = guard_words(chunks, voiced(4.0), RATE)
        self.assertEqual(heard, [])
        self.assertEqual(discarded[0]["reason"], "repetition")

    def test_loop_over_a_scripted_line_keeps_its_first_take(self):
        # #961: Whisper looped on the line it heard; the first take is real
        chunks = words("how", 0.0) + words(
            " ".join(["the lid is gone"] * 8), 1.0, step=0.3, length=0.2
        )
        answer = run(["the lid is gone"], chunks, voiced(12.0))
        self.assertEqual(answer["lines"][0]["heard"], "the lid is gone")
        self.assertEqual(answer["lines"][0]["similarity"], 1.0)
        self.assertEqual(len(answer["discarded"]), 1)
        self.assertEqual(answer["discarded"][0]["count"], 28)
        loops = [f for f in answer["findings"] if f["rule"] == "transcript_loop"]
        self.assertEqual(len(loops), 1)
        self.assertEqual(loops[0]["value"], 32)
        self.assertEqual(loops[0]["at"]["seconds"], 1.0)

    def test_line_after_a_loop_is_not_transcribed_not_a_zero(self):
        chunks = words("how", 0.0) + words(
            " ".join(["the lid is gone"] * 8), 1.0, step=0.3, length=0.2
        )
        answer = run(["the lid is gone", "i wore it as a hat"], chunks, voiced(12.0))
        line = answer["lines"][1]
        self.assertIsNone(line["heard"])
        self.assertIsNone(line["similarity"])
        self.assertEqual(line["not_transcribed"], "transcript_loop")
        rules = [f["rule"] for f in answer["findings"]]
        self.assertNotIn("line_mismatch", rules)
        self.assertIn("transcript_loop", rules)

    def test_loop_of_unscripted_words_keeps_nothing_and_is_no_finding(self):
        chunks = words(" ".join(["pre"] * 10), 1.0, step=0.1, length=0.05)
        answer = run(["hello"], chunks, voiced(4.0))
        self.assertEqual(answer["lines"][0]["heard"], "")
        self.assertFalse(
            [f for f in answer["findings"] if f["rule"] == "transcript_loop"]
        )

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
            [s["rule"] for s in answer["rules_skipped"]],
            ["speech_where_silent", "speech_in_silent_shot", "unscripted_speech"],
        )
        self.assertTrue(all(s["reason"] for s in answer["rules_skipped"]))

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
            list(script_check.LINE_RULES + script_check.SHOT_RULES),
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
        waveform = voiced(1.4)[numpy.newaxis, :]
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


SHOTS = [
    {"name": "A", "start": 0.0, "end": 3.0},
    {"name": "B", "start": 3.0, "end": 6.0},
]


def shot_run(lines, chunks, mono=None, spans=SHOTS, source="argument"):
    return run(lines, chunks, mono, spans=spans, shots_source=source)


def skipped(answer, name):
    return [s for s in answer["rules_skipped"] if s["rule"] == name]


class TestShotPlacement(unittest.TestCase):
    def test_tagged_lines_carry_their_shot(self):
        chunks = words("alpha bravo", 0.0) + words("delta echo", 3.5)
        answer = shot_run(
            [{"text": "alpha bravo", "shot": "A"}, {"text": "delta echo", "shot": "B"}],
            chunks,
            mono=numpy.concatenate([voiced(0.9), silent(2.6), voiced(1.0)]),
        )
        self.assertEqual([x["shot"] for x in answer["lines"]], ["A", "B"])
        self.assertEqual(answer["findings"], [])

    def test_untagged_lines_take_the_shot_their_words_overlap_most(self):
        chunks = words("alpha bravo", 0.0) + words("delta echo", 3.5)
        answer = shot_run(["alpha bravo", "delta echo"], chunks)
        self.assertEqual([x["shot"] for x in answer["lines"]], ["A", "B"])

    def test_no_spans_leaves_shot_none(self):
        answer = run(["alpha bravo"], words("alpha bravo"))
        self.assertIsNone(answer["lines"][0]["shot"])
        self.assertIsNone(answer["shots"])
        self.assertEqual(answer["shots_source"], "none")

    def test_overlapping_shot(self):
        self.assertEqual(overlapping_shot(SHOTS, 2.0, 4.5), "B")
        self.assertEqual(overlapping_shot(SHOTS, 2.0, 3.5), "A")
        self.assertEqual(overlapping_shot(SHOTS, 3.5, 3.5), "B")
        self.assertIsNone(overlapping_shot(SHOTS, 7.0, 8.0))

    def test_answer_reports_rounded_shots_and_source(self):
        spans = [{"name": "A", "start": 0.123456, "end": 2.98765}]
        answer = shot_run(["alpha"], words("alpha"), spans=spans, source="artifact")
        self.assertEqual(answer["shots"], [{"name": "A", "start": 0.123, "end": 2.988}])
        self.assertEqual(answer["shots_source"], "artifact")

    def test_dropped_line_in_a_known_shot_is_at_the_shots_start(self):
        answer = shot_run(
            [
                {"text": "alpha bravo", "shot": "A"},
                {"text": "delta echo", "shot": "B"},
            ],
            words("alpha bravo", 0.0),
        )
        found = rules(answer, "line_mismatch")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["line"], 1)
        self.assertEqual(found[0]["at"]["seconds"], 3.0)
        self.assertEqual(found[0]["at"]["shot"], "B")


class TestUnscriptedSpeech(unittest.TestCase):
    def test_a_line_spoken_twice_in_its_shot_is_a_finding(self):
        chunks = words("alpha bravo charlie", 0.0) + words("alpha bravo charlie", 1.0)
        answer = shot_run([{"text": "alpha bravo charlie", "shot": "A"}], chunks)
        (found,) = rules(answer, "unscripted_speech")
        self.assertEqual(found["value"], 3)
        self.assertEqual(found["at"]["shot"], "A")
        self.assertTrue(found["at"]["repeats_line"])

    def test_a_doubled_line_inside_a_longer_cut_is_a_finding(self):
        # align alone keeps the second copy inside the line (similarity 0.75)
        chunks = (
            words("hal kettle froze", 0.0)
            + words("then wear socks", 6.3)
            + words("wear socks", 9.1)
            + words("the kettle is whistling hal", 11.0)
        )
        lines = [
            {"text": "Hal, the kettle froze.", "shot": "A"},
            {"text": "Then wear socks.", "shot": "B"},
            {"text": "The kettle is whistling, Hal.", "shot": "C"},
        ]
        spans = [
            {"name": "A", "start": 0.0, "end": 5.0},
            {"name": "B", "start": 5.0, "end": 10.3},
            {"name": "C", "start": 10.3, "end": 16.0},
        ]
        answer = shot_run(lines, chunks, mono=voiced(16.0), spans=spans)
        self.assertEqual(answer["lines"][1]["similarity"], 1.0)
        (found,) = rules(answer, "unscripted_speech")
        self.assertEqual(found["at"]["shot"], "B")
        self.assertTrue(found["at"]["repeats_line"])
        self.assertEqual([w["text"] for w in answer["unmatched"]], ["wear", "socks"])

    def test_a_misheard_repeat_scoring_as_the_line_does_is_a_repeat(self):
        # "hal" heard as "how" both times: 0.75 for the line, so also a repeat
        chunks = words("how kettle froze now", 0.0, 0.2, 0.15) + words(
            "how kettle froze now", 1.0, 0.2, 0.15
        )
        answer = run(
            [{"text": "hal kettle froze now", "shot": "A"}],
            chunks,
            voiced(6.0),
            similarity=0.6,
            spans=SHOTS,
            shots_source="argument",
        )
        self.assertEqual(answer["lines"][0]["similarity"], 0.75)
        (found,) = rules(answer, "unscripted_speech")
        self.assertTrue(found["at"]["repeats_line"])

    def test_an_ad_lib_in_a_scripted_shot_is_not_a_repeat(self):
        chunks = words("alpha bravo", 0.0) + words("zulu yankee", 1.0)
        answer = shot_run([{"text": "alpha bravo", "shot": "A"}], chunks)
        (found,) = rules(answer, "unscripted_speech")
        self.assertFalse(found["at"]["repeats_line"])

    def test_clean_take_has_none(self):
        answer = shot_run([{"text": "alpha bravo", "shot": "A"}], words("alpha bravo"))
        self.assertEqual(rules(answer, "unscripted_speech"), [])


class TestSpeechNotTranscribed(unittest.TestCase):
    LINES = [{"text": "alpha bravo charlie delta", "shot": "A"}]

    def test_sound_in_a_gap_between_a_lines_words_is_a_finding(self):
        # The ASR dropped an ad-lib between "bravo" and "charlie"
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 4.0)
        answer = shot_run(self.LINES, chunks, mono=voiced(5.0))
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertEqual(found["at"]["line"], 0)
        self.assertEqual(found["at"]["shot"], "A")
        self.assertAlmostEqual(found["at"]["seconds"], 0.9, places=2)
        self.assertAlmostEqual(found["at"]["end"], 4.0, places=2)

    def test_a_stretched_word_over_sound_is_a_finding(self):
        chunks = [
            {"start": 0.0, "end": 0.4, "text": " alpha"},
            {"start": 0.5, "end": 3.0, "text": " bravo"},
            {"start": 3.0, "end": 3.4, "text": " charlie"},
            {"start": 3.5, "end": 3.9, "text": " delta"},
        ]
        answer = shot_run(self.LINES, chunks, mono=voiced(4.0))
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertEqual(found["at"]["seconds"], 0.5)

    def test_a_quiet_gap_is_not(self):
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 4.0)
        mono = numpy.concatenate([voiced(0.9), silent(3.1), voiced(1.0)])
        answer = shot_run(self.LINES, chunks, mono=mono)
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    def test_a_short_gap_is_not(self):
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 1.5)
        answer = shot_run(self.LINES, chunks, mono=voiced(3.0))
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    TWO = [
        {"text": "alpha bravo", "shot": "A"},
        {"text": "charlie delta", "shot": "B"},
    ]

    def test_sound_between_two_lines_is_a_finding_in_its_shot(self):
        # #960: the untranscribed stretch sits outside both lines' spans;
        # #964: it is placed in the shot holding most of it (A), not the next line's
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 4.0)
        answer = shot_run(self.TWO, chunks, mono=voiced(5.0))
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertEqual(found["at"]["line"], 0)
        self.assertEqual(found["at"]["shot"], "A")
        self.assertAlmostEqual(found["at"]["seconds"], 0.9, places=2)
        self.assertAlmostEqual(found["at"]["end"], 4.0, places=2)

    def test_a_quiet_gap_between_lines_is_not(self):
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 4.0)
        mono = numpy.concatenate([voiced(0.9), silent(3.1), voiced(1.0)])
        answer = shot_run(self.TWO, chunks, mono=mono)
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    def test_a_short_gap_between_lines_is_not(self):
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 1.5)
        answer = shot_run(self.TWO, chunks, mono=voiced(3.0))
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    def test_a_gap_holding_unmatched_words_is_unscripted_speech_only(self):
        chunks = (
            words("alpha bravo", 0.0)
            + words("zulu yankee", 1.5)
            + words("charlie delta", 4.0)
        )
        answer = shot_run(self.TWO, chunks, mono=voiced(5.0))
        # Stray words 1.5-2.4; the 2.4-4.0 stretch after them is audible
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertAlmostEqual(found["at"]["seconds"], 2.4, places=2)

    def test_a_gap_fully_covered_by_unmatched_words_is_unscripted_only(self):
        chunks = (
            words("alpha bravo", 0.0)
            + words("zulu yankee xray whiskey", 0.9, step=0.5, length=0.5)
            + words("charlie delta", 3.0)
        )
        answer = shot_run(self.TWO, chunks, mono=voiced(4.0))
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    def test_audible_stretch_before_a_stray_word_is_a_finding(self):
        # #964: a stray word late in the gap must not hide the stretch before it
        chunks = (
            words("alpha bravo", 0.0) + words("zulu", 4.0) + words("charlie delta", 6.0)
        )
        answer = shot_run(self.TWO, chunks, mono=voiced(7.0))
        # the stretches on both sides of the stray word are findings
        before, after = rules(answer, "speech_not_transcribed")
        self.assertAlmostEqual(before["at"]["seconds"], 0.9, places=2)
        self.assertAlmostEqual(before["at"]["end"], 4.0, places=2)
        self.assertAlmostEqual(after["at"]["seconds"], 4.4, places=2)
        # each is placed in the shot holding most of it, not the next line's
        self.assertEqual((before["at"]["shot"], before["at"]["line"]), ("A", 0))
        self.assertEqual((after["at"]["shot"], after["at"]["line"]), ("B", 1))

    def test_only_the_audible_part_of_a_gap_sizes_and_places_the_finding(self):
        # #967: the gap 0.9-5.0 is mostly silence in shot B; the audible 0.9-2.2
        # sits in A and is all the finding counts
        spans = [
            {"name": "A", "start": 0.0, "end": 2.0},
            {"name": "B", "start": 2.0, "end": 6.0},
        ]
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 5.0)
        mono = numpy.concatenate([voiced(2.2), silent(2.8), voiced(1.0)])
        answer = shot_run(self.TWO, chunks, mono=mono, spans=spans)
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertEqual((found["at"]["shot"], found["at"]["line"]), ("A", 0))
        self.assertAlmostEqual(found["at"]["seconds"], 0.9, places=2)
        self.assertLess(found["at"]["end"], 2.5)
        self.assertLess(found["value"], 1.6)

    def test_under_a_second_of_audible_sound_in_a_long_gap_is_not(self):
        chunks = words("alpha bravo", 0.0) + words("charlie delta", 5.0)
        mono = numpy.concatenate([voiced(1.5), silent(3.5), voiced(1.0)])
        answer = shot_run(self.TWO, chunks, mono=mono)
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])


class TestTailAndRepeatedSpan(unittest.TestCase):
    LINES = [{"text": "alpha bravo", "shot": "A"}]

    def test_audible_tail_after_the_last_word_is_a_finding(self):
        # #970: ~1.2 s of sound past the last word, then silence
        chunks = words("alpha bravo", 0.0)
        mono = numpy.concatenate([voiced(2.2), silent(2.8)])
        answer = run(self.LINES, chunks, mono)
        (found,) = rules(answer, "speech_not_transcribed")
        self.assertEqual(found["at"]["line"], 0)
        self.assertLess(found["value"], 1.6)

    def test_quiet_tail_after_the_last_word_is_not(self):
        chunks = words("alpha bravo", 0.0)
        mono = numpy.concatenate([voiced(1.0), silent(4.0)])
        answer = run(self.LINES, chunks, mono)
        self.assertEqual(rules(answer, "speech_not_transcribed"), [])

    def test_chunks_sharing_one_span_are_heard_once(self):
        # #971: three words cannot share one span
        chunks = words("alpha bravo", 0.0) + [
            {"start": 3.0, "end": 3.6, "text": " How?"} for _ in range(3)
        ]
        mono = voiced(4.0)
        answer = run(self.LINES, chunks, mono)
        self.assertEqual(len(answer["unmatched"]), 1)
        (dup,) = [d for d in answer["discarded"] if d["reason"] == "repeated_span"]
        self.assertEqual(dup["count"], 2)


class TestSpeechInSilentShot(unittest.TestCase):
    LINES = [{"text": "alpha bravo", "shot": "A"}]

    def test_voiced_words_in_an_unnamed_shot_are_one_finding(self):
        chunks = words("alpha bravo", 0.0) + words("zulu yankee xray", 3.5)
        answer = shot_run(self.LINES, chunks)
        found = rules(answer, "speech_in_silent_shot")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["shot"], "B")
        self.assertEqual(found[0]["value"], 3)
        self.assertIn("speech_in_silent_shot", answer["rules_applied"])
        self.assertEqual(skipped(answer, "speech_in_silent_shot"), [])

    def test_a_shot_a_line_names_gets_no_finding(self):
        chunks = words("alpha bravo", 0.0) + words("delta echo", 3.5)
        answer = shot_run(
            [{"text": "alpha bravo", "shot": "A"}, {"text": "delta echo", "shot": "B"}],
            chunks,
        )
        self.assertEqual(rules(answer, "speech_in_silent_shot"), [])

    def test_silent_unnamed_shot_has_no_finding(self):
        answer = shot_run(self.LINES, words("alpha bravo", 0.0))
        self.assertEqual(rules(answer, "speech_in_silent_shot"), [])
        self.assertIn("speech_in_silent_shot", answer["rules_applied"])

    def test_a_word_a_line_claimed_across_the_cut_is_not_silent_shot_speech(self):
        # 'charlie' straddles the cut (2.8-3.4, most of it in B) but line A
        # matched it: only B's own unmatched words count, from the first of them
        chunks = [
            *words("alpha bravo", 1.8),
            {"start": 2.8, "end": 3.4, "text": " charlie"},
            *words("zulu yankee", 3.5),
        ]
        answer = shot_run([{"text": "alpha bravo charlie", "shot": "A"}], chunks)
        (found,) = rules(answer, "speech_in_silent_shot")
        self.assertEqual(found["value"], 2)
        self.assertEqual(found["at"]["word"], "zulu")
        self.assertEqual(found["at"]["seconds"], 3.5)
        self.assertEqual(found["at"]["shot"], "B")
        self.assertEqual(answer["lines"][0]["similarity"], 1.0)

    def test_an_unclaimed_word_straddling_the_cut_is_at_the_cut(self):
        chunks = [
            *words("alpha bravo", 0.0),
            {"start": 2.8, "end": 3.4, "text": " zulu"},
        ]
        answer = shot_run(self.LINES, chunks)
        (found,) = rules(answer, "speech_in_silent_shot")
        self.assertEqual(found["value"], 1)
        self.assertEqual(found["at"]["seconds"], 3.0)
        self.assertEqual(found["at"]["shot"], "B")

    def test_words_over_silence_in_an_unnamed_shot_are_not_found(self):
        mono = numpy.concatenate([voiced(3.0), silent(3.0)])
        chunks = words("alpha bravo", 0.0) + words("zulu yankee", 3.5)
        answer = shot_run(self.LINES, chunks, mono)
        self.assertEqual(rules(answer, "speech_in_silent_shot"), [])

    def test_skipped_without_spans(self):
        for source in ("none", "argument"):
            with self.subTest(source=source):
                answer = run(
                    self.LINES, words("alpha bravo"), spans=None, shots_source=source
                )
                self.assertNotIn("speech_in_silent_shot", answer["rules_applied"])
                (skip,) = skipped(answer, "speech_in_silent_shot")
                self.assertTrue(skip["reason"])

    def test_reason_names_the_shots_the_lines_name_when_none_are_known(self):
        answer = run(self.LINES, words("alpha bravo"))
        (skip,) = skipped(answer, "speech_in_silent_shot")
        self.assertIn("'A'", skip["reason"])

    def test_skipped_when_no_line_names_a_shot(self):
        answer = shot_run(["alpha bravo"], words("alpha bravo"))
        self.assertNotIn("speech_in_silent_shot", answer["rules_applied"])
        (skip,) = skipped(answer, "speech_in_silent_shot")
        self.assertIn("no line names", skip["reason"])


class TestClippedUnderBed(unittest.TestCase):
    CHUNKS = [
        {"start": 0.5, "end": 0.9, "text": " hello"},
        {"start": 1.6, "end": 1.95, "text": " world"},
    ]

    def test_bed_under_a_finished_word_is_not_clipped(self):
        # voice (-20 dBFS) up to 1.7 s, then a bed ~30 dB lower to the end
        mono = numpy.concatenate([voiced(1.7), voiced(0.3, amplitude=0.003)])
        answer = run(["hello world"], self.CHUNKS, mono)
        self.assertEqual(rules(answer, "line_clipped_at_end"), [])

    def test_bed_under_the_whole_word_still_clips_a_voiced_tail(self):
        mono = voiced(2.0) + voiced(2.0, amplitude=0.003)
        answer = run(["hello world"], self.CHUNKS, mono)
        self.assertEqual(len(rules(answer, "line_clipped_at_end")), 1)

    def test_last_word_stretched_over_the_bed_is_not_clipped(self):
        # the ASR places the last word wholly over the bed (#959): its own
        # level is the bed's, the line's voice is 30 dB above the tail
        mono = numpy.concatenate([voiced(1.2), voiced(0.8, amplitude=0.003)])
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 1.5, "end": 1.95, "text": " world"},
        ]
        answer = run(["hello world"], chunks, mono)
        self.assertEqual(rules(answer, "line_clipped_at_end"), [])


class TestClippedAtShotEnd(unittest.TestCase):
    LINES = [{"text": "hello world", "shot": "A"}]

    def test_loud_shot_end_clips_the_line_with_its_shot(self):
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 2.5, "end": 2.9, "text": " world"},
        ]
        answer = shot_run(self.LINES, chunks)
        found = rules(answer, "line_clipped_at_end")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["at"]["shot"], "A")
        self.assertEqual(found[0]["at"]["line"], 0)

    def test_quiet_tail_at_the_shot_end_is_not_clipped(self):
        mono = numpy.concatenate([voiced(2.7), silent(3.3)])
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 2.4, "end": 2.7, "text": " world"},
        ]
        answer = shot_run(self.LINES, chunks, mono)
        self.assertEqual(rules(answer, "line_clipped_at_end"), [])

    def test_file_end_clipping_still_works_without_spans(self):
        chunks = [
            {"start": 0.5, "end": 0.9, "text": " hello"},
            {"start": 1.6, "end": 1.95, "text": " world"},
        ]
        answer = run(["hello world"], chunks, voiced(2.0))
        found = rules(answer, "line_clipped_at_end")
        self.assertEqual(len(found), 1)
        self.assertNotIn("shot", found[0]["at"])


class TestShotSpans(unittest.TestCase):
    def test_sample_span_is_preferred_over_frames(self):
        records = [
            {
                "name": "A",
                "start_sample": RATE,
                "num_samples": 2 * RATE,
                "start_frame": 240,
                "num_frames": 240,
            }
        ]
        self.assertEqual(
            shot_spans(records, RATE, 24, 10.0),
            [{"name": "A", "start": 1.0, "end": 3.0}],
        )

    def test_frames_over_fps_is_the_fallback(self):
        records = [{"name": "A", "start_frame": 24, "num_frames": 48}]
        self.assertEqual(
            shot_spans(records, RATE, 24, 10.0),
            [{"name": "A", "start": 1.0, "end": 3.0}],
        )

    def test_none_when_a_record_cannot_be_placed(self):
        # No recorded samples and no fps: nothing places the shot.
        self.assertIsNone(
            shot_spans(
                [{"name": "A", "start_frame": 0, "num_frames": 5}], RATE, None, 1
            )
        )

    def test_missing_start_frame_is_frame_zero_as_the_probes_place_it(self):
        # assess.sample_span reads a record without start_frame as frame 0;
        # the map is placed, not skipped whole (#644 arch review).
        records = [
            {"name": "A", "start_sample": 0, "num_samples": RATE},
            {"name": "B", "num_frames": 24},
        ]
        self.assertEqual(
            shot_spans(records, RATE, 24, 10.0),
            [
                {"name": "A", "start": 0.0, "end": 1.0},
                {"name": "B", "start": 0.0, "end": 1.0},
            ],
        )

    def test_agrees_with_the_probes_sample_span(self):
        from types import SimpleNamespace

        from dw.tasks.assess import sample_span

        media = SimpleNamespace(fps=25, sample_rate=RATE)
        records = [
            {"name": "A", "start_frame": 0, "num_frames": 50},
            {"name": "B", "start_frame": 50, "num_frames": 30},
            {"name": "C", "start_sample": 3 * RATE, "num_samples": RATE // 2},
        ]
        for record, span in zip(records, shot_spans(records, RATE, 25, 60.0)):
            start, count, _ = sample_span(record, media)
            self.assertAlmostEqual(span["start"], start / RATE)
            self.assertAlmostEqual(span["end"], (start + count) / RATE)

    def test_clamped_to_duration(self):
        records = [
            {"name": "A", "start_sample": 2 * RATE, "num_samples": 4 * RATE},
            {"name": "B", "start_sample": 5 * RATE, "num_samples": RATE},
        ]
        spans = shot_spans(records, RATE, None, 3.0)
        self.assertEqual(spans[0], {"name": "A", "start": 2.0, "end": 3.0})
        self.assertEqual(spans[1], {"name": "B", "start": 3.0, "end": 3.0})


class TestShotValidation(unittest.TestCase):
    RECORDS = [{"name": "A"}, {"name": "B"}]

    def test_unknown_shot_is_refused_listing_the_known(self):
        message = shot_names_error(
            parse_lines([{"text": "x", "shot": "C"}]), self.RECORDS
        )
        self.assertIn("'C'", message)
        self.assertIn("'A'", message)
        self.assertIn("'B'", message)

    def test_duplicate_names_a_line_names_are_refused(self):
        records = [{"name": "A"}, {"name": "A"}]
        message = shot_names_error(parse_lines([{"text": "x", "shot": "A"}]), records)
        self.assertIn("more than once", message)
        self.assertIn("start_frame", message)

    def test_start_frame_picks_one_occurrence_of_a_repeated_shot(self):
        records = [
            {"name": "A", "start_frame": 0},
            {"name": "B", "start_frame": 10},
            {"name": "A", "start_frame": 20},
        ]
        lines = parse_lines(
            [{"text": "x", "shot": "A", "start_frame": 20}, {"text": "y", "shot": "B"}]
        )
        self.assertIsNone(shot_names_error(lines, records))
        resolved = with_full_shot_names(lines, records)
        self.assertEqual([line["shot"] for line in resolved], ["A#20", "B"])
        self.assertEqual(
            [record["name"] for record in disambiguated(records)], ["A#0", "B", "A#20"]
        )

    def test_a_start_frame_no_occurrence_has_is_refused(self):
        records = [{"name": "A", "start_frame": 0}, {"name": "A", "start_frame": 20}]
        lines = parse_lines([{"text": "x", "shot": "A", "start_frame": 5}])
        self.assertIn("matches no occurrence", shot_names_error(lines, records))

    def test_the_refusal_lists_only_the_nameable_shots(self):
        records = [
            {"name": "A", "start_frame": 0},
            {"name": "A", "start_frame": 20},
            {"name": "B", "start_frame": 40},
        ]
        message = shot_names_error(parse_lines([{"text": "x", "shot": "A"}]), records)
        self.assertIn("Shots named without it: 'B'", message)
        self.assertNotIn("'A', 'B'", message)

    def test_a_name_repeated_only_across_files_is_not_a_duplicate(self):
        # shots.duplicate_shot_names groups by file (#644 arch review).
        records = [{"name": "A", "file": "one.mp4"}, {"name": "A", "file": "two.mp4"}]
        lines = parse_lines([{"text": "x", "shot": "A"}])
        self.assertIsNone(shot_names_error(lines, records))

    def test_known_names_and_untagged_lines_pass(self):
        lines = parse_lines([{"text": "x", "shot": "B"}, "y"])
        self.assertIsNone(shot_names_error(lines, self.RECORDS))

    def test_a_short_member_name_matches_the_one_shot_carrying_it(self):
        records = [
            {"name": "x-shot@cold_open.0-0.0.mp4"},
            {"name": "x-shot@cold_open_two.0-0.0.mp4"},
        ]
        lines = parse_lines([{"text": "x", "shot": "cold_open"}])
        self.assertIsNone(shot_names_error(lines, records))
        resolved = with_full_shot_names(lines, records)
        self.assertEqual(resolved[0]["shot"], "x-shot@cold_open.0-0.0.mp4")

    def test_member_keys_come_from_the_for_each_separator(self):
        from dw.references import member_keys

        self.assertEqual(member_keys("shot@cold_open"), ["cold_open"])
        self.assertEqual(member_keys("x-shot@cold_open.0-0.0.mp4"), ["cold_open"])
        self.assertEqual(member_keys("plain.mp4"), [])

    def test_a_valid_variable_name_comes_back_whole_as_a_member_key(self):
        # A for_each entry's name is checked by validate_variable_name; the
        # key member_keys reads back must end exactly where that rule does,
        # for every character either could meet
        import string

        from dw.for_each import member_name
        from dw.references import member_keys
        from dw.security import InvalidInputError, validate_variable_name

        for char in string.printable:
            key = f"a{char}b"
            filename = member_name("shot", key) + ".0-0.0.mp4"
            try:
                validate_variable_name(key)
            except InvalidInputError:
                self.assertNotEqual(member_keys(filename)[:1], [key], repr(char))
            else:
                self.assertEqual(member_keys(filename), [key], repr(char))

    def test_a_short_name_two_shots_carry_is_refused(self):
        records = [{"name": "a-shot@open.mp4"}, {"name": "b-shot@open.mp4"}]
        message = shot_names_error(
            parse_lines([{"text": "x", "shot": "open"}]), records
        )
        self.assertIn("more than one", message)

    def test_parse_shots(self):
        self.assertIsNone(parse_shots(None))
        self.assertIsNone(parse_shots([]))
        self.assertEqual(parse_shots([{"name": "A"}]), [{"name": "A"}])
        for bad in (
            "A",
            {"name": "A"},
            [{"start": 0}],
            [{"name": ""}],
            [{"name": 3}],
            ["A"],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as caught:
                    parse_shots(bad)
                self.assertIn("shots", str(caught.exception))

    def test_parse_lines_shot_validation(self):
        self.assertEqual(parse_lines([{"text": "x", "shot": "A"}])[0]["shot"], "A")
        for bad in (
            [{"text": "x", "shot": ""}],
            [{"text": "x", "shot": 3}],
            [{"text": "x", "shot": "  "}],
            [{"text": "x", "shot": "A", "extra": 1}],
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_lines(bad)


class TestShotLinesErrors(unittest.TestCase):
    def test_line_naming_an_absent_literal_shot(self):
        errors = script_lines_errors(
            {
                "lines": [{"text": "x", "shot": "C"}],
                "shots": [{"name": "A"}, {"name": "B"}],
            }
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "lines")
        self.assertIn("'A'", errors[0][1])
        self.assertIn("'B'", errors[0][1])

    def test_bad_literal_shots(self):
        errors = script_lines_errors({"lines": ["x"], "shots": [{"start": 0}]})
        self.assertEqual([e[0] for e in errors], ["shots"])

    def test_reference_shots_are_left_alone(self):
        for ref in ("variable:x", "previous_result:y"):
            with self.subTest(ref=ref):
                self.assertEqual(
                    script_lines_errors(
                        {"lines": [{"text": "x", "shot": "C"}], "shots": ref}
                    ),
                    [],
                )

    def test_matching_literal_shots_pass(self):
        self.assertEqual(
            script_lines_errors(
                {"lines": [{"text": "x", "shot": "A"}], "shots": [{"name": "A"}]}
            ),
            [],
        )


RECORDS = [
    {"name": "A", "start_sample": 0, "num_samples": 3 * RATE},
    {"name": "B", "start_sample": 3 * RATE, "num_samples": 3 * RATE},
]


def take(shots=RECORDS):
    return SimpleNamespace(
        audio=voiced(6.0)[numpy.newaxis, :], sample_rate=RATE, fps=24, shots=shots
    )


CANNED = {
    "text": " alpha bravo zulu",
    "chunks": words("alpha bravo", 0.0) + words("zulu", 3.5),
}
TAGGED = [{"text": "alpha bravo", "shot": "A"}]


class TestCheckScriptShots(unittest.TestCase):
    def call(self, audio, lines=TAGGED, **kwargs):
        with patch(
            "dw.tasks.audio_transcription.transcribe_audio", return_value=CANNED
        ) as transcribe:
            return check_script(audio, lines, **kwargs), transcribe

    def test_shots_resolve_from_the_artifact(self):
        answer, _ = self.call(take())
        self.assertEqual(answer["shots_source"], "artifact")
        self.assertEqual([s["name"] for s in answer["shots"]], ["A", "B"])
        self.assertEqual(answer["lines"][0]["shot"], "A")
        found = rules(answer, "speech_in_silent_shot")
        self.assertEqual([f["at"]["shot"] for f in found], ["B"])

    def test_explicit_shots_argument_wins(self):
        explicit = [{"name": "A", "start_sample": 0, "num_samples": 6 * RATE}]
        answer, _ = self.call(take(), shots=explicit)
        self.assertEqual(answer["shots_source"], "argument")
        self.assertEqual([s["name"] for s in answer["shots"]], ["A"])

    def test_unknown_shot_is_refused_before_transcription(self):
        with patch("dw.tasks.audio_transcription.transcribe_audio") as transcribe:
            with self.assertRaises(ValueError) as caught:
                check_script(take(), [{"text": "x", "shot": "C"}])
        self.assertIn("'A'", str(caught.exception))
        self.assertIn("'B'", str(caught.exception))
        transcribe.assert_not_called()

    def test_path_resolves_shots_from_the_manifest(self):
        with (
            patch(
                "dw.tasks.audio_utils.load_audio",
                return_value=(voiced(6.0)[numpy.newaxis, :], RATE),
            ),
            patch("dw.locations.validate_media_path", return_value="take.wav"),
            patch("dw.runs.shots_beside", return_value=RECORDS) as beside,
        ):
            answer, _ = self.call("take.wav")
        beside.assert_called_once_with("take.wav")
        self.assertEqual(answer["shots_source"], "manifest")
        self.assertEqual(answer["lines"][0]["shot"], "A")

    def test_shotless_take_skips_the_shot_rule_with_a_reason(self):
        answer, _ = self.call(take(shots=None))
        self.assertEqual(answer["shots_source"], "none")
        self.assertIsNone(answer["shots"])
        (skip,) = skipped(answer, "speech_in_silent_shot")
        self.assertIn("'A'", skip["reason"])


if __name__ == "__main__":
    unittest.main()


def test_check_script_defaults_to_the_one_whisper_model():
    import inspect

    from dw.tasks import audio_transcription

    default = inspect.signature(script_check.check_script).parameters["model_name"]
    assert default.default is audio_transcription.WHISPER_DEFAULT_MODEL


def test_transcription_input_is_level_independent():
    import numpy

    from dw.tasks.script_check import level_for_transcription

    rng = numpy.random.default_rng(0)
    loud = rng.uniform(-0.5, 0.5, (1, 48000)).astype(numpy.float32)
    quiet = loud * 0.1
    (a, gain_a), (b, gain_b) = (
        level_for_transcription(loud),
        level_for_transcription(quiet),
    )
    assert gain_b > gain_a
    assert numpy.allclose(a, b, atol=1e-4)
    assert float(numpy.abs(a).max()) <= 1.0
    silent = numpy.zeros((1, 100), dtype=numpy.float32)
    assert level_for_transcription(silent) == (silent, 1.0)


class PerShotPassTest(unittest.TestCase):
    """#963: a transcript_loop on a cut with known shots is re-transcribed per shot."""

    def _run(self, whole, per_shot):
        calls = []
        slice_calls = iter(per_shot)

        def fake(audio, **kwargs):
            calls.append(len(audio[0]) / RATE)
            if len(calls) == 1:
                return {"text": "", "chunks": whole}
            return {"text": "", "chunks": next(slice_calls)}

        records = [
            {"name": "shot@a", "start_sample": 0, "num_samples": 6 * RATE},
            {"name": "shot@b", "start_sample": 6 * RATE, "num_samples": 6 * RATE},
        ]
        wave = voiced(12.0)[None, :]
        with (
            patch("dw.tasks.audio_transcription.transcribe_audio", fake),
            patch.object(script_check, "waveform_and_rate", return_value=(wave, RATE)),
            patch.object(
                script_check, "resolve_shots", return_value=(records, "manifest")
            ),
        ):
            answer = script_check.check_script(
                "x.mp4",
                [
                    {"text": "the lid is gone", "shot": "a"},
                    {"text": "i wore it", "shot": "b"},
                ],
            )
        return answer, calls

    def test_looped_cut_is_rechecked_shot_by_shot(self):
        whole = words("the lid is gone", 1.0) + words(
            " ".join(["the lid is gone"] * 8), 3.0, step=0.1, length=0.05
        )
        answer, calls = self._run(
            whole, [words("the lid is gone", 1.0), words("i wore it", 1.0)]
        )
        self.assertEqual(calls, [12.0, 6.0, 6.0])
        self.assertEqual(answer["transcription_pass"], "per_shot")
        self.assertEqual([x["similarity"] for x in answer["lines"]], [1.0, 1.0])
        self.assertEqual(answer["lines"][1]["start"], 7.0)
        self.assertEqual(answer["lines"][1]["pass"], "per_shot")
        self.assertNotIn("not_transcribed", answer["lines"][1])

    def test_clean_cut_is_one_pass(self):
        whole = words("the lid is gone", 1.0) + words("i wore it", 7.0)
        answer, calls = self._run(whole, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(answer["transcription_pass"], "whole_cut")
