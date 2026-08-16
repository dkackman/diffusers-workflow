"""
Unit tests for the A1111-style prompt attention parser and the tokenization
helpers that turn its output into weighted CLIP/T5 token chunks.

These are the pure-logic half of prompt weighting - no model, no device. The
device-handling half lives in test_prompt_weighting.py.
"""

import pytest

from dw.prompt_weighting import (
    _group_tokens_and_weights,
    _tokenize_clip_with_weights,
    _tokenize_t5_with_weights,
    parse_prompt_attention,
)

ROUND = 1.1
SQUARE = 1 / 1.1


def weights_for(parsed, word):
    """The weight attached to the fragment holding `word`."""
    return next(w for text, w in parsed if word in text)


class TestParsePromptAttention:
    """(word:1.5), ((word)), [word] and their escapes"""

    def test_plain_text_is_one_fragment_at_weight_one(self):
        assert parse_prompt_attention("a cat") == [["a cat", 1.0]]

    def test_bare_parens_apply_the_round_multiplier(self):
        assert parse_prompt_attention("a (cat)") == [["a ", 1.0], ["cat", ROUND]]

    def test_explicit_weight_overrides_the_multiplier(self):
        assert parse_prompt_attention("a (cat:1.5)") == [["a ", 1.0], ["cat", 1.5]]

    def test_nested_parens_compound(self):
        parsed = parse_prompt_attention("a ((cat))")
        assert weights_for(parsed, "cat") == pytest.approx(ROUND * ROUND)

    def test_brackets_de_emphasize(self):
        parsed = parse_prompt_attention("a [cat]")
        assert weights_for(parsed, "cat") == pytest.approx(SQUARE)

    def test_nested_brackets_compound(self):
        parsed = parse_prompt_attention("a [[cat]]")
        assert weights_for(parsed, "cat") == pytest.approx(SQUARE * SQUARE)

    def test_escaped_parens_are_literal_text(self):
        # \( \) must survive into the prompt as characters, not as emphasis
        assert parse_prompt_attention(r"a \(cat\)") == [["a (cat)", 1.0]]

    def test_negative_weights_are_allowed(self):
        parsed = parse_prompt_attention("a (cat:-1.2)")
        assert weights_for(parsed, "cat") == pytest.approx(-1.2)

    def test_weight_syntax_inside_brackets_is_not_a_weight(self):
        # Only round brackets carry the :weight form; inside [] it is literal
        assert parse_prompt_attention("a [cat:1.5]") == [
            ["a ", 1.0],
            ["cat:1.5", pytest.approx(SQUARE)],
        ]

    def test_independent_groups_keep_separate_weights(self):
        assert parse_prompt_attention("a (cat:1.5) and (dog:0.5)") == [
            ["a ", 1.0],
            ["cat", 1.5],
            [" and ", 1.0],
            ["dog", 0.5],
        ]

    def test_an_empty_prompt_yields_a_single_empty_fragment(self):
        # Downstream code indexes res[0] unconditionally, so this must never be []
        assert parse_prompt_attention("") == [["", 1.0]]

    def test_an_unclosed_paren_still_weights_the_remainder(self):
        # Malformed input must not raise - it weights to end of prompt
        assert parse_prompt_attention("a (cat") == [["a ", 1.0], ["cat", ROUND]]

    def test_an_unopened_paren_is_literal_text(self):
        assert parse_prompt_attention("a cat)") == [["a cat)", 1.0]]

    def test_adjacent_equal_weights_are_merged(self):
        # The merge pass keeps chunk counts down; "a " and "cat" both weight 1.0
        assert parse_prompt_attention("a cat and a dog") == [["a cat and a dog", 1.0]]

    def test_break_becomes_its_own_sentinel_fragment(self):
        assert parse_prompt_attention("one BREAK two") == [
            ["one", 1.0],
            ["BREAK", -1],
            ["two", 1.0],
        ]

    def test_break_as_a_substring_is_not_a_sentinel(self):
        # \bBREAK\b - BREAKFAST is a word, not a chunk boundary
        assert parse_prompt_attention("a BREAKFAST") == [["a BREAKFAST", 1.0]]


class FakeTokenizer:
    """Maps each whitespace-separated word to one id, so weight expansion is checkable.

    CLIP tokenizers wrap output in BOS/EOS, which _tokenize_clip_with_weights
    strips with [1:-1]; the T5 path keeps what the tokenizer returns.
    """

    def __init__(self, wrap=True):
        self.wrap = wrap
        self.seen = []

    def __call__(self, text, **kwargs):
        self.seen.append(text)
        ids = [100 + i for i, _ in enumerate(text.split())]
        if self.wrap:
            ids = [49406] + ids + [49407]
        return type("Encoding", (), {"input_ids": ids})()


class TestTokenizeWithWeights:
    def test_clip_expands_a_weight_across_every_token_of_its_fragment(self):
        tokens, weights = _tokenize_clip_with_weights(
            FakeTokenizer(), "plain (two weighted words:1.5)"
        )
        assert len(tokens) == len(weights)
        # "plain " -> 1 token at 1.0; "two weighted words" -> 3 tokens at 1.5
        assert weights == [1.0, 1.5, 1.5, 1.5]

    def test_clip_strips_the_tokenizer_bos_and_eos(self):
        tokens, _ = _tokenize_clip_with_weights(FakeTokenizer(), "a b c")
        assert 49406 not in tokens and 49407 not in tokens

    def test_t5_keeps_the_tokenizer_special_tokens(self):
        tokens, weights = _tokenize_t5_with_weights(FakeTokenizer(wrap=False), "a b")
        assert tokens == [100, 101]
        assert weights == [1.0, 1.0]

    @pytest.mark.parametrize("empty", ["", None])
    def test_an_empty_prompt_falls_back_to_a_placeholder(self, empty):
        # Encoders reject a zero-token input; "empty" stands in for a blank prompt
        tokenizer = FakeTokenizer()
        tokens, _ = _tokenize_clip_with_weights(tokenizer, empty)
        assert tokenizer.seen == ["empty"]
        assert tokens


class TestGroupTokensAndWeights:
    """Chunking into the 77-token CLIP window, BOS/EOS included"""

    def test_a_short_prompt_is_one_padded_chunk(self):
        tokens, weights = _group_tokens_and_weights([1, 2, 3], [1.0, 1.0, 1.0])
        assert len(tokens) == 1
        assert len(tokens[0]) == 77
        assert len(weights[0]) == 77
        assert tokens[0][0] == 49406
        assert tokens[0][1:4] == [1, 2, 3]
        assert set(tokens[0][4:]) == {49407}

    def test_padding_carries_weight_one(self):
        # A padded slot must not inherit the prompt's weight
        _, weights = _group_tokens_and_weights([1, 2], [2.0, 2.0])
        assert weights[0][0] == 1.0
        assert weights[0][1:3] == [2.0, 2.0]
        assert set(weights[0][3:]) == {1.0}

    def test_exactly_75_tokens_is_a_single_chunk(self):
        # Boundary: the >= 75 loop consumes them all, leaving no trailing chunk
        tokens, _ = _group_tokens_and_weights(list(range(75)), [1.0] * 75)
        assert len(tokens) == 1
        assert len(tokens[0]) == 77

    def test_a_long_prompt_splits_into_full_77_token_chunks(self):
        tokens, weights = _group_tokens_and_weights(list(range(80)), [2.0] * 80)
        assert len(tokens) == 2
        assert [len(chunk) for chunk in tokens] == [77, 77]
        assert [len(chunk) for chunk in weights] == [77, 77]
        # The second chunk resumes where the first left off
        assert tokens[1][1:6] == [75, 76, 77, 78, 79]

    def test_pad_last_block_false_leaves_the_tail_unpadded(self):
        tokens, weights = _group_tokens_and_weights(
            [1, 2], [1.0, 1.0], pad_last_block=False
        )
        assert tokens == [[49406, 1, 2, 49407]]
        assert weights == [[1.0, 1.0, 1.0, 1.0]]

    def test_the_input_lists_are_not_mutated(self):
        # The caller reuses prompt_tokens after grouping; popping in place would
        # empty them out from under it
        tokens_in = list(range(80))
        weights_in = [1.0] * 80
        _group_tokens_and_weights(tokens_in, weights_in)
        assert len(tokens_in) == 80
        assert len(weights_in) == 80
