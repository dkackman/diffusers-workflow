"""select reduces a list of candidates to one by a deterministic rule, so a
fan-out (for_each) can feed a single expensive stage without an agent in the
loop."""

import pytest

from dw.tasks.select import select


class TestSelect:
    def test_argmax_returns_the_highest_scoring_candidate(self):
        result = select(
            candidates=["a", "b", "c"], scores=[0.1, 0.9, 0.5], rule="argmax"
        )
        assert result.value == "b"
        assert result.position == 1
        assert result.score == 0.9

    def test_argmin_returns_the_lowest_scoring_candidate(self):
        result = select(
            candidates=["a", "b", "c"], scores=[0.1, 0.9, 0.5], rule="argmin"
        )
        assert result.value == "a"
        assert result.position == 0
        assert result.score == 0.1

    def test_first_above_returns_the_first_candidate_meeting_the_threshold(self):
        result = select(
            candidates=["a", "b", "c"],
            scores=[0.1, 0.9, 0.95],
            rule="first_above",
            threshold=0.8,
        )
        assert result.value == "b"
        assert result.position == 1
        assert result.score == 0.9

    def test_first_below_returns_the_first_candidate_meeting_the_threshold(self):
        result = select(
            candidates=["a", "b", "c"],
            scores=[0.9, 0.1, 0.05],
            rule="first_below",
            threshold=0.2,
        )
        assert result.value == "b"
        assert result.position == 1
        assert result.score == 0.1

    def test_index_returns_the_candidate_at_that_position(self):
        result = select(
            candidates=["a", "b", "c"], scores=[0.1, 0.9, 0.5], rule="index", index=2
        )
        assert result.value == "c"
        assert result.position == 2
        assert result.score == 0.5

    def test_ties_go_to_the_first_in_list_order(self):
        result = select(
            candidates=["a", "b", "c"], scores=[0.5, 0.9, 0.9], rule="argmax"
        )
        assert result.value == "b"
        assert result.position == 1

    def test_a_string_score_is_parsed_once(self):
        result = select(candidates=["a", "b"], scores=["0.1", "0.9"], rule="argmax")
        assert result.value == "b"
        assert result.score == 0.9

    def test_a_prose_score_is_refused(self):
        with pytest.raises(ValueError, match="score 1 is not a number"):
            select(candidates=["a", "b"], scores=[0.1, "pretty good"], rule="argmax")

    def test_length_mismatch_names_both_lengths(self):
        with pytest.raises(ValueError, match="3 candidates.*2 scores"):
            select(candidates=["a", "b", "c"], scores=[0.1, 0.9], rule="argmax")

    def test_first_above_with_nothing_passing_is_an_error(self):
        with pytest.raises(ValueError, match="no candidate"):
            select(
                candidates=["a", "b"],
                scores=[0.1, 0.2],
                rule="first_above",
                threshold=0.5,
            )

    def test_first_below_with_nothing_passing_is_an_error(self):
        with pytest.raises(ValueError, match="no candidate"):
            select(
                candidates=["a", "b"],
                scores=[0.6, 0.7],
                rule="first_below",
                threshold=0.5,
            )

    def test_index_out_of_range_is_an_error(self):
        with pytest.raises(ValueError, match="index 5"):
            select(candidates=["a", "b"], scores=[0.1, 0.2], rule="index", index=5)

    @pytest.mark.parametrize("index", [2, 2.0, "2"])
    def test_a_whole_number_index_selects_that_position(self, index):
        result = select(
            candidates=list("abcde"), scores=[0, 1, 2, 3, 4], rule="index", index=index
        )
        assert result.position == 2
        assert result.value == "c"

    @pytest.mark.parametrize("index", [2.5, "two", True, [1], {}])
    def test_a_non_whole_index_is_refused_by_name(self, index):
        with pytest.raises(ValueError, match="whole number"):
            select(candidates=list("abcde"), scores=[0] * 5, rule="index", index=index)

    def test_a_missing_index_is_refused(self):
        with pytest.raises(ValueError, match="requires an index"):
            select(candidates=["a", "b"], scores=[0, 1], rule="index")

    @pytest.mark.parametrize("index", [5, -1, "7"])
    def test_an_out_of_range_index_names_the_valid_span(self, index):
        with pytest.raises(ValueError, match=r"\(0 to 4\)"):
            select(candidates=list("abcde"), scores=[0] * 5, rule="index", index=index)

    def test_an_unknown_rule_is_an_error(self):
        with pytest.raises(ValueError, match="unknown rule"):
            select(candidates=["a", "b"], scores=[0.1, 0.2], rule="top_k")


class TestSelectAsATask:
    def test_it_runs_through_the_task_dispatch(self):
        from dw.tasks.task import Task

        task = Task({"command": "select", "arguments": {}}, "cpu")

        result = task.run(
            {"candidates": ["a", "b", "c"], "scores": [0.1, 0.9, 0.5], "rule": "argmax"}
        )

        assert result == "b"
