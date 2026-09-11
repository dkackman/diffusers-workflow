"""compose_text assembles a block of text out of parts written once - the
answer to hand-copying a character's description into every shot, without
introducing the string interpolation the engine deliberately does not have."""

import pytest

from dw.tasks.compose_text import compose_text


class TestComposeText:
    def test_parts_are_joined_in_order(self):
        assert compose_text(["first", "second"]) == "first\n\nsecond"

    def test_the_separator_is_the_caller_s(self):
        assert compose_text(["a", "b"], separator=" - ") == "a - b"

    def test_an_absent_part_is_dropped(self):
        """A character who does not speak in this shot is a null variable,
        not a hole in the prompt."""
        assert compose_text(["bible", None, "action"]) == "bible\n\naction"
        assert compose_text(["bible", "   ", "action"]) == "bible\n\naction"

    def test_an_absent_part_can_be_kept_instead(self):
        assert compose_text(["a", None, "b"], skip_empty=False) == "a\n\n\n\nb"

    def test_numbers_are_written_out(self):
        assert compose_text(["shot", 3], separator=" ") == "shot 3"

    def test_something_that_is_not_text_says_so(self):
        """A part that came back as an image or a dict means the reference in
        that position resolved to something other than the text meant."""
        with pytest.raises(ValueError, match="part 1 is a dict"):
            compose_text(["a", {"prompt": "b"}])

    def test_a_list_is_required(self):
        with pytest.raises(ValueError, match="list of parts"):
            compose_text("just a string")

    def test_nothing_to_join_says_so(self):
        with pytest.raises(ValueError, match="nothing to join"):
            compose_text([None, ""])


class TestComposeTextAsATask:
    def test_it_runs_through_the_task_dispatch(self):
        from dw.tasks.task import Task

        task = Task({"command": "compose_text", "arguments": {}}, "cpu")

        assert task.run({"parts": ["a", "b"], "separator": ", "}) == "a, b"
