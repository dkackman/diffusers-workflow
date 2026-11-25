"""#1618: the gpu_busy wait text names the card and counts minutes on read."""

from dw.server.dispatch import _gpu_busy_message


def test_one_card_is_worded_for_one_card():
    text = _gpu_busy_message([("cuda:0", 1000.0)], 0, now=1000.0 + 6 * 60)
    assert "behind the running job on cuda:0, 6 min in" in text
    assert "every card" not in text and "a card," not in text


def test_minutes_follow_now():
    running = [("mps", 0.0)]
    assert "0 min in" in _gpu_busy_message(running, 0, now=30)
    assert "10 min in" in _gpu_busy_message(running, 0, now=600)


def test_several_cards_are_listed():
    text = _gpu_busy_message([("cuda:0", None), ("cuda:1", 0.0)], 2, now=120)
    assert "2 running jobs on every card (cuda:0; cuda:1, 2 min in)" in text
    assert "2 queued job(s) ahead" in text
