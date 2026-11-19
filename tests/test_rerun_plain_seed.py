"""#1492: a plain rerun keeps the seed the original drew, as one with overrides does."""

from dw.server.job_record import RERUN_SEED_UNKNOWN, rerun_arguments


def test_plain_rerun_keeps_drawn_seed():
    args, warnings = rerun_arguments(
        {"prompt": "a"}, None, False, "seed", lambda: {"seed": 123}
    )
    assert args == {"prompt": "a", "seed": 123}
    assert warnings == []


def test_plain_rerun_warns_when_seed_unknown():
    args, warnings = rerun_arguments({"prompt": "a"}, None, False, "seed", lambda: {})
    assert "seed" not in args
    assert warnings and warnings[0].startswith(RERUN_SEED_UNKNOWN)


def test_new_seed_still_draws_fresh():
    args, warnings = rerun_arguments(
        {"prompt": "a"}, None, True, "seed", lambda: {"seed": 123}
    )
    assert args["seed"] != 123 and warnings == []
