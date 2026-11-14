"""The bench's verdict rule. Every measure is lower-is-better (the driver
negates the ones where higher is better before it gets here).

The floor a verdict reads is the larger, per measure, of the seed floor
(`noise_floor` of a control at two seeds) and the same-seed rerun gap
(`rerun_gap` over pairs of runs with the same arguments and seed): ruling
R35, after same-seed reruns on the bench server were found not to
reproduce (`bench/records/noise-floor.md`)."""


def noise_floor(a, b):
    """Per-measure absolute difference of two runs."""
    return {key: abs(a[key] - b[key]) for key in a if key in b}


def rerun_gap(pairs):
    """Per-measure largest `noise_floor` over same-seed, same-arguments
    pairs of runs."""
    gap = {}
    for a, b in pairs:
        for key, value in noise_floor(a, b).items():
            gap[key] = max(gap.get(key, 0.0), value)
    return gap


def combined_floor(*floors):
    """Per-measure maximum of several floors (seed floor, rerun gap, or the
    floors of two continuities a cross-continuity read spans)."""
    out = {}
    for floor in floors:
        for key, value in floor.items():
            out[key] = max(out.get(key, 0.0), value)
    return out


def verdict(control, arm, noise, primary):
    """`wins` when the primary beats control by more than its noise and no
    other measure is worse by more than its noise; `loses` when the primary
    is worse by more than noise; `mixed` when the primary wins but another
    measure is worse past noise; `tie` otherwise."""
    missing = [
        name for name, run in (("control", control), ("arm", arm)) if primary not in run
    ]
    if missing:
        raise ValueError(
            f"primary measure {primary!r} missing from {' and '.join(missing)}"
        )
    gain = control[primary] - arm[primary]
    floor = noise.get(primary, 0.0)
    worse_elsewhere = any(
        arm[key] - control[key] > noise.get(key, 0.0)
        for key in control
        if key != primary and key in arm
    )
    if gain > floor:
        return "mixed" if worse_elsewhere else "wins"
    if -gain > floor:
        return "loses"
    return "mixed" if worse_elsewhere else "tie"
