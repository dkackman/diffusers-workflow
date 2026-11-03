"""The bench's verdict rule. Every measure is lower-is-better (the driver
negates the ones where higher is better before it gets here)."""


def noise_floor(a, b):
    """Per-measure absolute difference of two same-seed runs."""
    return {key: abs(a[key] - b[key]) for key in a if key in b}


def verdict(control, arm, noise, primary):
    """`wins` when the primary beats control by more than its noise and no
    other measure is worse by more than its noise; `loses` when the primary
    is worse by more than noise; `mixed` when the primary wins but another
    measure is worse past noise; `tie` otherwise."""
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
