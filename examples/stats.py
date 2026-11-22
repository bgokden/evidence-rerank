"""Order statistics shared by the harnesses, so one definition cannot drift.

`percentile` is here because the obvious index arithmetic was wrong everywhere it
was written out by hand. `ordered[int(n * 0.2) - 1]` returns the *second smallest*
of eleven values, which is the 10th percentile, not the 20th; at n = 4 it returns
the minimum. The error was not merely conservative. A quantity that sits too low
biases a level downwards, but it biases a *difference* by however much the two arms
differ in their lower tail, and here the weaker arm had the longer tail, so the
reported gain was about ten points too large.
"""

from __future__ import annotations


def percentile(values, share: float) -> float:
    """Linear-interpolated percentile, the definition numpy calls 'linear'.

    `share` is a fraction, so the 20th percentile is 0.2. With one value that value
    is returned; with none, zero, because every caller is reporting a percentage and
    an empty group means no cases rather than a missing measurement.
    """
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = share * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (position - low) * (ordered[high] - ordered[low]))
