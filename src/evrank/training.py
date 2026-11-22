"""Losses for training a re-ranker whose ablation effects mean something.

Ordinary re-ranker training supervises `score(context, candidate)`. Nothing in it
asks the *differences between* scores to carry information, and the differences are
what this package reports: removing one evidence unit and reading what moved is the
whole method. Measured over 1100 planted-evidence cases, a cause's ablation effect
beats a junk one about 93 percent of the time on the best model available and 42
percent of the time on the worst -- below a coin flip, meaning its effects point away
from the cause. That spread is unsupervised luck.

So these terms supervise it directly. Given a request where evidence unit `e` is the
reason candidate `c` belongs on the page, and `d(e, k) = score(k | E) - score(k | E-e)`:

    attribution  removing `e` must cost `c` more than it costs any other candidate
    margin       it must cost `c` enough to clear a floor, in the model's own units
    rank         ordinary relevance, unchanged, so quality does not regress
    anchor       scores must stay near the base model's, so nothing collapses

Those four are what `total_loss` applies. `parallel_loss` is a fifth, written and
tested but deliberately not wired in, and the honest reason is that the corpus cannot
currently supply its input. It is aimed at the measured language skew: every
per-request statistic varies more than twentyfold across eleven languages for content
that is semantically identical, and three attempts to correct that after the fact
with a threshold all failed, because the scale is a property of the model.

To train it, a pair must be the same request in two languages. Only the `matched`
cases are genuinely translated; in the `mixed` cases just the action verb is
localised and the candidate text stays English, so pairing those would teach the
model that a verb swap must not change an effect size -- which it already does not.
Wiring this term in means translating the corpus content first, and a term trained on
false pairs would be worse than no term.

Everything here takes padded batches with a boolean mask, because slates have
different lengths and a padded slot must not win an argmax or enter a mean.
"""

from __future__ import annotations

import torch
from torch.nn import functional

from evrank.config import Settings

# What a reason has to be worth, in the model's own score units, for training to stop
# pushing on it. Derived from the floor the service prints at rather than written out
# again: the two were duplicated constants, and when the gate moved from flooring the
# raw effect at 1.5 to flooring the centred effect at 0.75 this one stayed behind and
# its stated rationale -- that it *is* the shipped floor -- silently became false.
#
# Equal to that floor, after a factor of two was measured to collapse the model. The
# argument for asking more than the floor was that a hinge stopping exactly at the
# threshold is satisfied by a cause sitting on it, which the next reordering drops
# below -- so buy some headroom. The cost of that headroom had not been measured, and
# it is real: on a corpus whose planted causes have a median centred effect of 0.562,
# a margin of 1.5 asks for nearly three times what the data offers, and the only way
# to supply it is to distort the score function. The general-ranking guard caught it
# at p = 0.004, and at a margin of 0.75 the same recipe passes on two seeds.
#
# So the factor is 1.0 because 2.0 failed a safety gate, which is a reason to believe
# 2.0 was wrong rather than evidence that 1.0 is right. It has not been swept, and the
# quantity to sweep it against is the corpus's own median planted effect, not a
# constant: a margin far above where the data sits is a request to distort.
HEADROOM = 1.0
DEFAULT_MARGIN = HEADROOM * Settings.min_standout


def _masked_logits(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Padded slots cannot win a softmax."""
    return values.masked_fill(~mask, torch.finfo(values.dtype).min)


def standout(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """How much of each effect was specific to that candidate.

    The same quantity `evrank.standout` reports, and computed the same way: the lift
    subtracted is the mean over the *other* candidates, not over all of them.
    Including the explained candidate would attenuate every effect by
    `(width-1)/width`, which is a bias that varies with slate size and nothing else,
    and training against a biased target teaches the bias.
    """
    counts = mask.sum(dim=1, keepdim=True)
    totals = (delta * mask).sum(dim=1, keepdim=True)
    lift = (totals - delta) / (counts - 1).clamp(min=1)
    return delta - lift


def attribution_loss(
    delta: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Removing the cause must cost its own candidate more than it costs the others.

    A softmax over candidates of the ablation effect, against the candidate the
    evidence actually justifies. This is the term that makes the measurement
    informative rather than incidental, and it is why training costs two forward
    passes per example instead of one: the ablated context has to be scored too.

    Note what it does *not* ask for. It is indifferent to the absolute size of the
    effect and to the model's ranking -- it only asks that the ordering of the
    effects across candidates be right. Size is `margin_loss`'s job and ranking is
    `rank_loss`'s, and keeping them separate is what makes a failure attributable.
    """
    return functional.cross_entropy(_masked_logits(delta, mask), target)


def margin_loss(
    delta: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    margin: float = DEFAULT_MARGIN,
) -> torch.Tensor:
    """The cause's effect must clear the floor that decides whether it is printed.

    A hinge on the centred effect, not the raw one, because the centred quantity is
    what the explanation ranks by. One-sided: there is no penalty for exceeding the
    margin, since the floor is a threshold and not a target.
    """
    own = standout(delta, mask).gather(1, target.unsqueeze(1)).squeeze(1)
    return functional.relu(margin - own).mean()


def rank_loss(
    scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Ordinary relevance: the right candidate should score highest.

    Present so that the attribution terms are not free to move the score function
    anywhere that makes the deltas tidy. Without it they are, and the result ranks
    worse while explaining beautifully, which is a regression -- a reader sees the
    ranking on every request and the explanation on some of them.
    """
    return functional.cross_entropy(_masked_logits(scores, mask), target)


def anchor_loss(
    scores: torch.Tensor, base_scores: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Scores must stay near the ones the untrained model gave.

    Computed against values cached from the base model before training, so it costs
    no extra forward pass. It is the measure that keeps general ranking from
    drifting, and unlike early stopping it acts throughout training rather than
    only deciding when to halt. Pair it with an adapter, so that if it fails the
    base model is still on disk and still servable.
    """
    squared = (scores - base_scores) ** 2
    return (squared * mask).sum() / mask.sum().clamp(min=1)


def parallel_loss(
    left_delta: torch.Tensor,
    left_target: torch.Tensor,
    left_mask: torch.Tensor,
    right_delta: torch.Tensor,
    right_target: torch.Tensor,
    right_mask: torch.Tensor,
) -> torch.Tensor:
    """The same cause, written in another language, must be worth the same.

    Aimed at the one defect that no threshold could fix. A logit is comparable
    within a request and not between requests, and the gap between languages is a
    gap in signal-to-noise rather than in scale alone, which is why rescaling after
    the fact slides along a trade-off curve without bending it. Asking the model for
    the same effect size on both members of a translation pair asks it to bend the
    curve.

    Absolute difference rather than squared, so one badly scaled pair does not
    dominate the batch.
    """
    left = standout(left_delta, left_mask).gather(1, left_target.unsqueeze(1))
    right = standout(right_delta, right_mask).gather(1, right_target.unsqueeze(1))
    return (left - right).abs().mean()


def total_loss(
    scores: torch.Tensor,
    ablated: torch.Tensor,
    base_scores: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: dict[str, float],
    margin: float = DEFAULT_MARGIN,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Every term, weighted, with each one reported separately.

    The parts are returned alongside the sum because a single falling number says
    nothing about which objective is improving. The weights are swept one term at a
    time, starting from all-zero, which reproduces the untrained model exactly and
    makes this a superset of doing nothing rather than a replacement for it.
    """
    delta = scores - ablated
    parts = {
        "rank": rank_loss(scores, target, mask),
        "attribution": attribution_loss(delta, target, mask),
        "margin": margin_loss(delta, target, mask, margin),
        "anchor": anchor_loss(scores, base_scores, mask),
    }
    total = sum(
        weights.get(name, 0.0) * value
        for name, value in parts.items()
        if weights.get(name, 0.0)
    )
    if not isinstance(total, torch.Tensor):
        total = torch.zeros((), device=scores.device, requires_grad=True)
    return total, {name: float(value.detach()) for name, value in parts.items()}
