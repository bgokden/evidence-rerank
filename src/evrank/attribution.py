"""Attribution over evidence, not over latent dimensions.

Every quantity here is produced by re-running the model on a context with some
evidence omitted, so nothing depends on the ranker having interpretable features.
The effect of evidence unit e on the ordering of A over B is measured on the
pairwise margin:

    delta_e = (s_A - s_B) - (s_A^{-e} - s_B^{-e})

A positive delta means the model prefers A over B more strongly when e is present.
This is a measurement of model sensitivity to omitted input, not a causal claim
about the user's preferences.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
from collections.abc import Iterable, Sequence

from evrank.scoring import Ablator, removal_key

LEAVE_ONE_OUT = "leave_one_out"
SHAPLEY = "shapley"
SAMPLED_SHAPLEY = "sampled_shapley"

MAX_SUBSETS_PER_WAVE = 4096

# Above this, 2^n enumeration stops being affordable and `sampled_shapley` takes over.
MAX_EXACT_SHAPLEY_UNITS = 12


@dataclasses.dataclass(frozen=True)
class ContributionSpread:
    """How much an evidence unit's marginal contribution varies across coalitions.

    A Shapley value is a weighted mean over every order in which the evidence could
    have arrived. When the model is strongly interactive that mean can sit far from
    anything the model actually does, so the extremes and the spread are kept too.
    """

    low: float
    high: float
    stdev: float
    solo: float
    """The contribution of this unit added to an empty context, the one coalition
    whose value can be stated without reference to other evidence."""

    stderr: float = 0.0
    """Standard error of the reported value. Zero when every coalition was
    enumerated, positive when the value is a sample mean, in which case `low`,
    `high` and `stdev` describe the sample rather than the full distribution."""

    @property
    def reverses(self) -> bool:
        """True when the unit helps in some contexts and hurts in others."""
        return self.low < 0.0 < self.high


@dataclasses.dataclass(frozen=True)
class EvidenceEffect:
    unit: str
    label: str
    delta: float
    spread: ContributionSpread | None = None


@dataclasses.dataclass(frozen=True)
class PairAttribution:
    """Effects of each evidence unit on the A-over-B margin."""

    a: str
    b: str
    method: str
    base_margin: float
    empty_margin: float
    effects: tuple[EvidenceEffect, ...]

    def by_effect(self) -> tuple[EvidenceEffect, ...]:
        """Effects from most supportive of A to most supportive of B."""
        return tuple(sorted(self.effects, key=lambda e: (-e.delta, e.unit)))

    def total_effect(self) -> float:
        return sum(e.delta for e in self.effects)

    def reversing_effects(self) -> tuple[EvidenceEffect, ...]:
        """Effects whose direction depends on what other evidence is present."""
        return tuple(
            e for e in self.by_effect() if e.spread is not None and e.spread.reverses
        )


@dataclasses.dataclass(frozen=True)
class PairFlip:
    """A minimal set of evidence whose omission reverses the A/B ordering."""

    removed: tuple[str, ...]
    labels: tuple[str, ...]
    base_margin: float
    margin_after: float


@dataclasses.dataclass(frozen=True)
class TopFlip:
    """A minimal set of evidence whose omission changes which candidate ranks first."""

    removed: tuple[str, ...]
    labels: tuple[str, ...]
    base_top: str
    new_top: str
    margin_after: float


def leave_one_out(ablator: Ablator, a: str, b: str) -> PairAttribution:
    """One omission per evidence unit: cheap, and blind to redundancy.

    If two units say the same thing, removing either one alone changes little and
    both look unimportant. Group such facts into one unit, or use `shapley`.
    """
    ablator = ablator.restrict([a, b])
    units = ablator.evidence.units()
    wave = [removal_key(())] + [removal_key((u,)) for u in units] + [removal_key(units)]
    rankings = ablator.rank_without(wave)
    base = rankings[removal_key(())].margin(a, b)
    effects = tuple(
        EvidenceEffect(
            unit=u,
            label=ablator.evidence.label(u),
            delta=base - rankings[removal_key((u,))].margin(a, b),
        )
        for u in units
    )
    empty = rankings[removal_key(units)].margin(a, b)
    return PairAttribution(
        a=a,
        b=b,
        method=LEAVE_ONE_OUT,
        base_margin=base,
        empty_margin=empty,
        effects=effects,
    )


def shapley(
    ablator: Ablator, a: str, b: str, max_units: int = MAX_EXACT_SHAPLEY_UNITS
) -> PairAttribution:
    """Exact Shapley values of the margin function over evidence units.

    The margin is treated as a set function v(S) = margin under the context holding
    only the units in S. Each unit gets its average marginal contribution over all
    orders in which the evidence could have arrived, so redundant units split the
    credit they share instead of both scoring zero. Cost is 2^n contexts, hence the
    unit cap: group evidence before raising it.
    """
    ablator = ablator.restrict([a, b])
    units = ablator.evidence.units()
    n = len(units)
    if n > max_units:
        raise ValueError(
            "exact Shapley over %d units needs %d contexts; use sampled_shapley for "
            "an estimate with a reported error, group evidence that genuinely "
            "repeats, or use leave_one_out" % (n, 2**n)
        )
    subsets = [s for size in range(n + 1) for s in itertools.combinations(units, size)]
    rankings = ablator.rank_without([removal_key(set(units) - set(s)) for s in subsets])
    value = {
        frozenset(s): rankings[removal_key(set(units) - set(s))].margin(a, b) for s in subsets
    }

    effects = []
    for unit in units:
        others = [u for u in units if u != unit]
        # Every coalition this unit could join, with the probability of joining it
        # at that point in a uniformly random arrival order. The weights sum to one,
        # so the weighted mean is the Shapley value and the weighted variance
        # describes the distribution that value summarises.
        contributions: list[tuple[float, float]] = []
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for coalition in itertools.combinations(others, size):
                base_set = frozenset(coalition)
                contributions.append((weight, value[base_set | {unit}] - value[base_set]))
        delta = math.fsum(weight * margin for weight, margin in contributions)
        variance = math.fsum(weight * (margin - delta) ** 2 for weight, margin in contributions)
        margins = [margin for _, margin in contributions]
        effects.append(
            EvidenceEffect(
                unit=unit,
                label=ablator.evidence.label(unit),
                delta=delta,
                spread=ContributionSpread(
                    low=min(margins),
                    high=max(margins),
                    stdev=math.sqrt(max(variance, 0.0)),
                    solo=value[frozenset({unit})] - value[frozenset()],
                ),
            )
        )

    return PairAttribution(
        a=a,
        b=b,
        method=SHAPLEY,
        base_margin=value[frozenset(units)],
        empty_margin=value[frozenset()],
        effects=tuple(effects),
    )


def flip_pair(ablator: Ablator, a: str, b: str, max_size: int = 2) -> PairFlip | None:
    """Smallest omission that reverses the model's ordering of A and B.

    Searched by increasing size, so the answer is minimal in the number of evidence
    units removed; within a size, the most decisive reversal is returned. The test
    is a sign change of the margin relative to the unablated context, so it does not
    matter which of the two is passed first.
    """
    ablator = ablator.restrict([a, b])
    base = ablator.base().margin(a, b)
    sign = 1.0 if base >= 0 else -1.0
    for layer in _layers(ablator.evidence.units(), max_size):
        rankings = ablator.rank_without(layer)
        flips = [
            (sign * rankings[key].margin(a, b), key)
            for key in layer
            if sign * rankings[key].margin(a, b) < 0
        ]
        if flips:
            signed_margin, removed = min(flips)
            return PairFlip(
                removed=removed,
                labels=_labels(ablator, removed),
                base_margin=base,
                margin_after=sign * signed_margin,
            )
    return None


def flip_top(ablator: Ablator, max_size: int = 2) -> TopFlip | None:
    """Smallest omission that changes which candidate the model ranks first."""
    base_top = ablator.base().top
    for layer in _layers(ablator.evidence.units(), max_size):
        rankings = ablator.rank_without(layer)
        flips = [
            (-rankings[key].margin(rankings[key].top, base_top), key)
            for key in layer
            if rankings[key].top != base_top
        ]
        if flips:
            negative_margin, removed = min(flips)
            return TopFlip(
                removed=removed,
                labels=_labels(ablator, removed),
                base_top=base_top,
                new_top=rankings[removed].top,
                margin_after=-negative_margin,
            )
    return None


def _layers(units: Sequence[str], max_size: int) -> Iterable[list[tuple[str, ...]]]:
    if max_size < 1:
        raise ValueError("max_size must be at least 1")
    for size in range(1, min(max_size, len(units)) + 1):
        layer = [removal_key(c) for c in itertools.combinations(units, size)]
        if len(layer) > MAX_SUBSETS_PER_WAVE:
            raise ValueError(
                "removal sets of size %d over %d units exceed the wave budget of %d"
                % (size, len(units), MAX_SUBSETS_PER_WAVE)
            )
        yield layer


def _labels(ablator: Ablator, units: Sequence[str]) -> tuple[str, ...]:
    return tuple(ablator.evidence.label(u) for u in units)


def sampled_shapley(
    ablator: Ablator,
    a: str,
    b: str,
    permutations: int = 64,
    seed: int = 0,
) -> PairAttribution:
    """Shapley values estimated by sampling arrival orders, for evidence sets too
    large to enumerate.

    Exact Shapley visits all `2^n` coalitions, which stops being possible around
    twelve units. This walks `permutations` random orders instead, taking each
    unit's marginal contribution at the point it arrives, and averages those. Cost
    is `permutations * (units + 1)` contexts rather than `2^n`, so it grows linearly
    in both.

    Two properties are worth knowing. The estimate is unbiased, and its standard
    error is reported per unit, so a value can be read together with how firm it is;
    halving that error costs four times the permutations. And the values still sum
    exactly to the margin the evidence explains however few permutations are drawn,
    because each permutation's contributions telescope, so a small sample misplaces
    credit between units without inventing or losing any.

    `seed` is fixed by default: the same request returns the same explanation.
    """
    if permutations < 1:
        raise ValueError("permutations must be at least 1, got %d" % permutations)

    ablator = ablator.restrict([a, b])
    units = ablator.evidence.units()
    n = len(units)
    if n == 0:
        return PairAttribution(
            a=a,
            b=b,
            method=SAMPLED_SHAPLEY,
            base_margin=ablator.base().margin(a, b),
            empty_margin=ablator.base().margin(a, b),
            effects=(),
        )

    rng = random.Random(seed)
    orders = []
    for _ in range(permutations):
        order = list(units)
        rng.shuffle(order)
        orders.append(order)

    # Every prefix the walk will ask for, plus the singletons `solo` needs and the
    # two endpoints, scored in one wave.
    needed = {frozenset(), frozenset(units)}
    needed.update(frozenset({unit}) for unit in units)
    for order in orders:
        grown: set[str] = set()
        for unit in order:
            grown.add(unit)
            needed.add(frozenset(grown))

    coalitions = sorted(needed, key=lambda s: (len(s), sorted(s)))
    rankings = ablator.rank_without(
        [removal_key(set(units) - set(coalition)) for coalition in coalitions]
    )
    value = {
        coalition: rankings[removal_key(set(units) - set(coalition))].margin(a, b)
        for coalition in coalitions
    }

    contributions: dict[str, list[float]] = {unit: [] for unit in units}
    for order in orders:
        grown = set()
        for unit in order:
            before = frozenset(grown)
            grown.add(unit)
            contributions[unit].append(value[frozenset(grown)] - value[before])

    effects = []
    for unit in units:
        drawn = contributions[unit]
        mean = math.fsum(drawn) / len(drawn)
        if len(drawn) > 1:
            variance = math.fsum((x - mean) ** 2 for x in drawn) / (len(drawn) - 1)
        else:
            variance = 0.0
        stdev = math.sqrt(max(variance, 0.0))
        effects.append(
            EvidenceEffect(
                unit=unit,
                label=ablator.evidence.label(unit),
                delta=mean,
                spread=ContributionSpread(
                    low=min(drawn),
                    high=max(drawn),
                    stdev=stdev,
                    solo=value[frozenset({unit})] - value[frozenset()],
                    stderr=stdev / math.sqrt(len(drawn)),
                ),
            )
        )

    return PairAttribution(
        a=a,
        b=b,
        method=SAMPLED_SHAPLEY,
        base_margin=value[frozenset(units)],
        empty_margin=value[frozenset()],
        effects=tuple(effects),
    )
