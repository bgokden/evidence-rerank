"""Which attributions survive a change of ablation strategy.

Every number this package produces depends on a choice the data cannot make: how a
removed fact is represented to the model. Omission shortens the context, neutral
filler puts a sentence in it that has to be written in some language, and kind-only
drops the content but keeps the line. Measured on real contexts, the largest effects
keep their direction under all three while small ones change sign.

That makes the disagreement itself the useful quantity. Running the same attribution
under several maskers and reporting where they agree separates "this evidence pushed
the model toward A" from "this number is an artifact of how we asked".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

from evrank.attribution import PairAttribution, flip_pair, shapley
from evrank.masking import KindOnly, Masker, NeutralFiller, Omission
from evrank.ranking import Candidate
from evrank.scoring import Ablator, Scorer

DEFAULT_MASKERS: tuple[tuple[str, Masker], ...] = (
    ("omission", Omission()),
    ("kind", KindOnly()),
    ("filler", NeutralFiller()),
)

Estimator = Callable[[Ablator, str, str], PairAttribution]


@dataclasses.dataclass(frozen=True)
class UnitAgreement:
    """One evidence unit's effect as measured under each masking strategy."""

    unit: str
    label: str
    deltas: Mapping[str, float]

    @property
    def low(self) -> float:
        return min(self.deltas.values())

    @property
    def high(self) -> float:
        return max(self.deltas.values())

    @property
    def agrees(self) -> bool:
        """True unless the maskers land on opposite sides of zero.

        Measuring the same effect as zero under every masker is agreement, not
        dispute: they concur that the evidence did nothing.
        """
        return not (self.low < 0.0 < self.high)

    @property
    def direction(self) -> str:
        """`a`, `b`, `none` when every masker measured no effect, or `mixed`."""
        if self.low < 0.0 < self.high:
            return "mixed"
        if self.high > 0.0:
            return "a"
        if self.low < 0.0:
            return "b"
        return "none"

    @property
    def spread(self) -> float:
        return self.high - self.low


@dataclasses.dataclass(frozen=True)
class FlipUnderMasker:
    removed: tuple[str, ...] | None
    margin_after: float | None


@dataclasses.dataclass(frozen=True)
class AgreementReport:
    a: str
    b: str
    method: str
    maskers: tuple[str, ...]
    base_margin: float
    units: tuple[UnitAgreement, ...]
    flips: Mapping[str, FlipUnderMasker]

    def agreed(self) -> tuple[UnitAgreement, ...]:
        """Units whose direction every masker agrees on, strongest first."""
        return tuple(
            sorted(
                (unit for unit in self.units if unit.agrees),
                key=lambda unit: -min(abs(unit.low), abs(unit.high)),
            )
        )

    def disputed(self) -> tuple[UnitAgreement, ...]:
        """Units whose direction depends on how the evidence was removed."""
        return tuple(sorted((u for u in self.units if not u.agrees), key=lambda u: -u.spread))

    @property
    def flip_agrees(self) -> bool:
        """True when every masker found the same minimal removal, and found one."""
        removals = {flip.removed for flip in self.flips.values()}
        return len(removals) == 1 and None not in removals


def compare_maskers(
    scorer: Scorer,
    evidence,
    candidates: Sequence[Candidate],
    a: str,
    b: str,
    maskers: Sequence[tuple[str, Masker]] = DEFAULT_MASKERS,
    estimate: Estimator = shapley,
    max_flip_size: int = 2,
) -> AgreementReport:
    """Attribute the same pair under each masker and report where they agree.

    `estimate` is the attribution to repeat, so the caller chooses between exact
    Shapley, sampling and leave-one-out rather than this deciding for them. Cost is
    that attribution once per masker; the unablated context is shared, so the
    scorer's cache absorbs a little of it.
    """
    if not maskers:
        raise ValueError("at least one masker is required")

    deltas: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}
    flips: dict[str, FlipUnderMasker] = {}
    base_margin = 0.0
    method = ""

    for name, masker in maskers:
        ablator = Ablator(scorer, evidence, candidates, masker=masker)
        attribution = estimate(ablator, a, b)
        method = attribution.method
        base_margin = attribution.base_margin
        for effect in attribution.effects:
            deltas.setdefault(effect.unit, {})[name] = effect.delta
            labels[effect.unit] = effect.label
        flip = flip_pair(ablator, a, b, max_size=max_flip_size)
        flips[name] = FlipUnderMasker(
            removed=flip.removed if flip else None,
            margin_after=flip.margin_after if flip else None,
        )

    units = tuple(
        UnitAgreement(unit=unit, label=labels[unit], deltas=dict(by_masker))
        for unit, by_masker in deltas.items()
    )
    return AgreementReport(
        a=a,
        b=b,
        method=method,
        maskers=tuple(name for name, _ in maskers),
        base_margin=base_margin,
        units=units,
        flips=flips,
    )
