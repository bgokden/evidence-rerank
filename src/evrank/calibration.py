"""A learned stand-in for the floors that decide which measured effect gets printed.

The shipped gate is three constants -- a floor on the centred effect, one on the raw
effect, one on specificity -- swept by hand on a corpus. They work, and they are the
default, because they need no model and no file. But a constant cannot express a
condition like "this specificity is fine when the effect is small and suspicious when
it is this large", and that turns out to be where most of the remaining quality is.

Measured over four folds of a 48-category corpus, every category held out exactly
once, at matched first-row false positives: the floors print 41.0% correct sentences
and invent on 10.2% of rows that have no cause, and this scores 59.6% and 4.8% from
the same measurements. It gains no recall -- it cannot, since it only judges effects
the ablation already found -- and it costs no ranking accuracy, since ranking has
already happened by the time it runs. That last property is the reason it exists: the
other way to buy the same precision is to fine-tune the ranker, and six training runs
measured that at 1.5 to 2 points of ordinary ranking accuracy.

Three deliberate constraints on what may ship here.

It is a slate-level judgement, not a per-row one. Two of its strongest inputs are the
width of the page and the length of the context, so a single row cannot be scored in
isolation and `select` takes the whole slate.

It is trees in a JSON file, not a pickle. A pickle executes code when it loads, which
is an indefensible thing to put inside a system whose purpose is to let someone check
why they were shown something. The format here is readable, diffable, and evaluated
by the twenty lines below, so serving needs no modelling library at all.

And `delta > 0` is enforced here regardless of what the model says. Citing evidence
that *lowered* a candidate's score as the reason it was shown is incoherent whatever
a probability claims, and a learned rule that never did so on one corpus is not a
rule that never will. It cost nothing to add: across 192 citations in the measured
folds, the constraint never bound.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
from collections.abc import Mapping, Sequence

# In this order, and the order is part of the file format: a calibrator written by one
# version and read by another has to agree about which column is which, so the names
# are stored alongside the trees and checked on load.
FEATURES = (
    "standout",
    "delta",
    "lift",
    "specificity",
    "delta_share",
    "standout_share",
    "delta_scaled",
    "standout_scaled",
    "row_rank",
    "row_score_margin",
    "size",
    "units",
    "is_row_best",
)


@dataclasses.dataclass(frozen=True)
class Context:
    """What a feature needs that the effect itself does not carry.

    Everything here is derived from the slate, which is why `select` is the entry
    point rather than a method on one row.
    """

    scale: float
    row_rank: int
    row_score_margin: float
    size: int
    units: int
    largest_delta: float
    best_standout: float


def features_for(effect, context: Context) -> list[float]:
    """One effect as the vector the trees were grown on.

    Both the training script and the service call this, because a calibrator fed
    features derived even slightly differently from the ones it learned on is not
    being asked the question it can answer.
    """
    scale = context.scale or 1.0
    largest = context.largest_delta or 1.0
    best = abs(context.best_standout) or 1.0
    return [
        effect.standout,
        effect.delta,
        effect.lift,
        effect.specificity,
        abs(effect.delta) / largest,
        effect.standout / best,
        effect.delta / scale,
        effect.standout / scale,
        float(context.row_rank),
        context.row_score_margin,
        float(context.size),
        float(context.units),
        float(effect.standout >= context.best_standout),
    ]


@dataclasses.dataclass(frozen=True)
class Tree:
    """One decision tree, as four parallel arrays plus the leaf values."""

    left: tuple[int, ...]
    right: tuple[int, ...]
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    value: tuple[float, ...]

    def apply(self, vector: Sequence[float]) -> float:
        node = 0
        while self.left[node] != -1:
            node = (
                self.left[node]
                if vector[self.feature[node]] <= self.threshold[node]
                else self.right[node]
            )
        return self.value[node]


@dataclasses.dataclass(frozen=True)
class Calibrator:
    """A gradient-boosted ensemble, read from JSON and evaluated here.

    `base` is the ensemble's constant starting point and `rate` the shrinkage each
    tree was fitted with, so the raw score is `base + rate * sum(trees)` and the
    probability its logistic transform -- which is what the exporter verifies
    against the library that produced it, rather than assuming.
    """

    features: tuple[str, ...]
    trees: tuple[Tree, ...]
    base: float
    rate: float
    model: str = ""
    """Which re-ranker's effects this was fitted on. A calibrator belongs to a model
    the way the floors do: the quantities it reads are uncalibrated logits, so one
    fitted on another model is reading a different scale."""

    def raw(self, vector: Sequence[float]) -> float:
        return self.base + self.rate * math.fsum(tree.apply(vector) for tree in self.trees)

    def confidence(self, vector: Sequence[float]) -> float:
        """The probability that this effect is the row's real cause."""
        return 1.0 / (1.0 + math.exp(-self.raw(vector)))

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Calibrator:
        document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        names = tuple(document["features"])
        if names != FEATURES:
            raise ValueError(
                "%s was fitted on a different feature set: expected %s, found %s. "
                "Refit it rather than reordering the file, because the trees index "
                "these columns by position."
                % (path, ", ".join(FEATURES), ", ".join(names))
            )
        trees = tuple(
            Tree(
                left=tuple(tree["left"]),
                right=tuple(tree["right"]),
                feature=tuple(tree["feature"]),
                threshold=tuple(tree["threshold"]),
                value=tuple(tree["value"]),
            )
            for tree in document["trees"]
        )
        return cls(
            features=names,
            trees=trees,
            base=float(document["base"]),
            rate=float(document["rate"]),
            model=str(document.get("model", "")),
        )


def contexts(rows: Mapping[str, object]) -> dict[str, Context]:
    """One context per candidate, derived from the slate the rows came from."""
    scores = sorted(
        ((getattr(row, "score", 0.0), cid) for cid, row in rows.items()), reverse=True
    )
    rank = {cid: index for index, (_score, cid) in enumerate(scores)}
    values = sorted(score for score, _cid in scores)
    middle = (
        values[len(values) // 2]
        if len(values) % 2
        else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
    ) if values else 0.0
    out = {}
    for cid, row in rows.items():
        effects = tuple(row.effects)
        out[cid] = Context(
            scale=getattr(row, "scale", 0.0),
            row_rank=rank[cid],
            row_score_margin=getattr(row, "score", 0.0) - middle,
            size=len(rows),
            units=len(effects),
            largest_delta=max((abs(e.delta) for e in effects), default=0.0),
            best_standout=max((e.standout for e in effects), default=0.0),
        )
    return out


def select(
    rows: Mapping[str, object],
    calibrator: Calibrator,
    min_confidence: float = 0.5,
    top_reasons: int = 2,
) -> dict[str, tuple]:
    """Which effects each candidate would print, most confident first.

    The same shape `StandoutRow.favouring` returns, so a caller can swap one for the
    other. `min_confidence` is the dial `min_standout` is for the floors, and it is
    on the same footing: a number chosen at an operating point, which belongs to the
    corpus and the model it was chosen on.
    """
    found = contexts(rows)
    out = {}
    for cid, row in rows.items():
        kept = []
        for effect in row.effects:
            # Not negotiable, and not left to the model. See the module docstring.
            if effect.delta <= 0.0 or effect.standout <= 0.0:
                continue
            confidence = calibrator.confidence(features_for(effect, found[cid]))
            if confidence >= min_confidence:
                kept.append((confidence, effect))
        kept.sort(key=lambda pair: pair[0], reverse=True)
        out[cid] = tuple(effect for _confidence, effect in kept[:top_reasons or None])
    return out
