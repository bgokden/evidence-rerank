"""A transparent stand-in for the neural ranker, used to check the attribution math.

Real scores are only reproducible to floating point, and the whole point of the
package is that the model is opaque, so correctness is tested against a scorer
whose contributions are known by construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class AdditiveScorer:
    """score(context, doc) = base[doc] + sum of weights[token, doc] for tokens present."""

    def __init__(
        self, base: Mapping[str, float], weights: Mapping[tuple[str, str], float]
    ) -> None:
        self.base = dict(base)
        self.weights = dict(weights)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for context, doc in pairs:
            total = self.base.get(doc, 0.0)
            for (token, target), weight in self.weights.items():
                if target == doc and token in context:
                    total += weight
            scores.append(total)
        return scores


class SaturatingScorer:
    """Rewards a doc once if any token of a redundant group is present.

    Two facts that say the same thing are jointly decisive but individually
    replaceable, which is the case leave-one-out cannot see.
    """

    def __init__(
        self, base: Mapping[str, float], groups: Mapping[str, tuple[Sequence[str], float]]
    ) -> None:
        self.base = dict(base)
        self.groups = dict(groups)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for context, doc in pairs:
            total = self.base.get(doc, 0.0)
            for target, (tokens, weight) in self.groups.items():
                if target == doc and any(token in context for token in tokens):
                    total += weight
            scores.append(total)
        return scores


class TableScorer:
    """Scores one doc from an explicit table keyed by the tokens present.

    Lets a test state an arbitrary interaction, including one where a token helps
    on its own and hurts in company, which no additive scorer can express.
    """

    TOKENS = ("alpha", "beta")

    def __init__(self, doc_a_scores: Mapping[tuple[str, ...], float]) -> None:
        self.doc_a_scores = {tuple(sorted(key)): value for key, value in doc_a_scores.items()}

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for context, doc in pairs:
            present = tuple(sorted(t for t in self.TOKENS if t in context))
            scores.append(self.doc_a_scores[present] if doc == "doc_a" else 0.0)
        return scores
