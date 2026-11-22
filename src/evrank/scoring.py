"""The boundary between this package and the ranking model.

Everything downstream treats the model as a black box behind `Scorer`: a batched
function from (context, document) pairs to real-valued scores. No access to
weights, gradients, attention or embedding coordinates is assumed, which is what
makes the attribution in `attribution.py` model-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from evrank.evidence import EvidenceSet
from evrank.masking import Masker, Omission
from evrank.ranking import Candidate, Ranking


def removal_key(units: Sequence[str] | set[str]) -> tuple[str, ...]:
    """Canonical form of a removal set, so callers and `Ablator` agree on dict keys."""
    return tuple(sorted(set(units)))


class Scorer(Protocol):
    """Scores (context, document) pairs. Higher is better. Must be deterministic."""

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


class CachingScorer:
    """Deduplicates and memoizes pairs before they reach the model.

    Subset enumeration asks for the same (context, document) pair many times: the
    same ablated context recurs across coalitions, and the base context recurs in
    every method. Counters are kept so the real inference cost of an explanation
    can be reported instead of estimated.
    """

    def __init__(self, inner: Scorer) -> None:
        self._inner = inner
        self._cache: dict[tuple[str, str], float] = {}
        self.model_pairs = 0
        self.requested_pairs = 0
        self.batches = 0

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.requested_pairs += len(pairs)
        missing: list[tuple[str, str]] = []
        for pair in pairs:
            if pair not in self._cache and pair not in missing:
                missing.append(pair)
        if missing:
            values = self._inner.score(missing)
            if len(values) != len(missing):
                raise ValueError(
                    "scorer returned %d scores for %d pairs" % (len(values), len(missing))
                )
            self._cache.update(zip(missing, values))
            self.model_pairs += len(missing)
            self.batches += 1
        return [self._cache[pair] for pair in pairs]

    def reset_counters(self) -> None:
        self.model_pairs = 0
        self.requested_pairs = 0
        self.batches = 0


def rank(scorer: Scorer, evidence: EvidenceSet, candidates: Sequence[Candidate]) -> Ranking:
    """Score a candidate list under one evidence set. This is the production path."""
    context = evidence.render()
    return Ranking.of(candidates, scorer.score([(context, c.text) for c in candidates]))


class Ablator:
    """Ranks a fixed candidate list under many evidence subsets.

    One call per wave of subsets, so a whole explanation costs a handful of batched
    inferences rather than one inference per subset.
    """

    def __init__(
        self,
        scorer: Scorer,
        evidence: EvidenceSet,
        candidates: Sequence[Candidate],
        masker: Masker | None = None,
    ) -> None:
        self.scorer = scorer
        self.evidence = evidence
        self.candidates = tuple(candidates)
        self.masker = masker if masker is not None else Omission()

    def rank_without(
        self, removals: Sequence[Sequence[str]]
    ) -> dict[tuple[str, ...], Ranking]:
        """Rank once per removal set, in a single batched call.

        Keys are the removal sets as sorted tuples; `()` is the unablated context.
        """
        keys = [removal_key(r) for r in removals]
        contexts = {key: self.masker.context(self.evidence, key) for key in set(keys)}
        pairs = [(contexts[key], c.text) for key in keys for c in self.candidates]
        flat = self.scorer.score(pairs)
        width = len(self.candidates)
        return {
            key: Ranking.of(self.candidates, flat[i * width : (i + 1) * width])
            for i, key in enumerate(keys)
        }

    def restrict(self, candidate_ids: Sequence[str]) -> Ablator:
        """An ablator over a subset of the candidates, sharing this one's scorer cache.

        Pairwise attribution needs two candidates, not the whole result page, so
        narrowing here is what keeps an explanation to a couple of batched calls.
        """
        wanted = list(candidate_ids)
        by_id = {c.id: c for c in self.candidates}
        missing = [cid for cid in wanted if cid not in by_id]
        if missing:
            raise KeyError("unknown candidate ids: %s" % missing)
        return Ablator(
            self.scorer, self.evidence, [by_id[cid] for cid in wanted], masker=self.masker
        )

    def base(self) -> Ranking:
        return self.rank_without([()])[()]
