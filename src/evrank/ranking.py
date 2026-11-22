"""Candidates and the order the ranker puts them in."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One item to be ranked. `text` is the document side of the ranker."""

    id: str
    name: str
    text: str


@dataclasses.dataclass(frozen=True)
class Ranking:
    """Scores for a candidate list, highest first."""

    scores: Mapping[str, float]
    candidates: Mapping[str, Candidate]

    @classmethod
    def of(cls, candidates: Sequence[Candidate], scores: Sequence[float]) -> Ranking:
        if len(candidates) != len(scores):
            raise ValueError("got %d candidates and %d scores" % (len(candidates), len(scores)))
        return cls(
            scores={c.id: float(s) for c, s in zip(candidates, scores)},
            candidates={c.id: c for c in candidates},
        )

    def order(self) -> tuple[str, ...]:
        """Candidate ids, best first. Ties break on id so the order is reproducible."""
        return tuple(sorted(self.scores, key=lambda cid: (-self.scores[cid], cid)))

    @property
    def top(self) -> str:
        return self.order()[0]

    def margin(self, a: str, b: str) -> float:
        return self.scores[a] - self.scores[b]

    def name(self, cid: str) -> str:
        return self.candidates[cid].name
