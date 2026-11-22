"""Evidence-level counterfactual attribution for an opaque neural re-ranker.

`evrank.cross_encoder` is deliberately not imported here: the core path has no
deep-learning dependency, so a different ranker can be plugged in behind
`Scorer` without pulling in this package's own model.
"""

from evrank.attribution import (
    ContributionSpread,
    EvidenceEffect,
    PairAttribution,
    PairFlip,
    TopFlip,
    flip_pair,
    flip_top,
    leave_one_out,
    sampled_shapley,
    shapley,
)
from evrank.config import Settings, phrases_from_mapping
from evrank.evidence import Evidence, EvidenceSet
from evrank.explain import (
    RankPhrases,
    explain_pair,
    explain_shown,
    explain_top,
    languages,
    phrases_for,
    register_phrases,
)
from evrank.limits import ContextTooLongError, batch_size_for
from evrank.masking import (
    KindOnly,
    Masker,
    NeutralFiller,
    Omission,
    baseline_word_counts,
)
from evrank.ranking import Candidate, Ranking
from evrank.scoring import Ablator, CachingScorer, Scorer, rank
from evrank.standout import (
    CandidateStandout,
    StandoutEffect,
    exact_standout,
    sampled_standout,
    standout,
)

__all__ = [
    "Ablator",
    "CachingScorer",
    "Candidate",
    "CandidateStandout",
    "ContextTooLongError",
    "ContributionSpread",
    "Evidence",
    "EvidenceEffect",
    "EvidenceSet",
    "KindOnly",
    "Masker",
    "NeutralFiller",
    "Omission",
    "PairAttribution",
    "PairFlip",
    "RankPhrases",
    "Ranking",
    "Scorer",
    "Settings",
    "StandoutEffect",
    "TopFlip",
    "baseline_word_counts",
    "batch_size_for",
    "exact_standout",
    "explain_pair",
    "explain_shown",
    "explain_top",
    "flip_pair",
    "flip_top",
    "languages",
    "leave_one_out",
    "phrases_for",
    "phrases_from_mapping",
    "rank",
    "register_phrases",
    "sampled_shapley",
    "sampled_standout",
    "shapley",
    "standout",
]
