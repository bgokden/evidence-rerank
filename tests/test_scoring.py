from __future__ import annotations

from toy_scorer import AdditiveScorer

from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer, rank, removal_key

CANDIDATES = [Candidate("a", "A", "doc_a"), Candidate("b", "B", "doc_b")]
EVIDENCE = EvidenceSet(
    [Evidence("e1", "searched for", "alpha"), Evidence("e2", "reserved", "beta")]
)


def build() -> tuple[CachingScorer, Ablator]:
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0},
        weights={("alpha", "doc_a"): 1.0, ("beta", "doc_b"): 0.5},
    )
    scorer = CachingScorer(inner)
    return scorer, Ablator(scorer, EVIDENCE, CANDIDATES)


def test_rank_orders_candidates_by_score():
    scorer, _ = build()
    ranking = rank(scorer, EVIDENCE, CANDIDATES)
    assert ranking.order() == ("a", "b")
    assert ranking.margin("a", "b") == 0.5


def test_ablator_returns_one_ranking_per_removal_set():
    _, ablator = build()
    rankings = ablator.rank_without([(), ("e1",), ("e1", "e2")])
    assert rankings[()].margin("a", "b") == 0.5
    assert rankings[("e1",)].margin("a", "b") == -0.5
    assert rankings[("e1", "e2")].margin("a", "b") == 0.0


def test_removal_sets_are_keyed_independently_of_the_order_given():
    _, ablator = build()
    rankings = ablator.rank_without([("e2", "e1")])
    assert set(rankings) == {("e1", "e2")}
    assert removal_key(["e2", "e1"]) == ("e1", "e2")


def test_repeated_pairs_reach_the_model_once():
    scorer, ablator = build()
    ablator.rank_without([(), ("e1",), (), ("e1",)])
    assert scorer.requested_pairs == 8
    assert scorer.model_pairs == 4
    assert scorer.batches == 1


def test_restrict_keeps_the_named_candidates_and_the_shared_cache():
    scorer, ablator = build()
    ablator.rank_without([()])
    narrowed = ablator.restrict(["b"])
    assert [c.id for c in narrowed.candidates] == ["b"]
    assert narrowed.scorer is scorer
    narrowed.rank_without([()])
    assert scorer.model_pairs == 2


def test_restrict_rejects_unknown_candidates():
    _, ablator = build()
    try:
        ablator.restrict(["nope"])
    except KeyError as error:
        assert "nope" in str(error)
    else:
        raise AssertionError("expected KeyError")
