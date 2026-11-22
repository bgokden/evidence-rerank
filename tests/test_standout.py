from __future__ import annotations

import math

import pytest
from toy_scorer import AdditiveScorer

from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer
from evrank.standout import exact_standout, sampled_standout, standout

CANDIDATES = [
    Candidate("a", "A", "doc_a"),
    Candidate("b", "B", "doc_b"),
    Candidate("c", "C", "doc_c"),
]
EVIDENCE = EvidenceSet(
    [
        Evidence("s1", "searched for", "alpha"),
        Evidence("s2", "viewed", "beta"),
        Evidence("w1", "told us", "everywhere"),
    ]
)


def slate_ablator(evidence: EvidenceSet = EVIDENCE, candidates=None) -> Ablator:
    """`everywhere` lifts the whole slate; `alpha` and `beta` pick one each."""
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={
            ("alpha", "doc_a"): 1.0,
            ("beta", "doc_b"): 0.6,
            ("everywhere", "doc_a"): 0.5,
            ("everywhere", "doc_b"): 0.5,
            ("everywhere", "doc_c"): 0.5,
        },
    )
    return Ablator(CachingScorer(inner), evidence, candidates or CANDIDATES)


def test_the_effect_on_a_candidate_s_own_score_is_reported_as_measured():
    rows = standout(slate_ablator())
    effects = {e.unit: e.delta for e in rows["a"].effects}
    assert effects["s1"] == pytest.approx(1.0)
    assert effects["w1"] == pytest.approx(0.5)


def test_evidence_that_lifted_the_whole_slate_singles_nobody_out():
    rows = standout(slate_ablator())
    for cid in ("a", "b", "c"):
        lifted = next(e for e in rows[cid].effects if e.unit == "w1")
        assert lifted.delta == pytest.approx(0.5)
        assert lifted.standout == pytest.approx(0.0)


def test_each_candidate_stands_out_on_its_own_evidence():
    rows = standout(slate_ablator())
    assert rows["a"].favouring()[0].unit == "s1"
    assert rows["b"].favouring()[0].unit == "s2"


def test_a_candidate_no_evidence_singled_out_is_given_no_reason():
    # doc_c is only ever lifted with everything else, so nothing is specific to it.
    assert standout(slate_ablator())["c"].favouring() == ()


def test_evidence_that_lowered_a_candidate_is_never_a_reason_it_is_here():
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_b"): -1.0, ("alpha", "doc_c"): -1.0},
    )
    ablator = Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES)
    rows = standout(ablator)
    # `alpha` hurt b and c, so a stands out on it, but it did nothing for a.
    standing = next(e for e in rows["a"].effects if e.unit == "s1")
    assert standing.standout > 0.0
    assert standing.delta == pytest.approx(0.0)
    assert rows["a"].favouring() == ()


def test_one_pass_covers_the_whole_slate():
    ablator = slate_ablator()
    standout(ablator)
    # One context per unit plus the unablated one, each scored against every row.
    assert ablator.scorer.model_pairs == (len(EVIDENCE.units()) + 1) * len(CANDIDATES)
    assert ablator.scorer.batches == 1


def test_a_wholly_specific_unit_reads_the_same_on_a_long_slate_as_a_short_one():
    """The lift excludes the row being explained, so slate size is not a discount.

    Averaging it in instead would attenuate every effect by (width-1)/width: the
    same unit would read 0.5 against one rival and 0.8 against four, under one floor.
    """
    inner = AdditiveScorer(
        base={"doc_%s" % name: 0.0 for name in "abcde"},
        weights={("alpha", "doc_a"): 1.0},
    )
    evidence = EvidenceSet([Evidence("s1", "searched for", "alpha")])
    measured = {}
    for width in (2, 5):
        slate = [Candidate("c%d" % i, "n%d" % i, "doc_%s" % name)
                 for i, name in enumerate("abcde"[:width])]
        rows = standout(Ablator(CachingScorer(inner), evidence, slate))
        effect = next(e for e in rows["c0"].effects if e.unit == "s1")
        measured[width] = (effect.delta, effect.standout, effect.specificity)

    assert measured[2] == measured[5]
    assert measured[2] == pytest.approx((1.0, 1.0, 1.0))


def test_standing_out_is_meaningless_against_a_single_candidate():
    with pytest.raises(ValueError, match="at least two candidates"):
        standout(slate_ablator(candidates=[CANDIDATES[0]]))


def test_no_evidence_means_no_effects_rather_than_a_failure():
    rows = standout(slate_ablator(evidence=EvidenceSet([])))
    assert rows["a"].effects == ()
    assert sampled_standout(slate_ablator(evidence=EvidenceSet([])))["a"].effects == ()


REDUNDANT = EvidenceSet(
    [
        Evidence("s1", "searched for", "alpha"),
        Evidence("s2", "searched for", "alpha again"),
        Evidence("w1", "told us", "everywhere"),
    ]
)


def saturating_ablator() -> Ablator:
    """Two facts that carry the same information about doc_a, capped together.

    Either one alone lifts doc_a by 1.0; both together still lift it by 1.0. So
    removing either changes nothing, which is what leave-one-out cannot see.
    """

    class CappedScorer:
        def score(self, pairs):
            scores = []
            for context, document in pairs:
                value = 0.0
                if document == "doc_a" and ("alpha" in context):
                    value += 1.0
                if "everywhere" in context:
                    value += 0.5
                scores.append(value)
            return scores

    return Ablator(CachingScorer(CappedScorer()), REDUNDANT, CANDIDATES)


def test_leave_one_out_misses_evidence_that_repeats_itself():
    rows = standout(saturating_ablator())
    effects = {e.unit: e.delta for e in rows["a"].effects}
    # Each copy looks worthless because the other covers for it.
    assert effects["s1"] == pytest.approx(0.0)
    assert effects["s2"] == pytest.approx(0.0)
    assert rows["a"].favouring() == ()


def test_sampling_splits_the_credit_repeated_evidence_shares():
    rows = sampled_standout(saturating_ablator(), permutations=48, seed=1)
    effects = {e.unit: e.delta for e in rows["a"].effects}
    assert effects["s1"] == pytest.approx(0.5, abs=0.05)
    assert effects["s2"] == pytest.approx(0.5, abs=0.05)
    assert rows["a"].favouring()[0].unit in ("s1", "s2")


def test_the_values_still_account_for_the_candidate_s_whole_score():
    ablator = saturating_ablator()
    rows = sampled_standout(ablator, permutations=16, seed=2)
    for cid, result in rows.items():
        empty = ablator.rank_without([tuple(REDUNDANT.units())])
        baseline = empty[tuple(sorted(REDUNDANT.units()))].scores[cid]
        assert math.fsum(e.delta for e in result.effects) == pytest.approx(
            result.score - baseline, abs=1e-9
        )


def test_which_estimator_produced_the_numbers_is_recorded():
    assert standout(slate_ablator())["a"].method == "leave_one_out"
    assert sampled_standout(slate_ablator())["a"].method == "sampled_shapley"


def test_sampling_needs_at_least_one_permutation():
    with pytest.raises(ValueError, match="at least 1"):
        sampled_standout(slate_ablator(), permutations=0)


def test_the_exact_values_match_what_sampling_estimates():
    exact = exact_standout(saturating_ablator())
    sampled = sampled_standout(saturating_ablator(), permutations=400, seed=3)
    for unit in REDUNDANT.units():
        a = next(e for e in exact["a"].effects if e.unit == unit)
        b = next(e for e in sampled["a"].effects if e.unit == unit)
        assert a.delta == pytest.approx(b.delta, abs=0.05)
    assert exact["a"].method == "shapley"


def test_exact_values_split_repeated_evidence_down_the_middle():
    effects = {e.unit: e.delta for e in exact_standout(saturating_ablator())["a"].effects}
    assert effects["s1"] == pytest.approx(0.5)
    assert effects["s2"] == pytest.approx(0.5)


def test_exact_values_account_for_the_candidate_s_whole_score():
    ablator = saturating_ablator()
    rows = exact_standout(ablator)
    removed = tuple(sorted(REDUNDANT.units()))
    empty = ablator.rank_without([removed])[removed]
    for cid, result in rows.items():
        assert math.fsum(e.delta for e in result.effects) == pytest.approx(
            result.score - empty.scores[cid], abs=1e-9
        )


def test_enumeration_is_refused_past_the_point_it_is_affordable():
    units = EvidenceSet([Evidence("e%d" % i, "viewed", "x%d" % i) for i in range(14)])
    with pytest.raises(ValueError, match="14 units needs 16384 contexts"):
        exact_standout(slate_ablator(evidence=units))


def test_sampling_never_costs_more_than_enumerating_would():
    """Sampling draws orders but scores each coalition once, so 2^n caps them both.

    Which is why enumeration is free accuracy below the cap rather than a trade: at
    this size the two measurements score the very same contexts.
    """
    exact, sampled = slate_ablator(), slate_ablator()
    exact_standout(exact)
    sampled_standout(sampled, permutations=32)
    assert sampled.scorer.model_pairs == exact.scorer.model_pairs
    assert exact.scorer.model_pairs == 2 ** len(EVIDENCE.units()) * len(CANDIDATES)


def test_a_reason_is_not_invented_where_the_model_drew_no_distinction():
    """Every candidate within a hair of the others: there is nothing to explain.

    One of them still has the largest tiny effect, and its share of that nothing can
    be anything at all, so the share alone lets a reason through.
    """
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_a"): 0.009, ("alpha", "doc_b"): 0.006},
    )
    rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
    assert rows["a"].favouring(min_specificity=0.1)[0].unit == "s1"
    assert rows["a"].favouring(min_specificity=0.1, min_effect=0.1) == ()


def test_the_size_floor_catches_what_the_share_floor_cannot():
    """A unit that barely helped this row while hurting the others.

    `standout` comes out healthy because the lift is negative, so only a floor on the
    raw effect sees that nothing happened here.
    """
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_a"): 0.03, ("alpha", "doc_b"): -1.0, ("alpha", "doc_c"): -1.0},
    )
    rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
    effect = next(e for e in rows["a"].effects if e.unit == "s1")
    assert effect.standout > 0.6
    assert effect.specificity > 20.0
    assert rows["a"].favouring(min_specificity=0.1) != ()
    assert rows["a"].favouring(min_specificity=0.1, min_effect=0.1) == ()


def test_the_centred_floor_catches_what_the_size_floor_cannot():
    """A unit that lifted the whole slate and this row barely more than the rest.

    The mirror of the test above. `delta` is large, so a floor on it passes the unit
    happily; `standout` is almost nothing, because the lift took it away. This is the
    on-topic request for something else -- the realistic way to invent a reason --
    and only a floor on the centred effect sees it.
    """
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={
            ("alpha", "doc_a"): 1.02,
            ("alpha", "doc_b"): 1.0,
            ("alpha", "doc_c"): 1.0,
        },
    )
    rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
    effect = next(e for e in rows["a"].effects if e.unit == "s1")
    assert effect.delta == pytest.approx(1.02), "a large effect on its own score"
    assert effect.standout == pytest.approx(0.02), "and almost none of it specific"
    assert rows["a"].favouring(min_effect=0.5) != (), "the size floor lets it through"
    assert rows["a"].favouring(min_standout=0.5) == (), "the centred floor does not"


def test_the_two_floors_are_both_needed_and_neither_subsumes_the_other():
    """Each floor alone admits a unit the other rejects.

    Measured on real slates, keeping both costs nothing, which is why both ship. The
    pair is the point: one guards against a unit that did nothing here, the other
    against a unit that did the same thing everywhere.
    """
    slate_lifting = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_a"): 1.02, ("alpha", "doc_b"): 1.0, ("alpha", "doc_c"): 1.0},
    )
    barely_touched = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_a"): 0.03, ("alpha", "doc_b"): -1.0, ("alpha", "doc_c"): -1.0},
    )
    both = {"min_effect": 0.5, "min_standout": 0.5, "min_specificity": 0.1}
    for inner in (slate_lifting, barely_touched):
        rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
        assert rows["a"].favouring(**both) == ()


def test_the_dynamic_floor_is_measured_against_this_candidate_s_own_largest_effect():
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        # On doc_a one unit does 2.0 and another 0.3, so the second carries 15%.
        weights={("alpha", "doc_a"): 2.0, ("beta", "doc_a"): 0.3},
    )
    rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
    assert [e.unit for e in rows["a"].favouring()] == ["s1", "s2"]
    assert [e.unit for e in rows["a"].favouring(min_share=0.1)] == ["s1", "s2"]
    assert [e.unit for e in rows["a"].favouring(min_share=0.2)] == ["s1"]


def test_the_dynamic_floor_counts_harm_as_well_as_help():
    """The largest effect on a row can be one that pushed it down.

    A unit worth +0.3 beside one worth -2.0 carries 15% of that row's movement, not
    100%, and reading only the positive side would call it the whole story.
    """
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={("alpha", "doc_a"): -2.0, ("beta", "doc_a"): 0.3},
    )
    rows = standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))
    assert rows["a"]._largest_effect() == pytest.approx(2.0)
    assert rows["a"].favouring(min_share=0.2) == ()


def test_the_scale_is_the_request_s_own_median_effect():
    rows = standout(slate_ablator())
    everything = sorted(
        abs(effect.standout) for result in rows.values() for effect in result.effects
    )
    expected = everything[len(everything) // 2]
    # One yardstick for the whole request, not one per candidate.
    assert len({result.scale for result in rows.values()}) == 1
    assert rows["a"].scale == pytest.approx(expected, abs=1e-9)
