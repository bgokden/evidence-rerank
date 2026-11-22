from __future__ import annotations

import pytest
from toy_scorer import AdditiveScorer, SaturatingScorer, TableScorer

from evrank.attribution import (
    flip_pair,
    flip_top,
    leave_one_out,
    sampled_shapley,
    shapley,
)
from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer

CANDIDATES = [Candidate("a", "A", "doc_a"), Candidate("b", "B", "doc_b")]


def additive_ablator(evidence: EvidenceSet, candidates=None) -> Ablator:
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.5, "doc_c": 0.2},
        weights={
            ("alpha", "doc_a"): 1.0,
            ("beta", "doc_b"): 0.4,
            ("gamma", "doc_c"): 0.9,
        },
    )
    return Ablator(CachingScorer(inner), evidence, candidates or CANDIDATES)


def three_facts() -> EvidenceSet:
    return EvidenceSet(
        [
            Evidence("e1", "searched for", "alpha"),
            Evidence("e2", "reserved", "beta"),
            Evidence("e3", "constraint", "gamma"),
        ]
    )


def redundant_ablator(group: str | None) -> Ablator:
    evidence = EvidenceSet(
        [
            Evidence("e1", "searched for", "alpha", group=group),
            Evidence("e2", "searched for", "beta", group=group),
        ]
    )
    inner = SaturatingScorer(
        base={"doc_a": 0.0, "doc_b": 0.6},
        groups={"doc_a": (("alpha", "beta"), 1.0)},
    )
    return Ablator(CachingScorer(inner), evidence, CANDIDATES)


def effects_of(attribution):
    return {effect.unit: effect.delta for effect in attribution.effects}


def test_leave_one_out_recovers_known_contributions():
    attribution = leave_one_out(additive_ablator(three_facts()), "a", "b")
    assert attribution.base_margin == pytest.approx(0.1)
    assert attribution.empty_margin == pytest.approx(-0.5)
    assert effects_of(attribution) == pytest.approx({"e1": 1.0, "e2": -0.4, "e3": 0.0})


def test_delta_is_positive_when_evidence_supports_the_higher_ranked_item():
    attribution = leave_one_out(additive_ablator(three_facts()), "a", "b")
    supporting = attribution.by_effect()[0]
    assert supporting.unit == "e1"
    assert supporting.delta > 0


def test_shapley_matches_leave_one_out_on_an_additive_model():
    ablator = additive_ablator(three_facts())
    assert effects_of(shapley(ablator, "a", "b")) == pytest.approx(
        effects_of(leave_one_out(ablator, "a", "b"))
    )


def test_shapley_effects_sum_to_the_margin_the_evidence_explains():
    attribution = shapley(redundant_ablator(group=None), "a", "b")
    assert attribution.total_effect() == pytest.approx(
        attribution.base_margin - attribution.empty_margin
    )


def test_leave_one_out_is_blind_to_redundant_evidence():
    attribution = leave_one_out(redundant_ablator(group=None), "a", "b")
    assert effects_of(attribution) == pytest.approx({"e1": 0.0, "e2": 0.0})
    assert attribution.base_margin - attribution.empty_margin == pytest.approx(1.0)


def test_shapley_splits_credit_between_redundant_facts():
    attribution = shapley(redundant_ablator(group=None), "a", "b")
    assert effects_of(attribution) == pytest.approx({"e1": 0.5, "e2": 0.5})


def test_grouping_redundant_facts_restores_leave_one_out():
    attribution = leave_one_out(redundant_ablator(group="japanese"), "a", "b")
    assert effects_of(attribution) == pytest.approx({"japanese": 1.0})


def test_shapley_refuses_to_enumerate_too_many_units():
    evidence = EvidenceSet(
        [Evidence("e%d" % i, "searched for", "token%d" % i) for i in range(6)]
    )
    with pytest.raises(ValueError, match="group evidence"):
        shapley(additive_ablator(evidence), "a", "b", max_units=5)


def test_flip_pair_finds_the_single_fact_that_carries_the_ordering():
    flip = flip_pair(additive_ablator(three_facts()), "a", "b")
    assert flip is not None
    assert flip.removed == ("e1",)
    assert flip.base_margin > 0 > flip.margin_after


def test_flip_pair_reports_nothing_when_no_small_omission_reverses_the_order():
    irrelevant = EvidenceSet([Evidence("e3", "constraint", "gamma")])
    assert flip_pair(additive_ablator(irrelevant), "a", "b") is None


def test_flip_pair_does_not_care_which_candidate_is_named_first():
    forward = flip_pair(additive_ablator(three_facts()), "a", "b")
    backward = flip_pair(additive_ablator(three_facts()), "b", "a")
    assert forward is not None and backward is not None
    assert forward.removed == backward.removed == ("e1",)
    assert forward.base_margin == -backward.base_margin
    assert forward.margin_after == -backward.margin_after


def test_flip_pair_grows_the_removal_set_only_when_it_has_to():
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0},
        weights={("alpha", "doc_a"): 0.6, ("beta", "doc_a"): 0.6, ("gamma", "doc_b"): 0.5},
    )
    ablator = Ablator(CachingScorer(inner), three_facts(), CANDIDATES)
    assert flip_pair(ablator, "a", "b", max_size=1) is None
    flip = flip_pair(ablator, "a", "b", max_size=2)
    assert flip is not None
    assert flip.removed == ("e1", "e2")


def test_flip_top_names_the_candidate_that_takes_first_place():
    candidates = CANDIDATES + [Candidate("c", "C", "doc_c")]
    ablator = additive_ablator(three_facts(), candidates)
    assert ablator.base().top == "c"
    flip = flip_top(ablator)
    assert flip is not None
    assert flip.removed == ("e3",)
    assert flip.new_top == "a"
    assert flip.margin_after > 0


def test_pair_attribution_only_scores_the_pair_being_explained():
    candidates = CANDIDATES + [Candidate("c", "C", "doc_c")]
    ablator = additive_ablator(three_facts(), candidates)
    shapley(ablator, "a", "b")
    # 2^3 evidence subsets for 2 of the 3 candidates, in one batched call.
    assert ablator.scorer.model_pairs == 2 * 2**3
    assert ablator.scorer.batches == 1


def test_leave_one_out_needs_a_single_batched_call():
    ablator = additive_ablator(three_facts())
    leave_one_out(ablator, "a", "b")
    assert ablator.scorer.batches == 1


def test_counterfactual_search_reuses_the_attribution_cache():
    ablator = additive_ablator(three_facts())
    shapley(ablator, "a", "b")
    before = ablator.scorer.model_pairs
    flip_pair(ablator, "a", "b", max_size=3)
    assert ablator.scorer.model_pairs == before


def table_ablator(doc_a_scores) -> Ablator:
    """Two units over a hand-written score table, so Shapley values are checkable."""
    evidence = EvidenceSet(
        [
            Evidence("e1", "searched for", "alpha"),
            Evidence("e2", "reserved", "beta"),
        ]
    )
    return Ablator(CachingScorer(TableScorer(doc_a_scores)), evidence, CANDIDATES)


def test_spread_is_flat_when_the_model_is_additive():
    attribution = shapley(additive_ablator(three_facts()), "a", "b")
    for effect in attribution.effects:
        assert effect.spread is not None
        assert effect.spread.stdev == pytest.approx(0.0)
        assert effect.spread.low == pytest.approx(effect.delta)
        assert effect.spread.high == pytest.approx(effect.delta)
        assert effect.spread.solo == pytest.approx(effect.delta)
        assert not effect.spread.reverses


def test_spread_shows_redundancy_as_variation_without_reversal():
    attribution = shapley(redundant_ablator(group=None), "a", "b")
    spread = {e.unit: e.spread for e in attribution.effects}
    assert spread["e1"].low == pytest.approx(0.0)
    assert spread["e1"].high == pytest.approx(1.0)
    assert spread["e1"].solo == pytest.approx(1.0)
    assert spread["e1"].stdev == pytest.approx(0.5)
    assert not spread["e1"].reverses


def test_spread_flags_a_unit_whose_direction_depends_on_the_rest():
    # alpha helps doc_a by 4 on its own and hurts it by 2 once beta is present.
    attribution = shapley(
        table_ablator({(): 0.0, ("alpha",): 4.0, ("beta",): 0.0, ("alpha", "beta"): -2.0}),
        "a",
        "b",
    )
    effects = {e.unit: e for e in attribution.effects}
    assert effects["e1"].delta == pytest.approx(1.0)
    assert effects["e1"].spread.solo == pytest.approx(4.0)
    assert effects["e1"].spread.low == pytest.approx(-2.0)
    assert effects["e1"].spread.high == pytest.approx(4.0)
    assert effects["e1"].spread.stdev == pytest.approx(3.0)
    assert effects["e1"].spread.reverses
    assert attribution.reversing_effects() == (effects["e1"],)


def test_a_unit_that_only_ever_hurts_is_not_called_unstable():
    attribution = shapley(
        table_ablator({(): 0.0, ("alpha",): 4.0, ("beta",): 0.0, ("alpha", "beta"): -2.0}),
        "a",
        "b",
    )
    effects = {e.unit: e for e in attribution.effects}
    assert effects["e2"].delta == pytest.approx(-3.0)
    assert effects["e2"].spread.high == pytest.approx(0.0)
    assert not effects["e2"].spread.reverses


def test_leave_one_out_reports_no_spread_because_it_measures_one_coalition():
    attribution = leave_one_out(additive_ablator(three_facts()), "a", "b")
    assert all(effect.spread is None for effect in attribution.effects)


def test_sampled_shapley_approaches_the_exact_values():
    ablator = additive_ablator(three_facts())
    exact = effects_of(shapley(ablator, "a", "b"))
    sampled = effects_of(sampled_shapley(ablator, "a", "b", permutations=64))
    assert sampled == pytest.approx(exact, abs=1e-9)


def test_sampled_values_sum_exactly_however_few_permutations():
    # Each permutation's contributions telescope to v(all) - v(none), so the mean
    # does too. A small sample misplaces credit; it never invents or loses any.
    for permutations in (1, 2, 7):
        attribution = sampled_shapley(
            redundant_ablator(group=None), "a", "b", permutations=permutations
        )
        assert attribution.total_effect() == pytest.approx(
            attribution.base_margin - attribution.empty_margin
        )


def test_sampling_reports_a_standard_error_and_exact_enumeration_does_not():
    ablator = redundant_ablator(group=None)
    for effect in shapley(ablator, "a", "b").effects:
        assert effect.spread.stderr == 0.0
    for effect in sampled_shapley(ablator, "a", "b", permutations=16).effects:
        assert effect.spread.stderr > 0.0


def test_more_permutations_shrink_the_standard_error():
    ablator = redundant_ablator(group=None)
    few = sampled_shapley(ablator, "a", "b", permutations=16).effects[0].spread.stderr
    many = sampled_shapley(ablator, "a", "b", permutations=256).effects[0].spread.stderr
    assert many < few


def test_the_same_seed_gives_the_same_explanation():
    ablator = redundant_ablator(group=None)
    first = effects_of(sampled_shapley(ablator, "a", "b", permutations=8, seed=3))
    second = effects_of(sampled_shapley(ablator, "a", "b", permutations=8, seed=3))
    other = effects_of(sampled_shapley(ablator, "a", "b", permutations=8, seed=4))
    assert first == pytest.approx(second)
    assert first != pytest.approx(other)


def test_sampling_works_where_exact_enumeration_refuses():
    evidence = EvidenceSet(
        [Evidence("e%d" % i, "searched for", "token%d" % i) for i in range(14)]
    )
    ablator = additive_ablator(evidence)
    with pytest.raises(ValueError, match="group evidence"):
        shapley(ablator, "a", "b")
    attribution = sampled_shapley(ablator, "a", "b", permutations=8)
    assert len(attribution.effects) == 14
    assert attribution.method == "sampled_shapley"


def test_sampling_rejects_a_nonsense_permutation_count():
    with pytest.raises(ValueError, match="permutations must be at least 1"):
        sampled_shapley(additive_ablator(three_facts()), "a", "b", permutations=0)


def test_sampling_converges_on_a_model_where_permutations_disagree():
    # Redundant evidence contributes 1.0 when it arrives first and 0.0 when it
    # arrives second, so the estimate really is an average over orders.
    ablator = redundant_ablator(group=None)
    exact = effects_of(shapley(ablator, "a", "b"))
    sampled = sampled_shapley(ablator, "a", "b", permutations=400)
    assert effects_of(sampled) == pytest.approx(exact, abs=0.08)
    assert all(effect.spread.stderr < 0.05 for effect in sampled.effects)
