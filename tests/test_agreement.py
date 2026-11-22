from __future__ import annotations

import pytest
from toy_scorer import AdditiveScorer

from evrank.agreement import DEFAULT_MASKERS, compare_maskers
from evrank.attribution import leave_one_out
from evrank.evidence import Evidence, EvidenceSet
from evrank.masking import KindOnly, NeutralFiller, Omission
from evrank.ranking import Candidate
from evrank.scoring import CachingScorer

CANDIDATES = [Candidate("a", "A", "doc_a"), Candidate("b", "B", "doc_b")]
EVIDENCE = EvidenceSet(
    [
        Evidence("e1", "searched for", "alpha"),
        Evidence("e2", "reserved", "beta"),
        Evidence("e3", "constraint", "gamma"),
    ]
)


def content_scorer() -> CachingScorer:
    """Reads content only, so every masker that removes the token agrees."""
    return CachingScorer(
        AdditiveScorer(
            base={"doc_a": 0.0, "doc_b": 0.4},
            weights={("alpha", "doc_a"): 1.0, ("beta", "doc_b"): 0.5},
        )
    )


def filler_sensitive_scorer() -> CachingScorer:
    """Also reacts to a word that only the filler introduces, so maskers disagree."""
    return CachingScorer(
        AdditiveScorer(
            base={"doc_a": 0.0, "doc_b": 0.4},
            weights={
                ("alpha", "doc_a"): 1.0,
                ("beta", "doc_b"): 0.5,
                ("routine", "doc_b"): 3.0,
            },
        )
    )


def report(scorer, **kwargs):
    return compare_maskers(scorer, EVIDENCE, CANDIDATES, "a", "b", **kwargs)


def test_every_masker_is_reported_per_unit():
    result = report(content_scorer())
    assert result.maskers == ("omission", "kind", "filler")
    assert {unit.unit for unit in result.units} == {"e1", "e2", "e3"}
    assert set(result.units[0].deltas) == {"omission", "kind", "filler"}


def test_units_agree_when_the_model_only_reads_content():
    result = report(content_scorer())
    assert result.disputed() == ()
    assert {unit.unit for unit in result.agreed()} == {"e1", "e2", "e3"}
    assert next(u for u in result.units if u.unit == "e1").direction == "a"
    assert next(u for u in result.units if u.unit == "e2").direction == "b"


def test_a_unit_is_disputed_when_the_filler_itself_moves_the_margin():
    result = report(filler_sensitive_scorer())
    disputed = {unit.unit for unit in result.disputed()}
    assert disputed
    for unit in result.disputed():
        assert unit.direction == "mixed"
        assert unit.low < 0 < unit.high


def test_disputed_units_are_ordered_by_how_much_the_maskers_differ():
    result = report(filler_sensitive_scorer())
    spreads = [unit.spread for unit in result.disputed()]
    assert spreads == sorted(spreads, reverse=True)


def test_the_counterfactual_is_compared_across_maskers_too():
    result = report(content_scorer())
    assert result.flip_agrees
    assert {flip.removed for flip in result.flips.values()} == {("e1",)}


def test_a_unit_every_masker_measures_as_zero_is_agreement_not_dispute():
    result = report(content_scorer())
    gamma = next(unit for unit in result.units if unit.unit == "e3")
    assert set(gamma.deltas.values()) == {0.0}
    assert gamma.agrees
    assert gamma.direction == "none"


def test_a_counterfactual_only_some_maskers_find_does_not_count_as_agreement():
    # Here the filler's own word props up the candidate that would otherwise be
    # overtaken, so no removal reverses the pair under that masker.
    scorer = CachingScorer(
        AdditiveScorer(
            base={"doc_a": 0.0, "doc_b": 0.4},
            weights={("alpha", "doc_a"): 1.0, ("routine", "doc_a"): 1.0},
        )
    )
    result = report(scorer)
    assert result.flips["omission"].removed == ("e1",)
    assert result.flips["filler"].removed is None
    assert not result.flip_agrees


def test_the_estimator_is_the_caller_s_choice():
    result = report(content_scorer(), estimate=leave_one_out)
    assert result.method == "leave_one_out"


def test_a_custom_masker_set_is_honoured():
    result = report(
        content_scorer(), maskers=(("omission", Omission()), ("kind", KindOnly()))
    )
    assert result.maskers == ("omission", "kind")
    assert set(result.units[0].deltas) == {"omission", "kind"}


def test_an_empty_masker_set_is_rejected():
    with pytest.raises(ValueError, match="at least one masker"):
        report(content_scorer(), maskers=())


def test_the_default_set_covers_the_three_strategies():
    assert [name for name, _ in DEFAULT_MASKERS] == ["omission", "kind", "filler"]
    assert isinstance(dict(DEFAULT_MASKERS)["filler"], NeutralFiller)
