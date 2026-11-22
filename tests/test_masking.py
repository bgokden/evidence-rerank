from __future__ import annotations

import pytest
from toy_scorer import AdditiveScorer

from evrank.attribution import shapley
from evrank.evidence import Evidence, EvidenceSet
from evrank.masking import KindOnly, NeutralFiller, Omission
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer

EVIDENCE = EvidenceSet(
    [
        Evidence("s1", "searched for", "alpha", group="japanese"),
        Evidence("s2", "viewed", "alpha again", group="japanese"),
        Evidence("r1", "reserved", "beta"),
    ]
)
CANDIDATES = [Candidate("a", "A", "doc_a"), Candidate("b", "B", "doc_b")]
FILLER = "nothing of note"


def test_omission_shortens_the_context():
    context = Omission().context(EVIDENCE, ["japanese"])
    assert context.splitlines() == ["- reserved: beta"]


def test_neutral_filler_keeps_each_line_and_its_kind():
    context = NeutralFiller(text=FILLER).context(EVIDENCE, ["japanese"])
    assert context.splitlines() == [
        "- searched for: nothing of note",
        "- viewed: nothing of note",
        "- reserved: beta",
    ]


def test_neutral_filler_baseline_is_a_full_size_context():
    context = NeutralFiller(text=FILLER).context(EVIDENCE, EVIDENCE.units())
    assert context.splitlines() == [
        "- searched for: nothing of note",
        "- viewed: nothing of note",
        "- reserved: nothing of note",
    ]
    assert "alpha" not in context
    assert "beta" not in context


def test_removing_nothing_reproduces_the_production_context():
    for masker in (Omission(), NeutralFiller(text=FILLER)):
        assert masker.context(EVIDENCE, []) == EVIDENCE.render()


@pytest.mark.parametrize("masker", [Omission(), NeutralFiller(text=FILLER)])
def test_unknown_units_are_rejected_by_every_masker(masker):
    with pytest.raises(KeyError):
        masker.context(EVIDENCE, ["no_such_unit"])


def build(masker) -> Ablator:
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.5},
        weights={("alpha", "doc_a"): 1.0, ("beta", "doc_b"): 0.4},
    )
    return Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES, masker=masker)


def test_the_ablator_omits_by_default():
    assert isinstance(build(None).masker, Omission)


def test_restrict_carries_the_masking_strategy():
    masker = NeutralFiller(text=FILLER)
    assert build(masker).restrict(["a"]).masker is masker


def test_attribution_agrees_across_maskers_when_only_content_matters():
    # The toy scorer reads content and ignores length, so a filler line it does not
    # recognise must give the same answer as dropping the line altogether. Any
    # difference on a real model is the length artifact, not the evidence.
    omitted = shapley(build(Omission()), "a", "b")
    filled = shapley(build(NeutralFiller(text=FILLER)), "a", "b")
    assert {e.unit: e.delta for e in omitted.effects} == pytest.approx(
        {e.unit: e.delta for e in filled.effects}
    )
    assert omitted.empty_margin == pytest.approx(filled.empty_margin)


def test_a_filler_the_model_reacts_to_changes_the_answer():
    # A filler sharing wording with a candidate contributes an effect of its own,
    # which is the failure mode the docstring warns about.
    leaky = NeutralFiller(text="alpha")
    effects = {e.unit: e.delta for e in shapley(build(leaky), "a", "b").effects}
    assert effects["japanese"] == pytest.approx(0.0)


def test_baseline_word_counts_exposes_a_badly_sized_filler():
    from evrank.masking import baseline_word_counts

    unablated, omitted = baseline_word_counts(EVIDENCE, Omission())
    assert omitted == 0  # nothing is left once every fact is dropped
    _, matched = baseline_word_counts(EVIDENCE, NeutralFiller(text="three word filler"))
    _, oversized = baseline_word_counts(
        EVIDENCE, NeutralFiller(text=" ".join(["padding"] * 40))
    )
    assert abs(matched - unablated) < abs(oversized - unablated)


def test_kind_only_keeps_the_line_and_the_kind_but_drops_the_content():
    context = KindOnly().context(EVIDENCE, ["japanese"])
    assert context.splitlines() == [
        "- searched for:",
        "- viewed:",
        "- reserved: beta",
    ]


def test_kind_only_introduces_no_word_that_was_not_already_there():
    original = set(EVIDENCE.render().split())
    masked = set(KindOnly().context(EVIDENCE, EVIDENCE.units()).split())
    assert masked <= original


def test_kind_only_rejects_unknown_units():
    with pytest.raises(KeyError):
        KindOnly().context(EVIDENCE, ["nope"])
