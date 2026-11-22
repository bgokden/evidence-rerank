"""The learned gate: its file format, its arithmetic, and what it refuses to do.

`examples/calibrator.py` measures whether it is *better* than the floors. These are
the properties it has to hold whatever it scores: that the file it loads means what
it says, that the trees are evaluated correctly, that the judgement is made over the
slate rather than a row, and that no probability can talk it into citing evidence
which lowered the candidate it is explaining.
"""

from __future__ import annotations

import json
import math

import pytest
from toy_scorer import AdditiveScorer

from evrank.calibration import (
    FEATURES,
    Calibrator,
    Context,
    Tree,
    contexts,
    features_for,
    select,
)
from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer
from evrank.standout import standout

CANDIDATES = [
    Candidate("a", "A", "doc_a"),
    Candidate("b", "B", "doc_b"),
    Candidate("c", "C", "doc_c"),
]
EVIDENCE = EvidenceSet(
    [
        Evidence("s1", "searched for", "alpha"),
        Evidence("w1", "told us", "everywhere"),
    ]
)


def rows():
    """`everywhere` lifts the whole slate; `alpha` picks out one candidate."""
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.0, "doc_c": 0.0},
        weights={
            ("alpha", "doc_a"): 1.0,
            ("everywhere", "doc_a"): 0.5,
            ("everywhere", "doc_b"): 0.5,
            ("everywhere", "doc_c"): 0.5,
        },
    )
    return standout(Ablator(CachingScorer(inner), EVIDENCE, CANDIDATES))


def constant(value: float) -> Calibrator:
    """A calibrator that answers `value` as a raw score for everything."""
    leaf = Tree(left=(-1,), right=(-1,), feature=(-2,), threshold=(-2.0,), value=(0.0,))
    return Calibrator(features=FEATURES, trees=(leaf,), base=value, rate=1.0)


def on_feature(index: int, cut: float, below: float, above: float) -> Calibrator:
    """One stump splitting on a single feature, so a test can aim it."""
    tree = Tree(
        left=(1, -1, -1),
        right=(2, -1, -1),
        feature=(index, -2, -2),
        threshold=(cut, -2.0, -2.0),
        value=(0.0, below, above),
    )
    return Calibrator(features=FEATURES, trees=(tree,), base=0.0, rate=1.0)


def test_a_file_fitted_on_other_features_is_refused(tmp_path):
    """The trees index columns by position, so a reordered file is a wrong answer.

    Silently accepting it would feed every tree the wrong number and still return a
    probability, which is the worst available failure: plausible and wrong.
    """
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "features": ["standout", "delta"], "trees": [], "base": 0.0, "rate": 0.1,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="different feature set"):
        Calibrator.load(path)


def test_the_file_round_trips_through_the_loader(tmp_path):
    path = tmp_path / "c.json"
    original = on_feature(FEATURES.index("standout"), 0.4, -3.0, 3.0)
    path.write_text(json.dumps({
        "features": list(original.features),
        "trees": [{
            "left": list(original.trees[0].left),
            "right": list(original.trees[0].right),
            "feature": list(original.trees[0].feature),
            "threshold": list(original.trees[0].threshold),
            "value": list(original.trees[0].value),
        }],
        "base": original.base,
        "rate": original.rate,
        "model": "toy",
    }), encoding="utf-8")
    loaded = Calibrator.load(path)
    assert loaded.model == "toy"
    vector = [0.0] * len(FEATURES)
    vector[FEATURES.index("standout")] = 1.0
    assert loaded.raw(vector) == pytest.approx(3.0)


def test_a_stump_sends_values_either_side_of_its_cut():
    calibrator = on_feature(FEATURES.index("standout"), 0.4, -4.0, 4.0)
    low = [0.0] * len(FEATURES)
    high = list(low)
    high[FEATURES.index("standout")] = 1.0
    assert calibrator.confidence(low) < 0.5
    assert calibrator.confidence(high) > 0.5


def test_shrinkage_and_base_combine_as_the_exporter_promises():
    """raw = base + rate * sum(trees), which is what the export is verified against."""
    tree = Tree(left=(-1,), right=(-1,), feature=(-2,), threshold=(-2.0,), value=(2.0,))
    calibrator = Calibrator(features=FEATURES, trees=(tree, tree), base=0.5, rate=0.1)
    assert calibrator.raw([0.0] * len(FEATURES)) == pytest.approx(0.5 + 0.1 * 4.0)


def test_evidence_that_lowered_the_candidate_is_never_cited():
    """Enforced in code, not left to the model.

    A confident calibrator is still wrong to explain a candidate by something that
    argued against it, and a rule that never did so on one corpus is not a rule that
    never will.
    """
    chosen = select(rows(), constant(20.0), min_confidence=0.0, top_reasons=0)
    for cid, effects in chosen.items():
        for effect in effects:
            assert effect.delta > 0.0, cid
            assert effect.standout > 0.0, cid


def test_the_slate_wide_unit_is_the_one_a_confident_gate_still_drops():
    """`everywhere` raises every candidate equally, so its standout is zero.

    That is the citation the floors exist to stop and the one a calibrator has to
    stop too, however sure it is.
    """
    chosen = select(rows(), constant(20.0), min_confidence=0.0, top_reasons=0)
    for effects in chosen.values():
        assert "w1" not in {effect.unit for effect in effects}


def test_the_context_is_read_off_the_slate_not_the_row():
    """Width and context length are slate properties, which is why `select` takes it."""
    found = contexts(rows())
    assert {c.size for c in found.values()} == {3}
    assert {c.units for c in found.values()} == {2}
    assert found["a"].row_rank == 0
    assert found["a"].row_score_margin > 0


def test_a_lower_threshold_can_only_cite_more():
    """Monotone in the threshold, so the setting means what a reader expects."""
    slate = rows()
    calibrator = on_feature(FEATURES.index("standout"), 0.4, -1.0, 1.0)
    loose = select(slate, calibrator, min_confidence=0.0, top_reasons=0)
    tight = select(slate, calibrator, min_confidence=0.99, top_reasons=0)
    for cid in slate:
        assert set(tight[cid]).issubset(set(loose[cid]))


def test_features_are_derived_from_one_implementation():
    """The vector is positional, so its length and order are part of the contract."""
    slate = rows()
    found = contexts(slate)
    for cid, row in slate.items():
        for effect in row.effects:
            assert len(features_for(effect, found[cid])) == len(FEATURES)


def test_a_zero_scale_does_not_divide_by_zero():
    """A slate the model drew no distinction on has a scale of zero."""
    slate = rows()
    effect = slate["a"].effects[0]
    context = Context(scale=0.0, row_rank=0, row_score_margin=0.0, size=3, units=2,
                      largest_delta=0.0, best_standout=0.0)
    vector = features_for(effect, context)
    # Finite rather than merely not-NaN: an infinity would pass a tree's comparison
    # and come back as a confident probability.
    assert all(math.isfinite(value) for value in vector)
