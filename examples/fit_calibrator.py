"""Fit a calibrator on a dumped matrix and write it out as JSON.

The output is read by `evrank.calibration.Calibrator.load`, which evaluates the trees
itself, so nothing about serving depends on scikit-learn or on a pickle. That only
helps if the exported file means the same thing as the model it came from, which is
not something to assume: this script reconstructs every training row's score with the
same arithmetic the service will use and refuses to write the file unless it matches
the library to within a tight tolerance.

Fitted on the categories a fold does not reserve, so a calibrator can be measured on
the same held-out pairs as everything else.

Run: uv run python examples/fit_calibrator.py [dump.json] [out.json] [fold]
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy
from calibrator import matrix
from corpus import category_fold
from sklearn.ensemble import GradientBoostingClassifier

from evrank.calibration import FEATURES, Calibrator

# Measured against a non-portable histogram ensemble over four folds: 59.6% precision
# against 59.5%, 4.8% invention against 4.7%. Portability costs nothing here, so the
# shallow exportable form is what ships. Depth three keeps each tree small enough to
# read, which matters for a file that decides what a user is told.
TREES = 120
DEPTH = 3
RATE = 0.1
# A reconstruction further than this from the library's own score means the export is
# wrong. It is far tighter than any threshold a deployment would set, so a file that
# passes is one whose decisions are identical.
TOLERANCE = 1e-9


def export(model: GradientBoostingClassifier, features, name: str) -> dict:
    """The fitted ensemble as plain data, verified against the model it came from."""
    trees = []
    for stage in model.estimators_:
        inner = stage[0].tree_
        trees.append({
            "left": inner.children_left.tolist(),
            "right": inner.children_right.tolist(),
            "feature": inner.feature.tolist(),
            "threshold": inner.threshold.tolist(),
            "value": inner.value.reshape(-1).tolist(),
        })
    # The ensemble's constant starting point is whatever the library's own raw score
    # is, minus the trees' contribution. Derived rather than looked up, because the
    # attribute that holds it has moved between versions, and then checked.
    raw = model.decision_function(features)
    contribution = numpy.array([
        sum(stage[0].tree_.value.reshape(-1)[stage[0].apply(features[index: index + 1])[0]]
            for stage in model.estimators_)
        for index in range(min(len(features), 64))
    ])
    base = float(numpy.mean(raw[:len(contribution)] - RATE * contribution))
    document = {
        "features": list(FEATURES),
        "trees": trees,
        "base": base,
        "rate": RATE,
        "model": name,
    }
    return document


def verify(document: dict, model: GradientBoostingClassifier, features) -> None:
    """Score every training row both ways and refuse to ship a disagreement."""
    # Round-tripped through the public loader on purpose: what gets verified is the
    # path the service takes, file parsing included, not an object built in memory.
    scratch = pathlib.Path("tmp/calibrator-verify.json")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(json.dumps(document), encoding="utf-8")
    calibrator = Calibrator.load(scratch)
    scratch.unlink()

    theirs = model.decision_function(features)
    worst = 0.0
    for index in range(len(features)):
        mine = calibrator.raw(features[index])
        worst = max(worst, abs(mine - float(theirs[index])))
    if worst > TOLERANCE:
        raise ValueError(
            "the exported trees disagree with the model they came from by %.3g, "
            "which is larger than %.3g. The file would decide differently from what "
            "was measured, so it is not written." % (worst, TOLERANCE)
        )
    print("export verified against the library on %d rows, worst difference %.3g"
          % (len(features), worst))


def main(dump: str, out: str, fold: int, model_name: str = "") -> None:
    cases = json.loads(pathlib.Path(dump).read_text(encoding="utf-8"))
    reserved = category_fold(fold)
    train = [case for case in cases if case.get("category") not in reserved]
    _rows, features, labels = matrix(train)
    print("fitting on %d cases outside fold %d: %d effects, %d real causes"
          % (len(train), fold, len(labels), labels.sum()))

    # What class_weight="balanced" computes. Applied as sample weights because
    # GradientBoostingClassifier takes no class_weight, and it matters: without it
    # the same ensemble reads 53.8% precision instead of 59.6%, because the real
    # causes are two percent of the rows.
    weight = numpy.where(
        labels == 1,
        len(labels) / (2 * labels.sum()),
        len(labels) / (2 * (len(labels) - labels.sum())),
    )
    model = GradientBoostingClassifier(
        n_estimators=TREES, max_depth=DEPTH, learning_rate=RATE, random_state=0
    ).fit(features, labels, sample_weight=weight)

    document = export(model, features, model_name or pathlib.Path(dump).stem)
    verify(document, model, features)
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(document), encoding="utf-8")
    size = pathlib.Path(out).stat().st_size
    print("written to %s, %d trees, %.0f KB" % (out, len(document["trees"]), size / 1024))


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "tmp/matrices-base-v4.json",
        sys.argv[2] if len(sys.argv) > 2 else "tmp/calibrator-fold0.json",
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
