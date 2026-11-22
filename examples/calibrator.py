"""Can the transferable half of the adapter be had without touching the ranker?

The adapter was measured to learn two separable things. *Which* unit to cite on a
pair it has trained on is largely memorised -- the trained-on recall gain is six
times the gain on unseen pairs. But *being right when it speaks at all* transfers one
for one: precision rises from 39.3% to 64.1% on categories training never saw, and it
does so while printing fewer sentences and getting more of them right in absolute
terms. It reallocates speech rather than rationing it.

That second skill is a judgement over measured effects, not knowledge about a pair.
If so it should be learnable from the effects themselves, which is what this file
tests: a small model over the dumped matrix, deciding for each (candidate, unit)
whether that unit is the row's real cause. No gradient reaches the re-ranker.

Why that would matter. The adapter costs about 1.6 points of general ranking accuracy
on an independent guard, consistently across seeds, and the whole guard apparatus
exists to watch for exactly that. A rule over effects cannot damage ranking, because
ranking is upstream of it and unchanged -- so if it captures most of the precision,
it is strictly the better instrument, and it carries to any base model.

The comparison is three-way, all at the same first-row false-positive rate, all on
categories no training saw:

    base + standout floor      what ships today
    base + calibrator          this file
    adapter + standout floor   what the LoRA buys, with its ranking cost

Run: uv run python examples/calibrator.py [base.json] [lora.json] [fold]
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

import numpy
from corpus import CATEGORY_FOLDS, category_fold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from stats import percentile

from evrank.calibration import FEATURES, Context, features_for

TOP_REASONS = 2


class _Effect:
    """The three fields `features_for` reads, from a dumped row rather than a live one."""

    __slots__ = ("delta", "lift", "specificity", "standout")

    def __init__(self, effect: dict) -> None:
        self.delta = effect["delta"]
        self.lift = effect["lift"]
        self.standout = effect["standout"]
        self.specificity = effect["specificity"]


def rows_of(case: dict) -> list[dict]:
    """Every (candidate, unit) effect in one case, with its feature vector.

    The vector comes from `evrank.calibration.features_for`, which is also what the
    service calls. Deriving them twice is how a model ends up scoring something
    slightly different from what it was fitted on, so there is one implementation
    and both paths use it.
    """
    order = sorted(case["scores"], key=lambda cid: -case["scores"][cid])
    rank = {cid: index for index, cid in enumerate(order)}
    middle = statistics.median(case["scores"].values())
    out = []
    for cid, effects in case["effects"].items():
        if not effects:
            continue
        parsed = [_Effect(effect) for effect in effects]
        context = Context(
            scale=case.get("scale") or 0.0,
            row_rank=rank[cid],
            row_score_margin=case["scores"][cid] - middle,
            size=case["size"],
            units=case["units"],
            largest_delta=max(abs(e.delta) for e in parsed),
            best_standout=max(e.standout for e in parsed),
        )
        for effect, raw in zip(parsed, effects):
            vector = features_for(effect, context)
            row = dict(zip(FEATURES, vector))
            row.update({
                "case": case["id"],
                "cid": cid,
                "unit": raw["unit"],
                "label": int(cid == case["target"] and raw["unit"] == case["key"]),
            })
            out.append(row)
    return out


def matrix(cases: list[dict]):
    rows = [row for case in cases for row in rows_of(case)]
    features = numpy.array([[row[name] for name in FEATURES] for row in rows],
                           dtype=numpy.float64)
    labels = numpy.array([row["label"] for row in rows], dtype=numpy.int64)
    return rows, features, labels


def cited_by_score(cases: list[dict], scores: dict, floor: float) -> dict:
    """Which units each row would print, strongest score first, under one threshold."""
    said = {}
    for case in cases:
        per_row = {}
        for row in rows_of(case):
            value = scores[(row["case"], row["cid"], row["unit"])]
            if value >= floor:
                per_row.setdefault(row["cid"], []).append((value, row["unit"]))
        said[case["id"]] = {
            cid: [unit for _value, unit in sorted(kept, reverse=True)[:TOP_REASONS]]
            for cid, kept in per_row.items()
        }
    return said


def measure(cases: list[dict], scores: dict, floor: float) -> dict:
    """Recall per language, precision and first-row false positives, as the harness
    computes them, so a learned rule and a floor are read off the same page."""
    said = cited_by_score(cases, scores, floor)
    per_language, spoke, right = {}, 0, 0
    off_target, top_spoke = 0, 0
    no_cause, invented = 0, 0
    for case in cases:
        order = sorted(case["scores"], key=lambda cid: -case["scores"][cid])
        here = said[case["id"]]
        spoke += sum(1 for cid in order if here.get(cid))
        hit = case["target"] is not None and case["key"] in here.get(case["target"], [])
        right += int(hit)
        if case["target"] is not None:
            per_language.setdefault(case["language"], []).append(int(hit))
        if order[0] != case["target"]:
            off_target += 1
            top_spoke += int(bool(here.get(order[0])))
        # Every row that has no cause, not just the first one. Matching the arms on
        # the first row alone would let a rule buy its precision by going silent
        # everywhere else, so this is the column that catches that.
        for cid in order:
            if cid != case["target"]:
                no_cause += 1
                invented += int(bool(here.get(cid)))
    recalls = [100 * sum(v) / len(v) for v in per_language.values()]
    return {
        "mean": statistics.mean(recalls) if recalls else 0.0,
        "p20": percentile(recalls, 0.2),
        "precision": 100 * right / max(spoke, 1),
        "first_row_fp": 100 * top_spoke / max(off_target, 1),
        "invention": 100 * invented / max(no_cause, 1),
        "sentences": spoke,
        "right": right,
    }


def at_rate(cases: list[dict], scores: dict, target_fp: float, steps: int = 44) -> dict:
    """Bisect the threshold to a first-row false-positive rate, strict side.

    The same discipline `compare.py` uses, and for the same reason: a rule compared at
    a looser operating point than its rival is not being compared.
    """
    values = sorted(scores.values())
    low, high = values[0] - 1.0, values[-1] + 1.0
    for _ in range(steps):
        middle = (low + high) / 2
        if measure(cases, scores, middle)["first_row_fp"] > target_fp:
            low = middle
        else:
            high = middle
    found = measure(cases, scores, high)
    found["floor"] = high
    return found


def floor_scores(cases: list[dict], field: str, min_specificity: float) -> dict:
    """The shipped gate expressed as a score, so one bisection serves both arms.

    A unit failing any of the gate's other conditions is pushed below every real
    score rather than removed, which keeps the threshold search over one quantity.
    """
    out = {}
    for case in cases:
        for row in rows_of(case):
            key = (row["case"], row["cid"], row["unit"])
            usable = (row["standout"] > 0.0 and row["delta"] > 0.0
                      and row["delta"] >= 0.1
                      and row["specificity"] >= min_specificity)
            out[key] = row[field] if usable else -1e6
    return out


def main(base_path: str, lora_path: str, fold: int, target_fp: float = 25.0) -> None:
    base = json.loads(pathlib.Path(base_path).read_text(encoding="utf-8"))
    lora = json.loads(pathlib.Path(lora_path).read_text(encoding="utf-8"))
    reserved = category_fold(fold)

    def held(case):
        return case.get("category") in reserved

    train = [case for case in base if not held(case)]
    test = [case for case in base if held(case)]
    test_lora = [case for case in lora if held(case)]
    print("fold %d: %d training cases, %d held-out cases" % (fold, len(train), len(test)))

    _rows, features, labels = matrix(train)
    print("training effects: %d, of which real causes %d (%.2f%%)"
          % (len(labels), labels.sum(), 100 * labels.mean()))

    scaler = StandardScaler().fit(features)
    models = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "boosted": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=4,
            class_weight="balanced", random_state=0),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(scaler.transform(features) if name == "logistic" else features, labels)
        fitted[name] = model

    test_rows, test_features, _ = matrix(test)
    keys = [(row["case"], row["cid"], row["unit"]) for row in test_rows]

    print("\nAll three at the same first-row false-positive rate, on held-out "
          "categories only:")
    print("  %-26s %7s %7s %9s %7s %10s"
          % ("rule", "mean", "p20", "precision", "FP", "sentences"))
    for label, cases, scores in (
        ("base, standout floor", test, floor_scores(test, "standout", 0.3)),
        ("base, delta floor", test, floor_scores(test, "delta", 0.1)),
        ("adapter, standout floor", test_lora,
         floor_scores(test_lora, "standout", 0.3)),
    ):
        found = at_rate(cases, scores, target_fp)
        print("  %-26s %6.1f%% %6.1f%% %8.1f%% %6.1f%% %10d"
              % (label, found["mean"], found["p20"], found["precision"],
                 found["first_row_fp"], found["sentences"]))

    for name, model in fitted.items():
        data = scaler.transform(test_features) if name == "logistic" else test_features
        probability = model.predict_proba(data)[:, 1]
        scores = dict(zip(keys, probability))
        found = at_rate(test, scores, target_fp)
        print("  %-26s %6.1f%% %6.1f%% %8.1f%% %6.1f%% %10d"
              % ("base, %s" % name, found["mean"], found["p20"], found["precision"],
                 found["first_row_fp"], found["sentences"]))

    weights = fitted["logistic"].coef_[0]
    print("\nWhat the linear rule learned, in standardised units:")
    for name, weight in sorted(zip(FEATURES, weights), key=lambda kv: -abs(kv[1])):
        print("  %-18s %+.3f" % (name, weight))


def fit(cases: list[dict], reserved: frozenset[str]):
    """A calibrator trained on everything outside the reserved categories.

    Refitted per arm, not shared. The adapter changes the distribution of the very
    quantities this model reads, so a calibrator fitted on base-model effects and
    applied to an adapter's effects would be measuring the shift rather than the
    rule. Fitting costs seconds on a dump that already exists, so there is no reason
    to reuse one.
    """
    train = [case for case in cases if case.get("category") not in reserved]
    _rows, features, labels = matrix(train)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=4,
        class_weight="balanced", random_state=0).fit(features, labels)
    return model


def scored(model, cases: list[dict]) -> dict:
    rows, features, _labels = matrix(cases)
    keys = [(row["case"], row["cid"], row["unit"]) for row in rows]
    return dict(zip(keys, model.predict_proba(features)[:, 1]))


def sweep(base_path: str, pattern: str, target_fp: float = 25.0) -> None:
    """Four arms on every fold, all at the same first-row false-positive rate.

    The arms answer one question each. The floor is what ships. The calibrator asks
    whether the adapter's transferable half can be had without touching the ranker.
    The adapter is what a training run buys, at a measured cost to general ranking.
    And the two together ask whether they compose -- the adapter's gain is mostly
    recall and the calibrator's is mostly precision, so they may add rather than
    overlap.
    """
    base = json.loads(pathlib.Path(base_path).read_text(encoding="utf-8"))
    totals = {}
    for fold in range(CATEGORY_FOLDS):
        reserved = category_fold(fold)
        adapter = json.loads(
            pathlib.Path(pattern % fold).read_text(encoding="utf-8"))
        base_test = [case for case in base if case.get("category") in reserved]
        lora_test = [case for case in adapter if case.get("category") in reserved]
        arms = {
            "floor (ships)": (base_test, floor_scores(base_test, "standout", 0.3)),
            "calibrator": (base_test, scored(fit(base, reserved), base_test)),
            "adapter + floor": (lora_test, floor_scores(lora_test, "standout", 0.3)),
            "adapter + calibrator": (
                lora_test, scored(fit(adapter, reserved), lora_test)),
        }
        print("\n=== fold %d: %d held-out cases ===" % (fold, len(base_test)))
        print("  %-22s %7s %7s %9s %9s %7s %6s"
              % ("arm", "mean", "p20", "precision", "invention", "FP", "sent"))
        for label, (cases, scores) in arms.items():
            found = at_rate(cases, scores, target_fp)
            totals.setdefault(label, []).append(found)
            print("  %-22s %6.1f%% %6.1f%% %8.1f%% %8.1f%% %6.1f%% %6d"
                  % (label, found["mean"], found["p20"], found["precision"],
                     found["invention"], found["first_row_fp"], found["sentences"]))

    print("\n=== averaged over the four folds, every category held out exactly once ===")
    print("  %-22s %7s %7s %9s %9s %6s"
          % ("arm", "mean", "p20", "precision", "invention", "sent"))
    for label, runs in totals.items():
        print("  %-22s %6.1f%% %6.1f%% %8.1f%% %8.1f%% %6d"
              % (label,
                 statistics.mean(r["mean"] for r in runs),
                 statistics.mean(r["p20"] for r in runs),
                 statistics.mean(r["precision"] for r in runs),
                 statistics.mean(r["invention"] for r in runs),
                 sum(r["sentences"] for r in runs)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--folds":
        sweep("tmp/matrices-base-v4.json", "tmp/matrices-fold%d.json")
        sys.exit(0)
    main(
        sys.argv[1] if len(sys.argv) > 1 else "tmp/matrices-base-v4.json",
        sys.argv[2] if len(sys.argv) > 2 else "tmp/matrices-lora-v4.json",
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
