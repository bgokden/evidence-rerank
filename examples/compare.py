"""Compare two selection rules, or two models, at a matched false-positive rate.

Every interval this project has quoted came from throwaway scripts that no longer
exist, which is how a headline survived being measured at 26.4% first-row false
positives against a baseline arm at 24.7% -- a gain overstated by about a tenth of
itself. This file exists so that comparison is a repository artefact: it reads the
dumped matrices, applies the shipped gate's own semantics, bisects each arm to the
same false-positive rate, and reports the bracket rather than a single floor.

Why dumps rather than a model: every rule then faces identical numbers, so a
difference between two rules cannot be a difference between two inference runs.
The dumps come from `evaluation.py`, which writes one per model.

Two things it reports that a single number cannot:

    the bracket   `delta` is quantised -- bf16 logits land on steps of 1/16 and
                  1/32 -- so the realised false-positive rate moves in jumps of
                  about 1.7 points and an exact target is usually unattainable.
                  Both bracketing floors are printed so a reader can see which side
                  a figure was taken from, and the arms are matched from the same
                  side.
    the split     three of them, and they are not interchangeable. The category
                  hold-out is the one to quote: a reserved category's pairs appear
                  in no training slate in any position, so a model that memorised
                  the training pairs can say nothing about them. The domain hold-out
                  is coarser and brings a different novelty -- an unseen listing
                  template and distractor pool, not just an unseen pair. The
                  language hold-out is weak and is printed to show how weak: for a
                  `mixed` case only the action verb is localised and the candidate
                  text stays English, so a held-out language there is a trained-on
                  case with one string swapped, and the adapter once scored above
                  its own trained-on level on it.

                  The category split carries planted cases only, because a control
                  case has no planted category to be held out by. Its
                  false-positive rate is therefore measured on the non-target rows
                  of planted cases rather than on control pages as well. Both arms
                  face the same split, so the comparison holds; the level is not
                  comparable with the other splits'.

Run: uv run python examples/compare.py base.json lora.json [target_fp]
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import random
import statistics
import sys

from corpus import HELD_OUT_CATEGORIES
from stats import percentile

TOP_REASONS = 2
# The two arms the gate decision rests on. `delta` floors the raw effect, which is
# what the package shipped first; `standout` floors the effect net of what the same
# evidence did for the rest of the slate.
DELTA_ARM = ("delta", 0.1)
STANDOUT_ARM = ("standout", 0.3)
HELD_OUT_LANGUAGES = ("fr", "pl", "zh")
HELD_OUT_DOMAINS = ("flights",)


@dataclasses.dataclass(frozen=True)
class Rule:
    """One operating point of the shipped gate.

    `field` is the quantity the floor applies to. The other conditions are the
    shipped gate's and are not optional: a positive standout, a positive delta, and
    a specificity above its own floor. Mirrored from `StandoutRow.favouring` rather
    than approximated, because a sweep that is not the shipped rule measures
    something the product does not do.
    """

    field: str
    floor: float
    min_specificity: float
    min_effect: float = 0.1

    def keeps(self, effect: dict) -> bool:
        if effect["standout"] <= 0.0:
            return False
        if effect["delta"] <= 0.0 or effect["delta"] < self.min_effect:
            return False
        if effect["specificity"] < self.min_specificity:
            return False
        return effect[self.field] >= self.floor


def cited(case: dict, rule: Rule) -> dict[str, list[str]]:
    """The units each row would print, strongest standout first."""
    out = {}
    for cid, effects in case["effects"].items():
        kept = [effect for effect in effects if rule.keeps(effect)]
        kept.sort(key=lambda effect: effect["standout"], reverse=True)
        out[cid] = [effect["unit"] for effect in kept[:TOP_REASONS]]
    return out


def tally(case: dict, rule: Rule) -> dict:
    """One case's counts under one rule, mirroring `evaluation._tally`."""
    order = sorted(case["scores"], key=lambda cid: -case["scores"][cid])
    said = cited(case, rule)
    target = case["target"]
    top = order[0]
    no_cause = [cid for cid in order if cid != target]
    right = 1 if (target is not None and case["key"] in said.get(target, [])) else 0
    return {
        "planted": target is not None,
        "language": case["language"],
        "right": right,
        "top_has_cause": top == target,
        "top_spoke": bool(said.get(top)),
        "no_cause_rows": len(no_cause),
        "no_cause_spoke": sum(1 for cid in no_cause if said.get(cid)),
        "rows_spoke": sum(1 for cid in order if said.get(cid)),
    }


def metrics(cases: list[dict], rule: Rule) -> dict:
    """Recall per language, its mean and 20th percentile, precision, first-row FP."""
    tallies = [tally(case, rule) for case in cases]
    languages = sorted({t["language"] for t in tallies if t["planted"]})
    per_language = {}
    for language in languages:
        group = [t for t in tallies if t["planted"] and t["language"] == language]
        per_language[language] = 100 * sum(t["right"] for t in group) / max(len(group), 1)
    said = sum(t["rows_spoke"] for t in tallies)
    no_cause = sum(t["no_cause_rows"] for t in tallies)
    off_target = [t for t in tallies if not t["top_has_cause"]]
    return {
        "per_language": per_language,
        "mean": statistics.mean(per_language.values()) if per_language else 0.0,
        "p20": percentile(list(per_language.values()), 0.2),
        "precision": 100 * sum(t["right"] for t in tallies) / max(said, 1),
        "invention": 100 * sum(t["no_cause_spoke"] for t in tallies) / max(no_cause, 1),
        "first_row_fp": 100 * sum(t["top_spoke"] for t in off_target) / max(len(off_target), 1),
        "sentences": said,
    }


def bracket(cases: list[dict], field: str, specificity: float, target_fp: float,
            high: float = 8.0, steps: int = 40) -> tuple[float, float]:
    """The two floors either side of `target_fp`, as a (looser, stricter) pair.

    A bracket rather than a point, because the realised rate is a step function of
    the floor: the quantity being floored is quantised, so there is generally no
    floor that lands on the target. Returning the pair is what lets two arms be
    matched from the same side of it.
    """
    low = 0.0
    for _ in range(steps):
        middle = (low + high) / 2
        rate = metrics(cases, Rule(field, middle, specificity))["first_row_fp"]
        if rate > target_fp:
            low = middle
        else:
            high = middle
    return low, high


def arm(cases: list[dict], field: str, specificity: float, target_fp: float) -> dict:
    """The stricter side of the bracket: the first floor at or under the target.

    Taking the same side for both arms is the point. Taking the loose side for one
    and the strict side for the other is exactly the error this file was written to
    stop.
    """
    low, high = bracket(cases, field, specificity, target_fp)
    chosen = Rule(field, high, specificity)
    out = metrics(cases, chosen)
    out["floor"] = high
    out["loose_floor"] = low
    out["loose_fp"] = metrics(cases, Rule(field, low, specificity))["first_row_fp"]
    return out


def paired_bootstrap(a_cases: list[dict], b_cases: list[dict], a_rule: Rule,
                     b_rule: Rule, draws: int = 2000,
                     seed: int = 20261120) -> dict[str, tuple[float, float]]:
    """95% intervals for b minus a, resampling cases rather than rows.

    Cases are the independent unit: rows within a case share a slate, a ranking and
    one another's lift, so resampling rows would treat correlated observations as
    independent and report intervals narrower than the data earns.
    """
    by_id = {case["id"]: case for case in b_cases}
    shared = [case for case in a_cases if case["id"] in by_id]
    rng = random.Random(seed)
    spread: dict[str, list[float]] = {"mean": [], "p20": [], "precision": []}
    for _ in range(draws):
        sample = [shared[rng.randrange(len(shared))] for _ in range(len(shared))]
        before = metrics(sample, a_rule)
        after = metrics([by_id[case["id"]] for case in sample], b_rule)
        for key, values in spread.items():
            values.append(after[key] - before[key])
    return {
        key: (percentile(values, 0.025), percentile(values, 0.975))
        for key, values in spread.items()
    }


def splits(cases: list[dict]) -> dict[str, list[dict]]:
    """The splits worth reporting separately, by what training actually saw.

    The category split is the one to quote. A reserved category's pairs appear in no
    training slate in any position, so a model that memorised the training pairs can
    say nothing about them. The older axes are kept because they are still
    informative about the domain, not because they are equivalent:

        category   the honest generalisation test. 12 of 48 categories, held out
                   before any training run.
        domain     two of six domains. Overlaps the category split and is coarser;
                   it was the only honest axis before the corpus was rebuilt.
        language   weak, and kept only to show how weak. For a `mixed` case just the
                   action verb is localised and the candidate text stays English, so
                   a held-out-language mixed case is a trained-on case with one
                   string swapped, and the adapter scored above its own trained-on
                   level there.
    """
    held_category = [c for c in cases if c.get("category") in HELD_OUT_CATEGORIES]
    held_domain = [c for c in cases if c["domain"] in HELD_OUT_DOMAINS]
    held_language = [c for c in cases if c["language"] in HELD_OUT_LANGUAGES]
    trained = [
        c for c in cases
        if c.get("category") not in HELD_OUT_CATEGORIES
        and c["domain"] not in HELD_OUT_DOMAINS
        and c["language"] not in HELD_OUT_LANGUAGES
    ]
    return {
        "held-out category": held_category,
        "held-out domain": held_domain,
        "held-out language, mixed": [
            c for c in held_language
            if c["condition"] == "mixed" and c["domain"] not in HELD_OUT_DOMAINS
        ],
        "trained on": trained,
    }


def report(base_path: str, lora_path: str, target_fp: float) -> None:
    base = json.loads(pathlib.Path(base_path).read_text(encoding="utf-8"))
    lora = json.loads(pathlib.Path(lora_path).read_text(encoding="utf-8"))
    print("base %s, adapter %s, target first-row FP %.1f%%"
          % (base_path, lora_path, target_fp))
    print("Floors are bisected per arm and taken from the stricter side of the")
    print("bracket, the same side for both, so the comparison is not a free ride on")
    print("a looser operating point.\n")

    base_splits = splits(base)
    lora_splits = splits(lora)
    for name in ("held-out category", "held-out domain", "held-out language, mixed",
                 "trained on"):
        here, there = base_splits[name], lora_splits[name]
        print("=== %s: %d cases ===" % (name, len(here)))
        for field, specificity in (DELTA_ARM, STANDOUT_ARM):
            before = arm(here, field, specificity, target_fp)
            after = arm(there, field, specificity, target_fp)
            print("  %-9s floor %.2f/%.2f  FP %4.1f%%/%4.1f%%"
                  % (field, before["floor"], after["floor"],
                     before["first_row_fp"], after["first_row_fp"]))
            print("            mean %5.1f%% -> %5.1f%%   p20 %5.1f%% -> %5.1f%%   "
                  "precision %5.1f%% -> %5.1f%%"
                  % (before["mean"], after["mean"], before["p20"], after["p20"],
                     before["precision"], after["precision"]))
        print()


if __name__ == "__main__":
    report(
        sys.argv[1] if len(sys.argv) > 1 else "tmp/matrices-base-v3.json",
        sys.argv[2] if len(sys.argv) > 2 else "tmp/matrices-lora-v3.json",
        float(sys.argv[3]) if len(sys.argv) > 3 else 25.0,
    )
