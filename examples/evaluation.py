"""A planted-evidence evaluation: does the explanation name the right cause?

Reading a hundred sentences and nodding at them measures nothing. Each case here is
built so the answer is known in advance: one evidence unit names a feature exactly
one candidate has, and everything else in the slate is a distractor. The system is
then asked three questions whose answers can be scored.

    ranking      does the planted candidate come first?
    attribution  is the planted evidence the reason given for it?
    abstention   on a control case with nothing planted, is the answer silence?

The third matters as much as the others. A system that always produces a confident
sentence is easy to build and worthless, so half the control cases carry evidence
about nothing on the page, and the right output there is no sentence at all.

Latency is recorded per case so cost can be read against slate size rather than
asserted.

Run: uv run python examples/evaluation.py [cases] [min_standout] [model] [dump.json] [adapter]
"""

from __future__ import annotations

import bisect
import collections
import json
import pathlib
import random
import statistics
import sys
import time
from collections.abc import Sequence

from corpus import (
    DECOY,
    GENERIC,
    build_case,
    build_matched_case,
)
from stats import percentile

from evrank.cross_encoder import MULTILINGUAL_MODEL, CrossEncoderScorer
from evrank.explain import PHRASES, explain_shown
from evrank.scoring import Ablator, CachingScorer, rank
from evrank.standout import standout

# The service defaults, which this harness is what chose. The swept argument is
# `min_standout`, the floor on the centred effect, because that is the one that
# carries the decision; `min_effect` stays a small backstop and `min_specificity` a
# fixed shape gate. See `standout.favouring` for which case each one catches.
MIN_SPECIFICITY = 0.3
MIN_EFFECT = 0.1
MIN_STANDOUT = 0.75
# The no-model baseline's floor is bisected to the system's own first-row
# false-positive rate rather than fixed, for the reason the first attempt proves: it
# was pinned at 0.9 share of a listing's trigrams, and the highest share any row in
# this corpus attains is 0.528. The baseline therefore reported 0.0% cited and 0.0%
# invented by construction, which looked like a strong result for the system and was
# a threshold that could not be met. A baseline has to sit at the same operating
# point as the thing it is a baseline for, or it is not one.
TOP_REASONS = 2
# The floor is applied after the measurement, so a whole curve costs one pass.
FLOORS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5)

# One listing and the request that asks for it in different words.
#
# `want` names the same category as `target` without reusing its words -- "a Japanese
# noodle soup restaurant" for a tonkotsu ramen counter. Both extremes were measured
# and both measure the wrong thing. Copying the listing's own words in made the task
# solvable by character overlap, and a trigram matcher then beat the system on every
# metric at once. Describing the listing by a property only someone who knows the
# category could connect to it ("noodles in a pork broth") went too far the other
# way: the planted unit's median effect fell to 0.038 with a median specificity of
# 0.000, below the on-topic unit, so the harness measured the model's world knowledge
# rather than the attribution. A paraphrase of the category sits between the two,
# which is where real evidence sits.
#
# There is deliberately no near-twin on the page. A twin was tried, and a paraphrase
# that names a category describes the twin exactly as well as the target -- "a
# Japanese noodle soup restaurant" is a true description of a shio ramen bar -- so
# the twin made the label wrong rather than the task hard. The other families on the
# slate are the in-domain alternatives, and `generic` is the realistic way to be
# wrong.
# The corpus moved to `examples/corpus.py` when it stopped being written out case by
# case and became a generator. It is a lot of content and one idea, and neither
# belongs in the middle of the harness that measures against it.


def trigrams(text: str) -> set[str]:
    lowered = text.lower()
    return {lowered[i : i + 3] for i in range(max(len(lowered) - 2, 1))}


def lexical_best(case: dict) -> dict[str, tuple[float, str]]:
    """The no-model baseline: which unit is most nearly quoted inside each listing.

    Reported beside every metric, because a metric this passes is not a measurement
    of the attribution. On the first version of this corpus it passed all three --
    the plant was the listing's own words, so character overlap separated signal
    from noise perfectly (1.000 against a maximum of 0.773) and beat the system on
    recall and invention at once. The paraphrased plants exist to take that away,
    and this stays to prove it stayed taken away.
    """
    texts = {unit: case["evidence"].phrase(unit)[1] for unit in case["evidence"].units()}
    best = {}
    for candidate in case["candidates"]:
        listing = trigrams(candidate.text)
        scored = [
            (len(trigrams(text) & listing) / max(len(trigrams(text)), 1), unit)
            for unit, text in texts.items()
        ]
        best[candidate.id] = max(scored)
    return best


def matrix(case: dict, rows) -> dict:
    """Every measured number for one case, so a rule can be tried without the model.

    Scoring is the whole cost here and no candidate rule changes it: each is a
    function of this matrix. Dumping it once turns an afternoon of sweeps into
    milliseconds, and it keeps the comparison fair by making every rule face exactly
    the same measurements.
    """
    return {
        "id": case["id"],
        "language": case["language"],
        "condition": case["condition"],
        "domain": case["domain"],
        "size": case["size"],
        "units": case["units"],
        "target": case["target"],
        "key": case["key"],
        # The category the planted pair was generated from. The split that matters
        # is expressed in these, so a dump without them cannot be compared on the
        # one axis whose pairs training never saw.
        "category": case.get("category"),
        "generic": GENERIC,
        "scale": next(iter(rows.values())).scale if rows else 0.0,
        "scores": {cid: result.score for cid, result in rows.items()},
        "effects": {
            cid: [
                {
                    "unit": e.unit,
                    "delta": e.delta,
                    "lift": e.lift,
                    "standout": e.standout,
                    "specificity": e.specificity,
                }
                for e in result.effects
            ]
            for cid, result in rows.items()
        },
    }


def separation(results: list[dict]) -> dict:
    """How far apart the causes and the junk are, before any floor is chosen.

    The go/no-go check, and the one that would have saved two rebuilds of this
    corpus. A floor can only separate two distributions that are separated; if the
    planted effects and the junk effects overlap, every floor is a point chosen out
    of noise and the run has nothing to say about thresholds. Run it first and read
    it first: the first version of this corpus failed it by separating perfectly
    (every plant above every junk effect, so a string matcher sufficed) and the
    second by inverting (the median plant below the junk 75th percentile).
    """
    planted, junk = {"delta": [], "standout": []}, {"delta": [], "standout": []}
    for result in results:
        case = result["matrix"]
        for cid, effects in case["effects"].items():
            for effect in effects:
                into = planted if cid == case["target"] and effect["unit"] == case["key"] else junk
                into["delta"].append(effect["delta"])
                into["standout"].append(effect["standout"])
    report = {}
    for field in ("delta", "standout"):
        good, bad = sorted(planted[field]), sorted(junk[field])
        if not good or not bad:
            continue
        # The share of pairs the right answer wins, which is the area under the ROC
        # curve: 0.5 is no separation at all and 1.0 is perfect.
        wins = sum(bisect.bisect_left(bad, value) for value in good)
        ties = sum(
            bisect.bisect_right(bad, value) - bisect.bisect_left(bad, value) for value in good
        )
        report[field] = {
            "auc": (wins + 0.5 * ties) / (len(good) * len(bad)),
            "good": good,
            "bad": bad,
        }
    return report


def lexical_tally(results: list[dict], floor: float) -> dict:
    """What the no-model baseline says at one floor, counted as the system is.

    `invented` shares the system's denominator -- rows that have no cause -- which
    the first version did not: its numerator ran over every row while its
    denominator counted only the causeless ones, inflating the baseline's invention
    rate by about a quarter against the system's. Both are counted on the same rows
    here.
    """
    cited = spoke = invented = no_cause = 0
    off_target = top_spoke = 0
    for result in results:
        shares = result["lexical"]
        target = result["lexical_target"]
        if target is not None and shares[target][0] >= floor and shares[target][1] == result["key"]:
            cited += 1
        for cid, (share, _unit) in shares.items():
            if share >= floor:
                spoke += 1
                if cid != target:
                    invented += 1
            if cid != target:
                no_cause += 1
        if result["lexical_top"] != target:
            off_target += 1
            if shares[result["lexical_top"]][0] >= floor:
                top_spoke += 1
    planted = sum(1 for r in results if r["lexical_target"] is not None)
    return {
        "floor": floor,
        "cited": 100 * cited / max(planted, 1),
        "precision": 100 * cited / max(spoke, 1),
        "invention": 100 * invented / max(no_cause, 1),
        "first_row_fp": 100 * top_spoke / max(off_target, 1),
        "sentences": spoke,
    }


def lexical_at_rate(results: list[dict], target_fp: float, steps: int = 40) -> dict:
    """The baseline tuned to the system's own first-row false-positive rate.

    Without this the two columns are not comparable: the system's floor is chosen at
    an operating point and the baseline's was a constant, so the comparison rewarded
    the system for the baseline being set where nothing could reach it.
    """
    low, high = 0.0, 1.0
    for _ in range(steps):
        middle = (low + high) / 2
        if lexical_tally(results, middle)["first_row_fp"] > target_fp:
            low = middle
        else:
            high = middle
    return lexical_tally(results, high)


def _tally(case: dict, rows, order: Sequence[str], floor: float) -> dict:
    """One floor's worth of counts, with ground truth defined per row.

    A row has a cause when it is the planted candidate, and every other row on every
    other page has none -- including the rows of a planted case, which the earlier
    version left unscored and which are 60 percent of the rows that could carry an
    invented sentence.

    `top_*` is the same accounting restricted to the first-ranked row, which is the
    one the product explains. The two differ by more than rounding: the top row is
    selected on the score every one of its effects is measured from, so it carries a
    bias the average row does not.
    """
    cited = {
        cid: [
            effect.unit
            for effect in rows[cid].favouring(
                min_specificity=MIN_SPECIFICITY, min_effect=MIN_EFFECT, min_standout=floor
            )[:TOP_REASONS]
        ]
        for cid in order
    }
    target = case["target"]
    top = order[0]
    no_cause = [cid for cid in order if cid != target]
    return {
        "target_right": target is not None and case["key"] in cited[target],
        "target_leads": target is not None and cited[target][:1] == [case["key"]],
        "top_has_cause": top == target,
        "top_spoke": bool(cited[top]),
        "no_cause_rows": len(no_cause),
        "no_cause_spoke": sum(1 for cid in no_cause if cited[cid]),
        "generic_cited": sum(1 for cid in order if GENERIC in cited[cid]),
        "decoy_cited": sum(
            1 for cid in order if any(unit.startswith(DECOY) for unit in cited[cid])
        ),
        "rows_spoke": sum(1 for cid in order if cited[cid]),
        # Precision needs both halves counted on the same rows: of the sentences
        # printed, how many were about the row that has a cause and named it.
        "right": 1 if (target is not None and case["key"] in cited[target]) else 0,
    }


def evaluate(case: dict, scorer_factory, min_standout: float = MIN_STANDOUT) -> dict:
    scorer = scorer_factory()
    start = time.perf_counter()
    ranking = rank(scorer, case["evidence"], case["candidates"])
    rank_ms = (time.perf_counter() - start) * 1000

    scorer.reset_counters()
    start = time.perf_counter()
    ablator = Ablator(scorer, case["evidence"], case["candidates"])
    rows = standout(ablator)
    reason_ms = (time.perf_counter() - start) * 1000

    order = ranking.order()
    phrases = PHRASES[case["language"]]
    favouring = {
        cid: rows[cid].favouring(
            min_specificity=MIN_SPECIFICITY,
            min_effect=MIN_EFFECT,
            min_standout=min_standout,
        )
        for cid in order
    }
    sentence = {
        cid: explain_shown(
            [case["evidence"].phrase(e.unit) for e in favouring[cid][:TOP_REASONS]],
            TOP_REASONS,
            phrases,
        )
        for cid in order
    }

    # The same measured effects under every candidate floor, which is free: the
    # floor filters `rows`, it does not change them.
    sweep = {
        floor: _tally(case, rows, order, floor)
        for floor in sorted(set(FLOORS) | {min_standout})
    }

    target = case["target"]
    lexical = lexical_best(case)
    lexical_order = sorted(lexical, key=lambda cid: -lexical[cid][0])
    return {
        "matrix": matrix(case, rows),
        "sweep": sweep,
        **{
            k: case[k]
            for k in ("id", "domain", "language", "size", "units", "condition", "key")
        },
        "planted": target is not None,
        "ranked_first": target is not None and order[0] == target,
        "target_place": (order.index(target) + 1) if target is not None else None,
        # The no-model baseline on the identical case, so the two are never compared
        # across runs or corpora.
        "lexical_first": target is not None and lexical_order[0] == target,
        # The shares themselves, not a verdict at one floor: the baseline's floor is
        # chosen afterwards by bisection, so a single pass serves every floor.
        "lexical": {cid: [lexical[cid][0], lexical[cid][1]] for cid in order},
        "lexical_top": order[0],
        "lexical_target": target,
        "cited_key_for_target": sweep[min_standout]["target_right"]
        if min_standout in sweep
        else None,
        "any_sentence": any(sentence[cid] for cid in order),
        "sentences": sum(1 for cid in order if sentence[cid]),
        "rank_ms": rank_ms,
        "reason_ms": reason_ms,
        "reason_pairs": scorer.model_pairs,
        "top_sentence": sentence[order[0]],
    }


def share(rows, predicate) -> str:
    matching = [r for r in rows if predicate(r)]
    if not rows:
        return "   n/a"
    return "%5.1f%%" % (100 * len(matching) / len(rows))


def main(
    count: int = 120,
    min_standout: float = MIN_STANDOUT,
    model_name: str = MULTILINGUAL_MODEL,
    dump: str | None = None,
    adapter: str | None = None,
) -> None:
    inner = CrossEncoderScorer(model_name, adapter=adapter)
    # Fixed seed, so a change in the floor is the only thing that moved between runs.
    rng = random.Random(20261120)
    cases = [(build_matched_case if i % 2 else build_case)(rng, i) for i in range(count)]
    print(
        "model: %s%s, %d cases, min_standout %.2f\n"
        % (model_name, " + adapter %s" % adapter if adapter else "", count, min_standout)
    )

    # One untimed pass, or the first case carries the model's warm-up into the p50.
    evaluate(cases[0], lambda: CachingScorer(inner), min_standout)

    results = []
    started = time.perf_counter()
    for case in cases:
        results.append(evaluate(case, lambda: CachingScorer(inner), min_standout))
        if len(results) % 20 == 0:
            print(
                "  %d/%d, %.0f s elapsed" % (len(results), count, time.perf_counter() - started),
                flush=True,
            )
    wall = time.perf_counter() - started

    planted = [r for r in results if r["planted"]]
    control = [r for r in results if not r["planted"]]

    # The floor in force. Every table below reads the same measurements through it,
    # so the one number this run was given is the one the headline reports.
    #
    # It used to fall back to FLOORS[0] when the requested floor was not one of the
    # swept values, which is 0.0 -- no floor at all. A run given 1.50 therefore
    # printed its headline at 0.0 and reported 84% of unexplained first choices
    # speaking, a number that described nothing anyone would ship. Silently
    # answering a different question than the one asked is the worst kind of bug in
    # a measurement, so the requested floor now joins the sweep instead.
    here = min_standout
    swept = sorted(set(FLOORS) | {here})

    def at(floor: float) -> list[dict]:
        return [result["sweep"][floor] for result in results]

    off_target = [t for t in at(here) if not t["top_has_cause"]]
    system_fp = 100 * sum(t["top_spoke"] for t in off_target) / max(len(off_target), 1)
    baseline = lexical_at_rate(results, system_fp)

    print("\n=== Can a floor separate the causes from the junk at all?")
    print("  Read this before anything below it. A floor is a cut through two")
    print("  distributions, so if they overlap there is no cut and no finding.")
    found_all = separation(results)
    for field, found in sorted(found_all.items()):
        good, bad = found["good"], found["bad"]
        print(
            "  %-9s planted  p10 %7.3f  p50 %7.3f  p90 %7.3f  (n=%d)"
            % (
                field,
                good[len(good) // 10],
                good[len(good) // 2],
                good[int(len(good) * 0.9)],
                len(good),
            )
        )
        print(
            "  %-9s junk     p10 %7.3f  p50 %7.3f  p90 %7.3f  (n=%d)"
            % ("", bad[len(bad) // 10], bad[len(bad) // 2], bad[int(len(bad) * 0.9)], len(bad))
        )
        verdict = (
            "perfect -- suspect a shortcut"
            if found["auc"] > 0.995
            else "none -- no floor exists here"
            if found["auc"] < 0.6
            else "usable"
        )
        print("  %-9s separation %.3f  (%s)" % ("", found["auc"], verdict))

    print("\n=== Accuracy, %d planted and %d control cases" % (len(planted), len(control)))
    print("  The no-model column is a trigram-overlap baseline whose floor is")
    print("  bisected to the system's own first-row false-positive rate: floor")
    print("  %.3f, %.1f%% against the system's %.1f%%. Pinned at a constant it read"
          % (baseline["floor"], baseline["first_row_fp"], system_fp))
    print("  0.0% on everything, because no row in this corpus reaches 0.9.")
    print("  %-38s %8s %10s" % ("", "system", "no model"))
    print(
        "  %-38s %7.1f%% %9.1f%%"
        % (
            "planted candidate ranked first",
            100 * sum(r["ranked_first"] for r in planted) / max(len(planted), 1),
            100 * sum(r["lexical_first"] for r in planted) / max(len(planted), 1),
        )
    )
    print(
        "  %-38s %7.1f%% %9.1f%%"
        % (
            "planted evidence cited for it",
            100 * sum(r["sweep"][here]["target_right"] for r in planted) / max(len(planted), 1),
            baseline["cited"],
        )
    )
    rows_all = sum(t["no_cause_rows"] for t in at(here))
    print(
        "  %-38s %7.1f%% %9.1f%%"
        % (
            "rows with no cause given a sentence",
            100 * sum(t["no_cause_spoke"] for t in at(here)) / max(rows_all, 1),
            baseline["invention"],
        )
    )
    print(
        "  %-38s %7.1f%%"
        % (
            "planted evidence leads its sentence",
            100 * sum(r["sweep"][here]["target_leads"] for r in planted) / max(len(planted), 1),
        )
    )
    print(
        "  %-38s %7.1f%%"
        % (
            "cited, given it was ranked first",
            100
            * sum(r["sweep"][here]["target_right"] for r in planted if r["ranked_first"])
            / max(sum(1 for r in planted if r["ranked_first"]), 1),
        )
    )
    print(
        "  %-38s %7.1f%%"
        % (
            "a request for something else cited",
            100 * sum(t["decoy_cited"] for t in at(here)) / max(sum(r["size"] for r in results), 1),
        )
    )
    print(
        "  %-38s %7.1f%%"
        % (
            "the on-topic unit cited as a reason",
            100
            * sum(t["generic_cited"] for t in at(here))
            / max(sum(r["size"] for r in results), 1),
        )
    )
    said = sum(t["rows_spoke"] for t in at(here))
    print(
        "  %-38s %7.1f%% (%d of %d sentences)"
        % (
            "of the sentences shown, right",
            100 * sum(t["right"] for t in at(here)) / max(said, 1),
            sum(t["right"] for t in at(here)),
            said,
        )
    )

    print("\n  The product explains the first choice, so that row is scored on its own:")
    no_cause_top = [t for t in at(here) if not t["top_has_cause"]]
    print(
        "  %-38s %7.1f%% (%d of %d)"
        % (
            "first choice spoke with no cause",
            100 * sum(t["top_spoke"] for t in no_cause_top) / max(len(no_cause_top), 1),
            sum(t["top_spoke"] for t in no_cause_top),
            len(no_cause_top),
        )
    )
    print(
        "  %-38s %7.1f%%"
        % (
            "control cases wholly silent",
            100 * sum(1 for r in control if not r["any_sentence"]) / max(len(control), 1),
        )
    )

    print("\n=== Where the planted candidate landed")
    places = collections.Counter(r["target_place"] for r in planted)
    for place in sorted(places):
        print("  place %-3d %4d case(s)" % (place, places[place]))

    matched = [r for r in results if r["condition"] == "matched"]
    languages = sorted({r["language"] for r in matched})

    print("\n=== By floor and language, matched-language cases only")
    print("  Recall is the planted unit cited for its own row. Invention is a sentence")
    print("  on a row with no cause -- every such row, in planted and control cases")
    print("  alike, and then the same accounting for the first-ranked row alone.")
    print("  The headline is the 20th percentile across languages, not the minimum:")
    print("  the smallest of eleven noisy cells is biased low and moves between runs")
    print("  on identical code, which makes it the worst quantity to tune against.")
    print(
        "  %-8s %s %5s %6s %7s %7s %7s"
        % (
            "floor",
            " ".join("%4s" % lang for lang in languages),
            "p20",
            "worst",
            "prec",
            "invent",
            "on top",
        )
    )
    for floor in swept:
        per_lang = {}
        for lang in languages:
            group = [r for r in matched if r["planted"] and r["language"] == lang]
            per_lang[lang] = (
                100 * sum(r["sweep"][floor]["target_right"] for r in group) / max(len(group), 1)
            )
        tallies = [r["sweep"][floor] for r in matched]
        no_cause = sum(t["no_cause_rows"] for t in tallies)
        invented = sum(t["no_cause_spoke"] for t in tallies)
        said = sum(t["rows_spoke"] for t in tallies)
        top_rows = [t for t in tallies if not t["top_has_cause"]]
        ordered = sorted(per_lang.values())
        print(
            "  %-8s %s %4.0f%% %5.0f%% %6.1f%% %6.1f%% %6.1f%%"
            % (
                "%.2f" % floor,
                " ".join("%3.0f%%" % per_lang[lang] for lang in languages),
                percentile(ordered, 0.2),
                ordered[0],
                100 * sum(t["right"] for t in tallies) / max(said, 1),
                100 * invented / max(no_cause, 1),
                100 * sum(t["top_spoke"] for t in top_rows) / max(len(top_rows), 1),
            )
        )
    print("  Watch the precision column across the whole range before reading the")
    print("  others: if it is flat, the floor is trading volume and not quality.")
    print("\n=== What one floor costs each language, at the floor in force")
    print("  A single floor cannot sit at one false-positive rate in every language:")
    print("  %-6s %7s %9s %9s %8s" % ("lang", "cases", "recall", "invent", "on top"))
    for lang in languages:
        group = [r for r in matched if r["language"] == lang]
        planted_here = [r for r in group if r["planted"]]
        tallies = [r["sweep"][here] for r in group]
        no_cause = sum(t["no_cause_rows"] for t in tallies)
        top_rows = [t for t in tallies if not t["top_has_cause"]]
        print(
            "  %-6s %7d %8.0f%% %8.1f%% %7.1f%%"
            % (
                lang,
                len(group),
                100
                * sum(r["sweep"][here]["target_right"] for r in planted_here)
                / max(len(planted_here), 1),
                100 * sum(t["no_cause_spoke"] for t in tallies) / max(no_cause, 1),
                100 * sum(t["top_spoke"] for t in top_rows) / max(len(top_rows), 1),
            )
        )

    print("\n=== Of the misses, how many a floor could ever recover")
    print("  Ordering puts the planted unit in the top two or it does not. Where it")
    print("  does, the floor is what silenced it; where it does not, silence is right.")
    for lang in languages:
        group = [r for r in matched if r["planted"] and r["language"] == lang]
        misses = [r for r in group if not r["sweep"][here]["target_right"]]
        reachable = [r for r in misses if r["sweep"][0.0]["target_right"]]
        print(
            "  %-6s %3d miss(es), %d reachable by a lower floor, %d not"
            % (lang, len(misses), len(reachable), len(misses) - len(reachable))
        )

    for axis in ("size", "units", "language", "domain"):
        print("\n=== By %s" % axis)
        print(
            "  %-14s %6s %8s %8s %9s %9s"
            % (axis, "cases", "first", "cited", "rank ms", "reason ms")
        )
        for value in sorted({r[axis] for r in results}, key=str):
            group = [r for r in results if r[axis] == value]
            grouped_planted = [r for r in group if r["planted"]]
            print(
                "  %-14s %6d %8s %8s %9.0f %9.0f"
                % (
                    value,
                    len(group),
                    share(grouped_planted, lambda r: r["ranked_first"]),
                    share(grouped_planted, lambda r: r["sweep"][here]["target_right"]),
                    statistics.median(r["rank_ms"] for r in group),
                    statistics.median(r["reason_ms"] for r in group),
                )
            )

    if dump:
        pathlib.Path(dump).write_text(
            json.dumps([r["matrix"] for r in results], ensure_ascii=False)
        )
        print("\n  wrote %d measured cases to %s" % (len(results), dump))

    rank_ms = sorted(r["rank_ms"] for r in results)
    reason_ms = sorted(r["reason_ms"] for r in results)
    print("\n=== Latency over %d cases, %.0f s of wall clock" % (len(results), wall))
    print("  %-10s %8s %8s %8s %8s" % ("stage", "p50", "p90", "p99", "max"))
    for name, series in (("ranking", rank_ms), ("reasons", reason_ms)):
        print(
            "  %-10s %8.0f %8.0f %8.0f %8.0f"
            % (
                name,
                series[len(series) // 2],
                series[int(len(series) * 0.9)],
                series[int(len(series) * 0.99)],
                series[-1],
            )
        )
    print(
        "  model pairs for reasons: %d total, %d median"
        % (
            sum(r["reason_pairs"] for r in results),
            statistics.median(r["reason_pairs"] for r in results),
        )
    )

    print("\n=== What the size floor buys over the whole corpus, same measurements")
    print("  The matched table above is half the cases and the smaller slates. Read on")
    print("  every case, one answer gained costs a different number of invented rows.")
    print(
        "  %-10s %10s %12s %12s %12s"
        % ("min_standout", "cited", "invented", "on top", "per answer")
    )
    previous = None
    for floor in sorted(swept, reverse=True):
        tallies = at(floor)
        cited = sum(r["sweep"][floor]["target_right"] for r in planted)
        no_cause = sum(t["no_cause_rows"] for t in tallies)
        spoke = sum(t["no_cause_spoke"] for t in tallies)
        top_rows = [t for t in tallies if not t["top_has_cause"]]
        exchange = "  n/a"
        if previous is not None and cited > previous[0]:
            exchange = "%5.1f" % ((spoke - previous[1]) / (cited - previous[0]))
        print(
            "  %-10.2f %9.1f%% %11.1f%% %11.1f%% %12s"
            % (
                floor,
                100 * cited / max(len(planted), 1),
                100 * spoke / max(no_cause, 1),
                100 * sum(t["top_spoke"] for t in top_rows) / max(len(top_rows), 1),
                exchange,
            )
        )
        previous = (cited, spoke)
    print("  The last column is the price of lowering the floor one step: invented")
    print("  rows per extra answer. Below 1 the step is cheap, above it the step is not.")

    missed = [r for r in planted if r["ranked_first"] and not r["cited_key_for_target"]]
    print("\n=== Ranked the planted candidate first and explained it with something else")
    print("  %d of %d" % (len(missed), len([r for r in planted if r["ranked_first"]])))
    for row in missed[:8]:
        print(
            "  case %-4d %-12s %-3s size %-3d units %-3d  %s"
            % (
                row["id"],
                row["domain"],
                row["language"],
                row["size"],
                row["units"],
                row["top_sentence"][:88] or "(silent)",
            )
        )


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 120,
        float(sys.argv[2]) if len(sys.argv) > 2 else MIN_STANDOUT,
        sys.argv[3] if len(sys.argv) > 3 else MULTILINGUAL_MODEL,
        sys.argv[4] if len(sys.argv) > 4 else None,
        sys.argv[5] if len(sys.argv) > 5 else None,
    )
