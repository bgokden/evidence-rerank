"""The same planted-evidence cases through several re-rankers, side by side.

Attribution is model-agnostic by construction, so the interesting question is not
whether it runs on another model but what it reports when the model underneath is
better or worse. Every number here comes from `examples/evaluation.py`, on the same
cases in the same order, so the only thing that differs between rows is the model.

The no-model column is a character-trigram matcher and it is the reason the rest is
readable: a model that cannot beat it is not contributing, and a corpus where it
wins is measuring string overlap rather than attribution.

Only models whose weights are downloadable under a licence that permits commercial
use are listed. The licence is a column because it decides whether a result is
actionable, and a non-commercial model that wins is not a candidate.

Each model runs in its own process. One of them aborts CUDA on load, which leaves
the context unusable for everything scheduled after it, so isolation is correctness
here rather than tidiness.

Run: uv run python examples/model_comparison.py [cases] [min_standout]
"""

from __future__ import annotations

import collections
import json
import random
import statistics
import subprocess
import sys
import time

from corpus import build_case, build_matched_case
from evaluation import evaluate, separation
from stats import percentile

from evrank.config import Settings
from evrank.cross_encoder import CrossEncoderScorer
from evrank.scoring import CachingScorer

Entry = collections.namedtuple("Entry", "name licence note trust")

MODELS = (
    Entry("BAAI/bge-reranker-v2-m3", "Apache-2.0",
          "multilingual, 8192 tokens, the current default", False),
    Entry("mixedbread-ai/mxbai-rerank-base-v2", "Apache-2.0",
          "multilingual, 32768 tokens, unbounded scores", False),
    Entry("Qwen/Qwen3-Reranker-0.6B", "Apache-2.0",
          "multilingual, 32768 tokens, scored from yes/no token logits", False),
    Entry("Alibaba-NLP/gte-multilingual-reranker-base", "Apache-2.0",
          "multilingual, custom architecture", True),
    Entry("BAAI/bge-reranker-base", "MIT",
          "multilingual, 512 tokens", False),
    Entry("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "Apache-2.0",
          "multilingual, 512 tokens", False),
    Entry("cross-encoder/ettin-reranker-68m-v1", "Apache-2.0",
          "English only, 68M parameters", False),
    Entry("cross-encoder/ettin-reranker-150m-v1", "Apache-2.0",
          "English only, 150M parameters", False),
    Entry("cross-encoder/ms-marco-MiniLM-L-6-v2", "Apache-2.0",
          "English only, 512 tokens", False),
)

# Weights exist but the licence forbids commercial use, so a result would not be
# actionable. Recorded rather than silently dropped.
EXCLUDED = (
    ("jinaai/jina-reranker-v2-base-multilingual", "CC-BY-NC-4.0", "non-commercial"),
    ("jinaai/jina-reranker-v3", "CC-BY-NC-4.0",
     ("non-commercial, and listwise: removing one candidate changes every other "
      "candidate's score, so an ablation isolates nothing")),
)


def measure(model_name: str, trust: bool, count: int, min_standout: float) -> dict:
    """One model over every case. Runs in its own process; prints one JSON line."""
    rng = random.Random(20261120)
    cases = [(build_matched_case if i % 2 else build_case)(rng, i) for i in range(count)]
    inner = CrossEncoderScorer(model_name, trust_remote_code=trust)

    results, refused = [], 0
    started = time.perf_counter()
    for case in cases:
        try:
            results.append(evaluate(case, lambda: CachingScorer(inner), min_standout))
        except Exception as error:  # noqa: BLE001 - counted and reported, not hidden
            # A short-context model refuses the long cases rather than truncating
            # them, which is the designed behaviour and is its own outcome.
            refused += 1
            if refused == 1:
                print("note: %s refused a case: %s" % (model_name, str(error)[:70]),
                      file=sys.stderr)
    if not results:
        raise RuntimeError("every case was refused")

    planted = [r for r in results if r["planted"]]
    tallies = [r["sweep"][min_standout] for r in results]
    said = sum(t["rows_spoke"] for t in tallies)
    no_cause = sum(t["no_cause_rows"] for t in tallies)
    top_rows = [t for t in tallies if not t["top_has_cause"]]
    per_lang = {}
    for lang in sorted({r["language"] for r in results if r["condition"] == "matched"}):
        group = [r for r in planted
                 if r["condition"] == "matched" and r["language"] == lang]
        if group:
            per_lang[lang] = 100 * sum(
                r["sweep"][min_standout]["target_right"] for r in group
            ) / len(group)
    ordered = sorted(per_lang.values())
    by_condition = {}
    for condition in ("mixed", "matched"):
        group = [r for r in planted if r["condition"] == condition]
        by_condition[condition] = [
            100 * sum(r["ranked_first"] for r in group) / len(group) if group else 0.0,
            100 * sum(r["sweep"][min_standout]["target_right"] for r in group) / len(group)
            if group else 0.0,
        ]
    scores = [s for r in results for s in r["matrix"]["scores"].values()]
    return {
        "cases": len(results),
        "refused": refused,
        "max_seq_length": inner.max_seq_length,
        "first": 100 * sum(r["ranked_first"] for r in planted) / max(len(planted), 1),
        "baseline_first": 100 * sum(r["lexical_first"] for r in planted)
        / max(len(planted), 1),
        "cited": 100 * sum(t["target_right"] for t in tallies) / max(len(planted), 1),
        "precision": 100 * sum(t["right"] for t in tallies) / max(said, 1),
        "on_top": 100 * sum(t["top_spoke"] for t in top_rows) / max(len(top_rows), 1),
        "invention": 100 * sum(t["no_cause_spoke"] for t in tallies) / max(no_cause, 1),
        "separation": separation(results).get("delta", {}).get("auc", 0.0),
        "p20": percentile(ordered, 0.2),
        "by_condition": by_condition,
        # A score squashed into [0,1] compresses exactly the differences this method
        # measures, so the observed range is reported rather than assumed from a card.
        "score_low": min(scores),
        "score_high": max(scores),
        "reason_ms": statistics.median(r["reason_ms"] for r in results),
        "wall": time.perf_counter() - started,
    }


def run(entry: Entry, count: int, min_standout: float) -> dict | None:
    """One model in a clean process, so a crash costs that row and not the run."""
    finished = subprocess.run(
        [sys.executable, __file__, "--one", entry.name, str(entry.trust),
         str(count), str(min_standout)],
        capture_output=True, text=True, check=False,
    )
    for line in finished.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    reason = (finished.stderr.strip().splitlines() or ["no output"])[-1]
    print("  %-44s FAILED  %s" % (entry.name, reason[:72]))
    return None


def main(count: int = 220, min_standout: float = Settings.min_standout) -> None:
    print("%d cases, min_standout %.2f, commercially usable weights only\n"
          % (count, min_standout))
    rows = {}
    for entry in MODELS:
        print("  running %-44s %s" % (entry.name, entry.licence), flush=True)
        found = run(entry, count, min_standout)
        if found:
            rows[entry.name] = found
    order = sorted(rows, key=lambda name: -rows[name]["separation"])
    licences = {entry.name: entry.licence for entry in MODELS}

    print("\n=== Can the ablation deltas tell a cause from junk? (the gate on everything)")
    print("  %-44s %11s %9s %9s %9s"
          % ("model", "separation", "cited", "precision", "licence"))
    for name in order:
        row = rows[name]
        print("  %-44s %10.3f %8.1f%% %8.1f%% %9s"
              % (name, row["separation"], row["cited"], row["precision"],
                 licences[name]))

    print("\n=== Ranking: does the asked-for item come first?")
    print("  %-44s %8s %10s %8s" % ("model", "first", "no model", "gain"))
    for name in order:
        row = rows[name]
        print("  %-44s %7.1f%% %9.1f%% %7.1f"
              % (name, row["first"], row["baseline_first"],
                 row["first"] - row["baseline_first"]))

    print("\n=== Split by corpus: English content, against eleven languages written out")
    print("  %-44s %10s %10s %10s %10s"
          % ("model", "en first", "en cited", "all first", "all cited"))
    for name in order:
        mixed, matched = rows[name]["by_condition"]["mixed"], rows[name]["by_condition"]["matched"]
        print("  %-44s %9.1f%% %9.1f%% %9.1f%% %9.1f%%"
              % (name, mixed[0], mixed[1], matched[0], matched[1]))

    print("\n=== Score range observed, and cost")
    print("  Scores squashed into [0,1] compress the differences this method measures.")
    print("  %-44s %16s %8s %9s %8s %7s"
          % ("model", "score range", "tokens", "reason ms", "refused", "p20"))
    for name in order:
        row = rows[name]
        print("  %-44s %7.2f to %5.2f %8d %9.0f %8d %6.0f%%"
              % (name, row["score_low"], row["score_high"], row["max_seq_length"],
                 row["reason_ms"], row["refused"], row["p20"]))

    print("\n=== Excluded: weights exist, licence does not permit commercial use")
    for name, licence, why in EXCLUDED:
        print("  %-44s %-14s %s" % (name, licence, why))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(measure(sys.argv[2], sys.argv[3] == "True",
                                 int(sys.argv[4]), float(sys.argv[5]))))
    else:
        main(
            int(sys.argv[1]) if len(sys.argv) > 1 else 220,
            float(sys.argv[2]) if len(sys.argv) > 2 else Settings.min_standout,
        )
