"""What each model costs to rank with, and what it costs to explain with.

Two different questions, and conflating them is the mistake this file exists to stop.
Ranking is one scoring per candidate and happens on every request. Explaining is one
scoring per (candidate, evidence unit) and happens only when asked. A slate of ten
candidates with twelve units is ten pairs to rank and a hundred and twenty to
explain, so a model that is tolerable in the first role can be impossible in the
second, and the ratio is fixed by the method rather than by the model.

Everything here reduces to one measured quantity -- pairs scored per second, at a
realistic context length, batched as the service batches -- from which the two costs
follow by arithmetic. Reported per device, because the gap between them is the whole
deployment question: a 0.6B re-ranker that is comfortable on a GPU may be unusable on
a CPU for explanation while still being fine for ranking.

Each model runs in its own process. One of the candidates aborts CUDA on load, and a
crash in a subprocess is one row missing from a table rather than a lost run.

Run: uv run python examples/benchmark.py [slate] [units] [repeats]
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

import torch

from evrank.cross_encoder import CrossEncoderScorer
from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer, rank

# The models worth the comparison, with what each is for. Licences and separation
# scores are in `cross_encoder`'s docstring; this file is only about cost.
MODELS = (
    ("cross-encoder/ettin-reranker-68m-v1", None),
    ("cross-encoder/ettin-reranker-150m-v1", None),
    ("mixedbread-ai/mxbai-rerank-base-v2", None),
    ("mixedbread-ai/mxbai-rerank-base-v2", "tmp/lora-fold0"),
    ("Qwen/Qwen3-Reranker-0.6B", None),
    ("BAAI/bge-reranker-v2-m3", None),
)
# A listing and a request of the length the corpus actually uses, so the token count
# is the one a deployment would see rather than a round number.
LISTING = (
    "A canal-side guesthouse in old Amsterdam. On a quiet street. Breakfast "
    "included. About 140 EUR a night."
)
REQUEST = "a small waterside hotel in Amsterdam for a few nights"
NOISE = "the schedule for refuse collection"


def slate(candidates: int, units: int):
    evidence = EvidenceSet(
        [Evidence("k", "searched for", REQUEST)]
        + [Evidence("n%d" % i, "viewed", "%s %d" % (NOISE, i)) for i in range(units - 1)]
    )
    rows = [
        Candidate("c%d" % i, "Option %d" % (i + 1), "%s (%d)" % (LISTING, i))
        for i in range(candidates)
    ]
    return evidence, rows


def one(model_name: str, adapter: str | None, device: str, candidates: int,
        units: int, repeats: int) -> dict:
    """Load, warm up, then time ranking and explaining on the same slate."""
    started = time.perf_counter()
    scorer = CrossEncoderScorer(model_name, device=device, adapter=adapter)
    load_seconds = time.perf_counter() - started
    evidence, rows = slate(candidates, units)

    inner = scorer._model.model
    parameters = sum(p.numel() for p in inner.parameters())
    tokens = max(scorer.token_counts([(evidence.render(), row.text) for row in rows]))

    # Warm up: the first call pays for kernel selection and allocation, which is a
    # one-off a long-running service never pays again, so timing it would overstate
    # every number below.
    rank(CachingScorer(scorer), evidence, rows[:2])

    rank_times, explain_times = [], []
    for _ in range(repeats):
        cached = CachingScorer(scorer)
        started = time.perf_counter()
        rank(cached, evidence, rows)
        rank_times.append(time.perf_counter() - started)

        cached = CachingScorer(scorer)
        ablator = Ablator(cached, evidence, rows)
        # Leave-one-out over every unit, in the one batched call the service makes:
        # the unablated context plus one removal set per unit, every candidate scored
        # under each. The ranking pass is inside it, which is honest -- a request
        # that explains pays for both.
        removals = [[]] + [[unit] for unit in evidence.units()]
        started = time.perf_counter()
        ablator.rank_without(removals)
        explain_times.append(time.perf_counter() - started)

    memory = (
        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    )
    pairs = units * candidates + candidates
    return {
        "model": model_name,
        "adapter": adapter,
        "device": device,
        "parameters": parameters,
        "tokens": tokens,
        "load": load_seconds,
        "rank_ms": 1000 * statistics.median(rank_times),
        "explain_ms": 1000 * statistics.median(explain_times),
        "pairs": pairs,
        "pairs_per_second": pairs / statistics.median(explain_times),
        "gpu_gb": memory,
    }


def main(candidates: int = 10, units: int = 12, repeats: int = 3) -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    print("slate of %d candidates, %d evidence units: %d pairs to rank, %d to explain\n"
          % (candidates, units, candidates, units * candidates + candidates))

    found = []
    for model_name, adapter in MODELS:
        for device in devices:
            label = model_name.split("/")[-1] + (" +lora" if adapter else "")
            print("  %-28s %-5s " % (label, device), end="", flush=True)
            result = subprocess.run(
                [sys.executable, __file__, "--one", model_name, str(adapter), device,
                 str(candidates), str(units), str(repeats)],
                capture_output=True, text=True, timeout=3600, check=False,
            )
            if result.returncode != 0:
                print("FAILED  %s" % (result.stderr.strip().splitlines() or [""])[-1][:60])
                continue
            row = json.loads(result.stdout.strip().splitlines()[-1])
            found.append(row)
            print("rank %7.1f ms   explain %8.1f ms   %7.1f pairs/s"
                  % (row["rank_ms"], row["explain_ms"], row["pairs_per_second"]))

    print("\n=== Ranking: what every request pays ===")
    print("  %-30s %6s %10s %10s %9s"
          % ("model", "params", "GPU ms", "CPU ms", "slowdown"))
    for row in [r for r in found if r["device"] == "cuda"]:
        cpu = next((c for c in found
                    if c["device"] == "cpu" and c["model"] == row["model"]
                    and c["adapter"] == row["adapter"]), None)
        label = row["model"].split("/")[-1] + (" +lora" if row["adapter"] else "")
        print("  %-30s %5.0fM %9.1f %10s %9s"
              % (label, row["parameters"] / 1e6, row["rank_ms"],
                 "%.1f" % cpu["rank_ms"] if cpu else "-",
                 "%.0fx" % (cpu["rank_ms"] / row["rank_ms"]) if cpu else "-"))

    print("\n=== Explaining: what a request pays only when it asks ===")
    print("  %-30s %10s %10s %12s" % ("model", "GPU ms", "CPU ms", "GPU pairs/s"))
    for row in [r for r in found if r["device"] == "cuda"]:
        cpu = next((c for c in found
                    if c["device"] == "cpu" and c["model"] == row["model"]
                    and c["adapter"] == row["adapter"]), None)
        label = row["model"].split("/")[-1] + (" +lora" if row["adapter"] else "")
        print("  %-30s %9.1f %10s %12.1f"
              % (label, row["explain_ms"],
                 "%.0f" % cpu["explain_ms"] if cpu else "-", row["pairs_per_second"]))

    print("\n=== Memory and load ===")
    print("  %-30s %9s %10s %8s" % ("model", "GPU GB", "load s", "tokens"))
    for row in [r for r in found if r["device"] == "cuda"]:
        label = row["model"].split("/")[-1] + (" +lora" if row["adapter"] else "")
        print("  %-30s %9.2f %10.1f %8d"
              % (label, row["gpu_gb"], row["load"], row["tokens"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        adapter = None if sys.argv[3] == "None" else sys.argv[3]
        print(json.dumps(one(sys.argv[2], adapter, sys.argv[4], int(sys.argv[5]),
                             int(sys.argv[6]), int(sys.argv[7]))))
    else:
        main(
            int(sys.argv[1]) if len(sys.argv) > 1 else 10,
            int(sys.argv[2]) if len(sys.argv) > 2 else 12,
            int(sys.argv[3]) if len(sys.argv) > 3 else 3,
        )
