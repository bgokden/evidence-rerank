"""Measured cost of ranking against the cost of explaining.

Ranking is one batched inference whatever the context holds, so its cost barely
moves with the evidence. Explanation re-scores ablated contexts, and what it costs
is set by the number of evidence units: leave-one-out is linear in units, exact
Shapley is exponential. This script measures both rather than asserting them.

Run: uv run python examples/cost_scaling.py [model-name]
"""

from __future__ import annotations

import sys
import time

from walkthrough import build_candidates, build_evidence

from evrank.attribution import flip_pair, flip_top, leave_one_out, shapley
from evrank.cross_encoder import MULTILINGUAL_MODEL, CrossEncoderScorer
from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer, rank
from evrank.standout import standout

SHAPLEY_LIMIT = 12
LINEAR_SIZES = (20, 50, 100)
# The shape a production request is expected to take.
WORKLOAD_UNITS = (20, 25, 30)
WORKLOAD_CANDIDATES = (5, 10)


def evidence_of_size(n: int) -> EvidenceSet:
    """The real facts first, then filler of the same shape, so lengths stay realistic."""
    facts = list(build_evidence().items)
    while len(facts) < n:
        index = len(facts)
        facts.append(
            Evidence("x%d" % index, "viewed", "a listing near the river, entry %d" % index)
        )
    return EvidenceSet(facts[:n])


NEIGHBOURHOODS = ("Montmartre", "Belleville", "Saint-Germain", "Bastille", "Pigalle")
CUISINES = ("Korean barbecue", "Lebanese mezze", "Breton creperie", "Thai canteen", "wine bar")


def candidate_pool(count: int, base: list[Candidate]) -> list[Candidate]:
    """Distinct candidates, because duplicates make flip search measure its worst case.

    Two candidates with identical text sit at a zero margin that no removal can
    reverse, so the search exhausts every subset and times a situation a real
    result page does not contain.
    """
    pool = list(base[:count])
    while len(pool) < count:
        index = len(pool) - len(base)
        pool.append(
            Candidate(
                "extra%d" % index,
                "%s house" % CUISINES[index % len(CUISINES)].split()[0],
                "%s in %s. Open until %d:00. About %d EUR per person."
                % (
                    CUISINES[index % len(CUISINES)],
                    NEIGHBOURHOODS[index % len(NEIGHBOURHOODS)],
                    21 + index % 3,
                    22 + index * 4,
                ),
            )
        )
    return pool


def timed(call, *args, **kwargs):
    """Call it with its arguments rather than closing over loop variables."""
    start = time.perf_counter()
    result = call(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def main(model_name: str = MULTILINGUAL_MODEL) -> None:
    inner = CrossEncoderScorer(model_name)
    candidates = build_candidates()
    print("model: %s, %d candidates\n" % (model_name, len(candidates)))

    header = "%5s %7s | %5s %7s | %8s %6s %8s | %8s %7s %9s"
    print(
        header
        % ("units", "tokens", "pairs", "rank ms", "contexts", "pairs", "loo ms",
           "contexts", "pairs", "shapley ms")
    )
    print("-" * 94)

    for n in range(1, SHAPLEY_LIMIT + 1):
        evidence = evidence_of_size(n)
        tokens = inner.token_counts([(evidence.render(), candidates[0].text)])[0]

        ranking_scorer = CachingScorer(inner)
        _, rank_ms = timed(rank, ranking_scorer, evidence, candidates)

        loo_scorer = CachingScorer(inner)
        loo_ablator = Ablator(loo_scorer, evidence, candidates)
        pair = rank(CachingScorer(inner), evidence, candidates).order()[:2]
        loo_scorer.reset_counters()
        _, loo_ms = timed(leave_one_out, loo_ablator, pair[0], pair[1])

        shap_scorer = CachingScorer(inner)
        shap_ablator = Ablator(shap_scorer, evidence, candidates)
        _, shap_ms = timed(shapley, shap_ablator, pair[0], pair[1])

        print(
            header
            % (
                n,
                tokens,
                ranking_scorer.model_pairs,
                "%.1f" % rank_ms,
                n + 2,
                loo_scorer.model_pairs,
                "%.1f" % loo_ms,
                2**n,
                shap_scorer.model_pairs,
                "%.1f" % shap_ms,
            ),
            flush=True,
        )

    print("\nleave-one-out alone, past the point where exact Shapley is affordable:")
    for n in LINEAR_SIZES:
        evidence = evidence_of_size(n)
        tokens = inner.token_counts([(evidence.render(), candidates[0].text)])[0]
        scorer = CachingScorer(inner)
        ablator = Ablator(scorer, evidence, candidates)
        pair = ablator.base().order()[:2]
        scorer.reset_counters()
        _, loo_ms = timed(leave_one_out, ablator, pair[0], pair[1])
        print(
            "%5d units, %6d tokens: %4d contexts, %5d pairs, %.0f ms"
            % (n, tokens, n + 2, scorer.model_pairs, loo_ms),
            flush=True,
        )

    print("\na full explanation at production scale (leave-one-out, flips up to size 2):")
    print(
        "%5s %6s | %8s | %9s | %9s | %9s | %8s"
        % ("units", "cands", "rank ms", "loo ms", "flip pair", "flip top", "total ms")
    )
    print("-" * 74)
    for count in WORKLOAD_CANDIDATES:
        for n in WORKLOAD_UNITS:
            evidence = evidence_of_size(n)
            pool = candidate_pool(count, candidates)
            scorer = CachingScorer(inner)
            ranking, rank_ms = timed(rank, scorer, evidence, pool)
            pair = ranking.order()[:2]
            ablator = Ablator(scorer, evidence, pool)
            _, loo_ms = timed(leave_one_out, ablator, pair[0], pair[1])
            _, flip_pair_ms = timed(flip_pair, ablator, pair[0], pair[1], max_size=2)
            _, flip_top_ms = timed(flip_top, ablator, max_size=2)
            print(
                "%5d %6d | %8.0f | %9.0f | %9.0f | %9.0f | %8.0f"
                % (
                    n,
                    count,
                    rank_ms,
                    loo_ms,
                    flip_pair_ms,
                    flip_top_ms,
                    rank_ms + loo_ms + flip_pair_ms + flip_top_ms,
                ),
                flush=True,
            )

    print("\nexplaining the whole page instead of one pair:")
    print("%5s %6s | %8s | %9s %6s | %8s"
          % ("units", "cands", "rank ms", "standout", "pairs", "total ms"))
    print("-" * 60)
    for count in WORKLOAD_CANDIDATES:
        for n in WORKLOAD_UNITS:
            evidence = evidence_of_size(n)
            pool = candidate_pool(count, candidates)
            scorer = CachingScorer(inner)
            _, rank_ms = timed(rank, scorer, evidence, pool)
            ablator = Ablator(scorer, evidence, pool)
            scorer.reset_counters()
            _, standout_ms = timed(standout, ablator)
            print(
                "%5d %6d | %8.0f | %9.0f %6d | %8.0f"
                % (n, count, rank_ms, standout_ms, scorer.model_pairs,
                   rank_ms + standout_ms),
                flush=True,
            )
    print(
        "  Every row gets a reason from one wave of ablations, where the pairwise\n"
        "  explanation answers for one pair."
    )

    print(
        "\nRanking costs one context times %d candidates, whatever the evidence says.\n"
        "Leave-one-out costs (units + 2) contexts times the 2 candidates compared.\n"
        "Exact Shapley costs 2^units contexts times those same 2 candidates."
        % len(candidates)
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else MULTILINGUAL_MODEL)
