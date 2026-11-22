"""Does a reasoning reranker's stated reason match what its score depends on?

Rank1 writes out why it thinks a passage is relevant before judging it. That makes
it the one case where a generated explanation and a measured one can be put side by
side, which is the question the rest of this package exists to avoid having to trust:
an explanation that is produced by the model is a claim, not evidence.

The check is not a string comparison. For every evidence unit the trace quotes, the
unit is removed from the context and the pair rescored. A unit the model said it
relied on, whose removal does not move the score, was not relied on. A unit the trace
never mentions, whose removal moves the score a great deal, was.

Slow on purpose: each score is a generation, so this runs a handful of pairs rather
than a page. Run: uv run python examples/faithfulness.py
"""

from __future__ import annotations

import re

from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate
from evrank.reasoning import Rank1Scorer
from evrank.scoring import Ablator, CachingScorer, rank
from evrank.standout import standout

# Deliberately short. Every number below costs a few hundred generated tokens.
EVIDENCE = EvidenceSet(
    [
        Evidence("s1", "searched for", "ramen restaurants in Paris"),
        Evidence("r1", "reserved", "a hotel on Rue de Rivoli, 1st arrondissement"),
        Evidence("c1", "told us", "two people, dinner tonight, mid-range budget"),
    ]
)
CANDIDATES = [
    Candidate("a", "Shiso Ramen", "Tonkotsu ramen counter in the 2nd. Open until 23:00. "
              "About 22 EUR per person."),
    Candidate("b", "Izakaya Komachi", "Japanese small plates and sake in the 1st. Open "
              "until 00:00. About 40 EUR per person."),
    Candidate("c", "Trattoria Verde", "Handmade pasta near Bastille. Open until 22:30. "
              "About 35 EUR per person."),
]
MIN_EFFECT = 0.1
TRACE_PREVIEW = 420
# An omission is only worth calling out if the unit did a meaningful share of the work
# on that candidate. A flat floor gets this wrong: +0.125 clears 0.1 while being 3.7
# percent of the largest effect on its row, and a trace that leaves that out has left
# out nothing.
LOAD_BEARING_SHARE = 0.2


def mentions(trace: str, text: str) -> bool:
    """Whether the trace quotes this evidence, by its longest distinctive words.

    Crude, and only used to sort units into mentioned and not. Nothing is concluded
    from the match itself: every unit is measured either way, and the measurement is
    what the report rests on.
    """
    words = [word for word in re.findall(r"\w{5,}", text.lower())]
    if not words:
        return False
    lowered = trace.lower()
    return sum(word in lowered for word in words) >= max(1, len(words) // 2)


def audit(name: str, stated, result, trace_preview: int = TRACE_PREVIEW) -> int:
    """One candidate: what it said, what moved its score, and where those differ.

    Returns the number of disagreements, so the caller can total them.
    """
    print("\n--- %s" % name)
    print("  said (%s, %d chars): %s"
          % ("complete" if stated.complete else "cut off", len(stated.trace),
             stated.trace.replace("\n", " ")[:trace_preview]))
    print("  %-46s %9s %9s %s" % ("measured", "own", "specific", "in the trace"))
    for effect in sorted(result.effects, key=lambda e: -abs(e.delta)):
        print(
            "  %-46s %+9.3f %+9.3f %s"
            % (effect.label[:46], effect.delta, effect.standout,
               "yes" if mentions(stated.trace, effect.label) else "no")
        )

    disagreements = 0
    largest = max((abs(effect.delta) for effect in result.effects), default=0.0)
    floor = max(MIN_EFFECT, LOAD_BEARING_SHARE * largest)
    for effect in result.effects:
        said = mentions(stated.trace, effect.label)
        # Either direction counts. Evidence that drags a candidate down is as
        # load-bearing as evidence that lifts it, and reading only the positive side
        # reports a unit worth -12.4 as inert.
        moved = abs(effect.delta) >= floor
        if said and not moved:
            disagreements += 1
            print("  -> talked about, did nothing: %r at %+.3f" % (effect.label[:44], effect.delta))
        if moved and not said:
            disagreements += 1
            print("  -> never mentioned, carried %3.0f%% of this row: %r at %+.3f"
                  % (100 * abs(effect.delta) / largest, effect.label[:44], effect.delta))
    return disagreements


def main() -> None:
    scorer = Rank1Scorer(keep_traces=True)
    caching = CachingScorer(scorer)
    print("model: %s, batch_size %d (one at a time, or the trace is not reproducible)"
          % (scorer.model_name, scorer.batch_size))
    print("auditing every candidate, so this is one generation per pair.\n")

    # Ranking generates a trace per candidate; `keep_traces` keeps them, so the audit
    # reads the reasoning that produced the ranking rather than generating it twice.
    ranking = rank(caching, EVIDENCE, CANDIDATES)
    order = ranking.order()
    print("ranking by logit(true) - logit(false):")
    for position, cid in enumerate(order, start=1):
        print("  %d. %-18s %+8.3f" % (position, ranking.name(cid), ranking.scores[cid]))

    rows = standout(Ablator(caching, EVIDENCE, CANDIDATES))
    by_id = {candidate.id: candidate for candidate in CANDIDATES}

    total = 0
    for cid in order:
        stated = scorer.traces[(EVIDENCE.render(), by_id[cid].text)]
        total += audit(ranking.name(cid), stated, rows[cid])

    print("\n=== Across the slate")
    print("  %d disagreement(s) between what was said and what moved the score." % total)
    print("  Agreement on which unit mattered is the easy half. The trace cannot give")
    print("  a size, and sizes are what a ranking is made of:")
    for cid in order:
        ordered = sorted(rows[cid].effects, key=lambda e: -abs(e.delta))
        ratio = abs(ordered[0].delta) / max(abs(ordered[1].delta), 1e-9)
        print("    %-18s largest %+8.3f, %.1fx the next, on %r"
              % (ranking.name(cid), ordered[0].delta, ratio, ordered[0].label[:40]))

    print("\n=== The measured reason, in the form the service would return it")
    for cid in order:
        favouring = rows[cid].favouring(min_effect=MIN_EFFECT)
        if not favouring:
            print("  %-18s (nothing singled it out)" % ranking.name(cid))
            continue
        kind, text = EVIDENCE.phrase(favouring[0].unit)
        print("  %-18s Because you %s %s." % (ranking.name(cid), kind, text))

    print("\n  %d model pairs over %d calls, %d traces kept."
          % (caching.model_pairs, caching.batches, len(scorer.traces)))


if __name__ == "__main__":
    main()
