"""Evidence-level explanation of a real cross-encoder re-ranking.

Run: uv run python examples/walkthrough.py

The ranking itself is one batched inference over the candidate list. The
explanation is a second, separate stage that re-scores the same candidates under
contexts with evidence omitted, and the stage costs are printed so the split is
visible rather than asserted.
"""

from __future__ import annotations

from evrank.attribution import PairFlip, flip_pair, flip_top, leave_one_out, shapley
from evrank.cross_encoder import CrossEncoderScorer
from evrank.evidence import Evidence, EvidenceSet
from evrank.explain import explain_pair, explain_shown, explain_top
from evrank.masking import NeutralFiller, Omission, baseline_word_counts
from evrank.ranking import Candidate, Ranking
from evrank.scoring import Ablator, CachingScorer, rank
from evrank.standout import CandidateStandout, sampled_standout, standout

MAX_FLIP_SIZE = 2
STANDOUT_PERMUTATIONS = 32
# Matches the service default; see RankRequest.min_specificity for why it is a ratio.
MIN_SPECIFICITY = 0.1
# The other floor: was there an effect at all. See RankRequest.min_effect.
MIN_EFFECT = 0.1
# Shares no wording with any candidate, or it would contribute an effect of its own,
# and chosen so the fully masked context matches the real one in length.
FILLER_TEXT = "a routine entry, nothing specific noted"


def build_evidence() -> EvidenceSet:
    """The preserved user context, one unit per fact.

    These searches are not grouped. Grouping them as "Japanese food" looked
    reasonable and measured badly: the model reads "ramen" and "izakaya" as
    evidence for different restaurants, so averaging them into one unit reported
    their net and hid the largest effect in the explanation. `Evidence.group` is
    for facts that genuinely repeat, not for facts on a common topic.
    """
    return EvidenceSet(
        [
            Evidence("s1", "searched for", "ramen restaurants in Paris"),
            Evidence("s2", "searched for", "izakaya near the Louvre"),
            Evidence("s3", "viewed", "a Japanese noodle bar listing"),
            Evidence("h1", "reserved", "hotel on Rue de Rivoli, 1st arrondissement, central Paris"),
            # Kinds read as past actions, because the sentence in stage 6
            # makes the kind its verb.
            Evidence("t1", "told us", "dinner after 20:00, table for two"),
            Evidence("b1", "told us", "mid-range budget, about 40 EUR per person"),
            Evidence("v1", "searched for", "vegetarian lunch spots in Lyon last month"),
        ]
    )


def build_candidates() -> list[Candidate]:
    return [
        Candidate(
            "shiso",
            "Shiso Ramen",
            "Japanese ramen bar in the 1st arrondissement, a short walk from Rue de Rivoli. "
            "Open until 23:00. Bowls around 18 EUR.",
        ),
        Candidate(
            "marais",
            "Le Petit Marais",
            "Classic French bistro in the Marais, 3rd arrondissement. Last orders 21:30. "
            "Three-course menu 42 EUR.",
        ),
        Candidate(
            "komachi",
            "Izakaya Komachi",
            "Japanese izakaya serving small plates and sake near the Louvre. Open until "
            "midnight. About 35 EUR per person.",
        ),
        Candidate(
            "verde",
            "Trattoria Verde",
            "Italian trattoria with vegetarian pasta dishes in the 11th arrondissement. "
            "Closes at 22:00. About 25 EUR per person.",
        ),
        Candidate(
            "nord",
            "Brasserie du Nord",
            "Large brasserie by Gare du Nord serving steak frites all day. Kitchen closes "
            "at 20:30. About 30 EUR per person.",
        ),
    ]


def print_ranking(ranking: Ranking) -> None:
    for position, cid in enumerate(ranking.order(), start=1):
        print("  %d. %-18s %+.3f" % (position, ranking.name(cid), ranking.scores[cid]))


def describe_flip(flip: PairFlip | None) -> str:
    if flip is None:
        return "no reversal within %d evidence units" % MAX_FLIP_SIZE
    return "drop %s, margin %+.3f" % ("+".join(flip.removed), flip.margin_after)


def print_cost(label: str, scorer: CachingScorer) -> None:
    print(
        "  [%s] %d pairs requested, %d scored by the model in %d batch(es)"
        % (label, scorer.requested_pairs, scorer.model_pairs, scorer.batches)
    )
    scorer.reset_counters()


def describe_standout(
    evidence: EvidenceSet,
    result: CandidateStandout,
    min_specificity: float = MIN_SPECIFICITY,
    min_effect: float = MIN_EFFECT,
) -> str:
    """The sentence this measurement would print, or that there is none to print."""
    favouring = result.favouring(
        min_specificity=min_specificity, min_effect=min_effect
    )
    if not favouring:
        return "(no reason to show)"
    return explain_shown([evidence.phrase(favouring[0].unit)], limit=1)


def main() -> None:
    evidence = build_evidence()
    candidates = build_candidates()
    scorer = CachingScorer(CrossEncoderScorer())

    print("=== Context given to the ranker ===")
    print(evidence.render())
    print("\nAblation units: %s" % ", ".join(evidence.units()))

    print("\n=== Stage 1: ranking (production path) ===")
    ranking = rank(scorer, evidence, candidates)
    print_ranking(ranking)
    print_cost("ranking", scorer)

    order = ranking.order()
    top, runner_up = order[0], order[1]
    ablator = Ablator(scorer, evidence, candidates)

    print("\n=== Stage 2: evidence attribution for the top pair ===")
    print("  %s over %s" % (ranking.name(top), ranking.name(runner_up)))
    loo = leave_one_out(ablator, top, runner_up)
    print_cost("leave-one-out", scorer)
    shap = shapley(ablator, top, runner_up)
    print_cost("shapley", scorer)
    print("  %-14s %12s %10s" % ("unit", "leave-one-out", "shapley"))
    for effect in shap.by_effect():
        loo_delta = next(e.delta for e in loo.effects if e.unit == effect.unit)
        print("  %-14s %+12.3f %+10.3f" % (effect.unit, loo_delta, effect.delta))

    print("\n=== Stage 3: counterfactual search ===")
    pair_flip = flip_pair(ablator, top, runner_up, max_size=MAX_FLIP_SIZE)
    print_cost("pair reversal", scorer)
    top_flip = flip_top(ablator, max_size=MAX_FLIP_SIZE)
    print_cost("first place", scorer)

    print("\n=== Stage 4: explanation (templates, no inference) ===")
    print(
        explain_pair(shap, ranking, pair_flip, searched_up_to=MAX_FLIP_SIZE)
    )
    print()
    print(explain_top(ranking, top_flip, searched_up_to=MAX_FLIP_SIZE))

    print("\n=== Stage 5: does the explanation depend on how evidence is removed? ===")
    filler = NeutralFiller(text=FILLER_TEXT)
    unablated_words, filler_words = baseline_word_counts(evidence, filler)
    _, omission_words = baseline_word_counts(evidence, Omission())
    print(
        "  fully masked baseline: %d words by omission, %d by neutral filler, "
        "against %d unablated" % (omission_words, filler_words, unablated_words)
    )
    filled_ablator = Ablator(scorer, evidence, candidates, masker=filler)
    filled = shapley(filled_ablator, top, runner_up)
    print_cost("neutral filler", scorer)
    print("  %-14s %10s %15s %12s" % ("unit", "omission", "neutral filler", "difference"))
    for effect in shap.by_effect():
        same_unit = next(e for e in filled.effects if e.unit == effect.unit)
        print(
            "  %-14s %+10.3f %+15.3f %+12.3f"
            % (effect.unit, effect.delta, same_unit.delta, same_unit.delta - effect.delta)
        )
    print(
        "  margin with no evidence: omission %+.3f, neutral filler %+.3f"
        % (shap.empty_margin, filled.empty_margin)
    )
    filled_flip = flip_pair(filled_ablator, top, runner_up, max_size=MAX_FLIP_SIZE)
    print("  counterfactual by omission:       %s" % describe_flip(pair_flip))
    print("  counterfactual by neutral filler: %s" % describe_flip(filled_flip))

    print("\n=== Stage 6: why each candidate is here at all, not why it beat the next ===")
    rows = standout(ablator)
    print_cost("standout", scorer)
    print("  The line a product would print:\n")
    for cid in ranking.order():
        candidate = next(c for c in candidates if c.id == cid)
        favouring = rows[cid].favouring(
            min_specificity=MIN_SPECIFICITY, min_effect=MIN_EFFECT
        )
        if not favouring:
            print("  %-19s (nothing recorded singled it out)" % candidate.name)
            continue
        print(
            "  %-19s %s"
            % (candidate.name, explain_shown([evidence.phrase(favouring[0].unit)], limit=1))
        )

    print("\n  The numbers under those lines:")
    print(
        "  %-19s %9s %9s %6s  %s"
        % ("candidate", "specific", "own", "share", "evidence")
    )
    for cid in ranking.order():
        for index, effect in enumerate(rows[cid].by_standout()[:2]):
            print(
                "  %-19s %+9.3f %+9.3f %5.0f%%  %s"
                % (
                    ranking.name(cid) if index == 0 else "",
                    effect.standout,
                    effect.delta,
                    100 * effect.specificity,
                    effect.label,
                )
            )
    print(
        "\n  `share` is how much of what that evidence did was specific to this row.\n"
        "  A unit can lift a candidate and lift the whole slate by as much, and\n"
        "  citing it is then true and useless, so %.0f%% is the floor for a printed\n"
        "  reason, alongside a floor of %.2f on `own` so that a page the model drew\n"
        "  no distinction on gets no reasons rather than its largest piece of noise."
        % (100 * MIN_SPECIFICITY, MIN_EFFECT)
    )

    print(
        "\n  Three rows above are explained by a search for ramen, which says more\n"
        "  about the ranker than about the restaurants. The number behind it is not\n"
        "  a mistake, though. Dropping that search really\n"
        "  does cost the bistro more (-2.52) than it costs the ramen counter\n"
        "  (-2.20), because the counter has other Japanese evidence holding it up\n"
        "  and the bistro has nothing but \"restaurants in Paris\". That is a finding\n"
        "  about the ranker, not an artifact of the attribution.\n\n"
        "  Shapley values answer a different question -- what a unit is worth\n"
        "  averaged over every order the history could have arrived in -- and the\n"
        "  coalitions they average over are contexts this user never had. They\n"
        "  apportion credit between units more fairly, which matters when duplicate\n"
        "  evidence would cancel itself out, so they are worth having and worth not\n"
        "  defaulting to. Where they differ here, both readings are defensible:"
    )
    sampled = sampled_standout(ablator, permutations=STANDOUT_PERMUTATIONS)
    print_cost("standout, sampled Shapley", scorer)
    for cid in ranking.order():
        print("  %s" % ranking.name(cid))
        print("    leave-one-out:   %s" % describe_standout(evidence, rows[cid]))
        print("    sampled Shapley: %s" % describe_standout(evidence, sampled[cid]))

    if top_flip is not None:
        print("\n=== Check: re-rank with that evidence omitted ===")
        print_ranking(ablator.rank_without([top_flip.removed])[top_flip.removed])
        print_cost("verification", scorer)


if __name__ == "__main__":
    main()
