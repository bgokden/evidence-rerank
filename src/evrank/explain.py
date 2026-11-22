"""Natural-language rendering of measured attribution.

Templates only. Nothing here invents a reason, names a latent dimension or labels
a learned feature: every sentence restates evidence the user's own context already
contains. Cost is negligible next to the inference the numbers came from.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from evrank.attribution import PairAttribution, PairFlip, TopFlip
from evrank.ranking import Ranking

CAVEAT = (
    "Measured by omitting evidence and re-scoring. These are the model's "
    "sensitivities to the recorded context, not claims about the user's reasons."
)


def explain_pair(
    attribution: PairAttribution,
    ranking: Ranking,
    flip: PairFlip | None = None,
    searched_up_to: int | None = None,
    min_delta: float = 0.01,
) -> str:
    """Why the model put A above B, in terms of the evidence that was present."""
    a_name = ranking.name(attribution.a)
    b_name = ranking.name(attribution.b)
    above, below = (
        (a_name, b_name) if attribution.base_margin >= 0 else (b_name, a_name)
    )
    lines = [
        "%s ranks above %s (margin %+.3f)." % (above, below, abs(attribution.base_margin)),
        "",
        "Evidence effects on this pair (%s):" % attribution.method,
    ]
    lines.extend(_effect_lines(attribution, a_name, b_name, min_delta))
    lines.append(
        "With no user context the margin is %+.3f; the evidence accounts for %+.3f of it."
        % (attribution.empty_margin, attribution.total_effect())
    )
    lines.extend(_reversal_warning(attribution))
    lines.append("")
    lines.append(_pair_counterfactual(attribution, a_name, b_name, flip, searched_up_to))
    lines.append("")
    lines.append(CAVEAT)
    return "\n".join(lines)


def explain_top(
    ranking: Ranking, flip: TopFlip | None, searched_up_to: int | None = None
) -> str:
    """What would have to be forgotten for a different item to rank first."""
    top = ranking.top
    lines = ["%s ranks first (score %+.3f)." % (ranking.name(top), ranking.scores[top])]
    if flip is None:
        lines.append(
            "No omission of %s reverses first place."
            % _removal_budget(searched_up_to)
        )
    else:
        lines.append(
            "Without %s, %s ranks first instead, by %+.3f."
            % (_quote_all(flip.labels), ranking.name(flip.new_top), flip.margin_after)
        )
    lines.append("")
    lines.append(CAVEAT)
    return "\n".join(lines)


def _effect_lines(
    attribution: PairAttribution, a_name: str, b_name: str, min_delta: float
) -> list[str]:
    lines = []
    negligible = 0
    for effect in attribution.by_effect():
        if abs(effect.delta) < min_delta:
            negligible += 1
            continue
        favoured = a_name if effect.delta > 0 else b_name
        lines.append(
            "  %+.3f  %-14s %s  [favours %s]"
            % (effect.delta, effect.unit, effect.label, favoured)
        )
        if effect.spread is not None:
            line = (
                "          depending on the rest of the context: %+.3f to %+.3f (sd %.3f)"
                % (effect.spread.low, effect.spread.high, effect.spread.stdev)
            )
            if effect.spread.stderr > 0:
                line += ", sampled to within %s%.3f" % ("\u00b1", effect.spread.stderr)
            lines.append(line)
    if negligible:
        lines.append(
            "  %d further evidence unit(s) moved the margin by less than %.3f."
            % (negligible, min_delta)
        )
    return lines


def _reversal_warning(attribution: PairAttribution) -> list[str]:
    reversing = attribution.reversing_effects()
    if not reversing:
        return []
    lines = [
        "",
        (
            "Unstable evidence: the direction of the following depends on what else is "
            "present, so the value above is an average over contexts, not a description "
            "of one."
        ),
    ]
    for effect in reversing:
        lines.append(
            "  %-14s alone %+.3f, across contexts %+.3f to %+.3f, averaged %+.3f"
            % (
                effect.unit,
                effect.spread.solo,
                effect.spread.low,
                effect.spread.high,
                effect.delta,
            )
        )
    return lines


def _pair_counterfactual(
    attribution: PairAttribution,
    a_name: str,
    b_name: str,
    flip: PairFlip | None,
    searched_up_to: int | None,
) -> str:
    if flip is None:
        return "Counterfactual: no omission of %s reverses this pair." % _removal_budget(
            searched_up_to
        )
    winner, loser = (
        (b_name, a_name) if attribution.base_margin >= 0 else (a_name, b_name)
    )
    return "Counterfactual: without %s, %s ranks above %s (margin %+.3f)." % (
        _quote_all(flip.labels),
        winner,
        loser,
        abs(flip.margin_after),
    )


def _removal_budget(searched_up_to: int | None) -> str:
    if searched_up_to is None:
        return "the evidence sets searched"
    if searched_up_to == 1:
        return "any single evidence unit"
    return "up to %d evidence units" % searched_up_to


def _quote_all(labels: Sequence[str]) -> str:
    quoted = ['"%s"' % label for label in labels]
    if len(quoted) == 1:
        return quoted[0]
    return "%s and %s" % (", ".join(quoted[:-1]), quoted[-1])


@dataclasses.dataclass(frozen=True)
class RankPhrases:
    """Wording for the one-line explanation, so it is not locked to English.

    Nothing in the attribution is language-specific, and neither is this: the
    sentence is assembled from templates the caller supplies.

    `shown_because` is %-formatted with `evidence`, which is the reasons already
    joined. `evidence_phrase` is %-formatted with `kind` and `text`, and it exists
    because word order is not universal: English puts the action first ("searched for
    ramen"), a verb-final language puts it last. `join_two` takes two positional
    strings and `join_more` separates everything before the last.
    """

    join_two: str
    join_more: str
    shown_because: str
    """The line a reader sees: why this was shown at all, naming their own
    actions rather than the candidate it beat. Formatted with `evidence`."""
    evidence_phrase: str
    """How an evidence item reads in this language, from its `kind` and `text`.
    English puts the action first ("searched for ramen"); a verb-final language puts
    it last. The kind is the caller's wording, so for this line it should read as a
    past action in the same language."""


ENGLISH = RankPhrases(
    join_two="%s and %s",
    join_more=", ",
    shown_because="Because you %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

# Turkish makes no singular/plural verb distinction in these sentences, so both
# forms of each case are the same string.
TURKISH = RankPhrases(
    join_two="%s ve %s",
    join_more=", ",
    shown_because="%(evidence)s için öneriliyor.",
    evidence_phrase="%(text)s %(kind)s",
)

GERMAN = RankPhrases(
    join_two="%s und %s",
    join_more=", ",
    shown_because="Angezeigt, weil Sie %(evidence)s.",
    evidence_phrase="%(text)s %(kind)s",
)

FRENCH = RankPhrases(
    join_two="%s et %s",
    join_more=", ",
    shown_because="Proposé parce que vous %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

SPANISH = RankPhrases(
    join_two="%s y %s",
    join_more=", ",
    shown_because="Sugerido porque %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

ITALIAN = RankPhrases(
    join_two="%s e %s",
    join_more=", ",
    shown_because="Suggerito perché %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

JAPANESE = RankPhrases(
    join_two="%s と %s",
    join_more="、",
    shown_because="%(evidence)s ため表示されています。",
    evidence_phrase="%(text)s を%(kind)s",
)

# Chinese, like Japanese, writes no spaces between the parts, so the evidence runs
# straight into the verb rather than being joined by one.
CHINESE = RankPhrases(
    join_two="%s和%s",
    join_more="、",
    shown_because="因为您%(evidence)s。",
    evidence_phrase="%(kind)s%(text)s",
)

RUSSIAN = RankPhrases(
    join_two="%s и %s",
    join_more=", ",
    shown_because="Показано, потому что вы %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

POLISH = RankPhrases(
    join_two="%s i %s",
    join_more=", ",
    shown_because="Pokazane, ponieważ %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

PORTUGUESE = RankPhrases(
    join_two="%s e %s",
    join_more=", ",
    shown_because="Sugerido porque %(evidence)s.",
    evidence_phrase="%(kind)s %(text)s",
)

# The ranker handles far more languages than this; a sentence needs wording, which
# has to be written. `register_phrases` is the only thing standing between a
# language the model already reads and a sentence in it, and it is deliberately
# callable at runtime: the packs that ship here are a starting set, not the limit.
PHRASES = {
    "de": GERMAN,
    "en": ENGLISH,
    "es": SPANISH,
    "fr": FRENCH,
    "it": ITALIAN,
    "ja": JAPANESE,
    "pl": POLISH,
    "pt": PORTUGUESE,
    "ru": RUSSIAN,
    "tr": TURKISH,
    "zh": CHINESE,
}


def explain_shown(
    items: Sequence[tuple[str, str]],
    limit: int = 2,
    phrases: RankPhrases = ENGLISH,
) -> str:
    """The line a reader sees: why this candidate was shown at all.

    This names the viewer's own actions and nothing else: no rank, no competitor,
    no margin, no numbers. It is the weaker claim -- it says what the evidence was
    and not what it did -- and it is the only sentence this package puts in front of
    a reader, because "ranked 2 because X outweighs Y" is a sentence about the
    attribution rather than about the person reading it.

    `items` are the (kind, text) pairs of the evidence that favoured this candidate,
    strongest first.
    """
    if not items:
        return ""
    phrased = [
        phrases.evidence_phrase % {"kind": kind, "text": text} for kind, text in items[:limit]
    ]
    if len(phrased) == 1:
        joined = phrased[0]
    else:
        joined = phrases.join_two % (phrases.join_more.join(phrased[:-1]), phrased[-1])
    return phrases.shown_because % {"evidence": joined}


def languages() -> tuple[str, ...]:
    """Every language a sentence can currently be written in."""
    return tuple(sorted(PHRASES))


def phrases_for(code: str) -> RankPhrases:
    """The pack for one language, by code, or a message naming what is on offer."""
    try:
        return PHRASES[code]
    except KeyError:
        raise LookupError(
            "no phrase pack for %r; registered: %s" % (code, ", ".join(languages()))
        ) from None


def register_phrases(code: str, phrases: RankPhrases) -> None:
    """Add or replace the wording for one language.

    The pack is exercised here rather than trusted, because the failure it guards
    against is silent: a template with the wrong placeholder name formats cleanly
    and leaves `%(evidence)s` sitting in the sentence a reader sees. Registration is
    the last moment at which that can be blamed on the pack rather than on a
    request, so it is checked on the way in and named if it is wrong.
    """
    if not code:
        raise ValueError("a language pack needs a code")
    if not isinstance(phrases, RankPhrases):
        raise TypeError(
            "language %r: expected RankPhrases, got %s" % (code, type(phrases).__name__)
        )
    sample = [("searched for", "one thing"), ("viewed", "another")]
    for count in (1, 2):
        try:
            line = explain_shown(sample[:count], limit=count, phrases=phrases)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("language %r has a bad template: %s" % (code, error)) from error
        if not line:
            raise ValueError("language %r produced no sentence from %d reason(s)" % (code, count))
        for leftover in ("%(", "%s"):
            if leftover in line:
                raise ValueError(
                    "language %r left %s unfilled: %r" % (code, leftover, line)
                )
    PHRASES[code] = phrases
