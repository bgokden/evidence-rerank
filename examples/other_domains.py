"""The explanation on cases it was not built around.

walkthrough.py is the example every design decision here was tuned against, which
makes it the least informative thing to read. These are four slates it has never
seen: another domain entirely, a page where the user's constraint is contradicted, a
page the model cannot tell apart, and one where the evidence and the listings are in
different languages.

Some of what it prints is wrong, and that is the point of running it. Each case ends
with what the output shows about the ranker rather than about the explanation.

Run: uv run python examples/other_domains.py [model-name]
"""

from __future__ import annotations

import sys

from evrank.cross_encoder import MULTILINGUAL_MODEL, CrossEncoderScorer
from evrank.evidence import Evidence, EvidenceSet
from evrank.explain import PHRASES, explain_shown
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer, rank
from evrank.standout import standout

# The service defaults. See RankRequest.min_specificity and min_effect.
MIN_SPECIFICITY = 0.1
MIN_EFFECT = 0.1


def berlin_flats() -> tuple[EvidenceSet, list[Candidate], str]:
    evidence = EvidenceSet(
        [
            Evidence("q1", "searched for", "two-bedroom flats near a U-Bahn station"),
            Evidence("q2", "filtered to", "pets allowed"),
            Evidence("b1", "told us", "budget around 1800 EUR cold rent"),
            Evidence("s1", "saved", "a listing in Kreuzberg"),
        ]
    )
    candidates = [
        Candidate("a", "Gneisenaustrasse 12", "Bright two-bedroom in Kreuzberg, four "
                  "minutes from U7. Cats and dogs welcome. 1750 EUR cold rent."),
        Candidate("b", "Prenzlauer Allee 88", "Renovated three-bedroom in Prenzlauer "
                  "Berg with a balcony. No pets. 2600 EUR cold rent."),
        Candidate("c", "Tempelhofer Weg 4", "Studio in an industrial conversion, 20 "
                  "minutes by bus to the nearest station. 990 EUR cold rent."),
        Candidate("d", "Wiener Strasse 31", "Two-bedroom above a bakery in Kreuzberg, "
                  "no lift, fifth floor. 1820 EUR cold rent."),
    ]
    return evidence, candidates, (
        "The transport search and the budget land on the right rows. Two others are\n"
        "  wrong in ways worth reading. A 990 EUR studio is cited for a budget of\n"
        "  1800, because the model matches the shape of a\n"
        "  price and does not compare numbers. The flat that says 'No pets' is cited\n"
        "  for a pets filter, because the model does not read the negation. Both are\n"
        "  reported faithfully: the explanation is right about a ranker that is wrong."
    )


def vegetarian() -> tuple[EvidenceSet, list[Candidate], str]:
    evidence = EvidenceSet(
        [
            Evidence("d1", "told us", "strictly vegetarian, no fish"),
            Evidence("q1", "searched for", "tasting menu for an anniversary"),
            Evidence("b1", "told us", "happy to spend, special occasion"),
        ]
    )
    candidates = [
        Candidate("a", "Maison Verte", "Vegetarian tasting menu, seven courses, "
                  "seasonal vegetables. 95 EUR per person."),
        Candidate("b", "The Grill Room", "Dry-aged beef and chops over charcoal. "
                  "Tasting menu 110 EUR per person."),
        Candidate("c", "Kanto", "Omakase sushi counter, twelve seats. 140 EUR per person."),
        Candidate("d", "Cafe Oriel", "All-day cafe with sandwiches and salads. About "
                  "18 EUR per person."),
    ]
    return evidence, candidates, (
        "A steakhouse and a sushi counter rank above nothing for a strict vegetarian,\n"
        "  and the explanation says why: every row is carried by the tasting-menu\n"
        "  search or the budget, never by the dietary constraint. That constraint\n"
        "  moves the vegetarian restaurant's own score by +0.03, which is to say the\n"
        "  ranker is ignoring it. The explanation cannot fix that and should not\n"
        "  disguise it."
    )


def indistinguishable() -> tuple[EvidenceSet, list[Candidate], str]:
    evidence = EvidenceSet(
        [
            Evidence("o1", "searched for", "winter tyres for a Volvo V60"),
            Evidence("o2", "viewed", "a guide to filing German tax returns"),
            Evidence("o3", "told us", "two people"),
        ]
    )
    candidates = [
        Candidate("a", "Shiso Ramen", "Tonkotsu ramen counter in the 2nd. Open until "
                  "23:00. About 22 EUR per person."),
        Candidate("b", "Trattoria Verde", "Handmade pasta near Bastille. Open until "
                  "22:30. About 35 EUR per person."),
        Candidate("c", "Brasserie du Nord", "Steak frites all day by the station. "
                  "About 30 EUR per person."),
    ]
    return evidence, candidates, (
        "Nothing in the evidence is about dinner, the three candidates score within\n"
        "  0.01 of each other, and the honest output is silence. One of them still\n"
        "  has the largest tiny effect, so without a floor on the size of an effect\n"
        "  this page reads 'Because you viewed a guide to filing German tax returns'\n"
        "  on a delta of +0.009."
    )


def cross_lingual() -> tuple[EvidenceSet, list[Candidate], str]:
    evidence = EvidenceSet(
        [
            Evidence("s1", "aradığınız", "Paris'te ramen restoranları"),
            Evidence("h1", "rezervasyon yaptığınız", "Rue de Rivoli'de otel, 1. bölge"),
            Evidence("b1", "belirttiğiniz", "orta bütçe, kişi başı yaklaşık 40 EUR"),
        ]
    )
    candidates = [
        Candidate("a", "Shiso Ramen", "ルーヴル美術館の近く、リヴォリ通りから徒歩圏内の"
                  "日本のラーメン店。23時まで営業。一杯およそ18ユーロ。"),
        Candidate("b", "Le Petit Marais", "Bistrot français classique dans le Marais, "
                  "3e arrondissement. Menu trois plats 42 EUR."),
        Candidate("c", "Brasserie du Nord", "Große Brasserie am Gare du Nord, den "
                  "ganzen Tag Steak Frites. Etwa 30 EUR pro Person."),
    ]
    return evidence, candidates, (
        "Turkish evidence, Japanese, French and German listings, and a Turkish\n"
        "  sentence. The quoted clause stays in the language it was written in,\n"
        "  because the explanation reports the listing rather than translating it."
    )


CASES = {
    "Flats in Berlin": berlin_flats,
    "A vegetarian, with a steakhouse on the page": vegetarian,
    "Evidence about nothing on the page": indistinguishable,
    "Turkish evidence, listings in three languages": cross_lingual,
}


def main(model_name: str = MULTILINGUAL_MODEL) -> None:
    inner = CrossEncoderScorer(model_name)
    print("model: %s\n" % model_name)
    for title, build in CASES.items():
        evidence, candidates, note = build()
        language = "tr" if build is cross_lingual else "en"
        phrases = PHRASES[language]
        scorer = CachingScorer(inner)
        ranking = rank(scorer, evidence, candidates)
        rows = standout(Ablator(scorer, evidence, candidates))

        print("=== %s" % title)
        for cid in ranking.order():
            candidate = next(c for c in candidates if c.id == cid)
            favouring = rows[cid].favouring(
                min_specificity=MIN_SPECIFICITY, min_effect=MIN_EFFECT
            )
            if not favouring:
                print("  %-21s (nothing singled it out)" % candidate.name)
                continue
            best = favouring[0]
            print(
                "  %-21s %s  [%+.2f on its own score]"
                % (
                    candidate.name,
                    explain_shown([evidence.phrase(best.unit)], 1, phrases),
                    best.delta,
                )
            )
        print("\n  %s\n" % note)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else MULTILINGUAL_MODEL)
