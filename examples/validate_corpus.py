"""Check the evaluation corpus's own data before trusting a number measured on it.

Three claims hold up every figure measured on it, and none of them had been checked
by anyone who speaks the languages:

    discriminative  `want` picks out its own `target` over the others on the page
    absent          no decoy request describes a listing better than its own does
    faithful        `want` in each language means what `want` in English means

The first is checked on every pair the corpus can plant, against every rival that
pair can meet, because that is the form the label is used in. It passes by
construction -- the table in `corpus.py` was filtered on exactly this criterion --
and that is what makes it worth running: it is a regression check, and content is
what drifts.

A failure of the first makes the ground-truth label wrong, so the system is marked
down for giving an answer that is at least as good as the one recorded. A failure of
the second makes the per-language comparison a measurement of the translations.

Deliberately not scored with the re-ranker under test. Selecting the phrasings that
model happens to like would raise its own score, and the question here is what a
competent reader would say, not what the ranker says. Two multilingual sentence
encoders from unrelated families stand in for that reader, and they have to agree.

Run: uv run python examples/validate_corpus.py [--verbose]
"""

from __future__ import annotations

import collections
import sys

import torch
from corpus import DOMAINS, MATCHED
from sentence_transformers import SentenceTransformer

# Trained for cross-lingual paraphrase similarity, which is exactly the question,
# and from a different lineage than any re-ranker in the benchmark.
PARAPHRASE_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# A second opinion from an unrelated family. It wants its inputs prefixed.
SECOND_MODEL = "intfloat/multilingual-e5-large"

# Below this, two phrasings of the same thing are not the same thing. Calibrated
# against the English pack, where `want` and `target` are known to correspond.
FAITHFUL_FLOOR = 0.70
# A want has to beat the runners-up by this much to be an unambiguous label.
MARGIN_FLOOR = 0.02


Pairing = collections.namedtuple("Pairing", "want target")


class Reader:
    """Two encoders that have to agree, so one model's quirk is not a verdict."""

    def __init__(self) -> None:
        self.paraphrase = SentenceTransformer(PARAPHRASE_MODEL)
        self.second = SentenceTransformer(SECOND_MODEL)

    def similarity(self, left: list[str], right: list[str]) -> torch.Tensor:
        """Mean cosine over both encoders, pairwise between the two lists."""
        totals = []
        for model, prefix in ((self.paraphrase, ""), (self.second, "query: ")):
            a = model.encode([prefix + text for text in left], convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
            b = model.encode([prefix + text for text in right], convert_to_tensor=True,
                             normalize_embeddings=True, show_progress_bar=False)
            totals.append(a @ b.T)
        return sum(totals) / len(totals)


def discriminative(reader: Reader, name: str, families, verbose: bool) -> list[tuple]:
    """Does each `want` pick out its own `target` over the others on the page?"""
    wants = [family.want for family in families]
    targets = [family.target for family in families]
    scores = reader.similarity(wants, targets)
    failures = []
    for index, family in enumerate(families):
        row = scores[index]
        own = float(row[index])
        rivals = [(float(row[other]), families[other].target)
                  for other in range(len(families)) if other != index]
        best_rival, rival_text = max(rivals)
        margin = own - best_rival
        if margin < MARGIN_FLOOR:
            failures.append((name, family.want, family.target, rival_text, own, margin))
        if verbose:
            print("    %-6s %-52s own %.3f  margin %+.3f%s"
                  % (name, family.want[:52], own, margin,
                     "   <- %s" % rival_text[:34] if margin < MARGIN_FLOOR else ""))
    return failures


def decoys_match_nothing(reader: Reader, name: str, requests, families, verbose: bool):
    """A decoy must not be closer to any listing than that listing's own request.

    Decoys are request-shaped lines for things absent from the page, and they only do
    their job if they are genuinely absent. A decoy that happens to describe a
    listing better than that listing's own request does would make the label wrong in
    the opposite direction from before -- the model would be marked down for citing
    the better answer.
    """
    if not requests:
        return []
    targets = [family.target for family in families]
    wants = [family.want for family in families]
    decoy_scores = reader.similarity(list(requests), targets)
    own_scores = reader.similarity(wants, targets)
    failures = []
    for index, request in enumerate(requests):
        for column, target in enumerate(targets):
            theirs = float(decoy_scores[index][column])
            owner = float(own_scores[column][column])
            if theirs >= owner:
                failures.append((name, request, target, theirs, owner))
            if verbose:
                print("    %-6s %-40s vs %-32s %.3f (own %.3f)"
                      % (name, request[:40], target[:32], theirs, owner))
    return failures


def shipped(reader: Reader, name: str, catalogue, verbose: bool) -> list[tuple]:
    """Every pair the corpus can actually plant, against every rival it can meet.

    This is the claim in the form the corpus uses it: a want is drawn with one of its
    phrasings, the other candidates come from other categories with any of theirs,
    and the planted label says this want belongs to this listing. So each shipped
    want is checked against every target of every other category rather than against
    one representative -- a phrasing that only beat some of its rivals would produce
    a wrong label whenever the others turned up.

    It passes by construction, because the table in `corpus.py` was filtered on this
    criterion. That is what makes it worth running: it is a regression check, and
    content is what drifts.
    """
    failures = []
    wants = [(row[0], row[1]) for row in catalogue.pairs]
    targets = [(row[0], row[2]) for row in catalogue.pairs]
    scores = reader.similarity([w for _, w in wants], [t for _, t in targets])
    for index, (category, want) in enumerate(wants):
        own = float(scores[index][index])
        rivals = [
            (float(scores[index][other]), targets[other][1])
            for other in range(len(targets))
            if targets[other][0] != category
        ]
        if not rivals:
            continue
        best_rival, rival_text = max(rivals)
        margin = own - best_rival
        if margin < MARGIN_FLOOR:
            failures.append((name, want, targets[index][1], rival_text, own, margin))
        if verbose:
            print("    %-6s %-52s own %.3f  margin %+.3f"
                  % (name, want[:52], own, margin))
    return failures


def faithful(reader: Reader, verbose: bool) -> list[tuple]:
    """Does each language's `want` mean what the English one means?

    The packs carry the same three categories in the same order, so family `i` is
    the same concept in every language and can be compared position by position.
    """
    english = MATCHED["en"].families
    failures = []
    for code in sorted(MATCHED):
        if code == "en":
            continue
        pack = MATCHED[code]
        theirs = [family.want for family in pack.families]
        ours = [family.want for family in english]
        scores = reader.similarity(theirs, ours)
        for index in range(min(len(theirs), len(ours))):
            score = float(scores[index][index])
            if score < FAITHFUL_FLOOR:
                failures.append((code, theirs[index], ours[index], score))
            if verbose:
                print("    %-4s %-46s vs en  %.3f%s"
                      % (code, theirs[index][:46], score,
                         "   <- adrift" if score < FAITHFUL_FLOOR else ""))
    return failures


def main(*flags: str) -> None:
    verbose = "--verbose" in flags
    reader = Reader()
    print("read by %s and %s, both unrelated to any re-ranker benchmarked here\n"
          % (PARAPHRASE_MODEL.split("/")[-1], SECOND_MODEL.split("/")[-1]))

    print("=== Does each request pick out its own listing? (margin over the runner-up)")
    failures = []
    for name, domain in sorted(DOMAINS.items()):
        failures += shipped(reader, name, domain, verbose)
    for code in sorted(MATCHED):
        failures += discriminative(reader, code, MATCHED[code].families, verbose)
    if failures:
        print("  %d ambiguous label(s): the request does not clearly prefer its own listing"
              % len(failures))
        for name, want, target, rival, own, margin in failures:
            print("    %-6s %s" % (name, want))
            print("    %-6s   labelled  %-44s %.3f" % ("", target[:44], own))
            print("    %-6s   but       %-44s %+.3f" % ("", rival[:44], -margin + own))
    else:
        print("  every request prefers its own listing by at least %.2f" % MARGIN_FLOOR)

    print("\n=== Do the decoy requests match nothing on the page?")
    decoys = []
    for name, domain in sorted(DOMAINS.items()):
        decoys += decoys_match_nothing(
            reader, name, domain.requests,
            [Pairing(row[1], row[2]) for row in domain.pairs], verbose
        )
    for code in sorted(MATCHED):
        decoys += decoys_match_nothing(
            reader, code, MATCHED[code].requests, MATCHED[code].families, verbose
        )
    if decoys:
        print("  %d decoy(s) describe a listing better than its own request does:"
              % len(decoys))
        for name, request, target, theirs, owner in decoys:
            print("    %-6s %s" % (name, request))
            print("    %-6s   beats %-44s %.3f against %.3f" % ("", target[:44], theirs, owner))
    else:
        print("  every decoy sits below the listing's own request, for every listing")

    print("\n=== Does each translated request mean what the English one means?")
    adrift = faithful(reader, verbose)
    if adrift:
        print("  %d translation(s) below %.2f:" % (len(adrift), FAITHFUL_FLOOR))
        for code, theirs, ours, score in adrift:
            print("    %-4s %.3f  %s" % (code, score, theirs))
            print("    %-4s        should mean: %s" % ("", ours))
    else:
        print("  every translation is at or above %.2f against its English original"
              % FAITHFUL_FLOOR)

    print("\n%d problem(s) found." % (len(failures) + len(decoys) + len(adrift)))


if __name__ == "__main__":
    main(*sys.argv[1:])
