"""The corpus's own invariants, which nothing else checks.

`examples/validate_corpus.py` checks whether the labels are *true*, using two
sentence encoders. These are the structural claims underneath that, which hold or
fail without any model: that a slate never contradicts itself, and that a held-out
category is genuinely held out.
"""

from __future__ import annotations

import collections
import random

import pytest
from corpus import (
    DECOY_SHARE,
    DOMAINS,
    HELD_OUT_CATEGORIES,
    NOISE,
    build_case,
    build_matched_case,
)

CASES = 240


def cases(skip=frozenset()):
    rng = random.Random(20261120)
    return [
        build_matched_case(rng, index) if index % 2 else build_case(rng, index, skip)
        for index in range(CASES)
    ]


def test_every_pair_is_generated_from_a_listed_category():
    for name, catalogue in DOMAINS.items():
        listed = set(catalogue.categories())
        for category, _want, _target in catalogue.pairs:
            assert category in listed, "%s: %r" % (name, category)


def test_several_phrasings_share_a_category():
    """The point of the table: more plant strings than semantic pairs.

    If every category had one phrasing, the plant string would identify the pair and
    a lookup table would be as good as understanding it.
    """
    total = sum(len(c.pairs) for c in DOMAINS.values())
    categories = sum(len(c.categories()) for c in DOMAINS.values())
    assert total > 2 * categories, (total, categories)


def test_a_slate_never_draws_one_category_twice():
    """Two candidates of the same kind would both answer the planted want.

    That is a wrong label rather than a hard case, and it is the failure an earlier
    near-twin candidate actually caused.
    """
    rng = random.Random(1)
    for name, catalogue in DOMAINS.items():
        for width in catalogue.widths(frozenset()):
            families = catalogue.families(rng, width, frozenset())
            seen = [family.category for family in families]
            assert len(set(seen)) == len(seen), name


def test_asking_for_more_families_than_exist_is_refused():
    rng = random.Random(2)
    for catalogue in DOMAINS.values():
        too_many = len(catalogue.categories()) + 1
        with pytest.raises(ValueError, match="usable categories"):
            catalogue.families(rng, too_many, frozenset())


def test_no_evidence_line_appears_twice_in_one_case():
    """A repeated unit cancels itself out under leave-one-out.

    Each copy is measured with the other still present, so both look worthless and
    the distractor stops distracting. Drawing with replacement from pools of three
    was how the previous corpus lost most of the strength of its distractors.
    """
    for case in cases():
        counts = collections.Counter(item.text for item in case["evidence"].items)
        assert max(counts.values()) == 1, case["id"]


def test_skipped_categories_appear_nowhere_in_a_slate():
    """Not as the plant, and not as a distractor either.

    A held-out pair sitting among the candidates would have taught the model that
    pair's listing, which is most of what holding it out was for.
    """
    for case in cases(HELD_OUT_CATEGORIES):
        if case["condition"] != "mixed":
            continue
        assert case["category"] not in HELD_OUT_CATEGORIES
        wanted = {
            want for catalogue in DOMAINS.values()
            for category, want, _target in catalogue.pairs
            if category in HELD_OUT_CATEGORIES
        }
        targets = {
            target for catalogue in DOMAINS.values()
            for category, _want, target in catalogue.pairs
            if category in HELD_OUT_CATEGORIES
        }
        for item in case["evidence"].items:
            assert item.text not in wanted
        for candidate in case["candidates"]:
            assert not any(candidate.text.startswith(target) for target in targets)


def test_held_out_categories_are_reachable_when_not_skipped():
    """The evaluation corpus has to be able to plant them, or there is no split."""
    planted = {case["category"] for case in cases() if case["category"]}
    assert planted & HELD_OUT_CATEGORIES


def test_the_planted_unit_carries_its_family_want():
    for case in cases():
        if case["key"] is None:
            continue
        kind, text = case["evidence"].phrase(case["key"])
        assert kind
        assert text


def test_declared_widths_and_unit_counts_are_all_achievable():
    """Both are stated rather than clipped, so both have to be honest.

    The previous `min(size, len(families))` silently collapsed four of six nominal
    widths onto one value, which left every claim about cost against slate width
    measured over a range the corpus never produced.
    """
    rng = random.Random(3)
    for name, catalogue in DOMAINS.items():
        for skip in (frozenset(), HELD_OUT_CATEGORIES):
            for width in catalogue.widths(skip):
                assert len(catalogue.families(rng, width, skip)) == width
            for units in catalogue.unit_counts(len(NOISE)):
                # The control case is the binding one: no plant, so one more
                # distractor is needed than a planted case of the same size.
                wanted = units - 1
                decoys = round(wanted * DECOY_SHARE)
                assert decoys <= len(catalogue.requests), (name, units)
                assert wanted - decoys <= len(NOISE), (name, units)


def test_the_corpus_is_reproducible_from_its_seed():
    first = [case["id"] for case in cases()]
    second = [case["id"] for case in cases()]
    assert first == second
    one = cases()[7]
    two = cases()[7]
    assert [i.text for i in one["evidence"].items] == [i.text for i in two["evidence"].items]
