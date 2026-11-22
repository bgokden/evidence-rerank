from __future__ import annotations

from toy_scorer import AdditiveScorer

from evrank.attribution import flip_pair, flip_top, shapley
from evrank.evidence import Evidence, EvidenceSet
from evrank.explain import CAVEAT, explain_pair, explain_top
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer

CANDIDATES = [
    Candidate("a", "Shiso Ramen", "doc_a"),
    Candidate("b", "Le Petit Marais", "doc_b"),
]


def build() -> Ablator:
    inner = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.5},
        weights={("ramen", "doc_a"): 1.0, ("bistro", "doc_b"): 0.4},
    )
    evidence = EvidenceSet(
        [
            Evidence("japanese", "searched for", "ramen in Paris"),
            Evidence("french", "searched for", "bistro menus"),
        ]
    )
    return Ablator(CachingScorer(inner), evidence, CANDIDATES)


def test_pair_explanation_names_items_evidence_and_counterfactual():
    ablator = build()
    text = explain_pair(
        shapley(ablator, "a", "b"), ablator.base(), flip_pair(ablator, "a", "b")
    )
    assert "Shiso Ramen ranks above Le Petit Marais" in text
    assert "ramen in Paris" in text
    assert "favours Shiso Ramen" in text
    assert 'Counterfactual: without "ramen in Paris", Le Petit Marais ranks above' in text
    assert CAVEAT in text


def test_pair_explanation_reports_an_absent_counterfactual_honestly():
    ablator = build()
    text = explain_pair(
        shapley(ablator, "a", "b"), ablator.base(), None, searched_up_to=1
    )
    assert "no omission of any single evidence unit reverses this pair" in text.lower()


def test_explanation_never_mentions_latent_structure():
    ablator = build()
    text = explain_pair(
        shapley(ablator, "a", "b"), ablator.base(), flip_pair(ablator, "a", "b")
    )
    for word in ("dimension", "neuron", "embedding", "feature"):
        assert word not in text.lower()


def test_top_explanation_names_the_replacement_item():
    ablator = build()
    text = explain_top(ablator.base(), flip_top(ablator))
    assert "Shiso Ramen ranks first" in text
    assert "Le Petit Marais ranks first instead" in text


def test_the_sentence_names_the_action_and_never_the_rank():
    """The one sentence a reader sees. No rank, no rival, no margin, no numbers.

    "X is ranked 2 because A outweighs B" is a sentence about the attribution. This
    is a sentence about the person reading it, so that is the only one that ships.
    """
    from evrank.explain import PHRASES, explain_shown

    text = explain_shown([("searched for", "ramen in Paris")], phrases=PHRASES["en"])
    assert text == "Because you searched for ramen in Paris."
    for word in ("ranked", "outweighs", "ahead of", "1.42", "+"):
        assert word not in text


def test_the_sentence_joins_the_reasons_it_was_given():
    from evrank.explain import PHRASES, explain_shown

    text = explain_shown(
        [("searched for", "ramen in Paris"), ("viewed", "a noodle bar")],
        limit=2,
        phrases=PHRASES["en"],
    )
    assert text == "Because you searched for ramen in Paris and viewed a noodle bar."


def test_nothing_to_cite_is_no_sentence_rather_than_an_empty_one():
    from evrank.explain import PHRASES, explain_shown

    assert explain_shown([], phrases=PHRASES["en"]) == ""


def test_every_pack_builds_a_sentence_from_one_reason_and_from_several():
    from evrank.explain import PHRASES, explain_shown

    items = [("aradı", "ramen"), ("gördü", "izakaya"), ("dedi", "iki kisi")]
    for code, phrases in PHRASES.items():
        for count in (1, 2, 3):
            text = explain_shown(items[:count], limit=count, phrases=phrases)
            assert text, code
            assert "%(" not in text, (code, count)
            assert "%s" not in text, (code, count)


def test_turkish_keeps_its_diacritics():
    from evrank.explain import TURKISH, explain_shown

    text = explain_shown([("aradığınız", "Paris'te ramen restoranları")], phrases=TURKISH)
    assert text == "Paris'te ramen restoranları aradığınız için öneriliyor."
