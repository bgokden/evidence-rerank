from __future__ import annotations

import pytest

from evrank.evidence import Evidence, EvidenceSet


def make_set() -> EvidenceSet:
    return EvidenceSet(
        [
            Evidence("s1", "searched for", "ramen in Paris", group="japanese"),
            Evidence("s2", "searched for", "izakaya near the Louvre", group="japanese"),
            Evidence("r1", "reserved", "hotel in the 1st arrondissement"),
        ]
    )


def test_units_group_related_facts():
    assert make_set().units() == ("japanese", "r1")


def test_a_label_is_the_unit_s_own_words_with_no_prose_added():
    # Prose in a label would be English inside whatever language the caller wrote.
    assert make_set().label("japanese") == "ramen in Paris"


def test_how_many_facts_a_unit_holds_is_a_number_not_a_suffix():
    assert make_set().count("japanese") == 2
    assert make_set().count("r1") == 1


def test_a_unit_s_phrase_is_the_action_and_the_fact():
    assert make_set().phrase("japanese") == ("searched for", "ramen in Paris")
    assert make_set().label("r1") == "hotel in the 1st arrondissement"


def test_without_drops_every_fact_in_the_unit():
    reduced = make_set().without(["japanese"])
    assert [item.id for item in reduced.items] == ["r1"]
    assert "ramen" not in reduced.render()
    assert "izakaya" not in reduced.render()


def test_render_omits_rather_than_masks():
    rendered = make_set().without(["japanese"]).render()
    assert rendered.splitlines() == ["- reserved: hotel in the 1st arrondissement"]


def test_empty_set_renders_as_the_no_context_baseline():
    empty = make_set().without(["japanese", "r1"])
    assert len(empty) == 0
    assert empty.render() == ""


def test_no_header_is_written_by_default():
    assert not make_set().render().startswith("Traveler")


def test_a_header_that_is_set_opens_the_context():
    labelled = EvidenceSet(make_set().items, header="Traveler context:")
    assert labelled.render().splitlines()[0] == "Traveler context:"
    assert labelled.header_lines() == ["Traveler context:"]


def test_a_blank_header_leaves_no_line_at_all():
    assert EvidenceSet(make_set().items, header="   ").header_lines() == []


def test_keeping_is_the_complement_of_without():
    assert make_set().keeping(["r1"]).render() == make_set().without(["japanese"]).render()


def test_unknown_unit_is_an_error():
    with pytest.raises(KeyError):
        make_set().without(["no_such_unit"])


def test_duplicate_evidence_id_is_an_error():
    with pytest.raises(ValueError):
        EvidenceSet([Evidence("s1", "searched for", "a"), Evidence("s1", "reserved", "b")])
