from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from toy_scorer import AdditiveScorer

from evrank.config import Settings
from evrank.explain import languages
from evrank.service import ScorerProvider, create_app

EVIDENCE = [
    {"id": "s1", "kind": "aradı", "text": "alpha", "group": "japon"},
    {"id": "s2", "kind": "görüntüledi", "text": "alpha tekrar", "group": "japon"},
    {"id": "r1", "kind": "rezervasyon", "text": "beta"},
]
CANDIDATES = [
    {"id": "a", "name": "Birinci", "text": "doc_a"},
    {"id": "b", "name": "İkinci", "text": "doc_b"},
]


def _provider() -> ScorerProvider:
    scorer = AdditiveScorer(
        base={"doc_a": 0.0, "doc_b": 0.5},
        weights={("alpha", "doc_a"): 1.0, ("beta", "doc_b"): 0.4},
    )
    return ScorerProvider(lambda: scorer)


# The floor shipped in `Settings` is in the real model's own logits. The toy scorer
# here has its own scale -- contributions of 1.0 and 0.4 by construction -- so the
# test app sets a floor to match it. That is the same coupling the settings
# docstring warns about, exercised rather than worked around.
TOY_SETTINGS = Settings(min_effect=0.1, min_standout=0.1, min_specificity=0.1)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(_provider(), settings=TOY_SETTINGS))


def explain(client: TestClient, **overrides):
    body = {"evidence": EVIDENCE, "candidates": CANDIDATES, "explain": True}
    body.update(overrides)
    return client.post("/api/rank", json=body)


def detail(client: TestClient, **overrides):
    return explain(client, detail=True, **overrides)


def test_ranking_is_one_batched_call(client):
    response = client.post("/api/rank", json={"evidence": EVIDENCE, "candidates": CANDIDATES})
    assert response.status_code == 200
    body = response.json()
    assert [row["name"] for row in body["ranking"]] == ["Birinci", "İkinci"]
    assert body["timing"]["modelPairs"] == 2
    assert body["timing"]["stages"] == [
        {"name": "ranking", "ms": pytest.approx(0, abs=1000), "modelPairs": 2, "batches": 1}
    ]


def test_explanation_is_null_unless_the_request_asks_for_it(client):
    body = client.post(
        "/api/rank", json={"evidence": EVIDENCE, "candidates": CANDIDATES}
    ).json()
    assert body["explanation"] is None
    assert body["agreement"] is None


def test_asking_for_an_explanation_costs_extra_stages(client):
    plain = client.post(
        "/api/rank", json={"evidence": EVIDENCE, "candidates": CANDIDATES}
    ).json()
    explained = explain(client).json()
    assert [stage["name"] for stage in plain["timing"]["stages"]] == ["ranking"]
    assert [stage["name"] for stage in explained["timing"]["stages"]] == ["ranking", "reasons"]
    assert [stage["name"] for stage in detail(client).json()["timing"]["stages"]] == [
        "ranking",
        "reasons",
        "attribution",
        "counterfactual: pair",
        "counterfactual: first place",
    ]
    assert explained["timing"]["modelPairs"] > plain["timing"]["modelPairs"]


def test_explaining_adds_reasons_to_each_candidate_and_nothing_else(client):
    body = explain(client).json()
    assert set(body) == {"ranking", "context", "explanation", "agreement", "timing"}
    assert body["explanation"] is None
    for row in body["ranking"]:
        assert set(row) == {"rank", "id", "name", "score", "method", "scale", "reasons"}


def test_the_response_says_each_thing_once(client):
    body = detail(client).json()
    pair = body["explanation"]["pair"]
    assert set(pair) == {"a", "b", "method", "marginAOverB", "marginWithNoEvidence"}
    # Names are not repeated here: they are in `ranking`, keyed by the same id.
    assert {row["id"]: row["name"] for row in body["ranking"]}[pair["a"]] == "Birinci"
    assert body["explanation"]["effects"][0]["favours"] in {"Birinci", "İkinci"}


def test_derivable_fields_are_left_out(client):
    effect = detail(client).json()["explanation"]["effects"][0]
    assert set(effect["acrossContexts"]) == {"lowest", "highest"}
    sampled = detail(client, estimator="sampled", permutations=8).json()
    assert "samplingError" in sampled["explanation"]["effects"][0]["acrossContexts"]


def test_the_masked_baseline_rides_with_the_detail(client):
    explanation = detail(client, masker="kind").json()["explanation"]
    assert "narrative" not in explanation
    assert explanation["maskedBaseline"].splitlines() == [
        "- aradı:",
        "- görüntüledi:",
        "- rezervasyon:",
    ]


def test_the_counterfactual_says_who_wins_and_by_how_much(client):
    flip = detail(client).json()["explanation"]["counterfactual"]
    assert flip["thenFirst"] == "İkinci"
    assert flip["insteadOf"] == "Birinci"
    assert flip["byMargin"] > 0


def test_the_first_place_counterfactual_is_null_when_it_repeats_the_pair(client):
    # The explained pair is the top two by default, so the two answers coincide.
    assert detail(client).json()["explanation"]["firstPlaceCounterfactual"] is None


def test_non_ascii_evidence_survives_the_round_trip(client):
    body = explain(client).json()
    assert "aradı" in body["context"]
    assert "rezervasyon" in body["context"]
    assert any(row["name"] == "İkinci" for row in body["ranking"])


def test_explanation_groups_evidence_into_units(client):
    body = detail(client).json()["explanation"]
    assert [effect["unit"] for effect in body["effects"]] == ["japon", "r1"]
    assert body["effects"][0]["effectOnMargin"] == pytest.approx(1.0)


def test_shapley_reports_spread_and_leave_one_out_does_not(client):
    shapley_effects = detail(client, estimator="shapley").json()["explanation"]["effects"]
    loo_effects = detail(client, estimator="loo").json()["explanation"]["effects"]
    assert all("acrossContexts" in effect for effect in shapley_effects)
    assert all("acrossContexts" not in effect for effect in loo_effects)


def test_the_pair_defaults_to_the_top_two_and_can_be_chosen(client):
    assert detail(client).json()["explanation"]["pair"]["a"] == "a"
    chosen = detail(client, pair=["b", "a"]).json()["explanation"]["pair"]
    assert (chosen["a"], chosen["b"]) == ("b", "a")


def test_counterfactual_is_returned_with_its_evidence(client):
    flip = detail(client).json()["explanation"]["counterfactual"]
    assert flip["remove"] == ["japon"]
    assert flip["evidence"][0].startswith("alpha")


def test_the_filler_masker_keeps_the_context_full_size(client):
    omission = detail(client, masker="omission").json()["explanation"]["maskedBaseline"]
    filler = detail(
        client, masker="filler", filler_text="hiçbir şey"
    ).json()["explanation"]["maskedBaseline"]
    assert omission == ""
    assert len(filler.splitlines()) == 3
    assert "hiçbir şey" in filler


def test_timing_accounts_for_cache_hits_as_well_as_model_calls(client):
    timing = explain(client).json()["timing"]
    assert timing["modelPairs"] > 2
    assert timing["pairsServedFromCache"] >= 0
    assert timing["totalMs"] >= sum(stage["ms"] for stage in timing["stages"]) - 1


def test_an_unknown_masker_is_rejected(client):
    assert explain(client, masker="nope").status_code == 400


def test_an_unknown_candidate_in_the_pair_is_rejected(client):
    assert explain(client, pair=["a", "ghost"]).status_code == 400


def test_a_single_candidate_is_rejected(client):
    assert explain(client, candidates=[CANDIDATES[0]]).status_code == 400


def test_duplicate_evidence_ids_are_rejected(client):
    assert explain(client, evidence=[EVIDENCE[0], EVIDENCE[0]]).status_code == 400


def test_too_many_units_for_exact_shapley_is_a_client_error_not_a_crash(client):
    many = [
        {"id": "e%d" % i, "kind": "aradı", "text": "token%d" % i} for i in range(13)
    ]
    # The reasons on each row sample above the cap rather than failing: an
    # explanation of the page should not stop being available at a size the ranking
    # itself handles.
    assert explain(client, evidence=many, estimator="shapley").status_code == 200
    assert explain(client, evidence=many, estimator="shapley").json()["ranking"][0][
        "method"
    ] == "sampled_shapley"
    # The margin attribution under `detail` is the one with a hard cap, and it says so.
    response = explain(client, evidence=many, estimator="shapley", detail=True)
    assert response.status_code == 400
    assert "group evidence" in response.json()["detail"]


def agreement(client: TestClient, **overrides):
    return explain(client, compare_maskers=True, **overrides)


def test_agreement_reports_every_masker_per_unit(client):
    body = agreement(client).json()["agreement"]
    assert body["maskers"] == ["omission", "kind", "filler"]
    assert set(body["units"][0]["effectByMasker"]) == {"omission", "kind", "filler"}
    assert {unit["unit"] for unit in body["units"]} == {"japon", "r1"}


def test_agreement_states_a_verdict_in_words(client):
    body = agreement(client).json()["agreement"]
    assert body["unitsTotal"] == 2
    for unit in body["units"]:
        assert unit["lowest"] <= unit["highest"]
        assert unit["verdict"] in {
            "disputed",
            "no effect",
            "favours Birinci",
            "favours İkinci",
        }


def test_agreement_compares_the_counterfactual_too(client):
    body = agreement(client).json()["agreement"]
    assert set(body["counterfactualByMasker"]) == {"omission", "kind", "filler"}
    assert isinstance(body["counterfactualIsTheSameUnderAll"], bool)


def test_agreement_costs_more_than_a_single_explanation(client):
    single = explain(client).json()
    both = agreement(client).json()
    assert both["timing"]["modelPairs"] > single["timing"]["modelPairs"]


def test_agreement_rejects_an_unknown_estimator(client):
    assert agreement(client, estimator="nope").status_code == 400


def test_reasons_carry_only_what_was_in_the_candidate_s_favour(client):
    rows = explain(client, top_reasons=0).json()["ranking"]
    for row in rows:
        assert row["reasons"], "every candidate here has something going for it"
        assert all(reason["effect"] > 0 for reason in row["reasons"])


def test_what_held_a_candidate_back_is_a_separate_opt_in_list(client):
    plain = explain(client).json()["ranking"][1]
    assert "against" not in plain
    rows = explain(client, include_against=True, top_reasons=0).json()["ranking"]
    assert all(entry["effect"] < 0 for entry in rows[1]["against"])
    # The evidence that carried the winner is what the runner-up is held back by.
    helped = {r["unit"]: r["effect"] for r in rows[0]["reasons"]}
    held = {r["unit"]: r["effect"] for r in rows[1]["against"]}
    assert helped["japon"] == pytest.approx(-held["japon"])


def test_reasons_lead_with_whatever_helped_most(client):
    reasons = explain(client, top_reasons=0).json()["ranking"][0]["reasons"]
    sizes = [reason["effect"] for reason in reasons]
    assert sizes == sorted(sizes, reverse=True)


def test_how_many_reasons_to_carry_is_the_caller_s_choice(client):
    rows = explain(client, include_against=True, top_reasons=1).json()["ranking"]
    assert len(rows[0]["reasons"]) == 1
    assert len(rows[0]["against"]) == 1


def test_ranking_without_an_explanation_carries_no_reasons(client):
    rows = client.post(
        "/api/rank", json={"evidence": EVIDENCE, "candidates": CANDIDATES}
    ).json()["ranking"]
    assert all(set(row) == {"rank", "id", "name", "score"} for row in rows)


def test_a_reason_names_the_evidence_it_came_from(client):
    reason = explain(client).json()["ranking"][0]["reasons"][0]
    assert reason["unit"] == "japon"
    assert reason["evidence"].startswith("alpha")


def test_reasons_cost_is_reported_as_its_own_stage(client):
    stages = {s["name"]: s for s in explain(client).json()["timing"]["stages"]}
    assert stages["reasons"]["modelPairs"] >= 0
    assert stages["reasons"]["ms"] >= 0


def test_the_detail_is_opt_in_and_costs_more(client):
    plain = explain(client).json()
    detailed = detail(client).json()
    assert detailed["explanation"]["effects"]
    assert detailed["timing"]["modelPairs"] >= plain["timing"]["modelPairs"]


def test_sentences_are_opt_in(client):
    assert "because" not in explain(client).json()["ranking"][0]
    with_sentence = explain(client, sentences=True).json()["ranking"][0]
    assert with_sentence["because"].startswith("Because you ")
    assert with_sentence["reasons"], "the list stays alongside the sentence"


def test_the_sentence_can_be_written_in_another_language(client):
    turkish = explain(client, sentences=True, language="tr").json()["ranking"]
    assert turkish[0]["because"].endswith("için öneriliyor.")
    assert turkish[1]["because"].endswith("için öneriliyor.")
    # Turkish puts the action after the fact, which is the pack's business, not the
    # caller's: the same request in English reads the other way round.
    english = explain(client, sentences=True).json()["ranking"]
    assert english[0]["because"].startswith("Because you ")
    # The measurements are the same whichever language states them.
    english = explain(client, sentences=True, language="en").json()["ranking"]
    assert turkish[0]["reasons"] == english[0]["reasons"]


def test_an_unknown_language_is_rejected_with_the_ones_that_exist(client):
    response = explain(client, sentences=True, language="xx")
    assert response.status_code == 400
    # Against the registry rather than a list written here: packs are meant to be
    # added, and a test that has to be edited to add one is a test against that.
    assert ", ".join(languages()) in response.json()["detail"]


def test_the_sentence_agrees_with_the_reasons_it_summarises(client):
    row = explain(client, sentences=True, top_reasons=0).json()["ranking"][0]
    strongest = max(row["reasons"], key=lambda reason: reason["effect"])
    assert strongest["evidence"][:12] in row["because"]


def test_the_languages_with_wording_are_listed(client):
    assert client.get("/api/model").json()["sentenceLanguages"] == list(languages())


def test_the_installations_own_defaults_are_published(client):
    """What is published is this installation's settings, not the built-in ones."""
    defaults = client.get("/api/model").json()["defaults"]
    assert defaults["min_effect"] == TOY_SETTINGS.min_effect
    assert defaults["language"] == TOY_SETTINGS.language
    assert defaults["model"] == TOY_SETTINGS.model


def test_a_request_that_says_nothing_takes_the_installations_floor():
    """A deployment's floor is not the constant in the request model."""
    settings = Settings(min_effect=99.0)  # nothing any scorer here produces
    with TestClient(create_app(provider=_provider(), settings=settings)) as strict:
        response = explain(strict, sentences=True)
        assert response.status_code == 200
        rows = response.json()["ranking"]
        assert all(not row["reasons"] for row in rows), "nothing clears a floor of 99"
        # And the request still outranks it when it says so.
        relaxed = explain(strict, sentences=True, min_effect=0.0).json()["ranking"]
        assert any(row["reasons"] for row in relaxed)


def test_the_sentence_names_the_action_not_the_rival(client):
    row = explain(client, sentences=True).json()["ranking"][0]
    # The kind becomes the verb and the text follows it.
    assert row["because"].startswith("Because you aradı alpha")


def test_the_sentence_cites_exactly_its_own_reasons(client):
    rows = explain(client, sentences=True, top_reasons=0).json()["ranking"]
    for row in rows:
        for reason in row["reasons"]:
            assert reason["evidence"] in row["because"]
        if not row["reasons"]:
            assert row["because"] == ""


def test_two_reasons_are_cited_unless_asked_for_more():
    # Its own scorer, because the shared one has only one reason to offer and a cap
    # tested against one reason is not tested at all.
    two = TestClient(
        create_app(
            ScorerProvider(
                lambda: AdditiveScorer(
                    base={"doc_a": 0.0, "doc_b": 0.5},
                    weights={("alpha", "doc_a"): 1.0, ("beta", "doc_a"): 0.6},
                )
            ),
            settings=TOY_SETTINGS,
        )
    )
    assert len(explain(two, top_reasons=0).json()["ranking"][0]["reasons"]) == 2
    # Unset means two, the shape of "Because you searched X and viewed Y".
    assert len(explain(two).json()["ranking"][0]["reasons"]) == 2
    assert len(explain(two, top_reasons=1).json()["ranking"][0]["reasons"]) == 1


def test_a_reason_that_did_as_much_for_everyone_is_not_printed(client):
    rows = explain(client, top_reasons=0).json()["ranking"]
    for row in rows:
        for reason in row["reasons"]:
            assert reason["specificity"] >= 0.1
    # Dropping the floor can only admit more reasons, never fewer.
    loose = explain(client, top_reasons=0, min_specificity=0.0).json()
    for row, relaxed in zip(rows, loose["ranking"]):
        assert len(relaxed["reasons"]) >= len(row["reasons"])


def test_a_reason_separates_the_specific_part_from_the_raw_effect(client):
    row = explain(client, top_reasons=0).json()["ranking"][0]
    reason = row["reasons"][0]
    # The unit raised this row by 1.0 and the only other row by nothing, so all of
    # the effect was specific to it and the two numbers agree.
    assert reason["onItsOwnScore"] == pytest.approx(1.0)
    assert reason["effect"] == pytest.approx(1.0)
    assert reason["specificity"] == pytest.approx(1.0)


def test_the_sentence_follows_the_language_pack(client):
    row = explain(client, sentences=True, language="tr").json()["ranking"][0]
    # Turkish puts the action after the fact, so the pack orders them the other way.
    assert row["because"].startswith("alpha aradı")
    assert row["because"].endswith("için öneriliyor.")


def test_the_page_is_revalidated_so_an_edit_is_not_served_stale(client):
    response = client.get("/")
    assert response.status_code == 200
    # Without this a browser picks its own freshness window and can show an
    # explanation the running code no longer produces.
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["etag"]


def test_the_dynamic_floor_is_off_unless_asked_for(client):
    loose = explain(client, top_reasons=0).json()["ranking"]
    tight = explain(client, top_reasons=0, min_share=0.9).json()["ranking"]
    assert sum(len(row["reasons"]) for row in loose) >= sum(
        len(row["reasons"]) for row in tight
    )


def test_the_page_version_changes_when_the_page_does(client):
    first = client.get("/api/version").json()["page"]
    assert isinstance(first, int)
    # An open tab polls this; it has to move when the file moves, not when the
    # process restarts, or a reload prompt would appear for no reason.
    assert client.get("/api/version").json()["page"] == first


def test_only_the_first_choice_is_explained_when_asked(client):
    rows = explain(client, sentences=True, top_only=True).json()["ranking"]
    assert "because" in rows[0] and rows[0]["reasons"]
    for row in rows[1:]:
        assert set(row) == {"rank", "id", "name", "score"}


def test_explaining_only_the_first_choice_changes_nothing_about_it(client):
    whole = explain(client, sentences=True).json()["ranking"][0]
    alone = explain(client, sentences=True, top_only=True).json()["ranking"][0]
    assert whole == alone


def test_narrowing_the_report_does_not_narrow_the_measurement(client):
    """The effects are centred across the slate, so every candidate is still scored.

    If this ever got cheaper it would mean a reason was being computed without
    anything to stand out from.
    """
    whole = explain(client, sentences=True).json()["timing"]
    alone = explain(client, sentences=True, top_only=True).json()["timing"]
    assert whole["modelPairs"] == alone["modelPairs"]


def test_sentences_survive_the_masker_comparison(client):
    # Asking for both used to answer with the agreement report and no sentences at
    # all, saying nothing about having dropped one of the two things requested.
    body = explain(client, sentences=True, compare_maskers=True).json()
    assert body["agreement"]
    assert body["ranking"][0]["because"].startswith("Because you ")


def test_each_row_carries_the_scale_its_effects_are_measured_on(client):
    rows = explain(client).json()["ranking"]
    # One unit for the whole request, so two rows can be compared with each other.
    assert len({row["scale"] for row in rows}) == 1
    assert rows[0]["scale"] > 0
