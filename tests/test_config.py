"""Settings and language packs come from a file, so the file is what is tested.

The failure these guard against is silent rather than loud: a pack with a wrong
placeholder formats cleanly and puts `%(evidence)s` in front of a reader, and a
mistyped setting name would otherwise be accepted and ignored.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from evrank.config import ENV_VAR, Settings, phrases_from_mapping
from evrank.explain import (
    PHRASES,
    RankPhrases,
    explain_shown,
    languages,
    phrases_for,
    register_phrases,
)
from evrank.service import create_app

DUTCH = {
    "join_two": "%s en %s",
    "join_more": ", ",
    "shown_because": "Weergegeven omdat u %(evidence)s.",
    "evidence_phrase": "%(kind)s %(text)s",
}


@pytest.fixture(autouse=True)
def _restore_registry():
    """Registration is process-wide, so each test puts the registry back."""
    before = dict(PHRASES)
    yield
    PHRASES.clear()
    PHRASES.update(before)


def test_defaults_stand_when_no_file_is_given(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert Settings.from_env() == Settings()


def test_a_file_that_only_adds_a_language_is_a_valid_file(tmp_path):
    path = tmp_path / "evrank.json"
    path.write_text(json.dumps({"languages": {"nl": DUTCH}}), encoding="utf-8")

    settings = Settings.load(path)

    assert settings == Settings(), "nothing else in the file, nothing else changed"
    assert "nl" in languages()
    assert explain_shown([("gezocht naar", "ramen")], phrases=phrases_for("nl")) == (
        "Weergegeven omdat u gezocht naar ramen."
    )


def test_the_environment_names_the_file(tmp_path, monkeypatch):
    path = tmp_path / "evrank.json"
    path.write_text(json.dumps({"min_effect": 0.33, "language": "nl",
                                "languages": {"nl": DUTCH}}), encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(path))

    settings = Settings.from_env()

    assert settings.min_effect == 0.33
    assert settings.language == "nl"


def test_a_mistyped_setting_is_refused_with_the_known_names(tmp_path):
    with pytest.raises(ValueError, match="min_efect") as caught:
        Settings.from_mapping({"min_efect": 0.2})
    assert "min_effect" in str(caught.value), "the message offers the right spelling"


def test_a_pack_missing_a_field_is_refused_rather_than_half_english():
    incomplete = {name: value for name, value in DUTCH.items() if name != "evidence_phrase"}
    with pytest.raises(ValueError, match="evidence_phrase"):
        phrases_from_mapping("nl", incomplete)


def test_a_pack_with_the_wrong_placeholder_fails_on_the_way_in():
    wrong = dict(DUTCH, shown_because="Weergegeven omdat u %(reason)s.")
    with pytest.raises(ValueError, match="bad template"):
        register_phrases("nl", phrases_from_mapping("nl", wrong))
    assert "nl" not in languages(), "a pack that failed is not half-registered"


def test_a_pack_that_leaves_a_placeholder_in_the_sentence_is_refused():
    leftover = dict(DUTCH, evidence_phrase="%(kind)s %(text)s %(extra)")
    with pytest.raises(ValueError, match="bad template|unfilled"):
        register_phrases("nl", phrases_from_mapping("nl", leftover))


def test_an_unknown_language_names_what_is_registered():
    with pytest.raises(LookupError, match="registered:"):
        phrases_for("xx")


def test_registering_replaces_rather_than_duplicates():
    register_phrases("nl", RankPhrases(**DUTCH))
    register_phrases("nl", RankPhrases(**dict(DUTCH, shown_because="Omdat u %(evidence)s.")))
    assert languages().count("nl") == 1
    assert phrases_for("nl").shown_because == "Omdat u %(evidence)s."


def test_an_adapter_path_round_trips_through_a_config_file(tmp_path):
    """The field has to survive the file, because that is how it is set.

    An adapter that is configured but never reaches the scorer looks exactly like an
    adapter that is working: the service still answers, with the base model's
    numbers. Nothing in the response would say so, which is why `/api/model` now
    reports it and why this test exists.
    """
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"adapter": "tmp/lora-fold0"}), encoding="utf-8")
    settings = Settings.load(path)
    assert settings.adapter == "tmp/lora-fold0"
    assert Settings().adapter is None


def test_the_service_passes_the_adapter_to_the_scorer_it_builds():
    """Threaded, not merely stored.

    `create_app` builds its own scorer when none is injected, and that factory is
    the only place the setting can be honoured. Checked by building the app with a
    deliberately missing adapter directory and reading the error: reaching the load
    at all proves the path was passed through.
    """
    settings = Settings(adapter="tmp/definitely-not-an-adapter")
    app = create_app(settings=settings)
    client = TestClient(app)
    with pytest.raises(Exception) as raised:
        client.get("/api/model")
    assert "definitely-not-an-adapter" in str(raised.value)
