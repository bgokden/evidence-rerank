"""Deployment settings and language packs, loaded from a file rather than edited in.

Two things here are properties of a deployment and not of the method: which model
ranks, and the floors under what counts as a reason. A third, the wording of the
sentence, is a property of the audience. None of them is a reason to change code,
so all three come from one JSON file:

    {
      "model": "BAAI/bge-reranker-v2-m3",
      "min_effect": 0.1,
      "languages": {
        "nl": {
          "join_two": "%s en %s",
          "join_more": ", ",
          "shown_because": "Weergegeven omdat u %(evidence)s.",
          "evidence_phrase": "%(kind)s %(text)s"
        }
      }
    }

Every key is optional and anything absent keeps its built-in default, so a file that
only adds a language is a valid file. Packs are registered as they load and checked
on the way in, so a wrong placeholder fails at startup with the language named
rather than on the first request that asks for it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from collections.abc import Mapping

from evrank import cross_encoder
from evrank.explain import RankPhrases, register_phrases

ENV_VAR = "EVRANK_CONFIG"

# Resolving the default model name here would import the model package, and the core
# path deliberately has no deep-learning dependency. The service resolves it.
# Imported rather than repeated. Two constants of this name with different values is
# how the English-only model ended up being what a bare scorer loaded.
DEFAULT_MODEL = cross_encoder.DEFAULT_MODEL


@dataclasses.dataclass(frozen=True)
class Settings:
    """What a deployment chose, standing behind every request that does not say.

    A request may still override any of these per call; these are the defaults it
    falls back to, which is the difference between tuning one query and configuring
    an installation.
    """

    model: str = DEFAULT_MODEL
    """Which re-ranker scores. It is paired with `min_effect`: that floor is in the
    model's own uncalibrated logits, so changing one without re-sweeping the other
    is a silent misconfiguration. The two defaults here were swept together."""
    adapter: str | None = None
    """A directory written by `examples/finetune.py`, layered on at load time.

    The base weights are left exactly as downloaded and the adapter is applied on
    top, which is the point: if it turns out to make the ranking worse there is
    nothing to roll back, because omitting this setting restores the model exactly.

    It is not free, and the cost is in general ranking rather than in attribution.
    Measured over six independent training runs against a held-out retrieval set of
    a different dataset lineage, an adapter trained this way cost around 1.5 to 2
    points of ordinary ranking accuracy; every run leaned the same way, and two of
    the six fell outside what the guard could explain as noise. What it buys, on
    pairs its training never saw, is about 13 points of attribution recall and 20 of
    precision. Whether that trade is worth making is a deployment's decision, which
    is why this is a setting and not a default."""
    language: str = "en"
    top_reasons: int = 2
    top_only: bool = False
    min_specificity: float = 0.3
    min_standout: float = 0.75
    min_effect: float = 0.1
    min_share: float = 0.0
    calibrator: str | None = None
    """A JSON file written by `examples/fit_calibrator.py`, used in place of the
    floors above when it is set.

    The floors are three constants swept by hand; this is a small tree ensemble over
    the same measurements, and it is better at the one thing the floors are for.
    Measured over four folds of a 48-category corpus with every category held out
    exactly once, at matched first-row false positives: the floors print 41.0%
    correct sentences and invent on 10.2% of rows with no cause, and this reads
    59.6% and 4.8%. It gains no recall, because it only judges effects the ablation
    already found, and it cannot affect ranking, because ranking has finished before
    it runs.

    It belongs to the model it was fitted on, for the same reason `min_effect` does:
    it reads uncalibrated logits. Refitting takes seconds against a dumped matrix,
    so pair it with `model` and refit rather than carry one across."""
    min_confidence: float = 0.92
    """The floor under a calibrator's probability that an effect is a real cause.

    What `min_standout` is for the floors, and chosen the same way: bisected to a
    25% first-row false-positive rate on held-out categories, which landed at 0.914,
    0.928, 0.942 and 0.921 across four folds. The default is their median.

    It is deliberately not 0.5. A probability's own midpoint looks like the natural
    threshold and is a different operating point entirely -- at 0.5 this gate prints
    a reason on nearly every row, including the on-topic line that is true of the
    whole page, which is the failure the gate exists to prevent. Every figure quoted
    for a calibrator in this package was measured at a bisected threshold, so a
    deployment that moves this is choosing a point none of those numbers describe.

    Ignored when no `calibrator` is set."""

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Settings:
        """Read one JSON file, registering any language packs it carries."""
        document = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise TypeError("%s: expected an object at the top level" % path)
        return cls.from_mapping(document, source=str(path))

    @classmethod
    def from_mapping(cls, document: Mapping[str, object], source: str = "settings") -> Settings:
        fields = {field.name for field in dataclasses.fields(cls)}
        unknown = set(document) - fields - {"languages"}
        if unknown:
            raise ValueError(
                "%s: unknown setting(s) %s; known are %s"
                % (source, ", ".join(sorted(unknown)), ", ".join(sorted(fields)))
            )
        for code, pack in dict(document.get("languages") or {}).items():
            register_phrases(code, phrases_from_mapping(code, pack))
        return cls(**{name: document[name] for name in fields if name in document})

    @classmethod
    def from_env(cls) -> Settings:
        """The deployment's settings, or the built-in defaults if none were given."""
        path = os.environ.get(ENV_VAR)
        return cls.load(path) if path else cls()


def phrases_from_mapping(code: str, pack: object) -> RankPhrases:
    """One language pack out of plain data, with every field required.

    No field defaults to English: a half-written pack would otherwise produce a
    sentence in two languages, which is worse than refusing to produce one.
    """
    if not isinstance(pack, Mapping):
        raise TypeError(
            "language %r: expected an object, got %s" % (code, type(pack).__name__)
        )
    wanted = [field.name for field in dataclasses.fields(RankPhrases)]
    missing = [name for name in wanted if name not in pack]
    if missing:
        raise ValueError("language %r is missing %s" % (code, ", ".join(missing)))
    extra = set(pack) - set(wanted)
    if extra:
        raise ValueError(
            "language %r has unknown field(s) %s" % (code, ", ".join(sorted(extra)))
        )
    return RankPhrases(**{name: str(pack[name]) for name in wanted})
