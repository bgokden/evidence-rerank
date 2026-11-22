"""An HTTP service that ranks and explains, with the model loaded once.

There is one route. `POST /api/rank` always ranks, which is one batched inference
over the candidate list, and explains only when the request asks it to. The response
reports what each stage cost in pairs scored and milliseconds, so the price of an
explanation is visible rather than asserted, and the explanation itself is `null`
when it was not requested.

The model is multilingual by default, because evidence and candidates are the user's
own text and need not share a language. Nothing in the attribution is
language-specific: the context is rendered, masked and re-scored the same way
whatever it says.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Callable, Sequence

import pydantic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from evrank import calibration
from evrank.agreement import compare_maskers
from evrank.attribution import (
    MAX_EXACT_SHAPLEY_UNITS,
    SAMPLED_SHAPLEY,
    flip_pair,
    flip_top,
    leave_one_out,
    sampled_shapley,
    shapley,
)
from evrank.config import Settings
from evrank.cross_encoder import MULTILINGUAL_MODEL, CrossEncoderScorer
from evrank.evidence import DEFAULT_HEADER, Evidence, EvidenceSet
from evrank.explain import PHRASES, explain_shown
from evrank.limits import ContextTooLongError
from evrank.masking import DEFAULT_FILLER_TEXT, KindOnly, Masker, NeutralFiller, Omission
from evrank.ranking import Candidate
from evrank.scoring import Ablator, CachingScorer, Scorer, rank
from evrank.standout import MAX_EXACT_UNITS, exact_standout, sampled_standout, standout

WEB_ROOT = pathlib.Path(__file__).parent / "web"


class EvidenceIn(pydantic.BaseModel):
    id: str
    kind: str
    text: str
    group: str | None = None


class CandidateIn(pydantic.BaseModel):
    id: str
    name: str
    text: str


class RankRequest(pydantic.BaseModel):
    """Rank, and explain only if asked.

    `explain` is the switch: false returns the ranking alone. `compare_maskers`
    repeats the attribution under every masking strategy and reports where they
    agree, which costs roughly three explanations.
    """

    evidence: list[EvidenceIn]
    candidates: list[CandidateIn]
    header: str = DEFAULT_HEADER
    explain: bool = False
    compare_maskers: bool = False
    masker: str = "omission"
    filler_text: str = DEFAULT_FILLER_TEXT
    estimator: str | None = None
    """`shapley`, `sampled` or `loo`, defaulting to `loo`. That default is about the
    question rather than about quality: `loo` removes one unit from the real context
    and reports the difference between two scores the model actually produced, which
    is what "because you searched for X" asserts, while the Shapley paths average
    over arrival orders the user never had. Use those where duplicate evidence would
    otherwise cancel itself out and cannot be grouped. Each row's `method` reports
    which ran; `shapley` enumerates up to `standout.MAX_EXACT_UNITS` units and samples
    above that rather than failing."""
    permutations: int = 64
    seed: int = 0
    pair: tuple[str, str] | None = None
    max_flip_size: int = 2
    sentences: bool = False
    """Adds a one-line `because` to each candidate, built from the same reasons."""
    language: str | None = None
    """Which wording the sentences use. Unset takes the deployment's own default.
    `GET /api/model` lists every registered pack; `evrank.config` adds one."""
    top_reasons: int | None = None
    """How many reasons to carry per candidate, strongest first. 0 for all of them.
    Unset means 2, which is the shape of the sentence: "Because you searched for X
    and viewed Y". More than that stops reading as a reason and starts reading as a
    list."""
    min_standout: float | None = None
    """The floor under a reason's *centred* effect: how much of what the evidence did
    was specific to this candidate, in the model's own score units.

    This is the floor to choose from, and the one to reach for when a surface is
    printing too much. Measured on held-out cases at matched false positives, moving
    the floor here rather than onto `min_effect` is worth 8 to 12 points of recall
    and 7 to 16 points of precision, for both the default model and a fine-tuned one.
    `min_effect` stays as a small backstop; see `standout.favouring` for which case
    each one catches and why neither subsumes the other. 0 keeps everything."""
    min_specificity: float | None = None
    """The floor under a reason's share of its own effect. A unit
    that lifts this candidate by +0.40 while lifting the whole slate by +0.396 is
    doing nothing for this row, and citing it is true and useless. Being a ratio, the
    floor carries across models where an absolute size would not. 0 keeps everything."""
    min_effect: float | None = None
    """The floor under a reason's raw effect on this candidate.

    Where `min_specificity` asks whether the effect was specific to this row, this
    asks whether there was an effect at all. Without it a reason gets invented out of
    noise: on a page the model drew no distinction on, one candidate still has the
    largest tiny effect, and a share of almost nothing is still a share.

    It ships at 0.1 and is a backstop, not the dial to turn. That is a change from
    1.5, and the reason is `min_standout`: when the floor that carries the decision
    moved onto the centred effect, this one went back to being small. The sweep table
    that used to sit here, and the argument for 1.5 built on it, described this field
    while it still carried the decision, and both moved to `min_standout` with the
    decision. A reader who inherited 1.5 from here would now be flooring the raw
    effect at fifteen times the shipped value on top of a `min_standout` that is
    already gating, and printing almost nothing.

    The number belongs to the model it was swept on: on the model that shipped before
    mixedbread-ai/mxbai-rerank-base-v2 the same operating point was a factor of
    fifteen away, so a floor carried from one model to another is a silent
    misconfiguration, which is why `Settings` pairs the two.

    Expressing the floor in multiples of the request's own scale has now been tried
    three times and lost three times, including with a free normaliser that predicts
    a request's largest effect with an R-squared of 0.71; see `standout.favouring`
    for why that does not help. This is an uncalibrated logit and so both model- and
    corpus-dependent -- re-run the sweep rather than inheriting this number, and
    read its separation check before trusting any floor it suggests. 0 keeps
    everything."""
    min_share: float | None = None
    """The dynamic floor, as a share of the largest effect on this
    candidate in either direction. Off by default and never shown to pay for itself.
    The numbers that used to be quoted here -- good reasons from 18 to 100 percent of
    their row, wrong ones from 66 to 100 -- came from a corpus whose planted evidence
    was the candidate's own words copied into the history, which a character-trigram
    matcher solved outright, so they say nothing about this filter. Sweep it on your
    own corpus before raising it."""
    top_only: bool | None = None
    """Attach the reasons and the sentence to the first choice alone.

    Usually the first choice is the only one being explained, and the rest of the
    page is noise in the response. Every other row keeps its rank, name and score.

    It does not make the request cheaper. The effects are centred across the slate,
    so every candidate has to be scored under every ablation whatever is reported --
    narrowing the measurement to one candidate is refused rather than approximated,
    because a reason that was not compared against anything is not a reason."""
    include_against: bool = False
    """Adds an `against` list of the evidence that held the candidate back. `reasons`
    is what was in its favour; for a candidate that is not first, the two answer
    different questions and the `against` list is the one explaining its position."""
    detail: bool = False
    """Adds the numbers behind the sentences: the pair, every effect, and the
    counterfactuals. The sentences alone answer "why this one"; the detail is for
    checking them."""


class ScorerProvider:
    """Holds one model for the process, loaded on first use rather than on import."""

    def __init__(self, factory: Callable[[], Scorer], gate: str = "floors") -> None:
        self._factory = factory
        self._scorer: Scorer | None = None
        self.gate = gate

    def get(self) -> Scorer:
        if self._scorer is None:
            self._scorer = self._factory()
        return self._scorer

    def describe(self) -> dict[str, object]:
        scorer = self.get()
        return {
            "name": getattr(scorer, "model_name", type(scorer).__name__),
            # Reported because the same model name serves two different behaviours.
            # A reader comparing two deployments' explanations needs to know which
            # of them is running an adapter.
            "adapter": getattr(scorer, "adapter", None),
            # Which gate produced the reasons on the page. "floors" and a calibrator
            # answer the same question differently, and a reader comparing two
            # deployments needs to know which one they are reading.
            "gate": self.gate,
            "maxSeqLength": getattr(scorer, "max_seq_length", None),
            "hiddenSize": getattr(scorer, "hidden_size", None),
            "maxExactShapleyUnits": MAX_EXACT_SHAPLEY_UNITS,
            # The ranker reads far more languages than these; a sentence needs
            # wording, and only these have been written.
            "sentenceLanguages": sorted(PHRASES),
        }


def build_evidence(items: Sequence[EvidenceIn], header: str) -> EvidenceSet:
    if not items:
        raise HTTPException(status_code=400, detail="at least one evidence item is required")
    try:
        return EvidenceSet(
            [
                Evidence(id=item.id, kind=item.kind, text=item.text, group=item.group)
                for item in items
            ],
            header=header,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def build_candidates(items: Sequence[CandidateIn]) -> list[Candidate]:
    if len(items) < 2:
        raise HTTPException(status_code=400, detail="at least two candidates are required")
    ids = [item.id for item in items]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=400, detail="candidate ids must be unique")
    return [Candidate(id=item.id, name=item.name, text=item.text) for item in items]


def pick_masker(name: str, filler_text: str) -> Masker:
    if name == "omission":
        return Omission()
    if name == "kind":
        return KindOnly()
    if name == "filler":
        return NeutralFiller(text=filler_text)
    raise HTTPException(status_code=400, detail="unknown masker: %s" % name)


class Stages:
    """Times each stage and attributes the pairs it sent to the model."""

    def __init__(self, scorer: CachingScorer) -> None:
        self._scorer = scorer
        self._started = time.perf_counter()
        self.records: list[dict[str, object]] = []

    def run(self, name: str, call: Callable[[], object]):
        pairs, batches = self._scorer.model_pairs, self._scorer.batches
        began = time.perf_counter()
        result = _guarded(call)
        self.records.append(
            {
                "name": name,
                "ms": round((time.perf_counter() - began) * 1000, 1),
                "modelPairs": self._scorer.model_pairs - pairs,
                "batches": self._scorer.batches - batches,
            }
        )
        return result

    def report(self) -> dict[str, object]:
        return {
            "totalMs": round((time.perf_counter() - self._started) * 1000, 1),
            "modelPairs": self._scorer.model_pairs,
            "pairsServedFromCache": self._scorer.requested_pairs - self._scorer.model_pairs,
            "stages": self.records,
        }


def create_app(
    provider: ScorerProvider | None = None, settings: Settings | None = None
) -> FastAPI:
    """One route, one model, and the deployment's defaults behind every request.

    `settings` is read from the file named by `EVRANK_CONFIG` when it is not passed,
    which is also what registers any language packs that file carries.
    """
    settings = settings or Settings.from_env()
    # Loaded once at startup, not per request: it is a small JSON file, and a bad
    # path should stop a deployment coming up rather than surface as a failed
    # explanation on somebody's page an hour later.
    loaded_calibrator = (
        calibration.Calibrator.load(settings.calibrator) if settings.calibrator else None
    )
    provider = provider or ScorerProvider(
        lambda: CrossEncoderScorer(
            settings.model or MULTILINGUAL_MODEL, adapter=settings.adapter
        ),
        gate="calibrator" if loaded_calibrator is not None else "floors",
    )
    app = FastAPI(title="Evidence Ablation Bench")

    @app.get("/")
    def index() -> FileResponse:
        # Revalidate every load. FileResponse sends an etag and a last-modified, but
        # without a Cache-Control a browser is free to pick its own freshness window
        # and serve a stale copy without asking, which on a page that changes with
        # every edit means reading an explanation the running code no longer gives.
        # "no-cache" means check, not do not store: the etag still makes it a 304.
        return FileResponse(
            WEB_ROOT / "index.html", headers={"cache-control": "no-cache"}
        )

    @app.get("/api/model")
    def model() -> dict[str, object]:
        return {
            **provider.describe(),
            # What a request gets when it asks for nothing, so a caller can read the
            # installation's choices instead of inferring them from the output.
            "defaults": dataclasses.asdict(settings),
        }

    @app.get("/api/version")
    def version() -> dict[str, object]:
        """When the page on disk last changed, for an open tab to notice.

        `cache-control: no-cache` makes a reload fetch the current page, but a tab
        that is already open never asks, so an edit here is invisible to whoever is
        looking at it. This is cheap enough to poll and touches nothing else, which
        is why it is its own route rather than a field on the model description.
        """
        return {"page": WEB_ROOT.joinpath("index.html").stat().st_mtime_ns}

    @app.post("/api/rank")
    def rank_endpoint(request: RankRequest) -> dict[str, object]:
        evidence = build_evidence(request.evidence, request.header)
        candidates = build_candidates(request.candidates)
        masker = pick_masker(request.masker, request.filler_text)
        scorer = CachingScorer(provider.get())
        stages = Stages(scorer)

        ranking = stages.run("ranking", lambda: rank(scorer, evidence, candidates))
        response: dict[str, object] = {
            "ranking": _ranking_rows(ranking),
            "context": evidence.render(),
            "explanation": None,
            "agreement": None,
        }
        if not request.explain:
            response["timing"] = stages.report()
            return response

        order = ranking.order()
        pair = tuple(request.pair) if request.pair else (order[0], order[1])
        for cid in pair:
            if cid not in ranking.scores:
                raise HTTPException(status_code=400, detail="unknown candidate: %s" % cid)
        if pair[0] == pair[1]:
            raise HTTPException(status_code=400, detail="the pair must name two candidates")
        estimate = _estimator(
            request.estimator or "shapley", request.permutations, request.seed
        )
        size = max(1, min(request.max_flip_size, len(evidence.units())))

        ablator = Ablator(scorer, evidence, candidates, masker=masker)
        # Anything the request left unset is the deployment's choice, not a constant
        # written here: see `evrank.config`.
        language = request.language or settings.language
        if language not in PHRASES:
            raise HTTPException(
                status_code=400,
                detail="unknown language: %s (have %s)"
                % (language, ", ".join(sorted(PHRASES))),
            )
        # Before the masker-agreement branch, which used to return without these and
        # so answered a request for `sentences` with none, saying nothing about it.
        reasons = stages.run(
            "reasons",
            lambda: _standout_reasons(
                ablator,
                estimator=request.estimator or "loo",
                permutations=request.permutations,
                seed=request.seed,
                limit=settings.top_reasons if request.top_reasons is None
                else request.top_reasons,
                min_specificity=settings.min_specificity
                if request.min_specificity is None else request.min_specificity,
                min_effect=settings.min_effect
                if request.min_effect is None else request.min_effect,
                min_standout=settings.min_standout
                if request.min_standout is None else request.min_standout,
                min_share=settings.min_share
                if request.min_share is None else request.min_share,
                only=order[0] if (
                    settings.top_only if request.top_only is None else request.top_only
                ) else None,
                include_against=request.include_against,
                sentences=request.sentences,
                phrases=PHRASES[language],
                calibrator=loaded_calibrator,
                min_confidence=settings.min_confidence,
            ),
        )

        if request.compare_maskers:
            maskers = (
                ("omission", Omission()),
                ("kind", KindOnly()),
                ("filler", NeutralFiller(text=request.filler_text)),
            )
            report = stages.run(
                "agreement",
                lambda: compare_maskers(
                    scorer,
                    evidence,
                    candidates,
                    pair[0],
                    pair[1],
                    maskers=maskers,
                    estimate=estimate,
                    max_flip_size=size,
                ),
            )
            response["agreement"] = _agreement_rows(report, ranking)
            response["ranking"] = _ranking_rows(ranking, reasons)
            response["timing"] = stages.report()
            return response

        response["ranking"] = _ranking_rows(ranking, reasons)

        if request.detail:
            attribution = stages.run(
                "attribution", lambda: estimate(ablator, pair[0], pair[1])
            )
            pair_flip = stages.run(
                "counterfactual: pair",
                lambda: flip_pair(ablator, pair[0], pair[1], max_size=size),
            )
            top_flip = stages.run(
                "counterfactual: first place", lambda: flip_top(ablator, max_size=size)
            )
            response["explanation"] = _explanation_rows(
                attribution, ranking, evidence, masker, pair_flip, top_flip
            )
        response["timing"] = stages.report()
        return response

    return app


def _estimator(name: str, permutations: int, seed: int):
    if name == "shapley":
        return shapley
    if name == "sampled":
        return lambda ablator, a, b: sampled_shapley(
            ablator, a, b, permutations=permutations, seed=seed
        )
    if name == "loo":
        return leave_one_out
    raise HTTPException(status_code=400, detail="unknown estimator: %s" % name)


def _explanation_rows(attribution, ranking, evidence, masker, pair_flip, top_flip):
    """One way to say each thing. Names come from `ranking`, keyed by the same id,
    and anything derivable from another field is left out."""
    a_name = ranking.name(attribution.a)
    b_name = ranking.name(attribution.b)
    explanation: dict[str, object] = {
        "pair": {
            "a": attribution.a,
            "b": attribution.b,
            "method": attribution.method,
            "marginAOverB": attribution.base_margin,
            "marginWithNoEvidence": attribution.empty_margin,
        },
        "effects": [
            _effect_row(effect, evidence, a_name, b_name, attribution.method == SAMPLED_SHAPLEY)
            for effect in attribution.by_effect()
        ],
        "counterfactual": _flip_row(pair_flip, _pair_winner(attribution, ranking)),
        "firstPlaceCounterfactual": _first_place_row(top_flip, pair_flip, ranking),
    }
    explanation["maskedBaseline"] = masker.context(evidence, evidence.units())
    return explanation


def _standout_reasons(
    ablator: Ablator,
    estimator: str,
    permutations: int,
    seed: int,
    limit: int,
    min_specificity: float,
    min_effect: float,
    min_share: float,
    min_standout: float,
    only: str | None,
    include_against: bool,
    sentences: bool,
    phrases,
    calibrator: calibration.Calibrator | None = None,
    min_confidence: float = 0.5,
) -> dict[str, dict[str, object]]:
    """Why each candidate is on the page at all, for a reader rather than an auditor.

    The measurement is the candidate's own score with each unit removed, minus what
    the same unit did for the rest of the slate, because "Because you searched for
    ramen" claims the evidence is why this item is here. It is deliberately not the
    margin over the row below: that is what a rank is made of, but "ranked 2 because
    X outweighs Y" is a sentence about the attribution rather than about the person
    reading it. The margin attribution is still here, under `detail` and `agreement`,
    where it is read by someone checking the method.

    Only evidence that favoured the candidate is cited, so the sentence never
    explains an item by what argued against it. `only` reports one candidate and
    measures all of them, because the effects are centred across the slate.

    `effect` is the part specific to this candidate and `onItsOwnScore` the raw
    effect, so the two numbers show what centring did.
    """
    if estimator == "loo":
        rows = standout(ablator)
    elif estimator == "sampled" or len(ablator.evidence.units()) > MAX_EXACT_UNITS:
        rows = sampled_standout(ablator, permutations=permutations, seed=seed)
    else:
        rows = exact_standout(ablator)
    # A calibrator judges the slate, not a row: two of its strongest inputs are the
    # width of the page and the length of the context, so it cannot be asked about
    # one candidate in isolation. Computed once here for every row, then read per
    # row below, which is also why `only` still measures the whole slate.
    chosen = (
        calibration.select(rows, calibrator, min_confidence, limit or 0)
        if calibrator is not None
        else None
    )
    reasons: dict[str, dict[str, object]] = {}
    for cid, result in rows.items():
        if only is not None and cid != only:
            continue
        favouring = (
            chosen[cid]
            if chosen is not None
            else result.favouring(
                min_specificity=min_specificity,
                min_effect=min_effect,
                min_share=min_share,
                min_standout=min_standout,
            )
        )
        against = tuple(
            effect for effect in reversed(result.by_standout()) if effect.standout < 0.0
        )
        cited = _capped(favouring, limit)
        entry: dict[str, object] = {
            "score": result.score,
            "method": result.method,
            # The request's own median effect. An effect of +0.3 means one thing
            # where this is 0.05 and another where it is 0.5, and nothing in a raw
            # logit says which request you are looking at.
            "scale": result.scale,
            "reasons": [_standout_row(ablator, e) for e in cited],
        }
        if include_against:
            entry["against"] = [_standout_row(ablator, e) for e in _capped(against, limit)]
        if sentences:
            entry["because"] = explain_shown(
                [ablator.evidence.phrase(effect.unit) for effect in cited],
                limit=max(limit, 1),
                phrases=phrases,
            )
        reasons[cid] = entry
    return reasons


def _capped(effects, limit: int):
    return effects if limit <= 0 else effects[:limit]


def _standout_row(ablator: Ablator, effect) -> dict[str, object]:
    row: dict[str, object] = {
        "unit": effect.unit,
        "evidence": effect.label,
        "effect": effect.standout,
        "onItsOwnScore": effect.delta,
        "specificity": effect.specificity,
    }
    count = ablator.evidence.count(effect.unit)
    if count > 1:
        row["facts"] = count
    return row


def _effect_row(
    effect, evidence: EvidenceSet, a_name: str, b_name: str, sampled: bool
) -> dict[str, object]:
    row: dict[str, object] = {
        "unit": effect.unit,
        "evidence": evidence.label(effect.unit),
        "effectOnMargin": effect.delta,
        "favours": a_name if effect.delta > 0 else b_name if effect.delta < 0 else None,
    }
    if effect.spread is not None:
        across: dict[str, object] = {
            "lowest": effect.spread.low,
            "highest": effect.spread.high,
        }
        # Present exactly when the value is an estimate, so its absence is not
        # confused with a sample that happened to have no variance.
        if sampled:
            across["samplingError"] = effect.spread.stderr
        row["acrossContexts"] = across
    return row


def _pair_winner(attribution, ranking) -> tuple[str, str]:
    """Who leads the pair once it is reversed, and who they displace."""
    winner = attribution.b if attribution.base_margin >= 0 else attribution.a
    loser = attribution.a if winner == attribution.b else attribution.b
    return ranking.name(winner), ranking.name(loser)


def _flip_row(flip, names: tuple[str, str]) -> dict[str, object] | None:
    if flip is None:
        return None
    return {
        "remove": list(flip.removed),
        "evidence": list(flip.labels),
        "thenFirst": names[0],
        "insteadOf": names[1],
        "byMargin": abs(flip.margin_after),
    }


def _first_place_row(top_flip, pair_flip, ranking) -> dict[str, object] | None:
    """Null when it is the same answer as the pair counterfactual, which it is
    whenever the explained pair is the top two."""
    if top_flip is None:
        return None
    if pair_flip is not None and tuple(top_flip.removed) == tuple(pair_flip.removed):
        return None
    return _flip_row(
        top_flip, (ranking.name(top_flip.new_top), ranking.name(top_flip.base_top))
    )


def _agreement_rows(report, ranking) -> dict[str, object]:
    a_name = ranking.name(report.a)
    b_name = ranking.name(report.b)
    return {
        "pair": {
            "a": report.a,
            "aName": a_name,
            "b": report.b,
            "bName": b_name,
            "method": report.method,
            "marginAOverB": report.base_margin,
        },
        "maskers": list(report.maskers),
        "unitsHoldingDirection": len([u for u in report.units if u.agrees]),
        "unitsTotal": len(report.units),
        "units": [
            {
                "unit": unit.unit,
                "evidence": unit.label,
                "effectByMasker": dict(unit.deltas),
                "lowest": unit.low,
                "highest": unit.high,
                "verdict": (
                    "disputed"
                    if unit.direction == "mixed"
                    else "no effect"
                    if unit.direction == "none"
                    else "favours %s" % (a_name if unit.direction == "a" else b_name)
                ),
            }
            for unit in report.agreed() + report.disputed()
        ],
        "counterfactualByMasker": {
            name: (
                None
                if flip.removed is None
                else {"remove": list(flip.removed), "byMargin": abs(flip.margin_after)}
            )
            for name, flip in report.flips.items()
        },
        "counterfactualIsTheSameUnderAll": report.flip_agrees,
    }


def _guarded(call: Callable[[], object]):
    """Turn the library's refusals into honest HTTP errors rather than 500s."""
    try:
        return call()
    except ContextTooLongError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except HTTPException:
        raise
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _ranking_rows(ranking, reasons=None) -> list[dict[str, object]]:
    """The answer, with the reasons attached to the candidate they are about."""
    rows = []
    for position, cid in enumerate(ranking.order(), start=1):
        row: dict[str, object] = {
            "rank": position,
            "id": cid,
            "name": ranking.name(cid),
            "score": ranking.scores[cid],
        }
        if reasons and cid in reasons:
            row.update(reasons[cid])
        rows.append(row)
    return rows


app = create_app()
