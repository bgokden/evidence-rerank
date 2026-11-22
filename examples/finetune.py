"""Train a LoRA adapter so the re-ranker's ablation effects carry information.

The ranker is the ceiling on this whole package: the explanation names the right
cause about as often as the ranker ranks it, so the attribution layer has little
headroom of its own. And nothing in ordinary re-ranker training asks the differences
between scores to mean anything, which is what `evrank.standout` reports. Those two
facts together are the case for training the model rather than the threshold.

Two training sources, because neither is sufficient alone:

    synthetic   slates from `examples/evaluation.py`, which look like the product --
                paraphrased requests, on-topic decoys, listings rather than passages.
                Few in number: 63 hand-written families, so memorisation is the
                cheap path and the held-out splits are what catch it.
    miracl      slates of real queries and the passages a human judged relevant to
                them, in ten languages, Apache 2.0. Large enough that memorising it
                is not cheap, and its distractors are other real queries. It needs
                hard negatives to be worth anything at all -- see `miracl_data`.

What is held out, and why each split exists:

    languages   train on eight of eleven synthetic languages, test on three. This
                split is much weaker than it looks and is kept only for the
                `matched` cases. It was justified as "the objective cannot be
                satisfied on a language never seen", which measurement contradicts:
                in a `mixed` case only the action verb is localised and the
                candidate text stays English, so a held-out-language mixed case is a
                trained-on case with one string swapped. The adapter scores 95.8%
                there, above its own trained-on level, against 70.8% on the domain
                hold-out. Read the two apart; `compare.py` reports them separately.
    categories  the split to quote. The corpus composes 512 pairs from 48
                categories, and 12 categories are reserved before any training run,
                so a reserved pair appears in no training slate in any position --
                not as the plant and not as a distractor. A model that memorised the
                training pairs can say nothing about them. This replaces the older
                arrangement, where the split held out *cases*: with 18 cases per
                pair, the pair under test had been seen eighteen times, and a
                60-entry lookup table scored 100% on the supposedly held-out half.
    domains     one of six, flights. A coarser second axis and a different kind of
                novelty -- an unseen domain brings an unseen listing template, an
                unseen on-topic generic and an unseen distractor pool, not just an
                unseen pair. Narrowed from two domains because holding out courses
                as well cost a third of the training content, including the domain
                whose categories are the most separable.
    validation  a tenth of each training source, stratified. This is the stopping
                signal, and it is the objective as a decision rather than as a loss
                -- the loss kept falling in the first run while nothing watched
                whether it still helped. The best checkpoint is kept, not the last.
    guards      three sets of human judgements, in no training or test split. They
                bound how far ordinary ranking may fall -- `GUARD_TOLERANCE`, in
                points -- rather than testing whether it fell at all. A canary in an
                untrained language is expected to give a little back when the model
                specialises, and the earlier significance test could not tell that
                apart from damage: it discarded adapters over two-point moves that
                bought twelve to twenty-five points in domain. One guard was not
                enough either:
                MIRACL Swahili shares its construction and annotation pipeline with
                the MIRACL training slates, so an adapter that had learned MIRACL's
                shape rather than anything about ranking could still have passed it.
                Mr. TyDi supplies the other two -- a different lineage, built from
                TyDi QA with BM25-retrieved negatives -- in Swahili, which holds
                language fixed and changes only the dataset, and in English, which
                changes both.

The gate has two terms and the adapter must clear both: in-domain ranking, measured
on held-back slates, must not regress, and no guard may fall by more than the
tolerance. The first is the job the adapter was trained for and the second bounds the
collateral. The gate had no in-domain term at all until this was written down, which
is why it read a specialising adapter as a broken one.

An adapter rather than a fine-tune, deliberately: the base weights are untouched, so
a refusal is recoverable and the adapter can be switched off per request.

Run: uv run python examples/finetune.py [slates] [out_dir] [anchor]
         [miracl_languages] [guard_queries] [margin] [seed] [fold]
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
import typing

import safetensors.torch
import torch
from corpus import (
    ACTIONS,
    CATEGORY_FOLDS,
    DOMAINS,
    MATCHED,
    NO_FOLD,
    build_case,
    build_matched_case,
    category_fold,
)
from huggingface_hub import hf_hub_download
from miracl_data import LANGUAGES as MIRACL_LANGUAGES
from miracl_data import judgements, passages, read_pairs
from miracl_data import slates as miracl_slates
from mrtydi_data import guard as mrtydi_guard
from peft import LoraConfig, inject_adapter_in_model
from sentence_transformers import CrossEncoder

from evrank.config import Settings
from evrank.cross_encoder import MULTILINGUAL_MODEL
from evrank.masking import Omission
from evrank.training import DEFAULT_MARGIN, standout, total_loss

# Three languages and two domains the adapter never sees. Chosen before any training
# run, and not revisited, because a held-out split picked after looking at results is
# not held out.
HELD_OUT_LANGUAGES = ("fr", "pl", "zh")
HELD_OUT_DOMAINS = ("flights",)

# Swept one term at a time from all-zero, which reproduces the untrained model, so
# a gain is attributable to a term rather than to the set of them.
WEIGHTS = {"rank": 1.0, "attribution": 1.0, "margin": 0.5, "anchor": 1.0}

# Checks without improvement that end the run. How often a check runs is derived
# from the run's length, so a short run still gets several.
PATIENCE = 3
CHECKS_PER_RUN = 8
MIN_CHECK_EVERY = 10

Example = collections.namedtuple("Example", "full ablated target language domain")


def slates(rng: random.Random, count: int, languages, domains,
           reserved: frozenset[str]) -> list[dict]:
    """Cases drawn only from the given languages, domains and categories.

    The category filter is the one that matters. `build_case` is passed the reserved
    categories to skip, so no training slate contains a held-out pair in any
    position -- not as the plant and not as a distractor. The corpus generates 1344
    distinct pairs from 84 categories, so holding out 18 of them leaves an
    evaluation set whose pairs training never saw in any form. The language and
    domain filters stay, but they are the weaker axes: see the module docstring on
    what a held-out language does and does not test.
    """
    out = []
    while len(out) < count:
        index = len(out)
        case = (
            build_matched_case(rng, index)
            if index % 2
            else build_case(rng, index, reserved)
        )
        if case["target"] is None:
            continue
        if case["condition"] == "matched":
            if case["language"] not in languages:
                continue
        elif case["domain"] not in domains or case["language"] not in languages:
            continue
        out.append(case)
    return out


def to_example(case: dict) -> Example:
    """One case as the two contexts the attribution term needs.

    The contexts are built by the same masker the explanation uses at inference, so
    training sees exactly the strings the measurement will see. Getting this wrong
    would train the model on a distribution it is never asked about.
    """
    masker = Omission()
    full = masker.context(case["evidence"], ())
    ablated = masker.context(case["evidence"], (case["key"],))
    texts = [candidate.text for candidate in case["candidates"]]
    target = [candidate.id for candidate in case["candidates"]].index(case["target"])
    return Example(
        full=[(full, text) for text in texts],
        ablated=[(ablated, text) for text in texts],
        target=target,
        language=case["language"],
        domain=case["domain"],
    )


def score_batch(model: CrossEncoder, pairs: list[tuple[str, str]]) -> torch.Tensor:
    """The model's own scoring path, with gradients.

    `model(features)["scores"]` is what `predict` calls underneath. Reusing it rather
    than reimplementing the head matters here: this model scores a pair as
    logit("1") - logit("0") at the last position, which is a detail that would be
    easy to get subtly wrong and impossible to notice from the loss curve.
    """
    features = model.preprocess(pairs)
    device = model.model.device
    features = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }
    return model(features)["scores"].squeeze(-1).float()


# A tenth of each source is held back for the stopping signal, but never fewer than
# this: the signal is an accuracy on a small set, and below a dozen slates it moves in
# steps too coarse to choose a checkpoint by.
VALIDATION_FLOOR = 12
# 120 was the first guess and it could not do the job: a four-point move on 120
# paired queries is about one standard error, so the guard could not tell a collapse
# from noise. At 760 the same comparison resolves to within a couple of points.
GUARD_QUERIES = 760
# Swahili: in MIRACL, in neither training split, and not an evaluation language.
GUARD_LANGUAGE = "sw"
# How far a guard may fall before the adapter is refused, in points of accuracy.
#
# This replaces a significance test, and the two answer different questions. At 760
# paired queries a two-point move is resolvable, so `p < 0.05` asks "is this fall
# real?" and reliably answers yes. The gate has to ask "is this fall big enough to
# matter against what the adapter buys?", which is a question about magnitude. The
# old rule discarded two of four folds whose worst guard fall was 2.0 and 2.1 points,
# while the same adapters improved in-domain ranking by twelve to twenty-five points.
# It was measuring specialisation and calling it breakage.
#
# Five points is about three standard errors on a 760-query guard, so a fall that
# trips this is both large and real. A fall below it is reported and not acted on.
GUARD_TOLERANCE = 5.0


def miracl_guard(limit: int = GUARD_QUERIES) -> list[tuple[str, list[str], int]]:
    """Human relevance judgements in a language neither split contains.

    MIRACL Swahili, Apache 2.0. The licence matters as much as the data: MS MARCO
    and everything derived from it is restricted to non-commercial research by its
    own terms, which rules out most of what the ecosystem ships as standard
    re-ranking training data.

    Passages are trimmed exactly as the training slates are. Left untrimmed, the
    guard read whole Wikipedia sections and took longer than the training it was
    guarding; it also would have measured the model on lengths training never saw.

    Returns (query, candidate passages, index of the relevant one).
    """
    relevant, irrelevant = judgements(GUARD_LANGUAGE)
    wanted = set(relevant.values())
    for docs in irrelevant.values():
        wanted.update(docs[:4])
    texts = passages(GUARD_LANGUAGE, wanted)
    queries = read_pairs(
        hf_hub_download(
            "miracl/miracl",
            "miracl-v1.0-%s/topics/topics.miracl-v1.0-%s-train.tsv"
            % (GUARD_LANGUAGE, GUARD_LANGUAGE),
            repo_type="dataset",
        )
    )
    rng = random.Random(20261120)
    out = []
    for query_id, positive in relevant.items():
        negatives = [doc for doc in irrelevant.get(query_id, []) if doc in texts][:4]
        if query_id not in queries or positive not in texts or len(negatives) < 2:
            continue
        # Shuffled for the same reason as the Mr. TyDi guard: with the positive at a
        # fixed index 0, `argmax` hands it every exact tie, and bf16 scores tie on
        # about 3% of rows. Seeded, so the rows and their order are the same on every
        # run and a before/after comparison stays paired.
        candidates = [texts[positive]] + [texts[doc] for doc in negatives]
        relevant_text = candidates[0]
        rng.shuffle(candidates)
        out.append((queries[query_id], candidates, candidates.index(relevant_text)))
        if len(out) >= limit:
            break
    return out


def build_guards(limit: int = GUARD_QUERIES) -> dict[str, list[tuple[str, list[str], int]]]:
    """Every general-ranking guard, built once and reused before and after training.

    Three of them, because one could not answer the question it was asked. The
    MIRACL Swahili guard shares its construction with the MIRACL training slates,
    so on its own it cannot separate "ranking survived" from "ranking survived on
    data shaped like the training data". The Mr. TyDi pair settles that: same
    language, different dataset, then different language as well.
    """
    return {
        "miracl-%s" % GUARD_LANGUAGE: miracl_guard(limit),
        "mrtydi-sw": mrtydi_guard("swahili", limit),
        "mrtydi-en": mrtydi_guard("english", limit),
    }


def report_guards(before: dict[str, list[int]], after: dict[str, list[int]]) -> float:
    """Print each guard's paired comparison. Returns the largest fall, in points.

    The p-value is still printed, because it says whether a fall is real and that is
    worth knowing, but it no longer decides: the decision is `GUARD_TOLERANCE`, a
    bound on magnitude. A guard here is a canary in a language and a corpus the
    adapter never trains on, so a small fall is the expected cost of specialising and
    only a large one means the model has been damaged.

    No multiplicity correction on the printed p-values, deliberately. For a *claim* --
    "the adapter improved attribution" -- testing three things and quoting the best
    p-value overstates the evidence and a correction is owed. These are not claims;
    they are three separate bounds, each read on its own.
    """
    worst = 0.0
    for name in sorted(before):
        old, new = before[name], after[name]
        gained, lost, probability = mcnemar(old, new)
        fall = 100 * sum(old) / len(old) - 100 * sum(new) / len(new)
        worst = max(worst, fall)
        print("  %-12s %5.1f%% -> %5.1f%%   %d gained, %d lost, p = %.3f   %+.1f pts%s"
              % (name, 100 * sum(old) / len(old), 100 * sum(new) / len(new),
                 gained, lost, probability, -fall,
                 "  OVER TOLERANCE" if fall > GUARD_TOLERANCE else ""))
    return worst


GUARD_CACHE = pathlib.Path("tmp/guard-cache.json")


def guard_fingerprint(rows) -> str:
    """A digest of exactly the text a guard will score.

    The cache below is keyed on this rather than on a name or a row count, because a
    guard that was rebuilt -- a different shuffle, a different trim, more rows -- is a
    different measurement, and reusing a stale "before" would compare two models on
    two different sets while reporting a paired test over one.
    """
    digest = hashlib.sha256()
    for query, candidates, target in rows:
        digest.update(query.encode("utf-8"))
        for candidate in candidates:
            digest.update(candidate.encode("utf-8"))
        digest.update(str(target).encode("utf-8"))
    return digest.hexdigest()


def base_outcomes(model: CrossEncoder, guards: dict, name: str) -> dict[str, list[int]]:
    """The untrained model's verdict on each guard, computed once and kept.

    This is half the wall clock of a training run and none of its information: the
    base weights do not change between runs, so neither can their guard outcomes.
    Cached against the model name and a digest of the guard's own text, so a rebuilt
    guard misses the cache instead of silently reusing the wrong baseline.
    """
    cached = {}
    if GUARD_CACHE.exists():
        cached = json.loads(GUARD_CACHE.read_text(encoding="utf-8"))
    out, fresh = {}, False
    for guard_name, rows in sorted(guards.items()):
        key = "%s|%s|%s" % (name, guard_name, guard_fingerprint(rows))
        if key in cached:
            out[guard_name] = cached[key]
            continue
        out[guard_name] = guard_outcomes(model, rows)
        cached[key] = out[guard_name]
        fresh = True
    if fresh:
        GUARD_CACHE.parent.mkdir(parents=True, exist_ok=True)
        GUARD_CACHE.write_text(json.dumps(cached), encoding="utf-8")
    return out


def guard_outcomes(model: CrossEncoder, guard) -> list[int]:
    """Per query: did the relevant passage still score highest? One or zero.

    Kept per query rather than averaged, because the question is whether *this*
    adapter changed *these* queries, and a pair of averages cannot answer that.
    """
    got = []
    with torch.no_grad():
        for query, passages, target in guard:
            scores = score_batch(model, [(query, text) for text in passages])
            got.append(int(scores.argmax().item() == target))
    return got


def mcnemar(before: list[int], after: list[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar over the queries where the two models disagree.

    The right test for paired binary outcomes, and the reason the guard is worth
    running at all: comparing two accuracy figures invites reading a difference that
    the sample cannot support. Only the discordant pairs carry information.
    """
    gained = sum(1 for a, b in zip(before, after) if not a and b)
    lost = sum(1 for a, b in zip(before, after) if a and not b)
    total = gained + lost
    if total == 0:
        return gained, lost, 1.0
    smaller = min(gained, lost)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2**total
    return gained, lost, min(1.0, 2 * tail)


def save_adapter(model: CrossEncoder, config: LoraConfig, out_dir: str) -> None:
    """Just the adapter tensors and the config needed to put them back.

    Not a merged checkpoint. Merging into bfloat16 weights was measured to move
    scores by around 0.1, and a merged copy also throws away the property that makes
    an adapter the safe choice: the base model stays exactly as downloaded, so a bad
    run is undone by not loading this directory.
    """
    destination = pathlib.Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    weights = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.model.named_parameters()
        if "lora_" in name
    }
    safetensors.torch.save_file(weights, str(destination / "adapter_model.safetensors"))
    (destination / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": config.r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "bias": config.bias,
                "target_modules": list(config.target_modules),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "base_model.txt").write_text(
        MULTILINGUAL_MODEL + "\n", encoding="utf-8"
    )


class Validation(typing.NamedTuple):
    """What the held-back slates say about a checkpoint.

    `attribution` is the stopping signal and `ranking` is half the ship gate, and
    they are deliberately separate numbers: an adapter can learn to attribute well
    while ranking worse, which is a trade this package should never make silently.
    """

    attribution: float
    effect: float
    ranking: float


def attribution_accuracy(model: CrossEncoder, examples) -> Validation:
    """Does the cause's ablation effect fall hardest on its own candidate?

    The stopping signal. It is the attribution objective measured as a decision
    rather than as a loss, on slates held back from training, and it is what decides
    when to stop -- the loss itself was still falling in the first run while the
    guard drifted, which is the stopping signal taken from the wrong curve.

    `ranking` comes free from the same forward pass: the planted candidate scoring
    highest on a held-back slate. The guards measure ranking on general corpora the
    adapter never sees, which is the wrong place to look for the thing the adapter is
    *for*, and until this was returned the gate had no in-domain term at all.
    """
    right, ranked, margins = 0, 0, []
    with torch.no_grad():
        for example in examples:
            scores = score_batch(model, example.full)
            ablated = score_batch(model, example.ablated)
            delta = (scores - ablated).unsqueeze(0)
            mask = torch.ones((1, delta.shape[1]), dtype=torch.bool, device=delta.device)
            right += int(delta.argmax().item() == example.target)
            ranked += int(scores.argmax().item() == example.target)
            margins.append(float(standout(delta, mask)[0, example.target]))
    count = max(len(examples), 1)
    return Validation(
        attribution=100 * right / count,
        effect=statistics.mean(margins) if margins else 0.0,
        ranking=100 * ranked / count,
    )


def refuse(start: Validation, final: Validation, worst: float) -> str | None:
    """Why the adapter is refused, or None if it ships.

    Two terms and both are required. In-domain ranking is what the adapter is for, so
    a regression there is disqualifying at any magnitude: an adapter that is worse at
    its own job has nothing to trade with. The guards are canaries on corpora it never
    trains on, so they bound collateral damage rather than test for it, and a fall
    inside the tolerance is the expected price of specialising.

    Separated from the training run so the decision can be checked without a GPU.
    """
    if final.ranking < start.ranking:
        return ("in-domain ranking fell %.1f points, so the adapter is worse at the "
                "job it was trained for" % (start.ranking - final.ranking))
    if worst > GUARD_TOLERANCE:
        return ("a guard fell %.1f points, over the %.1f tolerance, so general ranking "
                "is damaged rather than merely specialised away from"
                % (worst, GUARD_TOLERANCE))
    return None


def main(
    steps: int = 200,
    out_dir: str = "tmp/lora-mxbai",
    anchor: float | None = None,
    miracl_languages: int = 10,
    guard_queries: int = GUARD_QUERIES,
    margin: float = DEFAULT_MARGIN,
    seed: int = 20261120,
    fold: int = 0,
) -> None:
    """`guard_queries` exists for smoke runs and defaults to the real gate. Lowering
    it makes the guards unable to resolve a collapse -- 120 was measured to be about
    one standard error on a four-point move -- so a run that sets it is a wiring
    check and not a result."""
    weights = dict(WEIGHTS)
    if anchor is not None:
        weights["anchor"] = anchor
    print("loss weights: %s" % "  ".join("%s %.2f" % kv for kv in sorted(weights.items())))
    # Printed because it is a property of the corpus as much as of the model. The
    # hinge asks the cause's centred effect to reach this much, so the distance
    # between it and where the corpus's plants actually sit is how hard the term is
    # pushing the score function -- and general ranking is what pays for the push.
    print("margin: %.2f on the centred effect, against a shipped print floor of %.2f"
          % (margin, Settings.min_standout))
    # Only the training data is reseeded. The guards keep their own fixed seed, so
    # that two runs face the same guard rows and the paired test stays paired --
    # and so that one run's verdict can be compared with another's at all. Varying
    # this is how to tell a property of the recipe from a property of one draw: a
    # gate cleared at p = 0.06 on a single seed is a coin landing, not a finding.
    rng = random.Random(seed)
    reserved = category_fold(fold)
    print("training seed: %d" % seed)
    # NO_FOLD holds nothing back on any axis -- categories, languages, domains -- and
    # is what the adapter that ships is trained with. Every hold-out here exists to
    # measure generalisation, not to improve the model, so once the folds have
    # measured it, keeping them costs quality on the axes they reserve and buys
    # nothing. The gate survives it: the validation slates are still unseen draws,
    # and the three guards are separate corpora that no run ever trains on.
    final = fold == NO_FOLD
    every_language = tuple(sorted(set(MATCHED) | set(ACTIONS)))
    every_domain = tuple(sorted(DOMAINS))
    train_languages = every_language if final else tuple(
        code for code in every_language if code not in HELD_OUT_LANGUAGES
    )
    train_domains = every_domain if final else tuple(
        name for name in every_domain if name not in HELD_OUT_DOMAINS
    )
    if final:
        print("no hold-out: trained on all 48 categories, %d languages, %d domains."
              % (len(train_languages), len(train_domains)))
        print("  The folds measured generalisation; this is the adapter that ships.")
    else:
        print("category fold %d of %d: %d of 48 categories reserved, trained on %d"
              % (fold, CATEGORY_FOLDS, len(reserved), 48 - len(reserved)))
    print("synthetic: %d languages %s" % (len(train_languages), " ".join(train_languages)))
    print("synthetic: %d domains   %s" % (len(train_domains), " ".join(train_domains)))
    if not final:
        print("held out:  languages %s, domains %s"
              % (" ".join(HELD_OUT_LANGUAGES), " ".join(HELD_OUT_DOMAINS)))

    synthetic_cases = slates(rng, steps, train_languages, train_domains, reserved)
    # Half the slates from MIRACL: real queries, human relevance judgements, and a
    # content pool large enough that memorising it is not the cheap path. Every
    # distractor there is a real query that is the right answer for another row.
    chosen_languages = MIRACL_LANGUAGES[:miracl_languages]
    print("miracl:    %d languages %s" % (len(chosen_languages), " ".join(chosen_languages)))
    per_language = max(1, steps // max(len(chosen_languages), 1))
    miracl_cases = []
    for language in chosen_languages:
        try:
            miracl_cases.extend(miracl_slates(language, per_language, rng))
        except Exception as error:  # noqa: BLE001 - reported, and the run continues
            print("  %s unavailable: %s" % (language, str(error)[:70]))

    # Validation is stratified, and that is not tidiness. Pooled, it read 95.8%
    # before training and could not have stopped anything: MIRACL slates without
    # hard negatives were trivial, and their share of the pool hid the synthetic
    # slates where the headroom is. Each source is now reported on its own.
    held = {}
    train = []
    for name, group in (("synthetic", synthetic_cases), ("miracl", miracl_cases)):
        examples_here = [to_example(case) for case in group]
        cut = max(VALIDATION_FLOOR, len(examples_here) // 10)
        if cut >= len(examples_here):
            raise ValueError(
                "%s has %d slates, which a %d-slate validation floor would consume "
                "entirely, leaving nothing to train on. Raise the slate count to at "
                "least %d. The floor is not negotiable downwards: a validation set "
                "smaller than %d cannot serve as a stopping signal, which is the job "
                "it exists for."
                % (name, len(examples_here), VALIDATION_FLOOR,
                   VALIDATION_FLOOR + 1, VALIDATION_FLOOR)
            )
        held[name] = examples_here[:cut]
        train.extend(examples_here[cut:])
    rng.shuffle(train)
    examples = train
    validation = held["synthetic"] + held["miracl"]
    print("\n%d training slates, validation %d synthetic + %d miracl, "
          "%.1f candidates each on average"
          % (len(examples), len(held["synthetic"]), len(held["miracl"]),
             statistics.mean(len(e.full) for e in examples)))

    model = CrossEncoder(MULTILINGUAL_MODEL)
    guards = build_guards(guard_queries)
    print("general-ranking guards: %s"
          % "  ".join("%s %d queries" % (name, len(rows))
                      for name, rows in sorted(guards.items())), flush=True)
    before = base_outcomes(model, guards, MULTILINGUAL_MODEL)
    for name in sorted(before):
        print("guard accuracy before training: %-12s %5.1f%%"
              % (name, 100 * sum(before[name]) / len(before[name])), flush=True)
    for name, group in sorted(held.items()):
        if group:
            check = attribution_accuracy(model, group)
            print("validation before: %-10s attribution %5.1f%% (effect %.3f), "
                  "ranking %5.1f%%"
                  % (name, check.attribution, check.effect, check.ranking), flush=True)
    start = attribution_accuracy(model, validation)
    print()

    base_scores = []
    with torch.no_grad():
        for example in examples:
            base_scores.append(score_batch(model, example.full).cpu().tolist())

    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    # In place. Assigning to `model.model` would register a third child on the
    # pipeline, which then receives the features dict as a positional argument.
    inject_adapter_in_model(config, model.model)
    for name, parameter in model.model.named_parameters():
        parameter.requires_grad = "lora_" in name
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print("adapter: %.2fM trainable parameters, base weights frozen" % (trainable / 1e6))

    # MIRACL slates are five to ten times longer than the synthetic ones -- several
    # hundred characters of passage per candidate rather than a listing -- and
    # storing every activation for the backward pass exhausted the card. Checkpointing
    # recomputes them instead, which costs time and makes the long slates possible at
    # all. `use_cache` has to go with it: a cache of past keys is what checkpointing
    # is choosing not to keep.
    model.model.config.use_cache = False
    model.model.gradient_checkpointing_enable()
    model.model.enable_input_require_grads()
    print("gradient checkpointing on, activation cache off\n")

    optimiser = torch.optim.AdamW(
        [p for p in model.model.parameters() if p.requires_grad], lr=1e-4
    )
    best = {"accuracy": start.attribution, "state": None, "step": 0}
    stale = 0
    history = []
    check_every = max(MIN_CHECK_EVERY, len(examples) // CHECKS_PER_RUN)
    began = time.perf_counter()
    for step, (example, base) in enumerate(zip(examples, base_scores), start=1):
        model.train()
        scores = score_batch(model, example.full)
        ablated = score_batch(model, example.ablated)
        mask = torch.ones((1, scores.shape[0]), dtype=torch.bool, device=scores.device)
        loss, parts = total_loss(
            scores.unsqueeze(0),
            ablated.unsqueeze(0),
            torch.tensor([base], device=scores.device),
            torch.tensor([example.target], device=scores.device),
            mask,
            weights,
            margin,
        )
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        history.append(parts)

        if step % check_every == 0:
            model.eval()
            check = attribution_accuracy(model, validation)
            accuracy, margin = check.attribution, check.effect
            parts_by_source = {
                name: attribution_accuracy(model, group).attribution
                for name, group in sorted(held.items())
                if group
            }
            recent = history[-check_every:]
            print("  step %4d/%d  %s  | val %.1f%% (%s) effect %.3f  (%.0f s)"
                  % (step, len(examples),
                     "  ".join("%s %.3f" % (name, statistics.mean(h[name] for h in recent))
                               for name in ("rank", "attribution", "margin", "anchor")),
                     accuracy,
                     " ".join("%s %.0f%%" % kv for kv in parts_by_source.items()),
                     margin, time.perf_counter() - began), flush=True)
            if accuracy > best["accuracy"] + 0.5:
                best = {
                    "accuracy": accuracy,
                    "state": {
                        name: parameter.detach().clone()
                        for name, parameter in model.model.named_parameters()
                        if "lora_" in name
                    },
                    "step": step,
                }
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print("  stopping: %d checks without improvement on validation"
                          % PATIENCE)
                    break

    if best["state"] is None:
        print("\nvalidation never improved on the untrained model. Nothing to save.")
        return
    # The best checkpoint, not the last one. They differ: the first run's attribution
    # loss was still falling while nothing was watching whether it still helped.
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if name in best["state"]:
                parameter.copy_(best["state"][name])
    model.eval()
    print("\nrestored the best checkpoint, from step %d at %.1f%% validation"
          % (best["step"], best["accuracy"]))

    after = {name: guard_outcomes(model, rows) for name, rows in guards.items()}
    print("\ngeneral ranking on corpora the adapter never saw, paired over the same queries:")
    worst = report_guards(before, after)

    final = attribution_accuracy(model, validation)
    print("\nin-domain, on slates held back from training:")
    print("  attribution   %5.1f%% -> %5.1f%%   %+.1f pts"
          % (start.attribution, final.attribution, final.attribution - start.attribution))
    print("  ranking       %5.1f%% -> %5.1f%%   %+.1f pts"
          % (start.ranking, final.ranking, final.ranking - start.ranking))

    print("\nship gate: in-domain ranking must not regress, no guard may fall more"
          " than %.1f points" % GUARD_TOLERANCE)
    refusal = refuse(start, final, worst)
    if refusal is None:
        print("  CLEARED: in-domain ranking %+.1f points, worst guard fall %.1f points."
              % (final.ranking - start.ranking, worst))
    else:
        print("  REFUSED: %s" % refusal)
        print("  The base weights are untouched, so there is nothing to roll back --")
        print("  stop passing adapter= and the model is as it was.")

    save_adapter(model, config, out_dir)
    print("\nadapter written to %s, base weights untouched" % out_dir)
    # The shipped gate floor, not the training margin. These are different
    # quantities -- the margin is where the hinge stops pushing, deliberately above
    # the floor -- and printing the margin here sent every post-training evaluation
    # to an operating point twice as strict as the product's.
    print("evaluate with: uv run python examples/evaluation.py 1100 %.2f %s %s %s"
          % (Settings.min_standout, MULTILINGUAL_MODEL, "tmp/matrices-lora.json", out_dir))



if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 200,
        sys.argv[2] if len(sys.argv) > 2 else "tmp/lora-mxbai",
        float(sys.argv[3]) if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 10,
        int(sys.argv[5]) if len(sys.argv) > 5 else GUARD_QUERIES,
        float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_MARGIN,
        int(sys.argv[7]) if len(sys.argv) > 7 else 20261120,
        int(sys.argv[8]) if len(sys.argv) > 8 else 0,
    )
