# evidence-rerank

**Explain a neural re-ranker by the user context it was given, not by the numbers
inside it.**

When a ranker puts one result above another, the useful question is which part of the
person's recorded activity caused it — the search they ran, the item they reserved,
the filter they set. This library answers that by removing one piece of user context
at a time and re-scoring, so every explanation is the difference between two scores
the model actually produced.

```
Because you searched for a two-bedroom apartment near a metro station
and reserved parking for the same dates.
```

Nothing looks inside the model. A cross-encoder has no individually meaningful
features, and labelling hidden dimension 147 after the fact invents an explanation
rather than finding one. The model stays a plain function from `(context, document)`
pairs to scores, which means any re-ranker works, including one behind an API you
cannot see into.

- **Model-agnostic.** Anything that scores pairs fits behind one small protocol.
- **Ranking stays fast.** Explanations are computed on demand, not per request.
- **Honest by default.** A result nothing singled out gets no explanation.

---

## Install

```bash
pip install -e .                      # or: uv sync
pip install -e ".[serve]"             # adds the HTTP service and its UI
```

Python 3.12+. The only required dependencies are `sentence-transformers` and `peft`.

## Quickstart

```python
from evrank import Candidate, Evidence, EvidenceSet, rank
from evrank.config import Settings
from evrank.cross_encoder import CrossEncoderScorer
from evrank.scoring import Ablator
from evrank.standout import standout

# The user context, kept as discrete facts rather than one blob of prose.
context = EvidenceSet([
    Evidence("s1", "searched", "two-bedroom apartment near a metro station"),
    Evidence("r1", "reserved", "parking for the same dates"),
    Evidence("v1", "viewed", "a listing in the old town"),
    Evidence("f1", "filtered", "under 1,400 a month"),
])

candidates = [
    Candidate("a", "Lindenstrasse 8", "Two bedrooms, 4 minutes from the metro, parking included."),
    Candidate("b", "Old Town Loft", "One bedroom in the old town, no parking."),
    Candidate("c", "Riverside Studio", "Studio flat by the river, 1,250 a month."),
]

scorer = CrossEncoderScorer()
ranking = rank(scorer, context, candidates)
rows = standout(Ablator(scorer, context, candidates))
settings = Settings()

for place, cid in enumerate(ranking.order(), start=1):
    reasons = rows[cid].favouring(
        min_specificity=settings.min_specificity,
        min_effect=settings.min_effect,
        min_standout=settings.min_standout,
    )
    print("%d. %s" % (place, ranking.name(cid)))
    if not reasons:
        print("     (nothing singled it out)")
    for effect in reasons[:2]:
        kind, said = context.phrase(effect.unit)
        print("     %-9s %-42s %+.2f" % (kind, said, effect.standout))
```

Ranked, with the reasons that cleared the floors:

```
1. Lindenstrasse 8
     searched  two-bedroom apartment near a metro station +4.34
2. Riverside Studio
     filtered  under 1,400 a month                        +3.34
3. Old Town Loft
     viewed    a listing in the old town                  +2.09
     reserved  parking for the same dates                 +0.94
```

Each number is what that piece of user context did for *that* result, after
subtracting what it did for the rest of the list — see [How it works](#how-it-works)
for why the subtraction is the whole trick. Two of the four facts cleared the floors
on no row at all, which is the intended behaviour: a result nothing singled out gets
no sentence.

The last line is worth reading closely, because it is the library being honest about a
ranker that is wrong. The parking reservation is cited for **Old Town Loft**, whose
text says "no parking" — the model matched the word and missed the negation, and on
the listing that actually includes parking the same fact came out *negative*. The
explanation is reporting that faithfully. Cross-encoders do not do negation or number
comparison, no threshold on the measurement finds it, and what finds it is reading the
reason. That is the argument for keeping reasons short and quotable.

For a runnable version against the real model:

```bash
uv run python examples/walkthrough.py
```

---

## User context

User context is kept as a list of facts, not flattened into a sentence. Each one has
an id, a `kind`, and the text the person's activity produced:

```python
Evidence("s1", "searched", "two-bedroom apartment near a metro station")
```

The `kind` is the verb the explanation will use, so it reads best as something the
person did. The vocabulary is yours and should come from your own event schema, but
these are the ones the shipped sentence templates know how to speak:

| kind | what it records |
| ---- | --------------- |
| `searched` | a query the person ran |
| `viewed` | an item they opened |
| `reserved` | a booking or hold they placed |
| `saved` | an item they kept for later |
| `compared` | items they put side by side |
| `filtered` | a constraint they set, such as a price ceiling |
| `requested` | something they asked for explicitly |
| `dismissed` | an item they rejected |
| `rated` | feedback they left |
| `purchased` | an order they completed |

Keep the vocabulary stable between requests, since the labels carry a little signal
of their own. Kinds can stay in one language while the text varies: holding the text
fixed and swapping English kinds for Turkish ones, or for a generic `event` label,
produced the same ranking order and the same effect signs, with magnitudes moving by
up to 1.5 — less than changing how removal is done costs.

Each fact is its own unit unless `Evidence.group` says otherwise. Grouping is for
facts that genuinely repeat — the same search fired five times — where removing one
copy changes nothing and every copy measures zero. It is **not** for facts on a
common topic. An earlier version of the walkthrough grouped three searches under one
topic label: measured separately they were `+10.25`, `-4.89` and `-1.61`, because the
model read them as evidence for different results. Grouped, they reported their net
of `-1.74` and hid the largest effect in the explanation.

`Evidence.at` is kept for you and never rendered. Putting timestamps in the context
costs 22% more tokens and shrinks the margin — 1.42 down to 0.50 as an ISO date, 0.83
as "3 days ago" — without changing the order or which fact was decisive. This model
reads dates as noise. Recency belongs in your own weighting, or in `text` if the model
should see it.

---

## How it works

### Two questions, two measurements

"Why is this above that?" and "Why am I being shown this?" are different questions,
and a margin only answers the first.

**Why this is above that.** For a fact `e` and two results A and B:

```
delta_e = (s_A - s_B) - (s_A^-e - s_B^-e)
```

where `s^-e` is the score after that fact is left out of the context. Positive means
the model prefers A over B more strongly when `e` is present.

**Why you are seeing this at all** is a claim about one result's own score, and
measured directly that score is too blunt to say anything. One search for a
two-bedroom flat lifts *every* two-bedroom listing, so the strongest cause comes out
as the same sentence on every row. Subtracting what the same fact did for the rest of
the list fixes it:

```
standout_e,A = (s_A - s_A^-e) - mean over other results (s_C - s_C^-e)
```

That is what sends the metro search to the listing near the metro and the parking
reservation to the listing with parking. The model held those associations already;
centring is what reads them back out. A result nothing singled out gets no sentence,
which is both the honest outcome and what a product wants.

`standout` costs one context per fact against the whole list in a single batched
call, and equals the pairwise effect exactly when there are two results.

### Deciding what is worth printing

A true effect is not automatically worth showing. Three floors stand between a
measurement and a sentence:

| floor | sits under | default | catches |
| ----- | ---------- | ------- | ------- |
| `min_standout` | the centred effect | 0.75 | the fact that moved the whole list, not this row |
| `min_specificity` | `standout / delta` | 0.3 | effects that are real but shared |
| `min_effect` | the raw effect on this row | small | the fact that barely touched this row |

`specificity` is the share of a fact's work that was specific to this result. A fact
can lift one listing by +0.40 while lifting the whole list by +0.396 — citing it is
true and useless. Being a ratio of two differences in the same model's scores, it
carries across models where an absolute size would not. Above 1.0 means the fact
actively cost the other results, which is the strongest form of singling a row out.

`min_standout` carries the decision. Over 1,100 cases in eleven languages:

| floor | precision | invented | first rows speaking with no cause |
| ----- | --------- | -------- | --------------------------------- |
| 0.00 | 22.2% | 61.7% | 73.3% |
| 0.25 | 25.0% | 49.0% | 61.5% |
| 0.50 | 32.3% | 28.9% | 43.8% |
| **0.75** | **42.1%** | **15.8%** | **29.8%** |
| 1.00 | 51.0% | 9.3% | 21.1% |
| 1.50 | 65.1% | 3.5% | 12.1% |

Precision triples across that range and unexplained first choices fall sixfold, so
the floor is a real quality control rather than a volume dial. Pick from the last
column: it is how many unexplained first results you are willing to let speak. The
weakest languages pay for it.

A floor belongs to the model it was swept on — see
[Choosing a model](#choosing-a-model). `Settings` pairs them and says why.

### How a fact is removed

Removal is a modelling choice with real consequences, so it is a parameter rather
than a default nobody sees.

- **`Omission`** drops the line, shortening the context.
- **`NeutralFiller`** keeps the line and its kind and replaces only the content, so
  what is removed is the substance of a fact rather than the fact of its existence.
- **`KindOnly`** keeps the line and the kind and drops the content, so no new wording
  enters the context. Useful when you cannot safely write filler in the user's
  language.

Neither is ground truth. Omission asks the model about a shorter context than it ever
sees in production; filler asks it about a context holding a sentence the person's
activity never produced. `compare_maskers` runs the attribution under each and reports
where they agree, which separates an effect from an artifact of how the question was
asked. A fact every strategy measures as zero counts as agreement — they concur it did
nothing. Only facts landing on opposite sides of zero are disputed.

### Estimators

| function | cost | use when |
| -------- | ---- | -------- |
| `leave_one_out` / `standout` | `units + 1` contexts | the default |
| `shapley` | `2^units` contexts | duplicate facts would otherwise cancel out |
| `sampled_shapley` | `permutations × (units + 1)` | too many facts to enumerate |
| `flip_pair` / `flip_top` | search | "what would have to go for this to change?" |

`leave_one_out` is the default on purpose: it removes one piece of the person's real
context and reports the difference between two scores the model actually produced,
which is exactly what "because you searched for X" asserts. The Shapley estimators
average a fact's contribution over every order the activity could have arrived in.
They apportion shared credit more fairly, but the coalitions they average over are
contexts the user never had — and a Shapley value can sit far from anything the model
does in any single context, so `shapley` also reports the spread of what it averaged
and flags a fact whose contribution changes sign.

`sampled_shapley` carries a standard error that falls as the square root of the
permutation count, and its values still sum exactly to the margin however few orders
are drawn. The seed is fixed, so the same request returns the same explanation.

---

## The HTTP service

```bash
uv run uvicorn evrank.service:app --port 8137
```

Open <http://127.0.0.1:8137/>. The page *is* the API: an editable request on the left,
the response on the right, ranking and stage timings below. The model loads once, on
the first request.

One route. `POST /api/rank` always ranks, and explains only when asked:

```
{"evidence": [...], "candidates": [...]}                    ranking alone
{..., "explain": true}                                      reasons per result
{..., "explain": true, "sentences": true}                    plus a sentence
{..., "explain": true, "detail": true}                       plus the numbers
{..., "explain": true, "compare_maskers": true}              plus the agreement
```

Evidence is `{id, kind, text}` with an optional `group`; candidates are
`{id, name, text}`.

With `explain`, reasons ride on the result they are about:

```json
{"rank": 1, "id": "a", "name": "Lindenstrasse 8", "score": -2.60,
 "method": "leave_one_out",
 "reasons": [{"unit": "s1", "evidence": "two-bedroom apartment near a metro station",
              "effect": 2.91, "onItsOwnScore": 3.50, "specificity": 0.83}],
 "because": "Because you searched for a two-bedroom apartment near a metro station."}
```

`effect` is the part specific to this result, `onItsOwnScore` the raw effect and
`specificity` the share, so the three together show what centring did.

`reasons` carries only what was in the result's favour, strongest first. What held it
back is a different question with a different answer, so negatives go in `against`
behind `include_against` rather than being mixed into one list sorted by magnitude.
`top_reasons` caps each list. A row nothing singled out carries an empty `reasons` and
no `because`.

`sentences` adds a `because` line built from the same reasons and naming nothing else:

```
Lindenstrasse 8    Because you searched for a two-bedroom apartment near a
                   metro station and reserved parking for the same dates.
Riverside Studio   (no sentence: nothing singled it out)
```

No rank, no competitor, no margin, no numbers. An earlier version wrote sentences
about the attribution itself — "ranked 1 because X outweighs Y, leaving it 1.42
ahead" — which is a sentence about the method rather than about the person reading it.
It was removed rather than made optional. The margin attribution it was built on is
still available under `detail`.

Sentences come from a `RankPhrases` pack rather than hard-coded English. Seven ship
with the package and are listed by `GET /api/model` as `sentenceLanguages`: `de`,
`en`, `es`, `fr`, `it`, `ja`, `tr`. The ranker reads far more languages than that, and
your context and results can be in any of them whatever the sentence language —
registering a pack is all that stands between a language the model already handles and
a sentence in it. An unknown `language` returns 400 listing the ones that exist,
because a request for Korean quietly answered in English is an explanation in a
language the reader did not ask for.

Every stage reports what it spent, so the price of an explanation is in the response
rather than in the documentation:

```json
"timing": {"totalMs": 360.8, "modelPairs": 280, "pairsServedFromCache": 37,
           "stages": [{"name": "ranking", "ms": 19.0, "modelPairs": 5},
                      {"name": "attribution", "ms": 307.7, "modelPairs": 254}]}
```

Refusals surface as HTTP errors rather than 500s or silent zeros: a context that would
be truncated returns 413, and a context too large for exact Shapley returns 400 with
the message saying what to use instead.

---

## Choosing a model

Any model that scores `(context, document)` pairs fits behind `Scorer`, and the choice
matters more than anything else here: the explanation names the right cause about as
often as the ranker ranks it, so the model is the ceiling.

What decides it is **not** the quality a leaderboard reports. This method needs score
*differences* that mean something, and a model can rank acceptably while its
differences say nothing. `examples/model_comparison.py` measures that as separation —
how often a real cause's effect beats a junk one, where 0.5 is a coin flip — over 220
planted cases. Only weights licensed for commercial use are listed.

| model | separation | cited | first | licence |
| ----- | ---------- | ----- | ----- | ------- |
| `Qwen/Qwen3-Reranker-0.6B` | **0.794** | 52.6% | 56.9% | Apache 2.0 |
| `mixedbread-ai/mxbai-rerank-base-v2` *(default)* | 0.763 | 41.6% | 52.6% | Apache 2.0 |
| `cross-encoder/ettin-reranker-68m-v1` | 0.722 | 31.4% | 51.1% | Apache 2.0 |
| `cross-encoder/ettin-reranker-150m-v1` | 0.715 | 32.8% | 54.0% | Apache 2.0 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.668 | 5.1% | 34.3% | Apache 2.0 |
| `BAAI/bge-reranker-v2-m3` | 0.652 | 28.5% | 47.4% | Apache 2.0 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 0.591 | 5.1% | 27.7% | Apache 2.0 |
| `BAAI/bge-reranker-base` | 0.590 | 24.1% | 40.9% | MIT |
| no model, character trigrams | – | – | 30.7% | – |

Read the separation column and nothing else: it is the only quantity here that needs
no floor, while `cited` and `first` are measured at the floor this package ships,
which was swept on mxbai.

The last row is a twenty-line character-trigram matcher, reported permanently, because
a metric a string matcher can pass is not a measurement of attribution. Note that
`mmarco-mMiniLMv2` ranks *below* it despite being nominally multilingual, so it
carries no attributable signal at all.

**Check the English column separately.** `ettin-reranker-150m` is the strongest model
here on English — 61.8% of planted results ranked first, ahead of everything else —
and across eleven languages its attribution falls to 13.1%. A single averaged column
hides that:

| model | English | all languages | drop |
| ----- | ------- | ------------- | ---- |
| `Qwen/Qwen3-Reranker-0.6B` | 53.9% | 50.8% | 1.1× |
| `mixedbread-ai/mxbai-rerank-base-v2` | 47.4% | 34.4% | 1.4× |
| `cross-encoder/ettin-reranker-68m-v1` | 42.1% | 18.0% | 2.3× |
| `cross-encoder/ettin-reranker-150m-v1` | 48.7% | 13.1% | **3.7×** |

The two Ettin models are named `ENGLISH_MODEL` and `FAST_ENGLISH_MODEL` in
`evrank.cross_encoder` for that reason — the name is the only thing standing between a
reader of the first column and a bad choice.

Qwen leads separation on two different corpora, so it is a property of the models
rather than the content. It is not the default yet: mxbai is what the fine-tuned
adapter was trained on, Qwen costs 63 ms a reason against 45, and the two are level on
precision.

**Two model families do not fit this method.** Listwise re-rankers score the whole
list jointly, so there is no score for A independent of B and removing one result
changes every other — an ablation isolates nothing. Hosted API re-rankers are a poor
fit for a different reason: an exact Shapley pass is `2^n` scored contexts per
explanation, priced per call, and they usually return a squashed score rather than the
raw one a margin needs.

Scores are read *before* any output activation. Several re-rankers ship a sigmoid,
which is monotonic and leaves the ranking untouched but compresses differences at the
extremes: two results 0.26 apart in raw score can land 0.0005 apart after squashing.
Attribution measures differences, so it reads the raw score.

---

## Cost

Ranking and explaining are separate stages and scale differently. **Ranking** is one
batched inference over the result list whatever the context says. **Explanation**
re-scores modified contexts, and its cost is set by the number of facts, not the
number of results.

At the scale a request is expected to take — 20 to 30 facts against 5 to 10 results —
a full explanation with counterfactual search is well under two seconds:

| facts | results | ranking | leave-one-out | pair reversal | first place | total |
| ----- | ------- | ------- | ------------- | ------------- | ----------- | ----- |
| 20 | 5 | 27 ms | 174 ms | 0 ms | 251 ms | 452 ms |
| 30 | 5 | 36 ms | 367 ms | 0 ms | 508 ms | 912 ms |
| 20 | 10 | 55 ms | 174 ms | 0 ms | 605 ms | 834 ms |
| 30 | 10 | 66 ms | 367 ms | 0 ms | 1314 ms | 1747 ms |

The pair reversal is free because leave-one-out has already scored every single-fact
removal it needs. First-place search is the largest line and the only stage whose cost
grows with the result count.

Exact Shapley costs `2^facts` contexts: 13 seconds at the twelve-fact cap, three and a
half minutes at fifteen. Past that it is not slow, it is impossible — twenty facts is
a million contexts. That cap is why genuine duplicates should be grouped and why
`sampled_shapley` exists. At 25 facts, where enumeration is refused:

| estimator | pairs scored | wall clock | error on the largest effect |
| --------- | ------------ | ---------- | --------------------------- |
| leave-one-out | 132 | 0.6 s | none reported |
| sampled, 32 permutations | 1,584 | 4.5 s | ±0.302 |
| sampled, 128 permutations | 5,726 | 15.9 s | ±0.153 |
| sampled, 512 permutations | 21,370 | 61.4 s | ±0.074 |

Throughput also falls as contexts grow, since attention is quadratic in length: 354
pairs per second at 194 tokens against 49 at 1,338. Counting pairs alone understates
what a long context costs.

Within one explanation the stages share a cache, so counterfactual search is usually
free.

---

## Languages

Nothing in the attribution is language-specific. The context is rendered, modified and
re-scored the same way whatever it says, and your context and results need not even
share a language: with `MULTILINGUAL_MODEL`, Turkish context raises a result described
in Japanese, and the attribution reports which fact did it.

Capacity is not language-specific either. The same fact costs 14 tokens in English, 15
in Turkish and 17 in French, Japanese, Korean or Arabic, so an 8,192-token budget holds
481 to 585 facts whatever the script.

---

## What it does not claim

An explanation here is a statement about **the model's sensitivity to its input**. It
is not a causal account of what the person actually wants, and the generated text says
so.

Attributed magnitudes depend on how removal is done, so read them as one estimate
under a stated method rather than as the effect of a fact. Running both and comparing
is the honest default.

Removal is itself a distribution shift: a context with a line missing is a context the
model was never asked about in production. That is why facts are dropped whole rather
than token-masked, and why an empty context still renders as a well-formed baseline.

The explanation can be right about a ranker that is wrong, and
`examples/other_domains.py` exists to show it. Some of what it prints is incorrect —
a 990 budget listing cited for a ceiling of 1,800 is the model matching the shape of a
price without comparing numbers, and a listing reading "No pets" cited for a pets
filter is the model missing the negation. Both are reported faithfully. Those failures
are negation and number comparison, which cross-encoders do not do, and no threshold
on the measurement finds them. What finds them is reading the reason, which is the
argument for keeping reasons short and quotable.

Other limits worth knowing: pairs exceeding the model's context are **refused, not
truncated**, so a long context fails loudly rather than reporting zeros. Minimal flip
sets are minimal in count and not unique. Grouping changes the answer, since grouped
facts are credited together. Margins are uncalibrated logits, comparable within a pair
under one model and not across models.

---

## Layout

```
src/evrank/evidence.py       Evidence, EvidenceSet: the user context, grouping, rendering
src/evrank/ranking.py        Candidate, Ranking: scores, order, margins
src/evrank/scoring.py        Scorer protocol, CachingScorer, Ablator
src/evrank/standout.py       Per-result effects, centred across the list
src/evrank/attribution.py    leave_one_out, shapley, flip_pair, flip_top
src/evrank/masking.py        Omission, KindOnly, NeutralFiller
src/evrank/agreement.py      compare_maskers: which effects survive the choice
src/evrank/limits.py         Token budget policy: refuse truncation, scale the batch
src/evrank/cross_encoder.py  Standard cross-encoders behind Scorer
src/evrank/calibration.py    A learned gate over the measured effects
src/evrank/reasoning.py      A ranker that states its own reasons, behind Scorer
src/evrank/explain.py        Sentence templates over measured numbers
src/evrank/service.py        HTTP service and its UI
```

```
examples/walkthrough.py      End-to-end run against the real model
examples/other_domains.py    Lists the design was not tuned on
examples/cost_scaling.py     What ranking costs against what explaining costs
examples/corpus.py           Planted-context corpus: 512 validated pairs
examples/evaluation.py       The harness: planted lists scored against known answers
examples/validate_corpus.py  Checks the corpus labels with two unrelated models
examples/model_comparison.py The same cases through eight re-rankers
examples/compare.py          Two rules or two models at matched false positives
examples/finetune.py         Trains a LoRA adapter on the measured effects
examples/faithfulness.py     A stated reason against the measured one
```

## Development

```bash
uv sync --extra dev --extra serve
uv run pytest
uv run python examples/walkthrough.py
```

Tests check the attribution arithmetic against a toy scorer with known
contributions, so the maths is verified without a model download.

## Licence

Apache 2.0. See [LICENSE](LICENSE).
