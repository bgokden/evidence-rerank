"""A general-ranking guard from Mr. TyDi, independent of the training data.

The guard exists to answer one question: did training the adapter on attribution
make the model worse at ordinary ranking? A guard drawn from the same dataset family
as the training data cannot answer it. The existing guard is MIRACL Swahili while
training uses ten other MIRACL languages -- same construction, same annotation
pipeline, same passage shape -- so a model that overfitted MIRACL's *form* rather
than learning anything would still pass.

Mr. TyDi is a different lineage: built from TyDi QA rather than from MIRACL's own
annotation effort, with its own query distribution and its own negatives. Apache 2.0,
and its training files ship positives and negatives inline, so there is no corpus to
download.

Two guards are built from it, and the pair is the point:

    swahili   the same language as the MIRACL guard. Holding language fixed and
              changing dataset isolates exactly the concern -- if the adapter looks
              fine on MIRACL Swahili and bad here, it learned MIRACL's shape.
    english   a language MIRACL training never saw, so a second axis of shift.

Not usable for this: MS MARCO and everything derived from it, restricted to
non-commercial research by its own terms. No commercially licensed multilingual
retrieval set outside the Wikipedia-derived family was found, so Mr. TyDi is as
independent as this gets without changing the corpus's provenance too.
"""

from __future__ import annotations

import gzip
import json
import random

from huggingface_hub import hf_hub_download
from passages import trim

# Languages whose training file is small enough to read directly. The guard needs a
# few hundred queries, not the whole set.
LANGUAGES = ("swahili", "english")
NEGATIVES = 4


def guard(
    language: str = "swahili", limit: int = 400, seed: int = 20261120
) -> list[tuple[str, list[str], int]]:
    """(query, candidate passages, index of the relevant one).

    Passages are trimmed exactly as the training slates are, so the guard measures
    the model on lengths it was trained at rather than on whole Wikipedia sections.

    The candidates are shuffled, and the shuffle is load-bearing rather than tidy.
    With the positive fixed at index 0, `argmax` resolves every exact tie in its
    favour, because it returns the first maximal index. Ties are not rare here --
    bf16 scores land on coarse steps, so about 3% of rows have one -- and that was
    worth roughly two points of inflated guard accuracy. Seeded, so the guard is the
    same set of rows on every run and before/after comparisons stay paired.
    """
    path = hf_hub_download(
        "castorini/mr-tydi",
        "mrtydi-v1.1-%s/train.jsonl.gz" % language,
        repo_type="dataset",
    )
    rng = random.Random(seed)
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            positives = row.get("positive_passages") or []
            negatives = row.get("negative_passages") or []
            if not positives or len(negatives) < 2:
                continue
            relevant = trim(positives[0]["text"])
            passages = [relevant]
            passages += [trim(item["text"]) for item in negatives[:NEGATIVES]]
            rng.shuffle(passages)
            out.append((row["query"], passages, passages.index(relevant)))
            if len(out) >= limit:
                break
    return out
