"""Training slates built from MIRACL, so the content is not 63 hand-written families.

The first LoRA run gained three to four times as much on data it had trained on as
on data it had not, which is memorisation measured rather than suspected. Sixty-three
families is too few, and no hyperparameter fixes that.

MIRACL supplies the way out, and the structure fits exactly. A slate is several real
queries and the passages they are relevant to: query `i` explains passage `i`, so the
attribution label comes free from a human relevance judgement rather than from
anything written here. It also closes the shortcut the synthetic corpus left open --
every distractor on the page is another real query, so "pick the line that reads like
a request" pays nothing, and each distractor is the correct answer for a *different*
candidate, which is the hardest version of the task.

Licence, which decided the choice: MIRACL is Apache 2.0 and its corpus is Wikipedia.
MS MARCO is restricted to non-commercial research by its own terms, and that rules
out mMARCO and most of what the ecosystem ships as standard re-ranking training data,
whatever licence a repackager attached.
"""

from __future__ import annotations

import collections
import gzip
import json
import pathlib
import random

from huggingface_hub import hf_hub_download
from passages import trim

from evrank.evidence import Evidence, EvidenceSet
from evrank.ranking import Candidate

# Languages to train on. Excluded on purpose: fr and zh are held-out evaluation
# languages, sw and yo are the general-ranking guard. Everything else MIRACL has,
# ordered by corpus download size so a short run can take a prefix.
LANGUAGES = (
    "bn", "te", "hi", "th", "id", "ko", "fa", "fi", "ar", "ja", "es", "ru", "de", "en",
)

CACHE = pathlib.Path("tmp/miracl-cache")

def read_pairs(path: str) -> dict[str, str]:
    out = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def judgements(language: str, split: str = "train"):
    """Which passages a human marked relevant, and which they marked not."""
    path = hf_hub_download(
        "miracl/miracl",
        "miracl-v1.0-%s/qrels/qrels.miracl-v1.0-%s-%s.tsv" % (language, language, split),
        repo_type="dataset",
    )
    relevant, irrelevant = {}, collections.defaultdict(list)
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        query_id, doc_id, grade = parts[0], parts[2], int(parts[3])
        if grade > 0:
            relevant.setdefault(query_id, doc_id)
        else:
            irrelevant[query_id].append(doc_id)
    return relevant, irrelevant


def passages(language: str, wanted: set[str]) -> dict[str, str]:
    """The text of the passages named in the judgements, cached after the first pass.

    The corpus ships as gzipped shards of a whole Wikipedia, and only a few thousand
    passages per language are ever referenced. Scanning once and caching the subset
    turns a multi-gigabyte read into a small file.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / ("%s.json" % language)
    if cached.exists():
        stored = json.loads(cached.read_text(encoding="utf-8"))
        if wanted <= set(stored):
            return stored
    from huggingface_hub import HfApi

    found = {}
    shards = [
        name
        for name in HfApi().list_repo_files("miracl/miracl-corpus", repo_type="dataset")
        if name.startswith("miracl-corpus-v1.0-%s/" % language) and name.endswith(".jsonl.gz")
    ]
    for shard in sorted(shards):
        path = hf_hub_download("miracl/miracl-corpus", shard, repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["docid"] in wanted:
                    found[row["docid"]] = trim(row["text"])
        if len(found) >= len(wanted):
            break
    cached.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")
    return found


def slates(
    language: str, count: int, rng: random.Random, sizes=(3, 4, 5), hard: int = 2
) -> list[dict]:
    """Slates where every evidence line is a real query for one of the candidates.

    Each slate holds `size` queries and the passages they are relevant to, so exactly
    one line explains each row. One query is named as the cause to ablate.

    `hard` is what makes this worth training on. Without it the other candidates are
    positives for unrelated queries, and the task is trivial -- measured, the base
    model scored 100% on slates built that way, so the attribution objective had
    nothing to teach. These are MIRACL's own grade-zero judgements for the target
    query: passages a retrieval system returned for that query and a human then
    marked not relevant. They are about the same topic as the cause, so the cause's
    effect has to land on the passage that actually answers it rather than on the
    three that merely share its subject.
    """
    relevant, irrelevant = judgements(language)
    usable = sorted(relevant)
    if not usable:
        return []
    wanted = {relevant[q] for q in usable}
    for query_id in usable:
        wanted.update(irrelevant.get(query_id, [])[:hard])
    texts = passages(language, wanted)
    usable = [q for q in usable if relevant[q] in texts]
    queries = read_pairs(
        hf_hub_download(
            "miracl/miracl",
            "miracl-v1.0-%s/topics/topics.miracl-v1.0-%s-train.tsv" % (language, language),
            repo_type="dataset",
        )
    )
    usable = [q for q in usable if q in queries]

    out = []
    while len(out) < count and len(usable) >= max(sizes):
        size = rng.choice(sizes)
        chosen = rng.sample(usable, size)
        target = rng.randrange(size)
        # The cause's own hard negatives join the page as extra candidates, so the
        # row it has to beat is on the same topic rather than on another one.
        decoys = [
            texts[doc]
            for doc in irrelevant.get(chosen[target], [])
            if doc in texts and doc != relevant[chosen[target]]
        ][:hard]
        texts_in_order = [texts[relevant[query_id]] for query_id in chosen] + decoys
        candidates = [
            Candidate("c%d" % index, "Option %d" % (index + 1), text)
            for index, text in enumerate(texts_in_order)
        ]
        items = [
            Evidence("k%d" % index, "searched for", trim(queries[query_id], 200))
            for index, query_id in enumerate(chosen)
        ]
        rng.shuffle(items)
        out.append(
            {
                "id": len(out),
                "domain": "miracl",
                "language": language,
                "condition": "miracl",
                "candidates": candidates,
                "evidence": EvidenceSet(items),
                "target": candidates[target].id,
                "key": "k%d" % target,
                "size": len(candidates),
                "units": len(items),
            }
        )
    return out
