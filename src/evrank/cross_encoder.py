"""Standard cross-encoder re-rankers behind the `Scorer` interface.

Four choices, all with downloadable weights under a licence that permits commercial
use, and all loaded the same way:

    DEFAULT_MODEL       what you get with no argument; the multilingual one
    MULTILINGUAL_MODEL  100+ languages, 32768 tokens
    LONG_CONTEXT_MODEL  100+ languages, 40960 tokens, a yes/no scoring head
    ENGLISH_MODEL       English only, 7999 tokens, 150M parameters
    FAST_ENGLISH_MODEL  English only, 7999 tokens, 68M parameters

"English only" is measured here rather than taken from a card, and the measurement is
what makes it worth naming in capitals. On English content ENGLISH_MODEL is the
strongest of the eight benchmarked -- 61.8% of planted candidates ranked first, better
than anything else -- and across the eleven-language corpus its attribution falls to
13.1% cited, second worst of the eight. A 3.7-fold collapse in the quantity this
package exists to produce, from a model that looks like the best choice if English is
all you test on.

That is also why DEFAULT_MODEL is the multilingual one despite being three times the
size. It was the English model, under this same name, while `config.DEFAULT_MODEL`
named the multilingual one -- two constants, one name, two values, and the quiet one
was the English model that a bare `CrossEncoderScorer()` would load. Someone
prototyping in French would have got 13% and concluded the method does not work.

Pick on language first, then context length, then speed -- but check the model
against `examples/model_comparison.py` before trusting it, because the quality that
matters here is not the one a leaderboard reports. What this package needs is score
*differences* that mean something, and a model can rank well while its differences
say nothing. Measured over 220 planted-evidence cases, where separation is how often
a real cause's ablation effect beats a junk one (0.5 is a coin flip):

    model                                   separation   cited   first   licence
    Qwen/Qwen3-Reranker-0.6B                  0.794   52.6%   56.9%   Apache-2.0
    mixedbread-ai/mxbai-rerank-base-v2        0.763   41.6%   52.6%   Apache-2.0
    cross-encoder/ettin-reranker-68m-v1       0.722   31.4%   51.1%   Apache-2.0
    cross-encoder/ettin-reranker-150m-v1      0.715   32.8%   54.0%   Apache-2.0
    cross-encoder/ms-marco-MiniLM-L-6-v2      0.668    5.1%   34.3%   Apache-2.0
    BAAI/bge-reranker-v2-m3                   0.652   28.5%   47.4%   Apache-2.0
    cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  0.591    5.1%   27.7%   Apache-2.0
    BAAI/bge-reranker-base                    0.590   24.1%   40.9%   MIT
    no model, character trigrams                  -       -   30.7%   -

Read the separation column across models and nothing else: it is the only quantity
here that needs no floor, while `cited` and `first` are measured at the floor this
package ships, which was swept on mxbai. A floor belongs to the model it was swept on.

Qwen leads on separation, and the shipped default does not. That has now been
measured on two different corpora -- 0.794 against 0.763 here, 0.867 against 0.820 on
the corpus this one replaced -- so it is a property of the models rather than of the
content. It is not yet a switch: mxbai is what the adapter was trained on, Qwen costs
63 ms a reason against 45 and is a 0.6B model, and the two are level on precision.

Two other readings. A nominally multilingual model can be useless here --
mmarco-mMiniLMv2 ranks *below* a twenty-line character-trigram matcher, 27.7% against
30.7%, so it carries no attributable signal at all. And the claim that
ms-marco-MiniLM sat *below* chance, its effects pointing away from the cause, is
retired: it was 0.424 on an earlier corpus and is 0.668 on this one, which is what a
corpus-specific finding looks like once the corpus changes.

Avoid listwise re-rankers entirely, whatever they score on a leaderboard. They score
the whole slate in one pass, so removing one candidate changes every other
candidate's score and an ablation isolates nothing.

None of them expose anything interpretable: their hidden dimensions carry no
individual meaning, which is the situation evidence-level attribution is built for.

Scores are read before the model's output activation by default. Several re-rankers
ship a sigmoid, which is monotonic and so leaves the ranking untouched, but it
compresses differences at the extremes: two candidates 0.26 apart in raw score can
land 0.0005 apart after squashing. Attribution measures differences, so it reads the
raw score and leaves the squashing to whatever consumes the ranking.

Pairs that would be truncated are refused rather than scored. See `evrank.limits`
for why truncation is not a degraded result here but a wrong one.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence

import safetensors.torch
import torch
from peft import LoraConfig, inject_adapter_in_model
from sentence_transformers import CrossEncoder

from evrank.limits import batch_size_for, ensure_within_limit

MULTILINGUAL_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
LONG_CONTEXT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
ENGLISH_MODEL = "cross-encoder/ettin-reranker-150m-v1"
FAST_ENGLISH_MODEL = "cross-encoder/ettin-reranker-68m-v1"
# The same model the service ships, so a bare scorer and a default deployment agree.
# They did not: this named the English-only model while `config.DEFAULT_MODEL` named
# this one.
DEFAULT_MODEL = MULTILINGUAL_MODEL

# Measured worse than the choices above and kept named rather than deleted, because
# it was this package's default and every number recorded before 2026-11-20 was
# measured on it.
PRIOR_MULTILINGUAL_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int | None = None,
        device: str | None = None,
        raw_scores: bool = True,
        trust_remote_code: bool = False,
        adapter: str | None = None,
    ) -> None:
        """`trust_remote_code` executes code the model repository ships.

        Off by default and never inferred from the model name. Several re-rankers
        worth comparing against are published as a custom architecture rather than
        as weights for a standard one, and loading them means running their code, so
        it is the caller who decides that for a given model rather than this class
        deciding it for them.

        `adapter` names a directory written by `examples/finetune.py`. The base
        weights are left exactly as downloaded and the adapter is layered on, which
        is the point: a trained adapter that turns out to make the ranking worse is
        removed by not passing this argument, with nothing to roll back. Merging it
        into the weights instead was measured to shift scores by around 0.1 in
        bfloat16, so it is layered rather than merged.
        """
        self.model_name = model_name
        self.adapter = adapter
        self._model = CrossEncoder(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        if adapter is not None:
            self._attach(adapter)
        # None tells sentence-transformers to use the model's configured activation.
        self.activation_fn = torch.nn.Identity() if raw_scores else None
        self.max_seq_length = int(self._model.max_seq_length)
        # None means: size each call from the pairs it carries.
        self.batch_size = batch_size

    def _attach(self, adapter: str) -> None:
        """Layer a saved adapter onto the loaded base weights."""
        directory = pathlib.Path(adapter)
        config = LoraConfig(
            **json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
        )
        # In place. Assigning to `self._model.model` would register an extra child on
        # the sentence-transformers pipeline, which then receives the features dict as
        # a positional argument and fails inside the embedding lookup.
        inject_adapter_in_model(config, self._model.model)
        weights = safetensors.torch.load_file(
            str(directory / "adapter_model.safetensors")
        )
        missing, unexpected = self._model.model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(
                "%s: %d adapter tensor(s) do not belong to %s, first is %s"
                % (adapter, len(unexpected), self.model_name, unexpected[0])
            )
        absent = [name for name in missing if "lora_" in name]
        if absent:
            raise ValueError(
                "%s: %d adapter tensor(s) missing from the file, first is %s"
                % (adapter, len(absent), absent[0])
            )
        self._model.to(self._model.model.device)

    @property
    def hidden_size(self) -> int:
        """Width of the representation the attribution deliberately does not read."""
        return int(self._model.model.config.hidden_size)

    def token_counts(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        """Length of each pair as the model would encode it, before any truncation."""
        encoded = self._model.tokenizer(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            truncation=False,
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        counts = self.token_counts(pairs)
        ensure_within_limit(
            counts, self.max_seq_length, contexts=[pair[0] for pair in pairs]
        )
        batch_size = (
            self.batch_size if self.batch_size is not None else batch_size_for(max(counts))
        )
        values = self._model.predict(
            [list(pair) for pair in pairs],
            batch_size=batch_size,
            activation_fn=self.activation_fn,
            show_progress_bar=False,
        )
        return [float(v) for v in values]
