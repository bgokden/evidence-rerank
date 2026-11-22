"""A reasoning reranker behind the same `Scorer` interface as the cross-encoder.

Rank1 writes a chain of thought about the pair and then answers "true" or "false",
and its relevance score is read from the logits of those two tokens. That makes it
useful here for a reason beyond ranking: it is a model that states its reasons, so
the reasons can be checked against what its score actually depends on. See
`examples/faithfulness.py`.

Two things differ from `cross_encoder.py` and both matter.

The score is read as `logit(true) - logit(false)` rather than as the probability the
model card computes from them. That probability is a sigmoid of this difference, and
it saturates hard: on one measured pair the difference between a ramen counter and an
Italian restaurant was -7.00 against -10.12, which the probability reports as 0.000911
against 0.000040. The ranking is the same and the margin is gone, and margins are what
this package attributes.

And it is slow. Each score is a generation of a few hundred tokens, about 1.3 to 2.2
seconds a pair on an RTX 5090 against roughly 0.4 milliseconds for a batched
cross-encoder pair. A leave-one-out explanation over 30 units and 10 candidates is 310
pairs, so minutes rather than seconds. That is the trade: this model can tell you why
it ranked something, and the ICLR 2026 finding is that the reasoning does not make the
ranking better. Use it to audit a fast ranker, not to replace one.

No extra install: transformers and torch arrive with sentence-transformers, so the
weights are the only download.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evrank.limits import ContextTooLongError

RANK1_MODEL = "jhu-clsp/rank1-0.5b"
THINK_CLOSE = "</think>"
TRUE_TOKEN = " true"
FALSE_TOKEN = " false"
# Long enough for the 0.5B model to close its own think section on real pairs, which
# it did in 271 tokens on the worked example. A trace cut off here still scores, and
# `Judgement.complete` says it was cut off.
DEFAULT_THINK_TOKENS = 640
PROMPT = (
    "Determine if the following passage is relevant to the query. "
    "Answer only with 'true' or 'false'.\n"
    "Query: %s\n"
    "Passage: %s\n"
    "<think>"
)


@dataclasses.dataclass(frozen=True)
class Judgement:
    """One pair scored, with the reasoning the model wrote to get there."""

    score: float
    trace: str
    """What the model said before answering. Its own account of its reasons, which is
    a claim to be checked rather than an explanation to be trusted."""
    complete: bool
    """False when the trace hit the token budget before the model closed it."""
    true_logit: float
    false_logit: float

    @property
    def probability(self) -> float:
        """The model card's relevance score: a sigmoid of the logit difference."""
        true_score = math.exp(self.true_logit)
        false_score = math.exp(self.false_logit)
        return true_score / (true_score + false_score)


def rank1_prompt(context: str, document: str) -> str:
    """The prompt the model was trained on, with the evidence context as the query."""
    return PROMPT % (context, document)


class Rank1Scorer:
    """Rank1 as a `Scorer`: deterministic, higher is better.

    Greedy decoding, so repeating a request repeats the trace and the score, which
    the `Scorer` protocol requires and which ablation depends on: a scorer that
    answered differently on the unablated context each time would make every
    attributed difference partly noise.

    `batch_size` defaults to 1 and should stay there for any use that subtracts one
    score from another. Batching is not merely imprecise here, it changes what the
    model thinks: padding and bfloat16 perturb the first greedy choice, and from there
    the chains diverge and never reconverge. Measured on three pairs, batching moved
    the trace from "I need to check if Tonkotsu ramen counter is a real place" to "I
    need to check if Tonkotsu is a real restaurant" and the score from -7.00 to -5.25,
    a difference several times larger than most effects this package attributes.
    One at a time, the same pair returned an identical trace and an identical score
    on every run. Raise it only for plain ranking, never behind an `Ablator`.
    """

    def __init__(
        self,
        model_name: str = RANK1_MODEL,
        max_think_tokens: int = DEFAULT_THINK_TOKENS,
        batch_size: int = 1,
        device: str | None = None,
        raw_scores: bool = True,
        keep_traces: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_think_tokens = max_think_tokens
        self.batch_size = batch_size
        self.raw_scores = raw_scores
        self.keep_traces = keep_traces
        self.traces: dict[tuple[str, str], Judgement] = {}
        """Every judgement made, by pair, when `keep_traces` is on.

        Off by default because it grows with the ablation: a scorer behind an
        `Ablator` is asked about one pair per evidence subset, and each trace is a
        kilobyte or two. On, it means a caller that ranked through `score` can read
        the reasoning afterwards without generating any of it a second time."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(self.device)
        self._model.eval()
        self.true_id = self._token_id(TRUE_TOKEN)
        self.false_id = self._token_id(FALSE_TOKEN)
        self.close_id = self._token_id(THINK_CLOSE)
        self.max_seq_length = int(self._model.config.max_position_embeddings)

    def _token_id(self, text: str) -> int:
        ids = self._tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError("%r is not a single token for %s" % (text, self.model_name))
        return int(ids[0])

    def token_counts(self, pairs: Sequence[tuple[str, str]]) -> list[int]:
        """Prompt length per pair, before any reasoning is generated."""
        prompts = [rank1_prompt(context, document) for context, document in pairs]
        return [
            len(ids) for ids in self._tokenizer(prompts, add_special_tokens=False).input_ids
        ]

    def judge(self, pairs: Sequence[tuple[str, str]]) -> list[Judgement]:
        """Score every pair and keep the reasoning each one was given."""
        if not pairs:
            return []
        self._ensure_room(pairs)
        judgements: list[Judgement] = []
        for start in range(0, len(pairs), self.batch_size):
            judgements.extend(self._judge_batch(pairs[start : start + self.batch_size]))
        if self.keep_traces:
            self.traces.update(zip((tuple(pair) for pair in pairs), judgements))
        return judgements

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [
            judgement.score if self.raw_scores else judgement.probability
            for judgement in self.judge(pairs)
        ]

    def _ensure_room(self, pairs: Sequence[tuple[str, str]]) -> None:
        """Refuse a prompt that leaves no room to reason, rather than truncating it.

        Same policy as the cross-encoder path, for the same reason: a silently
        truncated context reports the dropped evidence's effect as exactly zero.
        """
        budget = self.max_seq_length - self.max_think_tokens
        counts = self.token_counts(pairs)
        longest = max(counts)
        if longest > budget:
            raise ContextTooLongError(
                "prompt is %d tokens and %s leaves %d for it after reserving %d to "
                "reason with; shorten the evidence or lower max_think_tokens"
                % (longest, self.model_name, budget, self.max_think_tokens)
            )

    def _judge_batch(self, pairs: Sequence[tuple[str, str]]) -> list[Judgement]:
        prompts = [rank1_prompt(context, document) for context, document in pairs]
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        width = encoded.input_ids.shape[1]
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.max_think_tokens,
                do_sample=False,
                stop_strings=[THINK_CLOSE],
                tokenizer=self._tokenizer,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Generation stops when every row in the batch has closed its think section,
        # so a row that closed early has kept writing past it. Each row is cut back
        # to its own first close, or the whole budget when it never got there. (With
        # the default batch of one this is just the single row's own stop.)
        traces, complete = [], []
        for row in generated[:, width:]:
            closed = (row == self.close_id).nonzero()
            end = int(closed[0]) + 1 if len(closed) else len(row)
            traces.append(row[:end])
            complete.append(bool(len(closed)))

        return self._read_judgements(encoded, traces, complete)

    def _read_judgements(self, encoded, traces, complete):
        """One forward pass over prompt plus trace, reading the next token's logits.

        The judgement is whatever the model would say next, so an unclosed trace is
        scored at the point it ran out rather than discarded.
        """
        rows, masks = [], []
        for index, trace in enumerate(traces):
            prompt_ids = encoded.input_ids[index]
            prompt_mask = encoded.attention_mask[index]
            closing = (
                torch.tensor([], dtype=trace.dtype, device=trace.device)
                if complete[index]
                else torch.tensor([self.close_id], dtype=trace.dtype, device=trace.device)
            )
            rows.append(torch.cat([prompt_ids, trace, closing]))
            ones = torch.ones(len(trace) + len(closing), dtype=prompt_mask.dtype, device=prompt_mask.device)
            masks.append(torch.cat([prompt_mask, ones]))

        longest = max(len(row) for row in rows)
        pad = self._tokenizer.pad_token_id
        padded = torch.stack([
            torch.cat([torch.full((longest - len(row),), pad, dtype=row.dtype, device=row.device), row])
            for row in rows
        ])
        attention = torch.stack([
            torch.cat([torch.zeros(longest - len(mask), dtype=mask.dtype, device=mask.device), mask])
            for mask in masks
        ])
        # Positions have to be derived from the mask. Left padding means row 2 of a
        # batch may start twenty tokens in, and a model left to assume arange() reads
        # it at the wrong positions -- which showed up as the same pair scoring -5.25
        # batched and -7.00 alone, a difference far larger than anything the
        # attribution measures.
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            logits = self._model(
                input_ids=padded, attention_mask=attention, position_ids=positions
            ).logits[:, -1, :]

        judgements = []
        for index, trace in enumerate(traces):
            true_logit = float(logits[index, self.true_id])
            false_logit = float(logits[index, self.false_id])
            judgements.append(
                Judgement(
                    score=true_logit - false_logit,
                    trace=self._tokenizer.decode(trace, skip_special_tokens=True).strip(),
                    complete=complete[index],
                    true_logit=true_logit,
                    false_logit=false_logit,
                )
            )
        return judgements
