"""Token budget policy for transformer rankers.

A cross-encoder encodes the pair as one sequence, `[CLS] context [SEP] document
[SEP]`, so the whole evidence context competes with the candidate's own text for a
single fixed budget. There is no per-evidence allowance: thirty facts and a verbose
candidate description draw on the same limit.

Exceeding it is worse here than in ordinary retrieval. The tokenizer truncates the
longer side, which is always the context, so the evidence at the end of the context
never reaches the model. Its measured effect is then exactly zero, which is
indistinguishable from evidence the model saw and ignored. Attribution over a
truncated context is not approximate, it is wrong in a way that reads as a finding,
so this package refuses it instead of truncating.
"""

from __future__ import annotations

from collections.abc import Sequence

# Attention cost grows with the square of sequence length, so a batch that fits
# comfortably at 512 tokens will exhaust GPU memory at 8192. Holding tokens-per-batch
# roughly constant keeps the memory envelope stable as contexts grow.
ATTENTION_TOKEN_BUDGET = 16384

# Past this, batches stop buying throughput and only raise the memory floor.
MAX_BATCH_SIZE = 64


class ContextTooLongError(ValueError):
    """Raised when a pair would be silently truncated by the model's tokenizer."""


def batch_size_for(longest_pair_tokens: int) -> int:
    """A batch size scaled to the pairs actually being scored.

    Keyed to the longest pair in the call rather than the model's ceiling, because
    the two can differ by orders of magnitude: a 40960-token model scoring 130-token
    contexts has no reason to batch one at a time. The longest pair sets the bound
    because every sequence in a batch is padded to it.

    Gives 32 at 512 tokens, 2 at 8192, and the cap for anything short. Pass an
    explicit batch size to a scorer to override it.
    """
    if longest_pair_tokens < 1:
        raise ValueError("token count must be positive, got %d" % longest_pair_tokens)
    return max(1, min(MAX_BATCH_SIZE, ATTENTION_TOKEN_BUDGET // longest_pair_tokens))


def ensure_within_limit(
    token_counts: Sequence[int], limit: int, contexts: Sequence[str] | None = None
) -> None:
    """Raise if any pair would be truncated, naming the worst offender.

    `contexts` is optional and used only to make the message actionable.
    """
    over = [i for i, count in enumerate(token_counts) if count > limit]
    if not over:
        return
    worst = max(over, key=lambda i: token_counts[i])
    detail = ""
    if contexts is not None and worst < len(contexts):
        # Every context opens with the same header, so an excerpt has to run past it
        # to say anything about which one overflowed.
        excerpt = " / ".join(contexts[worst].splitlines())
        if len(excerpt) > 120:
            excerpt = excerpt[:117] + "..."
        detail = "\n  longest context begins: %s" % excerpt
    raise ContextTooLongError(
        "%d of %d pairs exceed the model's %d-token limit (longest is %d tokens). "
        "Truncation would drop evidence from the end of the context and report its "
        "effect as exactly zero, so scoring is refused. Group the evidence into "
        "fewer units, shorten the candidate text, or use a longer-context "
        "re-ranker.%s" % (len(over), len(token_counts), limit, token_counts[worst], detail)
    )
