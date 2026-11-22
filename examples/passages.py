"""How long a passage is allowed to be, shared by every dataset loader here.

This is one policy, not a per-dataset detail, and it is shared rather than copied
because the guards depend on it being identical everywhere. A guard exists to ask
whether training damaged ordinary ranking; if the guard trimmed differently from
the training slates it would be asking about lengths the model never trained at,
and a drop would be unattributable.
"""

from __future__ import annotations

# Passages run to a few thousand characters. Trimmed at a word boundary, because a
# step's cost is dominated by sequence length and the first sentences carry the
# relevance. This is a training-speed decision and it does change the data: at 500
# a slate of six passages plus its context exhausted 28 GB of activations on a 0.5B
# model, which is what gradient checkpointing and this number together fix.
MAX_PASSAGE = 300


def trim(text: str, limit: int = MAX_PASSAGE) -> str:
    """Collapse whitespace and cut at a word boundary, marking the cut."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    cut = collapsed[:limit].rsplit(" ", 1)[0]
    return cut + "..."
