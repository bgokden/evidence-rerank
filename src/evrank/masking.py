"""How omitted evidence is represented in the context the model sees.

Ablation has to put a question to the model that was never asked in production:
what would it have scored without this fact? The strategies here differ in what
they hold fixed. `Omission` drops the line, which is the smaller edit but also
shortens the query, so part of any measured effect is the query getting shorter
rather than the content going away. `NeutralFiller` holds the line count and
roughly the length and changes only content, at the cost of putting a sentence in
the context that the user's history never contained.

`KindOnly` sits between them, for context whose language is unknown or mixed: it
keeps the line and the event kind and drops the content, so the shape survives
without any new words being introduced in any language.

None of them is neutral. Running more than one and comparing is the honest option,
which is why the strategy is a parameter rather than a decision baked into the
ablation.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Protocol

from evrank.evidence import EvidenceSet

DEFAULT_FILLER_TEXT = "a routine entry, nothing specific noted"


class Masker(Protocol):
    """Turns an evidence set and a set of removed units into one context string."""

    def context(self, evidence: EvidenceSet, removed: Sequence[str]) -> str: ...


@dataclasses.dataclass(frozen=True)
class Omission:
    """Removed evidence leaves no trace, so the context gets shorter."""

    def context(self, evidence: EvidenceSet, removed: Sequence[str]) -> str:
        return evidence.without(removed).render()


@dataclasses.dataclass(frozen=True)
class NeutralFiller:
    """Removed evidence keeps its line and its kind, and loses only its content.

    What is ablated is therefore the substance of a fact and not the fact of its
    existence: the context still says the user searched for something and reserved
    something, but no longer what. That keeps the line count and roughly the length,
    so the measured effect is not confounded with the query getting shorter, and it
    gives a baseline context of the right size rather than a bare header. The latter
    matters for a ranker whose query is the context, since an empty query makes every
    candidate look irrelevant and anchors the attribution to a point the model never
    sees in production.

    The replacement text must share no wording with the candidates, or it contributes
    an effect of its own.
    """

    text: str = DEFAULT_FILLER_TEXT

    def context(self, evidence: EvidenceSet, removed: Sequence[str]) -> str:
        dropped = evidence.require_units(removed)
        lines = evidence.header_lines()
        lines.extend(
            dataclasses.replace(item, text=self.text).render()
            if item.unit in dropped
            else item.render()
            for item in evidence.items
        )
        return "\n".join(lines)


def baseline_word_counts(evidence: EvidenceSet, masker: Masker) -> tuple[int, int]:
    """Word counts of the unablated context and of the fully masked baseline.

    Filler text much longer or shorter than the facts it replaces changes the query's
    size as well as its content, which is the confound this strategy exists to remove.
    Compare the two numbers and tune the filler until they are close; under `Omission`
    the second number is just the header, which is the gap being measured.
    """
    return (
        len(evidence.render().split()),
        len(masker.context(evidence, evidence.units()).split()),
    )


@dataclasses.dataclass(frozen=True)
class KindOnly:
    """Removed evidence keeps its line and kind, and loses its content entirely.

    The strategy for context in an unknown or mixed language. `NeutralFiller` has to
    put a sentence in the context, and that sentence is written in some language,
    which is a guess about input the caller may not be able to make. This writes
    nothing: the line reads `- searched for:` with the search gone, so the model
    still sees that an event of that kind occurred and sees no new vocabulary. The
    kind itself was already in the context, so no word appears that was not there.

    Line count is preserved, but length is not: a long fact becomes a short line, so
    some of the measured effect is the context getting shorter, as with `Omission`.
    """

    def context(self, evidence: EvidenceSet, removed: Sequence[str]) -> str:
        dropped = evidence.require_units(removed)
        lines = evidence.header_lines()
        lines.extend(
            "- %s:" % item.kind if item.unit in dropped else item.render()
            for item in evidence.items
        )
        return "\n".join(lines)
