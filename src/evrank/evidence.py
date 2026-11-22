"""The user context, kept as discrete records rather than only as embeddings.

Attribution at the evidence level is only possible if the original search events,
reservation facts and trip constraints survive next to whatever the ranker learned
from them. This module is that store, plus the one place where evidence is turned
into model input.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Iterable

# No header by default. A label like "Traveler context:" is the caller's domain and
# language, not evidence, and removing it changes nothing that matters: on the worked
# example the order is identical, the margin moves 1.465 to 1.422, and every effect
# keeps its sign. Set one if your model was tuned to expect it.
DEFAULT_HEADER = ""


@dataclasses.dataclass(frozen=True)
class Evidence:
    """One preserved fact about the user.

    `kind` is the caller's own wording for what the user did ("searched for",
    "reserved"). It is model input, it is what `KindOnly` masking keeps, and in a
    sentence it becomes the verb, so it reads best as a past action
    in the same language as `text`.

    `at` is metadata for the caller, not model input; see `render`.

    `group` names the ablation unit this fact belongs to. Facts that say the same
    thing (three searches for ramen) share a group so they are removed together;
    otherwise each one hides the others' effect and both look unimportant. Leave it
    unset unless the facts really are duplicates: grouping by topic hides the
    largest single effect behind its neighbours.
    """

    id: str
    kind: str
    text: str
    at: datetime.datetime | None = None
    group: str | None = None

    @property
    def unit(self) -> str:
        return self.group or self.id

    def render(self) -> str:
        """One line of model input: the kind, then the fact.

        The layout is a nuisance parameter, measured not assumed. Against
        `- kind text`, `kind: text`, `[kind] text`, `1. kind: text` and the text
        alone, the top of the ranking and the sign of every large effect hold; what
        moves is the margin (1.05 to 1.92 on the worked example) and the order of
        candidates already within noise of each other. So the layout is not worth
        tuning, but a number carried across two different layouts is not comparable.

        `at` is deliberately absent. Rendering it costs 22 percent more tokens and
        shrinks the margin (1.42 to 0.50 as an ISO date, 0.83 as "3 days ago")
        without changing the order or which evidence is decisive: this model reads
        dates as noise. Recency belongs in the caller's own weighting, or in `text`
        if the model should see it.
        """
        return "- %s: %s" % (self.kind, self.text)


@dataclasses.dataclass(frozen=True)
class EvidenceSet:
    """An ordered, immutable set of evidence that can render itself as model input."""

    items: tuple[Evidence, ...]
    header: str = DEFAULT_HEADER

    def __init__(self, items: Iterable[Evidence], header: str = DEFAULT_HEADER) -> None:
        object.__setattr__(self, "items", tuple(items))
        object.__setattr__(self, "header", header)
        seen = set()
        for item in self.items:
            if item.id in seen:
                raise ValueError("duplicate evidence id: %s" % item.id)
            seen.add(item.id)

    def __len__(self) -> int:
        return len(self.items)

    def units(self) -> tuple[str, ...]:
        """Ablation units in first-appearance order."""
        ordered: list[str] = []
        for item in self.items:
            if item.unit not in ordered:
                ordered.append(item.unit)
        return tuple(ordered)

    def label(self, unit: str) -> str:
        """What a unit says, in the caller's own words: its first fact's text.

        No suffix for the rest of the unit. A label is read by people and dropped
        into sentences, so "(and 2 more of the same kind)" would put English prose
        inside a Turkish explanation; `count` carries that number instead, where a
        caller can word it themselves.
        """
        for item in self.items:
            if item.unit == unit:
                return item.text
        raise KeyError(unit)

    def count(self, unit: str) -> int:
        """How many facts the unit holds, which is how many were removed together."""
        total = sum(1 for item in self.items if item.unit == unit)
        if not total:
            raise KeyError(unit)
        return total

    def phrase(self, unit: str) -> tuple[str, str]:
        """The unit's kind and text, for an explanation that names the action.

        `label` answers what the evidence said; this also answers what the user did,
        which is what the sentence leads with. The kind is the first fact's,
        since a unit holds facts of one kind.
        """
        for item in self.items:
            if item.unit == unit:
                return item.kind, item.text
        raise KeyError(unit)

    def require_units(self, units: Iterable[str]) -> set[str]:
        """Validate unit names and return them as a set, so a typo cannot silently
        become an ablation that removes nothing."""
        requested = set(units)
        unknown = requested - set(self.units())
        if unknown:
            raise KeyError("unknown evidence units: %s" % sorted(unknown))
        return requested

    def without(self, units: Iterable[str]) -> EvidenceSet:
        """A copy with every fact in the named units omitted."""
        dropped = self.require_units(units)
        return EvidenceSet(
            [item for item in self.items if item.unit not in dropped], header=self.header
        )

    def keeping(self, units: Iterable[str]) -> EvidenceSet:
        """A copy holding only the named units."""
        kept = set(units)
        return self.without(set(self.units()) - kept)

    def header_lines(self) -> list[str]:
        """The opening line, or none when the header is blank.

        The header is a label the caller supplies, not evidence, so an empty one
        has to leave no trace rather than a blank first line.
        """
        return [self.header] if self.header.strip() else []

    def render(self) -> str:
        """The query side of the ranker.

        Omission, not token masking: a dropped fact leaves no placeholder, so every
        ablated context is still a well-formed context of the kind the model saw in
        training. An empty set renders as the header alone, which is the no-context
        baseline.
        """
        lines = self.header_lines()
        lines.extend(item.render() for item in self.items)
        return "\n".join(lines)

