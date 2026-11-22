"""Why an item is on the page at all, rather than why it beat the item below it.

The rest of this package attributes a *margin*: the pairwise quantity a rank is
made of. A product explains itself differently -- "Because you searched for ramen
restaurants in Paris" -- and that sentence is not a rewording of a margin. It claims
the evidence is why this item is here, which is a claim about the item's own score.

Measured on a real slate, the item's own score alone is too blunt to say that. One
search for "ramen restaurants in Paris" lifts every Parisian restaurant, the Italian
one included (+1.79 on its own score), so the strongest cause is the same sentence
for every row and explains nothing about any of them.

Subtracting what a unit did for the rest of the slate fixes that, and the result is
specific enough to be worth showing: on the worked example the ramen search goes to
the ramen counter, the hotel in the 1st arrondissement goes to the izakaya in the
1st, and "mid-range budget" goes to the one priced in the middle. The model held
those associations; centring is what reads them back out.

The cost is one context per unit, scored against every candidate -- so units + 1
contexts, where pairwise leave-one-out needs units + 2 but only two candidates. In
exchange every row gets its reason from one pass instead of one pass per pair.

`standout` is the default here, and that is a decision about the question rather
than about estimator quality. It removes one unit from the user's real context and
reads the difference between two scores the model actually produced, which is what
"because you searched for X" asserts. `exact_standout` and `sampled_standout` average
a unit's contribution over every order the history could have arrived in, which
apportions credit between units more fairly and is the right tool when duplicate
evidence would otherwise cancel itself out -- but the coalitions it averages over are
contexts the user never had, so it answers a question the sentence is not making a
claim about. Group genuine duplicates instead, which is what `Evidence.group` is for.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import random
import statistics
from collections.abc import Mapping, Sequence

from evrank.ranking import Ranking
from evrank.scoring import Ablator, removal_key


@dataclasses.dataclass(frozen=True)
class StandoutEffect:
    """What one evidence unit did for one candidate."""

    unit: str
    label: str
    delta: float
    """Effect on this candidate's own score: its score now, minus its score with the
    unit removed. Positive means the evidence raised it."""
    lift: float
    """What the same unit did for the average of the *other* candidates on the
    slate. The candidate being explained is left out, so a unit that moved only this
    row has a lift of zero however long the slate is."""

    @property
    def standout(self) -> float:
        """How much of the effect was specific to this candidate.

        This is the number to rank reasons by. `delta` says the evidence helped;
        `standout` says it helped *this* one more than it helped the others, which
        is what makes a reason worth reading.
        """
        return self.delta - self.lift

    @property
    def specificity(self) -> float:
        """The share of what this evidence did that was specific to this candidate.

        `standout` divided by `delta`, so 1.0 means the evidence did nothing for the
        rest of the slate and 0.0 that it did exactly as much for everyone. Being a
        ratio of two differences in the same model's scores, it compares across
        candidates and across models where the raw sizes do not, which is what makes
        it usable as a floor. Zero when the evidence did not help this candidate at
        all, since a share of nothing is not a quantity worth reporting, and above
        1.0 when the evidence actively cost the rest of the slate, which is the
        strongest form of singling one row out.
        """
        if self.delta <= 0.0:
            return 0.0
        return self.standout / self.delta


@dataclasses.dataclass(frozen=True)
class CandidateStandout:
    candidate: str
    score: float
    method: str
    effects: tuple[StandoutEffect, ...]
    scale: float = 0.0
    """The median effect size across the whole request, as a unit of measurement.

    A logit is not comparable between requests. Measured on content written out in
    eleven languages, every per-request statistic -- this one, a robust dispersion of
    the effects, the largest effect, and the spread of the scores themselves -- varies
    by more than twentyfold between the weakest and the strongest language for the
    same meaning. A floor written as a constant is therefore a floor tuned to
    whichever request it was chosen on.

    This is reported rather than acted on, and that is a measured decision rather
    than a gap. Flooring in multiples of a per-request scale has lost three times --
    against this median, against a robust dispersion of the effects, and against a
    dispersion of the ranking scores, which is free and predicts the request's
    largest effect with an R-squared of 0.71. See `favouring` for why: what differs
    between languages is the signal-to-noise ratio, and rescaling moves along a
    trade-off curve without bending it.

    It stays because it tells a caller comparing two requests that one of them is
    being measured on a scale an order of magnitude smaller, which a raw effect size
    alone hides.
    """

    def by_standout(self) -> tuple[StandoutEffect, ...]:
        """Effects most specific to this candidate first."""
        return tuple(sorted(self.effects, key=lambda effect: -effect.standout))

    def favouring(
        self,
        min_specificity: float = 0.0,
        min_effect: float = 0.0,
        min_share: float = 0.0,
        min_standout: float = 0.0,
    ) -> tuple[StandoutEffect, ...]:
        """Only the effects worth printing as a reason, strongest first.

        Helping and being specific are both required. A unit that raised every
        candidate equally did not single this one out, and a unit that lowered this
        candidate's own score is not why it is here however badly it treated the
        others.

        That second requirement was tested rather than assumed, because there is a
        real argument against it: when the evidence is a paraphrase rather than a
        quotation, adding it to the context lowers every candidate's score, and the
        right answer is the one it lowered least. Measured, that case exists and is
        rarer than the harm. Dropping the requirement, with the floor moved onto
        `standout` and both matched to the same false-positive rate on the row the
        product shows, changed recall by nothing the data could resolve and cost
        10.3 to 10.9 points of precision, with the bootstrap interval excluding zero
        at every operating point tried. It stays.

        Two floors, because they catch different kinds of worthless reason and
        neither catches the other.

        `min_specificity` floors `specificity`: a unit can lift this candidate by
        +0.40 while lifting the whole slate by +0.396, and citing it says nothing
        about this row. This is the gate that does the quality work -- see below.

        `min_standout` floors `standout`, the centred effect, and it is the floor that
        does most of the work. It is the one to choose from.

        `min_effect` floors `delta`, the effect on this candidate's own score. It is
        kept as a small backstop because the two catch different things and neither
        catches the other. A standout floor misses the unit that barely touched this
        row while hurting the others -- its lift is negative, so centring leaves it
        looking healthy, and `specificity` passes it too since the ratio is enormous.
        A delta floor misses the on-topic request that lifted the whole slate, which
        has a large `delta` and almost no `standout`. Measured on held-out cases at
        matched false positives, keeping `delta` floored at anything up to 0.5
        alongside the standout floor costs nothing at all -- precision moved from
        60.8% to 60.4% -- so both stay.

        Which of the two should carry the floor was got wrong twice before it was
        measured on a corpus containing the failure mode. Flooring `standout` was
        tried on a corpus whose only distractors were off-topic junk ("winter tyres
        for an estate car") and was no better, because there were no slate-lifting
        units for it to catch. With half the distractors now on-topic requests for
        things absent from the page, flooring `standout` with specificity at 0.3 beats
        flooring `delta` with specificity at 0.1 by 8.3 to 11.6 points of mean recall
        and 7.5 to 15.7 points of precision, for both the base model and a fine-tuned
        one, at two operating points, with every bootstrap interval excluding zero.
        A floor is only as good as the corpus it was chosen on.

        What the floor buys is precision, and it is paid for in recall. Swept over
        1100 cases in eleven languages whose causes are paraphrases rather than
        quotations, whose labels have been checked by two sentence encoders unrelated
        to any re-ranker here, and half of whose distractors are on-topic requests
        for things absent from the page. `min_standout` on the left, with `delta`
        floored at 0.1 and specificity at 0.3 throughout:

            standout  p20 language   precision   on the first row
            0.10             47%        22.7%          67.2%
            0.25             37%        26.3%          57.2%
            0.50             21%        34.8%          38.1%
            0.75             15%        48.7%          24.7%
            1.00              0%        59.7%          16.1%

        The same sweep on a model carrying the adapter from `examples/finetune.py`
        reads 75%/36%/62% at 0.10 and 53%/89%/5% at 0.75, which is the clearest
        statement of what training the model buys that flooring cannot.

        An older version of this docstring gave the equivalent table for a floor on
        `delta` and is superseded; the numbers there were measured before the corpus
        contained anything for a centred floor to catch.

            floor   p20 language   precision   invented   on the first row
            0.00           42%        22.7%      56.3%          94.0%
            0.10           26%        34.0%      25.8%          50.0%
            0.25           15%        38.0%      17.4%          33.2%
            0.50           12%        40.8%      12.3%          24.3%
            1.00            3%        45.8%       7.5%          15.6%
            2.00            0%        48.0%       3.3%           6.9%

        Precision roughly doubles across the range and the share of first-ranked rows
        that speak without a cause falls by a factor of fourteen, so this is a real
        quality control and not only a volume control. But the weakest languages pay
        for it: between 0 and 0.10 the trade is 11 points of precision for 16 points
        of 20th-percentile recall, and past 0.25 it turns bad -- 8 more points of
        precision for 23 more points of recall. The default sits at 0.10 for that
        reason, and the last column is the one to choose from: how many first-ranked
        rows with no cause a surface is willing to let speak.

        That table has been rebuilt three times, and the first two versions said
        something different. Both were wrong for corpus reasons rather than
        measurement reasons, which is the argument for reading
        `examples/evaluation.py`'s separation check and its no-model baseline before
        any number it prints.

        An earlier version of this docstring claimed the floor sat in a gap, "every
        reason worth printing had a delta above 0.31 and every junk one below 0.04".
        That was measured on a corpus whose planted evidence was the candidate's own
        words copied into the history, where a character-trigram matcher scored 100
        percent and invented nothing -- so the gap was a property of the corpus, not
        of the problem. Measured again over 32,170 junk effects on paraphrased
        evidence, the junk 90th percentile is 0.788 and the distributions overlap
        heavily (the planted effect beats a junk effect about 71 percent of the
        time). There is no gap. Tune the floor, expect to re-tune it per model and
        per corpus, and read `examples/evaluation.py`'s separation check first: if
        the two distributions do not separate, no floor exists to be found.

        `min_share` is the dynamic floor: how much of the largest effect on this
        candidate, in either direction, a unit has to carry. It is off by default and
        it stays off. The docstring used to say that the corpus it was measured on
        could not separate good reasons from wrong ones and that another might; that
        has now been tried on one built for the purpose, with on-topic decoys, and it
        does not pay. At matched first-row false positives on held-out cases,
        precision falls from 45.0% to 42.5% on the base model and from 61.9% to 59.4%
        on a fine-tuned one, at every setting from 0.1 to 0.75. The earlier claim
        that good reasons spanned 18 to 100 percent of their row and wrong ones 66 to
        100 came from the copied-words corpus and should not be relied on either.
        Measured twice, never paid: this is a candidate for removal.

        A floor on a logit is a floor tuned to whichever request it was chosen on,
        and that is a real cost rather than a theoretical one. Measured on content
        written out in eleven languages, every per-request statistic -- the median
        effect, its dispersion, the largest effect, the spread of the scores
        themselves -- varies by a factor above twenty between the weakest and
        strongest language, so a constant floor sits at a different place in each.

        Expressing the floor in multiples of the request's own scale is the obvious
        remedy and it has now lost three separate times: against the median effect,
        against a robust dispersion of the effects, and against a dispersion of the
        ranking scores, which is free, is not computed from the matrix being judged,
        and predicts the request's largest effect with an R-squared of 0.71. At
        matched false positives on the row the product shows, the absolute floor held
        the 20th-percentile language at 12 percent against the normalised rule's 8,
        with 30 percent precision against 23. Treat the question as settled.

        The reason it loses is worth stating, because it is not the one assumed here
        before. The languages whose effects are small in logits are also the ones
        whose signal is small *relative to their own noise*, and a monotone rescaling
        slides the operating point along one trade-off curve rather than bending it.
        Rescaling can move which language gets the benefit of the doubt; it cannot
        give any of them a better exchange rate. Equalising the operating point per
        language is a product decision about who gets shown fewer sentences, not a
        correction to a measurement error.

                All three default to off; the service sets them.
        """
        return tuple(
            effect
            for effect in self.by_standout()
            if effect.standout > 0.0
            and effect.standout >= max(min_standout, 0.0)
            and effect.delta >= max(min_effect, 0.0)
            and effect.delta > 0.0
            and effect.specificity >= min_specificity
            and abs(effect.delta) >= min_share * self._largest_effect()
        )

    def _largest_effect(self) -> float:
        """The biggest thing any evidence did to this candidate, in either direction."""
        return max((abs(effect.delta) for effect in self.effects), default=0.0)


LEAVE_ONE_OUT = "leave_one_out"
SHAPLEY = "shapley"
SAMPLED_SHAPLEY = "sampled_shapley"

# Above this, 2^n enumeration stops being affordable and sampling takes over.
MAX_EXACT_UNITS = 12


def standout(ablator: Ablator) -> dict[str, CandidateStandout]:
    """Per-candidate evidence effects, centred across the slate.

    One wave of ablations covering every unit, so this is a single batched call
    whatever the slate holds. Needs at least two candidates: with one there is
    nothing to stand out from.

    Each number is a difference between two scores the model really produced, on the
    user's real context and on that context with one unit removed. Nothing is averaged
    and no coalition is invented, which is why this is the default for a sentence that
    claims the evidence is why the item is here.

    It does measure each unit alone, so duplicate evidence cancels itself out: two
    searches carrying the same information each come out near zero because the other
    covers for it, and neither gets cited. Group them -- that is what `Evidence.group`
    is for -- or use `sampled_standout`, understanding that it answers the other
    question.

    What this is *not* is wrong when the answer is unflattering. On the five-candidate
    example it credits a French bistro to a search for "ramen restaurants in Paris",
    and that is a true statement about the model: dropping that search really does
    cost the bistro more (-2.52) than it costs the ramen counter (-2.20), because the
    counter has other Japanese evidence holding it up and the bistro has nothing but
    "restaurants in Paris". Replacing that reading with a Shapley average hides a
    finding about the ranker behind a smoother number.
    """
    units = ablator.evidence.units()
    _require_a_slate(ablator)
    if not units:
        return _nothing_to_attribute(ablator, LEAVE_ONE_OUT)

    ranked = ablator.rank_without([()] + [(unit,) for unit in units])
    base = ranked[()]
    deltas = {
        unit: {
            c.id: base.scores[c.id] - ranked[(unit,)].scores[c.id] for c in ablator.candidates
        }
        for unit in units
    }
    return _assemble(ablator, base, deltas, LEAVE_ONE_OUT)


def exact_standout(
    ablator: Ablator,
    max_units: int = MAX_EXACT_UNITS,
) -> dict[str, CandidateStandout]:
    """The same quantity with exact Shapley values, enumerating every coalition.

    Where `sampled_standout` is the right tool, this is the exact version of it at no
    extra cost. Sampling draws arrival orders but scores each distinct coalition once,
    so it never visits more than the 2^units coalitions enumeration visits; at four
    units both score the same sixteen contexts, and the only difference is that
    enumeration weights them by how often a random order would have produced them
    while sampling weights them by how often this particular draw did. Past the cap
    enumeration is not affordable and sampling is the only option.

    It answers the same question sampling does, which is not the question `standout`
    answers. See `sampled_standout` for why that matters.
    """
    _require_a_slate(ablator)
    units = ablator.evidence.units()
    n = len(units)
    if not units:
        return _nothing_to_attribute(ablator, SHAPLEY)
    if n > max_units:
        raise ValueError(
            "exact standout over %d units needs %d contexts; use sampled_standout "
            "for an estimate, or standout for the cheap one" % (n, 2**n)
        )

    subsets = [held for size in range(n + 1) for held in itertools.combinations(units, size)]
    rankings = ablator.rank_without([removal_key(set(units) - set(h)) for h in subsets])
    value = {
        frozenset(held): rankings[removal_key(set(units) - set(held))].scores
        for held in subsets
    }

    deltas: dict[str, dict[str, float]] = {}
    for unit in units:
        others = [other for other in units if other != unit]
        weighted: dict[str, list[tuple[float, float]]] = {c.id: [] for c in ablator.candidates}
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for coalition in itertools.combinations(others, size):
                before = value[frozenset(coalition)]
                after = value[frozenset(coalition) | {unit}]
                for candidate in ablator.candidates:
                    weighted[candidate.id].append(
                        (weight, after[candidate.id] - before[candidate.id])
                    )
        deltas[unit] = {
            cid: math.fsum(weight * margin for weight, margin in drawn)
            for cid, drawn in weighted.items()
        }
    return _assemble(ablator, rankings[removal_key(())], deltas, SHAPLEY)


def sampled_standout(
    ablator: Ablator,
    permutations: int = 32,
    seed: int = 0,
) -> dict[str, CandidateStandout]:
    """The same quantity, with each unit's effect taken as a Shapley value.

    A unit's marginal contribution is read at the point it arrives in a random order,
    so evidence that repeats itself splits the credit it shares instead of each copy
    measuring near nothing. That is the reason to reach for this: duplicate evidence
    that cannot be grouped, where `standout` would cite neither copy.

    It is not a more accurate `standout`, and treating it as one is a mistake this
    module made for a while. The average runs over every order the history could have
    arrived in, most of which the user never had, so where the two disagree neither is
    correcting the other -- they are answering different questions, and the sentence
    "because you searched for X" is a claim about the context the user actually has.

    The contexts are shared across the whole slate -- every ablated context is scored
    against every candidate in the same wave -- so this costs at most
    `permutations * (units + 1)` contexts and never more than the `2^units` that
    `exact_standout` visits, and it answers for every row rather than for one pair.
    """
    if permutations < 1:
        raise ValueError("permutations must be at least 1, got %d" % permutations)
    _require_a_slate(ablator)
    units = ablator.evidence.units()
    if not units:
        return _nothing_to_attribute(ablator, SAMPLED_SHAPLEY)

    rng = random.Random(seed)
    orders = []
    for _ in range(permutations):
        order = list(units)
        rng.shuffle(order)
        orders.append(order)

    needed = {frozenset(), frozenset(units)}
    for order in orders:
        grown: set[str] = set()
        for unit in order:
            grown.add(unit)
            needed.add(frozenset(grown))

    coalitions = sorted(needed, key=lambda held: (len(held), sorted(held)))
    rankings = ablator.rank_without(
        [removal_key(set(units) - set(coalition)) for coalition in coalitions]
    )
    value = {
        coalition: rankings[removal_key(set(units) - set(coalition))].scores
        for coalition in coalitions
    }

    drawn: dict[str, dict[str, list[float]]] = {
        unit: {c.id: [] for c in ablator.candidates} for unit in units
    }
    for order in orders:
        grown = set()
        for unit in order:
            before = value[frozenset(grown)]
            grown.add(unit)
            after = value[frozenset(grown)]
            for candidate in ablator.candidates:
                drawn[unit][candidate.id].append(
                    after[candidate.id] - before[candidate.id]
                )

    deltas = {
        unit: {
            cid: math.fsum(samples) / len(samples) for cid, samples in by_candidate.items()
        }
        for unit, by_candidate in drawn.items()
    }
    return _assemble(ablator, rankings[()], deltas, SAMPLED_SHAPLEY)


def _require_a_slate(ablator: Ablator) -> None:
    if len(ablator.candidates) < 2:
        raise ValueError("standing out needs at least two candidates to centre against")


def _nothing_to_attribute(ablator: Ablator, method: str) -> dict[str, CandidateStandout]:
    base = ablator.base()
    return {
        c.id: CandidateStandout(
            candidate=c.id, score=base.scores[c.id], method=method, effects=()
        )
        for c in ablator.candidates
    }


def _assemble(
    ablator: Ablator,
    base: Ranking,
    deltas: Mapping[str, Mapping[str, float]],
    method: str,
) -> dict[str, CandidateStandout]:
    """Centre each unit's effects across the slate and pack them per candidate."""
    # The average excludes the candidate being explained. Including it attenuates
    # every effect by (width-1)/width, which is a bias that varies with slate size
    # and nothing else: a unit that moved one row by D and no other read 0.67*D on a
    # three-row page and 0.90*D on a ten-row one, under the same floor. Excluding it
    # makes the comparison what the word says -- this row against the others -- and
    # lets `specificity` reach the 1.0 its own definition describes.
    others = len(ablator.candidates) - 1
    lift = {
        unit: {
            c.id: (math.fsum(by_candidate.values()) - by_candidate[c.id]) / others
            for c in ablator.candidates
        }
        for unit, by_candidate in deltas.items()
    }
    units: Sequence[str] = ablator.evidence.units()
    # One yardstick for the whole request, so every candidate is read against the
    # same noise level rather than against its own.
    spread = [
        abs(deltas[unit][c.id] - lift[unit][c.id])
        for unit in units
        for c in ablator.candidates
    ]
    scale = statistics.median(spread) if spread else 0.0
    return {
        c.id: CandidateStandout(
            candidate=c.id,
            score=base.scores[c.id],
            method=method,
            scale=scale,
            effects=tuple(
                StandoutEffect(
                    unit=unit,
                    label=ablator.evidence.label(unit),
                    delta=deltas[unit][c.id],
                    lift=lift[unit][c.id],
                )
                for unit in units
            ),
        )
        for c in ablator.candidates
    }
