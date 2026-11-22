"""The parts of the Rank1 adapter that do not need the model loaded.

Scoring itself is a generation and is exercised by examples/faithfulness.py against
the real weights; what is checked here is the arithmetic and the policy around it,
which is where this adapter actually went wrong while it was being written.
"""

from __future__ import annotations

import math

import pytest

from evrank.reasoning import Judgement, Rank1Scorer, rank1_prompt


def test_the_prompt_is_the_one_the_model_was_trained_on():
    prompt = rank1_prompt("- searched for: ramen", "A ramen bar.")
    assert prompt.startswith("Determine if the following passage is relevant")
    assert "Query: - searched for: ramen\n" in prompt
    assert "Passage: A ramen bar.\n" in prompt
    assert prompt.endswith("<think>")


def test_the_score_is_the_logit_difference_not_the_probability():
    judgement = Judgement(
        score=-7.0, trace="", complete=True, true_logit=19.25, false_logit=26.25
    )
    assert judgement.score == pytest.approx(judgement.true_logit - judgement.false_logit)


def test_the_probability_saturates_where_the_difference_does_not():
    """Why the raw difference is the score, measured on two real pairs.

    A ramen counter and an Italian restaurant sat 3.1 logits apart under the same
    evidence. As probabilities they are both indistinguishable from zero, and the
    margin this package attributes is the thing that got squashed.
    """
    ramen = Judgement(score=-7.0, trace="", complete=True, true_logit=19.25, false_logit=26.25)
    pasta = Judgement(
        score=-10.125, trace="", complete=True, true_logit=19.0, false_logit=29.125
    )
    assert ramen.score - pasta.score == pytest.approx(3.125)
    assert ramen.probability - pasta.probability < 0.001
    assert ramen.probability > pasta.probability


def test_the_probability_matches_the_model_card_formula():
    judgement = Judgement(score=1.0, trace="", complete=True, true_logit=1.0, false_logit=0.0)
    expected = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert judgement.probability == pytest.approx(expected)
    # Which is to say a sigmoid of the score.
    assert judgement.probability == pytest.approx(1 / (1 + math.exp(-judgement.score)))


def test_a_trace_that_ran_out_says_so_rather_than_being_discarded():
    cut = Judgement(score=0.5, trace="half a thought", complete=False, true_logit=1.0, false_logit=0.5)
    assert cut.complete is False
    assert cut.score == 0.5


def test_traces_are_kept_only_when_asked_for():
    """The dictionary grows with the ablation, so it is off unless a caller wants it.

    Checked on the attribute rather than through the model: what matters is that a
    scorer under an `Ablator` is not accumulating a trace per evidence subset by
    default.
    """
    assert "keep_traces" in Rank1Scorer.__init__.__annotations__
    assert Rank1Scorer.__init__.__defaults__[-1] is False
