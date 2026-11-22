"""The ship gate, checked without a GPU or a training run.

This is the most consequential decision in `examples/finetune.py` -- it is what keeps
a damaged adapter out of the service -- and until the decision was separated from the
training loop there was no way to exercise it. The cases below are the ones the rule
got wrong before it was re-specified: a small guard fall against a large in-domain
gain used to be refused, and an in-domain regression used to pass unnoticed because
the gate had no in-domain term at all.
"""

from __future__ import annotations

import pytest
from finetune import GUARD_TOLERANCE, Validation, refuse


def validation(ranking: float, attribution: float = 70.0) -> Validation:
    return Validation(attribution=attribution, effect=1.0, ranking=ranking)


def test_a_clean_run_ships():
    assert refuse(validation(50.0), validation(65.0), 0.0) is None


def test_a_small_guard_fall_is_the_price_of_specialising():
    """The case the old rule failed on.

    Folds 2 and 3 were refused over guard falls of 2.0 and 2.1 points while the same
    adapters gained twelve points or more in domain. A fall inside the tolerance is
    reported and not acted on.
    """
    assert refuse(validation(50.0), validation(65.0), 2.1) is None


def test_a_large_guard_fall_is_refused():
    reason = refuse(validation(50.0), validation(65.0), GUARD_TOLERANCE + 0.1)
    assert reason is not None
    assert "guard" in reason


def test_the_tolerance_boundary_is_inclusive():
    """Exactly at the tolerance still ships; the rule is 'more than'."""
    assert refuse(validation(50.0), validation(65.0), GUARD_TOLERANCE) is None


def test_an_in_domain_regression_is_refused_however_clean_the_guards():
    reason = refuse(validation(50.0), validation(49.9), 0.0)
    assert reason is not None
    assert "in-domain ranking" in reason


def test_in_domain_ranking_is_checked_before_the_guards():
    """Both terms fail, and the message names the disqualifying one.

    Ranking comes first deliberately: it is the adapter's own job, so it is the more
    informative thing to report when both are wrong.
    """
    reason = refuse(validation(50.0), validation(40.0), 20.0)
    assert reason is not None
    assert "in-domain ranking" in reason


def test_level_in_domain_ranking_is_not_a_regression():
    """Unchanged ranking clears. The rule refuses a fall, not a failure to improve;
    an adapter that holds ranking level while improving attribution is the trade this
    package exists to make."""
    assert refuse(validation(50.0), validation(50.0), 0.0) is None


def test_attribution_does_not_enter_the_gate():
    """Attribution is the stopping signal, not a gate term.

    It chose the checkpoint, so gating on it too would be scoring a decision against
    the quantity that made it.
    """
    assert refuse(validation(50.0, attribution=90.0),
                  validation(50.0, attribution=10.0), 0.0) is None


@pytest.mark.parametrize("fall", [0.0, 1.0, 2.1, GUARD_TOLERANCE])
def test_every_fall_measured_in_the_four_folds_clears(fall):
    """The largest guard fall observed across the four folds was 2.1 points."""
    assert refuse(validation(50.0), validation(60.0), fall) is None
