"""The training losses, against values worked out by hand.

Each loss is checked on a slate whose right answer is known by construction, the
same discipline the attribution math is tested with: a loss that is wrong in a way
the model can satisfy would train the wrong behaviour silently.
"""

from __future__ import annotations

import math

import pytest
import torch

from evrank.training import (
    anchor_loss,
    attribution_loss,
    margin_loss,
    parallel_loss,
    rank_loss,
    standout,
    total_loss,
)

# Three candidates. Removing the cause costs candidate 0 two points and the others
# nothing, which is the shape the attribution term is meant to reward.
DELTA = torch.tensor([[2.0, 0.0, 0.0]])
MASK = torch.tensor([[True, True, True]])
TARGET = torch.tensor([0])


def test_standout_subtracts_the_mean_over_the_other_candidates():
    # Candidate 0: 2.0 against the others' mean of 0.0, so all of it is specific.
    # Candidate 1: 0.0 against a mean of 1.0 over candidates 0 and 2.
    assert standout(DELTA, MASK).tolist() == [[2.0, -1.0, -1.0]]


def test_standout_ignores_padded_slots():
    delta = torch.tensor([[2.0, 0.0, 99.0]])
    mask = torch.tensor([[True, True, False]])
    # The padded 99.0 must not enter the lift: candidate 0 reads 2.0 - 0.0.
    assert standout(delta, mask)[0, 0].item() == pytest.approx(2.0)
    assert standout(delta, mask)[0, 1].item() == pytest.approx(-2.0)


def test_standout_does_not_shrink_as_the_slate_grows():
    """A unit that moved one row and no other reads the same at any slate size."""
    for width in (2, 5, 10):
        delta = torch.zeros((1, width))
        delta[0, 0] = 3.0
        mask = torch.ones((1, width), dtype=torch.bool)
        assert standout(delta, mask)[0, 0].item() == pytest.approx(3.0)


def test_attribution_loss_is_lowest_when_the_cause_hits_its_own_candidate():
    right = attribution_loss(DELTA, TARGET, MASK)
    wrong = attribution_loss(DELTA, torch.tensor([1]), MASK)
    assert right < wrong
    # Cross-entropy over [2, 0, 0] with the 2 as the target.
    expected = -math.log(math.exp(2) / (math.exp(2) + 2))
    assert right.item() == pytest.approx(expected, abs=1e-5)


def test_attribution_loss_is_indifferent_to_a_shared_shift():
    """Only the ordering across candidates matters, not the overall level."""
    shifted = attribution_loss(DELTA + 7.0, TARGET, MASK)
    assert shifted.item() == pytest.approx(attribution_loss(DELTA, TARGET, MASK).item())


def test_attribution_loss_never_prefers_a_padded_slot():
    delta = torch.tensor([[1.0, 50.0]])
    mask = torch.tensor([[True, False]])
    # With the 50.0 masked out, the only live candidate is the target and the loss
    # is essentially zero rather than enormous.
    assert attribution_loss(delta, torch.tensor([0]), mask).item() == pytest.approx(0.0)


def test_margin_loss_is_zero_once_the_effect_clears_the_floor():
    assert margin_loss(DELTA, TARGET, MASK, margin=1.5).item() == pytest.approx(0.0)
    # Standout of the target is 2.0, so a floor of 3.0 leaves a shortfall of 1.0.
    assert margin_loss(DELTA, TARGET, MASK, margin=3.0).item() == pytest.approx(1.0)


def test_margin_loss_reads_the_centred_effect_not_the_raw_one():
    """A unit that lifted the whole slate equally has no margin, however large."""
    delta = torch.tensor([[5.0, 5.0, 5.0]])
    assert margin_loss(delta, TARGET, MASK, margin=1.0).item() == pytest.approx(1.0)


def test_rank_loss_rewards_scoring_the_right_candidate_highest():
    scores = torch.tensor([[3.0, 1.0, 0.0]])
    assert rank_loss(scores, TARGET, MASK) < rank_loss(scores, torch.tensor([2]), MASK)


def test_anchor_loss_is_zero_only_when_nothing_moved():
    scores = torch.tensor([[1.0, 2.0, 3.0]])
    assert anchor_loss(scores, scores.clone(), MASK).item() == pytest.approx(0.0)
    moved = scores + torch.tensor([[2.0, 0.0, 0.0]])
    # One candidate off by 2.0, averaged over three live slots.
    assert anchor_loss(moved, scores, MASK).item() == pytest.approx(4.0 / 3.0)


def test_anchor_loss_ignores_padded_slots():
    scores = torch.tensor([[1.0, 0.0]])
    base = torch.tensor([[1.0, 99.0]])
    mask = torch.tensor([[True, False]])
    assert anchor_loss(scores, base, mask).item() == pytest.approx(0.0)


def test_parallel_loss_is_zero_when_a_translation_costs_the_same():
    other = torch.tensor([[0.0, 2.0, 0.0]])
    assert parallel_loss(
        DELTA, TARGET, MASK, other, torch.tensor([1]), MASK
    ).item() == pytest.approx(0.0)


def test_parallel_loss_grows_with_the_gap_between_two_languages():
    weak = torch.tensor([[0.4, 0.0, 0.0]])
    gap = parallel_loss(DELTA, TARGET, MASK, weak, TARGET, MASK)
    # Both are wholly specific, so standout equals delta: 2.0 against 0.4.
    assert gap.item() == pytest.approx(1.6, abs=1e-5)


def test_all_weights_zero_reproduces_doing_nothing():
    scores = torch.tensor([[3.0, 1.0, 0.0]], requires_grad=True)
    total, parts = total_loss(
        scores, torch.zeros_like(scores), scores.detach(), TARGET, MASK, weights={}
    )
    assert total.item() == pytest.approx(0.0)
    assert set(parts) == {"rank", "attribution", "margin", "anchor"}
    total.backward()
    assert scores.grad is None or scores.grad.abs().sum().item() == pytest.approx(0.0)


def test_total_loss_reports_every_term_even_when_it_is_not_weighted():
    # Effects too small to clear the 1.5 floor, so the margin term is live even
    # though nothing is training it.
    scores = torch.tensor([[1.0, 0.9, 0.8]])
    total, parts = total_loss(
        scores, torch.zeros_like(scores), scores, TARGET, MASK,
        weights={"attribution": 1.0},
    )
    assert parts["margin"] > 0.0, "reported so a sweep can see what it is not training"
    assert parts["anchor"] == pytest.approx(0.0)
    assert total.item() == pytest.approx(parts["attribution"])
