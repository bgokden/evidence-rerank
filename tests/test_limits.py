from __future__ import annotations

import pytest

from evrank.limits import (
    ATTENTION_TOKEN_BUDGET,
    MAX_BATCH_SIZE,
    ContextTooLongError,
    batch_size_for,
    ensure_within_limit,
)


def test_batch_size_shrinks_as_the_pairs_grow():
    assert batch_size_for(512) == 32
    assert batch_size_for(8192) == 2
    assert batch_size_for(512) * 512 == ATTENTION_TOKEN_BUDGET


def test_batch_size_never_drops_below_one():
    assert batch_size_for(ATTENTION_TOKEN_BUDGET * 4) == 1


def test_short_pairs_are_not_punished_for_a_long_context_model():
    # The regression this function exists to prevent: sizing from the model's
    # ceiling batched 130-token pairs two at a time on a 40960-token model.
    assert batch_size_for(130) > batch_size_for(40960)
    assert batch_size_for(130) == MAX_BATCH_SIZE


def test_batch_size_is_capped_for_tiny_pairs():
    assert batch_size_for(1) == MAX_BATCH_SIZE


def test_batch_size_rejects_a_nonsense_length():
    with pytest.raises(ValueError):
        batch_size_for(0)


def test_pairs_within_the_limit_pass():
    ensure_within_limit([10, 511, 512], 512)


def test_a_pair_over_the_limit_is_refused_not_truncated():
    with pytest.raises(ContextTooLongError, match="exceed the model's 512-token limit"):
        ensure_within_limit([10, 513], 512)


def test_the_message_names_the_worst_pair_and_how_many_failed():
    with pytest.raises(ContextTooLongError) as error:
        ensure_within_limit([900, 10, 1200], 512)
    message = str(error.value)
    assert "2 of 3 pairs" in message
    assert "longest is 1200 tokens" in message


def test_the_message_points_at_the_offending_context_when_given_one():
    with pytest.raises(ContextTooLongError, match="Traveler context:"):
        ensure_within_limit([600], 512, contexts=["Traveler context:\n- searched for: x"])


def test_the_excerpt_runs_past_the_shared_header():
    with pytest.raises(ContextTooLongError) as error:
        ensure_within_limit(
            [600], 512, contexts=["Traveler context:\n- searched for: ramen in Paris"]
        )
    assert "searched for: ramen in Paris" in str(error.value)
