from __future__ import annotations

import pytest

from live_lab.manifest import Split
from live_lab.split import assign_splits, pseudonym


def test_pseudonym_is_stable_and_hides_original() -> None:
    first = pseudonym("common-voice", "client-abc123")
    assert first == pseudonym("common-voice", "client-abc123")
    assert "abc123" not in first
    assert first != pseudonym("outra-fonte", "client-abc123")


def test_real_voice_goes_to_test_only() -> None:
    speakers = [(f"syn-{i}", False) for i in range(20)] + [("po", True)]
    assignment = assign_splits(speakers, seed=1)
    assert assignment["po"] is Split.TEST
    assert all(assignment[f"syn-{i}"] is not Split.TEST for i in range(20))
    assert sum(1 for s in assignment.values() if s is Split.VAL) == 2


def test_order_does_not_matter() -> None:
    speakers = [(f"syn-{i}", False) for i in range(30)]
    assert assign_splits(speakers, seed=5) == assign_splits(list(reversed(speakers)), seed=5)


def test_small_pool_still_gets_validation() -> None:
    assignment = assign_splits([("a", False), ("b", False), ("c", False)], seed=0)
    assert sorted(assignment.values()).count(Split.VAL) == 1


def test_speaker_cannot_be_real_and_synthetic() -> None:
    with pytest.raises(ValueError):
        assign_splits([("x", True), ("x", False)], seed=0)
