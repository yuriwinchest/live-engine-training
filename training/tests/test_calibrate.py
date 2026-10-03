from __future__ import annotations

import pytest

from live_lab.calibrate import calibrate
from live_lab.decode import Decision
from live_lab.metrics import Outcome, rates


def decision(best: int | None, posterior: float) -> Decision:
    return Decision(best, best, posterior, posterior * 10, -0.1)


def test_rates_count_wrong_reject_and_silence() -> None:
    result = rates([Outcome(5, 5), Outcome(5, 6), Outcome(5, None), Outcome(None, None), Outcome(None, 3)])
    assert result.n == 5
    assert result.accuracy == pytest.approx(2 / 5)
    assert result.wrong_rate == pytest.approx(2 / 5)
    assert result.reject_rate == pytest.approx(1 / 5)


def test_prefers_lower_wrong_rate_over_accuracy() -> None:
    """Opção A do PO: entre limiares que cumprem o acerto, o de menor erro vence."""
    decisions = [decision(1, 0.99)] * 90 + [decision(2, 0.40)] * 5 + [decision(3, 0.60)] * 5
    expected: list[int | None] = [1] * 90 + [9] * 5 + [3] * 5
    result = calibrate(decisions, expected, min_accuracy=0.90)
    assert result.target_met
    assert result.rates.wrong_rate == 0.0
    assert result.rates.accuracy == pytest.approx(0.95)


def test_reports_unmet_target() -> None:
    decisions = [decision(1, 0.9)] * 5 + [decision(2, 0.9)] * 5
    result = calibrate(decisions, [1] * 5 + [3] * 5, min_accuracy=0.99)
    assert not result.target_met


def test_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        calibrate([decision(1, 0.9)], [], min_accuracy=0.9)
