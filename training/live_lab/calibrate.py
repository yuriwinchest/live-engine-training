"""Escolha dos limiares de rejeição na validação (research R4 da 002).

Regra do PO (opção A): entre os limiares que alcançam o acerto mínimo, vence o de **menor taxa de número
errado**; empate → maior acerto. Se nenhum alcança o acerto, vence o de menor erro mesmo assim, e o
relatório diz que a meta não foi cumprida.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from live_lab.decode import Decision, Thresholds
from live_lab.metrics import Outcome, Rates, rates


@dataclass(frozen=True, slots=True)
class Calibration:
    thresholds: Thresholds
    rates: Rates
    target_met: bool


def apply(decisions: Sequence[Decision], thresholds: Thresholds) -> list[int | None]:
    return [
        d.best
        if d.best is not None
        and d.posterior >= thresholds.min_posterior
        and d.margin >= thresholds.min_margin
        and d.adherence >= thresholds.min_adherence
        else None
        for d in decisions
    ]


def _grid(values: Sequence[float], steps: int) -> list[float]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return [0.0]
    quantiles = np.quantile(finite, np.linspace(0.0, 0.5, steps))
    return sorted({float(q) for q in quantiles})


def calibrate(
    decisions: Sequence[Decision], expected: Sequence[int | None], min_accuracy: float, steps: int = 12
) -> Calibration:
    if len(decisions) != len(expected) or not decisions:
        raise ValueError("decisões e rótulos precisam ter o mesmo tamanho, maior que zero")
    accepted = [d for d in decisions if d.best is not None]
    posteriors = _grid([d.posterior for d in accepted], steps)
    margins = _grid([d.margin for d in accepted], steps)
    adherences = [-np.inf, *_grid([d.adherence for d in accepted], steps)]

    best: tuple[tuple[float, float], Thresholds, Rates] | None = None
    best_met: tuple[tuple[float, float], Thresholds, Rates] | None = None
    for posterior, margin, adherence in product(posteriors, margins, adherences):
        thresholds = Thresholds(posterior, margin, adherence)
        result = rates([Outcome(e, g) for e, g in zip(expected, apply(decisions, thresholds), strict=True)])
        key = (result.wrong_rate, -result.accuracy)
        if best is None or key < best[0]:
            best = (key, thresholds, result)
        if result.accuracy >= min_accuracy and (best_met is None or key < best_met[0]):
            best_met = (key, thresholds, result)
    chosen = best_met or best
    assert chosen is not None
    return Calibration(chosen[1], chosen[2], best_met is not None)
