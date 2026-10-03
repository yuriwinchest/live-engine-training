"""Métricas do ponto de vista do operador: acerto, número errado (o pior erro) e rejeição.

Um clipe sem número que termina rejeitado conta como acerto; qualquer número devolvido para ele é erro.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Outcome:
    expected: int | None
    got: int | None

    @property
    def correct(self) -> bool:
        return self.got == self.expected

    @property
    def wrong(self) -> bool:
        return self.got is not None and self.got != self.expected

    @property
    def rejected(self) -> bool:
        return self.got is None and self.expected is not None


@dataclass(frozen=True, slots=True)
class Rates:
    n: int
    accuracy: float
    wrong_rate: float
    reject_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "wrong_rate": self.wrong_rate,
            "reject_rate": self.reject_rate,
        }


def rates(outcomes: Sequence[Outcome]) -> Rates:
    n = len(outcomes)
    if n == 0:
        return Rates(0, 0.0, 0.0, 0.0)
    return Rates(
        n=n,
        accuracy=sum(o.correct for o in outcomes) / n,
        wrong_rate=sum(o.wrong for o in outcomes) / n,
        reject_rate=sum(o.rejected for o in outcomes) / n,
    )


def by_group(
    items: Iterable[T], key: Callable[[T], str], outcome: Callable[[T], Outcome]
) -> dict[str, Rates]:
    groups: dict[str, list[Outcome]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(outcome(item))
    return {name: rates(group) for name, group in sorted(groups.items())}
