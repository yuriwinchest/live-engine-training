"""Progresso legível para etapas longas: quantos, quanto falta e a que velocidade."""

from __future__ import annotations

import sys
import time
from typing import TextIO


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} min {secs:02d} s" if minutes else f"{secs} s"


class Progress:
    def __init__(self, total: int, label: str, every_s: float = 5.0, stream: TextIO = sys.stderr) -> None:
        self.total = total
        self.label = label
        self.every_s = every_s
        self.stream = stream
        self.count = 0
        self.start = time.perf_counter()
        self._last = 0.0

    def tick(self, amount: int = 1) -> None:
        self.count += amount
        now = time.perf_counter()
        if now - self._last >= self.every_s or self.count >= self.total:
            self._last = now
            elapsed = now - self.start
            rate = self.count / elapsed if elapsed > 0 else 0.0
            left = (self.total - self.count) / rate if rate > 0 else 0.0
            share = self.count / self.total if self.total else 1.0
            print(
                f"{self.label}: {self.count}/{self.total} ({share:.0%}) · {rate:.1f}/s · "
                f"decorrido {_clock(elapsed)} · faltam ~{_clock(left)}",
                file=self.stream,
                flush=True,
            )
