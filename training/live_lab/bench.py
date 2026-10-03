"""Medições do modelo exportado: tamanho, tempo por clipe com 1 thread e memória (SC-001, SC-009).

O tempo e a memória aqui são indicadores medidos no computador; o valor que vale é o do Samsung A11,
medido na feature do SDK Android.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

from live_lab.audio_io import SAMPLE_RATE
from live_lab.runtime import log_probs, open_session


@dataclass(frozen=True, slots=True)
class Bench:
    size_bytes: int
    latency_ms_3s_1thread: float
    rss_growth_mb: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def measure(model: Path, runs: int = 30) -> Bench:
    process = psutil.Process()
    before = process.memory_info().rss
    session = open_session(model, threads=1)
    clip = (0.05 * np.random.default_rng(0).standard_normal(3 * SAMPLE_RATE)).astype(np.float32)
    for _ in range(3):
        log_probs(session, clip)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        log_probs(session, clip)
        timings.append((time.perf_counter() - start) * 1000)
    growth = (process.memory_info().rss - before) / 2**20
    return Bench(model.stat().st_size, round(statistics.median(timings), 2), round(growth, 1))
