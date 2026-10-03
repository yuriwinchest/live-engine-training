"""Exemplos "sem número" feitos só de ruído (FR-012): o aperto do botão sem fala útil.

Frases faladas sem número vêm da síntese (planner.DISTRACTOR_PHRASES); aqui só se completa a cota
com ruído puro em volume variado.
"""

from __future__ import annotations

import numpy as np

from live_lab.audio_io import SAMPLE_RATE, Audio
from live_lab.augment import fit_length, microphone, power, prevent_clipping


def noise_only(noise: Audio, rng: np.random.Generator) -> Audio:
    length = int(rng.uniform(1.0, 3.0) * SAMPLE_RATE)
    segment = fit_length(noise, length, rng)
    level_db = rng.uniform(-35.0, -15.0)
    current = power(segment)
    if current > 0.0:
        segment = (segment * np.sqrt(10 ** (level_db / 10) / current)).astype(np.float32)
    colored = microphone(segment, rng.uniform(80.0, 200.0), rng.uniform(3400.0, 7000.0))
    return prevent_clipping(colored)
