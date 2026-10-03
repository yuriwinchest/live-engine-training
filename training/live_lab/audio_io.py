"""Entrada e saída de áudio no formato único do projeto: WAV PCM16, 16 kHz, mono (research R8)."""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

SAMPLE_RATE: Final = 16_000
AUDIO_EXTENSIONS: Final = frozenset({".wav", ".flac", ".ogg", ".mp3"})

Audio = NDArray[np.float32]


def to_mono(signal: NDArray[np.floating]) -> Audio:
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return np.asarray(signal, dtype=np.float32)


def resample(signal: Audio, rate_from: int, rate_to: int = SAMPLE_RATE) -> Audio:
    if rate_from == rate_to:
        return signal
    divisor = gcd(rate_from, rate_to)
    return np.asarray(resample_poly(signal, rate_to // divisor, rate_from // divisor), dtype=np.float32)


def load(path: Path) -> Audio:
    signal, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return resample(to_mono(signal), int(rate))


def load_segment(path: Path, seconds: float, rng: np.random.Generator) -> Audio:
    """Lê só um trecho sorteado: arquivos de ruído podem ter minutos, e o exemplo usa segundos."""
    info = sf.info(str(path))
    wanted = int(seconds * info.samplerate)
    if info.frames <= wanted:
        return load(path)
    start = int(rng.integers(0, info.frames - wanted + 1))
    signal, rate = sf.read(str(path), start=start, frames=wanted, dtype="float32", always_2d=False)
    return resample(to_mono(signal), int(rate))


def save(path: Path, signal: Audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(signal, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def duration(signal: Audio) -> float:
    return len(signal) / SAMPLE_RATE


def list_audio(directory: Path) -> list[Path]:
    """Arquivos de áudio em ordem estável (a ordem entra no sorteio determinístico)."""
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)
