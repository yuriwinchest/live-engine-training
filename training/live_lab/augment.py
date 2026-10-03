"""Degradação controlada que simula o celular do operador na largada (research R6).

Ordem: velocidade → tom → folga do Push-to-Talk → ruído em SNR → microfone → ganho → anti-clipping.
A SNR é exata no passo da mistura; o filtro de microfone vem depois porque o microfone real filtra
fala e ruído juntos, então a SNR efetiva final difere levemente da sorteada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.signal import butter, sosfilt

from live_lab import dsp
from live_lab.audio_io import SAMPLE_RATE, Audio
from live_lab.manifest import SnrLevel

PEAK_LIMIT: Final = 0.98
ACTIVE_FLOOR_DB: Final = -40.0

SNR_RANGES: Final[dict[SnrLevel, tuple[float, float]]] = {
    SnrLevel.LIGHT: (15.0, 25.0),
    SnrLevel.MEDIUM: (5.0, 15.0),
    SnrLevel.EXTREME: (-5.0, 5.0),
}
LEVEL_WEIGHTS: Final[dict[SnrLevel, float]] = {
    SnrLevel.CLEAN: 0.10,
    SnrLevel.LIGHT: 0.30,
    SnrLevel.MEDIUM: 0.35,
    SnrLevel.EXTREME: 0.25,
}


@dataclass(frozen=True, slots=True)
class AugmentParams:
    snr_level: SnrLevel
    snr_db: float | None
    stretch: float
    pitch_semitones: float
    highpass_hz: float
    lowpass_hz: float
    pad_before_s: float
    pad_after_s: float
    gain_db: float


def sample_params(rng: np.random.Generator, level: SnrLevel | None = None) -> AugmentParams:
    if level is None:
        levels = list(LEVEL_WEIGHTS)
        weights = np.array([LEVEL_WEIGHTS[lv] for lv in levels])
        level = levels[int(rng.choice(len(levels), p=weights / weights.sum()))]
    snr = None if level is SnrLevel.CLEAN else float(rng.uniform(*SNR_RANGES[level]))
    return AugmentParams(
        snr_level=level,
        snr_db=snr,
        stretch=float(rng.uniform(0.85, 1.15)),
        pitch_semitones=float(rng.uniform(-2.0, 2.0)),
        highpass_hz=float(rng.uniform(80.0, 200.0)),
        lowpass_hz=float(rng.uniform(3400.0, 7000.0)),
        pad_before_s=float(rng.uniform(0.1, 0.6)),
        pad_after_s=float(rng.uniform(0.1, 0.6)),
        gain_db=float(rng.uniform(-6.0, 6.0)),
    )


def active_power(signal: Audio) -> float:
    """Potência só das amostras de fala ativa; silêncio e folga não diluem a medida."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak == 0.0:
        return 0.0
    active = signal[np.abs(signal) >= peak * 10 ** (ACTIVE_FLOOR_DB / 20)]
    return float(np.mean(np.square(active, dtype=np.float64)))


def power(signal: Audio) -> float:
    return float(np.mean(np.square(signal, dtype=np.float64))) if signal.size else 0.0


def fit_length(noise: Audio, length: int, rng: np.random.Generator) -> Audio:
    """Trecho de ruído do tamanho pedido; repete o arquivo se ele for curto."""
    if noise.size == 0:
        raise ValueError("ruído vazio")
    if noise.size < length:
        noise = np.tile(noise, length // noise.size + 1)
    start = int(rng.integers(0, noise.size - length + 1))
    return noise[start : start + length]


def scale_noise(speech: Audio, noise: Audio, snr_db: float) -> Audio:
    speech_power, noise_power = active_power(speech), power(noise)
    if speech_power == 0.0 or noise_power == 0.0:
        raise ValueError("fala ou ruído sem energia: SNR indefinida")
    factor = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    return (noise * factor).astype(np.float32)


def measured_snr(speech: Audio, scaled_noise: Audio) -> float:
    return float(10 * np.log10(active_power(speech) / power(scaled_noise)))


time_stretch = dsp.time_stretch
pitch_shift = dsp.pitch_shift


def pad(signal: Audio, before_s: float, after_s: float) -> Audio:
    before, after = int(before_s * SAMPLE_RATE), int(after_s * SAMPLE_RATE)
    return np.pad(signal, (before, after)).astype(np.float32)


def microphone(signal: Audio, highpass_hz: float, lowpass_hz: float) -> Audio:
    nyquist = SAMPLE_RATE / 2
    sos = butter(
        4, [highpass_hz / nyquist, min(lowpass_hz, nyquist * 0.95) / nyquist], btype="band", output="sos"
    )
    return np.asarray(sosfilt(sos, signal), dtype=np.float32)


def prevent_clipping(signal: Audio, limit: float = PEAK_LIMIT) -> Audio:
    """Atenua a mistura inteira (fala e ruído juntos), preservando a SNR."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak <= limit:
        return signal
    return (signal * (limit / peak)).astype(np.float32)


def apply(speech: Audio, noise: Audio | None, params: AugmentParams, rng: np.random.Generator) -> Audio:
    shaped = pitch_shift(time_stretch(speech, params.stretch), params.pitch_semitones)
    shaped = pad(shaped, params.pad_before_s, params.pad_after_s)
    if params.snr_db is not None:
        if noise is None:
            raise ValueError(f"nível {params.snr_level} exige ruído")
        shaped = shaped + scale_noise(shaped, fit_length(noise, shaped.size, rng), params.snr_db)
    shaped = microphone(shaped, params.highpass_hz, params.lowpass_hz)
    shaped = (shaped * 10 ** (params.gain_db / 20)).astype(np.float32)
    return prevent_clipping(shaped)
