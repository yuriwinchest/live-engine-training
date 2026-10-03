"""Processamento de sinal sem librosa.

O librosa puxa o numba, cuja DLL o Smart App Control do Windows bloqueia nesta máquina; um phase
vocoder em numpy resolve o que o projeto precisa (velocidade, tom, corte de silêncio) e roda igual no
Windows e no Colab.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import get_window, resample

from live_lab.audio_io import Audio

N_FFT: Final = 512
HOP: Final = 128


def _stft(signal: Audio) -> tuple[np.ndarray, np.ndarray]:
    window = get_window("hann", N_FFT)
    padded = np.pad(signal, (N_FFT // 2, N_FFT // 2 + N_FFT))
    frames = sliding_window_view(padded, N_FFT)[::HOP] * window
    return np.fft.rfft(frames, axis=1), window


def _istft(spectrum: np.ndarray, window: np.ndarray, length: int) -> Audio:
    frames = np.fft.irfft(spectrum, n=N_FFT, axis=1) * window
    total = HOP * (len(frames) - 1) + N_FFT
    out = np.zeros(total)
    norm = np.zeros(total)
    for index, frame in enumerate(frames):
        start = index * HOP
        out[start : start + N_FFT] += frame
        norm[start : start + N_FFT] += window**2
    out /= np.where(norm > 1e-8, norm, 1.0)
    trimmed = out[N_FFT // 2 : N_FFT // 2 + length]
    return np.pad(trimmed, (0, max(0, length - trimmed.size))).astype(np.float32)


def time_stretch(signal: Audio, rate: float) -> Audio:
    """Muda a velocidade sem mudar o tom; rate > 1 acelera."""
    if rate <= 0:
        raise ValueError(f"rate precisa ser positivo: {rate}")
    if rate == 1.0 or signal.size < N_FFT:
        return signal
    spectrum, window = _stft(signal)
    bins = spectrum.shape[1]
    advance = 2 * np.pi * HOP * np.arange(bins) / N_FFT
    steps = np.arange(0, spectrum.shape[0] - 1, rate)
    phase = np.angle(spectrum[0])
    stretched = np.empty((steps.size, bins), dtype=np.complex128)
    for out_index, step in enumerate(steps):
        low = int(step)
        frac = step - low
        magnitude = (1 - frac) * np.abs(spectrum[low]) + frac * np.abs(spectrum[low + 1])
        stretched[out_index] = magnitude * np.exp(1j * phase)
        delta = np.angle(spectrum[low + 1]) - np.angle(spectrum[low]) - advance
        phase = phase + advance + (delta - 2 * np.pi * np.round(delta / (2 * np.pi)))
    return _istft(stretched, window, round(signal.size / rate))


def pitch_shift(signal: Audio, semitones: float) -> Audio:
    """Muda o tom mantendo a duração: estica no tempo e reamostra de volta ao tamanho original."""
    if semitones == 0.0 or signal.size < N_FFT:
        return signal
    factor = 2 ** (semitones / 12)
    stretched = time_stretch(signal, 1 / factor)
    return np.asarray(resample(stretched, signal.size), dtype=np.float32)


def trim(signal: Audio, top_db: float = 40.0, frame: int = 1024, hop: int = 256) -> Audio:
    """Corta silêncio nas pontas: quadros abaixo de `top_db` do quadro mais forte."""
    if signal.size < frame:
        return signal
    rms = np.sqrt(np.mean(sliding_window_view(signal, frame)[::hop] ** 2, axis=1))
    peak = float(rms.max())
    if peak == 0.0:
        return signal[:0]
    loud = np.flatnonzero(20 * np.log10(np.maximum(rms, 1e-12) / peak) > -top_db)
    return signal[loud[0] * hop : min(signal.size, loud[-1] * hop + frame)]
