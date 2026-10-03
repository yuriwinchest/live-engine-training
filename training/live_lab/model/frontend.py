"""Log-Mel dentro do grafo exportado (Princípio III, research R3 da 002).

A STFT é uma convolução com pesos fixos de DFT vezes a janela de Hann: exporta para qualquer runtime de ONNX,
ao contrário do operador STFT, de suporte irregular no celular.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

SAMPLE_RATE = 16_000
N_FFT = 400  # 25 ms
HOP = 160  # 10 ms
N_MELS = 40
LOG_FLOOR = 1e-6


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float) -> np.ndarray:
    """Filtros triangulares na escala Mel (HTK), forma [bins de frequência, n_mels]."""
    bins = n_fft // 2 + 1
    freqs = np.linspace(0, sample_rate / 2, bins)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array(f_min)), _hz_to_mel(np.array(f_max)), n_mels + 2))
    bank = np.zeros((bins, n_mels), dtype=np.float32)
    for band in range(n_mels):
        low, center, high = edges[band : band + 3]
        rising = (freqs - low) / (center - low)
        falling = (high - freqs) / (high - center)
        bank[:, band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def frame_count(samples: Tensor | int) -> Tensor | int:
    return samples // HOP + 1


class LogMel(nn.Module):
    def __init__(self, n_mels: int = N_MELS, f_min: float = 20.0, f_max: float = 7600.0) -> None:
        super().__init__()
        bins = N_FFT // 2 + 1
        window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
        angle = (
            2
            * math.pi
            * torch.outer(torch.arange(bins, dtype=torch.float64), torch.arange(N_FFT, dtype=torch.float64))
            / N_FFT
        )
        dft = torch.cat([torch.cos(angle) * window, -torch.sin(angle) * window])
        self.bins = bins
        self.register_buffer("dft", dft.float().unsqueeze(1), persistent=False)
        bank = mel_filterbank(N_FFT, n_mels, SAMPLE_RATE, f_min, f_max)
        self.register_buffer("mel", torch.from_numpy(bank), persistent=False)

    def power(self, audio: Tensor) -> Tensor:
        """Espectro de potência [lote, bins, quadros]; quadros centrados (preenchimento de N_FFT/2)."""
        padded = F.pad(audio.unsqueeze(1), (N_FFT // 2, N_FFT // 2))
        spectrum = F.conv1d(padded, self.dft, stride=HOP)
        real, imag = spectrum[:, : self.bins], spectrum[:, self.bins :]
        return real * real + imag * imag

    def forward(self, audio: Tensor) -> Tensor:
        """Áudio [lote, amostras] em [-1, 1] → log-Mel [lote, n_mels, quadros]."""
        mel = torch.matmul(self.power(audio).transpose(1, 2), self.mel)
        return torch.log(mel + LOG_FLOOR).transpose(1, 2)
