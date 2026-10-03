"""SpecAugment leve, só no treino: máscaras de frequência e de tempo sobre o log-Mel normalizado."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SpecAugment(nn.Module):
    def __init__(
        self, freq_masks: int = 2, max_freq: int = 8, time_masks: int = 2, max_time_share: float = 0.1
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.max_freq = max_freq
        self.time_masks = time_masks
        self.max_time_share = max_time_share

    def forward(self, features: Tensor) -> Tensor:
        batch, bands, frames = features.shape
        out = features.clone()
        max_time = max(1, int(frames * self.max_time_share))
        for item in range(batch):
            for _ in range(self.freq_masks):
                width = int(torch.randint(0, self.max_freq + 1, ()).item())
                start = int(torch.randint(0, max(1, bands - width), ()).item())
                out[item, start : start + width, :] = 0.0
            for _ in range(self.time_masks):
                width = int(torch.randint(0, max_time + 1, ()).item())
                start = int(torch.randint(0, max(1, frames - width), ()).item())
                out[item, :, start : start + width] = 0.0
        return out
