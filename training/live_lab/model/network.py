"""Rede acústica: convoluções 1-D separáveis em profundidade com saída CTC (research R2 da 002).

Só convoluções, BatchNorm e ReLU: quantiza bem em INT8 e roda rápido em ARM. A normalização por fala usa
só os quadros válidos, para o preenchimento do lote não deslocar a média.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from live_lab.model.frontend import N_MELS, LogMel, frame_count
from live_lab.model.specaugment import SpecAugment
from live_lab.vocab import TOKENS


@dataclass(frozen=True, slots=True)
class NetConfig:
    channels: int = 192
    blocks: int = 6
    repeats: int = 2
    kernels: tuple[int, ...] = (11, 13, 15, 17, 19, 21)
    dropout: float = 0.1
    vocab: int = len(TOKENS)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class SeparableConv(nn.Sequential):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__(
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )


class Block(nn.Module):
    """`repeats` camadas separáveis com atalho residual em volta do bloco (estilo QuartzNet)."""

    def __init__(self, channels: int, kernel: int, repeats: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(SeparableConv(channels, kernel) for _ in range(repeats))
        self.residual = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False), nn.BatchNorm1d(channels))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for index, layer in enumerate(self.layers):
            out = layer(out)
            if index < len(self.layers) - 1:
                out = self.dropout(self.activation(out))
        return self.dropout(self.activation(out + self.residual(x)))


def utterance_norm(features: Tensor, frames: Tensor | None) -> Tensor:
    if frames is None:
        mean = features.mean(dim=2, keepdim=True)
        std = features.std(dim=2, keepdim=True)
    else:
        mask = (torch.arange(features.shape[2], device=features.device)[None, :] < frames[:, None]).unsqueeze(
            1
        )
        count = mask.sum(dim=2, keepdim=True).clamp(min=1)
        mean = (features * mask).sum(dim=2, keepdim=True) / count
        std = (((features - mean) * mask) ** 2).sum(dim=2, keepdim=True).div(count).sqrt()
    return (features - mean) / (std + 1e-5)


class LiveNet(nn.Module):
    def __init__(self, config: NetConfig = NetConfig()) -> None:  # noqa: B008 - dataclass imutável
        super().__init__()
        if len(config.kernels) < config.blocks:
            raise ValueError("kernels precisa ter um tamanho por bloco")
        self.config = config
        self.frontend = LogMel()
        self.augment = SpecAugment()
        self.stem = nn.Sequential(
            nn.Conv1d(N_MELS, config.channels, 11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(config.channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *(
                Block(config.channels, config.kernels[i], config.repeats, config.dropout)
                for i in range(config.blocks)
            )
        )
        self.head = nn.Conv1d(config.channels, config.vocab, 1)

    @staticmethod
    def output_frames(samples: Tensor) -> Tensor:
        return (frame_count(samples) - 1) // 2 + 1  # type: ignore[operator]

    def forward(self, audio: Tensor, samples: Tensor | None = None) -> Tensor:
        """Áudio [lote, amostras] → log-probabilidades [lote, quadros, vocabulário]."""
        frames = None if samples is None else frame_count(samples)
        features = utterance_norm(self.frontend(audio), frames)  # type: ignore[arg-type]
        if self.training:
            features = self.augment(features)
        logits = self.head(self.blocks(self.stem(features)))
        return torch.log_softmax(logits.transpose(1, 2), dim=-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
