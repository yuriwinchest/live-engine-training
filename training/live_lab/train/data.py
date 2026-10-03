"""Leitura do manifesto da feature 001 em lotes por duração.

Lotes agrupam clipes de tamanho parecido: menos preenchimento, menos computação desperdiçada.
O orçamento é em segundos de áudio por lote (já contando o preenchimento), não em número de clipes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from live_lab import audio_io
from live_lab.manifest import Example, Split
from live_lab.vocab import TOKEN_INDEX


@dataclass(frozen=True, slots=True)
class Batch:
    audio: Tensor
    samples: Tensor
    targets: Tensor
    target_lengths: Tensor
    indices: Tensor


class ManifestDataset(Dataset[tuple[Tensor, Tensor, int]]):
    def __init__(self, examples: Sequence[Example], root: Path, split: Split, max_seconds: float) -> None:
        self.root = root
        self.examples = [e for e in examples if e.split is split and e.duration_s <= max_seconds]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        example = self.examples[index]
        audio = torch.from_numpy(audio_io.load(self.root / example.audio))
        tokens = [TOKEN_INDEX[t] for t in example.tokens or ()]
        return audio, torch.tensor(tokens, dtype=torch.long), index


def collate(items: Sequence[tuple[Tensor, Tensor, int]]) -> Batch:
    lengths = torch.tensor([audio.numel() for audio, _, _ in items])
    audio = torch.zeros(len(items), int(lengths.max()))
    for row, (clip, _, _) in enumerate(items):
        audio[row, : clip.numel()] = clip
    targets = torch.cat([tokens for _, tokens, _ in items]) if items else torch.zeros(0, dtype=torch.long)
    target_lengths = torch.tensor([tokens.numel() for _, tokens, _ in items])
    indices = torch.tensor([index for _, _, index in items])
    return Batch(audio, lengths, targets, target_lengths, indices)


class DurationBatchSampler(Sampler[list[int]]):
    def __init__(self, durations: Sequence[float], batch_seconds: float, seed: int, shuffle: bool) -> None:
        order = sorted(range(len(durations)), key=lambda i: (durations[i], i))
        self.batches: list[list[int]] = []
        current: list[int] = []
        for index in order:
            # o lote é preenchido até o maior clipe, que é o último por causa da ordenação
            if current and durations[index] * (len(current) + 1) > batch_seconds:
                self.batches.append(current)
                current = []
            current.append(index)
        if current:
            self.batches.append(current)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = np.arange(len(self.batches))
        if self.shuffle:
            np.random.default_rng((self.seed, self.epoch)).shuffle(order)
        return (self.batches[i] for i in order)

    def __len__(self) -> int:
        return len(self.batches)
