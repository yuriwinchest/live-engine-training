"""Partição por locutor (FR-013) e pseudônimos (FR-020).

Voz real vai inteira para `test`: o modelo nunca a ouve no treino, e a medida final é com voz humana
(Princípio V). Vozes sintéticas dividem-se entre `train` e `val`.
"""

from __future__ import annotations

from collections.abc import Iterable

from live_lab.manifest import Split
from live_lab.seeding import stable_hash


def pseudonym(source: str, original_id: str) -> str:
    """Pseudônimo estável; o identificador original não sai da máquina que o leu."""
    return "spk-" + stable_hash("speaker", source, original_id)[:12]


def assign_splits(
    speakers: Iterable[tuple[str, bool]], seed: int, val_fraction: float = 0.1
) -> dict[str, Split]:
    """`speakers`: pares (pseudônimo, é_voz_real). Ordem de entrada não altera o resultado."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction fora de [0, 1): {val_fraction}")
    real: set[str] = set()
    synthetic: set[str] = set()
    for speaker, is_real in speakers:
        (real if is_real else synthetic).add(speaker)
    overlap = real & synthetic
    if overlap:
        raise ValueError(f"locutor marcado como real e sintético: {sorted(overlap)[:3]}")

    ranked = sorted(synthetic, key=lambda s: stable_hash("split", seed, s))
    val_count = round(len(ranked) * val_fraction)
    if val_fraction > 0 and val_count == 0 and len(ranked) >= 2:
        val_count = 1
    assignment = {speaker: Split.VAL for speaker in ranked[:val_count]}
    assignment.update({speaker: Split.TRAIN for speaker in ranked[val_count:]})
    assignment.update({speaker: Split.TEST for speaker in real})
    return assignment


def split_of_file(path_key: str, seed: int, val_fraction: float) -> Split:
    """Arquivos de ruído também se separam: o mesmo trecho não pode estar no treino e na validação."""
    bucket = int(stable_hash("noise-split", seed, path_key)[:8], 16) / 0xFFFFFFFF
    return Split.VAL if bucket < val_fraction else Split.TRAIN
