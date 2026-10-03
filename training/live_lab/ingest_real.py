"""Ingestão da gravação real do PO: vira a partição de teste (FR-008, FR-009, FR-019).

Roda só na máquina local. Os nomes originais dos arquivos não são copiados para o manifesto:
cada clipe passa a ser conhecido pelo seu UUID.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Final

from live_lab import audio_io
from live_lab.grammar import Grammar
from live_lab.manifest import CleanClip, Mode, append_clip
from live_lab.numbers_pt import worded_forms
from live_lab.seeding import stable_uuid
from live_lab.split import pseudonym
from live_lab.synth.render import CLEAN_MANIFEST
from live_lab.vocab import tokenize

CLOUD_MARKERS: Final = ("COLAB_RELEASE_TAG", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE")


def running_in_cloud() -> bool:
    return "google.colab" in sys.modules or any(marker in os.environ for marker in CLOUD_MARKERS)


def refuse_cloud() -> None:
    if running_in_cloud():
        raise ValueError("voz real só é processada na máquina local (Princípio IV da constituição)")


def _row_clip(row: dict[str, str], line: int, source: str, grammar: Grammar) -> CleanClip:
    raw_number = (row.get("numero") or "").strip()
    number = int(raw_number) if raw_number else None
    spoken = (row.get("fala") or "").strip()
    tokens: tuple[str, ...] | None = None
    mode = Mode.NONE if number is None else Mode.UNKNOWN
    if spoken:
        tokens = tokenize(spoken)
        parsed = grammar.parse(tokens)
        if parsed != number:
            raise ValueError(f"linha {line}: fala {spoken!r} vale {parsed}, rótulo diz {number}")
        if number is not None:
            mode = Mode.WORDED if tokens in worded_forms(number) else Mode.DIGITS
    elif number is None:
        tokens = ()
    clip_id = str(stable_uuid(0, "real", source, row["arquivo"]))
    speaker = pseudonym(source, (row.get("locutor") or "po").strip())
    return CleanClip(clip_id, f"audio/{clip_id}.wav", tokens, number, mode, source, speaker)


def ingest(labels: Path, audio_dir: Path, out_dir: Path, source: str) -> int:
    """CSV com colunas `arquivo,numero` e opcionais `fala,locutor`; `numero` vazio = sem número."""
    refuse_cloud()
    grammar = Grammar.build()
    manifest = out_dir / CLEAN_MANIFEST
    if manifest.exists():
        raise ValueError(f"{manifest} já existe; use uma pasta nova")
    count = 0
    with labels.open(encoding="utf-8-sig", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            clip = _row_clip(row, line, source, grammar)
            audio_io.save(out_dir / clip.audio, audio_io.load(audio_dir / row["arquivo"]))
            append_clip(manifest, clip)
            count += 1
    return count
