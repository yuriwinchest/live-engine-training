"""Clipes limpos, exemplos preparados e o formato JSONL (data-model.md).

O manifesto não guarda data de criação: duas execuções com a mesma semente precisam gerar o mesmo
arquivo byte a byte (SC-004). A data vai para `run.json`, ao lado.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Mode(StrEnum):
    WORDED = "extenso"
    DIGITS = "digitos"
    NONE = "nenhum"
    UNKNOWN = "desconhecido"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SnrLevel(StrEnum):
    CLEAN = "limpo"
    LIGHT = "leve"
    MEDIUM = "medio"
    EXTREME = "extremo"


@dataclass(frozen=True, slots=True)
class CleanClip:
    id: str
    audio: str
    tokens: tuple[str, ...] | None
    number: int | None
    mode: Mode
    source: str
    speaker: str

    @property
    def has_number(self) -> bool:
        return self.number is not None


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    audio: str
    tokens: tuple[str, ...] | None
    number: int | None
    mode: Mode
    source: str
    speaker: str
    split: Split
    clean_id: str | None
    noise_source: str | None
    snr_level: SnrLevel
    snr_db: float | None
    stretch: float
    pitch_semitones: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class ManifestHeader:
    seed: int
    vocab_version: str
    provisional_test: bool
    sources: dict[str, str]
    lib_versions: dict[str, str] = field(default_factory=dict)


def _encode(record: Any) -> str:
    payload = asdict(record)
    for key, value in payload.items():
        if isinstance(value, tuple):
            payload[key] = list(value)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _tokens(value: Any) -> tuple[str, ...] | None:
    return None if value is None else tuple(value)


def clip_from_dict(data: dict[str, Any]) -> CleanClip:
    return CleanClip(
        id=data["id"],
        audio=data["audio"],
        tokens=_tokens(data.get("tokens")),
        number=data.get("number"),
        mode=Mode(data["mode"]),
        source=data["source"],
        speaker=data["speaker"],
    )


def example_from_dict(data: dict[str, Any]) -> Example:
    return Example(
        **{
            **data,
            "tokens": _tokens(data.get("tokens")),
            "mode": Mode(data["mode"]),
            "split": Split(data["split"]),
            "snr_level": SnrLevel(data["snr_level"]),
        }
    )


def append_clip(path: Path, clip: CleanClip) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_encode(clip) + "\n")


def read_clips(path: Path) -> Iterator[CleanClip]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield clip_from_dict(json.loads(line))


def write_manifest(path: Path, header: ManifestHeader, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"type": "header", **asdict(header)}, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        for example in examples:
            handle.write(_encode(example) + "\n")


def read_manifest(path: Path) -> tuple[ManifestHeader, list[Example]]:
    with path.open(encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path}: manifesto sem cabeçalho")
    head = {k: v for k, v in lines[0].items() if k != "type"}
    return ManifestHeader(**head), [example_from_dict(item) for item in lines[1:]]
