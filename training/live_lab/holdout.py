"""Ruído reservado para a gravação real do PO (US5 da feature 002).

O PO grava o teste tocando ruído numa caixa de som. Se esse ruído estivesse também no treino, o teste mediria
memória, não robustez. A reserva depende só do caminho relativo do arquivo,
nem da semente nem do nome da fonte:
nenhuma configuração futura devolve um reservado ao treino.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from live_lab import audio_io
from live_lab.audio_io import SAMPLE_RATE, Audio
from live_lab.augment import fit_length
from live_lab.manifest import Mode
from live_lab.numbers_pt import digit_forms, worded_forms
from live_lab.seeding import rng_for, stable_hash
from live_lab.sources import Registry, SourceKind

HOLDOUT_FRACTION: Final = 0.10
TARGET_RMS_DB: Final = -20.0
PEAK_DB: Final = -1.0
CROSSFADE_S: Final = 0.5


@dataclass(frozen=True, slots=True)
class Reserved:
    source: str
    path: Path
    key: str


def holdout_key(root: Path, file: Path) -> str:
    return file.relative_to(root).as_posix()


def is_reserved(key: str) -> bool:
    return int(stable_hash("holdout-v1", key)[:8], 16) / 0xFFFFFFFF < HOLDOUT_FRACTION


def reserved_files(registry: Registry, kind: SourceKind) -> list[Reserved]:
    found: list[Reserved] = []
    for source in registry.of_kind(kind):
        if source.path is None:
            continue
        for file in audio_io.list_audio(source.path):
            key = holdout_key(source.path, file)
            if is_reserved(key):
                found.append(Reserved(source.name, file, key))
    return found


def _rms_db(signal: Audio) -> float:
    return float(10 * np.log10(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))


def _gain(signal: Audio, db: float) -> Audio:
    return (signal * 10 ** (db / 20)).astype(np.float32)


def normalize(signal: Audio, rms_db: float = TARGET_RMS_DB, peak_db: float = PEAK_DB) -> Audio:
    leveled = _gain(signal, rms_db - _rms_db(signal))
    peak, limit = float(np.max(np.abs(leveled))), 10 ** (peak_db / 20)
    return _gain(leveled, 20 * np.log10(limit / peak)) if peak > limit else leveled


def _segment(item: Reserved, seconds: float, rng: np.random.Generator) -> Audio:
    loaded = audio_io.load_segment(item.path, seconds, rng)
    return normalize(fit_length(loaded, int(seconds * SAMPLE_RATE), rng))


def crowd(speech: Sequence[Reserved], seconds: float, rng: np.random.Generator) -> Audio:
    """Multidão: 6 a 8 falas reservadas sobrepostas, cada uma com ganho próprio."""
    if not speech:
        raise ValueError("nenhuma fala reservada para montar multidão")
    voices = int(rng.integers(6, 9))
    picks = rng.choice(len(speech), size=voices, replace=len(speech) < voices)
    mix = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    for index in picks:
        mix += _gain(_segment(speech[int(index)], seconds, rng), float(rng.uniform(-6.0, 0.0)))
    return normalize(mix)


def _chain(items: Sequence[Reserved], seconds: float, rng: np.random.Generator) -> Audio:
    """Trechos de 10–30 s encadeados com transição suave, até a duração pedida."""
    total = int(seconds * SAMPLE_RATE)
    fade = int(CROSSFADE_S * SAMPLE_RATE)
    out = np.zeros(0, dtype=np.float32)
    while out.size < total:
        piece = _segment(items[int(rng.integers(len(items)))], float(rng.uniform(10.0, 30.0)), rng)
        if out.size >= fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            out[-fade:] = out[-fade:] * (1 - ramp) + piece[:fade] * ramp
            piece = piece[fade:]
        out = np.concatenate([out, piece])
    return out[:total]


def street(
    speech: Sequence[Reserved], noise: Sequence[Reserved], seconds: float, rng: np.random.Generator
) -> Audio:
    """Rua: multidão por baixo de ruído e música reservados."""
    background = crowd(speech, seconds, rng)
    if not noise:
        return background
    foreground = _gain(_chain(noise, seconds, rng), float(rng.uniform(-6.0, 0.0)))
    return normalize(background + foreground)


_PLAIN: Final = frozenset({"uma", "duas", "catorze"})


def _plain_forms(number: int) -> list[tuple[str, ...]]:
    return [form for form in worded_forms(number) if not _PLAIN & set(form) and form[:2] != ("um", "mil")]


def _normative(number: int) -> str:
    """Norma culta: depois de "mil", "e" só antes de dezena/unidade ou de centena redonda (mil e duzentos)."""
    thousands, rest = divmod(number, 1000)
    if thousands == 0 or rest == 0:
        return " ".join(_plain_forms(number)[0])
    head = "mil" if thousands == 1 else f"{' '.join(_plain_forms(thousands)[0])} mil"
    tail = " ".join(_plain_forms(rest)[0])
    joiner = " e " if rest < 100 or rest % 100 == 0 else " "
    return f"{head}{joiner}{tail}"


@dataclass(frozen=True, slots=True)
class ScriptLine:
    file: str
    number: int | None
    mode: Mode
    spoken: str


def recording_script(count: int, seed: int) -> list[ScriptLine]:
    """Roteiro da gravação: números balanceados por algarismos e modo, mais apertos sem número."""
    lines: list[ScriptLine] = []
    buckets = ((0, 9), (10, 99), (100, 999), (1000, 9999))
    for index in range(count):
        rng = rng_for(seed, "script", index)
        low, high = buckets[index % len(buckets)]
        number = int(rng.integers(low, high + 1))
        mode = Mode.WORDED if index % 2 == 0 else Mode.DIGITS
        if mode is Mode.WORDED:
            spoken = _normative(number)
        else:
            forms = digit_forms(number)
            spoken = " ".join(forms[int(rng.integers(len(forms)))])
        lines.append(ScriptLine(f"{index + 1:03d}.wav", number, mode, spoken))
    for extra in range(max(1, count // 10)):
        lines.append(ScriptLine(f"{count + extra + 1:03d}.wav", None, Mode.NONE, "(aperte sem falar número)"))
    return lines


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    reserved: int
    tracks: list[Path]
    script: list[ScriptLine]


def _write_script(out_dir: Path, script: Sequence[ScriptLine]) -> None:
    """`rotulos.csv` já no formato do `ingest-real`; `roteiro.md` para ler durante a gravação."""
    with (out_dir / "rotulos.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arquivo", "numero", "fala"))
        for line in script:
            has_number = line.number is not None
            writer.writerow((line.file, line.number if has_number else "", line.spoken if has_number else ""))
    table = ["| Arquivo | Fale |", "|---|---|", *(f"| {line.file} | {line.spoken} |" for line in script)]
    (out_dir / "roteiro.md").write_text(
        "# Roteiro da gravação\n\n" + "\n".join(table) + "\n", encoding="utf-8"
    )


def build_playback(
    registry: Registry,
    out_dir: Path,
    seed: int,
    tracks: int = 6,
    minutes: float = 3.0,
    script_count: int = 100,
) -> PlaybackResult:
    speech = reserved_files(registry, SourceKind.DISTRACTOR)
    noise = reserved_files(registry, SourceKind.NOISE)
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = sorted(speech + noise, key=lambda r: (r.source, r.key))
    (out_dir / "reserved.txt").write_text(
        "".join(f"{r.source}\t{r.key}\n" for r in listing), encoding="utf-8"
    )
    written: list[Path] = []
    for index in range(tracks):
        rng = rng_for(seed, "playback", index)
        is_crowd = index % 2 == 0
        signal = crowd(speech, minutes * 60, rng) if is_crowd else street(speech, noise, minutes * 60, rng)
        path = out_dir / f"{'multidao' if is_crowd else 'rua'}-{index // 2 + 1:02d}.wav"
        audio_io.save(path, signal)
        written.append(path)
    script = recording_script(script_count, seed)
    _write_script(out_dir, script)
    return PlaybackResult(len(speech) + len(noise), written, script)
