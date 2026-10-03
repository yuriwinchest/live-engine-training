"""Orquestra a preparação: clipes limpos + ruído → exemplos degradados, partições e manifesto.

Cada exemplo depende só de (semente, clipe, cópia); a ordem do loop não altera nenhum byte.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Final

import numpy as np

from live_lab import audio_io, augment
from live_lab.grammar import Grammar
from live_lab.holdout import holdout_key, is_reserved
from live_lab.ingest_real import refuse_cloud
from live_lab.manifest import CleanClip, Example, ManifestHeader, Mode, SnrLevel, Split, read_clips
from live_lab.negatives import noise_only
from live_lab.seeding import rng_for, stable_uuid
from live_lab.sources import Registry, SourceKind
from live_lab.split import assign_splits, split_of_file
from live_lab.vocab import VOCAB_VERSION

MIN_DURATION_S: Final = 0.2
MAX_DURATION_S: Final = 10.0
SILENCE_PEAK: Final = 1e-3
NOISE_SPEAKER: Final = "sem-locutor"


@dataclass(frozen=True, slots=True)
class PrepConfig:
    clean_manifests: Sequence[Path]
    out_dir: Path
    seed: int
    copies: int = 4
    val_fraction: float = 0.1
    negative_share: float = 0.15


@dataclass(frozen=True, slots=True)
class Discard:
    clip_id: str
    reason: str


@dataclass(slots=True)
class PrepResult:
    header: ManifestHeader
    examples: list[Example] = field(default_factory=list)
    discards: list[Discard] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Located:
    clip: CleanClip
    base: Path


def _load_clips(registry: Registry, manifests: Sequence[Path]) -> list[_Located]:
    located: dict[str, _Located] = {}
    for manifest in manifests:
        for clip in read_clips(manifest):
            kind = registry[clip.source].kind
            if kind not in (SourceKind.SPEECH_SYNTHETIC, SourceKind.SPEECH_REAL):
                raise ValueError(f"clipe {clip.id}: fonte {clip.source!r} é {kind}, não fala")
            if clip.id in located:
                raise ValueError(f"clipe repetido entre manifestos: {clip.id}")
            located[clip.id] = _Located(clip, manifest.parent)
    return [located[key] for key in sorted(located)]


def _noise_pools(registry: Registry, seed: int, val_fraction: float) -> dict[Split, list[tuple[str, Path]]]:
    pools: dict[Split, list[tuple[str, Path]]] = {Split.TRAIN: [], Split.VAL: []}
    for source in registry.of_kind(SourceKind.NOISE):
        if source.path is None:
            raise ValueError(f"fonte de ruído {source.name!r} sem 'path'")
        for file in audio_io.list_audio(source.path):
            if is_reserved(holdout_key(source.path, file)):
                continue  # reservado para a gravação real do PO: nunca entra no treino (US5 da 002)
            key = f"{source.name}/{file.relative_to(source.path).as_posix()}"
            pools[split_of_file(key, seed, val_fraction)].append((source.name, file))
    return pools


def _invalid_reason(clip: CleanClip, signal: audio_io.Audio, grammar: Grammar) -> str | None:
    seconds = audio_io.duration(signal)
    if not MIN_DURATION_S <= seconds <= MAX_DURATION_S:
        return f"duração {seconds:.2f}s fora de {MIN_DURATION_S}–{MAX_DURATION_S}s"
    if float(np.max(np.abs(signal))) < SILENCE_PEAK:
        return "silencioso"
    if clip.number is None and clip.tokens:
        return "sem número, mas com palavras do vocabulário"
    if clip.number is not None and clip.tokens is not None and grammar.parse(clip.tokens) != clip.number:
        return "rótulo não confere com a gramática"
    return None


def _example(clip: CleanClip, split: Split, example_id: str, seconds: float, **extra: object) -> Example:
    fields: dict[str, object] = {
        "clean_id": clip.id, "noise_source": None, "snr_level": SnrLevel.CLEAN, "snr_db": None,
        "stretch": 1.0, "pitch_semitones": 0.0, **extra,
    }  # fmt: skip
    return Example(
        id=example_id, audio=f"audio/{split}/{example_id}.wav", tokens=clip.tokens, number=clip.number,
        mode=clip.mode, source=clip.source, speaker=clip.speaker, split=split, duration_s=round(seconds, 3),
        **fields,  # type: ignore[arg-type]
    )  # fmt: skip


class _Preparer:
    def __init__(self, config: PrepConfig, registry: Registry) -> None:
        self.config = config
        self.registry = registry
        self.grammar = Grammar.build()
        self.pools = _noise_pools(registry, config.seed, config.val_fraction)
        self.notes: list[str] = []
        if not self.pools[Split.TRAIN]:
            raise ValueError("nenhum arquivo de ruído no registro: o treino precisa de ruído (FR-011)")
        if not self.pools[Split.VAL]:
            self.pools[Split.VAL] = self.pools[Split.TRAIN]
            self.notes.append("validação usa os mesmos arquivos de ruído do treino (poucos arquivos)")

    def _noise(self, split: Split, seconds: float, rng: np.random.Generator) -> tuple[str, audio_io.Audio]:
        pool = self.pools[Split.VAL if split is Split.VAL else Split.TRAIN]
        name, path = pool[int(rng.integers(len(pool)))]
        return name, audio_io.load_segment(path, seconds + 0.5, rng)

    def _save(self, example: Example, signal: audio_io.Audio) -> Example:
        audio_io.save(self.config.out_dir / example.audio, signal)
        return example

    def degrade(self, clip: CleanClip, split: Split, signal: audio_io.Audio) -> list[Example]:
        seed = self.config.seed
        if split is Split.TEST:
            example_id = str(stable_uuid(seed, "example", clip.id, 0))
            return [self._save(_example(clip, split, example_id, audio_io.duration(signal)), signal)]
        examples = []
        for copy in range(self.config.copies):
            rng = rng_for(seed, "augment", clip.id, copy)
            params = augment.sample_params(rng)
            noise_name, noise = None, None
            if params.snr_db is not None:
                expected = (
                    audio_io.duration(signal) / params.stretch + params.pad_before_s + params.pad_after_s
                )
                noise_name, noise = self._noise(split, expected, rng)
            degraded = augment.apply(signal, noise, params, rng)
            example_id = str(stable_uuid(seed, "example", clip.id, copy))
            example = _example(
                clip, split, example_id, audio_io.duration(degraded), noise_source=noise_name,
                snr_level=params.snr_level,
                snr_db=None if params.snr_db is None else round(params.snr_db, 2),
                stretch=round(params.stretch, 4), pitch_semitones=round(params.pitch_semitones, 3),
            )  # fmt: skip
            examples.append(self._save(example, degraded))
        return examples

    def negatives(self, split: Split, examples: Sequence[Example]) -> list[Example]:
        in_split = [e for e in examples if e.split is split]
        without = sum(1 for e in in_split if e.number is None)
        target = math.ceil(self.config.negative_share * len(in_split) / (1 - self.config.negative_share))
        missing = max(0, target - without)
        created = []
        for index in range(missing):
            rng = rng_for(self.config.seed, "negative", split, index)
            noise_name, noise = self._noise(split, 3.0, rng)
            signal = noise_only(noise, rng)
            example_id = str(stable_uuid(self.config.seed, "negative", split, index))
            created.append(
                self._save(
                    Example(
                        id=example_id, audio=f"audio/{split}/{example_id}.wav", tokens=(), number=None,
                        mode=Mode.NONE, source=noise_name, speaker=NOISE_SPEAKER, split=split, clean_id=None,
                        noise_source=noise_name, snr_level=SnrLevel.EXTREME, snr_db=None, stretch=1.0,
                        pitch_semitones=0.0, duration_s=round(audio_io.duration(signal), 3),
                    ),
                    signal,
                )
            )  # fmt: skip
        return created


def _lib_versions() -> dict[str, str]:
    return {name: version(name) for name in ("numpy", "scipy", "soundfile")}


def run(config: PrepConfig, registry: Registry) -> PrepResult:
    if config.copies < 1:
        raise ValueError("copies precisa ser ≥ 1")
    if not 0.0 < config.negative_share < 1.0:
        raise ValueError("negative_share precisa estar em (0, 1)")
    registry.require_valid()
    clips = _load_clips(registry, config.clean_manifests)
    if any(registry[item.clip.source].kind is SourceKind.SPEECH_REAL for item in clips):
        refuse_cloud()
    preparer = _Preparer(config, registry)
    splits = assign_splits(
        ((item.clip.speaker, registry[item.clip.source].kind is SourceKind.SPEECH_REAL) for item in clips),
        config.seed,
        config.val_fraction,
    )

    examples: list[Example] = []
    discards: list[Discard] = []
    for item in clips:
        signal = audio_io.load(item.base / item.clip.audio)
        reason = _invalid_reason(item.clip, signal, preparer.grammar)
        if reason:
            discards.append(Discard(item.clip.id, reason))
            continue
        examples.extend(preparer.degrade(item.clip, splits[item.clip.speaker], signal))
    for split in (Split.TRAIN, Split.VAL):
        examples.extend(preparer.negatives(split, examples))

    used = sorted({e.source for e in examples} | {e.noise_source for e in examples if e.noise_source})
    header = ManifestHeader(
        seed=config.seed,
        vocab_version=VOCAB_VERSION,
        provisional_test=not any(e.split is Split.TEST for e in examples),
        sources={name: registry[name].license for name in used},
        lib_versions=_lib_versions(),
    )
    examples.sort(key=lambda e: (e.split, e.id))
    return PrepResult(header, examples, discards, preparer.notes)
