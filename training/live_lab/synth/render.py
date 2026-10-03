"""Renderiza trabalhos de síntese em clipes limpos, retomável: um clipe já gravado não é refeito.

Útil no Colab, onde a sessão cai no meio: rodar de novo continua de onde parou.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from live_lab import audio_io, dsp
from live_lab.manifest import CleanClip, append_clip, read_clips
from live_lab.synth.base import TtsEngine, Voice
from live_lab.synth.planner import SynthJob

CLEAN_MANIFEST: Final = "clean.jsonl"
TRIM_TOP_DB: Final = 40
TARGET_PEAK: Final = 0.9
MIN_SECONDS_PER_WORD: Final = 0.28  # abaixo disso a fala sai atropelada (PO, 2026-10-03)
SLOWER: Final = 0.8
RETRIES: Final = 2


def _normalize(signal: audio_io.Audio) -> audio_io.Audio:
    trimmed = dsp.trim(signal, top_db=TRIM_TOP_DB)
    peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
    if peak == 0.0:
        return np.asarray(trimmed, dtype=np.float32)
    return np.asarray(trimmed * (TARGET_PEAK / peak), dtype=np.float32)


def _synthesize_paced(engine: TtsEngine, job: SynthJob, voice: Voice) -> audio_io.Audio:
    """Gera e, se a fala sair rápida demais para o número de palavras, gera de novo mais devagar."""
    speed = job.speed
    words = max(1, len(job.tokens))
    for attempt in range(RETRIES + 1):
        signal, rate = engine.synthesize(job.text, voice, speed)
        clip = _normalize(audio_io.resample(audio_io.to_mono(signal), rate))
        if not job.tokens or audio_io.duration(clip) / words >= MIN_SECONDS_PER_WORD or attempt == RETRIES:
            return clip
        speed *= SLOWER
    return clip


def render(jobs: Sequence[SynthJob], engine: TtsEngine, out_dir: Path) -> int:
    """Devolve quantos clipes foram gravados nesta chamada."""
    voices = {voice.voice_id: voice for voice in engine.voices()}
    manifest = out_dir / CLEAN_MANIFEST
    done = {clip.id for clip in read_clips(manifest)} if manifest.exists() else set()
    written = 0
    for position, job in enumerate(jobs, start=1):
        if job.id in done:
            continue
        voice = voices.get(job.voice_id)
        if voice is None:
            raise ValueError(f"voz {job.voice_id!r} não existe no motor {engine.name!r}")
        clip_audio = _synthesize_paced(engine, job, voice)
        relative = f"audio/{job.id}.wav"
        audio_io.save(out_dir / relative, clip_audio)
        append_clip(
            manifest,
            CleanClip(job.id, relative, job.tokens, job.number, job.mode, engine.name, job.speaker),
        )
        written += 1
        if position % 200 == 0:
            print(f"{position}/{len(jobs)} trabalhos", file=sys.stderr)
    return written
