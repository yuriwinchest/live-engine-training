"""Fixtures compartilhadas: fontes e ruído gerados em pasta temporária, fora do repositório."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from live_lab import audio_io
from live_lab.sources import Source, SourceKind

CONSENT = "6f1c2a52-6a4c-4d7e-9b1a-2f6f6c2f8f10"


def make_source(path: Path | None = None, **overrides: object) -> Source:
    fields: dict[str, object] = {
        "name": "fonte",
        "kind": SourceKind.NOISE,
        "license": "CC0-1.0",
        "origin": "teste",
        "obtained_at": date(2026, 10, 3),
        "path": path,
    }
    fields.update(overrides)
    return Source(**fields)  # type: ignore[arg-type]


@pytest.fixture
def noise_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "ruido"
    rng = np.random.default_rng(123)
    for index in range(6):
        seconds = 2.0 + index
        signal = (0.2 * rng.standard_normal(int(seconds * audio_io.SAMPLE_RATE))).astype(np.float32)
        audio_io.save(directory / f"ruido-{index}.wav", signal)
    return directory


TINY_REGISTRY = """
[[source]]
name = "fake-tts"
kind = "speech_synthetic"
license = "MIT"
origin = "teste"
obtained_at = 2026-10-03

[[source]]
name = "ruido"
kind = "noise"
license = "CC0-1.0"
origin = "teste"
obtained_at = 2026-10-03
path = "ruido"
"""


@pytest.fixture(scope="session")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Dataset falso pequeno, preparado de ponta a ponta (treino e exportação de fumaça)."""
    from live_lab import report
    from live_lab.prep import PrepConfig, run
    from live_lab.sources import load_registry
    from live_lab.synth.base import FakeEngine
    from live_lab.synth.planner import plan_jobs
    from live_lab.synth.render import CLEAN_MANIFEST, render

    base = tmp_path_factory.mktemp("treino")
    rng = np.random.default_rng(1)
    for index in range(6):
        noise = (0.1 * rng.standard_normal(32_000)).astype(np.float32)
        audio_io.save(base / "ruido" / f"r{index}.wav", noise)
    (base / "fontes.toml").write_text(TINY_REGISTRY, encoding="utf-8")
    engine = FakeEngine(voice_count=4)
    render(plan_jobs(60, engine.voices(), seed=2), engine, base / "clean")
    registry = load_registry(base / "fontes.toml")
    config = PrepConfig([base / "clean" / CLEAN_MANIFEST], base / "ds", seed=3, copies=1, val_fraction=0.25)
    report.write_outputs(run(config, registry), registry, config)
    return base / "ds" / "manifest.jsonl"
