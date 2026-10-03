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
