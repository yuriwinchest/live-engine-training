"""US5 da 002: ruído reservado nunca entra no treino; faixas e roteiro prontos para a gravação real."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from live_lab import audio_io
from live_lab.grammar import Grammar
from live_lab.holdout import HOLDOUT_FRACTION, build_playback, holdout_key, is_reserved, recording_script
from live_lab.prep import _noise_pools
from live_lab.sources import load_registry

REGISTRY = """
[[source]]
name = "ruido"
kind = "noise"
license = "CC0-1.0"
origin = "teste"
obtained_at = 2026-10-03
path = "ruido"

[[source]]
name = "fala"
kind = "distractor"
license = "CC0-1.0"
origin = "teste"
obtained_at = 2026-10-03
path = "fala"
"""


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(5)
    for folder in ("ruido", "fala"):
        for index in range(60):
            signal = (0.1 * rng.standard_normal(audio_io.SAMPLE_RATE // 2)).astype(np.float32)
            audio_io.save(tmp_path / folder / f"sub/{folder}-{index:03d}.wav", signal)
    (tmp_path / "fontes.toml").write_text(REGISTRY, encoding="utf-8")
    return tmp_path


def test_reservation_rate_and_stability() -> None:
    keys = [f"noise/free-sound/noise-free-sound-{i:04d}.wav" for i in range(5000)]
    share = sum(is_reserved(k) for k in keys) / len(keys)
    assert abs(share - HOLDOUT_FRACTION) < 0.02
    assert [is_reserved(k) for k in keys[:50]] == [is_reserved(k) for k in keys[:50]]


def test_prep_never_uses_reserved_files(registry_dir: Path) -> None:
    """SC-008, para qualquer semente."""
    registry = load_registry(registry_dir / "fontes.toml")
    root = registry["ruido"].path
    assert root is not None
    reserved = {f for f in audio_io.list_audio(root) if is_reserved(holdout_key(root, f))}
    assert reserved, "fixture precisa ter ao menos um reservado"
    for seed in (0, 1, 99):
        pools = _noise_pools(registry, seed, 0.2)
        used = {path for pool in pools.values() for _, path in pool}
        assert not used & reserved
        assert used | reserved == set(audio_io.list_audio(root))


def test_playback_tracks_and_script(registry_dir: Path) -> None:
    out = registry_dir / "playback"
    result = build_playback(load_registry(registry_dir / "fontes.toml"), out, seed=1, tracks=2, minutes=0.1,
                            script_count=8)  # fmt: skip
    assert len(result.tracks) == 2
    for track in result.tracks:
        signal = audio_io.load(track)
        assert signal.size == int(0.1 * 60 * audio_io.SAMPLE_RATE)
        assert float(np.max(np.abs(signal))) <= 10 ** (-1 / 20) + 1e-3
    listed = (out / "reserved.txt").read_text(encoding="utf-8").splitlines()
    assert listed and all(is_reserved(line.split("\t")[1]) for line in listed)

    grammar = Grammar.build()
    with (out / "rotulos.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.script) == 9
    for row in rows:
        if row["numero"]:
            assert grammar.parse(row["fala"].split()) == int(row["numero"])
        else:
            assert row["fala"] == ""


def test_script_balances_digits_and_modes() -> None:
    script = [line for line in recording_script(100, seed=3) if line.number is not None]
    sizes = [len(str(line.number)) for line in script]
    assert all(sizes.count(size) == 25 for size in (1, 2, 3, 4))
    assert sum(line.mode == "extenso" for line in script) == 50
