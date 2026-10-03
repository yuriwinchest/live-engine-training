"""Treino de fumaça em CPU: dataset falso, rede minúscula, interrupção e retomada (SC-006)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from live_lab.model.network import NetConfig
from live_lab.train import checkpoint
from live_lab.train.loop import TrainConfig, train


def config(manifest: Path, out: Path, epochs: int) -> TrainConfig:
    return TrainConfig(
        manifest=manifest, out_dir=out, epochs=epochs, patience=10, batch_seconds=20.0, warmup_steps=5,
        workers=0, val_limit=40, net=NetConfig(channels=16, blocks=1, repeats=1, kernels=(5,)),
    )  # fmt: skip


def test_trains_and_resumes(manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    first = train(config(manifest, out, epochs=1))
    assert [h.epoch for h in first] == [0]
    assert (out / checkpoint.LAST).exists() and (out / checkpoint.BEST).exists()

    resumed = train(config(manifest, out, epochs=3))
    assert [h.epoch for h in resumed] == [0, 1, 2]  # a época 0 não foi refeita
    assert resumed[0] == first[0]
    assert resumed[-1].train_loss < resumed[0].train_loss
    saved = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert len(saved) == 3


def test_refuses_real_voice_in_cloud(manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import live_lab.train.loop as loop

    lines = manifest.read_text(encoding="utf-8").splitlines()
    for split in ('"split": "train"', '"split": "val"'):
        lines[1] = lines[1].replace(split, '"split": "test"')
    with_real = tmp_path / "manifest.jsonl"
    with_real.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(loop, "running_in_cloud", lambda: True)
    with pytest.raises(ValueError, match="nuvem"):
        train(config(with_real, tmp_path / "x", epochs=1))
