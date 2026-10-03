"""Exportação ponta a ponta com rede minúscula: pacote completo, entrada = áudio bruto, avaliação local."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from live_lab.evaluate import Package, evaluate_package
from live_lab.export import build_package
from live_lab.manifest import Split
from live_lab.model.network import NetConfig
from live_lab.train.loop import TrainConfig, train


def test_package_is_complete_and_evaluable(manifest: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    tiny = NetConfig(channels=16, blocks=1, repeats=1, kernels=(5,))
    train(
        TrainConfig(
            manifest=manifest, out_dir=run, epochs=1, batch_seconds=20.0, warmup_steps=5, workers=0,
            val_limit=20, net=tiny,
        )
    )  # fmt: skip
    result = build_package(run / "best.pt", manifest, tmp_path / "pkg", min_accuracy=0.5, val_limit=30)
    package = result.package
    for name in ("model.int8.onnx", "model.fp32.onnx", "grammar.json", "meta.json", "report.md"):
        assert (package / name).exists(), name
    meta = json.loads((package / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "live-model" and meta["sample_rate"] == 16_000
    assert meta["metrics"]["fp32_onnx_vs_torch_max_abs"] < 1e-3
    assert meta["bench"]["size_bytes"] < 5 * 2**20  # SC-001
    assert set(meta["thresholds"]) == {"min_posterior", "min_margin", "min_adherence"}

    report = evaluate_package(Package.open(package), manifest, Split.VAL, enrolled=[7, 233, 1500], limit=15)
    assert report["provisional"] is True
    assert all(e["got"] in (7, 233, 1500) for e in report["errors"])
