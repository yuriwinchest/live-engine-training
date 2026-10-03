"""Exportação para o celular: ONNX FP32 → INT8 (QDQ), equivalência, calibração e pacote (US4 da 002).

O front-end (log-Mel) fica em FP32: log e potência perdem precisão demais em 8 bits. Só as convoluções da
rede são quantizadas, com calibração estática em clipes da validação.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)

from live_lab import audio_io, bench
from live_lab.calibrate import calibrate
from live_lab.decode import ACCEPT_ALL, CompiledTrie
from live_lab.evaluate import decisions_for, halves, summarize, to_markdown, vars_of
from live_lab.grammar import Grammar
from live_lab.manifest import Split, read_manifest
from live_lab.model.network import LiveNet, NetConfig
from live_lab.runtime import log_probs, open_session
from live_lab.seeding import stable_hash
from live_lab.train import checkpoint
from live_lab.vocab import TOKENS, VOCAB_VERSION

MAX_SECONDS = 8.0


def load_model(path: Path) -> LiveNet:
    state = checkpoint.load(path)
    if state.get("vocab_version") != VOCAB_VERSION:
        raise ValueError("checkpoint de outra versão do vocabulário")
    net = dict(state["net"])
    net["kernels"] = tuple(net["kernels"])
    model = LiveNet(NetConfig(**net))
    model.load_state_dict(state["model"])
    return model.eval()


def export_fp32(model: LiveNet, path: Path) -> None:
    sample = torch.zeros(1, 16_000)
    torch.onnx.export(
        model, (sample,), str(path), input_names=["audio"], output_names=["log_probs"],
        dynamic_axes={"audio": {1: "samples"}, "log_probs": {1: "frames"}}, opset_version=17, dynamo=False,
    )  # fmt: skip


class _ClipReader(CalibrationDataReader):  # type: ignore[misc]
    def __init__(self, clips: Sequence[np.ndarray]) -> None:
        self._iter: Iterator[dict[str, np.ndarray]] = iter(
            {"audio": c[None, :].astype(np.float32)} for c in clips
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def quantize(fp32: Path, int8: Path, clips: Sequence[np.ndarray]) -> None:
    prepared = fp32.with_name("model.prep.onnx")
    quant_pre_process(str(fp32), str(prepared))
    frontend = [n.name for n in onnx.load(str(prepared)).graph.node if "frontend" in n.name]
    quantize_static(
        str(prepared), str(int8), _ClipReader(clips), quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, op_types_to_quantize=["Conv"],
        nodes_to_exclude=frontend,
    )  # fmt: skip
    prepared.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ExportResult:
    package: Path
    meta: dict[str, Any]


def _parity(model: LiveNet, fp32: Path, clips: Sequence[np.ndarray]) -> float:
    session = open_session(fp32)
    worst = 0.0
    with torch.no_grad():
        for clip in clips:
            expected = model(torch.from_numpy(clip)[None, :])[0].numpy()
            worst = max(worst, float(np.max(np.abs(log_probs(session, clip) - expected))))
    return worst


def build_package(
    checkpoint_path: Path, manifest: Path, out: Path, min_accuracy: float = 0.97, val_limit: int = 2000
) -> ExportResult:
    _, examples = read_manifest(manifest)
    val = [e for e in examples if e.split is Split.VAL and e.duration_s <= MAX_SECONDS]
    val = sorted(val, key=lambda e: stable_hash("export", e.id))[:val_limit]
    if not val:
        raise ValueError("manifesto sem partição de validação")
    root = manifest.parent
    clips = [audio_io.load(root / e.audio) for e in val[:300]]

    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint_path)
    fp32, int8 = out / "model.fp32.onnx", out / "model.int8.onnx"
    export_fp32(model, fp32)
    parity = _parity(model, fp32, clips[:10])
    print(f"FP32 ONNX vs PyTorch: diferença máxima {parity:.2e}", file=sys.stderr)
    quantize(fp32, int8, clips)

    grammar = Grammar.build()
    trie_dict = grammar.export_trie()
    (out / "grammar.json").write_text(json.dumps(trie_dict, separators=(",", ":")), encoding="utf-8")
    trie = CompiledTrie(trie_dict)
    calib_set, report_set = halves(val)
    print(f"decodificando {len(val)} clipes de validação (FP32 e INT8)…", file=sys.stderr)
    int8_calib = decisions_for(int8, calib_set, root, trie)
    calibration = calibrate(int8_calib, [e.number for e in calib_set], min_accuracy)
    int8_report = summarize(
        report_set, decisions_for(int8, report_set, root, trie), calibration.thresholds, True
    )
    fp32_raw = summarize(report_set, decisions_for(fp32, report_set, root, trie), ACCEPT_ALL, True)
    int8_raw = summarize(report_set, decisions_for(int8, report_set, root, trie), ACCEPT_ALL, True)
    measured = bench.measure(int8)

    meta: dict[str, Any] = {
        "format": "live-model", "format_version": 1, "vocab_version": VOCAB_VERSION, "tokens": list(TOKENS),
        "sample_rate": audio_io.SAMPLE_RATE, "frame_ms": 20, "max_seconds": MAX_SECONDS, "beam_width": 8,
        "thresholds": vars_of(calibration.thresholds),
        "sha256": {"model.int8.onnx": _sha256(int8), "grammar.json": _sha256(out / "grammar.json")},
        "metrics": {
            "val_calibrated_int8": int8_report["totals"], "val_no_reject_fp32": fp32_raw["totals"],
            "val_no_reject_int8": int8_raw["totals"], "target_met": calibration.target_met,
            "min_accuracy": min_accuracy, "provisional": True, "fp32_onnx_vs_torch_max_abs": parity,
        },
        "bench": measured.as_dict(),
        "net": model.config.as_dict(),
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(
        to_markdown("Validação (INT8, limiares calibrados)", int8_report), encoding="utf-8"
    )
    (out / "report.json").write_text(json.dumps(int8_report, indent=2, ensure_ascii=False), encoding="utf-8")
    return ExportResult(out, meta)
