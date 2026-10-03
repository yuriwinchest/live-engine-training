"""Execução do modelo exportado com ONNX Runtime: o mesmo arquivo e o mesmo caminho que irão ao celular."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from live_lab.decode import ACCEPT_ALL, CompiledTrie, Decision, Thresholds, decide


def open_session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def log_probs(session: ort.InferenceSession, audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Áudio 16 kHz [amostras] → log-probabilidades [quadros, vocabulário]."""
    output = session.run(None, {"audio": np.asarray(audio, dtype=np.float32)[None, :]})[0]
    return np.asarray(output[0], dtype=np.float32)


def recognize_all(
    session: ort.InferenceSession,
    clips: Sequence[NDArray[np.float32]],
    trie: CompiledTrie,
    thresholds: Thresholds = ACCEPT_ALL,
    beam_width: int = 8,
) -> list[Decision]:
    return [decide(log_probs(session, clip), trie, thresholds, beam_width) for clip in clips]
