"""Pontos de retomada: gravação atômica, porque a sessão do Colab pode cair no meio da escrita."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LAST = "last.pt"
BEST = "best.pt"


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def save(path: Path, state: dict[str, Any]) -> None:
    """Grava num arquivo temporário e troca de uma vez: um ponto anterior nunca fica corrompido."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    return state
