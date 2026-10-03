"""Aleatoriedade derivada de (semente, chave): o resultado de um exemplo não depende da ordem
nem de quantos exemplos vieram antes, então uma execução parcial reproduz a completa."""

from __future__ import annotations

import hashlib
import uuid
from typing import Final

import numpy as np

NAMESPACE: Final = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/yuriwinchest/live-engine-training")


def _key(parts: tuple[object, ...]) -> str:
    return "|".join(str(part) for part in parts)


def stable_uuid(seed: int, *parts: object) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, _key((seed, *parts)))


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(_key(parts).encode("utf-8")).hexdigest()


def rng_for(seed: int, *parts: object) -> np.random.Generator:
    digest = hashlib.sha256(_key((seed, *parts)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))
