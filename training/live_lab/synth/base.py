"""Protocolo dos motores de voz sintética e o motor falso usado nos testes.

Cada motor real vive no próprio módulo e importa suas dependências pesadas só quando instanciado,
para que gramática e preparação rodem sem torch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from live_lab.audio_io import Audio
from live_lab.split import pseudonym
from live_lab.vocab import TOKEN_INDEX


@dataclass(frozen=True, slots=True)
class Voice:
    voice_id: str
    speaker: str


class TtsEngine(Protocol):
    """`name` é também o nome da fonte no registro; `license`, a licença dos pesos do motor."""

    name: str
    license: str

    def voices(self) -> Sequence[Voice]: ...

    def synthesize(self, text: str, voice: Voice, speed: float = 1.0) -> tuple[Audio, int]:
        """`speed` < 1 fala mais devagar; motores sem controle de velocidade podem ignorar."""
        ...


class FakeEngine:
    """Motor determinístico sem dependências: cada palavra vira um tom. Só para testes."""

    name = "fake-tts"
    license = "MIT"
    rate = 16_000

    def __init__(self, voice_count: int = 6) -> None:
        self._voices = [Voice(f"fake-{i}", pseudonym(self.name, f"fake-{i}")) for i in range(voice_count)]

    def voices(self) -> Sequence[Voice]:
        return self._voices

    def synthesize(self, text: str, voice: Voice, speed: float = 1.0) -> tuple[Audio, int]:
        offset = 1.0 + 0.05 * self._voices.index(voice)
        t = np.arange(int(0.25 / speed * self.rate)) / self.rate
        gap = np.zeros(int(0.05 * self.rate), dtype=np.float32)
        parts: list[Audio] = []
        for word in text.replace(",", " ").split():
            frequency = (200 + 25 * TOKEN_INDEX.get(word, 50)) * offset
            parts += [(0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), gap]
        return np.concatenate(parts) if parts else gap, self.rate


def _kokoro(options: dict[str, Any]) -> TtsEngine:
    from live_lab.synth.kokoro_engine import KokoroEngine

    return KokoroEngine(**options)


def _chatterbox(options: dict[str, Any]) -> TtsEngine:
    from live_lab.synth.chatterbox_engine import ChatterboxEngine

    return ChatterboxEngine(**options)


def _parler(options: dict[str, Any]) -> TtsEngine:
    from live_lab.synth.parler_engine import ParlerEngine

    return ParlerEngine(**options)


ENGINES: dict[str, Callable[[dict[str, Any]], TtsEngine]] = {
    "fake": lambda options: FakeEngine(**options),
    "kokoro": _kokoro,
    "chatterbox": _chatterbox,
    "parler": _parler,
}


def get_engine(name: str, **options: Any) -> TtsEngine:
    try:
        factory = ENGINES[name]
    except KeyError:
        raise ValueError(f"motor desconhecido: {name!r} (opções: {sorted(ENGINES)})") from None
    return factory(options)
