"""Kokoro-82M (Apache-2.0), 3 vozes pt-BR. Exige o pacote `kokoro` e o espeak-ng do sistema.

No Colab: `apt-get install -y espeak-ng` e `pip install kokoro`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from live_lab.audio_io import Audio
from live_lab.split import pseudonym
from live_lab.synth.base import Voice

KOKORO_RATE: Final = 24_000
PT_BR_VOICES: Final = ("pf_dora", "pm_alex", "pm_santa")


class KokoroEngine:
    name = "kokoro-82m"
    license = "Apache-2.0"

    def __init__(self, device: str | None = None) -> None:
        from kokoro import KPipeline

        self._pipeline: Any = KPipeline(lang_code="p", device=device)
        self._voices = [Voice(v, pseudonym(self.name, v)) for v in PT_BR_VOICES]

    def voices(self) -> Sequence[Voice]:
        return self._voices

    def synthesize(self, text: str, voice: Voice, speed: float = 1.0) -> tuple[Audio, int]:
        chunks: list[Audio] = []
        for result in self._pipeline(text, voice=voice.voice_id, speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is not None:
                chunks.append(np.asarray(audio.detach().cpu().numpy(), dtype=np.float32).reshape(-1))
        if not chunks:
            raise RuntimeError(f"Kokoro não gerou áudio para {text!r}")
        return np.concatenate(chunks), KOKORO_RATE
