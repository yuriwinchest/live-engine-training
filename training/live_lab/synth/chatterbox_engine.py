"""Chatterbox Multilingual (MIT): clona a voz de clipes de referência CC0 (ex.: Common Voice pt).

Cada clipe de referência vira um locutor sintético; o nome do arquivo é pseudonimizado. A saída traz
marca d'água inaudível (PerTh) embutida pelo próprio motor. Requer GPU para volume (Colab).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from live_lab.audio_io import Audio, list_audio
from live_lab.split import pseudonym
from live_lab.synth.base import Voice


class ChatterboxEngine:
    name = "chatterbox-multilingual"
    license = "MIT"

    def __init__(self, references: str, device: str = "cuda", max_voices: int = 200) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._model: Any = ChatterboxMultilingualTTS.from_pretrained(device=device)
        paths = list_audio(Path(references))[:max_voices]
        if not paths:
            raise ValueError(f"nenhum clipe de referência em {references}")
        self._paths = {str(p): p for p in paths}
        self._voices = [Voice(str(p), pseudonym(self.name, p.stem)) for p in paths]

    def voices(self) -> Sequence[Voice]:
        return self._voices

    def synthesize(self, text: str, voice: Voice, speed: float = 1.0) -> tuple[Audio, int]:
        # sem controle de velocidade no motor: o time stretch da preparação cobre a variação
        wav = self._model.generate(text, language_id="pt", audio_prompt_path=str(self._paths[voice.voice_id]))
        return np.asarray(wav.detach().cpu().numpy(), dtype=np.float32).reshape(-1), int(self._model.sr)
