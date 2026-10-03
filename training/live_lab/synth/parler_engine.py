"""Parler-TTS Mini Multilingual v1.1 (Apache-2.0): a voz é descrita em texto.

Cada descrição é um locutor sintético; variar gênero, tom, ritmo e ambiente dá diversidade sem
clonar ninguém. Requer GPU para volume (Colab).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from live_lab.audio_io import Audio
from live_lab.split import pseudonym
from live_lab.synth.base import Voice

REPO: Final = "parler-tts/parler-tts-mini-multilingual-v1.1"

DESCRIPTIONS: Final = tuple(
    f"A {gender} speaker delivers {pace} speech with a {pitch} voice, recorded {room}."
    for gender in ("male", "female")
    for pace in ("quick", "moderate")
    for pitch in ("low-pitched", "high-pitched")
    for room in ("very close to the microphone", "in a slightly distant, reverberant space")
)


class ParlerEngine:
    name = "parler-tts-multilingual"
    license = "Apache-2.0"

    def __init__(self, device: str = "cuda") -> None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._device = device
        self._model: Any = ParlerTTSForConditionalGeneration.from_pretrained(REPO).to(device)
        self._prompt_tokenizer: Any = AutoTokenizer.from_pretrained(REPO)
        self._description_tokenizer: Any = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )
        self._voices = [Voice(d, pseudonym(self.name, d)) for d in DESCRIPTIONS]

    def voices(self) -> Sequence[Voice]:
        return self._voices

    def synthesize(self, text: str, voice: Voice) -> tuple[Audio, int]:
        description = self._description_tokenizer(voice.voice_id, return_tensors="pt").input_ids
        prompt = self._prompt_tokenizer(text, return_tensors="pt").input_ids
        generation = self._model.generate(
            input_ids=description.to(self._device), prompt_input_ids=prompt.to(self._device)
        )
        audio = np.asarray(generation.cpu().numpy(), dtype=np.float32).reshape(-1)
        return audio, int(self._model.config.sampling_rate)
