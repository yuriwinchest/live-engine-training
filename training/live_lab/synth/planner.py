"""Escolhe o que sintetizar: números balanceados, formas variadas e frases sem número.

Sorteio uniforme por inteiro deixaria números de 1 algarismo quase ausentes (10 em 10 000) e palavras
raras ("catorze", "meia", "uma") abaixo do mínimo de cobertura (SC-006); por isso o sorteio é por
quantidade de algarismos e há um reforço explícito das palavras pouco vistas.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Final

import numpy as np

from live_lab.manifest import Mode
from live_lab.numbers_pt import Form, digit_forms, worded_forms
from live_lab.seeding import rng_for, stable_uuid
from live_lab.synth.base import Voice
from live_lab.vocab import WORDS

MIN_WORD_SHARE: Final = 0.015  # margem: a validação leva vozes inteiras e reduz a fatia do treino
DISTRACTOR_SHARE: Final = 0.06
SHORT_SPEED: Final = (0.70, 0.85)
LONG_SPEED: Final = (0.85, 1.00)

# Falas comuns na largada, sem nenhuma palavra do vocabulário (nem "e"): ensinam o "não entendi".
DISTRACTOR_PHRASES: Final[tuple[str, ...]] = (
    "vai vai vai",
    "olha o próximo",
    "atenção na chegada",
    "força força",
    "segura aí",
    "boa prova",
    "corre corre",
    "quase lá",
    "pode passar",
    "chegou outro",
    "abre caminho",
    "cuidado aí",
    "isso aí",
    "muito bem",
    "vamos lá",
    "espera",
    "não vi o número",
    "repete por favor",
    "tá chegando gente",
    "anota aí",
)

_BUCKETS: Final = ((0, 9), (10, 99), (100, 999), (1000, 9999))


@dataclass(frozen=True, slots=True)
class SynthJob:
    id: str
    number: int | None
    tokens: tuple[str, ...]
    mode: Mode
    text: str
    voice_id: str
    speaker: str
    speed: float = 1.0


@cache
def _forms_by_mode(number: int, mode: Mode) -> tuple[Form, ...]:
    return worded_forms(number) if mode is Mode.WORDED else digit_forms(number)


@cache
def _word_index() -> dict[str, tuple[tuple[int, Mode, Form], ...]]:
    index: dict[str, list[tuple[int, Mode, Form]]] = {word: [] for word in WORDS}
    for number in range(10_000):
        for mode in (Mode.WORDED, Mode.DIGITS):
            for form in _forms_by_mode(number, mode):
                for word in set(form):
                    index[word].append((number, mode, form))
    return {word: tuple(entries) for word, entries in index.items()}


def _text_for(form: Form, mode: Mode, rng: np.random.Generator) -> str:
    """No ditado, vírgulas às vezes: o TTS faz pausa, como o operador que fala algarismo por algarismo."""
    if mode is Mode.DIGITS and rng.random() < 0.5:
        return ", ".join(form)
    return " ".join(form)


def speed_for(words: int, rng: np.random.Generator) -> float:
    """Falas curtas mais devagar: o Kokoro "atropela" palavra isolada ("uma" saía ininteligível, PO)."""
    low, high = SHORT_SPEED if words <= 2 else LONG_SPEED
    return round(float(rng.uniform(low, high)), 3)


def _job(
    seed: int, ordinal: int, number: int | None, form: Form, mode: Mode, voice: Voice, text: str, speed: float
) -> SynthJob:
    job_id = str(stable_uuid(seed, "job", ordinal, voice.voice_id, " ".join(form), text, speed))
    return SynthJob(job_id, number, form, mode, text, voice.voice_id, voice.speaker, speed)


def plan_jobs(count: int, voices: Sequence[Voice], seed: int) -> list[SynthJob]:
    if count <= 0:
        raise ValueError("count precisa ser positivo")
    if not voices:
        raise ValueError("motor sem vozes")
    jobs: list[SynthJob] = []
    word_counts: Counter[str] = Counter()

    def add(number: int | None, form: Form, mode: Mode, rng: np.random.Generator) -> None:
        voice = voices[len(jobs) % len(voices)]
        text = _text_for(form, mode, rng) if form else str(rng.choice(DISTRACTOR_PHRASES))
        speed = speed_for(len(form) if form else len(text.split()), rng)
        jobs.append(_job(seed, len(jobs), number, form, mode, voice, text, speed))
        word_counts.update(form)

    distractors = math.ceil(count * DISTRACTOR_SHARE)
    for ordinal in range(count - distractors):
        rng = rng_for(seed, "plan", ordinal)
        low, high = _BUCKETS[int(rng.integers(len(_BUCKETS)))]
        number = int(rng.integers(low, high + 1))
        mode = Mode.WORDED if rng.random() < 0.5 else Mode.DIGITS
        forms = _forms_by_mode(number, mode)
        add(number, forms[int(rng.integers(len(forms)))], mode, rng)
    for ordinal in range(distractors):
        add(None, (), Mode.NONE, rng_for(seed, "distractor", ordinal))

    minimum = math.ceil(count * MIN_WORD_SHARE)
    for word in sorted(WORDS):
        candidates = _word_index()[word]
        boost = 0
        while word_counts[word] < minimum:
            rng = rng_for(seed, "boost", word, boost)
            number, mode, form = candidates[int(rng.integers(len(candidates)))]
            add(number, form, mode, rng)
            boost += 1
    return jobs


def write_jobs(path: Path, jobs: Sequence[SynthJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            data = asdict(job)
            data["tokens"] = list(job.tokens)
            handle.write(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n")


def read_jobs(path: Path) -> list[SynthJob]:
    jobs: list[SynthJob] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                jobs.append(SynthJob(**{**data, "tokens": tuple(data["tokens"]), "mode": Mode(data["mode"])}))
    return jobs
