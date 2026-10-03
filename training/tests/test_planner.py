from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from live_lab.grammar import Grammar
from live_lab.manifest import Mode
from live_lab.synth.base import FakeEngine
from live_lab.synth.planner import DISTRACTOR_PHRASES, plan_jobs, read_jobs, write_jobs
from live_lab.vocab import WORDS

COUNT = 3000


@pytest.fixture(scope="module")
def jobs() -> list:  # type: ignore[type-arg]
    return plan_jobs(COUNT, FakeEngine().voices(), seed=7)


def test_distractor_phrases_have_no_vocabulary_word() -> None:
    for phrase in DISTRACTOR_PHRASES:
        assert not set(phrase.replace(",", " ").split()) & WORDS, phrase


def test_every_job_label_matches_grammar(jobs: list) -> None:  # type: ignore[type-arg]
    grammar = Grammar.build()
    for job in jobs:
        if job.number is None:
            assert job.tokens == () and job.mode is Mode.NONE
        else:
            assert grammar.parse(job.tokens) == job.number


def test_word_coverage_at_least_one_percent(jobs: list) -> None:  # type: ignore[type-arg]
    counts: Counter[str] = Counter(word for job in jobs for word in set(job.tokens))
    for word in WORDS:
        assert counts[word] / len(jobs) >= 0.01, (word, counts[word])


def test_both_modes_and_all_magnitudes(jobs: list) -> None:  # type: ignore[type-arg]
    modes = Counter(job.mode for job in jobs)
    assert modes[Mode.WORDED] > COUNT * 0.3 and modes[Mode.DIGITS] > COUNT * 0.3
    magnitudes = Counter(len(str(job.number)) for job in jobs if job.number is not None)
    assert all(magnitudes[size] > COUNT * 0.15 for size in (1, 2, 3, 4))


def test_deterministic_and_round_trip(tmp_path: Path, jobs: list) -> None:  # type: ignore[type-arg]
    assert plan_jobs(COUNT, FakeEngine().voices(), seed=7) == jobs
    assert plan_jobs(COUNT, FakeEngine().voices(), seed=8) != jobs
    path = tmp_path / "jobs.jsonl"
    write_jobs(path, jobs)
    assert read_jobs(path) == jobs
    assert len({job.id for job in jobs}) == len(jobs)


def test_short_utterances_are_slower(jobs: list) -> None:  # type: ignore[type-arg]
    """Feedback do PO: "uma" isolado saía atropelado; falas de 1–2 palavras vão mais devagar."""
    from live_lab.synth.planner import LONG_SPEED, SHORT_SPEED

    for job in jobs:
        words = len(job.tokens) if job.tokens else len(job.text.split())
        low, high = SHORT_SPEED if words <= 2 else LONG_SPEED
        assert low <= job.speed <= high, (job.text, job.speed)


def test_render_retries_rushed_speech(tmp_path: Path) -> None:
    import numpy as np

    from live_lab import audio_io
    from live_lab.manifest import read_clips
    from live_lab.synth.base import Voice
    from live_lab.synth.render import MIN_SECONDS_PER_WORD, render

    class Rushing:
        name, license = "rushing", "MIT"

        def __init__(self) -> None:
            self.speeds: list[float] = []

        def voices(self) -> list[Voice]:
            return [Voice("v", "spk-v")]

        def synthesize(self, text: str, voice: Voice, speed: float = 1.0) -> tuple[np.ndarray, int]:
            self.speeds.append(speed)
            seconds = 0.15 / speed  # rápido demais até desacelerar bastante
            t = np.arange(int(seconds * 16_000)) / 16_000
            return (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 16_000

    job = plan_jobs(1, [Voice("v", "spk-v")], seed=1)[0]
    single = type(job)(job.id, 1, ("uma",), Mode.DIGITS, "uma", "v", "spk-v", 0.8)
    engine = Rushing()
    render([single], engine, tmp_path)
    # 0,8 → 0,19 s (atropelado) → 0,64 → 0,23 s (atropelado) → 0,512 → 0,29 s (aceito)
    assert engine.speeds == pytest.approx([0.8, 0.64, 0.512])
    clip = next(read_clips(tmp_path / "clean.jsonl"))
    assert audio_io.duration(audio_io.load(tmp_path / clip.audio)) >= MIN_SECONDS_PER_WORD
