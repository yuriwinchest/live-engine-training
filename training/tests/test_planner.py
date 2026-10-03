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
