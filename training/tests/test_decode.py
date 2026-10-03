"""Decodificador restrito com probabilidades sintéticas: comportamento exato, sem modelo."""

from __future__ import annotations

import numpy as np
import pytest

from live_lab.decode import CompiledTrie, Thresholds, decide
from live_lab.grammar import Grammar
from live_lab.vocab import TOKEN_INDEX, TOKENS

V = len(TOKENS)


@pytest.fixture(scope="module")
def grammar() -> Grammar:
    return Grammar.build()


@pytest.fixture(scope="module")
def full(grammar: Grammar) -> CompiledTrie:
    return CompiledTrie(grammar.export_trie())


def emissions(text: str, confidence: float = 0.9, blanks: int = 2, token_frames: int = 3) -> np.ndarray:
    """Quadros "picudos": cada palavra domina alguns quadros, separada por brancos."""
    rows: list[int] = [0] * blanks
    for word in text.split():
        rows += [TOKEN_INDEX[word]] * token_frames + [0] * blanks
    probs = np.full((len(rows), V), (1 - confidence) / (V - 1))
    probs[np.arange(len(rows)), rows] = confidence
    return np.log(probs)


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("duzentos e trinta e três", 233),
        ("dois três três", 233),
        ("zero zero sete", 7),
        ("meia meia", 66),
        ("dois dois", 22),
        ("mil e quinhentos", 1500),
        ("nove mil novecentos e noventa e nove", 9999),
    ],
)
def test_decodes_clear_speech(full: CompiledTrie, text: str, number: int) -> None:
    decision = decide(emissions(text), full)
    assert decision.number == number
    assert decision.posterior > 0.9


def test_repeated_token_needs_blank(full: CompiledTrie) -> None:
    """ "meia" contínuo, sem branco no meio, é uma palavra só (6), não 66."""
    probs = np.full((8, V), 0.01 / (V - 1))
    probs[:, TOKEN_INDEX["meia"]] = 0.99
    assert decide(np.log(probs), full).best == 6


def test_noise_is_rejected(full: CompiledTrie) -> None:
    probs = np.full((40, V), 0.05 / (V - 1))
    probs[:, 0] = 0.95
    decision = decide(np.log(probs), full)
    assert decision.number is None and decision.best is None


def test_never_outside_enrolled(grammar: Grammar) -> None:
    restricted = CompiledTrie(grammar.restrict([7, 233, 1500]).export_trie())
    loose = decide(emissions("duzentos e trinta e quatro"), restricted)
    assert loose.number in (None, 7, 233, 1500)
    strict = decide(emissions("duzentos e trinta e quatro"), restricted, Thresholds(min_adherence=-0.5))
    assert strict.number is None
    assert (
        decide(emissions("duzentos e trinta e três"), restricted, Thresholds(min_adherence=-0.5)).number
        == 233
    )


def test_outputs_always_inside_grammar(grammar: Grammar, full: CompiledTrie) -> None:
    """Probabilidades aleatórias: qualquer saída aceita é um número da gramática (SC-003)."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        logits = rng.normal(size=(30, V))
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        decision = decide(log_probs, full)
        assert decision.best is None or decision.best in grammar.numbers


def test_shape_mismatch_raises(full: CompiledTrie) -> None:
    with pytest.raises(ValueError):
        decide(np.zeros((5, V - 1)), full)


def test_empty_input_rejected(full: CompiledTrie) -> None:
    assert decide(np.zeros((0, V)), full).number is None
