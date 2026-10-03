"""Juiz de inteligibilidade: transcrições reais do Whisper na amostra que o PO ouviu (2026-10-03)."""

from __future__ import annotations

import pytest

from live_lab.grammar import Grammar
from live_lab.transcript import says_no_number, understood_number


@pytest.fixture(scope="module")
def grammar() -> Grammar:
    return Grammar.build()


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("233.", 233),
        ("007", 7),
        ("3.", 3),
        ("Dois, três, três.", 233),
        ("Duzentos e trinta e um.", 231),
        ("A doze.", 12),
        ("Dois.", 2),
        ("1.500", 1500),
        ("dois tres tres", 233),
    ],
)
def test_understood(grammar: Grammar, text: str, number: int) -> None:
    assert understood_number(text, grammar) == number


@pytest.mark.parametrize(
    "text",
    [
        "Irons.", "Rons.", "Drogue!", "Estou mais...", "Rúmas...", "Rumor...", "Outra é isso.",
        "Dô, dô, dô, dô,", "", "12 e 13",
    ],
)  # fmt: skip
def test_not_understood(grammar: Grammar, text: str) -> None:
    """As falas que o Yuri não entendeu de ouvido: o juiz também reprova."""
    assert understood_number(text, grammar) is None


def test_distractor() -> None:
    assert says_no_number("Vai, vai, vai!")
    assert not says_no_number("Duzentos")
    assert not says_no_number("233")
