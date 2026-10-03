"""Vocabulário fechado do motor. A ordem define o índice de saída do modelo: só acrescentar no fim
e subir VOCAB_VERSION, nunca reordenar (quebraria todo modelo já treinado)."""

from __future__ import annotations

from typing import Final

VOCAB_VERSION: Final = "1"

BLANK: Final = "<blank>"

TOKENS: Final[tuple[str, ...]] = (
    BLANK,
    "zero", "um", "uma", "dois", "duas", "três", "quatro", "cinco", "seis", "meia", "sete", "oito", "nove",
    "dez", "onze", "doze", "treze", "quatorze", "catorze", "quinze",
    "dezesseis", "dezessete", "dezoito", "dezenove",
    "vinte", "trinta", "quarenta", "cinquenta", "sessenta", "setenta", "oitenta", "noventa",
    "cem", "cento", "duzentos", "trezentos", "quatrocentos", "quinhentos",
    "seiscentos", "setecentos", "oitocentos", "novecentos",
    "mil", "e",
)  # fmt: skip

TOKEN_INDEX: Final[dict[str, int]] = {token: index for index, token in enumerate(TOKENS)}

WORDS: Final[frozenset[str]] = frozenset(TOKENS[1:])


class UnknownWordError(ValueError):
    def __init__(self, word: str) -> None:
        super().__init__(f"palavra fora do vocabulário: {word!r}")
        self.word = word


def tokenize(text: str) -> tuple[str, ...]:
    """Divide uma fala transcrita em tokens; recusa palavra fora do vocabulário."""
    words = tuple(text.lower().replace(",", " ").split())
    for word in words:
        if word not in WORDS:
            raise UnknownWordError(word)
    return words


def to_indices(tokens: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(TOKEN_INDEX[token] for token in tokens)
