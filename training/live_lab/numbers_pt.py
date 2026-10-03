"""Regras do português do Brasil para falar um número de peito (0–9999).

Este módulo é a única definição da linguagem aceita: o conversor fala → número em `grammar.py`
é a tabela inversa destas formas, então os dois nunca divergem.
"""

from __future__ import annotations

from functools import cache
from itertools import product
from typing import Final

MAX_NUMBER: Final = 9999
MAX_DIGITS: Final = 4

Form = tuple[str, ...]

_UNITS: Final[dict[int, tuple[str, ...]]] = {
    1: ("um", "uma"),
    2: ("dois", "duas"),
    3: ("três",),
    4: ("quatro",),
    5: ("cinco",),
    6: ("seis",),
    7: ("sete",),
    8: ("oito",),
    9: ("nove",),
}

_TEENS: Final[dict[int, tuple[str, ...]]] = {
    10: ("dez",),
    11: ("onze",),
    12: ("doze",),
    13: ("treze",),
    14: ("quatorze", "catorze"),
    15: ("quinze",),
    16: ("dezesseis",),
    17: ("dezessete",),
    18: ("dezoito",),
    19: ("dezenove",),
}

_TENS: Final[dict[int, str]] = {
    2: "vinte",
    3: "trinta",
    4: "quarenta",
    5: "cinquenta",
    6: "sessenta",
    7: "setenta",
    8: "oitenta",
    9: "noventa",
}

_HUNDREDS: Final[dict[int, str]] = {
    1: "cento",
    2: "duzentos",
    3: "trezentos",
    4: "quatrocentos",
    5: "quinhentos",
    6: "seiscentos",
    7: "setecentos",
    8: "oitocentos",
    9: "novecentos",
}

# "meia" é o 6 do ditado ("meia meia" = 66); por extenso, "trinta e meia" não é número.
_DIGITS: Final[dict[int, tuple[str, ...]]] = {0: ("zero",), **_UNITS, 6: ("seis", "meia")}


def _check_range(number: int) -> None:
    if not 0 <= number <= MAX_NUMBER:
        raise ValueError(f"número fora de 0–{MAX_NUMBER}: {number}")


def _below_100(number: int) -> list[Form]:
    if number < 10:
        return [(word,) for word in _UNITS[number]]
    if number < 20:
        return [(word,) for word in _TEENS[number]]
    tens, unit = divmod(number, 10)
    if unit == 0:
        return [(_TENS[tens],)]
    return [(_TENS[tens], "e", word) for word in _UNITS[unit]]


def _below_1000(number: int) -> list[Form]:
    if number < 100:
        return _below_100(number)
    if number == 100:
        return [("cem",)]
    hundreds, rest = divmod(number, 100)
    head = _HUNDREDS[hundreds]
    if rest == 0:
        return [(head,)]
    return [(head, "e", *tail) for tail in _below_100(rest)]


def _thousand_prefixes(thousands: int) -> list[Form]:
    if thousands == 1:
        return [("mil",), ("um", "mil")]
    # "duas mil" não existe: o multiplicador do milhar é sempre masculino.
    return [(_UNITS[thousands][0], "mil")]


@cache
def worded_forms(number: int) -> tuple[Form, ...]:
    """Formas por extenso. Após "mil", o "e" é aceito com e sem (normativo e coloquial)."""
    _check_range(number)
    if number == 0:
        return (("zero",),)
    thousands, rest = divmod(number, 1000)
    if thousands == 0:
        return tuple(_below_1000(rest))
    forms: list[Form] = []
    for prefix in _thousand_prefixes(thousands):
        if rest == 0:
            forms.append(prefix)
            continue
        for tail in _below_1000(rest):
            forms.append((*prefix, "e", *tail))
            forms.append((*prefix, *tail))
    return tuple(forms)


@cache
def digit_forms(number: int) -> tuple[Form, ...]:
    """Formas dígito a dígito, de 1 a 4 algarismos, com zeros à esquerda ("zero zero sete" = 7)."""
    _check_range(number)
    digits = str(number)
    forms: list[Form] = []
    for length in range(len(digits), MAX_DIGITS + 1):
        padded = digits.rjust(length, "0")
        forms.extend(product(*(_DIGITS[int(d)] for d in padded)))
    return tuple(forms)


@cache
def spoken_forms(number: int) -> tuple[Form, ...]:
    """Todas as falas válidas do número, sem repetição ("sete" é extenso e dígito ao mesmo tempo)."""
    return tuple(dict.fromkeys((*worded_forms(number), *digit_forms(number))))
