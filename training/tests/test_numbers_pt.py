from __future__ import annotations

import pytest

from live_lab.numbers_pt import MAX_NUMBER, digit_forms, spoken_forms, worded_forms


def words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, "zero"),
        (6, "seis"),
        (14, "quatorze"),
        (14, "catorze"),
        (21, "vinte e um"),
        (21, "vinte e uma"),
        (100, "cem"),
        (101, "cento e um"),
        (200, "duzentos"),
        (233, "duzentos e trinta e três"),
        (1000, "mil"),
        (1000, "um mil"),
        (1001, "mil e um"),
        (1200, "mil e duzentos"),
        (1200, "mil duzentos"),
        (1230, "mil duzentos e trinta"),
        (1230, "mil e duzentos e trinta"),
        (2000, "dois mil"),
        (9999, "nove mil novecentos e noventa e nove"),
    ],
)
def test_worded_forms_include(number: int, expected: str) -> None:
    assert words(expected) in worded_forms(number)


@pytest.mark.parametrize(
    ("number", "rejected"),
    [
        (100, "cem e"),
        (101, "cem e um"),
        (6, "meia"),
        (36, "trinta e meia"),
        (2000, "duas mil"),
        (23, "vinte três"),
        (1000, "um um mil"),
    ],
)
def test_worded_forms_exclude(number: int, rejected: str) -> None:
    assert words(rejected) not in worded_forms(number)


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (7, "sete"),
        (7, "zero sete"),
        (7, "zero zero sete"),
        (7, "zero zero zero sete"),
        (0, "zero zero zero zero"),
        (66, "meia meia"),
        (66, "seis meia"),
        (233, "dois três três"),
        (1200, "um dois zero zero"),
        (1200, "uma dois zero zero"),
    ],
)
def test_digit_forms_include(number: int, expected: str) -> None:
    assert words(expected) in digit_forms(number)


def test_digit_forms_never_exceed_four_digits() -> None:
    assert all(len(form) <= 4 for form in digit_forms(7))
    assert words("zero zero zero zero sete") not in digit_forms(7)


def test_every_number_has_both_modes() -> None:
    for number in range(MAX_NUMBER + 1):
        assert worded_forms(number), number
        assert digit_forms(number), number


def test_spoken_forms_have_no_duplicates() -> None:
    forms = spoken_forms(7)
    assert len(forms) == len(set(forms))


@pytest.mark.parametrize("number", [-1, MAX_NUMBER + 1])
def test_out_of_range_rejected(number: int) -> None:
    with pytest.raises(ValueError):
        spoken_forms(number)
