from __future__ import annotations

import json

import pytest

from live_lab.grammar import Grammar, GrammarCollisionError, InvalidEnrollmentError
from live_lab.numbers_pt import MAX_NUMBER, spoken_forms
from live_lab.vocab import BLANK, TOKEN_INDEX, TOKENS


@pytest.fixture(scope="module")
def grammar() -> Grammar:
    return Grammar.build()


def words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("duzentos e trinta e três", 233),
        ("dois três três", 233),
        ("zero zero sete", 7),
        ("meia meia", 66),
        ("mil e quinhentos", 1500),
        ("quatorze", 14),
        ("catorze", 14),
    ],
)
def test_parse_accepts(grammar: Grammar, text: str, number: int) -> None:
    assert grammar.parse(words(text)) == number


@pytest.mark.parametrize(
    "text",
    [
        "",
        "trinta e",
        "vinte dez",
        "cem e um",
        "dois trinta e três",
        "zero zero zero zero sete",
        "e",
        "mil mil",
    ],
)
def test_parse_rejects(grammar: Grammar, text: str) -> None:
    assert grammar.parse(words(text)) is None


def test_exhaustive_round_trip(grammar: Grammar) -> None:
    """SC-001: toda forma de todo inteiro volta exatamente ao inteiro."""
    for number in range(MAX_NUMBER + 1):
        for form in spoken_forms(number):
            assert grammar.parse(form) == number, (number, form)


def test_table_matches_generator(grammar: Grammar) -> None:
    """SC-002: a tabela é exatamente a união das formas; nenhuma fala serve a dois inteiros."""
    total = sum(len(spoken_forms(n)) for n in range(MAX_NUMBER + 1))
    assert len(grammar) == total
    assert grammar.numbers == frozenset(range(MAX_NUMBER + 1))


def test_collision_aborts_build() -> None:
    def colliding(number: int) -> list[tuple[str, ...]]:
        return [("um",)]

    with pytest.raises(GrammarCollisionError):
        Grammar.build(numbers=range(2), forms_of=colliding)


def test_forms_of_1200(grammar: Grammar) -> None:
    forms = grammar.forms(1200)
    assert words("mil e duzentos") in forms
    assert words("um dois zero zero") in forms


def test_trie_export_matches_parse(grammar: Grammar) -> None:
    trie = grammar.export_trie()
    assert trie["format"] == "live-grammar-trie"
    assert trie["tokens"] == list(TOKENS)
    assert trie["restricted"] is False
    assert trie["numbers"] == MAX_NUMBER + 1
    json.dumps(trie)

    nodes = trie["nodes"]
    blank = str(TOKEN_INDEX[BLANK])
    assert all(blank not in node["next"] for node in nodes)

    def walk(tokens: tuple[str, ...]) -> int | None:
        node = 0
        for token in tokens:
            nxt = nodes[node]["next"].get(str(TOKEN_INDEX[token]))
            if nxt is None:
                return None
            node = nxt
        value = nodes[node]["value"]
        return int(value) if value is not None else None

    for form, number in grammar.items():
        assert walk(form) == number
    assert walk(words("trinta e")) is None
    assert walk(words("dois")) == 2
    assert nodes[nodes[0]["next"][str(TOKEN_INDEX["dois"])]]["next"]


class TestEnrollment:
    def test_only_enrolled_accepted(self, grammar: Grammar) -> None:
        restricted = grammar.restrict([7, 233, 1500])
        assert restricted.parse(words("mil e quinhentos")) == 1500
        assert restricted.parse(words("zero zero sete")) == 7
        assert restricted.parse(words("duzentos e trinta e quatro")) is None
        assert grammar.parse(words("duzentos e trinta e quatro")) == 234

    def test_exhaustive_restriction(self, grammar: Grammar) -> None:
        """SC-003: nenhum número fora da lista é aceito."""
        enrolled = frozenset(range(0, MAX_NUMBER + 1, 37))
        restricted = grammar.restrict(enrolled)
        assert restricted.numbers == enrolled
        for form, _ in grammar.items():
            result = restricted.parse(form)
            assert result is None or result in enrolled

    def test_restricted_trie(self, grammar: Grammar) -> None:
        trie = grammar.restrict([7, 233]).export_trie()
        assert trie["restricted"] is True
        assert trie["numbers"] == 2

    @pytest.mark.parametrize("enrolled", [[], [-1], [MAX_NUMBER + 1], [5, 10_000]])
    def test_invalid_enrollment(self, grammar: Grammar, enrolled: list[int]) -> None:
        with pytest.raises(InvalidEnrollmentError):
            grammar.restrict(enrolled)
