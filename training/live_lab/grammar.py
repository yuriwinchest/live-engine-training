"""Gramática de números válidos: fala (tokens) → inteiro ou rejeição.

A tabela é construída a partir de `numbers_pt.spoken_forms`; uma fala que sirva a dois inteiros
aborta a construção, porque devolver qualquer um deles seria inventar número.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

from live_lab.numbers_pt import MAX_NUMBER, Form, spoken_forms
from live_lab.vocab import TOKEN_INDEX, TOKENS, VOCAB_VERSION

TRIE_FORMAT: Final = "live-grammar-trie"
TRIE_FORMAT_VERSION: Final = 1


class GrammarCollisionError(RuntimeError):
    def __init__(self, form: Form, first: int, second: int) -> None:
        super().__init__(f"fala {' '.join(form)!r} serviria a {first} e {second}")


class InvalidEnrollmentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Grammar:
    _table: dict[Form, int]
    restricted: bool = False

    @classmethod
    def build(
        cls,
        numbers: Iterable[int] = range(MAX_NUMBER + 1),
        forms_of: Callable[[int], Sequence[Form]] = spoken_forms,
    ) -> Grammar:
        table: dict[Form, int] = {}
        for number in numbers:
            for form in forms_of(number):
                previous = table.setdefault(form, number)
                if previous != number:
                    raise GrammarCollisionError(form, previous, number)
        return cls(table)

    def __len__(self) -> int:
        return len(self._table)

    @property
    def numbers(self) -> frozenset[int]:
        return frozenset(self._table.values())

    def items(self) -> Iterator[tuple[Form, int]]:
        return iter(self._table.items())

    def parse(self, tokens: Sequence[str]) -> int | None:
        return self._table.get(tuple(tokens))

    def forms(self, number: int) -> list[Form]:
        found = [form for form, value in self._table.items() if value == number]
        if not found:
            raise ValueError(f"número fora desta gramática: {number}")
        return found

    def restrict(self, enrolled: Iterable[int]) -> Grammar:
        """Aceita só os inscritos da prova; uma lista vazia ou fora da faixa é erro de configuração."""
        allowed = frozenset(enrolled)
        if not allowed:
            raise InvalidEnrollmentError("lista de inscritos vazia")
        invalid = sorted(n for n in allowed if not 0 <= n <= MAX_NUMBER)
        if invalid:
            raise InvalidEnrollmentError(f"inscritos fora de 0–{MAX_NUMBER}: {invalid[:5]}")
        missing = sorted(allowed - self.numbers)
        if missing:
            raise InvalidEnrollmentError(f"inscritos ausentes desta gramática: {missing[:5]}")
        table = {form: value for form, value in self._table.items() if value in allowed}
        return Grammar(table, restricted=True)

    def export_trie(self) -> dict[str, Any]:
        """Árvore de prefixos sobre índices de token; contrato em contracts/grammar-export.md."""
        nodes: list[dict[str, Any]] = [{"next": {}, "value": None}]
        for form, value in sorted(self._table.items(), key=lambda item: item[0]):
            node = 0
            for token in form:
                key = str(TOKEN_INDEX[token])
                child = nodes[node]["next"].get(key)
                if child is None:
                    child = len(nodes)
                    nodes[node]["next"][key] = child
                    nodes.append({"next": {}, "value": None})
                node = child
            nodes[node]["value"] = value
        return {
            "format": TRIE_FORMAT,
            "format_version": TRIE_FORMAT_VERSION,
            "vocab_version": VOCAB_VERSION,
            "tokens": list(TOKENS),
            "restricted": self.restricted,
            "numbers": len(self.numbers),
            "nodes": nodes,
        }
