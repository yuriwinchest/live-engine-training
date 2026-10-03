"""Converte a transcrição de um reconhecedor genérico (Whisper) no número que ele entendeu.

Usado só para julgar a fala sintética: se um ouvinte genérico não entende o número do rótulo, a fala não
serve para ensinar o nosso modelo. O Yuri ouviu as falas e reprovou as mesmas que o Whisper reprovou.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from live_lab.grammar import Grammar
from live_lab.vocab import WORDS

MAX_FOREIGN_WORDS: Final = 1  # "A doze." passa; "Outra é isso." não

_ACCENTLESS: Final = {"tres": "três"}


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def _words(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [_ACCENTLESS.get(_strip_accents(w), w) for w in cleaned.split()]


def understood_number(text: str, grammar: Grammar) -> int | None:
    """Número entendido, ou None se a transcrição não forma exatamente um número."""
    compact = re.sub(r"(?<=\d)[.\s](?=\d{3}\b)", "", text.strip())  # "1.500" e "1 500" → "1500"
    groups = re.findall(r"\d+", compact)
    words = [w for w in _words(compact) if not w.isdigit()]
    known = [w for w in words if w in WORDS]
    if len(words) - len(known) > MAX_FOREIGN_WORDS:
        return None
    if groups:
        return int(groups[0]) if len(groups) == 1 and not known else None
    return grammar.parse(tuple(known)) if known else None


def says_no_number(text: str) -> bool:
    """Para frases distratoras: aprovada se nenhum número foi entendido."""
    return not re.search(r"\d", text) and not any(w in WORDS - {"e", "um", "uma"} for w in _words(text))
