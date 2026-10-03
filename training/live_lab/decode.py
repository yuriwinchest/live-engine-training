"""Decodificação CTC restrita à gramática, com rejeição (research R4 da 002).

Numpy puro, sem torch: é a referência que o SDK Android porta linha a linha. Cada hipótese do feixe é um nó
da trie exportada pela feature 001; como a trie é uma árvore de prefixos, o nó identifica o prefixo inteiro,
e só tokens com filho na trie podem estender uma hipótese. Saída fora da gramática é impossível por
construção.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

NEG_INF: Final = -np.inf
BLANK: Final = 0


@dataclass(frozen=True, slots=True)
class Thresholds:
    min_posterior: float = 0.0
    min_margin: float = 0.0
    min_adherence: float = -np.inf


ACCEPT_ALL: Final = Thresholds()


@dataclass(frozen=True, slots=True)
class Decision:
    number: int | None
    best: int | None
    posterior: float
    margin: float
    adherence: float
    candidates: list[tuple[int | None, float]] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return self.number is None


class CompiledTrie:
    """Trie do contrato `live-grammar-trie` em listas indexadas por nó (acesso rápido no laço)."""

    def __init__(self, trie: Mapping[str, Any]) -> None:
        if trie.get("format") != "live-grammar-trie":
            raise ValueError("formato de gramática desconhecido")
        nodes = trie["nodes"]
        self.children: list[dict[int, int]] = [{int(k): v for k, v in n["next"].items()} for n in nodes]
        self.values: list[int | None] = [n["value"] for n in nodes]
        self.token_into: list[int] = [BLANK] * len(nodes)
        for children in self.children:
            for token, child in children.items():
                self.token_into[child] = token
        self.vocab_size = len(trie["tokens"])


def _add(a: float, b: float) -> float:
    return float(np.logaddexp(a, b))


def prefix_beam_search(
    log_probs: NDArray[np.floating], trie: CompiledTrie, beam_width: int = 8
) -> dict[int, float]:
    """Probabilidade (log) de cada prefixo sobrevivente, somada sobre todos os alinhamentos CTC."""
    if log_probs.ndim != 2 or log_probs.shape[1] != trie.vocab_size:
        raise ValueError(f"log_probs precisa ser [quadros, {trie.vocab_size}]")
    beams: dict[int, tuple[float, float]] = {0: (0.0, NEG_INF)}  # nó → (termina em branco, não branco)
    for frame in np.asarray(log_probs, dtype=np.float64):
        nxt: dict[int, list[float]] = {}
        for node, (p_blank, p_token) in beams.items():
            total = _add(p_blank, p_token)
            entry = nxt.setdefault(node, [NEG_INF, NEG_INF])
            entry[0] = _add(entry[0], total + frame[BLANK])
            if node != 0:
                last = trie.token_into[node]
                entry[1] = _add(entry[1], p_token + frame[last])
            for token, child in trie.children[node].items():
                # Mesmo token repetido só abre palavra nova se houve branco entre eles (regra do CTC).
                source = p_blank if node != 0 and token == trie.token_into[node] else total
                target = nxt.setdefault(child, [NEG_INF, NEG_INF])
                target[1] = _add(target[1], source + frame[token])
        ranked = sorted(nxt.items(), key=lambda item: _add(*item[1]), reverse=True)[:beam_width]
        beams = {node: (pb, pt) for node, (pb, pt) in ranked}
    return {node: _add(pb, pt) for node, (pb, pt) in beams.items()}


def decide(
    log_probs: NDArray[np.floating],
    trie: CompiledTrie,
    thresholds: Thresholds = ACCEPT_ALL,
    beam_width: int = 8,
) -> Decision:
    frames = len(log_probs)
    if frames == 0:
        return Decision(None, None, 0.0, 0.0, NEG_INF)
    scores = prefix_beam_search(log_probs, trie, beam_width)
    empty = float(np.sum(log_probs[:, BLANK], dtype=np.float64))
    candidates: dict[int | None, float] = {None: empty}
    for node, score in scores.items():
        value = trie.values[node]
        if value is not None:
            candidates[value] = max(candidates.get(value, NEG_INF), score)
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else NEG_INF
    log_total = float(np.logaddexp.reduce([s for _, s in ranked]))
    posterior = float(np.exp(best_score - log_total))
    margin = float(best_score - second)
    free_path = float(np.sum(np.max(log_probs, axis=1), dtype=np.float64))
    adherence = (best_score - free_path) / frames
    accepted = (
        best is not None
        and posterior >= thresholds.min_posterior
        and margin >= thresholds.min_margin
        and adherence >= thresholds.min_adherence
    )
    return Decision(best if accepted else None, best, posterior, margin, adherence, ranked[:3])
