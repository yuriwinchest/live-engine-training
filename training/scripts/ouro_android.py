"""Gera o arquivo "ouro" do decodificador: o SDK Android precisa decidir exatamente como o Python.

Uso (em training/):
    uv run python scripts/ouro_android.py --package data/models/v1
        --manifest data/ds-v1/manifest.jsonl --out <arquivo>
Os casos usam saídas reais do modelo (clipes de validação, voz sintética) e emissões artificiais com respostas
conhecidas. Nenhum áudio vai no arquivo, só log-probabilidades.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from live_lab import audio_io
from live_lab.decode import ACCEPT_ALL, CompiledTrie, Thresholds, decide
from live_lab.evaluate import Package
from live_lab.grammar import Grammar
from live_lab.manifest import Split, read_manifest
from live_lab.runtime import log_probs, open_session
from live_lab.seeding import stable_hash
from live_lab.vocab import TOKEN_INDEX, TOKENS


def artificial(text: str) -> np.ndarray:
    rows = [0, 0]
    for word in text.split():
        rows += [TOKEN_INDEX[word]] * 3 + [0, 0]
    probs = np.full((len(rows), len(TOKENS)), 0.1 / (len(TOKENS) - 1))
    probs[np.arange(len(rows)), rows] = 0.9
    return np.log(probs).astype(np.float32)


def case(name: str, lp: np.ndarray, trie: CompiledTrie, thresholds: Thresholds) -> dict[str, object]:
    # o Kotlin recebe os números arredondados: o Python decide sobre exatamente os mesmos
    lp = np.round(lp.astype(np.float64), 5).astype(np.float32)
    loose, strict = decide(lp, trie), decide(lp, trie, thresholds)
    return {
        "nome": name,
        "log_probs": np.round(lp.astype(np.float64), 5).tolist(),
        "sem_rejeicao": {"numero": loose.number, "melhor": loose.best, "posterior": loose.posterior,
                         "margem": loose.margin, "aderencia": loose.adherence},
        "com_limiares": {"numero": strict.number, "melhor": strict.best},
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", type=int, default=12)
    args = parser.parse_args()

    package = Package.open(args.package)
    trie = CompiledTrie(Grammar.build().export_trie())
    session = open_session(args.package / "model.int8.onnx")
    _, examples = read_manifest(args.manifest)
    val = sorted((e for e in examples if e.split is Split.VAL), key=lambda e: stable_hash("ouro", e.id))
    chosen = [e for e in val if e.number is not None][: args.clips - 3] + [
        e for e in val if e.number is None
    ][:3]

    cases = []
    for e in chosen:
        lp = log_probs(session, audio_io.load(args.manifest.parent / e.audio))
        cases.append(case(f"real-{e.number}", lp, trie, package.thresholds))
    for text in ("duzentos e trinta e três", "zero zero sete", "meia meia", "mil e quinhentos"):
        cases.append(case(f"artificial-{text}", artificial(text), trie, package.thresholds))
    t = package.thresholds
    document = {
        "limiares": {"posterior": t.min_posterior, "margem": t.min_margin, "aderencia": t.min_adherence},
        "largura": 8,
        "casos": cases,
        "_aceita_tudo": ACCEPT_ALL.min_posterior,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")
    print(f"{len(cases)} casos → {args.out}")


if __name__ == "__main__":
    main()
