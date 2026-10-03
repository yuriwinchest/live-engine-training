"""Avaliação pelo modelo exportado, do ponto de vista do operador (US3 da 002).

Recortes por nível de ruído, modo de fala e quantidade de algarismos; lista de erros para inspeção.
Voz real só é avaliada na máquina local.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from live_lab import audio_io
from live_lab.calibrate import apply
from live_lab.decode import CompiledTrie, Decision, Thresholds
from live_lab.grammar import Grammar
from live_lab.ingest_real import refuse_cloud
from live_lab.manifest import Example, Split, read_manifest
from live_lab.metrics import Outcome, by_group, rates
from live_lab.progress import Progress
from live_lab.runtime import open_session, recognize_all
from live_lab.seeding import stable_hash


@dataclass(frozen=True, slots=True)
class Package:
    root: Path
    meta: dict[str, Any]

    @classmethod
    def open(cls, root: Path) -> Package:
        return cls(root, json.loads((root / "meta.json").read_text(encoding="utf-8")))

    @property
    def thresholds(self) -> Thresholds:
        return Thresholds(**self.meta["thresholds"])


def digits_of(example: Example) -> str:
    return "sem número" if example.number is None else f"{len(str(example.number))} alg."


def halves(examples: Sequence[Example]) -> tuple[list[Example], list[Example]]:
    """Metade para calibrar, metade para medir: medir onde se calibrou seria otimista."""
    first = [e for e in examples if int(stable_hash("half", e.id)[:2], 16) % 2 == 0]
    second = [e for e in examples if int(stable_hash("half", e.id)[:2], 16) % 2 == 1]
    return first, second


def decisions_for(
    model: Path, examples: Sequence[Example], root: Path, trie: CompiledTrie, label: str = "decodificando"
) -> list[Decision]:
    session = open_session(model)
    progress = Progress(len(examples), label)
    decisions: list[Decision] = []
    for example in examples:
        decisions.append(recognize_all(session, [audio_io.load(root / example.audio)], trie)[0])
        progress.tick()
    return decisions


def summarize(
    examples: Sequence[Example], decisions: Sequence[Decision], thresholds: Thresholds, provisional: bool
) -> dict[str, Any]:
    got = apply(decisions, thresholds)
    pairs = list(zip(examples, got, strict=True))

    def outcome(pair: tuple[Example, int | None]) -> Outcome:
        return Outcome(pair[0].number, pair[1])

    errors = [
        {"example_id": e.id, "expected": e.number, "got": g, "posterior": round(d.posterior, 4)}
        for (e, g), d in zip(pairs, decisions, strict=True)
        if g is not None and g != e.number
    ]
    return {
        "provisional": provisional,
        "thresholds": {k: float(v) for k, v in vars_of(thresholds).items()},
        "totals": rates([outcome(p) for p in pairs]).as_dict(),
        "by_snr_level": {
            k: v.as_dict() for k, v in by_group(pairs, lambda p: str(p[0].snr_level), outcome).items()
        },
        "by_mode": {k: v.as_dict() for k, v in by_group(pairs, lambda p: str(p[0].mode), outcome).items()},
        "by_digits": {k: v.as_dict() for k, v in by_group(pairs, lambda p: digits_of(p[0]), outcome).items()},
        "errors": errors[:200],
    }


def vars_of(thresholds: Thresholds) -> dict[str, float]:
    return {
        "min_posterior": thresholds.min_posterior,
        "min_margin": thresholds.min_margin,
        "min_adherence": thresholds.min_adherence,
    }


def to_markdown(title: str, report: dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    if report["provisional"]:
        lines += ["> **Provisório:** medido só com voz sintética (Princípio V).", ""]
    t = report["totals"]
    lines += [
        f"**Total ({t['n']} clipes):** acerto {t['accuracy']:.1%} · número errado {t['wrong_rate']:.2%} · "
        f"rejeição {t['reject_rate']:.1%}",
        "",
    ]
    for title_part, key in (
        ("Nível de ruído", "by_snr_level"),
        ("Modo", "by_mode"),
        ("Algarismos", "by_digits"),
    ):
        lines += [f"| {title_part} | n | Acerto | Errado | Rejeição |", "|---|---|---|---|---|"]
        for name, r in report[key].items():
            lines.append(
                f"| {name} | {r['n']} | {r['accuracy']:.1%} | {r['wrong_rate']:.2%} "
                f"| {r['reject_rate']:.1%} |"
            )
        lines.append("")
    if report["errors"]:
        lines += ["Erros (esperado → devolvido): " + ", ".join(
            f"{e['expected']}→{e['got']}" for e in report["errors"][:30]
        ), ""]  # fmt: skip
    return "\n".join(lines)


def evaluate_package(
    package: Package,
    manifest: Path,
    split: Split,
    enrolled: Sequence[int] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    _, examples = read_manifest(manifest)
    chosen = [e for e in examples if e.split is split]
    if split is Split.TEST:
        refuse_cloud()
    if limit is not None:
        chosen = sorted(chosen, key=lambda e: stable_hash("eval", e.id))[:limit]
    grammar = Grammar.build()
    if enrolled is not None:
        grammar = grammar.restrict(enrolled)
    trie = CompiledTrie(grammar.export_trie())
    decisions = decisions_for(package.root / "model.int8.onnx", chosen, manifest.parent, trie)
    return summarize(chosen, decisions, package.thresholds, provisional=split is not Split.TEST)
