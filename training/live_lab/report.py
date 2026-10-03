"""Relatório da preparação (FR-016) e atribuições exigidas por licenças CC-BY."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from live_lab.manifest import Example, Split, write_manifest
from live_lab.prep import PrepConfig, PrepResult
from live_lab.sources import ATTRIBUTION_REQUIRED, Registry
from live_lab.vocab import TOKENS


def _counts(examples: Sequence[Example], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(e, key)) for e in examples).items()))


def word_coverage(examples: Sequence[Example]) -> dict[str, float]:
    """Fração dos exemplos de treino com fala de número que contêm cada palavra (SC-006)."""
    spoken = [e for e in examples if e.split is Split.TRAIN and e.tokens]
    if not spoken:
        return {}
    counts: Counter[str] = Counter(word for e in spoken for word in set(e.tokens or ()))
    return {word: round(counts[word] / len(spoken), 4) for word in TOKENS[1:]}


def build(result: PrepResult) -> dict[str, Any]:
    examples = result.examples
    by_split: dict[str, Any] = {}
    for split in Split:
        part = [e for e in examples if e.split is split]
        without = sum(1 for e in part if e.number is None)
        by_split[split.value] = {
            "examples": len(part),
            "hours": round(sum(e.duration_s for e in part) / 3600, 3),
            "speakers": len({e.speaker for e in part}),
            "without_number_share": round(without / len(part), 4) if part else 0.0,
        }
    return {
        "provisional_test": result.header.provisional_test,
        "splits": by_split,
        "by_source": _counts(examples, "source"),
        "by_mode": _counts(examples, "mode"),
        "by_snr_level": _counts(examples, "snr_level"),
        "word_coverage_train": word_coverage(examples),
        "discarded": [asdict(d) for d in result.discards],
        "notes": result.notes,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# Relatório da preparação", ""]
    if report["provisional_test"]:
        lines += ["> **Teste provisório:** sem gravação real na partição `test` (Princípio V).", ""]
    lines += ["| Partição | Exemplos | Horas | Locutores | Sem número |", "|---|---|---|---|---|"]
    for name, data in report["splits"].items():
        share = f"{data['without_number_share']:.1%}"
        lines.append(f"| {name} | {data['examples']} | {data['hours']} | {data['speakers']} | {share} |")
    for title, key in (("Fonte", "by_source"), ("Modo", "by_mode"), ("Nível de ruído", "by_snr_level")):
        lines += ["", f"| {title} | Exemplos |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in report[key].items()]
    coverage = report["word_coverage_train"]
    if coverage:
        rare = sorted(coverage.items(), key=lambda item: item[1])[:8]
        lines += ["", "Palavras menos vistas no treino: " + ", ".join(f"{w} {s:.1%}" for w, s in rare)]
    lines += ["", f"Descartados: {len(report['discarded'])}"]
    reasons = Counter(item["reason"] for item in report["discarded"])
    lines += [f"- {reason}: {count}" for reason, count in reasons.most_common()]
    lines += [f"- Nota: {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def attributions(registry: Registry, used: Sequence[str]) -> str:
    lines = ["# Atribuições", ""]
    for name in used:
        source = registry[name]
        if source.license in ATTRIBUTION_REQUIRED:
            lines.append(f"- **{source.name}** ({source.license}) — {source.attribution} — {source.origin}")
    return "\n".join(lines) + "\n"


def write_outputs(result: PrepResult, registry: Registry, config: PrepConfig) -> dict[str, Any]:
    out = config.out_dir
    write_manifest(out / "manifest.jsonl", result.header, result.examples)
    report = build(result)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    (out / "ATRIBUICOES.md").write_text(
        attributions(registry, sorted(result.header.sources)), encoding="utf-8"
    )
    run_info = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": config.seed,
        "copies": config.copies,
        "val_fraction": config.val_fraction,
        "negative_share": config.negative_share,
        "clean_manifests": [p.name for p in config.clean_manifests],
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    return report
