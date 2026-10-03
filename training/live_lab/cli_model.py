"""Subcomandos da feature 002: `noise holdout` (e, a seguir, `train`, `export`, `evaluate`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from live_lab.holdout import build_playback
from live_lab.sources import load_registry


def register(commands: Any) -> None:
    noise = commands.add_parser("noise", help="ruído reservado").add_subparsers(
        dest="noise_command", required=True
    )
    holdout = noise.add_parser("holdout", help="faixas de multidão/rua para a gravação real")
    holdout.add_argument("--registry", type=Path, required=True)
    holdout.add_argument("--out", type=Path, required=True)
    holdout.add_argument("--seed", type=int, default=1)
    holdout.add_argument("--tracks", type=int, default=6)
    holdout.add_argument("--minutes", type=float, default=3.0)
    holdout.add_argument("--script-count", type=int, default=100)
    holdout.set_defaults(handler=_holdout)

    train = commands.add_parser("train", help="treina o modelo acústico (retoma se houver last.pt)")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-seconds", type=float, default=240.0)
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--channels", type=int, default=192)
    train.add_argument("--blocks", type=int, default=6)
    train.add_argument("--init-from", type=Path)
    train.set_defaults(handler=_train)

    export = commands.add_parser("export", help="ONNX INT8 + calibração dos limiares + pacote do modelo")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--min-accuracy", type=float, default=0.97)
    export.add_argument("--val-limit", type=int, default=2000)
    export.set_defaults(handler=_export)

    evaluate = commands.add_parser("evaluate", help="avalia o pacote (ONNX INT8) numa partição")
    evaluate.add_argument("--package", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--split", choices=["val", "test"], required=True)
    evaluate.add_argument("--inscritos", type=Path)
    evaluate.add_argument("--limit", type=int)
    evaluate.set_defaults(handler=_evaluate)


def _holdout(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    registry.require_valid()
    result = build_playback(registry, args.out, args.seed, args.tracks, args.minutes, args.script_count)
    print(f"{result.reserved} arquivos reservados; {len(result.tracks)} faixas; roteiro com "
          f"{len(result.script)} apertos → {args.out}", file=sys.stderr)  # fmt: skip
    return 0


def _train(args: argparse.Namespace) -> int:
    from live_lab.model.network import NetConfig
    from live_lab.train.loop import TrainConfig, train

    config = TrainConfig(
        manifest=args.manifest, out_dir=args.out, seed=args.seed, epochs=args.epochs, patience=args.patience,
        batch_seconds=args.batch_seconds, workers=args.workers, init_from=args.init_from,
        net=NetConfig(channels=args.channels, blocks=args.blocks),
    )  # fmt: skip
    history = train(config)
    best = max(history, key=lambda h: h.val_accuracy) if history else None
    if best is not None:
        print(
            f"melhor época {best.epoch}: acerto {best.val_accuracy:.1%}, errado {best.val_wrong_rate:.1%}",
            file=sys.stderr,
        )
    return 0


def _export(args: argparse.Namespace) -> int:
    from live_lab.export import build_package

    result = build_package(args.checkpoint, args.manifest, args.out, args.min_accuracy, args.val_limit)
    metrics, measured = result.meta["metrics"], result.meta["bench"]
    totals = metrics["val_calibrated_int8"]
    print(
        f"pacote em {result.package}: {measured['size_bytes'] / 2**20:.2f} MB, "
        f"{measured['latency_ms_3s_1thread']} ms/3 s (1 thread) | validação calibrada: "
        f"acerto {totals['accuracy']:.1%}, errado {totals['wrong_rate']:.2%}, "
        f"rejeição {totals['reject_rate']:.1%}"
        f" | meta {'cumprida' if metrics['target_met'] else 'NÃO cumprida'}",
        file=sys.stderr,
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import json

    from live_lab.cli_grammar import read_enrollment
    from live_lab.evaluate import Package, evaluate_package, to_markdown
    from live_lab.manifest import Split

    enrolled = read_enrollment(args.inscritos) if args.inscritos else None
    report = evaluate_package(
        Package.open(args.package), args.manifest, Split(args.split), enrolled, args.limit
    )
    name = f"avaliacao-{args.split}{'-inscritos' if enrolled else ''}"
    (args.package / f"{name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown = to_markdown(f"Avaliação: {args.split}", report)
    (args.package / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0
