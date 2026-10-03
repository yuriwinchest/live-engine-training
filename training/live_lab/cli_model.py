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
