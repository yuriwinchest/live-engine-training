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


def _holdout(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    registry.require_valid()
    result = build_playback(registry, args.out, args.seed, args.tracks, args.minutes, args.script_count)
    print(f"{result.reserved} arquivos reservados; {len(result.tracks)} faixas; roteiro com "
          f"{len(result.script)} apertos → {args.out}", file=sys.stderr)  # fmt: skip
    return 0
