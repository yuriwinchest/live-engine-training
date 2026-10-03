"""Subcomandos de dados: `sources check`, `synth plan|render`, `ingest-real`, `prep`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from live_lab import report
from live_lab.ingest_real import ingest
from live_lab.prep import PrepConfig, run
from live_lab.sources import load_registry
from live_lab.synth.base import get_engine
from live_lab.synth.planner import plan_jobs, read_jobs, write_jobs
from live_lab.synth.render import render


def _engine_options(raw: list[str] | None) -> dict[str, str]:
    options: dict[str, str] = {}
    for item in raw or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"opção de motor sem '=': {item!r}")
        options[key] = value
    return options


def register(commands: Any) -> None:
    sources = commands.add_parser("sources", help="registro de fontes").add_subparsers(
        dest="sources_command", required=True
    )
    check = sources.add_parser("check", help="valida licenças e caminhos")
    check.add_argument("--registry", type=Path, required=True)
    check.set_defaults(handler=_sources_check)

    synth = commands.add_parser("synth", help="voz sintética").add_subparsers(
        dest="synth_command", required=True
    )
    for name, handler in (("plan", _synth_plan), ("render", _synth_render)):
        sub = synth.add_parser(name)
        sub.add_argument("--engine", required=True)
        sub.add_argument("--engine-option", action="append", metavar="CHAVE=VALOR")
        sub.set_defaults(handler=handler)
        if name == "plan":
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--count", type=int, required=True)
            sub.add_argument("--seed", type=int, required=True)
        else:
            sub.add_argument("--jobs", type=Path, required=True)
            sub.add_argument("--out", type=Path, required=True)

    real = commands.add_parser("ingest-real", help="gravação real do PO → teste (só local)")
    real.add_argument("--labels", type=Path, required=True)
    real.add_argument("--audio", type=Path, required=True)
    real.add_argument("--out", type=Path, required=True)
    real.add_argument("--source", required=True, help="nome da fonte speech_real no registro")
    real.set_defaults(handler=_ingest_real)

    prep = commands.add_parser("prep", help="degrada, separa e gera o manifesto")
    prep.add_argument("--registry", type=Path, required=True)
    prep.add_argument("--clean", type=Path, nargs="+", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--copies", type=int, default=4)
    prep.add_argument("--val-fraction", type=float, default=0.1)
    prep.add_argument("--negative-share", type=float, default=0.15)
    prep.set_defaults(handler=_prep)


def _sources_check(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    rejected = registry.check()
    for name in sorted(registry.sources):
        status = "RECUSADA: " + "; ".join(rejected[name]) if name in rejected else "ok"
        print(f"{name}: {status}")
    return 2 if rejected else 0


def _synth_plan(args: argparse.Namespace) -> int:
    engine = get_engine(args.engine, **_engine_options(args.engine_option))
    jobs = plan_jobs(args.count, engine.voices(), args.seed)
    write_jobs(args.out, jobs)
    print(f"{len(jobs)} trabalhos → {args.out}", file=sys.stderr)
    return 0


def _synth_render(args: argparse.Namespace) -> int:
    engine = get_engine(args.engine, **_engine_options(args.engine_option))
    written = render(read_jobs(args.jobs), engine, args.out)
    print(f"{written} clipes novos em {args.out}", file=sys.stderr)
    return 0


def _ingest_real(args: argparse.Namespace) -> int:
    count = ingest(args.labels, args.audio, args.out, args.source)
    print(f"{count} clipes reais → {args.out}", file=sys.stderr)
    return 0


def _prep(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    config = PrepConfig(
        clean_manifests=args.clean,
        out_dir=args.out,
        seed=args.seed,
        copies=args.copies,
        val_fraction=args.val_fraction,
        negative_share=args.negative_share,
    )
    result = run(config, registry)
    summary = report.write_outputs(result, registry, config)
    for name, data in summary["splits"].items():
        print(f"{name}: {data['examples']} exemplos, {data['hours']} h", file=sys.stderr)
    return 0
