"""CLI `live-lab`. Contrato em specs/001-gramatica-dataprep/contracts/cli.md.

Códigos de saída: 0 sucesso · 1 erro inesperado · 2 entrada inválida · 3 fala rejeitada.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from live_lab import cli_data, cli_grammar

EXIT_INVALID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="live-lab", description="Laboratório do L.I.V.E.")
    commands = parser.add_subparsers(dest="command", required=True)
    cli_grammar.register(commands)
    cli_data.register(commands)
    return parser


def _utf8_console() -> None:
    # O console do Windows usa cp1252 por padrão e estraga "números" e "→" nas mensagens.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code: int = args.handler(args)
    except ValueError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_INVALID
    return code


if __name__ == "__main__":
    sys.exit(main())
