"""Subcomandos `grammar parse|forms|export`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from live_lab.grammar import Grammar
from live_lab.vocab import tokenize

EXIT_REJECTED = 3


def register(commands: Any) -> None:
    grammar = commands.add_parser("grammar", help="gramática de números").add_subparsers(
        dest="grammar_command", required=True
    )

    parse = grammar.add_parser("parse", help="fala → número")
    parse.add_argument("text")
    parse.set_defaults(handler=_parse)

    forms = grammar.add_parser("forms", help="número → falas válidas")
    forms.add_argument("number", type=int)
    forms.set_defaults(handler=_forms)

    export = grammar.add_parser("export", help="exporta a trie JSON")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--inscritos", type=Path, help="arquivo com um inteiro por linha")
    export.set_defaults(handler=_export)


def read_enrollment(path: Path) -> list[int]:
    numbers: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.isdigit():
            raise ValueError(f"{path.name}:{line_number}: não é inteiro: {stripped!r}")
        numbers.append(int(stripped))
    return numbers


def _parse(args: argparse.Namespace) -> int:
    number = Grammar.build().parse(tokenize(args.text))
    if number is None:
        print("REJEITADO")
        return EXIT_REJECTED
    print(number)
    return 0


def _forms(args: argparse.Namespace) -> int:
    for form in Grammar.build().forms(args.number):
        print(" ".join(form))
    return 0


def _export(args: argparse.Namespace) -> int:
    grammar = Grammar.build()
    if args.inscritos is not None:
        grammar = grammar.restrict(read_enrollment(args.inscritos))
    trie = grammar.export_trie()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trie, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{trie['numbers']} números, {len(trie['nodes'])} nós → {args.out}", file=sys.stderr)
    return 0
