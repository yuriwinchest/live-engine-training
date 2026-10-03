"""Registro de fontes de dados e política de licenças (research R3, FR-007).

Fonte recusada não é processada: o modelo treinado é o ativo comercial do projeto, e uma licença
não comercial no treino contaminaria esse licenciamento.
"""

from __future__ import annotations

import tomllib
import uuid
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

ALLOWED_LICENSES: Final = frozenset(
    {"CC0-1.0", "CC-BY-4.0", "Apache-2.0", "MIT", "public-domain", "own-recording-consented"}
)
ATTRIBUTION_REQUIRED: Final = frozenset({"CC-BY-4.0"})
REAL_SPEECH_LICENSE: Final = "own-recording-consented"

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
PRIVATE_DATA_DIR: Final = REPO_ROOT / "training" / "data"


class SourceKind(StrEnum):
    SPEECH_SYNTHETIC = "speech_synthetic"
    SPEECH_REAL = "speech_real"
    NOISE = "noise"
    DISTRACTOR = "distractor"


class SourceRejectedError(ValueError):
    def __init__(self, reasons: dict[str, list[str]]) -> None:
        lines = [f"{name}: {'; '.join(why)}" for name, why in reasons.items()]
        super().__init__("fontes recusadas → " + " | ".join(lines))
        self.reasons = reasons


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    kind: SourceKind
    license: str
    origin: str
    obtained_at: date
    path: Path | None = None
    attribution: str | None = None
    consent_ref: str | None = None


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def problems(source: Source) -> list[str]:
    """Motivos de recusa; lista vazia = fonte aceita."""
    found: list[str] = []
    if source.license not in ALLOWED_LICENSES:
        found.append(f"licença {source.license!r} fora da lista de permissão")
    if source.license in ATTRIBUTION_REQUIRED and not source.attribution:
        found.append("licença exige atribuição e o campo 'attribution' está vazio")
    is_real = source.kind is SourceKind.SPEECH_REAL
    if is_real != (source.license == REAL_SPEECH_LICENSE):
        found.append(f"voz real exige licença {REAL_SPEECH_LICENSE!r}, e só ela")
    if is_real and not _is_uuid(source.consent_ref):
        found.append("voz real exige 'consent_ref' (UUID do termo de consentimento)")
    if source.path is not None:
        resolved = source.path.resolve()
        if _inside(resolved, REPO_ROOT) and not _inside(resolved, PRIVATE_DATA_DIR):
            found.append("caminho dentro do repositório versionado (use training/data/ ou fora do repo)")
        elif not resolved.exists():
            found.append(f"caminho inexistente: {resolved}")
    return found


def _is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Registry:
    sources: dict[str, Source]

    def __getitem__(self, name: str) -> Source:
        try:
            return self.sources[name]
        except KeyError:
            raise ValueError(f"fonte {name!r} não está no registro") from None

    def of_kind(self, kind: SourceKind) -> list[Source]:
        return [s for s in self.sources.values() if s.kind is kind]

    def check(self) -> dict[str, list[str]]:
        return {name: why for name, s in self.sources.items() if (why := problems(s))}

    def require_valid(self) -> None:
        rejected = self.check()
        if rejected:
            raise SourceRejectedError(rejected)


def _source_from(entry: dict[str, Any], base: Path) -> Source:
    raw_path = entry.get("path")
    path = None if raw_path is None else (base / str(raw_path))
    obtained = entry.get("obtained_at")
    if not isinstance(obtained, date):
        raise ValueError(f"fonte {entry.get('name')!r}: 'obtained_at' precisa ser data ISO (AAAA-MM-DD)")
    return Source(
        name=str(entry["name"]),
        kind=SourceKind(entry["kind"]),
        license=str(entry.get("license", "")),
        origin=str(entry.get("origin", "")),
        obtained_at=obtained,
        path=path,
        attribution=entry.get("attribution"),
        consent_ref=entry.get("consent_ref"),
    )


def load_registry(path: Path) -> Registry:
    """Lê o TOML `[[source]]`; caminhos relativos partem da pasta do registro."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    sources: dict[str, Source] = {}
    for entry in data.get("source", []):
        source = _source_from(entry, path.parent)
        if source.name in sources:
            raise ValueError(f"fonte duplicada no registro: {source.name!r}")
        sources[source.name] = source
    return Registry(sources)
