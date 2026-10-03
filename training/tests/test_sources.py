from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CONSENT, make_source

from live_lab.sources import REPO_ROOT, SourceKind, SourceRejectedError, load_registry, problems


def test_accepts_cc0(tmp_path: Path) -> None:
    assert problems(make_source(tmp_path)) == []


@pytest.mark.parametrize("license_id", ["CC-BY-NC-4.0", "", "CPML", "GPL-3.0"])
def test_rejects_licenses_outside_allowlist(tmp_path: Path, license_id: str) -> None:
    assert any("licença" in p for p in problems(make_source(tmp_path, license=license_id)))


def test_cc_by_requires_attribution(tmp_path: Path) -> None:
    assert problems(make_source(tmp_path, license="CC-BY-4.0"))
    assert problems(make_source(tmp_path, license="CC-BY-4.0", attribution="MUSAN, Snyder et al.")) == []


def test_real_voice_requires_consent(tmp_path: Path) -> None:
    real = {"kind": SourceKind.SPEECH_REAL, "license": "own-recording-consented"}
    assert problems(make_source(tmp_path, **real))
    assert problems(make_source(tmp_path, **real, consent_ref="não-é-uuid"))
    assert problems(make_source(tmp_path, **real, consent_ref=CONSENT)) == []


def test_consented_license_is_only_for_real_voice(tmp_path: Path) -> None:
    assert problems(make_source(tmp_path, license="own-recording-consented"))


def test_rejects_path_inside_versioned_repo() -> None:
    assert any("repositório" in p for p in problems(make_source(REPO_ROOT / "docs")))


def test_rejects_missing_path(tmp_path: Path) -> None:
    assert any("inexistente" in p for p in problems(make_source(tmp_path / "nada")))


def test_registry_round_trip_and_rejection(tmp_path: Path) -> None:
    (tmp_path / "musan").mkdir()
    registry_file = tmp_path / "fontes.toml"
    registry_file.write_text(
        """
[[source]]
name = "musan-noise"
kind = "noise"
license = "CC-BY-4.0"
origin = "https://www.openslr.org/17/"
obtained_at = 2026-10-03
path = "musan"
attribution = "MUSAN (Snyder, Chen, Povey, 2015)"

[[source]]
name = "urbansound"
kind = "noise"
license = "CC-BY-NC-4.0"
origin = "x"
obtained_at = 2026-10-03
""",
        encoding="utf-8",
    )
    registry = load_registry(registry_file)
    assert registry["musan-noise"].path == tmp_path / "musan"
    assert set(registry.check()) == {"urbansound"}
    with pytest.raises(SourceRejectedError):
        registry.require_valid()
