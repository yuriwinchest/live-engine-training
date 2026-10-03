"""Ponta a ponta com motor falso e ruído gerado: SC-004, SC-005, SC-007 e a regra da partição de teste."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from conftest import CONSENT

from live_lab import audio_io, report
from live_lab.cli import main
from live_lab.holdout import is_reserved
from live_lab.ingest_real import ingest
from live_lab.manifest import Split, read_manifest
from live_lab.prep import PrepConfig, run
from live_lab.sources import load_registry
from live_lab.synth.base import FakeEngine
from live_lab.synth.planner import plan_jobs
from live_lab.synth.render import CLEAN_MANIFEST, render

REGISTRY = """
[[source]]
name = "fake-tts"
kind = "speech_synthetic"
license = "MIT"
origin = "motor falso dos testes"
obtained_at = 2026-10-03

[[source]]
name = "ruido-gerado"
kind = "noise"
license = "CC-BY-4.0"
origin = "gerado no teste"
obtained_at = 2026-10-03
path = "ruido"
attribution = "ruído gaussiano do teste"

[[source]]
name = "gravacao-po"
kind = "speech_real"
license = "own-recording-consented"
origin = "gravação do PO"
obtained_at = 2026-10-03
consent_ref = "{consent}"
"""


@pytest.fixture
def workspace(tmp_path: Path, noise_dir: Path) -> Path:
    (tmp_path / "fontes.toml").write_text(REGISTRY.format(consent=CONSENT), encoding="utf-8")
    engine = FakeEngine(voice_count=10)
    render(plan_jobs(120, engine.voices(), seed=3), engine, tmp_path / "clean")
    return tmp_path


def _prep(workspace: Path, out: str, manifests: list[Path]) -> PrepConfig:
    registry = load_registry(workspace / "fontes.toml")
    config = PrepConfig(
        clean_manifests=manifests, out_dir=workspace / out, seed=11, copies=2, val_fraction=0.2
    )
    report.write_outputs(run(config, registry), registry, config)
    return config


def test_render_is_resumable(workspace: Path) -> None:
    engine = FakeEngine(voice_count=10)
    assert render(plan_jobs(120, engine.voices(), seed=3), engine, workspace / "clean") == 0


def test_same_seed_same_bytes(workspace: Path) -> None:
    """SC-004."""
    manifests = [workspace / "clean" / CLEAN_MANIFEST]
    first, second = _prep(workspace, "a", manifests), _prep(workspace, "b", manifests)
    assert (first.out_dir / "manifest.jsonl").read_bytes() == (second.out_dir / "manifest.jsonl").read_bytes()
    files_a = sorted(p.relative_to(first.out_dir) for p in (first.out_dir / "audio").rglob("*.wav"))
    assert files_a == sorted(p.relative_to(second.out_dir) for p in (second.out_dir / "audio").rglob("*.wav"))
    for relative in files_a:
        assert (first.out_dir / relative).read_bytes() == (second.out_dir / relative).read_bytes()


def test_partitions_and_negatives(workspace: Path) -> None:
    config = _prep(workspace, "ds", [workspace / "clean" / CLEAN_MANIFEST])
    header, examples = read_manifest(config.out_dir / "manifest.jsonl")
    assert header.provisional_test is True
    assert "ruido-gerado" in header.sources

    speakers = {split: {e.speaker for e in examples if e.split is split and e.speaker != "sem-locutor"}
                for split in Split}  # fmt: skip
    assert not speakers[Split.TRAIN] & speakers[Split.VAL]  # SC-005
    assert not speakers[Split.TEST]
    noisy = [e for e in examples if e.noise_source]
    assert noisy and all(e.noise_file and not is_reserved(e.noise_file) for e in noisy)  # SC-008 da 002

    train = [e for e in examples if e.split is Split.TRAIN]
    share = sum(1 for e in train if e.number is None) / len(train)
    assert 0.10 <= share <= 0.20  # SC-007

    for example in examples[:20]:
        assert (config.out_dir / example.audio).exists()
    assert "MUSAN" not in (config.out_dir / "ATRIBUICOES.md").read_text(encoding="utf-8")
    assert "Teste provisório" in (config.out_dir / "report.md").read_text(encoding="utf-8")


def test_real_recording_goes_to_test_untouched(workspace: Path, tmp_path: Path) -> None:
    recordings = tmp_path / "gravacoes"
    rng = np.random.default_rng(0)
    for name in ("REC_PO_001.wav", "REC_PO_002.wav"):
        audio_io.save(recordings / name, (0.3 * rng.standard_normal(audio_io.SAMPLE_RATE)).astype(np.float32))
    labels = tmp_path / "rotulos.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ("arquivo", "numero", "fala"),
                ("REC_PO_001.wav", "233", "dois três três"),
                ("REC_PO_002.wav", "", ""),
            ]
        )
    assert ingest(labels, recordings, workspace / "real", "gravacao-po") == 2
    assert "REC_PO" not in (workspace / "real" / CLEAN_MANIFEST).read_text(encoding="utf-8")

    config = _prep(
        workspace, "ds-real", [workspace / "clean" / CLEAN_MANIFEST, workspace / "real" / CLEAN_MANIFEST]
    )
    header, examples = read_manifest(config.out_dir / "manifest.jsonl")
    test = [e for e in examples if e.split is Split.TEST]
    assert header.provisional_test is False
    assert {e.number for e in test} == {233, None}
    assert all(e.snr_level == "limpo" and e.source == "gravacao-po" for e in test)


def test_real_voice_refused_in_cloud(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "x")
    with pytest.raises(ValueError, match="local"):
        ingest(tmp_path / "x.csv", tmp_path, tmp_path / "out", "gravacao-po")


def test_cli_rejects_non_commercial_source(tmp_path: Path, noise_dir: Path) -> None:
    registry = tmp_path / "fontes.toml"
    registry.write_text(
        '[[source]]\nname = "esc50"\nkind = "noise"\nlicense = "CC-BY-NC-3.0"\n'
        'origin = "x"\nobtained_at = 2026-10-03\npath = "ruido"\n',
        encoding="utf-8",
    )
    assert main(["sources", "check", "--registry", str(registry)]) == 2


def test_cli_grammar(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["grammar", "parse", "zero zero sete"]) == 0
    assert capsys.readouterr().out.strip() == "7"
    assert main(["grammar", "parse", "trinta e"]) == 3
    assert main(["grammar", "parse", "banana"]) == 2
    enrolled = tmp_path / "inscritos.txt"
    enrolled.write_text("7\n233\n\n1500\n", encoding="utf-8")
    assert main(["grammar", "export", "--out", str(tmp_path / "g.json"), "--inscritos", str(enrolled)]) == 0
    assert main(["grammar", "forms", "10000"]) == 2
