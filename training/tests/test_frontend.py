"""Front-end e rede: forma, paridade com a STFT de referência e exportação ONNX (risco da 002)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from live_lab.model.frontend import HOP, N_FFT, N_MELS, LogMel, frame_count  # noqa: E402
from live_lab.model.network import LiveNet, NetConfig  # noqa: E402
from live_lab.vocab import TOKENS  # noqa: E402


def tone(seconds: float = 1.0, hz: float = 440.0) -> torch.Tensor:
    t = torch.arange(int(seconds * 16_000)) / 16_000
    return (0.5 * torch.sin(2 * torch.pi * hz * t)).unsqueeze(0)


def test_power_matches_torch_stft() -> None:
    audio = tone()
    ours = LogMel().power(audio)
    window = torch.hann_window(N_FFT, periodic=True)
    ref = torch.stft(audio, N_FFT, HOP, window=window, center=True, pad_mode="constant", return_complex=True)
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref.abs() ** 2, rtol=1e-3, atol=1e-3)


def test_logmel_shape_and_peak_band() -> None:
    features = LogMel()(tone(hz=1000.0))
    assert features.shape == (1, N_MELS, frame_count(16_000))
    assert torch.isfinite(features).all()


def test_network_output_shape_and_size() -> None:
    model = LiveNet().eval()
    samples = torch.tensor([16_000, 8_000])
    audio = torch.zeros(2, 16_000)
    audio[0] = tone()[0]
    out = model(audio, samples)
    assert out.shape == (2, int(model.output_frames(torch.tensor(16_000))), len(TOKENS))
    torch.testing.assert_close(out.exp().sum(-1), torch.ones(out.shape[:2]))
    assert model.parameter_count() < 1_500_000  # INT8 bem abaixo de 5 MB (SC-001)


def test_exports_to_onnx_and_matches(tmp_path: Path) -> None:
    """Risco principal: o log-Mel precisa caber no grafo ONNX e dar o mesmo resultado."""
    model = LiveNet(NetConfig(channels=32, blocks=2, repeats=1)).eval()
    audio = tone(1.3)
    path = tmp_path / "m.onnx"
    torch.onnx.export(
        model, (audio,), str(path), input_names=["audio"], output_names=["log_probs"],
        dynamic_axes={"audio": {1: "samples"}, "log_probs": {1: "frames"}}, opset_version=17, dynamo=False,
    )  # fmt: skip
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for seconds in (0.7, 1.3, 2.9):
        clip = tone(seconds)
        expected = model(clip).detach().numpy()
        got = session.run(None, {"audio": clip.numpy().astype(np.float32)})[0]
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, atol=1e-3)
