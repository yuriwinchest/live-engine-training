from __future__ import annotations

import numpy as np
import pytest

from live_lab import augment
from live_lab.audio_io import SAMPLE_RATE
from live_lab.manifest import SnrLevel


def tone(seconds: float = 1.5, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def noise(seconds: float = 4.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)


@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 7.5, 15.0, 25.0])
def test_snr_within_one_db(snr_db: float) -> None:
    """SC-008."""
    speech = augment.pad(tone(), 0.3, 0.3)
    rng = np.random.default_rng(1)
    scaled = augment.scale_noise(speech, augment.fit_length(noise(), speech.size, rng), snr_db)
    assert abs(augment.measured_snr(speech, scaled) - snr_db) <= 1.0


def test_padding_does_not_dilute_speech_power() -> None:
    assert augment.active_power(augment.pad(tone(), 1.0, 1.0)) == pytest.approx(augment.active_power(tone()))


def test_fit_length_repeats_short_noise() -> None:
    short = noise(0.5)
    assert augment.fit_length(short, SAMPLE_RATE * 2, np.random.default_rng(0)).size == SAMPLE_RATE * 2


def test_anti_clipping_limits_peak() -> None:
    loud = tone(amplitude=3.0)
    assert np.max(np.abs(augment.prevent_clipping(loud))) <= augment.PEAK_LIMIT + 1e-6


def test_stretch_changes_duration_and_pitch_keeps_it() -> None:
    signal = tone(2.0)
    assert augment.time_stretch(signal, 1.25).size == pytest.approx(signal.size / 1.25, rel=0.02)
    assert augment.pitch_shift(signal, 2.0).size == signal.size


@pytest.mark.parametrize("level", list(SnrLevel))
def test_apply_never_clips(level: SnrLevel) -> None:
    rng = np.random.default_rng(3)
    params = augment.sample_params(rng, level)
    out = augment.apply(tone(amplitude=0.9), noise(), params, rng)
    assert np.max(np.abs(out)) <= augment.PEAK_LIMIT + 1e-6
    assert np.isfinite(out).all()


def test_sampled_levels_respect_ranges() -> None:
    rng = np.random.default_rng(9)
    for _ in range(500):
        params = augment.sample_params(rng)
        if params.snr_db is None:
            assert params.snr_level is SnrLevel.CLEAN
        else:
            low, high = augment.SNR_RANGES[params.snr_level]
            assert low <= params.snr_db <= high
        assert 0.85 <= params.stretch <= 1.15 and -2 <= params.pitch_semitones <= 2


def test_noisy_level_requires_noise() -> None:
    params = augment.sample_params(np.random.default_rng(0), SnrLevel.MEDIUM)
    with pytest.raises(ValueError):
        augment.apply(tone(), None, params, np.random.default_rng(0))
