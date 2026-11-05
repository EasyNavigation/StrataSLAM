"""Allan deviation: noise of known size is recovered; the warnings say when the recording is too short."""

import numpy as np
import pytest

from strataslam_tools import allan

FS = 100.0


def synthetic(hours: float, n_density: float, k_walk: float, seed: int = 3) -> np.ndarray:
    """A rate signal with white noise of density `n_density` and a bias that walks with density `k_walk`."""
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * FS)
    white = rng.normal(0.0, n_density * np.sqrt(FS), n)
    walk = np.cumsum(rng.normal(0.0, k_walk * np.sqrt(1.0 / FS), n))
    return white + walk


def test_white_noise_density_is_read_at_one_second() -> None:
    x = synthetic(1.0, 2.0e-3, 0.0)
    taus, sig = allan.allan_deviation(x, FS)
    n = allan.axis_noise(taus, sig).noise_density
    assert n == pytest.approx(2.0e-3, rel=0.1)


def test_random_walk_is_the_asymptote_of_the_long_taus_taken_at_three_seconds() -> None:
    x = synthetic(3.0, 1.0e-4, 2.0e-5)  # at 3 s the white noise (5.8e-5) still dominates the walk (2e-5)
    taus, sig = allan.allan_deviation(x, FS)
    a = allan.axis_noise(taus, sig)
    assert a.random_walk == pytest.approx(2.0e-5, rel=0.3)
    assert a.walk_slope == pytest.approx(0.5, abs=0.25)  # few clusters at the longest taus: a noisy slope


def test_a_signal_without_random_walk_gives_an_upper_bound_not_a_number() -> None:
    x = synthetic(1.0, 1.5e-3, 0.0)
    a = allan.axis_noise(*allan.allan_deviation(x, FS))
    assert a.random_walk is not None and not a.walk_reached
    assert a.random_walk < 1.5e-3 / np.sqrt(3.0) * 1.5  # at most the white noise seen at the longest taus, back at 3 s


def test_four_numbers_margin_and_the_noise_yaml() -> None:
    acc = np.stack([synthetic(0.5, 1.5e-3, 0.0, seed=s) for s in (1, 2, 3)], axis=1)
    gyr = np.stack([synthetic(0.5, 1.2e-4, 0.0, seed=s) for s in (4, 5, 6)], axis=1)
    res, curves = allan.imu_noise(acc, gyr, FS, margin=2.0)
    assert res.accel_noise_density == pytest.approx(2 * 1.5e-3, rel=0.15)
    assert res.gyro_noise_density == pytest.approx(2 * 1.2e-4, rel=0.15)
    assert len(curves) == 6
    text = res.noise_yaml()
    assert "gyro_noise_density:" in text and text.startswith("sensors:\n  imu:\n    noise:")
    assert any("under 2 h" in w for w in res.warnings)  # half an hour is not enough for the random walk


def test_a_recording_that_does_not_reach_the_reading_says_so() -> None:
    x = np.random.default_rng(0).normal(0, 1e-3, 150)  # 1.5 s at 100 Hz: no 3 s
    res, _ = allan.imu_noise(np.stack([x] * 3, 1), np.stack([x] * 3, 1), FS)
    assert any("does not reach" in w for w in res.warnings)
    assert np.isnan(res.accel_random_walk)


def test_too_few_samples_is_an_error() -> None:
    with pytest.raises(ValueError):
        allan.allan_deviation(np.zeros(5), FS)


def test_cli_parser_has_the_subcommand() -> None:
    from strataslam_tools import cli

    a = cli.build_parser().parse_args(["imu-noise", "b", "--topic", "/imu", "--margin", "2"])
    assert a.command == "imu-noise" and a.margin == 2.0
