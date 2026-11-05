"""IMU noise from a static recording: the Allan deviation and the four numbers of a profile's `imu.noise`.

References: D. W. Allan, "Statistics of atomic frequency standards", Proc. IEEE 54(2), 1966; IEEE Std 952-1997 (Annex
C, the slopes of each noise term on the Allan deviation plot).

Record the IMU completely still for hours (2-3 h at least, 6 h or more for the random walk), with the sensor warm and
without vibration, then:

    strataslam imu-noise imu_still_bag --topic /oak/imu --margin 2 --plot allan.png

The Allan deviation of a rate signal over clusters of duration tau shows each noise as a slope on a log-log plot:
white noise (slope -1/2, read at tau = 1 s) is the noise density N, the random walk of the bias (slope +1/2, read at
tau = 3 s) is K and the minimum of the curve (slope 0) is the bias instability. The computation needs only numpy;
reading a rosbag2 needs ROS and the plot needs matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def allan_deviation(rate: Array, fs: float, points_per_decade: int = 12) -> tuple[Array, Array]:
    """Overlapping Allan deviation of a rate signal (rad/s or m/s^2) sampled at `fs` Hz.

    Returns (tau, sigma) for cluster sizes m = 1 ... N/9 spaced logarithmically."""
    x = np.asarray(rate, dtype=np.float64)
    n = len(x)
    if n < 20:
        raise ValueError("too few samples for an Allan deviation")
    theta = np.concatenate((np.zeros(1), np.cumsum(x))) / fs  # integral of the rate
    max_m = max(1, (n - 1) // 9)
    ms = np.unique(np.round(np.logspace(0, np.log10(max_m), max(2, int(np.log10(max_m) * points_per_decade) + 1))).astype(int))
    taus, sigmas = [], []
    for m in ms:
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]
        tau = m / fs
        taus.append(tau)
        sigmas.append(np.sqrt(np.mean(d * d) / (2.0 * tau * tau)))
    return np.array(taus), np.array(sigmas)


def _at(taus: Array, sigmas: Array, tau: float) -> float | None:
    """Value of the curve at `tau` (log-log interpolation), None if the recording does not reach it."""
    if tau < taus[0] or tau > taus[-1]:
        return None
    return float(np.exp(np.interp(np.log(tau), np.log(taus), np.log(sigmas))))


@dataclass
class AxisNoise:
    noise_density: float | None  # N: sigma at tau = 1 s (unit/sqrt(Hz))
    random_walk: float | None  # K: the +1/2 asymptote of the long taus, taken at tau = 3 s (unit/s/sqrt(Hz))
    bias_instability: float  # minimum of the curve
    tau_at_minimum: float
    walk_slope: float | None = None  # measured slope over the taus used for K (0.5 for a pure random walk)
    walk_reached: bool = False  # False: no +1/2 slope was seen and `random_walk` is an upper bound


def axis_noise(taus: Array, sigmas: Array) -> AxisNoise:
    i = int(np.argmin(sigmas))
    n1 = _at(taus, sigmas, 1.0)
    # sigma(tau) = K sqrt(tau/3) on the +1/2 slope. At tau = 3 s the white noise still dominates, so K is the asymptote
    # of the last decade of the curve brought back to 3 s. A recording shorter than 3 s gives nothing.
    if taus[-1] < 3.0:
        return AxisNoise(n1, None, float(sigmas[i]), float(taus[i]))
    tail = (taus >= taus[-1] / 10.0) & (taus > 2.0 * taus[i])
    if np.count_nonzero(tail) >= 3:
        slope = float(np.polyfit(np.log(taus[tail]), np.log(sigmas[tail]), 1)[0])
        if slope >= 0.25:
            k = float(np.exp(np.mean(np.log(sigmas[tail]) - 0.5 * np.log(taus[tail] / 3.0))))
            return AxisNoise(n1, k, float(sigmas[i]), float(taus[i]), slope, True)
    # no rise at the long taus (the bias does not walk within the recording): the last point bounds K from above
    last = max(1, len(taus) // 10)
    k_bound = float(np.max(sigmas[-last:] / np.sqrt(taus[-last:] / 3.0)))
    return AxisNoise(n1, k_bound, float(sigmas[i]), float(taus[i]), None, False)


@dataclass
class ImuNoise:
    accel_noise_density: float
    gyro_noise_density: float
    accel_random_walk: float
    gyro_random_walk: float
    warnings: list[str]

    def noise_yaml(self) -> str:
        """The `sensors.imu.noise` block of a profile."""
        return (
            "sensors:\n  imu:\n    noise:\n"
            f"      accel_noise_density: {self.accel_noise_density:.3e}\n"
            f"      gyro_noise_density: {self.gyro_noise_density:.3e}\n"
            f"      accel_random_walk: {self.accel_random_walk:.3e}\n"
            f"      gyro_random_walk: {self.gyro_random_walk:.3e}\n"
        )


def imu_noise(accel: Array, gyro: Array, fs: float, margin: float = 1.0) -> tuple[ImuNoise, dict[str, tuple[Array, Array]]]:
    """The four numbers of `imu.noise` from (N, 3) accelerometer and gyroscope samples taken at rest.

    Of the three axes the worst is kept; `margin` multiplies the result (a moving IMU is worse than a parked one)."""
    curves: dict[str, tuple[Array, Array]] = {}
    out: dict[str, list[AxisNoise]] = {"accel": [], "gyro": []}
    for kind, data in (("accel", accel), ("gyro", gyro)):
        for axis, name in enumerate("xyz"):
            taus, sig = allan_deviation(data[:, axis], fs)
            curves[f"{kind}_{name}"] = (taus, sig)
            out[kind].append(axis_noise(taus, sig))
    warnings: list[str] = []
    span = len(accel) / fs

    def worst(kind: str, field_: str, label: str) -> float:
        vals = [getattr(a, field_) for a in out[kind]]
        vals = [v for v in vals if v is not None]
        if not vals:
            warnings.append(f"{label}: the recording ({span / 3600:.1f} h at {fs:.0f} Hz) does not reach the tau of the reading")
            return float("nan")
        return float(max(vals))

    res = ImuNoise(
        accel_noise_density=worst("accel", "noise_density", "accel_noise_density") * margin,
        gyro_noise_density=worst("gyro", "noise_density", "gyro_noise_density") * margin,
        accel_random_walk=worst("accel", "random_walk", "accel_random_walk") * margin,
        gyro_random_walk=worst("gyro", "random_walk", "gyro_random_walk") * margin,
        warnings=warnings,
    )
    for kind in ("accel", "gyro"):
        for a, name in zip(out[kind], "xyz"):
            if a.random_walk is not None and not a.walk_reached:
                res.warnings.append(f"{kind}_{name}: no +1/2 slope at the long taus (the bias does not walk within the recording): its random walk is an upper bound")
    if span < 2 * 3600:
        res.warnings.append(f"the recording lasts {span / 3600:.1f} h: under 2 h the random walk and the bias instability are not reliable")
    # white noise must dominate at tau = 1 s: a curve that is not decreasing there means vibration or a quantized sensor
    for key, (taus, sig) in curves.items():
        if taus[0] < 0.5 and taus[-1] > 4 and (_at(taus, sig, 0.5) or 0) < (_at(taus, sig, 1.0) or 0) * 1.05:
            res.warnings.append(f"{key}: the curve does not fall as 1/sqrt(tau) before 1 s; the reading of N may be pessimistic (vibration?)")
    return res, curves


def read_imu_bag(bag: str, topic: str) -> tuple[Array, Array, float]:
    """(accelerometer (N,3), gyroscope (N,3), sample rate in Hz) of a sensor_msgs/Imu topic of a rosbag2."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions("", ""))
    stamps, acc, gyr = [], [], []
    while reader.has_next():
        name, raw, _ = reader.read_next()
        if name != topic:
            continue
        m = deserialize_message(raw, Imu)
        stamps.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        acc.append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
        gyr.append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
    if len(stamps) < 100:
        raise SystemExit(f"strataslam: {len(stamps)} message(s) on {topic}: nothing to measure")
    t = np.array(stamps)
    fs = (len(t) - 1) / (t[-1] - t[0])
    return np.array(acc), np.array(gyr), float(fs)


def plot_curves(curves: dict[str, tuple[Array, Array]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, (taus, sig) in curves.items():
        ax[0 if k.startswith("accel") else 1].loglog(taus, sig, label=k)
    for a, title, unit in ((ax[0], "accelerometer", "m/s^2"), (ax[1], "gyroscope", "rad/s")):
        a.axvline(1.0, c="grey", ls=":")
        a.axvline(3.0, c="grey", ls=":")
        a.set_title(f"Allan deviation, {title}")
        a.set_xlabel("tau (s)")
        a.set_ylabel(f"sigma ({unit})")
        a.grid(True, which="both", alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
