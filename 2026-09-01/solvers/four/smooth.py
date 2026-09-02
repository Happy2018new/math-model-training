"""Smooth the fused 10 Hz trajectory and estimate its derivatives.

The local polynomial fit used here is the same principle as a
Savitzky-Golay filter, but only depends on NumPy.  Timestamps are reconstructed
from the sampling frequency stored in SensorData.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .define import PosData, SensorData, TrajectoryPoint, sensor_data_from_raw
from .prepare import data


def smooth_data(
    sensor_data: SensorData,
    window_length: int = 11,
    polyorder: int = 3,
) -> list[TrajectoryPoint]:
    """Return smoothed positions, velocities, and accelerations.

    ``sensor_data`` supplies the sample frequency, start time, and position
    payload.  ``window_length`` is the number of samples used by each local
    fit and must be a positive odd number.  At the two ends of the trajectory,
    the available samples are used without padding artificial coordinates.
    """
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd number")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")

    if sensor_data.sensor_hz <= 0:
        raise ValueError("sensor_hz must be positive")
    if not sensor_data.payload:
        return []

    sample_count = len(sensor_data.payload)
    if window_length > sample_count:
        raise ValueError("window_length cannot exceed the number of samples")

    # SensorData stores the common sampling interval once, rather than
    # repeating a timestamp in every payload item.
    dt = 1.0 / sensor_data.sensor_hz
    times = sensor_data.start_time + np.arange(sample_count, dtype=float) * dt
    values = np.column_stack(
        (
            times,
            [point.posx for point in sensor_data.payload],
            [point.posy for point in sensor_data.payload],
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("sensor_data contains NaN or infinite values")

    radius = window_length // 2
    result: list[TrajectoryPoint] = []

    for index, (time, _, _) in enumerate(values):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        window = values[left:right]

        # Centering time at the sample makes the fitted coefficients directly
        # equal to position, first derivative, and half the second derivative.
        offsets = window[:, 0] - time
        degree = min(polyorder, len(window) - 1)
        design = np.vander(offsets, N=degree + 1, increasing=True)
        x_coefficients = np.linalg.lstsq(
            design, window[:, 1], rcond=None
        )[0]
        y_coefficients = np.linalg.lstsq(
            design, window[:, 2], rcond=None
        )[0]

        x_velocity = x_coefficients[1] if degree >= 1 else 0.0
        y_velocity = y_coefficients[1] if degree >= 1 else 0.0
        x_acceleration = 2.0 * x_coefficients[2] if degree >= 2 else 0.0
        y_acceleration = 2.0 * y_coefficients[2] if degree >= 2 else 0.0

        result.append(
            TrajectoryPoint(
                time=float(time),
                pos=PosData(float(x_coefficients[0]), float(y_coefficients[0])),
                velocity=PosData(float(x_velocity), float(y_velocity)),
                acceleration=PosData(
                    float(x_acceleration), float(y_acceleration)
                ),
            )
        )

    return result


def to_raw_data(trajectory: Sequence[TrajectoryPoint]) -> list[list[float]]:
    """Convert smoothed trajectory points back to ``[time, x, y]`` rows."""
    return [[point.time, point.pos.posx, point.pos.posy] for point in trajectory]


# Ready-to-use result for the data loaded by prepare.py.
sensor_data = sensor_data_from_raw(data, hz_data=10)
smoothed_data = smooth_data(sensor_data)
# Same result in the original ``[time, x, y]`` representation.
smoothed_raw_data = to_raw_data(smoothed_data)


if __name__ == "__main__":
    print(f"raw points: {len(data)}")
    print(f"smoothed points: {len(smoothed_data)}")
    print(f"first speed: {smoothed_data[0].speed:.6f}")
    print(
        "first acceleration: "
        f"{smoothed_data[0].acceleration_magnitude:.6f}"
    )
