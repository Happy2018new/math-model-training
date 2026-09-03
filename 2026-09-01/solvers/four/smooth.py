"""平滑 10 Hz 融合轨迹，并估计其速度和加速度。

这里的局部多项式拟合与 Savitzky-Golay 滤波原理相同，但仅依赖 NumPy。
时间戳由 ``SensorData`` 中保存的采样频率和起始时刻重建。
"""

from __future__ import annotations

import numpy as np
from collections.abc import Sequence
from .define import PosData, SensorData, TrajectoryPoint, sensor_data_from_raw
from .prepare import data


def smooth_data(
    sensor_data: SensorData,
    window_length: int = 11,
    polyorder: int = 3,
) -> list[TrajectoryPoint]:
    """返回平滑后的位置、速度和加速度。

    ``sensor_data`` 提供采样频率、起始时刻和位置序列。``window_length`` 是
    每次局部拟合使用的采样点数，必须为正奇数。轨迹两端仅使用实际存在的点，
    不额外填充虚构坐标。
    """
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("窗口长度必须为正奇数")
    if polyorder < 0:
        raise ValueError("多项式阶数不能为负数")
    if polyorder >= window_length:
        raise ValueError("多项式阶数必须小于窗口长度")

    if sensor_data.sensor_hz <= 0:
        raise ValueError("采样频率必须为正数")
    if not sensor_data.payload:
        return []

    sample_count = len(sensor_data.payload)
    if window_length > sample_count:
        raise ValueError("窗口长度不能超过采样点数量")

    # SensorData 只保存一次公共采样间隔信息，无需在每个位置点重复存储时间戳。
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
        raise ValueError("传感器数据中包含 NaN 或无穷值")

    radius = window_length // 2
    result: list[TrajectoryPoint] = []

    for index, (time, _, _) in enumerate(values):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        window = values[left:right]

        # 以当前采样时刻为中心后，拟合系数可直接给出位置、一阶导数和二阶导数的一半。
        offsets = window[:, 0] - time
        degree = min(polyorder, len(window) - 1)
        design = np.vander(offsets, N=degree + 1, increasing=True)
        x_coefficients = np.linalg.lstsq(design, window[:, 1], rcond=None)[0]
        y_coefficients = np.linalg.lstsq(design, window[:, 2], rcond=None)[0]

        x_velocity = x_coefficients[1] if degree >= 1 else 0.0
        y_velocity = y_coefficients[1] if degree >= 1 else 0.0
        x_acceleration = 2.0 * x_coefficients[2] if degree >= 2 else 0.0
        y_acceleration = 2.0 * y_coefficients[2] if degree >= 2 else 0.0

        result.append(
            TrajectoryPoint(
                time=float(time),
                pos=PosData(float(x_coefficients[0]), float(y_coefficients[0])),
                velocity=PosData(float(x_velocity), float(y_velocity)),
                acceleration=PosData(float(x_acceleration), float(y_acceleration)),
            )
        )

    return result


def to_raw_data(trajectory: Sequence[TrajectoryPoint]) -> list[list[float]]:
    """将平滑轨迹点转回 ``[时间, X坐标, Y坐标]`` 行列表。"""
    return [[point.time, point.pos.posx, point.pos.posy] for point in trajectory]


# 对 prepare.py 已读取数据的直接平滑结果。
sensor_data = sensor_data_from_raw(data, hz_data=10)
smoothed_data = smooth_data(sensor_data)
# 同一结果的原始 ``[时间, X坐标, Y坐标]`` 表示。
smoothed_raw_data = to_raw_data(smoothed_data)


if __name__ == "__main__":
    print(f"原始点数: {len(data)}")
    print(f"平滑后点数: {len(smoothed_data)}")
    print(f"第一个点的速度: {smoothed_data[0].speed:.6f}")
    print(f"第一个点的加速度: {smoothed_data[0].acceleration_magnitude:.6f}")
