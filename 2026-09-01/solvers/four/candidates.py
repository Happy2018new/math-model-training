"""计算问题四中射击和拍照任务的可行执行时刻。"""

from __future__ import annotations

import csv
from bisect import bisect_left
from collections.abc import Sequence
from math import atan2, degrees, hypot
from pathlib import Path

from .define import TaskCandidate, TrajectoryPoint


SHOOT_FILE = Path("inputs/table_4-1.csv")
PHOTO_FILE = Path("inputs/table_4-2.csv")


def read_targets(csv_file: str | Path) -> list[tuple[str, float, float]]:
    """从目标 CSV 文件读取 ``(目标编号, X坐标, Y坐标)``。"""
    path = Path(csv_file)
    if not path.is_file():
        raise FileNotFoundError(f"目标 CSV 文件不存在：{path}")

    required = {"编号", "X坐标(m)", "Y坐标(m)"}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} 必须包含列：{', '.join(required)}")
        targets: list[tuple[str, float, float]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                target_id = row["编号"]
                if not target_id:
                    continue
                targets.append(
                    (
                        target_id,
                        float(row["X坐标(m)"]),
                        float(row["Y坐标(m)"]),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} 第 {row_number} 行的目标数据无效"
                ) from exc
    if not targets:
        raise ValueError(f"目标 CSV 文件 {path} 中没有目标")
    return targets


def read_shoot_targets(
    csv_file: str | Path = SHOOT_FILE,
) -> list[tuple[str, float, float]]:
    return read_targets(csv_file)


def read_photo_targets(
    csv_file: str | Path = PHOTO_FILE,
) -> list[tuple[str, float, float]]:
    return read_targets(csv_file)


def _in_range(value: float, lower: float | None, upper: float | None) -> bool:
    return (lower is None or value >= lower) and (upper is None or value <= upper)


def _valid(
    point: TrajectoryPoint,
    target_x: float,
    target_y: float,
    min_distance: float | None,
    max_distance: float | None,
    min_speed: float | None,
    max_speed: float | None,
    min_acceleration: float | None,
    max_acceleration: float | None,
) -> bool:
    distance = hypot(point.pos.posx - target_x, point.pos.posy - target_y)
    return (
        _in_range(distance, min_distance, max_distance)
        and _in_range(point.speed, min_speed, max_speed)
        and _in_range(
            point.acceleration_magnitude,
            min_acceleration,
            max_acceleration,
        )
    )


def feasible_times(
    trajectory: Sequence[TrajectoryPoint],
    task_id: str,
    task_type: str,
    target_x: float,
    target_y: float,
    preparation_time: float,
    *,
    min_distance: float | None = None,
    max_distance: float | None = None,
    min_speed: float | None = None,
    max_speed: float | None = None,
    min_acceleration: float | None = None,
    max_acceleration: float | None = None,
    time_tolerance: float = 1e-6,
) -> list[TaskCandidate]:
    """计算单个目标的全部可行执行候选。

    对每个采样开始时刻 ``T``，闭区间 ``[T, T + 准备时长]`` 中的每个
    轨迹点都必须满足给定约束。返回的候选时刻是该区间终点，即实际执行时刻。
    """
    if preparation_time < 0 or time_tolerance < 0:
        raise ValueError("准备时长和时间容差不能为负数")
    if not trajectory:
        return []
    if any(later.time <= earlier.time for earlier, later in zip(trajectory, trajectory[1:])):
        raise ValueError("轨迹时间必须严格递增")

    times = [point.time for point in trajectory]
    if task_type not in {"shoot", "photo"}:
        raise ValueError("任务类型必须为 'shoot' 或 'photo'")

    result: list[TaskCandidate] = []
    for start_index, start_time in enumerate(times):
        end_time = start_time + preparation_time
        end_index = bisect_left(times, end_time - time_tolerance, lo=start_index)
        if end_index == len(times) or abs(times[end_index] - end_time) > time_tolerance:
            break
        if all(
            _valid(
                point,
                target_x,
                target_y,
                min_distance,
                max_distance,
                min_speed,
                max_speed,
                min_acceleration,
                max_acceleration,
            )
            for point in trajectory[start_index : end_index + 1]
        ):
            point = trajectory[end_index]
            angle = None
            if task_type == "photo":
                angle = (degrees(atan2(
                    point.pos.posy - target_y,
                    point.pos.posx - target_x,
                )) + 360.0) % 360.0
            result.append(
                TaskCandidate(
                    candidate_id=f"{task_id}_{len(result) + 1:03d}",
                    task_id=task_id,
                    task_type=task_type,
                    execute_time=float(times[end_index]),
                    angle_deg=angle,
                )
            )
    return result


def feasible_shoot_times(
    trajectory: Sequence[TrajectoryPoint],
    task_id: str,
    target_x: float,
    target_y: float,
    *,
    preparation_time: float = 1.5,
    min_distance: float | None = None,
    max_distance: float | None = None,
    min_speed: float | None = None,
    max_speed: float | None = None,
    min_acceleration: float | None = None,
    max_acceleration: float | None = None,
) -> list[TaskCandidate]:
    """计算单个射击目标的全部可行候选。"""
    return feasible_times(
        trajectory,
        task_id,
        "shoot",
        target_x,
        target_y,
        preparation_time,
        min_distance=min_distance,
        max_distance=max_distance,
        min_speed=min_speed,
        max_speed=max_speed,
        min_acceleration=min_acceleration,
        max_acceleration=max_acceleration,
    )


def feasible_photo_times(
    trajectory: Sequence[TrajectoryPoint],
    task_id: str,
    target_x: float,
    target_y: float,
    *,
    preparation_time: float = 0.5,
    min_distance: float | None = None,
    max_distance: float | None = None,
    min_speed: float | None = None,
    max_speed: float | None = None,
    min_acceleration: float | None = None,
    max_acceleration: float | None = None,
) -> list[TaskCandidate]:
    """计算单个拍照目标的全部可行候选。"""
    return feasible_times(
        trajectory,
        task_id,
        "photo",
        target_x,
        target_y,
        preparation_time,
        min_distance=min_distance,
        max_distance=max_distance,
        min_speed=min_speed,
        max_speed=max_speed,
        min_acceleration=min_acceleration,
        max_acceleration=max_acceleration,
    )


if __name__ == "__main__":
    print(f"射击目标：{len(read_shoot_targets())} 个")
    print(f"拍照目标：{len(read_photo_targets())} 个")
