"""完成问题四的候选生成、冲突构造与 0-1 整数规划求解。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from .candidates import (
    feasible_photo_times,
    feasible_shoot_times,
    read_photo_targets,
    read_shoot_targets,
)
from .conflicts import calculate_conflict_sets
from .define import PosData, TaskCandidate, TrajectoryPoint
from .prepare import data
from .smooth import sensor_data, smooth_data

# 位置直接取问题三的最终融合轨迹；仅用 41 点局部三次拟合估计速度和加速度。
TASK_SMOOTH_WINDOW = 41
TASK_POLYORDER = 3

SHOOT_SCORE = 75
PHOTO_SCORE = 100

SHOOT_CONSTRAINTS = {
    "min_distance": 5.0,
    "max_distance": 30.0,
    "max_speed": 2.0,
    "max_acceleration": 1.5,
}

PHOTO_CONSTRAINTS = {
    "min_distance": 10.0,
    "max_distance": 40.0,
    "max_speed": 1.5,
    "max_acceleration": 1.5,
}


def trajectory_from_result_csv(
    window_length: int = TASK_SMOOTH_WINDOW,
) -> list[TrajectoryPoint]:
    """从 ``result.csv`` 构造包含位置、速度和加速度的轨迹点。

    距离判断始终使用原始融合位置；局部三次拟合只用于估计速度和加速度，
    避免数值微分放大位置中的残余噪声。
    """
    if len(data) < 3:
        raise ValueError("result.csv 至少应包含三个轨迹点")

    times = [row[0] for row in data]
    positions = [PosData(row[1], row[2]) for row in data]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("result.csv 的时间必须严格递增")

    derivative_trajectory = smooth_data(
        sensor_data,
        window_length=window_length,
        polyorder=TASK_POLYORDER,
    )
    if len(derivative_trajectory) != len(data):
        raise ValueError("result.csv 的导数估计结果长度异常")

    return [
        TrajectoryPoint(
            time=float(time),
            pos=position,
            velocity=derivative_point.velocity,
            acceleration=derivative_point.acceleration,
        )
        for (time, _x, _y), position, derivative_point in zip(
            data, positions, derivative_trajectory
        )
    ]


def calculate_candidates() -> (
    tuple[dict[str, list[TaskCandidate]], dict[str, list[TaskCandidate]]]
):
    """计算每个射击目标和拍照目标的全部可行候选任务。"""
    trajectory = trajectory_from_result_csv()

    shoot_times = {
        target_id: feasible_shoot_times(
            trajectory, target_id, x, y, **SHOOT_CONSTRAINTS
        )
        for target_id, x, y in read_shoot_targets()
    }
    photo_times = {
        target_id: feasible_photo_times(
            trajectory, target_id, x, y, **PHOTO_CONSTRAINTS
        )
        for target_id, x, y in read_photo_targets()
    }
    return shoot_times, photo_times


def _group_shoot_candidates(
    candidates: Iterable[TaskCandidate],
) -> dict[str, list[TaskCandidate]]:
    """按射击目标编号对候选任务分组。"""
    grouped: dict[str, list[TaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.task_type == "shoot":
            grouped[candidate.task_id].append(candidate)
    return dict(grouped)


def solve_schedule(
    *,
    time_limit: float | None = None,
    message: bool = False,
) -> tuple[list[TaskCandidate], float, str]:
    """用成对冲突约束求解任务选择问题。

    每个候选任务对应一个 0-1 变量。目标函数为
    ``75 * 射击次数 + 100 * 拍照次数``。每一对时间冲突或角度冲突候选
    都直接加入 ``x_i + x_j <= 1``；每个射击目标至多执行一次。
    """
    shoot_candidates, photo_candidates = calculate_candidates()
    candidates, time_conflicts, angle_conflicts = calculate_conflict_sets(
        shoot_candidates,
        photo_candidates,
    )
    candidate_ids = list(candidates)
    candidate_index = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }

    conflict_pairs = sorted(time_conflicts | angle_conflicts)
    shoot_groups = [
        tuple(candidate.candidate_id for candidate in task_candidates)
        for task_candidates in _group_shoot_candidates(candidates.values()).values()
    ]
    constraint_sets = conflict_pairs + shoot_groups

    row_index: list[int] = []
    column_index: list[int] = []
    for row, candidate_set in enumerate(constraint_sets):
        row_index.extend([row] * len(candidate_set))
        column_index.extend(
            candidate_index[candidate_id] for candidate_id in candidate_set
        )
    constraint_matrix = coo_array(
        (np.ones(len(row_index)), (row_index, column_index)),
        shape=(len(constraint_sets), len(candidate_ids)),
    ).tocsr()

    scores = np.array(
        [
            (
                SHOOT_SCORE
                if candidates[candidate_id].task_type == "shoot"
                else PHOTO_SCORE
            )
            for candidate_id in candidate_ids
        ],
        dtype=float,
    )
    result = milp(
        c=-scores,
        integrality=np.ones(len(candidate_ids)),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(
            constraint_matrix,
            lb=np.full(len(constraint_sets), -np.inf),  # type: ignore
            ub=np.ones(len(constraint_sets)),  # type: ignore
        ),
        options={
            "disp": message,
            **({"time_limit": time_limit} if time_limit is not None else {}),
        },
    )
    if result.x is None:
        raise RuntimeError(f"HiGHS 未找到可行解：{result.message}")
    status = (
        "Optimal"
        if result.status == 0
        else f"HiGHS 状态 {result.status}: {result.message}"
    )

    chosen = [
        candidates[candidate_id]
        for candidate_id, value in zip(candidate_ids, result.x)
        if value > 0.5
    ]
    chosen.sort(key=lambda candidate: candidate.execute_time)

    chosen_ids = {candidate.candidate_id for candidate in chosen}
    time_violations = [
        pair
        for pair in time_conflicts
        if pair[0] in chosen_ids and pair[1] in chosen_ids
    ]
    angle_violations = [
        pair
        for pair in angle_conflicts
        if pair[0] in chosen_ids and pair[1] in chosen_ids
    ]
    if time_violations or angle_violations:
        raise RuntimeError(
            "所选方案存在 "
            f"{len(time_violations)} 对时间冲突和 "
            f"{len(angle_violations)} 对角度冲突"
        )

    objective = float(
        SHOOT_SCORE * sum(candidate.task_type == "shoot" for candidate in chosen)
        + PHOTO_SCORE * sum(candidate.task_type == "photo" for candidate in chosen)
    )
    return chosen, objective, status


def _print_solution(chosen: list[TaskCandidate], objective: float, status: str) -> None:
    """以中文表格输出最终任务安排。"""
    shoot_count = sum(candidate.task_type == "shoot" for candidate in chosen)
    photo_count = len(chosen) - shoot_count
    print(f"求解状态: {status}")
    print(f"最优目标值: {objective:.0f}")
    print(
        f"目标函数: {SHOOT_SCORE} * {shoot_count} + "
        f"{PHOTO_SCORE} * {photo_count} = {objective:.0f}"
    )
    print(f"射击次数: {shoot_count}")
    print(f"拍照次数: {photo_count}")
    print(f"已选任务总数: {len(chosen)}")
    print()
    print("候选编号     任务类型  目标编号  执行时刻(s)  拍照角度(度)")
    print("----------------------------------------------------------")
    for candidate in chosen:
        task_type = "射击" if candidate.task_type == "shoot" else "拍照"
        angle = "--" if candidate.angle_deg is None else f"{candidate.angle_deg:.2f}"
        print(
            f"{candidate.candidate_id:<10}  {task_type:<4}  {candidate.task_id:<6}  "
            f"{candidate.execute_time:>10.1f}  {angle:>11}"
        )


if __name__ == "__main__":
    _print_solution(*solve_schedule())
