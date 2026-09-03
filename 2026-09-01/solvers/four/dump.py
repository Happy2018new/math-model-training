"""导出问题四标准 0--1 整数规划方案。"""

from __future__ import annotations

import csv
from pathlib import Path

from .define import TaskCandidate, TrajectoryPoint
from .resolve import (
    PHOTO_SCORE,
    SHOOT_SCORE,
    solve_schedule,
    trajectory_from_result_csv,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "four"
HEADER = (
    "候选编号",
    "任务类型",
    "目标编号",
    "准备开始时刻(s)",
    "执行时刻(s)",
    "拍照角度(度)",
    "执行X坐标(m)",
    "执行Y坐标(m)",
    "任务收益",
)


def _position_by_time(
    trajectory: list[TrajectoryPoint],
) -> dict[float, tuple[float, float]]:
    """建立执行时刻到机器人位置的索引。"""
    return {
        round(point.time, 6): (point.pos.posx, point.pos.posy)
        for point in trajectory
    }


def _write_solution(
    chosen: list[TaskCandidate],
    trajectory: list[TrajectoryPoint],
    output_path: str | Path,
) -> Path:
    """将已选任务写入 CSV。"""
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = OUTPUT_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    positions = _position_by_time(trajectory)

    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for candidate in sorted(chosen, key=lambda item: item.execute_time):
            try:
                execute_x, execute_y = positions[round(candidate.execute_time, 6)]
            except KeyError as exc:
                raise ValueError(
                    f"轨迹中找不到候选 {candidate.candidate_id} 的执行时刻"
                ) from exc
            task_type = "射击" if candidate.task_type == "shoot" else "拍照"
            angle = "" if candidate.angle_deg is None else candidate.angle_deg
            score = SHOOT_SCORE if candidate.task_type == "shoot" else PHOTO_SCORE
            writer.writerow(
                (
                    candidate.candidate_id,
                    task_type,
                    candidate.task_id,
                    round(candidate.prepare_start, 10),
                    round(candidate.execute_time, 10),
                    angle,
                    execute_x,
                    execute_y,
                    score,
                )
            )
    return destination


def dump_standard_solution(
    output_path: str | Path = "result.csv",
) -> Path:
    """求解默认 41 点方案并导出已选任务。"""
    chosen, objective, status = solve_schedule()
    if status != "Optimal":
        raise RuntimeError(f"求解器未返回最优解：{status}")
    trajectory = trajectory_from_result_csv()
    destination = _write_solution(chosen, trajectory, output_path)
    shoot_count = sum(candidate.task_type == "shoot" for candidate in chosen)
    photo_count = len(chosen) - shoot_count
    print(f"求解状态：{status}")
    print(f"射击次数：{shoot_count}")
    print(f"拍照次数：{photo_count}")
    print(f"任务总数：{len(chosen)}")
    print(f"目标值：{objective:.0f}")
    print(f"结果文件：{destination}")
    return destination


if __name__ == "__main__":
    dump_standard_solution()
