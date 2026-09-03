"""问题四滑动窗口敏感性分析。"""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

from .resolve import solve_schedule


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs" / "four"
OUTPUT_PATH = OUTPUT_DIR / "sensitivity.csv"

# 41 点是当前正式方案；其余窗口用于观察平滑强度改变后的调度结果。
WINDOWS = (5, 11, 21, 31, 41, 51, 61, 81, 101)


def run_sensitivity(
    windows: tuple[int, ...] = WINDOWS,
    output_path: str | Path = OUTPUT_PATH,
) -> Path:
    """对每个窗口重新生成候选任务并求解，保存汇总结果。"""
    if not windows:
        raise ValueError("至少需要一个滑动窗口")
    if any(window < 5 or window % 2 == 0 for window in windows):
        raise ValueError("窗口必须是大于等于 5 的奇数")

    rows: list[dict[str, object]] = []
    for window in windows:
        started = perf_counter()
        chosen, objective, status = solve_schedule(window_length=window)
        elapsed = perf_counter() - started
        shoot_count = sum(candidate.task_type == "shoot" for candidate in chosen)
        photo_count = sum(candidate.task_type == "photo" for candidate in chosen)
        rows.append(
            {
                "experiment": "trajectory_smoothing_window",
                "window_points": window,
                "status": status,
                "objective": objective,
                "shoot_count": shoot_count,
                "photo_count": photo_count,
                "task_count": len(chosen),
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"窗口 {window:>3} 点：{status}，目标值 {objective:.0f}，"
            f"射击 {shoot_count} 次，拍照 {photo_count} 次，"
            f"总任务 {len(chosen)} 项"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


if __name__ == "__main__":
    print(f"结果文件：{run_sensitivity()}")
