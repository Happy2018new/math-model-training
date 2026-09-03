"""绘制问题四滑动窗口敏感性分析结果。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ..sensitivity_common import chinese_font_properties, configure_plot_fonts


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "four"


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("敏感性分析结果为空")
    return rows


def plot_sensitivity(
    csv_path: str | Path | None = None,
    save_name: str = "problem_four_sensitivity.svg",
) -> Path:
    """绘制窗口大小对问题四调度结果的影响。"""
    configure_plot_fonts()
    chinese_font = chinese_font_properties()
    source = Path(csv_path) if csv_path is not None else OUTPUT_DIR / "sensitivity.csv"
    rows = _read_rows(source)
    x = [int(row["window_points"]) for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")

    objective_axis = axes[0]
    objective = [float(row["objective"]) for row in rows]
    objective_axis.plot(x, objective, marker="o", color="#1f5d8f", linewidth=1.5)
    for window, value in zip(x, objective):
        objective_axis.annotate(
            f"${value:.0f}$",
            (window, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    objective_axis.set_title("目标函数值", fontproperties=chinese_font)
    objective_axis.set_xlabel("平滑窗口（点）", fontproperties=chinese_font)
    objective_axis.set_ylabel("$K$", fontproperties=chinese_font)
    objective_axis.grid(True, alpha=0.3)

    count_axis = axes[1]
    shoot = [int(row["shoot_count"]) for row in rows]
    photo = [int(row["photo_count"]) for row in rows]
    count_axis.plot(x, shoot, marker="o", color="#b83831", label="射击次数")
    count_axis.plot(x, photo, marker="s", color="#2168a6", label="拍照次数")
    count_axis.set_title("选中任务数量", fontproperties=chinese_font)
    count_axis.set_xlabel("平滑窗口（点）", fontproperties=chinese_font)
    count_axis.set_ylabel("次数", fontproperties=chinese_font)
    count_axis.set_yticks(range(0, max(photo + shoot, default=0) + 3, 5))
    count_axis.grid(True, alpha=0.3)
    count_axis.legend(prop=chinese_font, loc="best")

    total_axis = axes[2]
    total = [int(row["task_count"]) for row in rows]
    total_axis.plot(x, total, marker="D", color="#4c8b57", linewidth=1.5)
    for window, value in zip(x, total):
        total_axis.annotate(
            f"${value}$",
            (window, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    total_axis.set_title("选中任务总数", fontproperties=chinese_font)
    total_axis.set_xlabel("平滑窗口（点）", fontproperties=chinese_font)
    total_axis.set_ylabel("任务数", fontproperties=chinese_font)
    total_axis.grid(True, alpha=0.3)

    figure.suptitle(
        "问题四：滑动窗口敏感性分析",
        fontproperties=chinese_font,
        fontsize=14,
    )
    destination = (OUTPUT_DIR / save_name).with_suffix(".svg")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(figure)
    return destination


if __name__ == "__main__":
    print(f"图片文件：{plot_sensitivity()}")
