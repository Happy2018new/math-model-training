"""绘制问题四 0-1 整数规划调度结果的主体结论图。"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from .candidates import read_photo_targets, read_shoot_targets
from .define import TaskCandidate, TrajectoryPoint
from .resolve import solve_schedule, trajectory_from_result_csv
from ..sensitivity_common import (
    chinese_font_properties,
    configure_plot_fonts,
    mathtext_number,
)


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "four"


def _position_by_time(
    trajectory: list[TrajectoryPoint],
) -> dict[float, tuple[float, float]]:
    """建立执行时刻到机器人位置的索引。"""
    return {
        round(point.time, 6): (point.pos.posx, point.pos.posy)
        for point in trajectory
    }


def _candidate_position(
    candidate: TaskCandidate,
    positions: dict[float, tuple[float, float]],
) -> tuple[float, float]:
    """返回一个候选任务实际执行时刻对应的机器人位置。"""
    try:
        return positions[round(candidate.execute_time, 6)]
    except KeyError as exc:
        raise ValueError(
            f"轨迹中找不到候选 {candidate.candidate_id} 的执行时刻"
        ) from exc


def _split_tasks(
    chosen: list[TaskCandidate],
) -> tuple[list[TaskCandidate], list[TaskCandidate]]:
    """将最优解拆分为射击任务和拍照任务。"""
    shoot = [candidate for candidate in chosen if candidate.task_type == "shoot"]
    photo = [candidate for candidate in chosen if candidate.task_type == "photo"]
    return shoot, photo


def _task_groups(
    chosen: list[TaskCandidate],
    group_count: int = 9,
) -> list[list[TaskCandidate]]:
    """按执行时刻将任务尽量均匀地分为若干连续组。"""
    if group_count <= 0:
        raise ValueError("分组数量必须为正数")
    ordered = sorted(chosen, key=lambda candidate: candidate.execute_time)
    if not ordered:
        return []
    group_count = min(group_count, len(ordered))
    quotient, remainder = divmod(len(ordered), group_count)
    groups: list[list[TaskCandidate]] = []
    start = 0
    for group_index in range(group_count):
        size = quotient + (group_index < remainder)
        groups.append(ordered[start : start + size])
        start += size
    return groups


def _axis_limits(
    trajectory: list[TrajectoryPoint],
    shoot_targets: list[tuple[str, float, float]],
    photo_targets: list[tuple[str, float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """计算九张图共用的坐标范围，保证不同时段可以直接比较。"""
    x_values = [point.pos.posx for point in trajectory]
    y_values = [point.pos.posy for point in trajectory]
    x_values.extend(x for _, x, _ in shoot_targets + photo_targets)
    y_values.extend(y for _, _, y in shoot_targets + photo_targets)
    padding = 5.0
    return (
        (min(x_values) - padding, max(x_values) + padding),
        (min(y_values) - padding, max(y_values) + padding),
    )


def plot_schedule(
    chosen: list[TaskCandidate],
    objective: float,
    trajectory: list[TrajectoryPoint],
    save_name: str = "problem_four_schedule.svg",
) -> Path:
    """在一张九宫格 SVG 中绘制机器人轨迹中各时段的任务位置。"""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    configure_plot_fonts()
    chinese_font = chinese_font_properties()
    shoot_tasks, photo_tasks = _split_tasks(chosen)
    positions = _position_by_time(trajectory)
    shoot_targets = read_shoot_targets()
    photo_targets = read_photo_targets()
    shoot_target_map = {target_id: (x, y) for target_id, x, y in shoot_targets}
    photo_target_map = {target_id: (x, y) for target_id, x, y in photo_targets}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_limits, y_limits = _axis_limits(trajectory, shoot_targets, photo_targets)
    groups = _task_groups(chosen)
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(16, 15),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axes_flat = list(axes.flat)

    for group_index, (spatial_axis, group) in enumerate(
        zip(axes_flat, groups), start=1
    ):
        spatial_axis.plot(
            [point.pos.posx for point in trajectory],
            [point.pos.posy for point in trajectory],
            color="#68737c",
            linewidth=0.75,
            alpha=0.72,
            label=r"问题三 $10\,\mathrm{Hz}$ 融合轨迹",
            zorder=1,
        )
        spatial_axis.scatter(
            [x for _, x, _ in shoot_targets],
            [y for _, _, y in shoot_targets],
            marker="*",
            s=65,
            facecolors="none",
            edgecolors="#b83831",
            linewidths=0.9,
            alpha=0.35,
            label="全部射击目标",
            zorder=2,
        )
        spatial_axis.scatter(
            [x for _, x, _ in photo_targets],
            [y for _, _, y in photo_targets],
            marker="*",
            s=65,
            facecolors="none",
            edgecolors="#2168a6",
            linewidths=0.9,
            alpha=0.35,
            label="全部拍照目标",
            zorder=2,
        )

        group_shoot = [candidate for candidate in group if candidate.task_type == "shoot"]
        group_photo = [candidate for candidate in group if candidate.task_type == "photo"]
        group_shoot_ids = sorted({candidate.task_id for candidate in group_shoot})
        group_photo_ids = sorted({candidate.task_id for candidate in group_photo})
        spatial_axis.scatter(
            [shoot_target_map[target_id][0] for target_id in group_shoot_ids],
            [shoot_target_map[target_id][1] for target_id in group_shoot_ids],
            marker="*",
            s=170,
            color="#b31d1d",
            label="本组射击目标",
            zorder=4,
        )
        spatial_axis.scatter(
            [photo_target_map[target_id][0] for target_id in group_photo_ids],
            [photo_target_map[target_id][1] for target_id in group_photo_ids],
            marker="*",
            s=170,
            color="#1261a0",
            label="本组拍照目标",
            zorder=4,
        )
        task_types_seen: set[str] = set()
        for task_index, candidate in enumerate(group):
            execution_x, execution_y = _candidate_position(candidate, positions)
            color = "#e2736a" if candidate.task_type == "shoot" else "#64add7"
            label = "_nolegend_"
            if candidate.task_type not in task_types_seen:
                task_types_seen.add(candidate.task_type)
                label = (
                    "射击执行位置"
                    if candidate.task_type == "shoot"
                    else "拍照执行位置"
                )
            spatial_axis.scatter(
                [execution_x],
                [execution_y],
                marker="o",
                s=42,
                color=color,
                label=label,
                zorder=5,
            )
            offset_x, offset_y = ((7, 8), (7, -21), (-54, 8), (-54, -21))[
                task_index % 4
            ]
            spatial_axis.annotate(
                f"{candidate.task_id}\n${candidate.execute_time:.1f}\\,\\mathrm{{s}}$",
                (execution_x, execution_y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="left" if offset_x > 0 else "right",
                va="center",
            fontsize=6.5,
                color="#7f2420" if candidate.task_type == "shoot" else "#175a8c",
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
                zorder=6,
            )

        start_time = mathtext_number(group[0].execute_time, 1)
        end_time = mathtext_number(group[-1].execute_time, 1)
        spatial_axis.set_title(
            "第 "
            f"{mathtext_number(group_index, 0)} 组："
            f"{start_time} 至 {end_time} 秒",
            fontproperties=chinese_font,
        )
        # 将坐标轴中的拉丁字母交给 Latin Modern Math，中文仍使用宋体。
        spatial_axis.set_xlabel(r"$X$ 坐标（米）", fontproperties=chinese_font)
        spatial_axis.set_ylabel(r"$Y$ 坐标（米）", fontproperties=chinese_font)
        spatial_axis.set_xlim(x_limits)
        spatial_axis.set_ylim(y_limits)
        spatial_axis.set_aspect("equal", adjustable="box")
        spatial_axis.grid(True, alpha=0.25)
        if group_index > 6:
            spatial_axis.set_xlabel(r"$X$ 坐标（米）", fontproperties=chinese_font)
        if group_index in (1, 4, 7):
            spatial_axis.set_ylabel(r"$Y$ 坐标（米）", fontproperties=chinese_font)

    # 使用整张图的公共图例，避免九个子图各自重复占用绘图区。
    legend_handles = [
        Line2D(
            [0], [0], color="#68737c", linewidth=0.8,
            label=r"问题三 $10\,\mathrm{Hz}$ 融合轨迹",
        ),
        Line2D(
            [0], [0], marker="*", color="#b31d1d", linestyle="None",
            markersize=10, label="射击目标（星形）",
        ),
        Line2D(
            [0], [0], marker="o", color="#e2736a", linestyle="None",
            markersize=6, label="射击执行位置（圆点）",
        ),
        Line2D(
            [0], [0], marker="*", color="#1261a0", linestyle="None",
            markersize=10, label="拍照目标（星形）",
        ),
        Line2D(
            [0], [0], marker="o", color="#64add7", linestyle="None",
            markersize=6, label="拍照执行位置（圆点）",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        prop=chinese_font,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "问题四：最优任务安排九宫格"
        f"（射击 {mathtext_number(len(shoot_tasks), 0)} 次，"
        f"拍照 {mathtext_number(len(photo_tasks), 0)} 次，"
        f"目标值 {mathtext_number(objective, 0)}）",
        fontproperties=chinese_font,
        fontsize=16,
    )
    output_path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_photo_views(
    chosen: list[TaskCandidate],
    save_name: str = "problem_four_photo_views.svg",
) -> Path:
    """绘制拍照任务的角度覆盖和各目标拍照次数。"""
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    chinese_font = chinese_font_properties()
    _, photo_tasks = _split_tasks(chosen)
    photo_target_ids = [target_id for target_id, _, _ in read_photo_targets()]
    target_indices = {target_id: index for index, target_id in enumerate(photo_target_ids)}
    photo_counts = Counter(candidate.task_id for candidate in photo_tasks)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    angle_axis, count_axis = axes

    angle_axis.scatter(
        [target_indices[candidate.task_id] for candidate in photo_tasks],
        [float(candidate.angle_deg) for candidate in photo_tasks],
        s=48,
        color="#1f77b4",
        label="已选拍照视角",
        zorder=3,
    )
    for angle in range(60, 360, 60):
        angle_axis.axhline(
            angle,
            color="#909090",
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
            zorder=1,
            label=r"$60^\circ$ 间隔参考线" if angle == 60 else None,
        )
    angle_axis.set_xticks(range(len(photo_target_ids)), photo_target_ids, rotation=45)
    angle_axis.set_yticks(range(0, 361, 60))
    angle_axis.set_ylim(-12, 372)
    angle_axis.set_title("各拍照目标的已选视角", fontproperties=chinese_font)
    angle_axis.set_xlabel("拍照目标编号", fontproperties=chinese_font)
    angle_axis.set_ylabel("相对方向角（度）", fontproperties=chinese_font)
    angle_axis.grid(True, axis="x", alpha=0.18)
    angle_axis.legend(loc="upper right", prop=chinese_font)

    counts = [photo_counts[target_id] for target_id in photo_target_ids]
    bars = count_axis.bar(
        photo_target_ids,
        counts,
        color="#4b90c8",
        edgecolor="#28618f",
        linewidth=0.6,
        label="已选拍照次数",
    )
    for bar, count in zip(bars, counts):
        if count:
            count_axis.text(
                bar.get_x() + bar.get_width() / 2,
                count + 0.07,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    count_axis.set_ylim(0, max(counts, default=0) + 1.0)
    count_axis.set_title("各拍照目标的已选次数", fontproperties=chinese_font)
    count_axis.set_xlabel("拍照目标编号", fontproperties=chinese_font)
    count_axis.set_ylabel("拍照次数", fontproperties=chinese_font)
    count_axis.tick_params(axis="x", rotation=45)
    count_axis.grid(True, axis="y", alpha=0.25)
    count_axis.legend(loc="upper right", prop=chinese_font)

    figure.suptitle(
        "问题四：多视角拍照覆盖"
        f"（共 {mathtext_number(len(photo_tasks), 0)} 次）",
        fontproperties=chinese_font,
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_conclusion() -> tuple[Path, Path]:
    """求解当前模型并生成一张九宫格空间调度图与一张拍照覆盖图。"""
    chosen, objective, status = solve_schedule()
    if status != "Optimal":
        raise RuntimeError(f"求解器未返回最优解：{status}")
    trajectory = trajectory_from_result_csv()
    return (
        plot_schedule(chosen, objective, trajectory),
        plot_photo_views(chosen),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    schedule_path, photo_views_path = plot_conclusion()
    print(f"已生成：{schedule_path}")
    print(f"已生成：{photo_views_path}")
