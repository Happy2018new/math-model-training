import math
from pathlib import Path

from .define import PosData, PosWithTime
from .delta_time import s1, s2, linear_interpolation, compute_delta_time
from .delta_offset import (
    compute_delta_offset,
    estimate_block_mean_bias,
    offset_sensor_data,
)
from .rand_error import compute_rand_error

FINAL_FUSION_WINDOW = 11


def resolve() -> tuple[
    list[PosWithTime],
    tuple[float, float],
]:
    new_s1 = s1
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    result = []
    start = min(new_s1.start_time, new_s2.start_time)
    end = max(new_s1.end_time, new_s2.end_time)

    # Start on the common 10 Hz grid even when an aligned stream begins at a
    # fractional tenth of a second.
    t = math.ceil(start * 10 - 1e-9) / 10.0
    while t <= end:
        pos1 = linear_interpolation(new_s1, t)
        pos2 = linear_interpolation(new_s2, t)

        if pos1 is None and pos2 is not None:
            result.append(PosWithTime(t, pos2))
        elif pos1 is not None and pos2 is None:
            result.append(PosWithTime(t, pos1))
        elif pos1 is not None and pos2 is not None:
            result.append(
                PosWithTime(
                    t,
                    PosData(
                        (pos1.posx + pos2.posx) / 2,
                        (pos1.posy + pos2.posy) / 2,
                    ),
                )
            )

        t = round(t + 0.1, 1)

    return smooth_trajectory(result, FINAL_FUSION_WINDOW), compute_rand_error()


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "three"


def _configure_chinese_font() -> None:
    from ..sensitivity_common import configure_plot_fonts

    configure_plot_fonts()


def smooth_trajectory(
    trajectory: list[PosWithTime],
    window: int = 3,
) -> list[PosWithTime]:
    """对 10 Hz 轨迹进行居中滑动平均。"""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    if not trajectory:
        return []

    radius = window // 2
    result = []
    for index, item in enumerate(trajectory):
        left = max(0, index - radius)
        right = min(len(trajectory), index + radius + 1)
        points = trajectory[left:right]
        result.append(
            PosWithTime(
                item.time,
                PosData(
                    sum(point.pos.posx for point in points) / len(points),
                    sum(point.pos.posy for point in points) / len(points),
                ),
            )
        )
    return result


def plot_10hz_trajectory(
    trajectory: list[PosWithTime] | None = None,
    save_path: str | None = None,
    show: bool = False,
    smooth_window: int | None = None,
    show_raw: bool = True,
):
    """绘制问题三的 10 Hz 融合轨迹主体结果。"""
    import matplotlib.pyplot as plt

    _configure_chinese_font()
    if trajectory is None:
        trajectory, _ = resolve()
    if not trajectory:
        raise ValueError("The 10 Hz trajectory is empty")
    smoothed = (
        smooth_trajectory(trajectory, smooth_window)
        if smooth_window is not None
        else None
    )
    times = [item.time for item in trajectory]
    xs = [item.pos.posx for item in trajectory]
    ys = [item.pos.posy for item in trajectory]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    if show_raw:
        axes[0].plot(xs, ys, linewidth=0.7, alpha=0.6, label="10 Hz 融合轨迹")
    if smoothed is not None:
        axes[0].plot(
            [item.pos.posx for item in smoothed],
            [item.pos.posy for item in smoothed],
            linewidth=1.4,
            label=f"平滑后（{smooth_window}点）",
        )
    displayed = smoothed if smoothed is not None and not show_raw else trajectory
    axes[0].scatter(
        displayed[0].pos.posx,
        displayed[0].pos.posy,
        color="green",
        s=30,
        label="起点",
        zorder=3,
    )
    axes[0].scatter(
        displayed[-1].pos.posx,
        displayed[-1].pos.posy,
        color="red",
        s=30,
        label="终点",
        zorder=3,
    )
    axes[0].set_title("融合轨迹")
    axes[0].set_xlabel("X 坐标")
    axes[0].set_ylabel("Y 坐标")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    if show_raw:
        axes[1].plot(times, xs, linewidth=0.7, alpha=0.6, label="原始数据")
        axes[2].plot(times, ys, linewidth=0.7, alpha=0.6, label="原始数据")
    if smoothed is not None:
        smooth_times = [item.time for item in smoothed]
        axes[1].plot(
            smooth_times,
            [item.pos.posx for item in smoothed],
            linewidth=1.2,
            label=f"平滑后（{smooth_window}点）",
        )
        axes[2].plot(
            smooth_times,
            [item.pos.posy for item in smoothed],
            linewidth=1.2,
            label=f"平滑后（{smooth_window}点）",
        )
    axes[1].set_title("X 坐标随时间变化")
    axes[1].set_xlabel("时间（秒）")
    axes[1].set_ylabel("X 坐标")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title("Y 坐标随时间变化")
    axes[2].set_xlabel("时间（秒）")
    axes[2].set_ylabel("Y 坐标")
    axes[2].grid(True, alpha=0.3)

    if save_path is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = (OUTPUT_DIR / Path(save_path).name).with_suffix(".svg")
        figure.savefig(output_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_conclusion(save_name: str = "problem_three_conclusion.svg") -> Path:
    """生成问题三的主体结论图。"""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..sensitivity_common import (
        chinese_font_properties,
        configure_plot_fonts,
        mathtext_number,
    )

    configure_plot_fonts()
    chinese_font = chinese_font_properties()
    # 与实际求解器共用时间偏差、插值和分块均值估计，避免图表另算一套结果。
    delta = compute_delta_time()
    dx_list, dy_list = compute_delta_offset(delta)
    aligned_s2 = offset_sensor_data(s2, -delta)
    start_time = max(s1.start_time, aligned_s2.start_time)
    times = np.array(
        [round(start_time + index * 0.1, 10) for index in range(len(dx_list))]
    )
    x1_aligned = np.array([linear_interpolation(s1, time).posx for time in times])
    y1_aligned = np.array([linear_interpolation(s1, time).posy for time in times])
    x2_aligned = np.array(
        [linear_interpolation(aligned_s2, time).posx for time in times]
    )
    y2_aligned = np.array(
        [linear_interpolation(aligned_s2, time).posy for time in times]
    )
    block_size = 20
    estimate = estimate_block_mean_bias(dx_list, dy_list, block_size)
    block_times = (
        start_time
        + np.arange(len(estimate.x_block_means)) * block_size / 10.0
    )
    bx = np.asarray(estimate.x_block_means)
    by = np.asarray(estimate.y_block_means)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(x1_aligned, y1_aligned, label="传感器1")
    axes[0, 0].plot(x2_aligned, y2_aligned, label="传感器2（时间对齐后）")
    final_trajectory, _ = resolve()
    axes[0, 0].plot(
        [item.pos.posx for item in final_trajectory],
        [item.pos.posy for item in final_trajectory],
        linewidth=1.4,
        label=f"最终融合轨迹（{FINAL_FUSION_WINDOW} 点平滑）",
    )
    axes[0, 0].set_title(
        f"时间对齐后的轨迹（偏差 {mathtext_number(delta, 3)} 秒）",
        fontproperties=chinese_font,
    )
    axes[0, 0].set_xlabel("X 坐标（米）")
    axes[0, 0].set_ylabel("Y 坐标（米）")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].scatter(bx, by, s=10, alpha=0.5, label="20 点分块均值")
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].scatter(
        [estimate.mean_x],
        [estimate.mean_y],
        color="#c45b3c",
        s=45,
        label="分块均值的算术平均",
    )
    axes[0, 1].set_title("空间偏差：20 点分块均值法")
    axes[0, 1].set_xlabel("X 方向分块均值（米）")
    axes[0, 1].set_ylabel("Y 方向分块均值（米）")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right")

    for axis, block_times, values, direction, low_key, high_key, mean_key in (
        (axes[1, 0], block_times, bx, "X", "x_low", "x_high", "mean_x"),
        (axes[1, 1], block_times, by, "Y", "y_low", "y_high", "mean_y"),
    ):
        axis.plot(
            block_times,
            values,
            marker=".",
            linewidth=0.8,
            label=f"{direction} 方向 20 点分块均值",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, label="零偏差")
        axis.axhspan(
            getattr(estimate, low_key),
            getattr(estimate, high_key),
            color="#2f7d5a",
            alpha=0.18,
            label="95% 置信区间",
        )
        axis.axhline(
            getattr(estimate, mean_key),
            color="#c45b3c",
            linestyle="--",
            label=f"分块均值的均值 {mathtext_number(getattr(estimate, mean_key), 3)} 米",
        )
        axis.set_title(f"{direction} 轴系统偏差检验")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("分块均值（米）")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", prop=chinese_font)

    figure.suptitle(
        f"问题三：5 点预平滑与实际测量数据的系统偏差检验（{block_size} 点分块均值法，95% 置信水平）",
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_conclusion())
