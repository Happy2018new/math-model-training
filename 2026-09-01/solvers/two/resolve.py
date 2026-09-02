from pathlib import Path

from .define import PosData, PosWithTime
from .delta_time import linear_interpolation
from .rand_error import compute_rand_error


def resolve() -> tuple[
    list[PosWithTime],
    tuple[float, float],
]:
    ans, err = compute_rand_error()
    _, s1, s2 = ans

    result = []
    start = min(s1.start_time, s2.start_time)
    end = max(s1.end_time, s2.end_time)

    t = start
    while t <= end:
        pos1 = linear_interpolation(s1, t)
        pos2 = linear_interpolation(s2, t)

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

    return result, err


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "two"


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
    """绘制问题二的 10 Hz 融合轨迹主体结果。"""
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
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    if show_raw:
        axes[0].plot(xs, ys, linewidth=1.0, label="10 Hz 融合轨迹")
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
        s=40,
        label="起点",
        zorder=3,
    )
    axes[0].scatter(
        displayed[-1].pos.posx,
        displayed[-1].pos.posy,
        color="red",
        s=40,
        label="终点",
        zorder=3,
    )
    axes[0].set_title("融合轨迹")
    axes[0].set_xlabel("X 坐标")
    axes[0].set_ylabel("Y 坐标")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if show_raw:
        axes[1].plot(times, xs, linewidth=0.8, label="原始数据")
        axes[2].plot(times, ys, linewidth=0.8, label="原始数据")
    if smoothed is not None:
        smooth_times = [item.time for item in smoothed]
        axes[1].plot(
            smooth_times,
            [item.pos.posx for item in smoothed],
            linewidth=1.0,
            label=f"平滑后（{smooth_window}点）",
        )
        axes[2].plot(
            smooth_times,
            [item.pos.posy for item in smoothed],
            linewidth=1.0,
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

    figure.tight_layout()
    if save_path is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = (OUTPUT_DIR / Path(save_path).name).with_suffix(".svg")
        figure.savefig(output_path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    return figure, axes


def plot_conclusion(save_name: str = "problem_two_conclusion.svg") -> Path:
    """生成问题二的主体结论图。"""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..sensitivity_common import (
        aligned_differences,
        configure_plot_fonts,
        estimate_delta_time,
        load_sensor_csv,
        smooth_sensor,
    )

    configure_plot_fonts()
    data_dir = Path(__file__).resolve().parents[2] / "inputs"
    t1, x1, y1 = load_sensor_csv(data_dir / "table_2_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(data_dir / "table_2_sensor_2.csv")
    st1, sx1, sy1 = smooth_sensor(t1, x1, y1, 5)
    st2, sx2, sy2 = smooth_sensor(t2, x2, y2, 5)
    delta, _ = estimate_delta_time(
        st1, sx1, sy1, st2, sx2, sy2, 0.0, 1000.0
    )
    times, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
    x1_aligned = np.interp(times, t1, x1)
    y1_aligned = np.interp(times, t1, y1)
    x2_aligned = np.interp(times + delta, t2, x2)
    y2_aligned = np.interp(times + delta, t2, y2)
    mean_dx, mean_dy = float(dx.mean()), float(dy.mean())
    x2_corrected = x2_aligned - mean_dx
    y2_corrected = y2_aligned - mean_dy
    fused_x = (x1_aligned + x2_corrected) / 2.0
    fused_y = (y1_aligned + y2_corrected) / 2.0
    rmse = float(np.sqrt(np.mean(dx**2 + dy**2)))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(x1_aligned, y1_aligned, label="传感器1")
    axes[0, 0].plot(x2_aligned, y2_aligned, label="传感器2（时间对齐后）")
    axes[0, 0].set_title(f"时间对齐后的两条轨迹（偏差 {delta:.3f} 秒）")
    axes[0, 0].set_xlabel("X 坐标（米）")
    axes[0, 0].set_ylabel("Y 坐标（米）")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(x1_aligned, y1_aligned, label="传感器1")
    axes[0, 1].plot(x2_corrected, y2_corrected, label="传感器2（空间校正后）")
    axes[0, 1].plot(fused_x, fused_y, linewidth=1.6, label="融合轨迹")
    axes[0, 1].set_title(f"空间偏移校正与融合（RMSE {rmse:.3f} 米）")
    axes[0, 1].set_xlabel("X 坐标（米）")
    axes[0, 1].set_ylabel("Y 坐标（米）")
    axes[0, 1].axis("equal")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    for axis, values, mean_value, direction in (
        (axes[1, 0], dx, mean_dx, "X"),
        (axes[1, 1], dy, mean_dy, "Y"),
    ):
        axis.plot(times, values, linewidth=0.8, label=f"{direction} 方向差值")
        axis.axhline(
            mean_value,
            color="#c45b3c",
            linestyle="--",
            label=f"固定偏移 {mean_value:.3f} 米",
        )
        axis.set_title(f"{direction} 方向空间差异")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("差值（米）")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle("问题二：时间对齐、空间偏移校正与轨迹融合", fontsize=14)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_conclusion())
