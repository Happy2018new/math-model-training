from pathlib import Path

from .define import PosData, PosWithTime
from .delta_time import s1, s2, linear_interpolation, compute_delta_time
from .delta_offset import offset_sensor_data
from .rand_error import compute_rand_error


def resolve() -> tuple[
    list[PosWithTime],
    tuple[float, float],
]:
    new_s1 = s1
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    result = []
    start = min(new_s1.start_time, new_s2.start_time)
    end = max(new_s1.end_time, new_s2.end_time)

    t = start
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

    return result, compute_rand_error()


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
    axes[0].legend()

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


def _block_series(values, block_size: int):
    import numpy as np

    count = len(values) // block_size
    blocks = values[: count * block_size].reshape(count, block_size).mean(axis=1)
    return np.arange(count) * block_size / 10.0, blocks


def plot_conclusion(save_name: str = "problem_three_conclusion.svg") -> Path:
    """生成问题三的主体结论图。"""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..sensitivity_common import (
        aligned_differences,
        bias_confidence,
        configure_plot_fonts,
        estimate_delta_time,
        load_sensor_csv,
        smooth_sensor,
    )

    configure_plot_fonts()
    data_dir = Path(__file__).resolve().parents[2] / "inputs"
    t1, x1, y1 = load_sensor_csv(data_dir / "table_3_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(data_dir / "table_3_sensor_2.csv")
    st1, sx1, sy1 = smooth_sensor(t1, x1, y1, 5)
    st2, sx2, sy2 = smooth_sensor(t2, x2, y2, 5)
    delta, _ = estimate_delta_time(
        st1, sx1, sy1, st2, sx2, sy2, -1500.0, 1500.0
    )
    times, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
    x1_aligned = np.interp(times, t1, x1)
    y1_aligned = np.interp(times, t1, y1)
    x2_aligned = np.interp(times + delta, t2, x2)
    y2_aligned = np.interp(times + delta, t2, y2)
    block_size = 20
    confidence = bias_confidence(dx, dy, block_size=block_size, confidence=0.95)
    bx_time, bx = _block_series(dx, block_size)
    by_time, by = _block_series(dy, block_size)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(x1_aligned, y1_aligned, label="传感器1")
    axes[0, 0].plot(x2_aligned, y2_aligned, label="传感器2（时间对齐后）")
    axes[0, 0].set_title(f"时间对齐后的轨迹（偏差 {delta:.3f} 秒）")
    axes[0, 0].set_xlabel("X 坐标（米）")
    axes[0, 0].set_ylabel("Y 坐标（米）")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].scatter(bx, by, s=10, alpha=0.5, label="分块误差均值")
    axes[0, 1].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].scatter(
        [confidence["mean_x"]],
        [confidence["mean_y"]],
        color="#c45b3c",
        s=45,
        label="总体均值",
    )
    axes[0, 1].set_title("分块空间偏差散点")
    axes[0, 1].set_xlabel("X 方向分块均值（米）")
    axes[0, 1].set_ylabel("Y 方向分块均值（米）")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    for axis, block_times, values, direction, low_key, high_key, mean_key in (
        (axes[1, 0], bx_time, bx, "X", "x_low", "x_high", "mean_x"),
        (axes[1, 1], by_time, by, "Y", "y_low", "y_high", "mean_y"),
    ):
        axis.plot(
            block_times,
            values,
            marker=".",
            linewidth=0.8,
            label=f"{direction} 方向分块均值",
        )
        axis.axhline(0.0, color="black", linewidth=0.8, label="零偏差")
        axis.axhspan(
            confidence[low_key],
            confidence[high_key],
            color="#2f7d5a",
            alpha=0.18,
            label="95% 置信区间",
        )
        axis.axhline(
            confidence[mean_key],
            color="#c45b3c",
            linestyle="--",
            label=f"均值 {confidence[mean_key]:.3f} 米",
        )
        axis.set_title(f"{direction} 轴系统偏差检验")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("分块均值（米）")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(
        f"问题三：实际测量数据的系统偏差检验（{block_size} 点分块，95% 置信水平）",
        fontsize=14,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_conclusion())
