import math
import copy
from .prepare import data1, data2
from .define import PosData, PosWithTime, SensorData, sensor_data_from_raw

s1 = sensor_data_from_raw(data1, 4)
s2 = sensor_data_from_raw(data2, 5)


def linear_interpolation(data: SensorData, time: float) -> PosData | None:
    if time < data.start_time or time > data.end_time:
        return None

    ordinal = int((time - data.start_time) * data.sensor_hz + 1)
    start = max(0, ordinal - 1)
    end = min(ordinal, len(data.payload) - 1)
    if start == end:
        return data.payload[start]

    t1 = data.start_time + start / data.sensor_hz
    t2 = data.start_time + end / data.sensor_hz
    ratio = (time - t1) / (t2 - t1)

    pos1 = data.payload[start]
    pos2 = data.payload[end]
    return PosData(
        (1 - ratio) * pos1.posx + ratio * pos2.posx,
        (1 - ratio) * pos1.posy + ratio * pos2.posy,
    )


def compute_distance(pos1: PosData, pos2: PosData) -> float:
    dx = pos1.posx - pos2.posx
    dy = pos1.posy - pos2.posy
    return dx**2 + dy**2


def guess_delta_time(answer: float) -> float | None:
    start = max(s1.start_time, s2.start_time - answer)
    end = min(s1.end_time, s2.end_time - answer)
    if start > end:
        return None

    total = 0.0
    count = 0
    for index, pos1 in enumerate(s1.payload):
        current_time = s1.start_time + index / s1.sensor_hz
        if current_time < start:
            continue
        if current_time > end:
            break
        pos2 = linear_interpolation(s2, current_time + answer)
        if pos2 is None:
            continue
        total += compute_distance(pos1, pos2)
        count += 1

    if count < 0.8 * len(s1.payload):
        return None
    return total / count


def offset_sensor_data(data: SensorData, delta_time: float) -> SensorData:
    data = copy.deepcopy(data)
    data.start_time += delta_time
    data.end_time += delta_time
    return data


def generate_10hz_ans(delta_time: float) -> list[PosWithTime]:
    result = []

    new_s1, new_s2 = s1, offset_sensor_data(s2, -delta_time)
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

    return result


def resolve() -> list[PosWithTime]:
    ans = 0.0
    err = math.inf
    ptr = 0.0
    while ptr < 1500.0:
        temp = guess_delta_time(ptr)
        if temp is not None and temp < err:
            err = temp
            ans = ptr
        ptr += 0.1

    final_ans = ans
    final_err = err
    final_ptr = ans - 0.1
    while final_ptr <= ans + 0.1:
        temp = guess_delta_time(final_ptr)
        if temp is not None and temp < final_err:
            final_err = temp
            final_ans = final_ptr
        final_ptr += 0.0001

    return generate_10hz_ans(final_ans)


def plot_conclusion(save_name: str = "problem_one_conclusion.svg"):
    """生成问题一的主体结论图。"""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from ..sensitivity_common import (
        aligned_differences,
        chinese_font_properties,
        configure_plot_fonts,
        estimate_delta_time,
        fixed_mathtext_formatter,
        load_sensor_csv,
        mathtext_number,
        moving_average,
        smooth_sensor,
    )

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs" / "one"
    configure_plot_fonts()
    chinese_font = chinese_font_properties()

    t1, x1, y1 = load_sensor_csv(data_dir / "table_1_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(data_dir / "table_1_sensor_2.csv")
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
    raw_objective = float(
        np.mean((dx - dx.mean()) ** 2 + (dy - dy.mean()) ** 2)
    )
    residual_rmse = float(np.sqrt(np.mean(dx**2 + dy**2)))
    residual_window = 11
    raw_display_stride = 10
    smooth_dx = moving_average(dx, residual_window)
    smooth_dy = moving_average(dy, residual_window)

    # Reserve a dedicated header band for the title and the numeric summary;
    # otherwise the summary text can touch the top-row axes after tight export.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(x1, y1, label="方式1")
    axes[0, 0].plot(x2, y2, label="方式2（未对齐）")
    axes[0, 0].set_title("对齐前的原始轨迹")
    axes[0, 0].set_xlabel("X 坐标（米）")
    axes[0, 0].set_ylabel("Y 坐标（米）")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(x1_aligned, y1_aligned, label="方式1")
    axes[0, 1].plot(x2_aligned, y2_aligned, label="方式2（时间平移后）")
    axes[0, 1].set_title(
        f"时间对齐后的轨迹（偏差 {mathtext_number(delta, 3)} 秒）",
        fontproperties=chinese_font,
    )
    axes[0, 1].set_xlabel("X 坐标（米）")
    axes[0, 1].set_ylabel("Y 坐标（米）")
    axes[0, 1].axis("equal")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right")

    # The raw residual is sampled at 10 Hz after interpolating 4 Hz and 5 Hz
    # sources.  Show it lightly, then overlay a residual-domain moving average
    # so interpolation-scale oscillation does not obscure the conclusion.
    for axis, residual, smoothed, direction in (
        (axes[1, 0], dx, smooth_dx, "X"),
        (axes[1, 1], dy, smooth_dy, "Y"),
    ):
        residual_um = residual * 1_000_000.0
        smoothed_um = smoothed * 1_000_000.0
        mean_um = float(residual.mean() * 1_000_000.0)
        axis.scatter(
            times[::raw_display_stride],
            residual_um[::raw_display_stride],
            color="#98a6ad",
            s=5,
            alpha=0.55,
            linewidths=0,
            label="原始差值（每秒抽样）",
        )
        axis.plot(
            times,
            smoothed_um,
            color="#007c83",
            linewidth=1.2,
            label=f"{residual_window} 点平滑趋势",
        )
        axis.axhline(
            mean_um,
            color="#c45b3c",
            linestyle="--",
            linewidth=1.0,
            label=f"平均值 {mathtext_number(mean_um, 2)} 微米",
        )
        axis.set_title(f"对齐后的 {direction} 方向残差")
        axis.set_xlabel("时间（秒）")
        axis.set_ylabel("残差（微米）")
        axis.yaxis.set_major_formatter(fixed_mathtext_formatter(0))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", prop=chinese_font)

    residual_mm = residual_rmse * 1000.0
    objective_mm2 = raw_objective * 1_000_000.0
    figure.suptitle("问题一：两类数据时间对齐结果", fontsize=14, y=0.975)
    figure.text(
        0.5,
        0.935,
        rf"$\delta={delta:.3f}\ \mathrm{{s}},\quad "
        rf"\mathrm{{RMSE}}\approx{residual_mm:.3f}\ \mathrm{{mm}},\quad "
        rf"J\approx{objective_mm2:.5f}\ \mathrm{{mm}}^2$",
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / save_name).with_suffix(".svg")
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(plot_conclusion())
