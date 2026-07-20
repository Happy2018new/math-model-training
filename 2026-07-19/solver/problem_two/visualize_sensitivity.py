"""绘制第二题第二问的分位数敏感性分析图。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter

from .sensitivity import DEFAULT_CSV_PATH, PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "two" / "figures"

COLOR_INK = "#172033"
COLOR_MUTED = "#64748B"
COLOR_GRID = "#DCE5EF"
COLOR_BLUE = "#2563EB"
COLOR_CYAN = "#0891B2"
COLOR_GREEN = "#0F9F7A"
COLOR_ORANGE = "#F59E0B"
COLOR_RED = "#E5484D"
COLOR_WHITE = "#FFFFFF"
COLOR_SOFT = "#F7FAFC"
TEXT_FONT = "SimSun"
NUMBER_FONT = "Latin Modern Math"


@dataclass(frozen=True)
class SensitivityPlotRecord:
    """绘图所需的单个敏感性场景数据。"""

    supply_percentile: float
    loss_percentile: float
    feasible: bool
    meets_gap: bool | None
    provider_count: int | None
    provider_count_a: int | None
    provider_count_b: int | None
    provider_count_c: int | None
    optimized_cost: float | None
    optimized_transport_loss: float | None
    transport_loss_reduction_rate: float | None


def _configure_style() -> None:
    """设置统一的白底现代论文绘图风格。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    missing_fonts = {
        font_name
        for font_name in (TEXT_FONT, NUMBER_FONT)
        if font_name not in available_fonts
    }
    if missing_fonts:
        raise RuntimeError(f"缺少绘图字体：{', '.join(sorted(missing_fonts))}")
    matplotlib.rcParams.update(
        {
            # Latin Modern Math 优先渲染数字与字母，缺少的中文字形回退到宋体。
            "font.family": [NUMBER_FONT, TEXT_FONT],
            "font.serif": [NUMBER_FONT, TEXT_FONT],
            "font.sans-serif": [NUMBER_FONT, TEXT_FONT],
            "axes.unicode_minus": False,
            "figure.facecolor": COLOR_WHITE,
            "axes.facecolor": COLOR_WHITE,
            "savefig.facecolor": COLOR_WHITE,
            "savefig.edgecolor": COLOR_WHITE,
            "savefig.transparent": False,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_MUTED,
            "axes.titlecolor": COLOR_INK,
            "axes.titlesize": 15,
            "axes.titleweight": "semibold",
            "text.color": COLOR_INK,
            "font.size": 10.5,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "grid.color": COLOR_GRID,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.stretch": "normal",
        }
    )
    matplotlib.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": NUMBER_FONT,
            "mathtext.it": NUMBER_FONT,
            "mathtext.bf": NUMBER_FONT,
        }
    )


def _parse_optional_float(value: str | None) -> float | None:
    return None if value is None or value.strip() == "" else float(value)


def _parse_optional_int(value: str | None) -> int | None:
    return None if value is None or value.strip() == "" else int(value)


def _parse_bool(value: str | None) -> bool | None:
    if value is None or value.strip() == "":
        return None
    return value.strip().lower() == "true"


def load_sensitivity_data(
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[SensitivityPlotRecord]:
    """从敏感性分析生成的结构化 CSV 中读取绘图数据。"""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到敏感性分析数据：{csv_path}。请先运行 run_sensitivity()。"
        )

    records: list[SensitivityPlotRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {
            "supply_percentile",
            "loss_percentile",
            "feasible",
            "meets_gap",
            "provider_count",
            "provider_count_a",
            "provider_count_b",
            "provider_count_c",
            "optimized_cost",
            "optimized_transport_loss",
            "transport_loss_reduction_rate",
        }
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"敏感性 CSV 缺少字段：{', '.join(sorted(missing))}")

        for row in reader:
            records.append(
                SensitivityPlotRecord(
                    supply_percentile=float(row["supply_percentile"]),
                    loss_percentile=float(row["loss_percentile"]),
                    feasible=bool(_parse_bool(row["feasible"])),
                    meets_gap=_parse_bool(row["meets_gap"]),
                    provider_count=_parse_optional_int(row["provider_count"]),
                    provider_count_a=_parse_optional_int(row["provider_count_a"]),
                    provider_count_b=_parse_optional_int(row["provider_count_b"]),
                    provider_count_c=_parse_optional_int(row["provider_count_c"]),
                    optimized_cost=_parse_optional_float(row["optimized_cost"]),
                    optimized_transport_loss=_parse_optional_float(
                        row["optimized_transport_loss"]
                    ),
                    transport_loss_reduction_rate=_parse_optional_float(
                        row["transport_loss_reduction_rate"]
                    ),
                )
            )
    if not records:
        raise ValueError("敏感性 CSV 中没有数据")
    return records


def _percentile_label(value: float) -> str:
    percentage = value * 100
    digits = 0 if abs(percentage - round(percentage)) < 1e-8 else 1
    return f"P{percentage:.{digits}f}"


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    """保存可无损缩放和编辑的 SVG 矢量图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.svg"]
    figure.savefig(
        paths[0],
        bbox_inches="tight",
        facecolor=COLOR_WHITE,
        edgecolor=COLOR_WHITE,
    )
    plt.close(figure)
    return paths


def _build_matrix(
    records: Sequence[SensitivityPlotRecord],
    supply_values: Sequence[float],
    loss_values: Sequence[float],
    getter: Callable[[SensitivityPlotRecord], float | int | None],
) -> np.ndarray:
    matrix = np.full((len(supply_values), len(loss_values)), np.nan)
    supply_index = {value: index for index, value in enumerate(supply_values)}
    loss_index = {value: index for index, value in enumerate(loss_values)}
    for record in records:
        value = getter(record)
        if record.feasible and value is not None:
            matrix[
                supply_index[record.supply_percentile],
                loss_index[record.loss_percentile],
            ] = float(value)
    return matrix


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(COLOR_GRID)
    axis.spines["bottom"].set_color(COLOR_GRID)


def plot_sensitivity_heatmaps(
    records: Sequence[SensitivityPlotRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """绘制成本、运输损耗和供应商数量的二维敏感性热力图。"""
    supply_values = sorted({record.supply_percentile for record in records})
    loss_values = sorted({record.loss_percentile for record in records})
    cost = _build_matrix(
        records,
        supply_values,
        loss_values,
        lambda record: (
            record.optimized_cost / 10000
            if record.optimized_cost is not None
            else None
        ),
    )
    loss = _build_matrix(
        records,
        supply_values,
        loss_values,
        lambda record: record.optimized_transport_loss,
    )
    provider_count = _build_matrix(
        records,
        supply_values,
        loss_values,
        lambda record: record.provider_count,
    )
    record_map = {
        (record.supply_percentile, record.loss_percentile): record
        for record in records
    }

    colormaps = (
        LinearSegmentedColormap.from_list("cost", ("#EDF4FF", "#BFD5FF", COLOR_BLUE)),
        LinearSegmentedColormap.from_list("loss", ("#EAFBF7", "#9DE3D2", COLOR_GREEN)),
        LinearSegmentedColormap.from_list("provider", ("#FFF8E8", "#FBD58A", COLOR_ORANGE)),
    )
    titles = ("优化经济成本", "优化运输损耗", "最少供应商数量")
    subtitles = ("成本单位 ×10^4", "未来 12 周累计损耗", "A、B、C 三类合计")
    matrices = (cost, loss, provider_count)
    formats = (lambda value: f"{value:.2f}", lambda value: f"{value:.0f}", lambda value: f"{value:.0f} 家")

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.8), facecolor=COLOR_WHITE)
    figure.suptitle(
        "供货与损耗分位数的二维敏感性",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=20,
        fontweight="semibold",
        color=COLOR_INK,
    )
    figure.text(
        0.055,
        0.905,
        "横向比较损耗预测的保守程度，纵向比较供应能力预测水平",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    for axis, matrix, cmap, title, subtitle, formatter in zip(
        axes, matrices, colormaps, titles, subtitles, formats
    ):
        masked = np.ma.masked_invalid(matrix)
        image = axis.imshow(masked, origin="lower", cmap=cmap, aspect="auto")
        image.cmap.set_bad(COLOR_WHITE)
        axis.set_title(title, loc="left", pad=24)
        axis.text(
            0,
            1.02,
            subtitle,
            transform=axis.transAxes,
            color=COLOR_MUTED,
            fontsize=9,
        )
        axis.set_xticks(
            range(len(loss_values)),
            [_percentile_label(value) for value in loss_values],
        )
        axis.set_yticks(
            range(len(supply_values)),
            [_percentile_label(value) for value in supply_values],
        )
        axis.set_xlabel("转运损耗分位数", labelpad=10)
        if axis is axes[0]:
            axis.set_ylabel("供货预测分位数", labelpad=10)
        axis.set_xticks(np.arange(-0.5, len(loss_values), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(supply_values), 1), minor=True)
        axis.grid(which="minor", color=COLOR_WHITE, linewidth=3)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.tick_params(length=0, pad=7)

        valid_values = matrix[np.isfinite(matrix)]
        midpoint = (
            (float(valid_values.min()) + float(valid_values.max())) / 2
            if valid_values.size
            else 0.0
        )
        for row, supply_value in enumerate(supply_values):
            for column, loss_value in enumerate(loss_values):
                record = record_map[(supply_value, loss_value)]
                value = matrix[row, column]
                if not np.isfinite(value):
                    axis.add_patch(
                        Rectangle(
                            (column - 0.48, row - 0.48),
                            0.96,
                            0.96,
                            facecolor=COLOR_SOFT,
                            edgecolor=COLOR_GRID,
                            linewidth=1.0,
                            hatch="///",
                        )
                    )
                    axis.text(
                        column,
                        row,
                        "不可行\nB 类能力不足",
                        ha="center",
                        va="center",
                        color=COLOR_RED,
                        fontsize=8.5,
                        linespacing=1.4,
                    )
                    continue

                text_color = COLOR_WHITE if value > midpoint else COLOR_INK
                suffix = " †" if record.meets_gap is False else ""
                axis.text(
                    column,
                    row,
                    formatter(float(value)) + suffix,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="semibold",
                )
                if record.meets_gap is False:
                    axis.add_patch(
                        Rectangle(
                            (column - 0.46, row - 0.46),
                            0.92,
                            0.92,
                            fill=False,
                            edgecolor=COLOR_RED,
                            linewidth=1.4,
                            linestyle=(0, (3, 2)),
                        )
                    )

        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.outline.set_visible(False)
        colorbar.ax.tick_params(colors=COLOR_MUTED, labelsize=8, length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    figure.text(
        0.055,
        0.035,
        "注：斜线单元格表示模型不可行；† 和红色虚线框表示在 60 秒内未达到 0.5% MIP Gap。",
        color=COLOR_MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.17, wspace=0.30)
    return _save_figure(figure, output_dir, "sensitivity_heatmaps")


def plot_sensitivity_trends(
    records: Sequence[SensitivityPlotRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """绘制不同供货分位数下成本和损耗随损耗分位数的变化。"""
    supply_values = sorted({record.supply_percentile for record in records})
    loss_values = sorted({record.loss_percentile for record in records})
    colors = (COLOR_BLUE, COLOR_CYAN, COLOR_ORANGE, COLOR_RED)
    record_map = {
        (record.supply_percentile, record.loss_percentile): record
        for record in records
    }

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), facecolor=COLOR_WHITE)
    figure.suptitle(
        "预测分位数对订货方案的交互影响",
        x=0.065,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.92,
        "损耗预测越保守，运输损耗快速上升；供货分位数同时改变供应商结构",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    plotted_handles = []
    plotted_labels = []
    for color, supply_value in zip(colors, supply_values):
        scenario_records = [
            record_map[(supply_value, loss_value)] for loss_value in loss_values
        ]
        if not any(record.feasible for record in scenario_records):
            continue
        x = np.array(loss_values) * 100
        cost = np.array(
            [
                record.optimized_cost / 10000
                if record.feasible and record.optimized_cost is not None
                else np.nan
                for record in scenario_records
            ]
        )
        loss = np.array(
            [
                record.optimized_transport_loss
                if record.feasible and record.optimized_transport_loss is not None
                else np.nan
                for record in scenario_records
            ]
        )
        line = axes[0].plot(
            x,
            cost,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=7,
            markeredgecolor=COLOR_WHITE,
            markeredgewidth=1.2,
        )[0]
        axes[1].plot(
            x,
            loss,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=7,
            markeredgecolor=COLOR_WHITE,
            markeredgewidth=1.2,
        )
        plotted_handles.append(line)
        plotted_labels.append(f"供货 {_percentile_label(supply_value)}")

        for index, record in enumerate(scenario_records):
            if record.feasible and record.meets_gap is False:
                for axis, values in ((axes[0], cost), (axes[1], loss)):
                    axis.scatter(
                        x[index],
                        values[index],
                        s=78,
                        facecolor=COLOR_WHITE,
                        edgecolor=COLOR_RED,
                        linewidth=1.6,
                        zorder=5,
                    )

    axes[0].set_title("经济成本保持平稳，但供应结构发生变化", loc="left", pad=16)
    axes[0].set_ylabel("优化经济成本（×10^4）")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    axes[1].set_title("高损耗分位数显著放大运输损耗", loc="left", pad=16)
    axes[1].set_ylabel("未来 12 周运输损耗")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    for axis in axes:
        axis.set_xlabel("转运损耗分位数")
        axis.set_xticks(np.array(loss_values) * 100)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"P{value:g}"))
        axis.margins(x=0.06, y=0.16)
        _style_axis(axis)

    unavailable_patch = Patch(
        facecolor=COLOR_SOFT,
        edgecolor=COLOR_RED,
        hatch="///",
        label="供货 P90：不可行",
    )
    gap_handle = plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor=COLOR_WHITE,
        markeredgecolor=COLOR_RED,
        markeredgewidth=1.5,
        label="未达到 0.5% Gap",
    )
    figure.legend(
        plotted_handles + [unavailable_patch, gap_handle],
        plotted_labels + ["供货 P90：不可行", "未达到 0.5% Gap"],
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.015),
        columnspacing=1.5,
        handlelength=2.2,
    )
    figure.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.21, wspace=0.25)
    return _save_figure(figure, output_dir, "sensitivity_trends")


def plot_provider_structure(
    records: Sequence[SensitivityPlotRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """绘制供货分位数变化下的最少供应商结构。"""
    supply_values = sorted({record.supply_percentile for record in records})
    representative: dict[float, SensitivityPlotRecord] = {}
    for supply_value in supply_values:
        matching = [
            record for record in records if record.supply_percentile == supply_value
        ]
        representative[supply_value] = next(
            (record for record in matching if record.feasible), matching[0]
        )

    x = np.arange(len(supply_values))
    figure, axis = plt.subplots(figsize=(10.8, 6.2), facecolor=COLOR_WHITE)
    figure.suptitle(
        "供货预测水平改变最少供应商结构",
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.09,
        0.905,
        "较高的供货分位数提高单家供应能力，C 类供应商数量由 6 家降至 2 家",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    bottoms = np.zeros(len(supply_values))
    categories = (
        ("A 类", "provider_count_a", COLOR_BLUE),
        ("B 类", "provider_count_b", COLOR_GREEN),
        ("C 类", "provider_count_c", COLOR_ORANGE),
    )
    for label, attribute, color in categories:
        values = np.array(
            [
                float(getattr(representative[value], attribute) or 0)
                for value in supply_values
            ]
        )
        bars = axis.bar(
            x,
            values,
            bottom=bottoms,
            width=0.58,
            color=color,
            edgecolor=COLOR_WHITE,
            linewidth=1.5,
            label=label,
        )
        for bar, value, bottom, supply_value in zip(
            bars, values, bottoms, supply_values
        ):
            if value > 0 and representative[supply_value].feasible:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color=COLOR_WHITE,
                    fontsize=11,
                    fontweight="bold",
                )
        bottoms += values

    for index, supply_value in enumerate(supply_values):
        record = representative[supply_value]
        if record.feasible and record.provider_count is not None:
            axis.text(
                index,
                record.provider_count + 0.35,
                f"共 {record.provider_count} 家",
                ha="center",
                va="bottom",
                color=COLOR_INK,
                fontsize=10.5,
                fontweight="semibold",
            )
        else:
            axis.add_patch(
                Rectangle(
                    (index - 0.29, 0),
                    0.58,
                    2.1,
                    facecolor=COLOR_SOFT,
                    edgecolor=COLOR_RED,
                    linewidth=1.2,
                    hatch="///",
                )
            )
            axis.text(
                index,
                2.55,
                "不可行\nB 类能力不足",
                ha="center",
                va="bottom",
                color=COLOR_RED,
                fontsize=9.5,
                linespacing=1.35,
            )

    axis.set_xticks(x, [_percentile_label(value) for value in supply_values])
    axis.set_xlabel("供货预测分位数", labelpad=12)
    axis.set_ylabel("最少供应商数量（家）")
    axis.set_ylim(0, max(bottoms.max() + 1.7, 11.5))
    axis.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _style_axis(axis)
    axis.legend(
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.08),
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.14)
    return _save_figure(figure, output_dir, "sensitivity_provider_structure")


def run_visualization(
    csv_path: Path = DEFAULT_CSV_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """读取敏感性 CSV 并生成全部论文图。"""
    _configure_style()
    records = load_sensitivity_data(csv_path)
    paths: list[Path] = []
    paths.extend(plot_sensitivity_heatmaps(records, output_dir))
    paths.extend(plot_sensitivity_trends(records, output_dir))
    paths.extend(plot_provider_structure(records, output_dir))
    return paths


if __name__ == "__main__":
    for generated_path in run_visualization():
        print(generated_path)
