"""可视化第三题供货偏差与损耗下降敏感性分析结果。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .sensitivity import DEFAULT_OUTPUT_PATH as DEFAULT_SENSITIVITY_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "three" / "figures"

COLOR_WHITE = "#FFFFFF"
COLOR_INK = "#172033"
COLOR_MUTED = "#64748B"
COLOR_GRID = "#DCE5EF"
COLOR_SOFT = "#F7FAFC"
COLOR_BLUE = "#2563EB"
COLOR_CYAN = "#0891B2"
COLOR_GREEN = "#0F9F7A"
COLOR_ORANGE = "#F59E0B"
COLOR_RED = "#E5484D"
TEXT_FONT = "SimSun"
NUMBER_FONT = "Latin Modern Math"


@dataclass(frozen=True)
class SensitivityRecord:
    """敏感性分析汇总表中的一个情景。"""

    scenario_id: int
    supply_deviation: float
    loss_reduction: float
    completed: bool
    candidate_count: int | None
    active_provider_count: int | None
    total_cost: float | None
    transport_loss: float | None
    minimum_inventory_margin: float | None
    maximum_transfer_load: float | None
    primary_mip_gap: float | None
    secondary_mip_gap: float | None
    meets_gap: bool


def _configure_style() -> None:
    """设置纯白背景及宋体、Latin Modern Math 字体回退。"""
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
            # 数字和字母优先使用 Latin Modern Math，中文字形回退到宋体。
            "font.family": [NUMBER_FONT, TEXT_FONT],
            "font.serif": [NUMBER_FONT, TEXT_FONT],
            "font.sans-serif": [NUMBER_FONT, TEXT_FONT],
            "mathtext.fontset": "custom",
            "mathtext.rm": NUMBER_FONT,
            "mathtext.it": NUMBER_FONT,
            "mathtext.bf": NUMBER_FONT,
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
            "font.size": 10.5,
            "text.color": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "grid.color": COLOR_GRID,
            "grid.alpha": 0.72,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "font.stretch": "normal",
        }
    )


def _parse_percent(value: str) -> float:
    return float(value.removesuffix("%")) / 100.0


def load_sensitivity_data(
    report_path: Path = DEFAULT_SENSITIVITY_PATH,
) -> list[SensitivityRecord]:
    """从敏感性 TXT 的情景汇总表读取结构化绘图数据。"""
    if not report_path.exists():
        raise FileNotFoundError(
            f"未找到敏感性分析结果：{report_path}。请先运行 sensitivity.py。"
        )
    content = report_path.read_text(encoding="utf-8-sig")
    summary_match = re.search(
        r"二、情景汇总\s*(.*?)\s*三、各情景求解状态",
        content,
        flags=re.DOTALL,
    )
    if summary_match is None:
        raise ValueError("敏感性 TXT 中缺少‘二、情景汇总’章节")

    records: list[SensitivityRecord] = []
    for line in summary_match.group(1).splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        scenario_id = int(fields[0])
        if len(fields) >= 4 and fields[3] == "未完成":
            records.append(
                SensitivityRecord(
                    scenario_id=scenario_id,
                    supply_deviation=_parse_percent(fields[1]),
                    loss_reduction=_parse_percent(fields[2]),
                    completed=False,
                    candidate_count=None,
                    active_provider_count=None,
                    total_cost=None,
                    transport_loss=None,
                    minimum_inventory_margin=None,
                    maximum_transfer_load=None,
                    primary_mip_gap=None,
                    secondary_mip_gap=None,
                    meets_gap=False,
                )
            )
            continue
        if len(fields) < 13 or fields[3] != "完成":
            raise ValueError(f"无法解析敏感性汇总行：{line}")
        records.append(
            SensitivityRecord(
                scenario_id=scenario_id,
                supply_deviation=_parse_percent(fields[1]),
                loss_reduction=_parse_percent(fields[2]),
                completed=True,
                candidate_count=int(fields[4]),
                active_provider_count=int(fields[5]),
                total_cost=float(fields[6]),
                transport_loss=float(fields[7]),
                minimum_inventory_margin=float(fields[8]),
                maximum_transfer_load=float(fields[9]),
                primary_mip_gap=_parse_percent(fields[10]),
                secondary_mip_gap=_parse_percent(fields[11]),
                meets_gap=fields[12] == "是",
            )
        )
    if not records:
        raise ValueError("敏感性 TXT 中没有可解析的情景")
    return sorted(records, key=lambda record: record.scenario_id)


def _deviation_label(value: float) -> str:
    return f"{value:+.0%}"


def _reduction_label(value: float) -> str:
    return f"{value:.0%}"


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> Path:
    """仅保存可无损缩放和编辑的 SVG 矢量图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.svg"
    figure.savefig(
        path,
        bbox_inches="tight",
        facecolor=COLOR_WHITE,
        edgecolor=COLOR_WHITE,
    )
    plt.close(figure)
    return path


def _build_matrix(
    records: Sequence[SensitivityRecord],
    supply_values: Sequence[float],
    reduction_values: Sequence[float],
    getter: Callable[[SensitivityRecord], float | int | None],
) -> np.ndarray:
    matrix = np.full((len(supply_values), len(reduction_values)), np.nan)
    supply_index = {value: index for index, value in enumerate(supply_values)}
    reduction_index = {
        value: index for index, value in enumerate(reduction_values)
    }
    for record in records:
        value = getter(record)
        if record.completed and value is not None:
            matrix[
                supply_index[record.supply_deviation],
                reduction_index[record.loss_reduction],
            ] = float(value)
    return matrix


def _record_map(
    records: Sequence[SensitivityRecord],
) -> dict[tuple[float, float], SensitivityRecord]:
    return {
        (record.supply_deviation, record.loss_reduction): record
        for record in records
    }


def _draw_heatmap_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    matrix: np.ndarray,
    records: dict[tuple[float, float], SensitivityRecord],
    supply_values: Sequence[float],
    reduction_values: Sequence[float],
    *,
    title: str,
    subtitle: str,
    subtitle_math: str | None,
    colormap: LinearSegmentedColormap,
    formatter: Callable[[float], str],
) -> None:
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, origin="lower", cmap=colormap, aspect="auto")
    image.cmap.set_bad(COLOR_WHITE)
    axis.set_title(title, loc="left", pad=24)
    _add_text_with_math_suffix(
        axis,
        subtitle,
        subtitle_math,
        position=(0, 1.02),
        location="lower left",
    )
    axis.set_xticks(
        range(len(reduction_values)),
        [_reduction_label(value) for value in reduction_values],
    )
    axis.set_yticks(
        range(len(supply_values)),
        [_deviation_label(value) for value in supply_values],
    )
    axis.set_xlabel("每周损耗下降比例", labelpad=10)
    axis.set_xticks(np.arange(-0.5, len(reduction_values), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(supply_values), 1), minor=True)
    axis.grid(which="minor", color=COLOR_WHITE, linewidth=3)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(length=0, pad=7)

    finite = matrix[np.isfinite(matrix)]
    midpoint = (
        (float(finite.min()) + float(finite.max())) / 2
        if finite.size
        else 0.0
    )
    for row, supply_value in enumerate(supply_values):
        for column, reduction_value in enumerate(reduction_values):
            record = records[supply_value, reduction_value]
            value = matrix[row, column]
            if not np.isfinite(value):
                axis.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        facecolor=COLOR_SOFT,
                        edgecolor=COLOR_RED,
                        linewidth=1.2,
                        hatch="///",
                    )
                )
                axis.text(
                    column,
                    row,
                    "限时\n未完成",
                    ha="center",
                    va="center",
                    color=COLOR_RED,
                    fontsize=9,
                    linespacing=1.4,
                )
                continue
            text_color = COLOR_WHITE if value > midpoint else COLOR_INK
            marker = " †" if not record.meets_gap else ""
            axis.text(
                column,
                row,
                formatter(float(value)) + marker,
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="semibold",
            )
            if not record.meets_gap:
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


def _add_text_with_math_suffix(
    axis: plt.Axes,
    text: str,
    math_suffix: str | None,
    *,
    position: tuple[float, float],
    location: str,
) -> None:
    """将中文文本和数学公式分开渲染，避免中文被 MathText 解析。"""
    children = [
        TextArea(
            text,
            textprops={"color": COLOR_MUTED, "fontsize": 9, "fontfamily": TEXT_FONT},
        )
    ]
    if math_suffix:
        children.append(
            TextArea(
                math_suffix,
                textprops={
                    "color": COLOR_MUTED,
                    "fontsize": 9,
                    "fontfamily": NUMBER_FONT,
                },
            )
        )
    packed_text = HPacker(children=children, align="center", pad=0, sep=3)
    axis.add_artist(
        AnchoredOffsetbox(
            loc=location,
            child=packed_text,
            pad=0,
            borderpad=0,
            frameon=False,
            bbox_to_anchor=position,
            bbox_transform=axis.transAxes,
        )
    )


def plot_sensitivity_heatmaps(
    records: Sequence[SensitivityRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制成本、运输损耗和实际使用供应商数量热力图。"""
    supply_values = sorted({record.supply_deviation for record in records})
    reduction_values = sorted({record.loss_reduction for record in records})
    records_by_key = _record_map(records)
    cost = _build_matrix(
        records,
        supply_values,
        reduction_values,
        lambda record: record.total_cost / 10000 if record.total_cost else None,
    )
    loss = _build_matrix(
        records,
        supply_values,
        reduction_values,
        lambda record: record.transport_loss,
    )
    provider_count = _build_matrix(
        records,
        supply_values,
        reduction_values,
        lambda record: record.active_provider_count,
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.7), facecolor=COLOR_WHITE)
    figure.suptitle(
        "供货偏差与损耗下降的二维敏感性",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.905,
        "横向比较运输改善速度，纵向比较供应商履约偏差",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    panels = (
        (
            cost,
            "优化经济成本",
            "成本单位",
            r"$\times 10^{4}$",
            LinearSegmentedColormap.from_list(
                "cost", ("#EDF4FF", "#BFD5FF", COLOR_BLUE)
            ),
            lambda value: f"{value:.2f}",
        ),
        (
            loss,
            "优化运输损耗",
            "未来 12 周累计损耗",
            None,
            LinearSegmentedColormap.from_list(
                "loss", ("#EAFBF7", "#9DE3D2", COLOR_GREEN)
            ),
            lambda value: f"{value:.0f}",
        ),
        (
            provider_count,
            "实际使用供应商",
            "未来 12 周至少供货一次",
            None,
            LinearSegmentedColormap.from_list(
                "providers", ("#FFF8E8", "#FBD58A", COLOR_ORANGE)
            ),
            lambda value: f"{value:.0f} 家",
        ),
    )
    for axis, panel in zip(axes, panels):
        _draw_heatmap_panel(
            figure,
            axis,
            panel[0],
            records_by_key,
            supply_values,
            reduction_values,
            title=panel[1],
            subtitle=panel[2],
            subtitle_math=panel[3],
            colormap=panel[4],
            formatter=panel[5],
        )
    axes[0].set_ylabel("供货偏差", labelpad=10)
    figure.text(
        0.055,
        0.035,
        "注：斜线单元格表示限时未完成；† 和红色虚线框表示已找到整数解，但未达到 5% MIP Gap。",
        color=COLOR_MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.78,
        bottom=0.18,
        wspace=0.30,
    )
    return _save_figure(figure, output_dir, "sensitivity_heatmaps")


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(COLOR_GRID)
    axis.spines["bottom"].set_color(COLOR_GRID)


def plot_sensitivity_trends(
    records: Sequence[SensitivityRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制经济成本和运输损耗随损耗下降率变化的趋势。"""
    supply_values = sorted({record.supply_deviation for record in records})
    reduction_values = sorted({record.loss_reduction for record in records})
    records_by_key = _record_map(records)
    colors = (COLOR_BLUE, COLOR_CYAN, COLOR_ORANGE)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=COLOR_WHITE)
    figure.suptitle(
        "不确定因素对第三题方案的交互影响",
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.885,
        "每个情景独立优化；曲线比较参数变化对经济性与运输损耗的影响",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    handles: list[Line2D] = []
    labels: list[str] = []
    for color, supply_value in zip(colors, supply_values):
        scenario_records = [
            records_by_key[supply_value, reduction]
            for reduction in reduction_values
        ]
        x = np.array(reduction_values) * 100
        cost = np.array(
            [
                record.total_cost / 10000
                if record.completed and record.total_cost is not None
                else np.nan
                for record in scenario_records
            ]
        )
        loss = np.array(
            [
                record.transport_loss
                if record.completed and record.transport_loss is not None
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
        handles.append(line)
        labels.append(f"供货偏差 {_deviation_label(supply_value)}")

        for index, record in enumerate(scenario_records):
            if not record.completed:
                for axis in axes:
                    axis.scatter(
                        x[index],
                        axis.get_ylim()[0],
                        marker="x",
                        color=COLOR_RED,
                        s=58,
                        linewidth=1.8,
                        clip_on=False,
                        zorder=6,
                    )
            elif not record.meets_gap:
                for axis, values in ((axes[0], cost), (axes[1], loss)):
                    axis.scatter(
                        x[index],
                        values[index],
                        s=84,
                        facecolor=COLOR_WHITE,
                        edgecolor=COLOR_RED,
                        linewidth=1.6,
                        zorder=6,
                    )

    axes[0].set_title("经济成本对情景组合呈非单调变化", loc="left", pad=16)
    axes[0].set_ylabel("优化经济成本")
    _add_text_with_math_suffix(
        axes[0],
        "单位：",
        r"$\times 10^{4}$",
        position=(1.0, 1.02),
        location="lower right",
    )
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    axes[1].set_title("损耗改善通常降低累计运输损耗", loc="left", pad=16)
    axes[1].set_ylabel("未来 12 周运输损耗")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    for axis in axes:
        axis.set_xlabel("每周损耗下降比例")
        axis.set_xticks(np.array(reduction_values) * 100)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
        axis.margins(x=0.08, y=0.18)
        _style_axis(axis)

    gap_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor=COLOR_WHITE,
        markeredgecolor=COLOR_RED,
        markeredgewidth=1.5,
        label="未达到 5% Gap",
    )
    unfinished_handle = Line2D(
        [],
        [],
        marker="x",
        linestyle="None",
        color=COLOR_RED,
        markeredgewidth=1.8,
        label="限时未完成",
    )
    figure.legend(
        handles + [gap_handle, unfinished_handle],
        labels + ["未达到 5% Gap", "限时未完成"],
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.015),
        columnspacing=1.5,
        handlelength=2.2,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.75,
        bottom=0.21,
        wspace=0.25,
    )
    return _save_figure(figure, output_dir, "sensitivity_trends")


def plot_sensitivity_risk(
    records: Sequence[SensitivityRecord],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制安全库存裕量与转运容量占用率风险图。"""
    completed = [record for record in records if record.completed]
    figure, axis = plt.subplots(figsize=(10.8, 6.4), facecolor=COLOR_WHITE)
    figure.suptitle(
        "情景可行性与运营缓冲空间",
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.09,
        0.905,
        "库存裕量越高、转运容量占用越低，方案的运营缓冲越充分",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    color_map = {-0.05: COLOR_BLUE, 0.0: COLOR_CYAN, 0.05: COLOR_ORANGE}
    annotation_offsets = {
        1: (8, 7),
        2: (-25, 11),
        3: (9, -7),
        4: (8, 7),
        5: (8, -9),
        6: (8, 11),
        7: (8, 8),
        8: (8, 8),
    }
    for record in completed:
        utilization = float(record.maximum_transfer_load or 0.0) / 6200.0
        margin = float(record.minimum_inventory_margin or 0.0)
        size = 75 + 13 * float(record.active_provider_count or 0)
        axis.scatter(
            utilization,
            margin,
            s=size,
            color=color_map[record.supply_deviation],
            alpha=0.92,
            edgecolor=COLOR_WHITE if record.meets_gap else COLOR_RED,
            linewidth=1.8,
            zorder=3,
        )
        axis.annotate(
            f"S{record.scenario_id}",
            (utilization, margin),
            xytext=annotation_offsets.get(record.scenario_id, (8, 7)),
            textcoords="offset points",
            color=COLOR_INK,
            fontsize=10,
            fontweight="semibold",
        )

    axis.axvline(1.0, color=COLOR_RED, linestyle=(0, (4, 4)), linewidth=1.3)
    axis.text(
        0.9995,
        0.98,
        "转运容量上限",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="top",
        color=COLOR_RED,
        fontsize=9,
    )
    axis.axhline(0.0, color=COLOR_RED, linestyle=(0, (4, 4)), linewidth=1.3)
    axis.set_xlabel("最大转运容量占用率")
    axis.set_ylabel("最小安全库存裕量")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    axis.set_xlim(0.948, 1.005)
    axis.margins(y=0.16)
    _style_axis(axis, grid_axis="both")

    deviation_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=color_map[value],
            markeredgecolor=COLOR_WHITE,
            markersize=8,
            label=f"供货偏差 {_deviation_label(value)}",
        )
        for value in sorted(color_map)
    ]
    gap_handle = Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markerfacecolor=COLOR_WHITE,
        markeredgecolor=COLOR_RED,
        markeredgewidth=1.5,
        markersize=8,
        label="未达到 5% Gap",
    )
    axis.legend(
        handles=deviation_handles + [gap_handle],
        ncol=4,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.11),
        columnspacing=1.5,
    )
    figure.text(
        0.09,
        0.035,
        "注：圆点面积表示 12 周实际使用的供应商数量；S1—S8 为已完成情景，S9 限时未完成。",
        color=COLOR_MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.15)
    return _save_figure(figure, output_dir, "sensitivity_risk")


def run_visualization(
    report_path: Path = DEFAULT_SENSITIVITY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """读取第三题敏感性 TXT，并生成全部 SVG 论文图。"""
    _configure_style()
    records = load_sensitivity_data(report_path)
    return [
        plot_sensitivity_heatmaps(records, output_dir),
        plot_sensitivity_trends(records, output_dir),
        plot_sensitivity_risk(records, output_dir),
    ]


if __name__ == "__main__":
    for generated_path in run_visualization():
        print(generated_path)
