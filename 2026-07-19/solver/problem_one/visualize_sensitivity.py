"""Visualize the indicator-removal sensitivity analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from .critic import INDICATOR_NAMES, STD_MATRIX
from .sensitivity import (
    TOP_N,
    ScenarioResult,
    analyze_indicator_removal,
)
from ..utils.read import std_received

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "one" / "figures"
SHORT_INDICATOR_NAMES = (
    "总供货量(Q)",
    "供货频率(F)",
    "供货稳定性(V)",
    "订单匹配度(M)",
    "等效贡献量(P)",
)
INDICATOR_CODES = ("Q", "F", "V", "M", "P")
COLOR_BLUE = "#2563EB"
COLOR_CYAN = "#06B6D4"
COLOR_GREEN = "#10B981"
COLOR_ORANGE = "#F97316"
COLOR_RED = "#EF4444"
COLOR_INK = "#172033"
COLOR_MUTED = "#64748B"
COLOR_GRID = "#DCE5EF"
COLOR_CANVAS = "#FFFFFF"
TEXT_FONT = "SimSun"
NUMBER_FONT = "Latin Modern Math"


def _configure_font() -> None:
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
            "font.family": TEXT_FONT,
            "font.serif": [TEXT_FONT],
            "mathtext.fontset": "custom",
            "mathtext.rm": NUMBER_FONT,
            "mathtext.it": NUMBER_FONT,
            "mathtext.bf": NUMBER_FONT,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
            "figure.facecolor": COLOR_CANVAS,
            "axes.facecolor": COLOR_CANVAS,
            "axes.edgecolor": COLOR_GRID,
            "axes.labelcolor": COLOR_MUTED,
            "axes.titlecolor": COLOR_INK,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "font.size": 11,
            "text.color": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "grid.color": COLOR_GRID,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "savefig.facecolor": COLOR_CANVAS,
        }
    )


def _set_tick_fonts(
    axis: plt.Axes,
    *,
    x_numbers: bool = False,
    y_numbers: bool = False,
) -> None:
    if x_numbers:
        for label in axis.get_xticklabels():
            label.set_fontfamily(NUMBER_FONT)
    if y_numbers:
        for label in axis.get_yticklabels():
            label.set_fontfamily(NUMBER_FONT)


def _scenario_label(scenario: ScenarioResult) -> str:
    removed_names = [INDICATOR_CODES[index] for index in scenario.removed_indices]
    return "删除$\\mathrm{" + "+".join(removed_names) + "}$"


def _provider_label(provider_id: int) -> str:
    return f"$\\mathrm{{S{provider_id:03d}}}$"


def _latin_text(value: str) -> str:
    return f"$\\mathrm{{{value}}}$"


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(
        output_path,
        bbox_inches="tight",
        facecolor=COLOR_CANVAS,
    )
    plt.close(figure)


def plot_metric_comparison(
    scenarios: list[ScenarioResult],
    output_dir: Path,
) -> Path:
    labels = [_scenario_label(scenario) for scenario in scenarios]
    y = np.arange(len(scenarios))
    retention = np.array([scenario.retention_rate for scenario in scenarios])
    jaccard = np.array([scenario.jaccard for scenario in scenarios])
    spearman = np.array([scenario.spearman for scenario in scenarios])

    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.hlines(
        y,
        np.minimum.reduce([retention, jaccard, spearman]),
        np.maximum.reduce([retention, jaccard, spearman]),
        color=COLOR_GRID,
        linewidth=2.5,
        zorder=1,
    )
    axis.scatter(
        retention, y, s=70, color=COLOR_BLUE, label="前$\\mathrm{30}$名保留率", zorder=3
    )
    axis.scatter(
        jaccard, y, s=70, color=COLOR_GREEN, label="$\\mathrm{Jaccard}$系数", zorder=3
    )
    axis.scatter(
        spearman,
        y,
        s=70,
        color=COLOR_ORANGE,
        label="$\\mathrm{Spearman}$系数",
        zorder=3,
    )
    axis.axvline(0.9, color=COLOR_RED, linestyle=(0, (4, 4)), linewidth=1.4)
    axis.text(0.902, -0.7, "$\\mathrm{90\\%}$参考线", color=COLOR_RED, fontsize=9)
    worst_index = int(np.argmin(spearman))
    axis.annotate(
        f"{spearman[worst_index]:.3f}",
        (spearman[worst_index], worst_index),
        xytext=(-35, 0),
        textcoords="offset points",
        va="center",
        color=COLOR_ORANGE,
        fontweight="bold",
        fontfamily=NUMBER_FONT,
    )
    axis.set_xlim(0.45, 1.025)
    axis.set_xlabel("稳定性指标值")
    axis.set_title("核心供应商集合对指标删减保持稳定", loc="left", pad=20)
    axis.text(
        0,
        1.015,
        "点越靠右越稳健；横线连接同一删减方案的三项评价",
        transform=axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.02))
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()

    _set_tick_fonts(axis, x_numbers=True)
    path = output_dir / "sensitivity_metric_comparison.svg"
    _save_figure(figure, path)
    return path


def plot_stability_scatter(
    scenarios: list[ScenarioResult],
    output_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 6.5))
    one_removed = [
        scenario for scenario in scenarios if len(scenario.removed_indices) == 1
    ]
    two_removed = [
        scenario for scenario in scenarios if len(scenario.removed_indices) == 2
    ]

    for group, label, color, marker in (
        (one_removed, "删除$\\mathrm{1}$个指标", COLOR_BLUE, "o"),
        (two_removed, "删除$\\mathrm{2}$个指标", COLOR_ORANGE, "s"),
    ):
        axis.scatter(
            [scenario.retention_rate for scenario in group],
            [scenario.spearman for scenario in group],
            label=label,
            color=color,
            marker=marker,
            s=85,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.2,
        )

    worst = min(scenarios, key=lambda scenario: scenario.spearman)
    axis.annotate(
        _scenario_label(worst),
        xy=(worst.retention_rate, worst.spearman),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=9,
        color=COLOR_RED,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLOR_RED},
    )
    axis.axvline(0.9, color=COLOR_RED, linestyle=(0, (4, 4)), linewidth=1.2)
    axis.axhline(0.95, color=COLOR_RED, linestyle=(0, (4, 4)), linewidth=1.2)
    axis.set_xlim(0.87, 1.01)
    axis.set_ylim(min(0.45, min(s.spearman for s in scenarios) - 0.03), 1.01)
    axis.set_xlabel("前$\\mathrm{30}$名保留率")
    axis.set_ylabel("$\\mathrm{Spearman}$等级相关系数")
    axis.set_title("核心集合稳定，不代表完整排序同样稳定", loc="left", pad=20)
    axis.text(
        0,
        1.015,
        "右上区域表示前$\\mathrm{30}$名与完整排名同时稳健",
        transform=axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid()
    axis.legend(loc="lower right")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()

    _set_tick_fonts(axis, x_numbers=True, y_numbers=True)
    path = output_dir / "sensitivity_stability_scatter.svg"
    _save_figure(figure, path)
    return path


def plot_selection_frequency(
    scenarios: list[ScenarioResult],
    output_dir: Path,
    top_n: int = TOP_N,
) -> Path:
    """Plot selection status for every supplier and export the full table."""
    selection_counts = np.zeros(len(STD_MATRIX), dtype=int)
    selection_matrix = np.zeros((len(STD_MATRIX), len(scenarios)), dtype=int)
    for scenario_index, scenario in enumerate(scenarios):
        for row_index in scenario.topsis.ranking[:top_n]:
            selection_counts[row_index] += 1
            selection_matrix[row_index, scenario_index] = 1

    all_order = np.argsort(-selection_counts)
    selected_order = np.array(
        [index for index in all_order if selection_counts[index] > 0]
    )
    ordered_matrix = selection_matrix[selected_order]
    ordered_counts = selection_counts[selected_order]
    ordered_provider_ids = [std_received[index].provider_id for index in selected_order]
    row_count = len(ordered_provider_ids)
    never_selected_count = len(STD_MATRIX) - row_count

    scenario_labels = [
        "删除" + "+".join(INDICATOR_CODES[index] for index in scenario.removed_indices)
        for scenario in scenarios
    ]
    csv_path = output_dir / "sensitivity_selection_frequency.csv"
    csv_lines = [",".join(["供应商", *scenario_labels, "入选次数", "入选比例"])]
    for row_index in all_order:
        provider_id = std_received[row_index].provider_id
        count = int(selection_counts[row_index])
        csv_lines.append(
            ",".join(
                [
                    f"S{provider_id:03d}",
                    *(str(value) for value in selection_matrix[row_index]),
                    str(count),
                    f"{count / len(scenarios):.6f}",
                ]
            )
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8-sig")

    figure = plt.figure(figsize=(14, 10))
    grid = GridSpec(1, 2, figure=figure, width_ratios=(13, 2), wspace=0.08)
    axis = figure.add_subplot(grid[0, 0])
    frequency_axis = figure.add_subplot(grid[0, 1], sharey=axis)
    status_map = ListedColormap(["#EEF3F8", COLOR_BLUE])
    axis.imshow(
        ordered_matrix,
        aspect="auto",
        interpolation="none",
        cmap=status_map,
        vmin=0,
        vmax=1,
    )
    axis.set_xticks(
        np.arange(len(scenario_labels)),
        [_latin_text(label.replace("删除", "")) for label in scenario_labels],
        rotation=55,
        ha="right",
    )
    tick_step = 1
    tick_positions = np.arange(0, row_count, tick_step)
    axis.set_yticks(
        tick_positions,
        [
            _provider_label(ordered_provider_ids[position])
            for position in tick_positions
        ],
    )
    axis.set_xlabel("指标删减方案")
    axis.set_ylabel("供应商（按入选次数降序）")
    axis.set_title(
        f"{row_count} 家曾入选供应商的方案矩阵",
        loc="left",
        pad=20,
    )
    axis.text(
        0,
        1.015,
        f"蓝色 = 进入该方案的 TOPSIS 前$\\mathrm{{30}}$名；"
        f"其余$\\mathrm{{{never_selected_count}}}$家在所有方案中均未入选",
        transform=axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    axis.grid(axis="x", color="white", linewidth=0.7)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    _set_tick_fonts(axis, x_numbers=True, y_numbers=True)

    frequencies = ordered_counts / len(scenarios)
    frequency_axis.barh(
        np.arange(row_count),
        frequencies,
        color=np.where(frequencies >= 0.9, COLOR_GREEN, COLOR_ORANGE),
        height=0.8,
    )
    frequency_axis.set_xlim(0, 1.05)
    frequency_axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    frequency_axis.set_xlabel("入选比例")
    frequency_axis.set_title("频率", loc="left", pad=20)
    frequency_axis.tick_params(axis="y", left=False, labelleft=False)
    frequency_axis.grid(axis="x")
    for spine in frequency_axis.spines.values():
        spine.set_visible(False)
    _set_tick_fonts(frequency_axis, x_numbers=True)
    figure.suptitle(
        "供应商在敏感性方案中的入选情况",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=COLOR_INK,
    )
    figure.subplots_adjust(top=0.91, bottom=0.12, left=0.07, right=0.96)

    path = output_dir / "sensitivity_selection_frequency.svg"
    _save_figure(figure, path)
    return path


def plot_weight_heatmap(
    scenarios: list[ScenarioResult],
    output_dir: Path,
) -> Path:
    values = np.full((len(scenarios), len(INDICATOR_NAMES)), np.nan)
    labels = []
    for row, scenario in enumerate(scenarios):
        labels.append(_scenario_label(scenario))
        for column, indicator_index in enumerate(scenario.kept_indices):
            values[row, indicator_index] = scenario.weights[column]

    figure, axis = plt.subplots(figsize=(10, 8.5))
    masked_values = np.ma.masked_invalid(values)
    color_map = LinearSegmentedColormap.from_list(
        "modern_blue",
        ["#E8F2FF", "#67E8F9", COLOR_BLUE, "#172E7A"],
    )
    color_map.set_bad("#EDF2F7")
    image = axis.imshow(masked_values, aspect="auto", cmap=color_map, vmin=0, vmax=0.75)
    axis.set_xticks(
        np.arange(len(INDICATOR_CODES)),
        [_latin_text(code) for code in INDICATOR_CODES],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("指标")
    axis.set_ylabel("删除指标组合")
    axis.set_title("指标删减会重新分配 $\\mathrm{CRITIC}$ 权重", loc="left", pad=20)
    axis.text(
        0,
        1.015,
        "灰色单元格表示该指标已从当前方案中删除",
        transform=axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    color_bar = figure.colorbar(image, ax=axis, label="权重", fraction=0.035, pad=0.03)
    color_bar.outline.set_visible(False)
    _set_tick_fonts(color_bar.ax, y_numbers=True)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isnan(values[row, column]):
                axis.text(
                    column,
                    row,
                    "×",
                    ha="center",
                    va="center",
                    color="#94A3B8",
                    fontsize=13,
                )
            else:
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if values[row, column] >= 0.42 else COLOR_INK,
                    fontweight="bold" if values[row, column] >= 0.42 else "normal",
                    fontfamily=NUMBER_FONT,
                )
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.tight_layout()

    path = output_dir / "sensitivity_weight_heatmap.svg"
    _save_figure(figure, path)
    return path


def _dashboard_panel(axis: plt.Axes, title: str, subtitle: str) -> None:
    axis.set_facecolor("white")
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=13)
    axis.text(
        0,
        1.01,
        subtitle,
        transform=axis.transAxes,
        color=COLOR_MUTED,
        fontsize=8.5,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_dashboard(
    scenarios: list[ScenarioResult],
    output_dir: Path,
    top_n: int = TOP_N,
) -> Path:
    """Create a compact presentation-style overview of all results."""
    retention = np.array([scenario.retention_rate for scenario in scenarios])
    jaccard = np.array([scenario.jaccard for scenario in scenarios])
    spearman = np.array([scenario.spearman for scenario in scenarios])
    labels = [_scenario_label(scenario) for scenario in scenarios]

    selection_counts = np.zeros(len(STD_MATRIX), dtype=int)
    for scenario in scenarios:
        for row_index in scenario.topsis.ranking[:top_n]:
            selection_counts[row_index] += 1
    selected_order = np.argsort(-selection_counts)
    selected_order = [index for index in selected_order if selection_counts[index] > 0][
        :10
    ]

    weight_values = np.full((len(scenarios), len(INDICATOR_NAMES)), np.nan)
    for row, scenario in enumerate(scenarios):
        for column, indicator_index in enumerate(scenario.kept_indices):
            weight_values[row, indicator_index] = scenario.weights[column]

    figure = plt.figure(figsize=(16, 10), facecolor=COLOR_CANVAS)
    grid = GridSpec(
        2,
        2,
        figure=figure,
        left=0.06,
        right=0.97,
        top=0.66,
        bottom=0.08,
        hspace=0.48,
        wspace=0.22,
    )

    figure.text(
        0.06,
        0.955,
        "指标删减敏感性",
        fontsize=28,
        fontweight="bold",
        color=COLOR_INK,
    )
    figure.text(
        0.06,
        0.918,
        "CRITIC–TOPSIS 供应商评价稳健性总览  /  318 家供应商 · 15 种指标组合",
        fontsize=11,
        color=COLOR_MUTED,
    )
    figure.text(
        0.97,
        0.955,
        "SENSITIVITY / 01",
        fontsize=10,
        color=COLOR_BLUE,
        fontweight="bold",
        ha="right",
        fontfamily=NUMBER_FONT,
    )

    summary = [
        ("15", "删减方案", COLOR_BLUE),
        (f"{retention.mean():.1%}", "平均前30保留率", COLOR_GREEN),
        (f"{retention.min():.1%}", "最低前30保留率", COLOR_ORANGE),
        (f"{spearman.min():.3f}", "最低整体相关系数", COLOR_RED),
    ]
    card_x = np.linspace(0.06, 0.78, len(summary))
    for x, (value, label, accent) in zip(card_x, summary):
        figure.text(
            x,
            0.815,
            value,
            fontsize=23,
            fontweight="bold",
            color=accent,
            fontfamily=NUMBER_FONT,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#E2E8F0",
                "linewidth": 0.8,
            },
        )
        figure.text(x, 0.775, label, fontsize=9, color=COLOR_MUTED)

    axis = figure.add_subplot(grid[0, 0])
    _dashboard_panel(axis, "核心集合的稳定性", "每一行代表一个删减方案，越靠右越稳定")
    y = np.arange(len(scenarios))
    axis.hlines(
        y,
        np.minimum.reduce([retention, jaccard, spearman]),
        np.maximum.reduce([retention, jaccard, spearman]),
        color=COLOR_GRID,
        linewidth=2,
    )
    axis.scatter(retention, y, color=COLOR_BLUE, s=32, label="保留率", zorder=3)
    axis.scatter(jaccard, y, color=COLOR_GREEN, s=32, label="Jaccard", zorder=3)
    axis.scatter(spearman, y, color=COLOR_ORANGE, s=32, label="Spearman", zorder=3)
    axis.axvline(0.9, color=COLOR_RED, linestyle=(0, (3, 4)), linewidth=1)
    axis.set_xlim(0.45, 1.025)
    axis.set_yticks(y, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("指标值", fontsize=9)
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower right", ncol=3, fontsize=8)
    _set_tick_fonts(axis, x_numbers=True)

    axis = figure.add_subplot(grid[0, 1])
    _dashboard_panel(axis, "集合稳定性 vs. 整体排序", "右上区域表示两种稳定性同时较高")
    one_removed = [
        scenario for scenario in scenarios if len(scenario.removed_indices) == 1
    ]
    two_removed = [
        scenario for scenario in scenarios if len(scenario.removed_indices) == 2
    ]
    for group, color, marker, label in (
        (one_removed, COLOR_BLUE, "o", "删1个指标"),
        (two_removed, COLOR_ORANGE, "s", "删2个指标"),
    ):
        axis.scatter(
            [scenario.retention_rate for scenario in group],
            [scenario.spearman for scenario in group],
            color=color,
            marker=marker,
            s=55,
            edgecolors="white",
            linewidths=1,
            label=label,
        )
    worst = min(scenarios, key=lambda scenario: scenario.spearman)
    axis.annotate(
        _scenario_label(worst),
        (worst.retention_rate, worst.spearman),
        xytext=(8, -16),
        textcoords="offset points",
        color=COLOR_RED,
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": COLOR_RED},
    )
    axis.axvline(0.9, color=COLOR_RED, linestyle=(0, (3, 4)), linewidth=1)
    axis.axhline(0.95, color=COLOR_RED, linestyle=(0, (3, 4)), linewidth=1)
    axis.set_xlim(0.88, 1.01)
    axis.set_ylim(0.47, 1.01)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("前30名保留率", fontsize=9)
    axis.set_ylabel("Spearman", fontsize=9)
    axis.grid()
    axis.legend(loc="lower right", fontsize=8)
    _set_tick_fonts(axis, x_numbers=True, y_numbers=True)

    axis = figure.add_subplot(grid[1, 0])
    _dashboard_panel(axis, "稳定入选的核心供应商", "展示入选频率最高的 10 家供应商")
    order = list(reversed(selected_order))
    provider_labels = [f"S{std_received[index].provider_id:03d}" for index in order]
    frequencies = [selection_counts[index] / len(scenarios) for index in order]
    axis.barh(provider_labels, frequencies, color=COLOR_BLUE, height=0.56)
    axis.set_xlim(0, 1.05)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("进入前30名的方案比例", fontsize=9)
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    for row, frequency in enumerate(frequencies):
        axis.text(
            frequency + 0.015,
            row,
            f"{frequency:.0%}",
            va="center",
            fontsize=8,
            fontfamily=NUMBER_FONT,
        )
    _set_tick_fonts(axis, x_numbers=True, y_numbers=True)

    axis = figure.add_subplot(grid[1, 1])
    _dashboard_panel(axis, "CRITIC 权重重新分配", "灰色单元格表示指标在该方案中被删除")
    color_map = LinearSegmentedColormap.from_list(
        "dashboard_blue",
        ["#E8F2FF", "#67E8F9", COLOR_BLUE, "#172E7A"],
    )
    color_map.set_bad("#EDF2F7")
    image = axis.imshow(
        np.ma.masked_invalid(weight_values),
        aspect="auto",
        cmap=color_map,
        vmin=0,
        vmax=0.75,
    )
    axis.set_xticks(np.arange(len(SHORT_INDICATOR_NAMES)), INDICATOR_CODES, fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_xlabel(
        "指标代码：Q 总供货量 / F 供货频率 / V 稳定性 / M 匹配度 / P 等效贡献",
        fontsize=8,
    )
    for row in range(weight_values.shape[0]):
        for column in range(weight_values.shape[1]):
            value = weight_values[row, column]
            if np.isnan(value):
                axis.text(
                    column,
                    row,
                    "×",
                    ha="center",
                    va="center",
                    color="#94A3B8",
                    fontsize=9,
                )
            else:
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value >= 0.42 else COLOR_INK,
                    fontfamily=NUMBER_FONT,
                )
    axis.tick_params(length=0)
    color_bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02, label="权重")
    color_bar.outline.set_visible(False)
    _set_tick_fonts(color_bar.ax, y_numbers=True)
    for label in axis.get_xticklabels():
        label.set_fontfamily(NUMBER_FONT)

    path = output_dir / "sensitivity_dashboard.svg"
    _save_figure(figure, path)
    return path


def run_visualization(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Run sensitivity analysis and save all figures."""
    _configure_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, scenarios = analyze_indicator_removal()
    return [
        plot_metric_comparison(scenarios, output_dir),
        plot_stability_scatter(scenarios, output_dir),
        plot_selection_frequency(scenarios, output_dir),
        plot_weight_heatmap(scenarios, output_dir),
    ]


if __name__ == "__main__":
    for figure_path in run_visualization():
        print(f"已生成：{figure_path}")
