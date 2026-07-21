"""可视化第三题基本情景的订货与转运求解结果。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

from ..problem_two.step1 import MATERIAL_DEMANDS, TRANSFER_CAPACITY
from ..problem_two.step2 import SAFETY_STOCKS, WEEK_COUNT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "basic_result.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "three" / "figures"

COLOR_WHITE = "#FFFFFF"
COLOR_INK = "#172033"
COLOR_MUTED = "#64748B"
COLOR_GRID = "#DCE5EF"
COLOR_SOFT = "#F5F8FC"
COLOR_BLUE = "#2563EB"
COLOR_CYAN = "#0891B2"
COLOR_GREEN = "#0F9F7A"
COLOR_ORANGE = "#F59E0B"
COLOR_RED = "#E5484D"
TEXT_FONT = "SimSun"
NUMBER_FONT = "Latin Modern Math"

MATERIAL_COLORS = {"A": COLOR_BLUE, "B": COLOR_GREEN, "C": COLOR_ORANGE}
TRANSFER_COLORS = {
    "T1": "#2563EB",
    "T2": "#0891B2",
    "T3": "#F59E0B",
    "T4": "#7C3AED",
    "T5": "#E5484D",
    "T6": "#0F9F7A",
    "T7": "#D97706",
    "T8": "#475569",
}


@dataclass(frozen=True)
class PlanEntry:
    """某周某家供应商的订货与转运安排。"""

    week: int
    provider: str
    material: str
    nominal_supply: float
    order_quantity: float
    capacity_limit: float
    transfer: str


@dataclass(frozen=True)
class WeeklyResult:
    """基本情景下某一周的入库、库存和运营指标。"""

    week: int
    receipts: dict[str, float]
    inventories: dict[str, float]
    cost: float
    loss: float
    maximum_transfer_load: float


@dataclass(frozen=True)
class BasicResult:
    """从第三题基本情景 TXT 中提取的绘图数据。"""

    supply_deviation: float
    loss_reduction: float
    primary_gap: float
    secondary_gap: float
    candidate_count: int
    active_providers: tuple[str, ...]
    total_cost: float
    total_loss: float
    minimum_inventory_margin: float
    maximum_transfer_load: float
    plan: tuple[PlanEntry, ...]
    weeks: tuple[WeeklyResult, ...]
    actual_supply: dict[tuple[int, str], float]


def _configure_style() -> None:
    """设置纯白背景及宋体、Latin Modern Math 字体回退。"""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    missing_fonts = {
        name for name in (TEXT_FONT, NUMBER_FONT) if name not in available_fonts
    }
    if missing_fonts:
        raise RuntimeError(f"缺少绘图字体：{', '.join(sorted(missing_fonts))}")

    matplotlib.rcParams.update(
        {
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


def _required_match(pattern: str, content: str, label: str) -> re.Match[str]:
    match = re.search(pattern, content, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"基本结果 TXT 中缺少{label}")
    return match


def _parse_percent(value: str) -> float:
    return float(value.removesuffix("%")) / 100.0


def load_basic_result(report_path: Path = DEFAULT_REPORT_PATH) -> BasicResult:
    """读取基本情景 TXT，并校验汇总值与逐周值一致。"""
    if not report_path.exists():
        raise FileNotFoundError(
            f"未找到第三题基本结果：{report_path}。请先运行 solver.py。"
        )
    content = report_path.read_text(encoding="utf-8-sig")

    supply_deviation = _parse_percent(
        _required_match(r"^供货偏差情景：([^\s]+)$", content, "供货偏差").group(1)
    )
    loss_reduction = _parse_percent(
        _required_match(r"^每周损耗下降情景：([^\s]+)$", content, "损耗下降比例").group(1)
    )
    primary_gap = _parse_percent(
        _required_match(r"^第一阶段 MIP Gap：([^\s]+)$", content, "第一阶段 Gap").group(1)
    )
    secondary_gap = _parse_percent(
        _required_match(r"^第二阶段 MIP Gap：([^\s]+)$", content, "第二阶段 Gap").group(1)
    )
    candidate_count = int(
        _required_match(r"^候选供应商总数：(\d+)$", content, "候选供应商数").group(1)
    )
    active_match = _required_match(
        r"^12 周中至少供货一次的供应商：(\d+) 家，(.+)$",
        content,
        "实际使用供应商",
    )
    active_count = int(active_match.group(1))
    active_providers = tuple(
        provider.strip() for provider in active_match.group(2).split(",")
    )
    if active_count != len(active_providers):
        raise ValueError("实际使用供应商数与名单不一致")

    summary_match = _required_match(
        r"^\s*1\s+[-+\d.]+%\s+[\d.]+%\s+"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        content,
        "12 周汇总结果",
    )
    total_cost, total_loss, minimum_margin, maximum_load = (
        float(summary_match.group(index)) for index in range(1, 5)
    )

    plan_section = _required_match(
        r"四、未来 12 周共同订货与转运计划\s*([\s\S]*?)\s*"
        r"五、基本情景 12 周汇总",
        content,
        "订货与转运计划",
    ).group(1)
    plan: list[PlanEntry] = []
    for line in plan_section.splitlines():
        fields = line.split()
        if len(fields) != 7 or not fields[0].isdigit() or not fields[1].startswith("S"):
            continue
        plan.append(
            PlanEntry(
                week=int(fields[0]),
                provider=fields[1],
                material=fields[2],
                nominal_supply=float(fields[3]),
                order_quantity=float(fields[4]),
                capacity_limit=float(fields[5]),
                transfer=fields[6],
            )
        )

    weekly_section = _required_match(
        r"六、基本情景逐周库存、损耗和运输负载\s*([\s\S]*?)\s*"
        r"七、基本情景各供应商实际供货量",
        content,
        "逐周运营结果",
    ).group(1)
    weeks: list[WeeklyResult] = []
    for line in weekly_section.splitlines():
        fields = line.split()
        if len(fields) != 10 or not fields[0].isdigit():
            continue
        values = [float(value) for value in fields[1:]]
        weeks.append(
            WeeklyResult(
                week=int(fields[0]),
                receipts=dict(zip(("A", "B", "C"), values[0:3])),
                inventories=dict(zip(("A", "B", "C"), values[3:6])),
                cost=values[6],
                loss=values[7],
                maximum_transfer_load=values[8],
            )
        )

    actual_section = _required_match(
        r"七、基本情景各供应商实际供货量\s*([\s\S]*?)\s*"
        r"八、约束核验",
        content,
        "各供应商实际供货量",
    ).group(1)
    actual_supply: dict[tuple[int, str], float] = {}
    for match in re.finditer(r"^第\s+(\d+)\s+周：(.+)$", actual_section, re.MULTILINE):
        week = int(match.group(1))
        for assignment in match.group(2).split(","):
            provider, value = assignment.strip().split("=")
            actual_supply[week, provider] = float(value)

    if len(weeks) != WEEK_COUNT or {week.week for week in weeks} != set(
        range(1, WEEK_COUNT + 1)
    ):
        raise ValueError(f"逐周结果应完整包含 {WEEK_COUNT} 周")
    if not plan:
        raise ValueError("订货与转运计划为空")
    if {entry.provider for entry in plan} != set(active_providers):
        raise ValueError("计划中的供应商与实际使用名单不一致")
    if any((entry.week, entry.provider) not in actual_supply for entry in plan):
        raise ValueError("部分订货计划缺少实际供货量")

    calculated_cost = sum(week.cost for week in weeks)
    calculated_loss = sum(week.loss for week in weeks)
    calculated_margin = min(
        week.inventories[material] - SAFETY_STOCKS[material]
        for week in weeks
        for material in MATERIAL_DEMANDS
    )
    calculated_load = max(week.maximum_transfer_load for week in weeks)
    checks = (
        (calculated_cost, total_cost, "总成本"),
        (calculated_loss, total_loss, "总运输损耗"),
        (calculated_margin, minimum_margin, "最小库存裕量"),
        (calculated_load, maximum_load, "最大转运负载"),
    )
    for calculated, reported, label in checks:
        if not np.isclose(calculated, reported, atol=0.01):
            raise ValueError(
                f"{label}汇总值与逐周计算不一致：{reported} != {calculated}"
            )

    return BasicResult(
        supply_deviation=supply_deviation,
        loss_reduction=loss_reduction,
        primary_gap=primary_gap,
        secondary_gap=secondary_gap,
        candidate_count=candidate_count,
        active_providers=active_providers,
        total_cost=total_cost,
        total_loss=total_loss,
        minimum_inventory_margin=minimum_margin,
        maximum_transfer_load=maximum_load,
        plan=tuple(plan),
        weeks=tuple(sorted(weeks, key=lambda week: week.week)),
        actual_supply=actual_supply,
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> Path:
    """仅保存可无损缩放和编辑的 SVG 矢量图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}.svg"
    figure.savefig(
        output_path,
        bbox_inches="tight",
        facecolor=COLOR_WHITE,
        edgecolor=COLOR_WHITE,
    )
    plt.close(figure)
    return output_path


def _style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(axis=grid_axis)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color(COLOR_GRID)
    axis.spines["bottom"].set_color(COLOR_GRID)


def _add_text_with_math_suffix(
    axis: plt.Axes,
    text: str,
    math_suffix: str,
    *,
    position: tuple[float, float],
    location: str = "lower right",
) -> None:
    """将中文单位和数学公式分开渲染。"""
    packed = HPacker(
        children=[
            TextArea(
                text,
                textprops={"color": COLOR_MUTED, "fontsize": 9, "fontfamily": TEXT_FONT},
            ),
            TextArea(
                math_suffix,
                textprops={
                    "color": COLOR_MUTED,
                    "fontsize": 9,
                    "fontfamily": NUMBER_FONT,
                },
            ),
        ],
        align="center",
        pad=0,
        sep=3,
    )
    axis.add_artist(
        AnchoredOffsetbox(
            loc=location,
            child=packed,
            pad=0,
            borderpad=0,
            frameon=False,
            bbox_to_anchor=position,
            bbox_transform=axis.transAxes,
        )
    )


def plot_material_inventory(
    result: BasicResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制三类原材料的实际入库量和期末库存。"""
    week_numbers = np.array([week.week for week in result.weeks])
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14.8, 8.3),
        facecolor=COLOR_WHITE,
        gridspec_kw={"height_ratios": (1.0, 1.08), "hspace": 0.40},
    )
    figure.suptitle(
        "未来 12 周原材料入库与库存路径",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.075,
        0.925,
        "三类材料均满足生产需求，期末库存全程不低于三周安全库存线",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    bar_width = 0.24
    for index, material in enumerate(("A", "B", "C")):
        receipts = [week.receipts[material] for week in result.weeks]
        axes[0].bar(
            week_numbers + (index - 1) * bar_width,
            receipts,
            width=bar_width,
            color=MATERIAL_COLORS[material],
            alpha=0.90,
            label=f"{material} 类实际入库",
            zorder=3,
        )
    axes[0].set_title("周度实际入库量", loc="left", pad=14)
    axes[0].set_ylabel("入库量")
    axes[0].set_xticks(week_numbers)
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    axes[0].legend(
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        borderaxespad=0,
    )
    _style_axis(axes[0])

    for material in ("A", "B", "C"):
        inventories = np.array(
            [week.inventories[material] for week in result.weeks]
        )
        safety = SAFETY_STOCKS[material]
        axes[1].plot(
            week_numbers,
            inventories,
            color=MATERIAL_COLORS[material],
            linewidth=2.6,
            marker="o",
            markersize=6,
            markeredgecolor=COLOR_WHITE,
            markeredgewidth=1.1,
            label=f"{material} 类期末库存",
            zorder=4,
        )
        axes[1].plot(
            week_numbers,
            np.full(WEEK_COUNT, safety),
            color=MATERIAL_COLORS[material],
            linewidth=1.4,
            linestyle=(0, (5, 4)),
            alpha=0.72,
            zorder=2,
        )
        axes[1].fill_between(
            week_numbers,
            safety,
            inventories,
            color=MATERIAL_COLORS[material],
            alpha=0.055,
            zorder=1,
        )
    axes[1].set_title("期末库存与安全库存线", loc="left", pad=14)
    axes[1].set_xlabel("周次")
    axes[1].set_ylabel("库存量")
    axes[1].set_xticks(week_numbers)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    inventory_handles = [
        Line2D([], [], color=MATERIAL_COLORS[material], linewidth=2.6, marker="o", label=f"{material} 类")
        for material in ("A", "B", "C")
    ]
    inventory_handles.append(
        Line2D([], [], color=COLOR_MUTED, linewidth=1.4, linestyle=(0, (5, 4)), label="同色虚线：安全库存")
    )
    axes[1].legend(handles=inventory_handles, ncol=4, loc="upper right")
    _style_axis(axes[1])

    minimum_inventory = min(
        (
            week.inventories[material] - SAFETY_STOCKS[material],
            material,
            week.week,
        )
        for week in result.weeks
        for material in MATERIAL_DEMANDS
    )
    figure.text(
        0.075,
        0.025,
        f"注：全期最小安全库存裕量为 {result.minimum_inventory_margin:.3f}，"
        f"出现在 {minimum_inventory[1]} 类材料第 {minimum_inventory[2]} 周。",
        color=COLOR_MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.075, right=0.98, top=0.86, bottom=0.10)
    return _save_figure(figure, output_dir, "basic_material_inventory")


def plot_weekly_operations(
    result: BasicResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制逐周成本、运输损耗和最大转运容量占用率。"""
    week_numbers = np.array([week.week for week in result.weeks])
    costs = np.array([week.cost for week in result.weeks]) / 10000
    losses = np.array([week.loss for week in result.weeks])
    load_rates = np.array(
        [week.maximum_transfer_load / TRANSFER_CAPACITY for week in result.weeks]
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), facecolor=COLOR_WHITE)
    figure.suptitle(
        "基本方案的周度经济性与运营负载",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.915,
        f"12 周总成本 {result.total_cost:,.3f}｜累计运输损耗 {result.total_loss:,.3f}"
        f"｜最高转运容量占用率 {result.maximum_transfer_load / TRANSFER_CAPACITY:.3%}",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    axes[0].bar(week_numbers, costs, color=COLOR_BLUE, width=0.68, alpha=0.90, zorder=3)
    axes[0].set_title("备货周抬高阶段性成本", loc="left", pad=16)
    axes[0].set_ylabel("本周成本")
    _add_text_with_math_suffix(
        axes[0], "单位：", r"$\times 10^{4}$", position=(1.0, 1.02)
    )
    for index in np.argsort(costs)[-3:]:
        axes[0].text(
            week_numbers[index],
            costs[index] + 0.06,
            f"{costs[index]:.2f}",
            ha="center",
            va="bottom",
            color=COLOR_INK,
            fontsize=8.5,
        )
    axes[0].set_ylim(0, max(costs) * 1.22)

    axes[1].plot(
        week_numbers,
        losses,
        color=COLOR_ORANGE,
        linewidth=2.7,
        marker="o",
        markersize=6.5,
        markeredgecolor=COLOR_WHITE,
        markeredgewidth=1.1,
        zorder=4,
    )
    axes[1].fill_between(week_numbers, losses, color=COLOR_ORANGE, alpha=0.10)
    axes[1].axhline(
        losses.mean(), color=COLOR_MUTED, linewidth=1.2, linestyle=(0, (5, 4))
    )
    axes[1].text(
        12,
        losses.mean() + 5,
        f"周均 {losses.mean():.2f}",
        ha="right",
        color=COLOR_MUTED,
        fontsize=8.5,
    )
    axes[1].set_title("运输损耗总体逐周回落", loc="left", pad=16)
    axes[1].set_ylabel("本周运输损耗")
    axes[1].set_ylim(0, max(losses) * 1.25)

    load_colors = [COLOR_RED if rate >= 0.99 else COLOR_CYAN for rate in load_rates]
    axes[2].bar(
        week_numbers,
        load_rates,
        color=load_colors,
        width=0.68,
        alpha=0.90,
        zorder=3,
    )
    axes[2].axhline(1.0, color=COLOR_RED, linewidth=1.4, linestyle=(0, (5, 4)))
    axes[2].set_title("转运系统长期接近容量上限", loc="left", pad=16)
    axes[2].set_ylabel("最大转运容量占用率")
    axes[2].set_ylim(0.93, 1.006)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].legend(
        handles=[
            Patch(facecolor=COLOR_RED, label="占用率≥99%"),
            Patch(facecolor=COLOR_CYAN, label="占用率＜99%"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    for axis in axes:
        axis.set_xlabel("周次")
        axis.set_xticks(week_numbers)
        _style_axis(axis)
    figure.subplots_adjust(left=0.055, right=0.90, top=0.79, bottom=0.16, wspace=0.28)
    return _save_figure(figure, output_dir, "basic_weekly_operations")


def _provider_order(result: BasicResult) -> list[str]:
    material_by_provider = {entry.provider: entry.material for entry in result.plan}
    totals = {
        provider: sum(
            value
            for (week, current_provider), value in result.actual_supply.items()
            if current_provider == provider
        )
        for provider in result.active_providers
    }
    return sorted(
        result.active_providers,
        key=lambda provider: (
            ("A", "B", "C").index(material_by_provider[provider]),
            -totals[provider],
            provider,
        ),
    )


def plot_provider_transfer_schedule(
    result: BasicResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """绘制供应商、转运商及实际供货量的 12 周排程。"""
    providers = _provider_order(result)
    material_by_provider = {entry.provider: entry.material for entry in result.plan}
    plan_by_key = {(entry.week, entry.provider): entry for entry in result.plan}
    used_transfers = sorted(
        {entry.transfer for entry in result.plan}, key=lambda value: int(value[1:])
    )
    transfer_index = {transfer: index for index, transfer in enumerate(used_transfers)}
    matrix = np.full((len(providers), WEEK_COUNT), np.nan)
    totals = np.zeros(len(providers))
    for row, provider in enumerate(providers):
        for week in range(1, WEEK_COUNT + 1):
            entry = plan_by_key.get((week, provider))
            if entry is None:
                continue
            matrix[row, week - 1] = transfer_index[entry.transfer]
            totals[row] += result.actual_supply[week, provider]

    colors = [TRANSFER_COLORS[transfer] for transfer in used_transfers]
    colormap = ListedColormap(colors)
    colormap.set_bad(COLOR_SOFT)
    normalizer = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))

    figure, (schedule_axis, total_axis) = plt.subplots(
        1,
        2,
        figsize=(16.2, 8.0),
        facecolor=COLOR_WHITE,
        gridspec_kw={"width_ratios": (5.0, 1.25), "wspace": 0.08},
        sharey=True,
    )
    figure.suptitle(
        "未来 12 周供应商—转运商排程",
        x=0.065,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.925,
        "单元格上行为转运商，下行为基本情景下运输前实际供货量（千单位）",
        color=COLOR_MUTED,
        fontsize=10.5,
    )

    schedule_axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        norm=normalizer,
        origin="upper",
        aspect="auto",
    )
    schedule_axis.set_xticks(range(WEEK_COUNT), range(1, WEEK_COUNT + 1))
    schedule_axis.set_yticks(
        range(len(providers)),
        [f"{provider}   {material_by_provider[provider]}" for provider in providers],
    )
    schedule_axis.set_xlabel("周次", labelpad=10)
    schedule_axis.set_ylabel("供应商及材料类型", labelpad=12)
    schedule_axis.set_xticks(np.arange(-0.5, WEEK_COUNT, 1), minor=True)
    schedule_axis.set_yticks(np.arange(-0.5, len(providers), 1), minor=True)
    schedule_axis.grid(which="minor", color=COLOR_WHITE, linewidth=2.4)
    schedule_axis.tick_params(which="minor", bottom=False, left=False)
    schedule_axis.tick_params(length=0, pad=7)
    for spine in schedule_axis.spines.values():
        spine.set_visible(False)

    dark_text_transfers = {"T3"}
    for row, provider in enumerate(providers):
        for week in range(1, WEEK_COUNT + 1):
            entry = plan_by_key.get((week, provider))
            if entry is None:
                continue
            value = result.actual_supply[week, provider] / 1000
            schedule_axis.text(
                week - 1,
                row,
                f"{entry.transfer}\n{value:.2f}",
                ha="center",
                va="center",
                color=COLOR_INK if entry.transfer in dark_text_transfers else COLOR_WHITE,
                fontsize=7.8,
                linespacing=1.15,
                fontweight="semibold",
            )

    group_ends: list[int] = []
    for material in ("A", "B"):
        group_ends.append(
            max(
                index
                for index, provider in enumerate(providers)
                if material_by_provider[provider] == material
            )
        )
    for row in group_ends:
        schedule_axis.axhline(row + 0.5, color=COLOR_INK, linewidth=1.2, alpha=0.55)

    bar_colors = [MATERIAL_COLORS[material_by_provider[provider]] for provider in providers]
    total_axis.barh(
        range(len(providers)),
        totals / 1000,
        color=bar_colors,
        height=0.62,
        alpha=0.90,
        zorder=3,
    )
    total_axis.set_title("12 周累计实际供货", loc="left", pad=14)
    total_axis.set_xlabel("千单位")
    total_axis.tick_params(axis="y", left=False, labelleft=False)
    total_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    total_axis.set_xlim(0, max(totals / 1000) * 1.24)
    for row, total in enumerate(totals / 1000):
        total_axis.text(
            total + max(totals / 1000) * 0.025,
            row,
            f"{total:.1f}",
            va="center",
            color=COLOR_INK,
            fontsize=8.5,
        )
    _style_axis(total_axis, grid_axis="x")

    transfer_handles = [
        Patch(facecolor=TRANSFER_COLORS[transfer], label=transfer)
        for transfer in used_transfers
    ]
    material_handles = [
        Patch(facecolor=MATERIAL_COLORS[material], label=f"{material} 类累计量")
        for material in ("A", "B", "C")
    ]
    figure.legend(
        handles=transfer_handles + material_handles,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.02),
        ncol=len(used_transfers) + 3,
        columnspacing=1.35,
        handlelength=1.5,
    )
    figure.text(
        0.065,
        0.012,
        "注：空白单元格表示该供应商当周不供货；"
        f"本方案实际使用 {len(providers)} 家供应商和 {len(used_transfers)} 家转运商。",
        color=COLOR_MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(left=0.065, right=0.98, top=0.84, bottom=0.14)
    return _save_figure(figure, output_dir, "basic_provider_transfer_schedule")


def run_visualization(
    report_path: Path = DEFAULT_REPORT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """读取第三题基本结果并生成全部 SVG 论文图。"""
    _configure_style()
    result = load_basic_result(report_path)
    return [
        plot_material_inventory(result, output_dir),
        plot_weekly_operations(result, output_dir),
        plot_provider_transfer_schedule(result, output_dir),
    ]


if __name__ == "__main__":
    for generated_path in run_visualization():
        print(generated_path)
