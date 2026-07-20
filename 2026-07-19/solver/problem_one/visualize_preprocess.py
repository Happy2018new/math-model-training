"""Visualize the raw-data checks used by problem one."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

from .preprocess import PreprocessAudit, run_audit, write_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "output" / "problems" / "one" / "preprocess" / "figures"
)

TEXT_FONT = "SimSun"
NUMBER_FONT = "Latin Modern Math"
COLOR_BLUE = "#2563EB"
COLOR_GREEN = "#10B981"
COLOR_ORANGE = "#F97316"
COLOR_RED = "#EF4444"
COLOR_INK = "#172033"
COLOR_MUTED = "#64748B"
COLOR_GRID = "#DCE5EF"
COLOR_CANVAS = "#FFFFFF"


def _configure_style() -> None:
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
            "savefig.facecolor": COLOR_CANVAS,
        }
    )


def _save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, bbox_inches="tight", facecolor=COLOR_CANVAS)
    plt.close(figure)


def plot_data_quality(audit: PreprocessAudit, output_dir: Path) -> Path:
    """Plot integrity checks and meaningful zero-value rates."""
    figure, (status_axis, zero_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": (1.35, 1)},
    )

    checks = (
        ("缺失单元格", audit.order.missing_cells + audit.supply.missing_cells),
        (
            "重复供应商",
            len(audit.order.duplicate_ids) + len(audit.supply.duplicate_ids),
        ),
        (
            "非数值记录",
            audit.order.non_numeric_cells + audit.supply.non_numeric_cells,
        ),
        ("负值记录", audit.order.negative_cells + audit.supply.negative_cells),
        ("跨表类别不一致", len(audit.material_mismatches)),
    )
    rows = list(range(len(checks)))
    status_axis.scatter(
        [0] * len(checks),
        rows,
        s=90,
        color=COLOR_GREEN,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    for row, (label, value) in enumerate(checks):
        status_axis.text(0.05, row, label, va="center", fontsize=11)
        status_axis.text(
            0.95,
            row,
            str(value),
            va="center",
            ha="right",
            fontsize=13,
            color=COLOR_GREEN if value == 0 else COLOR_RED,
            fontfamily=NUMBER_FONT,
            fontweight="bold",
        )
    status_axis.set_xlim(-0.05, 1.0)
    status_axis.set_ylim(-0.7, len(checks) - 0.3)
    status_axis.invert_yaxis()
    status_axis.set_xticks([])
    status_axis.set_yticks([])
    status_axis.set_title("数据完整性检查", loc="left", pad=18)
    status_axis.text(
        0,
        1.02,
        "$\\mathrm{318}$ 家供应商 · $\\mathrm{240}$ 周记录 · 全部检查通过",
        transform=status_axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    for spine in status_axis.spines.values():
        spine.set_visible(False)

    zero_rates = (audit.order.zero_rate, audit.supply.zero_rate)
    bars = zero_axis.bar(
        ("企业订货量", "供应商供货量"),
        zero_rates,
        color=(COLOR_BLUE, COLOR_ORANGE),
        width=0.58,
    )
    zero_axis.set_ylim(0, 0.8)
    zero_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    zero_axis.set_ylabel("零值比例")
    zero_axis.set_title("零值是有效业务记录", loc="left", pad=18)
    zero_axis.text(
        0,
        1.02,
        "零值表示当周未订货或未供货，不进行插补",
        transform=zero_axis.transAxes,
        color=COLOR_MUTED,
        fontsize=10,
    )
    for bar, value in zip(bars, zero_rates):
        zero_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.1%}",
            ha="center",
            fontfamily=NUMBER_FONT,
            fontweight="bold",
        )
    zero_axis.grid(axis="y")
    for spine in ("top", "right", "left"):
        zero_axis.spines[spine].set_visible(False)
    for label in zero_axis.get_yticklabels():
        label.set_fontfamily(NUMBER_FONT)

    figure.suptitle(
        "原始数据完整，零值具有明确业务含义",
        x=0.06,
        y=1.01,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=COLOR_INK,
    )
    figure.tight_layout()
    output_path = output_dir / "preprocess_data_quality.svg"
    _save_figure(figure, output_path)
    return output_path


def run_visualization(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Check source data and generate the single preprocessing figure."""
    _configure_style()
    audit = run_audit()
    write_outputs(audit)
    if not audit.passed:
        raise ValueError("原始数据检查未通过，已停止生成预处理图")

    output_dir.mkdir(parents=True, exist_ok=True)
    return plot_data_quality(audit, output_dir)


if __name__ == "__main__":
    written_path = run_visualization()
    print(f"已生成：{written_path}")
