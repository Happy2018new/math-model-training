import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ..sensitivity_common import configure_plot_fonts


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "three"


def _configure_chinese_font() -> None:
    configure_plot_fonts()


def plot_sensitivity(
    csv_path: str | Path | None = None,
    save_name: str = "problem_three_sensitivity.svg",
) -> Path:
    """绘制问题三敏感性分析图。"""
    _configure_chinese_font()
    csv_path = Path(csv_path) if csv_path is not None else OUTPUT_DIR / "sensitivity.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("sensitivity CSV is empty")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    smooth_rows = [row for row in rows if row["experiment"] == "smoothing_window"]
    x = [float(row["parameter"]) for row in smooth_rows]
    axes[0, 0].plot(x, [float(row["delta_time"]) for row in smooth_rows], marker="o", label="时间偏差")
    axes[0, 0].set_xlabel("平滑窗口（点）")
    axes[0, 0].set_ylabel("时间偏差（秒）")
    axes[0, 0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(x, [float(row["mean_dx"]) for row in smooth_rows], marker="o", label="X 方向平均差值")
    axes[0, 1].plot(x, [float(row["mean_dy"]) for row in smooth_rows], marker="s", label="Y 方向平均差值")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8, label="零偏差")
    axes[0, 1].set_xlabel("平滑窗口（点）")
    axes[0, 1].set_ylabel("平均空间差值（米）")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right")

    block_rows = [row for row in rows if row["experiment"] == "block_confidence"]
    for confidence, marker in ((0.90, "o"), (0.95, "s"), (0.99, "^")):
        selected = [row for row in block_rows if abs(float(row["confidence"]) - confidence) < 1e-9]
        selected.sort(key=lambda row: float(row["parameter"]))
        bx = [float(row["parameter"]) for row in selected]
        axes[1, 0].plot(
            bx,
            [float(row["x_high"]) - float(row["x_low"]) for row in selected],
            marker=marker,
            label=f"{confidence:.0%} 置信区间宽度",
        )
        axes[1, 1].plot(
            bx,
            [float(row["y_high"]) - float(row["y_low"]) for row in selected],
            marker=marker,
            label=f"{confidence:.0%} 置信区间宽度",
        )
    axes[1, 0].set_xlabel("分块大小（点）")
    axes[1, 0].set_ylabel("X 轴置信区间宽度（米）")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="upper right")
    axes[1, 1].set_xlabel("分块大小（点）")
    axes[1, 1].set_ylabel("Y 轴置信区间宽度（米）")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(plot_sensitivity())
