import csv
from pathlib import Path

import matplotlib.pyplot as plt

from ..sensitivity_common import configure_plot_fonts


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "two"


def _configure_chinese_font() -> None:
    configure_plot_fonts()


def plot_sensitivity(
    csv_path: str | Path | None = None,
    save_name: str = "problem_two_sensitivity.svg",
) -> Path:
    """绘制问题二敏感性分析图。"""
    _configure_chinese_font()
    csv_path = Path(csv_path) if csv_path is not None else OUTPUT_DIR / "sensitivity.csv"
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("sensitivity CSV is empty")

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for experiment, axis, xlabel in (
        ("smoothing_window", axes[0, 0], "平滑窗口（点）"),
        ("overlap_trim", axes[0, 1], "两端裁剪比例"),
    ):
        selected = [row for row in rows if row["experiment"] == experiment]
        x = [float(row["parameter"]) for row in selected]
        axis.plot(x, [float(row["delta_time"]) for row in selected], marker="o", label="时间偏差")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("时间偏差（秒）")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, alpha=0.3)
        axis.legend()

    selected = [row for row in rows if row["experiment"] == "smoothing_window"]
    x = [float(row["parameter"]) for row in selected]
    axes[1, 0].plot(x, [float(row["mean_dx"]) for row in selected], marker="o", label="X 方向平均差值")
    axes[1, 0].plot(x, [float(row["mean_dy"]) for row in selected], marker="s", label="Y 方向平均差值")
    axes[1, 0].set_xlabel("平滑窗口（点）")
    axes[1, 0].set_ylabel("平均空间差值（米）")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(x, [float(row["rmse"]) for row in selected], marker="o", label="均方根误差（RMSE）")
    axes[1, 1].plot(x, [float(row["std_dx"]) for row in selected], marker="s", label="X 方向标准差")
    axes[1, 1].plot(x, [float(row["std_dy"]) for row in selected], marker="^", label="Y 方向标准差")
    axes[1, 1].set_xlabel("平滑窗口（点）")
    axes[1, 1].set_ylabel("误差（米）")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(plot_sensitivity())
