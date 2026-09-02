import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from ..sensitivity_common import configure_plot_fonts


BASE_DIR = Path(__file__).resolve().parents[2]
CSV_DIR = BASE_DIR / "outputs" / "one"
OUTPUT_DIR = BASE_DIR / "outputs" / "one"

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_sensitivity(
    csv_path: str | Path | None = None,
    save_name: str = "problem_one_sensitivity.svg",
) -> Path:
    csv_path = Path(csv_path) if csv_path is not None else CSV_DIR / "sensitivity.csv"
    configure_plot_fonts()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError("sensitivity CSV is empty")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for experiment, axis, xlabel in (
        ("smoothing_window", axes[0], "平滑窗口（点）"),
        ("coarse_step", axes[1], "粗搜索步长（秒）"),
    ):
        selected = [row for row in rows if row["experiment"] == experiment]
        x = [float(row["parameter"]) for row in selected]
        delta = [float(row["delta_time"]) for row in selected]
        objective = [float(row["objective"]) * 1_000_000.0 for row in selected]
        axis.plot(x, delta, marker="o", label="估计时间偏差")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("时间偏差（秒）")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        axis.grid(True, alpha=0.3)
        twin = axis.twinx()
        twin.plot(x, objective, marker="s", color="#c45b3c", label="目标函数")
        # Keep Chinese labels as plain text; math symbols are rendered in
        # dedicated mathtext objects where needed so font fallback is stable.
        twin.set_ylabel("目标函数（平方毫米）")
        twin.ticklabel_format(axis="y", style="plain", useOffset=False)
        twin.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(
            handles + twin_handles,
            labels + twin_labels,
            loc="best",
            frameon=True,
        )
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = (OUTPUT_DIR / save_name).with_suffix(".svg")
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(plot_sensitivity())
