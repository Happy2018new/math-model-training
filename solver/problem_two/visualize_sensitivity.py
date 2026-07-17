"""将 25 个敏感性分析场景绘制为现代亮色三维响应曲面。"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity_results.txt"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity_surface.png"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def load_results(input_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从敏感性分析文本中读取 alpha、beta 和总修正航程。"""
    text = input_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"alpha=([0-9.]+)\s+beta=([0-9.]+)\s+"
        r"total_corrected_distance_km=([0-9.]+)"
    )
    records = [(float(a), float(b), float(cost)) for a, b, cost in pattern.findall(text)]
    if len(records) != 25:
        raise ValueError(f"期望读取 25 个场景，实际读取到 {len(records)} 个。")

    alphas = np.array(sorted({record[0] for record in records}))
    betas = np.array(sorted({record[1] for record in records}))
    costs = {(alpha, beta): cost for alpha, beta, cost in records}
    alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="ij")
    cost_grid = np.array(
        [[costs[alpha, beta] for beta in betas] for alpha in alphas]
    )
    return alpha_grid, beta_grid, cost_grid


def create_visualization(
    alpha_grid: np.ndarray, beta_grid: np.ndarray, cost_grid: np.ndarray
) -> None:
    """创建三维曲面、底部等高线和基准场景标记。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(15, 10), dpi=180, facecolor="#f7fafc")
    axis = figure.add_subplot(111, projection="3d")
    axis.set_facecolor("#f7fafc")

    colormap = LinearSegmentedColormap.from_list(
        "clear_blue", ["#dff7f2", "#7dd3c7", "#38a6c6", "#2563a8"]
    )
    surface = axis.plot_surface(
        alpha_grid,
        beta_grid,
        cost_grid,
        cmap=colormap,
        edgecolor=(1, 1, 1, 0.7),
        linewidth=0.8,
        antialiased=True,
        alpha=0.96,
    )

    floor = float(cost_grid.min() - 2.0)
    axis.contourf(
        alpha_grid,
        beta_grid,
        cost_grid,
        zdir="z",
        offset=floor,
        levels=12,
        cmap=colormap,
        alpha=0.42,
    )

    baseline_cost = float(cost_grid[2, 2])
    axis.scatter(
        [1.0], [1.0], [baseline_cost], s=90, color="#f97316", edgecolor="white", linewidth=1.8
    )
    axis.text(
        1.03,
        1.03,
        baseline_cost + 0.35,
        f"基准场景  {baseline_cost:.2f} km",
        color="#9a3412",
        fontsize=10,
    )

    axis.set_title("天气与风力参数敏感性响应曲面", fontsize=22, color="#16324f", pad=24)
    axis.set_xlabel("天气影响强度 α", fontsize=12, color="#334e68", labelpad=12)
    axis.set_ylabel("风力影响强度 β", fontsize=12, color="#334e68", labelpad=12)
    axis.set_zlabel("总修正航程 / km", fontsize=12, color="#334e68", labelpad=12)
    axis.set_xticks(np.unique(alpha_grid))
    axis.set_yticks(np.unique(beta_grid))
    axis.set_zlim(floor, float(cost_grid.max() + 1.5))
    axis.view_init(elev=29, azim=-53)
    axis.set_box_aspect((1.1, 1.1, 0.7))

    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
        pane.set_edgecolor((0.87, 0.91, 0.94, 1.0))
    axis.grid(True, color="#d9e2ec", linewidth=0.7, alpha=0.7)
    axis.tick_params(colors="#526777", labelsize=10)

    colorbar = figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.08, aspect=22)
    colorbar.set_label("总修正航程 / km", color="#334e68", fontsize=11)
    colorbar.ax.tick_params(colors="#526777")
    colorbar.outline.set_visible(False)

    figure.text(
        0.075,
        0.055,
        "α = 1、β = 1 为基准模型；曲面高度越低，方案总修正航程越小。",
        color="#66788a",
        fontsize=10,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    """读取敏感性分析结果并保存三维可视化图片。"""
    alpha_grid, beta_grid, cost_grid = load_results(INPUT_PATH)
    create_visualization(alpha_grid, beta_grid, cost_grid)
    print(f"Saved sensitivity visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
