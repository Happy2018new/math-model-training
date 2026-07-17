"""显示 alpha、beta 对多无人机路线选择的影响。"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity_results.txt"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "route_selection_impact.png"


def load_scenarios(input_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """读取每个情景的参数和各无人机访问顺序。"""
    text = input_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n(?=alpha=)", text)[1:]
    scenarios: list[dict[str, object]] = []

    for block in blocks:
        alpha = float(re.search(r"alpha=([0-9.]+)", block).group(1))
        beta = float(re.search(r"beta=([0-9.]+)", block).group(1))
        routes = tuple(
            tuple(map(int, match.split(" -> ")))
            for match in re.findall(r"drone_\d+_route=(.+)", block)
        )
        scenarios.append({"alpha": alpha, "beta": beta, "routes": tuple(sorted(routes))})

    if len(scenarios) != 25:
        raise ValueError(f"期望读取 25 个场景，实际读取到 {len(scenarios)} 个。")
    return (
        np.array(sorted({scenario["alpha"] for scenario in scenarios})),
        np.array(sorted({scenario["beta"] for scenario in scenarios})),
        scenarios,
    )


def create_visualization(
    alphas: np.ndarray, betas: np.ndarray, scenarios: list[dict[str, object]]
) -> None:
    """创建路线方案编号和相对基准路线变化数量的双热力图。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    scenario_map = {(item["alpha"], item["beta"]): item["routes"] for item in scenarios}
    baseline = scenario_map[1.0, 1.0]
    signatures = sorted(set(scenario_map.values()))
    signature_id = {signature: index + 1 for index, signature in enumerate(signatures)}

    scheme_grid = np.array(
        [[signature_id[scenario_map[alpha, beta]] for beta in betas] for alpha in alphas]
    )
    changed_routes_grid = np.array(
        [
            [
                sum(route not in baseline for route in scenario_map[alpha, beta])
                for beta in betas
            ]
            for alpha in alphas
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 7.4), dpi=180, facecolor="#ffffff")
    figure.subplots_adjust(left=0.07, right=0.94, top=0.84, bottom=0.15, wspace=0.28)
    scheme_colors = ["#dbeafe", "#bfdbfe", "#93c5fd", "#67e8f9", "#a7f3d0", "#fde68a", "#fbcfe8"]
    scheme_map = ListedColormap(scheme_colors[: len(signatures)])
    change_map = ListedColormap(["#dff7f2", "#9ee7d8", "#55c7c0", "#258ca5", "#1d4f78", "#162f50"])

    for axis in axes:
        axis.set_facecolor("#ffffff")
        axis.set_xticks(range(len(betas)), [f"{value:.1f}" for value in betas])
        axis.set_yticks(range(len(alphas)), [f"{value:.1f}" for value in alphas])
        axis.set_xlabel("风力影响强度 β", color="#334e68", labelpad=10)
        axis.set_ylabel("天气影响强度 α", color="#334e68", labelpad=10)
        axis.tick_params(colors="#526777")
        for spine in axis.spines.values():
            spine.set_color("#d9e2ec")

    image_scheme = axes[0].imshow(scheme_grid, cmap=scheme_map, vmin=1, vmax=len(signatures))
    axes[0].set_title("最优路线方案的变化", color="#16324f", fontsize=16, pad=15)
    for row, alpha in enumerate(alphas):
        for column, beta in enumerate(betas):
            scheme = scheme_grid[row, column]
            axes[0].text(column, row, f"方案 {scheme}", ha="center", va="center", color="#16324f", fontsize=9)

    image_change = axes[1].imshow(changed_routes_grid, cmap=change_map, vmin=0, vmax=5)
    axes[1].set_title("相对基准的路线变化数量", color="#16324f", fontsize=16, pad=15)
    for row, alpha in enumerate(alphas):
        for column, beta in enumerate(betas):
            changed = changed_routes_grid[row, column]
            color = "white" if changed >= 3 else "#16324f"
            axes[1].text(column, row, str(changed), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    baseline_row = int(np.where(alphas == 1.0)[0][0])
    baseline_column = int(np.where(betas == 1.0)[0][0])
    for axis in axes:
        axis.add_patch(
            Rectangle(
                (baseline_column - 0.5, baseline_row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#f97316",
                linewidth=3,
            )
        )

    colorbar = figure.colorbar(image_change, ax=axes[1], shrink=0.82, pad=0.04)
    colorbar.set_label("变化的无人机路线数量", color="#334e68")
    colorbar.ax.tick_params(colors="#526777")
    colorbar.outline.set_visible(False)
    figure.suptitle("天气与风力参数对航程选择的影响", color="#16324f", fontsize=22, y=0.95)
    figure.text(
        0.07,
        0.06,
        "橙色边框为基准场景 (α=1, β=1)。路线方向保留在方案比较中，因为风向修正 k 使正反向航程代价不同。",
        color="#66788a",
        fontsize=10,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    """读取结果文本并生成路线选择影响图。"""
    alphas, betas, scenarios = load_scenarios(INPUT_PATH)
    create_visualization(alphas, betas, scenarios)
    print(f"Saved route selection impact visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
