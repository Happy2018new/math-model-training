"""可视化分析问题三的 100 组 eta 敏感性结果。"""

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "eta_sensitivity_results.txt"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "eta_sensitivity_analysis.png"


@dataclass(frozen=True)
class EtaResult:
    """一个 eta 场景的汇总指标。"""

    eta: float
    total_energy: float
    drone_count: int
    max_drone_energy: float
    mean_drone_energy: float
    route_signature: tuple[str, ...]


def load_results(input_path: Path) -> list[EtaResult]:
    """解析 100 个 eta 场景及其路线方案。"""
    text = input_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n(?=eta=)", text)[1:]
    results: list[EtaResult] = []
    for block in blocks:
        eta = float(re.search(r"eta=([0-9.]+)", block).group(1))
        total_energy = float(re.search(r"total_fuel_consumption=([0-9.]+)", block).group(1))
        drone_count = int(re.search(r"drone_count=(\d+)", block).group(1))
        drone_energies = [
            float(value)
            for value in re.findall(r"drone_\d+_fuel_consumption=([0-9.]+)", block)
        ]
        package_groups = tuple(
            sorted(re.findall(r"drone_\d+_package_ids=(.+)", block))
        )
        results.append(
            EtaResult(
                eta=eta,
                total_energy=total_energy,
                drone_count=drone_count,
                max_drone_energy=max(drone_energies),
                mean_drone_energy=float(np.mean(drone_energies)),
                route_signature=package_groups,
            )
        )

    results.sort(key=lambda result: result.eta)
    if len(results) != 100:
        raise ValueError(f"期望读取 100 个 eta 场景，实际读取到 {len(results)} 个。")
    return results


def configure_axis(axis: plt.Axes) -> None:
    """应用统一的白底现代图表样式。"""
    axis.set_facecolor("#ffffff")
    axis.grid(axis="y", color="#e7edf4", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors="#64748b", labelsize=9)
    for spine in (axis.spines["top"], axis.spines["right"]):
        spine.set_visible(False)
    for spine in (axis.spines["left"], axis.spines["bottom"]):
        spine.set_color("#dbe4ee")


def create_visualization(results: list[EtaResult]) -> None:
    """生成燃料消耗趋势、单机负担、方案切换和变化率综合图。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False

    eta = np.array([result.eta for result in results])
    total_energy = np.array([result.total_energy for result in results])
    max_energy = np.array([result.max_drone_energy for result in results])
    mean_energy = np.array([result.mean_drone_energy for result in results])
    baseline = total_energy[0]
    change_percent = (total_energy / baseline - 1.0) * 100.0
    signatures = {signature: index + 1 for index, signature in enumerate(sorted({result.route_signature for result in results}))}
    scheme_ids = np.array([signatures[result.route_signature] for result in results])
    switch_positions = np.where(scheme_ids[1:] != scheme_ids[:-1])[0] + 1

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180, facecolor="#ffffff")
    figure.subplots_adjust(left=0.075, right=0.96, top=0.86, bottom=0.09, hspace=0.34, wspace=0.25)
    for axis in axes.flat:
        configure_axis(axis)

    axes[0, 0].plot(eta, total_energy, color="#2563eb", linewidth=2.5)
    axes[0, 0].fill_between(eta, total_energy, total_energy.min() - 8, color="#dbeafe", alpha=0.55)
    minimum_index = int(np.argmin(total_energy))
    axes[0, 0].scatter(eta[minimum_index], total_energy[minimum_index], s=65, color="#f97316", edgecolor="white", linewidth=1.5, zorder=5)
    axes[0, 0].annotate(
        f"最低 {total_energy[minimum_index]:.2f}",
        (eta[minimum_index], total_energy[minimum_index]),
        xytext=(10, 14),
        textcoords="offset points",
        color="#9a3412",
        fontsize=9,
    )
    axes[0, 0].set_title("系统总等效燃料消耗响应", fontsize=14, color="#16324f", loc="left", pad=12)
    axes[0, 0].set_ylabel("等效燃料消耗", color="#526777")

    axes[0, 1].plot(eta, max_energy, color="#e85d75", linewidth=2.2, label="最大单机燃料消耗")
    axes[0, 1].plot(eta, mean_energy, color="#0d9488", linewidth=2.2, label="平均单机燃料消耗")
    axes[0, 1].fill_between(eta, mean_energy, max_energy, color="#ccfbf1", alpha=0.42)
    axes[0, 1].set_title("单机燃料消耗负担", fontsize=14, color="#16324f", loc="left", pad=12)
    axes[0, 1].set_ylabel("等效燃料消耗", color="#526777")
    axes[0, 1].legend(frameon=False, fontsize=9, labelcolor="#526777")

    axes[1, 0].step(eta, scheme_ids, where="mid", color="#7c3aed", linewidth=1.8)
    axes[1, 0].scatter(eta[switch_positions], scheme_ids[switch_positions], s=22, color="#f59e0b", edgecolor="white", linewidth=0.6, zorder=4)
    axes[1, 0].set_title("包裹分配方案切换", fontsize=14, color="#16324f", loc="left", pad=12)
    axes[1, 0].set_ylabel("方案编号", color="#526777")
    axes[1, 0].set_xlabel("载重影响参数 η", color="#526777")

    positive = change_percent >= 0
    axes[1, 1].bar(eta[positive], change_percent[positive], width=0.009, color="#38b2ac", alpha=0.86)
    axes[1, 1].bar(eta[~positive], change_percent[~positive], width=0.009, color="#60a5fa", alpha=0.86)
    axes[1, 1].axhline(0, color="#94a3b8", linewidth=1)
    axes[1, 1].set_title("相对 η=0.01 的燃料消耗变化", fontsize=14, color="#16324f", loc="left", pad=12)
    axes[1, 1].set_ylabel("变化率 / %", color="#526777")
    axes[1, 1].set_xlabel("载重影响参数 η", color="#526777")

    figure.suptitle("载重影响参数 η 的敏感性分析", x=0.075, y=0.955, ha="left", fontsize=23, color="#0f2942", fontweight="bold")
    figure.text(
        0.075,
        0.905,
        f"100 个场景  ·  η=0.01–1.00  ·  {len(signatures)} 种分配方案  ·  无人机数量始终为 {results[0].drone_count}",
        fontsize=11,
        color="#64748b",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    """读取 eta 敏感性结果并生成综合可视化。"""
    results = load_results(INPUT_PATH)
    create_visualization(results)
    print(f"Saved eta sensitivity visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
