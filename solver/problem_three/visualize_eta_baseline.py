"""可视化问题三 eta=0.30 时的包裹配送方案。"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageEnhance


Point = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "points_with_packages.json"
RESULTS_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "eta_sensitivity_results.txt"
MAP_PATH = PROJECT_ROOT / "others" / "input_picture.png"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "eta_030_routes.png"
TARGET_ETA = 0.30
ROUTE_COLORS = (
    "#2563eb",
    "#0d9488",
    "#f59e0b",
    "#e85d75",
    "#7c3aed",
    "#06b6d4",
    "#65a30d",
    "#ea580c",
)


@dataclass(frozen=True)
class RouteResult:
    """一架无人机在目标 eta 场景中的规划结果。"""

    package_ids: list[str]
    point_ids: list[int]
    initial_load: float
    energy: float


def load_eta_result(results_path: Path) -> tuple[float, float, list[RouteResult]]:
    """读取 eta=0.30 场景的最大载重、总燃料消耗和路线。"""
    text = results_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n(?=eta=)", text)[1:]
    for block in blocks:
        eta = float(re.search(r"eta=([0-9.]+)", block).group(1))
        if abs(eta - TARGET_ETA) > 1e-12:
            continue
        max_payload = float(re.search(r"max_payload_kg=([0-9.]+)", block).group(1))
        total_energy = float(re.search(r"total_fuel_consumption=([0-9.]+)", block).group(1))
        package_lines = re.findall(r"drone_\d+_package_ids=(.+)", block)
        route_lines = re.findall(r"drone_\d+_route=(.+)", block)
        loads = [float(value) for value in re.findall(r"drone_\d+_initial_load_kg=([0-9.]+)", block)]
        energies = [float(value) for value in re.findall(r"drone_\d+_fuel_consumption=([0-9.]+)", block)]
        routes = [
            RouteResult(
                package_ids=packages.split(","),
                point_ids=list(map(int, route.split(" -> "))),
                initial_load=load,
                energy=energy,
            )
            for packages, route, load, energy in zip(
                package_lines, route_lines, loads, energies
            )
        ]
        return max_payload, total_energy, routes
    raise ValueError(f"结果文件中不存在 eta={TARGET_ETA:.2f} 场景。")


def load_points(points_path: Path) -> tuple[Point, dict[int, Point]]:
    """读取仓库及配送点坐标。"""
    points: list[Point] = json.loads(points_path.read_text(encoding="utf-8"))
    depot = next(point for point in points if point.get("color") == "blue")
    return depot, {int(point["id"]): point for point in points}


def draw_route(
    axis: plt.Axes,
    route: RouteResult,
    points: dict[int, Point],
    color: str,
) -> None:
    """绘制一架无人机的有向路线和去重后的配送节点。"""
    for start_id, end_id in zip(route.point_ids, route.point_ids[1:]):
        if start_id == end_id:
            continue
        start = points[start_id]
        end = points[end_id]
        axis.add_patch(
            FancyArrowPatch(
                (start["x_math"], start["y_math"]),
                (end["x_math"], end["y_math"]),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.65,
                color=color,
                alpha=0.80,
                shrinkA=7,
                shrinkB=7,
                connectionstyle="arc3,rad=0.012",
                zorder=2,
            )
        )

    delivery_ids = list(dict.fromkeys(route.point_ids[1:-1]))
    delivery_points = [points[point_id] for point_id in delivery_ids]
    axis.scatter(
        [point["x_math"] for point in delivery_points],
        [point["y_math"] for point in delivery_points],
        s=54,
        color=color,
        edgecolor="white",
        linewidth=1.35,
        zorder=4,
    )
    for order, point in enumerate(delivery_points, start=1):
        axis.text(
            point["x_math"] + 5,
            point["y_math"] + 5,
            str(order),
            fontsize=7.5,
            color="#334155",
            zorder=5,
        )


def create_visualization(
    depot: Point,
    points: dict[int, Point],
    routes: list[RouteResult],
    max_payload: float,
    total_energy: float,
) -> None:
    """创建 eta=0.30 的白底现代配送方案图。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(16, 10), dpi=180, facecolor="#ffffff")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.13, 0.87),
        width_ratios=(4.1, 1.4),
        hspace=0.02,
        wspace=0.06,
    )
    header_axis = figure.add_subplot(grid[0, :])
    map_axis = figure.add_subplot(grid[1, 0])
    info_axis = figure.add_subplot(grid[1, 1])

    header_axis.axis("off")
    header_axis.text(
        0.025,
        0.68,
        "载重影响下的多无人机配送方案",
        transform=header_axis.transAxes,
        fontsize=23,
        color="#0f2942",
        fontweight="bold",
        va="center",
    )
    header_axis.text(
        0.025,
        0.22,
        "η=0.30 基准场景  ·  剩余载重驱动的等效燃料消耗优化",
        transform=header_axis.transAxes,
        fontsize=11,
        color="#64748b",
        va="center",
    )
    header_axis.plot([0, 1], [0.02, 0.02], transform=header_axis.transAxes, color="#dbe4ee", linewidth=1)

    map_image = Image.open(MAP_PATH).convert("RGB")
    map_image = ImageEnhance.Color(map_image).enhance(0.45)
    map_image = ImageEnhance.Contrast(map_image).enhance(0.72)
    map_image = ImageEnhance.Brightness(map_image).enhance(1.18)
    map_axis.imshow(
        np.asarray(map_image),
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="upper",
        alpha=0.88,
        zorder=0,
    )
    map_axis.imshow(
        np.full((2, 2, 4), (1.0, 1.0, 1.0, 0.38)),
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="lower",
        zorder=1,
    )
    for route, color in zip(routes, ROUTE_COLORS):
        draw_route(map_axis, route, points, color)

    map_axis.scatter(
        [depot["x_math"]],
        [depot["y_math"]],
        s=230,
        marker="D",
        color="#fb923c",
        edgecolor="white",
        linewidth=2.3,
        zorder=7,
    )
    map_axis.text(
        depot["x_math"] + 8,
        depot["y_math"] + 8,
        "仓库",
        color="#9a3412",
        fontsize=10,
        fontweight="bold",
        zorder=8,
    )
    map_axis.text(
        0.025,
        0.035,
        f"载重修正  W(t)=1+η·w(t)/Q\nη = {TARGET_ETA:.2f}  ·  Q = {max_payload:.0f} kg",
        transform=map_axis.transAxes,
        fontsize=9.5,
        color="#334155",
        linespacing=1.55,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.65,rounding_size=0.25",
            "facecolor": "white",
            "edgecolor": "#dbe4ee",
            "alpha": 0.92,
        },
        zorder=10,
    )
    map_axis.set_xlim(0, map_image.width - 1)
    map_axis.set_ylim(0, map_image.height - 1)
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.set_xlabel("横向坐标 / 像素", color="#64748b", labelpad=10)
    map_axis.set_ylabel("纵向坐标 / 像素", color="#64748b", labelpad=10)
    map_axis.tick_params(colors="#94a3b8")
    for spine in map_axis.spines.values():
        spine.set_color("#e2e8f0")

    info_axis.axis("off")
    info_axis.text(0.03, 0.95, "方案概览", fontsize=20, fontweight="bold", color="#0f2942")
    info_axis.text(0.03, 0.89, f"{len(routes)} 架无人机  ·  47 个包裹", fontsize=10.5, color="#64748b")
    info_axis.text(0.03, 0.82, "系统总等效燃料消耗", fontsize=10.5, color="#64748b")
    info_axis.text(0.03, 0.765, f"{total_energy:.2f}", fontsize=24, fontweight="bold", color="#0f2942")
    info_axis.plot([0.03, 0.96], [0.71, 0.71], color="#dbe4ee", linewidth=1)

    y = 0.655
    for index, (route, color) in enumerate(zip(routes, ROUTE_COLORS), start=1):
        info_axis.scatter([0.055], [y], s=72, color=color, edgecolor="white", linewidth=1.1)
        info_axis.text(0.12, y + 0.014, f"无人机 {index}", fontsize=10, fontweight="bold", color="#243b53")
        info_axis.text(
            0.12,
            y - 0.019,
            f"{len(route.package_ids)} 件  ·  {route.initial_load:.2f} kg  ·  燃料消耗 {route.energy:.2f}",
            fontsize=8.7,
            color="#78909f",
        )
        y -= 0.073

    info_axis.text(
        0.03,
        0.035,
        "节点数字表示该无人机访问顺序\n同一地点的多个包裹合并显示",
        fontsize=9,
        color="#78909f",
        linespacing=1.5,
    )
    info_axis.set_xlim(0, 1)
    info_axis.set_ylim(0, 1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="#ffffff")
    plt.close(figure)


def main() -> None:
    """读取 eta=0.30 结果并生成配送方案图。"""
    max_payload, total_energy, routes = load_eta_result(RESULTS_PATH)
    depot, points = load_points(POINTS_PATH)
    create_visualization(depot, points, routes, max_payload, total_energy)
    print(f"Saved eta=0.30 route visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
