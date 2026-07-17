"""可视化敏感性分析中 alpha=1、beta=1 的基准配送路线。"""

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.patches import FancyArrowPatch

try:
    from .resolve import (
        MAP_SCALE_KM_PER_PIXEL,
        WEATHER_CODE_OVERRIDE,
        weather_coefficient,
    )
except ImportError:
    from resolve import (
        MAP_SCALE_KM_PER_PIXEL,
        WEATHER_CODE_OVERRIDE,
        weather_coefficient,
    )


Point = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTS_PATH = PROJECT_ROOT / "output" / "processed" / "points.json"
MAP_PATH = PROJECT_ROOT / "others" / "input_picture.png"
RESULTS_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity_results.txt"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "baseline_routes.png"
ROUTE_COLORS = ("#2563eb", "#0d9488", "#f59e0b", "#e85d75", "#7c3aed")


def load_baseline_result(
    results_path: Path,
) -> tuple[float, list[list[int]], list[float]]:
    """读取 alpha=1、beta=1 场景的总航程、路线和单机航程。"""
    text = results_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n(?=alpha=)", text)[1:]
    for block in blocks:
        alpha = float(re.search(r"alpha=([0-9.]+)", block).group(1))
        beta = float(re.search(r"beta=([0-9.]+)", block).group(1))
        if alpha != 1.0 or beta != 1.0:
            continue
        total = float(
            re.search(r"total_corrected_distance_km=([0-9.]+)", block).group(1)
        )
        routes = [
            list(map(int, route.split(" -> ")))
            for route in re.findall(r"drone_\d+_route=(.+)", block)
        ]
        costs = [
            float(cost)
            for cost in re.findall(r"drone_\d+_corrected_distance_km=([0-9.]+)", block)
        ]
        return total, routes, costs
    raise ValueError("结果文件中不存在 alpha=1、beta=1 的基准场景。")


def load_points(points_path: Path) -> tuple[Point, dict[int, Point]]:
    """读取唯一仓库和按 ID 索引的配送点。"""
    points: list[Point] = json.loads(points_path.read_text(encoding="utf-8"))
    depots = [point for point in points if point.get("color") == "blue"]
    if len(depots) != 1:
        raise ValueError(f"应存在唯一蓝色仓库，实际数量为 {len(depots)}。")
    return depots[0], {int(point["id"]): point for point in points}


def draw_route(
    axis: plt.Axes,
    depot: Point,
    route: list[int],
    points: dict[int, Point],
    color: str,
) -> None:
    """绘制一条带方向箭头的仓库闭环路线。"""
    closed_route = [int(depot["id"])] + route + [int(depot["id"])]
    for start_id, end_id in zip(closed_route, closed_route[1:]):
        start = points[start_id]
        end = points[end_id]
        arrow = FancyArrowPatch(
            (start["x_math"], start["y_math"]),
            (end["x_math"], end["y_math"]),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.8,
            color=color,
            alpha=0.82,
            shrinkA=7,
            shrinkB=7,
            connectionstyle="arc3,rad=0.015",
            zorder=2,
        )
        axis.add_patch(arrow)

    route_points = [points[point_id] for point_id in route]
    axis.scatter(
        [point["x_math"] for point in route_points],
        [point["y_math"] for point in route_points],
        s=58,
        color=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
    )
    for order, point in enumerate(route_points, start=1):
        axis.text(
            point["x_math"] + 5,
            point["y_math"] + 5,
            str(order),
            fontsize=8,
            color="#334155",
            zorder=5,
        )


def create_visualization(
    depot: Point,
    points: dict[int, Point],
    routes: list[list[int]],
    costs: list[float],
    total_cost: float,
) -> None:
    """创建基准多无人机路线地图与统计信息。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(16, 10), dpi=180, facecolor="#f8fafc")
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.13, 0.87),
        width_ratios=(4.25, 1.25),
        hspace=0.02,
        wspace=0.06,
    )
    header_axis = figure.add_subplot(grid[0, :])
    map_axis = figure.add_subplot(grid[1, 0])
    info_axis = figure.add_subplot(grid[1, 1])

    header_axis.set_facecolor("#f8fafc")
    header_axis.axis("off")
    header_axis.text(
        0.025,
        0.68,
        "雷暴情景下的多无人机基准配送路线",
        transform=header_axis.transAxes,
        fontsize=23,
        color="#0f2942",
        fontweight="bold",
        va="center",
    )
    header_axis.text(
        0.025,
        0.22,
        "基于修正航程 F(i,j) 的 α=1 且 β=1 的最优方案",
        transform=header_axis.transAxes,
        fontsize=11,
        color="#64748b",
        va="center",
    )
    header_axis.plot(
        [0.0, 1.0],
        [0.02, 0.02],
        transform=header_axis.transAxes,
        color="#dbe4ee",
        linewidth=1,
    )

    map_image = Image.open(MAP_PATH).convert("RGB")
    map_image = ImageEnhance.Color(map_image).enhance(0.45)
    map_image = ImageEnhance.Contrast(map_image).enhance(0.72)
    map_image = ImageEnhance.Brightness(map_image).enhance(1.18)
    map_array = np.asarray(map_image)
    map_axis.imshow(
        map_array,
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="upper",
        alpha=0.88,
        zorder=0,
    )
    map_axis.imshow(
        np.full((2, 2, 4), (1.0, 1.0, 1.0, 0.36)),
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="lower",
        zorder=1,
    )
    map_axis.set_facecolor("#ffffff")
    map_axis.grid(color="#ffffff", linewidth=0.8, alpha=0.5)
    map_axis.set_axisbelow(True)
    for route, color in zip(routes, ROUTE_COLORS):
        draw_route(map_axis, depot, route, points, color)

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
        (
            "假设雷暴天气模拟\n"
            f"WMO 天气代码 {WEATHER_CODE_OVERRIDE}  (μ = {weather_coefficient(WEATHER_CODE_OVERRIDE):.2f})\n"
            f"比例尺 L = {MAP_SCALE_KM_PER_PIXEL:.3f} km/像素"
        ),
        transform=map_axis.transAxes,
        fontsize=9.5,
        color="#334155",
        linespacing=1.55,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.65,rounding_size=0.25",
            "facecolor": "white",
            "edgecolor": "#dbe4ee",
            "alpha": 0.90,
        },
        zorder=10,
    )
    map_axis.set_aspect("equal", adjustable="box")
    map_axis.set_xlim(0, map_image.width - 1)
    map_axis.set_ylim(0, map_image.height - 1)
    map_axis.set_xlabel("横向坐标 / 像素", color="#64748b", labelpad=10)
    map_axis.set_ylabel("纵向坐标 / 像素", color="#64748b", labelpad=10)
    map_axis.tick_params(colors="#94a3b8")
    for spine in map_axis.spines.values():
        spine.set_color("#e2e8f0")

    info_axis.set_facecolor("#f8fafc")
    info_axis.axis("off")
    info_axis.text(
        0.03, 0.94, "基准方案", fontsize=21, fontweight="bold", color="#0f2942"
    )
    info_axis.text(0.03, 0.885, "α = 1.0   β = 1.0", fontsize=11, color="#64748b")
    info_axis.text(0.03, 0.80, "总修正航程", fontsize=11, color="#64748b")
    info_axis.text(
        0.03,
        0.735,
        f"{total_cost:.2f} km",
        fontsize=25,
        fontweight="bold",
        color="#0f2942",
    )
    info_axis.plot([0.03, 0.95], [0.68, 0.68], color="#dbe4ee", linewidth=1)

    y = 0.62
    for index, (route, cost, color) in enumerate(
        zip(routes, costs, ROUTE_COLORS), start=1
    ):
        info_axis.scatter(
            [0.06], [y], s=82, color=color, edgecolor="white", linewidth=1.2
        )
        info_axis.text(
            0.13,
            y + 0.018,
            f"无人机 {index}",
            fontsize=11,
            fontweight="bold",
            color="#243b53",
        )
        info_axis.text(
            0.13,
            y - 0.018,
            f"7 个配送点   {cost:.2f} km",
            fontsize=9.5,
            color="#78909f",
        )
        y -= 0.105

    info_axis.text(
        0.03,
        0.055,
        "数字表示各无人机的访问顺序\n箭头表示实际飞行方向",
        fontsize=9.5,
        color="#78909f",
        linespacing=1.55,
    )
    info_axis.set_xlim(0, 1)
    info_axis.set_ylim(0, 1)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    """读取基准场景并保存路线可视化。"""
    total_cost, routes, costs = load_baseline_result(RESULTS_PATH)
    depot, points = load_points(POINTS_PATH)
    create_visualization(depot, points, routes, costs, total_cost)
    print(f"Saved baseline route visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
