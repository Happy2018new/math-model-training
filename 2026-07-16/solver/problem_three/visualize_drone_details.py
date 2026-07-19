"""为 eta=0.30 方案中的每架无人机生成独立配送详情图。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageEnhance

try:
    from .resolve import (
        DEPARTURE_TIME,
        build_distance_lookup,
        load_weather_snapshot,
    )
    from .visualize_eta_baseline import (
        POINTS_PATH,
        RESULTS_PATH,
        TARGET_ETA,
        RouteResult,
        load_eta_result,
        load_points,
    )
except ImportError:
    from resolve import DEPARTURE_TIME, build_distance_lookup, load_weather_snapshot
    from visualize_eta_baseline import (
        POINTS_PATH,
        RESULTS_PATH,
        TARGET_ETA,
        RouteResult,
        load_eta_result,
        load_points,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = PROJECT_ROOT / "others" / "input_picture.png"
OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "three" / "drone_routes"
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

Segment = dict[str, object]


def package_weights_by_id(points: dict[int, dict]) -> dict[str, float]:
    """建立包裹编号到重量的映射。"""
    return {
        f"{point_id}-{package_index}": float(weight)
        for point_id, point in points.items()
        for package_index, weight in enumerate(point.get("packages", []), start=1)
    }


def calculate_segments(
    route: RouteResult,
    points: dict[int, dict],
    package_weights: dict[str, float],
    max_payload: float,
) -> list[Segment]:
    """按实际停靠点聚合包裹，并计算各航段载重与等效燃料消耗。"""
    weather = load_weather_snapshot(DEPARTURE_TIME)
    base_costs = build_distance_lookup(list(points.values()), weather)
    remaining_load = route.initial_load
    cumulative_energy = 0.0
    segments: list[Segment] = []

    package_index = 0
    start_id = route.point_ids[0]
    while package_index < len(route.package_ids):
        end_id = route.point_ids[package_index + 1]
        stop_package_ids: list[str] = []
        while (
            package_index < len(route.package_ids)
            and route.point_ids[package_index + 1] == end_id
        ):
            stop_package_ids.append(route.package_ids[package_index])
            package_index += 1

        package_weight = sum(package_weights[package_id] for package_id in stop_package_ids)
        load_before = remaining_load
        load_factor = 1.0 + TARGET_ETA * load_before / max_payload
        segment_energy = base_costs[start_id, end_id] * load_factor
        remaining_load -= package_weight
        cumulative_energy += segment_energy
        segments.append(
            {
                "start": start_id,
                "end": end_id,
                "package_ids": stop_package_ids,
                "package_weight": package_weight,
                "load_before": load_before,
                "load_after": remaining_load,
                "load_factor": load_factor,
                "segment_energy": segment_energy,
                "cumulative_energy": cumulative_energy,
            }
        )
        start_id = end_id

    return_start = start_id
    return_end = route.point_ids[-1]
    return_energy = base_costs[return_start, return_end]
    cumulative_energy += return_energy
    segments.append(
        {
            "start": return_start,
            "end": return_end,
            "package_ids": [],
            "package_weight": 0.0,
            "load_before": 0.0,
            "load_after": 0.0,
            "load_factor": 1.0,
            "segment_energy": return_energy,
            "cumulative_energy": cumulative_energy,
        }
    )
    return segments


def draw_map(
    axis: plt.Axes,
    depot: dict,
    points: dict[int, dict],
    route: RouteResult,
    segments: list[Segment],
    color: str,
) -> None:
    """绘制单机路线，并在停靠点标注货物和载重参数。"""
    map_image = Image.open(MAP_PATH).convert("RGB")
    map_image = ImageEnhance.Color(map_image).enhance(0.38)
    map_image = ImageEnhance.Contrast(map_image).enhance(0.68)
    map_image = ImageEnhance.Brightness(map_image).enhance(1.20)
    axis.imshow(
        np.asarray(map_image),
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="upper",
        alpha=0.84,
        zorder=0,
    )
    axis.imshow(
        np.full((2, 2, 4), (1.0, 1.0, 1.0, 0.42)),
        extent=(0, map_image.width - 1, 0, map_image.height - 1),
        origin="lower",
        zorder=1,
    )

    delivery_points = [point for point in points.values() if point.get("color") == "green"]
    axis.scatter(
        [point["x_math"] for point in delivery_points],
        [point["y_math"] for point in delivery_points],
        s=30,
        color="#cbd5e1",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.78,
        zorder=2,
    )

    for start_id, end_id in zip(route.point_ids, route.point_ids[1:]):
        if start_id == end_id:
            continue
        start, end = points[start_id], points[end_id]
        axis.add_patch(
            FancyArrowPatch(
                (start["x_math"], start["y_math"]),
                (end["x_math"], end["y_math"]),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=2.35,
                color=color,
                alpha=0.92,
                shrinkA=8,
                shrinkB=8,
                connectionstyle="arc3,rad=0.012",
                zorder=4,
            )
        )

    used_ids = [
        point_id
        for index, point_id in enumerate(route.point_ids[1:-1])
        if index == 0 or point_id != route.point_ids[index]
    ]
    used_points = [points[point_id] for point_id in used_ids]
    axis.scatter(
        [point["x_math"] for point in used_points],
        [point["y_math"] for point in used_points],
        s=82,
        color=color,
        edgecolor="white",
        linewidth=1.8,
        zorder=6,
    )
    occupied_labels: list[tuple[float, float, float, float]] = []
    labels = []
    target_x = []
    target_y = []
    candidate_offsets = (
        (12, 12, "left", "bottom"),
        (-12, 12, "right", "bottom"),
        (12, -12, "left", "top"),
        (-12, -12, "right", "top"),
        (24, 24, "left", "bottom"),
        (-24, 24, "right", "bottom"),
        (24, -24, "left", "top"),
        (-24, -24, "right", "top"),
        (40, 0, "left", "center"),
        (-40, 0, "right", "center"),
    )

    for order, (point, segment) in enumerate(zip(used_points, segments[:-1]), start=1):
        package_ids = ", ".join(segment["package_ids"])
        label = (
            f"{order}. 点 {int(segment['end'])}\n"
            f"包裹 {package_ids} / {float(segment['package_weight']):.2f} kg\n"
            f"载重 {float(segment['load_before']):.2f}→{float(segment['load_after']):.2f} kg\n"
            f"W={float(segment['load_factor']):.3f}  燃料消耗={float(segment['segment_energy']):.2f}"
        )
        label_width = 82.0
        label_height = 39.0
        label_x = float(point["x_math"])
        label_y = float(point["y_math"])
        horizontal_alignment = "left"
        vertical_alignment = "bottom"

        for dx, dy, horizontal, vertical in candidate_offsets:
            candidate_x = float(point["x_math"]) + dx
            candidate_y = float(point["y_math"]) + dy
            left = candidate_x if horizontal == "left" else candidate_x - label_width
            bottom = (
                candidate_y
                if vertical == "bottom"
                else candidate_y - label_height
                if vertical == "top"
                else candidate_y - label_height / 2
            )
            box = (left, bottom, left + label_width, bottom + label_height)
            inside_map = box[0] >= 3 and box[1] >= 3 and box[2] <= map_image.width - 4 and box[3] <= map_image.height - 4
            overlaps = any(
                box[0] < used_box[2]
                and box[2] > used_box[0]
                and box[1] < used_box[3]
                and box[3] > used_box[1]
                for used_box in occupied_labels
            )
            if inside_map and not overlaps:
                label_x = candidate_x
                label_y = candidate_y
                horizontal_alignment = horizontal
                vertical_alignment = vertical
                occupied_labels.append(box)
                break

        annotation = axis.annotate(
            label,
            (point["x_math"], point["y_math"]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=7.4,
            color="#334155",
            ha=horizontal_alignment,
            va=vertical_alignment,
            linespacing=1.28,
            bbox={
                "boxstyle": "round,pad=0.42,rounding_size=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.92,
                "linewidth": 0.9,
            },
            zorder=10,
        )
        labels.append(annotation)
        target_x.append(float(point["x_math"]))
        target_y.append(float(point["y_math"]))

    adjust_text(
        labels,
        target_x=target_x,
        target_y=target_y,
        ax=axis,
        ensure_inside_axes=True,
        expand=(1.12, 1.18),
        force_text=(0.8, 1.0),
        force_static=(0.3, 0.4),
        force_pull=(0.04, 0.04),
        max_move=(18, 24),
        iter_lim=500,
        prevent_crossings=True,
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.8},
    )

    axis.scatter(
        [depot["x_math"]],
        [depot["y_math"]],
        s=230,
        marker="D",
        color="#fb923c",
        edgecolor="white",
        linewidth=2.3,
        zorder=8,
    )
    axis.text(
        depot["x_math"] + 8,
        depot["y_math"] + 8,
        "仓库",
        color="#9a3412",
        fontsize=10,
        fontweight="bold",
        zorder=9,
    )
    axis.set_xlim(0, map_image.width - 1)
    axis.set_ylim(0, map_image.height - 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("横向坐标 / 像素", color="#64748b")
    axis.set_ylabel("纵向坐标 / 像素", color="#64748b")
    axis.tick_params(colors="#94a3b8", labelsize=8)
    for spine in axis.spines.values():
        spine.set_color("#e2e8f0")


def draw_load_chart(
    axis: plt.Axes,
    segments: list[Segment],
    color: str,
    max_payload: float,
) -> None:
    """绘制每次配送前后的剩余载重阶梯变化。"""
    deliveries = segments[:-1]
    before = [float(segment["load_before"]) for segment in deliveries]
    after = [float(segment["load_after"]) for segment in deliveries]
    x = np.arange(len(deliveries))
    axis.step(
        np.arange(len(deliveries) + 1),
        before + [after[-1]],
        where="post",
        color=color,
        linewidth=2.4,
    )
    axis.scatter(x, before, color=color, s=40, edgecolor="white", linewidth=1.1, zorder=4, label="停靠前")
    axis.scatter(x + 1, after, color="#94a3b8", s=34, edgecolor="white", linewidth=1.0, zorder=4, label="停靠后")
    axis.axhline(max_payload, color="#ef4444", linestyle="--", linewidth=1.1, alpha=0.65, label="载重上限")
    axis.fill_between(np.arange(len(deliveries) + 1), before + [after[-1]], step="post", color=color, alpha=0.10)
    axis.set_xlim(-0.3, len(deliveries) + 0.3)
    axis.set_ylim(0, max_payload * 1.13)
    axis.set_xticks(np.arange(len(deliveries) + 1))
    axis.set_xticklabels(["起飞"] + [f"停靠 {index}" for index in range(1, len(deliveries) + 1)], fontsize=8)
    axis.set_ylabel("剩余载重 / kg", color="#526777")
    axis.grid(axis="y", color="#e7edf4", linewidth=0.75)
    axis.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axis.tick_params(colors="#64748b", labelsize=8)
    for spine in (axis.spines["top"], axis.spines["right"]):
        spine.set_visible(False)
    for spine in (axis.spines["left"], axis.spines["bottom"]):
        spine.set_color("#dbe4ee")


def draw_segment_table(
    axis: plt.Axes,
    segments: list[Segment],
) -> None:
    """绘制各航段参数变化明细表。"""
    axis.axis("off")
    columns = ["航段", "投递包裹 / 总重", "载重变化", "W(t)", "段能耗", "累计"]
    rows = []
    for index, segment in enumerate(segments, start=1):
        package_ids = segment["package_ids"]
        task = "空载返仓" if not package_ids else (
            f"{', '.join(package_ids)} / {float(segment['package_weight']):.2f}kg"
        )
        rows.append(
            [
                f"{segment['start']}→{segment['end']}",
                task,
                f"{float(segment['load_before']):.2f}→{float(segment['load_after']):.2f}",
                f"{float(segment['load_factor']):.3f}",
                f"{float(segment['segment_energy']):.3f}",
                f"{float(segment['cumulative_energy']):.3f}",
            ]
        )
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.13, 0.22, 0.18, 0.12, 0.15, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(color="#334155", fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")
            cell.set_text_props(color="#526777")


def create_drone_image(
    drone_index: int,
    depot: dict,
    points: dict[int, dict],
    route: RouteResult,
    segments: list[Segment],
    max_payload: float,
) -> None:
    """生成仅包含地图、路线和点位货物参数的单机详情图。"""
    color = ROUTE_COLORS[drone_index - 1]
    figure = plt.figure(figsize=(11, 11), dpi=180, facecolor="#ffffff")
    header = figure.add_axes((0.06, 0.91, 0.88, 0.065))
    map_axis = figure.add_axes((0.08, 0.07, 0.84, 0.80))

    header.axis("off")
    stop_count = len(segments) - 1
    header.text(
        0.02,
        0.72,
        f"无人机 {drone_index} · 配送路线与货物状态",
        transform=header.transAxes,
        fontsize=20,
        fontweight="bold",
        color="#0f2942",
        va="center",
    )
    header.text(
        0.02,
        0.18,
        f"η={TARGET_ETA:.2f}  ·  初始载重 {route.initial_load:.2f}/{max_payload:.0f} kg  ·  "
        f"{len(route.package_ids)} 件包裹 / {stop_count} 个停靠点  ·  总燃料消耗 {route.energy:.2f}",
        transform=header.transAxes,
        fontsize=10.5,
        color="#64748b",
        va="center",
    )
    header.plot([0, 1], [0.02, 0.02], transform=header.transAxes, color="#dbe4ee", linewidth=1)

    draw_map(map_axis, depot, points, route, segments, color)
    map_axis.text(
        0.025,
        0.025,
        "灰色节点为本机未使用的配送点\n点位标签依次显示包裹、载重变化、W(t) 与航段燃料消耗",
        transform=map_axis.transAxes,
        fontsize=8.5,
        color="#64748b",
        linespacing=1.5,
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.2",
            "facecolor": "white",
            "edgecolor": "#dbe4ee",
            "alpha": 0.92,
        },
        zorder=12,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"drone_{drone_index:02d}.png"
    figure.savefig(output_path, facecolor="#ffffff")
    plt.close(figure)


def main() -> None:
    """为 eta=0.30 的全部无人机生成独立详情图。"""
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    max_payload, _total_energy, routes = load_eta_result(RESULTS_PATH)
    depot, points = load_points(POINTS_PATH)
    package_weights = package_weights_by_id(points)

    for drone_index, route in enumerate(routes, start=1):
        segments = calculate_segments(
            route, points, package_weights, max_payload
        )
        create_drone_image(
            drone_index, depot, points, route, segments, max_payload
        )
    print(f"Saved {len(routes)} drone detail images to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
