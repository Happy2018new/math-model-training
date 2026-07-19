"""
问题一解决方案

(1) n<=11 时考虑暴力求解;
(2) n>11 时考虑先使用最近邻算法求出粗略近似解，然后再对该解执行 2-opt 进行调优
"""

import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any

Point = dict[str, Any]
DistanceMatrix = list[list[float]]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_POINT_PATH = PROJECT_ROOT / "output" / "processed" / "points.json"
EXACT_THRESHOLD = 11
EPSILON = 1e-12


def distance(a: Point, b: Point) -> float:
    """返回点 A 到点 B 之间的直线距离"""
    delta_x = abs(a["x_math"] - b["x_math"])
    delta_z = abs(a["y_math"] - b["y_math"])
    return math.sqrt(delta_x**2 + delta_z**2)


def build_distance_matrix(points: list[Point]) -> DistanceMatrix:
    """预先计算点对距离，避免在全排列中重复计算"""
    return [[distance(point, other) for other in points] for point in points]


def route_length(
    route: tuple[int, ...] | list[int], distances: DistanceMatrix
) -> float:
    """计算给定路线的总长度"""
    return sum(distances[start][end] for start, end in zip(route, route[1:]))


def exact_tsp(distances: DistanceMatrix) -> tuple[list[int], float]:
    """这是我们的暴力求解算法，它进行全排列枚举"""
    count = len(distances)
    if count == 1:
        return [0, 0], 0.0

    best_route: list[int] = []
    best_length = math.inf

    for visit_order in permutations(range(1, count)):
        route = (0, *visit_order, 0)
        total_length = route_length(route, distances)
        if total_length < best_length:
            best_length = total_length
            best_route = list(route)

    return best_route, best_length


def nearest_neighbour_tsp(distances: DistanceMatrix) -> list[int]:
    """使用最近邻算法生成初始解"""
    current = 0
    route = [current]
    unvisited = set(range(1, len(distances)))

    while unvisited:
        nearest_node = min(unvisited, key=lambda node: distances[current][node])
        route.append(nearest_node)
        unvisited.remove(nearest_node)
        current = nearest_node

    route.append(0)
    return route


def two_opt(route: list[int], distances: DistanceMatrix) -> list[int]:
    """使用 Two opt 算法优化初始解"""
    improved = True
    while improved:
        improved = False
        for start in range(1, len(route) - 2):
            for end in range(start + 1, len(route) - 1):
                before = (
                    distances[route[start - 1]][route[start]]
                    + distances[route[end]][route[end + 1]]
                )
                after = (
                    distances[route[start - 1]][route[end]]
                    + distances[route[start]][route[end + 1]]
                )
                if after < before - EPSILON:
                    route[start : end + 1] = reversed(route[start : end + 1])
                    improved = True
                    break
            if improved:
                break
    return route


def solve_tsp(depot: Point, deliveries: list[Point]) -> tuple[list[int], float]:
    """相当于是入口函数（求解的入口函数）"""
    points = [depot] + deliveries
    distances = build_distance_matrix(points)

    if len(deliveries) <= EXACT_THRESHOLD:
        route, total_length = exact_tsp(distances)
    else:
        route = two_opt(nearest_neighbour_tsp(distances), distances)
        total_length = route_length(route, distances)

    return route, total_length


def load_points(points_path: Path) -> tuple[Point, list[Point]]:
    """加载点，分别返回起点（蓝色点）和送货点（绿色点）"""
    with points_path.open(encoding="utf-8") as file:
        points: list[Point] = json.load(file)

    depot = [point for point in points if point.get("color") == "blue"]
    deliveries = [point for point in points if point.get("color") == "green"]

    return depot[0], deliveries


def main() -> None:
    depot, deliveries = load_points(INPUT_POINT_PATH)  # 加载点
    route, total_length = solve_tsp(depot, deliveries)  # 求解最佳路径

    all_points = [depot] + deliveries
    route_ids = [all_points[index]["id"] for index in route]

    print(f"Delivery locations (n): {len(deliveries)}")
    print(f"Route (point IDs): {' -> '.join(map(str, route_ids))}")
    print(f"Total distance: {total_length:.4f}")


if __name__ == "__main__":
    main()
