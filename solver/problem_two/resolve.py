"""问题二：基于极角分组和跨组交换的多无人机路径规划。"""

import csv
import json
import math
from itertools import permutations
from pathlib import Path
from typing import Any

Point = dict[str, Any]
Route = list[Point]
Group = list[Point]
GroupSolution = tuple[Route, float]
DistanceLookup = dict[tuple[int, int], float]
WeatherSnapshot = tuple[float, float, float]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_POINT_PATH = PROJECT_ROOT / "output" / "processed" / "points.json"
HOURLY_WEATHER_PATH = PROJECT_ROOT / "output" / "weather" / "paris_hourly_weather_2016-05-01_to_2026-05-01.csv"
DEPARTURE_TIME = "2025-05-01 08:00:00+00:00"
WEATHER_CODE_OVERRIDE = 95
MAX_DELIVERIES_PER_DRONE = 7
EPSILON = 1e-12
MAP_SCALE_KM_PER_PIXEL = 0.040
DRONE_AIRSPEED_KMH = 50.0

WEATHER_COEFFICIENTS = {
    0: 1.00,
    1: 1.01,
    2: 1.01,
    3: 1.01,
    45: 1.08,
    48: 1.08,
    51: 1.06,
    53: 1.06,
    55: 1.06,
    56: 1.18,
    57: 1.18,
    61: 1.10,
    63: 1.10,
    65: 1.10,
    66: 1.18,
    67: 1.18,
    71: 1.15,
    73: 1.15,
    75: 1.15,
    77: 1.15,
    80: 1.12,
    81: 1.12,
    82: 1.12,
    85: 1.18,
    86: 1.18,
    95: 1.30,
    96: 1.50,
    99: 1.50,
}


def distance(a: Point, b: Point) -> float:
    """返回两点在地图坐标中的欧氏距离，单位为像素。"""
    return math.hypot(a["x_math"] - b["x_math"], a["y_math"] - b["y_math"])


def weather_coefficient(weather_code: int) -> float:
    """按 WMO 天气代码返回非定向天气修正系数 μ。"""
    return WEATHER_COEFFICIENTS.get(weather_code, 1.00)


def load_weather_snapshot(departure_time: str) -> WeatherSnapshot:
    """读取任务出发时刻的风数据和雷暴压力测试的天气修正系数。"""
    with HOURLY_WEATHER_PATH.open(encoding="utf-8-sig", newline="") as file:
        hourly_rows = csv.DictReader(file)
        hourly = next((row for row in hourly_rows if row["date"] == departure_time), None)
    if hourly is None:
        raise ValueError(f"小时天气数据中不存在出发时刻 {departure_time}。")

    wind_speed = float(hourly["wind_speed_100m"])
    wind_from_direction = float(hourly["wind_direction_100m"])
    weather_mu = weather_coefficient(WEATHER_CODE_OVERRIDE)
    return wind_speed, wind_from_direction, weather_mu


def wind_coefficient(a: Point, b: Point, wind_speed: float, wind_from_direction: float) -> float:
    """按航向和气象风向计算风向修正系数 k。"""
    flight_direction = math.atan2(b["y_math"] - a["y_math"], b["x_math"] - a["x_math"])
    wind_to_direction = math.radians((wind_from_direction + 180.0) % 360.0)
    angle = wind_to_direction - flight_direction
    crosswind_speed = wind_speed * math.sin(angle)
    along_track_speed = wind_speed * math.cos(angle)
    ground_speed = math.sqrt(DRONE_AIRSPEED_KMH**2 - crosswind_speed**2) + along_track_speed
    if ground_speed <= EPSILON:
        raise ValueError("当前风况下无人机无法维持该航向的正地速。")
    return DRONE_AIRSPEED_KMH / ground_speed


def build_distance_lookup(
    points: list[Point], weather: WeatherSnapshot, alpha: float = 1.0, beta: float = 1.0
) -> DistanceLookup:
    """预计算 f(i,j)=L*d(i,j)*μ'*k' 的有向航段修正距离。"""
    wind_speed, wind_from_direction, weather_mu = weather
    adjusted_mu = 1.0 + alpha * (weather_mu - 1.0)
    return {
        (start["id"], end["id"]): (
            MAP_SCALE_KM_PER_PIXEL
            * distance(start, end)
            * adjusted_mu
            * (
                1.0
                + beta
                * (wind_coefficient(start, end, wind_speed, wind_from_direction) - 1.0)
            )
        )
        for start in points
        for end in points
    }


def corrected_route_cost(depot: Point, route: Route, costs: DistanceLookup) -> float:
    """计算闭环路线的修正航程，即逐段累加 f(i,j)。"""
    if len(route) == 0:
        return 0.0
    return (
        costs[depot["id"], route[0]["id"]]
        + sum(
            costs[start["id"], end["id"]] for start, end in zip(route, route[1:])
        )
        + costs[route[-1]["id"], depot["id"]]
    )


def exact_group_tsp(
    depot: Point,
    group: Group,
    distances: DistanceLookup,
    cache: dict[tuple[int, ...], GroupSolution],
) -> GroupSolution:
    """通过惰性全排列枚举，求一个最多 7 点分组的最优闭环路线。"""
    group_key = tuple(sorted(point["id"] for point in group))
    if group_key in cache:
        return cache[group_key]

    best_route: Route = []
    best_length = math.inf

    for visit_order in permutations(group):
        route = list(visit_order)
        total_length = corrected_route_cost(depot, route, distances)
        if total_length < best_length:
            best_route = route
            best_length = total_length

    cache[group_key] = best_route, best_length
    return cache[group_key]


def polar_angle(depot: Point, point: Point) -> float:
    """计算送货点相对于仓库的平面极角。"""
    return math.atan2(
        point["y_math"] - depot["y_math"], point["x_math"] - depot["x_math"]
    )


def split_by_angle(
    depot: Point, deliveries: list[Point], start_index: int
) -> list[Group]:
    """从指定极角位置开始顺时针扫描，将点按容量切分为多个组。"""
    ordered = sorted(
        deliveries, key=lambda point: polar_angle(depot, point), reverse=True
    )
    rotated = ordered[start_index:] + ordered[:start_index]
    return [
        rotated[index : index + MAX_DELIVERIES_PER_DRONE]
        for index in range(0, len(rotated), MAX_DELIVERIES_PER_DRONE)
    ]


def solve_groups(
    depot: Point,
    groups: list[Group],
    distances: DistanceLookup,
    cache: dict[tuple[int, ...], GroupSolution],
) -> list[GroupSolution]:
    """求每个分组的最优闭环路线。"""
    return [exact_group_tsp(depot, group, distances, cache) for group in groups]


def total_distance(solutions: list[GroupSolution]) -> float:
    """计算所有无人机路线的总距离。"""
    return sum(length for _route, length in solutions)


def improve_by_swapping(
    depot: Point,
    groups: list[Group],
    distances: DistanceLookup,
    cache: dict[tuple[int, ...], GroupSolution],
) -> list[GroupSolution]:
    """反复接受可降低总距离的跨组单点交换，直到无法改进。"""
    solutions = solve_groups(depot, groups, distances, cache)

    while True:
        improved = False
        for first_group_index in range(len(groups) - 1):
            for second_group_index in range(first_group_index + 1, len(groups)):
                first_group = groups[first_group_index]
                second_group = groups[second_group_index]
                old_length = (
                    solutions[first_group_index][1] + solutions[second_group_index][1]
                )

                for first_point_index in range(len(first_group)):
                    for second_point_index in range(len(second_group)):
                        candidate_first = first_group.copy()
                        candidate_second = second_group.copy()
                        (
                            candidate_first[first_point_index],
                            candidate_second[second_point_index],
                        ) = (
                            candidate_second[second_point_index],
                            candidate_first[first_point_index],
                        )
                        candidate_first_solution = exact_group_tsp(
                            depot, candidate_first, distances, cache
                        )
                        candidate_second_solution = exact_group_tsp(
                            depot, candidate_second, distances, cache
                        )
                        candidate_length = (
                            candidate_first_solution[1] + candidate_second_solution[1]
                        )

                        if candidate_length < old_length - EPSILON:
                            groups[first_group_index] = candidate_first
                            groups[second_group_index] = candidate_second
                            solutions[first_group_index] = candidate_first_solution
                            solutions[second_group_index] = candidate_second_solution
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return solutions


def load_points(points_path: Path) -> tuple[Point, list[Point]]:
    """加载唯一蓝色仓库点和所有绿色送货点。"""
    with points_path.open(encoding="utf-8") as file:
        points: list[Point] = json.load(file)

    depots = [point for point in points if point.get("color") == "blue"]
    deliveries = [point for point in points if point.get("color") == "green"]
    if len(depots) != 1:
        raise ValueError(f"应存在唯一的蓝色仓库点，实际数量为 {len(depots)}。")
    if not deliveries:
        raise ValueError("未找到绿色送货点。")
    return depots[0], deliveries


def solve_multi_drone(
    depot: Point, deliveries: list[Point], alpha: float = 1.0, beta: float = 1.0
) -> list[GroupSolution]:
    """比较每个起始极角下的局部最优解，返回其中总距离最短的解。"""
    best_solutions: list[GroupSolution] = []
    best_length = math.inf
    weather = load_weather_snapshot(DEPARTURE_TIME)
    distances = build_distance_lookup([depot] + deliveries, weather, alpha, beta)
    cache: dict[tuple[int, ...], GroupSolution] = {}

    for start_index in range(len(deliveries)):
        groups = split_by_angle(depot, deliveries, start_index)
        local_solutions = improve_by_swapping(depot, groups, distances, cache)
        local_length = total_distance(local_solutions)
        if local_length < best_length - EPSILON:
            best_solutions = local_solutions
            best_length = local_length

    return best_solutions


def main() -> None:
    """求解问题二并输出各架无人机的最优路线。"""
    depot, deliveries = load_points(INPUT_POINT_PATH)
    wind_speed, wind_from_direction, weather_mu = load_weather_snapshot(DEPARTURE_TIME)
    solutions = solve_multi_drone(depot, deliveries)

    print(f"送货点数量: {len(deliveries)}")
    print(f"比例尺 L: {MAP_SCALE_KM_PER_PIXEL:.3f} km/像素")
    print(f"无人机空速 v_a: {DRONE_AIRSPEED_KMH:.1f} km/h")
    print(f"天气快照: {DEPARTURE_TIME}")
    print(
        f"风速: {wind_speed:.3f} km/h, 风向: {wind_from_direction:.3f}°, "
        f"假设天气代码: {WEATHER_CODE_OVERRIDE}, μ: {weather_mu:.2f}"
    )
    print(f"无人机数量: {len(solutions)}")
    for index, (route, length) in enumerate(solutions, start=1):
        route_ids = [depot["id"]] + [point["id"] for point in route] + [depot["id"]]
        print(f"无人机 {index}: {' -> '.join(map(str, route_ids))}")
        print(f"  配送点数: {len(route)}, 修正航程: {length:.4f} km")
    print(f"总修正航程: {total_distance(solutions):.4f} km")


if __name__ == "__main__":
    main()
