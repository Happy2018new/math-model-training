"""问题三：考虑剩余载重的多无人机等效燃料消耗路径规划。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ..problem_two.resolve import (
        DEPARTURE_TIME,
        INPUT_POINT_PATH,
        Point,
        build_distance_lookup,
        load_points,
        load_weather_snapshot,
    )
except ImportError:
    import importlib.util
    import sys

    problem_two_path = Path(__file__).resolve().parents[1] / "problem_two" / "resolve.py"
    module_spec = importlib.util.spec_from_file_location("problem_two_resolve", problem_two_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"无法加载问题二求解器: {problem_two_path}")
    problem_two_resolve = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = problem_two_resolve
    module_spec.loader.exec_module(problem_two_resolve)
    DEPARTURE_TIME = problem_two_resolve.DEPARTURE_TIME
    INPUT_POINT_PATH = problem_two_resolve.INPUT_POINT_PATH
    Point = problem_two_resolve.Point
    build_distance_lookup = problem_two_resolve.build_distance_lookup
    load_points = problem_two_resolve.load_points
    load_weather_snapshot = problem_two_resolve.load_weather_snapshot


EPSILON = 1e-12
DEFAULT_MAX_PAYLOAD = 10.0
DEFAULT_ETA = 0.3
PACKAGE_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "problems"
    / "three"
    / "points_with_packages.json"
)
BaseCostLookup = dict[tuple[int, int], float]


@dataclass(frozen=True)
class PackageTask:
    """一个不可拆分包裹及其配送地点。"""

    package_id: str
    point: Point
    weight: float


@dataclass
class DroneRoute:
    """一架无人机的配送任务顺序及其等效燃料消耗。"""

    tasks: list[PackageTask]
    energy: float = 0.0

    @property
    def initial_load(self) -> float:
        return sum(task.weight for task in self.tasks)


def create_package_tasks(delivery_points: list[Point]) -> list[PackageTask]:
    """将各点的 packages 重量列表展开为独立包裹任务。"""
    tasks: list[PackageTask] = []
    for point in delivery_points:
        packages = point.get("packages")
        if not isinstance(packages, list):
            raise ValueError(f"配送点 {point.get('id')} 缺少 packages 列表。")
        for package_index, weight in enumerate(packages, start=1):
            numeric_weight = float(weight)
            if numeric_weight <= 0:
                raise ValueError(f"配送点 {point.get('id')} 包含非正包裹重量。")
            tasks.append(
                PackageTask(
                    package_id=f"{point['id']}-{package_index}",
                    point=point,
                    weight=numeric_weight,
                )
            )
    return tasks


def route_energy(
    depot: Point,
    tasks: list[PackageTask],
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> float:
    """按每段起飞前的剩余载重计算一条完整路线的等效燃料消耗。"""
    if not tasks:
        return 0.0

    remaining_load = sum(task.weight for task in tasks)
    if remaining_load > max_payload + EPSILON:
        return float("inf")

    energy = 0.0
    previous_point = depot
    for task in tasks:
        load_factor = 1.0 + eta * remaining_load / max_payload
        energy += base_costs[previous_point["id"], task.point["id"]] * load_factor
        remaining_load -= task.weight
        previous_point = task.point

    energy += base_costs[previous_point["id"], depot["id"]]
    return energy


def build_initial_routes(
    depot: Point,
    package_tasks: list[PackageTask],
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> list[DroneRoute]:
    """按重量降序，使用最小燃料消耗增量插入法构造初始方案。"""
    routes: list[DroneRoute] = []
    for task in sorted(package_tasks, key=lambda item: item.weight, reverse=True):
        if task.weight > max_payload + EPSILON:
            raise ValueError(f"包裹 {task.package_id} 重量超过单架无人机最大载重。")

        best_delta = float("inf")
        best_route_index: int | None = None
        best_position = 0
        for route_index, route in enumerate(routes):
            if route.initial_load + task.weight > max_payload + EPSILON:
                continue
            for position in range(len(route.tasks) + 1):
                candidate = route.tasks[:position] + [task] + route.tasks[position:]
                candidate_energy = route_energy(
                    depot, candidate, base_costs, max_payload, eta
                )
                delta = candidate_energy - route.energy
                if delta < best_delta - EPSILON:
                    best_delta = delta
                    best_route_index = route_index
                    best_position = position

        new_route_energy = route_energy(
            depot, [task], base_costs, max_payload, eta
        )
        if new_route_energy < best_delta - EPSILON or best_route_index is None:
            routes.append(DroneRoute([task], new_route_energy))
        else:
            route = routes[best_route_index]
            route.tasks.insert(best_position, task)
            route.energy += best_delta
    return routes


def improve_route_order(
    depot: Point,
    route: DroneRoute,
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> bool:
    """通过任意两包裹交换和单包裹重插改善一条路线。"""
    for first_index in range(len(route.tasks) - 1):
        for second_index in range(first_index + 1, len(route.tasks)):
            candidate = route.tasks.copy()
            candidate[first_index], candidate[second_index] = (
                candidate[second_index],
                candidate[first_index],
            )
            candidate_energy = route_energy(
                depot, candidate, base_costs, max_payload, eta
            )
            if candidate_energy < route.energy - EPSILON:
                route.tasks = candidate
                route.energy = candidate_energy
                return True

    for source_index in range(len(route.tasks)):
        task = route.tasks[source_index]
        without_task = route.tasks[:source_index] + route.tasks[source_index + 1 :]
        for target_index in range(len(without_task) + 1):
            candidate = without_task[:target_index] + [task] + without_task[target_index:]
            if candidate == route.tasks:
                continue
            candidate_energy = route_energy(
                depot, candidate, base_costs, max_payload, eta
            )
            if candidate_energy < route.energy - EPSILON:
                route.tasks = candidate
                route.energy = candidate_energy
                return True
    return False


def optimize_task_order(
    depot: Point,
    tasks: list[PackageTask],
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> DroneRoute:
    """使用路线内交换和重插，优化给定任务集合直至局部最优。"""
    route = DroneRoute(
        tasks.copy(),
        route_energy(depot, tasks, base_costs, max_payload, eta),
    )
    while improve_route_order(
        depot, route, base_costs, max_payload, eta
    ):
        pass
    return route


def relocate_package(
    depot: Point,
    routes: list[DroneRoute],
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> bool:
    """尝试将一个包裹移至另一架无人机并接受首次改进。"""
    for source_index, source in enumerate(routes):
        for task_index, task in enumerate(source.tasks):
            for target_index, target in enumerate(routes):
                if source_index == target_index or target.initial_load + task.weight > max_payload + EPSILON:
                    continue
                for position in range(len(target.tasks) + 1):
                    source_candidate = source.tasks[:task_index] + source.tasks[task_index + 1 :]
                    target_candidate = target.tasks[:position] + [task] + target.tasks[position:]
                    new_source_energy = route_energy(
                        depot, source_candidate, base_costs, max_payload, eta
                    )
                    new_target_energy = route_energy(
                        depot, target_candidate, base_costs, max_payload, eta
                    )
                    if new_source_energy + new_target_energy < source.energy + target.energy - EPSILON:
                        source.tasks = source_candidate
                        source.energy = new_source_energy
                        target.tasks = target_candidate
                        target.energy = new_target_energy
                        if not source.tasks:
                            routes.pop(source_index)
                        return True
    return False


def swap_packages(
    depot: Point,
    routes: list[DroneRoute],
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> bool:
    """尝试交换两架无人机的包裹并接受首次改进。"""
    for first_index in range(len(routes) - 1):
        for second_index in range(first_index + 1, len(routes)):
            first = routes[first_index]
            second = routes[second_index]
            for first_task_index, first_task in enumerate(first.tasks):
                for second_task_index, second_task in enumerate(second.tasks):
                    first_load = first.initial_load - first_task.weight + second_task.weight
                    second_load = second.initial_load - second_task.weight + first_task.weight
                    if first_load > max_payload + EPSILON or second_load > max_payload + EPSILON:
                        continue
                    first_candidate = first.tasks.copy()
                    second_candidate = second.tasks.copy()
                    first_candidate[first_task_index] = second_task
                    second_candidate[second_task_index] = first_task
                    optimized_first = optimize_task_order(
                        depot,
                        first_candidate,
                        base_costs,
                        max_payload,
                        eta,
                    )
                    optimized_second = optimize_task_order(
                        depot,
                        second_candidate,
                        base_costs,
                        max_payload,
                        eta,
                    )
                    if (
                        optimized_first.energy + optimized_second.energy
                        < first.energy + second.energy - EPSILON
                    ):
                        first.tasks = optimized_first.tasks
                        first.energy = optimized_first.energy
                        second.tasks = optimized_second.tasks
                        second.energy = optimized_second.energy
                        return True
    return False


def solve_payload_routes(
    depot: Point,
    delivery_points: list[Point],
    max_payload: float,
    eta: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[DroneRoute]:
    """求解包裹分配、访问顺序和无人机数量不受限的配送方案。"""
    if max_payload <= 0 or eta < 0:
        raise ValueError("max_payload 必须为正，eta 不得为负。")

    package_tasks = create_package_tasks(delivery_points)
    weather = load_weather_snapshot(DEPARTURE_TIME)
    base_costs = build_distance_lookup([depot] + delivery_points, weather, alpha, beta)
    routes = build_initial_routes(
        depot, package_tasks, base_costs, max_payload, eta
    )

    while True:
        improved = any(
            improve_route_order(
                depot, route, base_costs, max_payload, eta
            )
            for route in routes
        )
        if improved:
            continue
        if relocate_package(
            depot, routes, base_costs, max_payload, eta
        ):
            continue
        if swap_packages(
            depot, routes, base_costs, max_payload, eta
        ):
            continue
        return routes


def route_segments(
    depot: Point,
    route: DroneRoute,
    base_costs: BaseCostLookup,
    max_payload: float,
    eta: float,
) -> list[dict[str, Any]]:
    """返回一条路线各航段的剩余载重与等效燃料消耗明细。"""
    segments: list[dict[str, Any]] = []
    remaining_load = route.initial_load
    previous = depot
    for task in route.tasks:
        energy = base_costs[previous["id"], task.point["id"]] * (
            1.0 + eta * remaining_load / max_payload
        )
        segments.append(
            {
                "from": previous["id"],
                "to": task.point["id"],
                "package_id": task.package_id,
                "remaining_load": remaining_load,
                "energy": energy,
            }
        )
        remaining_load -= task.weight
        previous = task.point
    segments.append(
        {
            "from": previous["id"],
            "to": depot["id"],
            "package_id": None,
            "remaining_load": 0.0,
            "energy": base_costs[previous["id"], depot["id"]],
        }
    )
    return segments


def print_solution(
    depot: Point,
    delivery_points: list[Point],
    routes: list[DroneRoute],
    max_payload: float,
    eta: float,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> None:
    """输出各无人机路线、载重、航段燃料消耗及系统总燃料消耗。"""
    weather = load_weather_snapshot(DEPARTURE_TIME)
    base_costs = build_distance_lookup(
        [depot] + delivery_points, weather, alpha, beta
    )
    print(f"无人机数量: {len(routes)}")
    print(f"最大载重 Q: {max_payload:.3f} kg, 载重参数 eta: {eta:.3f}")

    for route_index, route in enumerate(routes, start=1):
        point_ids = [depot["id"]] + [task.point["id"] for task in route.tasks] + [depot["id"]]
        print(f"\n无人机 {route_index}")
        print(f"  路线: {' -> '.join(map(str, point_ids))}")
        print(f"  初始装载重量: {route.initial_load:.3f} kg")
        print(
            "  包裹: "
            + ", ".join(
                f"{task.package_id}({task.weight:.3f} kg)" for task in route.tasks
            )
        )
        for segment_index, segment in enumerate(
            route_segments(depot, route, base_costs, max_payload, eta), start=1
        ):
            package = segment["package_id"] or "返仓"
            print(
                f"  航段 {segment_index}: {segment['from']} -> {segment['to']}, "
                f"任务={package}, 剩余载重={segment['remaining_load']:.3f} kg, "
                f"等效燃料消耗={segment['energy']:.6f}"
            )
        print(f"  单机总等效燃料消耗: {route.energy:.6f}")

    print(f"\n系统总等效燃料消耗: {sum(route.energy for route in routes):.6f}")


def main() -> None:
    """从 points.json 读取 packages 接口数据并运行问题三求解器。"""
    depot, delivery_points = load_points(PACKAGE_INPUT_PATH)
    routes = solve_payload_routes(
        depot,
        delivery_points,
        max_payload=DEFAULT_MAX_PAYLOAD,
        eta=DEFAULT_ETA,
    )
    print_solution(
        depot,
        delivery_points,
        routes,
        max_payload=DEFAULT_MAX_PAYLOAD,
        eta=DEFAULT_ETA,
    )


if __name__ == "__main__":
    main()
