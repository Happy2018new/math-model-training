"""并行计算天气系数与风向系数的 25 个敏感性分析场景。"""

from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

try:
    from .resolve import (
        INPUT_POINT_PATH,
        PROJECT_ROOT,
        load_points,
        solve_multi_drone,
        total_distance,
    )
except ImportError:
    from resolve import (
        INPUT_POINT_PATH,
        PROJECT_ROOT,
        load_points,
        solve_multi_drone,
        total_distance,
    )


SCENARIO_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0)
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity_results.txt"


def solve_scenario(
    scenario: tuple[float, float],
) -> tuple[float, float, float, list[list[int]], list[float]]:
    """计算一个 (alpha, beta) 场景并返回用于保存的结果。"""
    alpha, beta = scenario
    depot, deliveries = load_points(INPUT_POINT_PATH)
    solutions = solve_multi_drone(depot, deliveries, alpha, beta)
    routes = [[point["id"] for point in route] for route, _length in solutions]
    route_costs = [length for _route, length in solutions]
    return alpha, beta, total_distance(solutions), routes, route_costs


def format_result(result: tuple[float, float, float, list[list[int]], list[float]]) -> str:
    """将单个场景结果格式化为便于后续读取的文本块。"""
    alpha, beta, total_cost, routes, route_costs = result
    lines = [
        f"alpha={alpha:.1f}",
        f"beta={beta:.1f}",
        f"total_corrected_distance_km={total_cost:.6f}",
        f"drone_count={len(routes)}",
    ]
    for index, (route, cost) in enumerate(zip(routes, route_costs), start=1):
        lines.append(f"drone_{index}_route={' -> '.join(map(str, route))}")
        lines.append(f"drone_{index}_corrected_distance_km={cost:.6f}")
    return "\n".join(lines)


def main() -> None:
    """使用多个 CPU 进程计算 25 个情景，并保存为文本文件。"""
    scenarios = list(product(SCENARIO_VALUES, repeat=2))
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(solve_scenario, scenarios))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        "# Sensitivity analysis for f(i,j) = L * d(i,j) * mu' * k'",
        "# Weather condition: hypothetical thunderstorm (WMO weather_code=95, mu=1.30)",
        "# mu' = 1 + alpha * (mu - 1)",
        "# k' = 1 + beta * (k - 1)",
        "",
    ]
    blocks.extend(format_result(result) for result in results)
    OUTPUT_PATH.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} scenarios to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
