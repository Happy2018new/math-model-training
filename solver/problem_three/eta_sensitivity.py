"""并行求解并导出不同载重影响参数 eta 下的问题三规划结果。"""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

try:
    from .resolve import (
        DEFAULT_MAX_PAYLOAD,
        PACKAGE_INPUT_PATH,
        load_points,
        solve_payload_routes,
    )
except ImportError:
    from resolve import (
        DEFAULT_MAX_PAYLOAD,
        PACKAGE_INPUT_PATH,
        load_points,
        solve_payload_routes,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "eta_sensitivity_results.txt"
ETA_VALUES = tuple(round(index * 0.01, 2) for index in range(1, 101))


def solve_eta_scenario(
    eta: float,
) -> tuple[float, float, list[tuple[list[str], list[int], float, float]]]:
    """求解一个 eta 场景并返回总燃料消耗和各无人机路线信息。"""
    depot, delivery_points = load_points(PACKAGE_INPUT_PATH)
    routes = solve_payload_routes(
        depot,
        delivery_points,
        max_payload=DEFAULT_MAX_PAYLOAD,
        eta=eta,
    )
    route_results = [
        (
            [task.package_id for task in route.tasks],
            [int(depot["id"])]
            + [int(task.point["id"]) for task in route.tasks]
            + [int(depot["id"])],
            route.initial_load,
            route.energy,
        )
        for route in routes
    ]
    return eta, sum(route.energy for route in routes), route_results


def format_result(
    result: tuple[float, float, list[tuple[list[str], list[int], float, float]]]
) -> str:
    """将一个 eta 场景格式化为可供后续可视化解析的文本块。"""
    eta, total_energy, routes = result
    lines = [
        f"eta={eta:.2f}",
        f"max_payload_kg={DEFAULT_MAX_PAYLOAD:.3f}",
        f"drone_count={len(routes)}",
        f"total_fuel_consumption={total_energy:.6f}",
    ]
    for index, (package_ids, point_ids, initial_load, route_energy) in enumerate(
        routes, start=1
    ):
        lines.extend(
            [
                f"drone_{index}_package_ids={','.join(package_ids)}",
                f"drone_{index}_route={' -> '.join(map(str, point_ids))}",
                f"drone_{index}_initial_load_kg={initial_load:.3f}",
                f"drone_{index}_fuel_consumption={route_energy:.6f}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    """使用多进程计算全部 eta 场景并保存文本结果。"""
    worker_count = min(len(ETA_VALUES), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(solve_eta_scenario, ETA_VALUES))

    header = [
        "# Problem three eta sensitivity analysis",
        "# W(t) = 1 + eta * w(t) / Q",
        "# Package data are simulated with a fixed random seed",
        "",
    ]
    blocks = header + [format_result(result) for result in results]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} eta scenarios to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
