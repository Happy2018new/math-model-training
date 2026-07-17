"""为问题三生成可复现的模拟包裹重量数据。"""

import json
import random
from pathlib import Path
from typing import Any


Point = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "output" / "processed" / "points.json"
OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "three" / "points_with_packages.json"
RANDOM_SEED = 20260717


def sample_package_weight(random_generator: random.Random) -> float:
    """按轻、中、重三档混合分布生成单个包裹重量，单位为 kg。"""
    category = random_generator.random()
    if category < 0.50:
        return round(random_generator.uniform(0.3, 1.2), 2)
    if category < 0.85:
        return round(random_generator.uniform(1.2, 2.5), 2)
    return round(random_generator.uniform(2.5, 4.0), 2)


def add_package_data(points: list[Point], random_seed: int) -> list[Point]:
    """为每个绿色配送点添加包含 1 至 2 个重量的 packages 列表。"""
    random_generator = random.Random(random_seed)
    result: list[Point] = []
    for point in points:
        point_with_packages = point.copy()
        if point.get("color") == "green":
            package_count = 1 if random_generator.random() < 0.68 else 2
            point_with_packages["packages"] = [
                sample_package_weight(random_generator) for _ in range(package_count)
            ]
        else:
            point_with_packages["packages"] = []
        result.append(point_with_packages)
    return result


def main() -> None:
    """读取原始点数据、生成模拟包裹并保存独立输入文件。"""
    points: list[Point] = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    points_with_packages = add_package_data(points, RANDOM_SEED)
    weights = [
        weight
        for point in points_with_packages
        for weight in point["packages"]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(points_with_packages, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved simulated package data to: {OUTPUT_PATH}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Packages: {len(weights)}")
    print(f"Total weight: {sum(weights):.2f} kg")
    print(f"Weight range: {min(weights):.2f}-{max(weights):.2f} kg")


if __name__ == "__main__":
    main()
