import csv
from pathlib import Path

import numpy as np

from ..sensitivity_common import (
    aligned_differences,
    bias_confidence,
    estimate_delta_time,
    load_sensor_csv,
    smooth_sensor,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs" / "three"


def run_sensitivity() -> Path:
    t1, x1, y1 = load_sensor_csv(DATA_DIR / "table_3_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(DATA_DIR / "table_3_sensor_2.csv")
    rows = []

    for window in (1, 3, 5, 7):
        st1, sx1, sy1 = smooth_sensor(t1, x1, y1, window)
        st2, sx2, sy2 = smooth_sensor(t2, x2, y2, window)
        delta, objective = estimate_delta_time(
            st1, sx1, sy1, st2, sx2, sy2, -1500.0, 1500.0
        )
        _, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
        result = bias_confidence(dx, dy, block_size=20, confidence=0.95)
        rows.append(
            {
                "experiment": "smoothing_window",
                "parameter": window,
                "parameter_unit": "points",
                "delta_time": delta,
                "objective": objective,
                "mean_dx": result["mean_x"],
                "mean_dy": result["mean_y"],
                "x_low": result["x_low"],
                "x_high": result["x_high"],
                "y_low": result["y_low"],
                "y_high": result["y_high"],
                "has_x_bias": result["has_x_bias"],
                "has_y_bias": result["has_y_bias"],
                "block_count": result["block_count"],
            }
        )

    st1, sx1, sy1 = smooth_sensor(t1, x1, y1, 5)
    st2, sx2, sy2 = smooth_sensor(t2, x2, y2, 5)
    delta, objective = estimate_delta_time(
        st1, sx1, sy1, st2, sx2, sy2, -1500.0, 1500.0
    )
    _, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
    for block_size in (10, 20, 50, 100):
        for confidence in (0.90, 0.95, 0.99):
            result = bias_confidence(dx, dy, block_size, confidence)
            rows.append(
                {
                    "experiment": "block_confidence",
                    "parameter": block_size,
                    "parameter_unit": "points",
                    "confidence": confidence,
                    "delta_time": delta,
                    "objective": objective,
                    "mean_dx": result["mean_x"],
                    "mean_dy": result["mean_y"],
                    "x_low": result["x_low"],
                    "x_high": result["x_high"],
                    "y_low": result["y_low"],
                    "y_high": result["y_high"],
                    "has_x_bias": result["has_x_bias"],
                    "has_y_bias": result["has_y_bias"],
                    "block_count": result["block_count"],
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sensitivity.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    print(run_sensitivity())
