import csv
from pathlib import Path

import numpy as np

from ..sensitivity_common import (
    aligned_differences,
    estimate_delta_time,
    load_sensor_csv,
    smooth_sensor,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs" / "two"


def _spatial_metrics(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float, float, float, float]:
    mean_x, mean_y = float(dx.mean()), float(dy.mean())
    std_x, std_y = float(dx.std()), float(dy.std())
    rmse = float(np.sqrt(np.mean(dx**2 + dy**2)))
    return mean_x, mean_y, std_x, std_y, rmse


def run_sensitivity() -> Path:
    t1, x1, y1 = load_sensor_csv(DATA_DIR / "table_2_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(DATA_DIR / "table_2_sensor_2.csv")
    rows = []

    for window in (1, 3, 5, 7):
        st1, sx1, sy1 = smooth_sensor(t1, x1, y1, window)
        st2, sx2, sy2 = smooth_sensor(t2, x2, y2, window)
        delta, objective = estimate_delta_time(
            st1, sx1, sy1, st2, sx2, sy2, 0.0, 1000.0
        )
        _, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
        metrics = _spatial_metrics(dx, dy)
        rows.append(
            {
                "experiment": "smoothing_window",
                "parameter": window,
                "parameter_unit": "points",
                "delta_time": delta,
                "objective": objective,
                "mean_dx": metrics[0],
                "mean_dy": metrics[1],
                "std_dx": metrics[2],
                "std_dy": metrics[3],
                "rmse": metrics[4],
            }
        )

    st1, sx1, sy1 = smooth_sensor(t1, x1, y1, 5)
    st2, sx2, sy2 = smooth_sensor(t2, x2, y2, 5)
    delta, objective = estimate_delta_time(st1, sx1, sy1, st2, sx2, sy2, 0.0, 1000.0)
    _, dx, dy = aligned_differences(t1, x1, y1, t2, x2, y2, delta)
    n = len(dx)
    for trim in (0.0, 0.05, 0.10):
        left = int(n * trim)
        right = n - left
        metrics = _spatial_metrics(dx[left:right], dy[left:right])
        rows.append(
            {
                "experiment": "overlap_trim",
                "parameter": trim,
                "parameter_unit": "fraction_each_end",
                "delta_time": delta,
                "objective": objective,
                "mean_dx": metrics[0],
                "mean_dy": metrics[1],
                "std_dx": metrics[2],
                "std_dy": metrics[3],
                "rmse": metrics[4],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sensitivity.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    print(run_sensitivity())
