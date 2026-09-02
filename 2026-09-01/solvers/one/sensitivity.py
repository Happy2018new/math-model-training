import csv
from pathlib import Path

from ..sensitivity_common import (
    estimate_delta_time,
    load_sensor_csv,
    smooth_sensor,
    time_objective,
)


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs" / "one"


def run_sensitivity() -> Path:
    t1, x1, y1 = load_sensor_csv(DATA_DIR / "table_1_sensor_1.csv")
    t2, x2, y2 = load_sensor_csv(DATA_DIR / "table_1_sensor_2.csv")

    rows = []
    for window in (1, 3, 5, 7):
        st1, sx1, sy1 = smooth_sensor(t1, x1, y1, window)
        st2, sx2, sy2 = smooth_sensor(t2, x2, y2, window)
        delta, objective = estimate_delta_time(
            st1, sx1, sy1, st2, sx2, sy2, 0.0, 1000.0
        )
        rows.append(
            {
                "experiment": "smoothing_window",
                "parameter": window,
                "parameter_unit": "points",
                "delta_time": delta,
                "objective": objective,
            }
        )

    st1, sx1, sy1 = smooth_sensor(t1, x1, y1, 5)
    st2, sx2, sy2 = smooth_sensor(t2, x2, y2, 5)
    for coarse_step in (0.5, 0.2, 0.1, 0.05):
        delta, objective = estimate_delta_time(
            st1,
            sx1,
            sy1,
            st2,
            sx2,
            sy2,
            0.0,
            1000.0,
            coarse_step=coarse_step,
            fine_step=0.001,
        )
        rows.append(
            {
                "experiment": "coarse_step",
                "parameter": coarse_step,
                "parameter_unit": "seconds",
                "delta_time": delta,
                "objective": objective,
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
