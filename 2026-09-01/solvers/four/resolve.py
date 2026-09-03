"""Run the constraint screening for problem four.

Problem three keeps its original 11-point smoothing result. For this task,
we use a 51-point local cubic fit only when estimating velocity and
acceleration, because the constraints are applied to derivatives and those
are much more sensitive to residual position noise.
"""

from __future__ import annotations

from .candidates import (
    feasible_photo_times,
    feasible_shoot_times,
    read_photo_targets,
    read_shoot_targets,
)
from .smooth import sensor_data, smooth_data


TASK_SMOOTH_WINDOW = 51
TASK_POLYORDER = 3

SHOOT_CONSTRAINTS = {
    "min_distance": 5.0,
    "max_distance": 30.0,
    "max_speed": 2.0,
    "max_acceleration": 1.5,
}

PHOTO_CONSTRAINTS = {
    "min_distance": 10.0,
    "max_distance": 40.0,
    "max_speed": 1.5,
    "max_acceleration": 1.5,
}


def calculate_candidates() -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return feasible execution times for every shooting and photo target."""
    trajectory = smooth_data(
        sensor_data,
        window_length=TASK_SMOOTH_WINDOW,
        polyorder=TASK_POLYORDER,
    )

    shoot_times = {
        target_id: feasible_shoot_times(trajectory, x, y, **SHOOT_CONSTRAINTS)
        for target_id, x, y in read_shoot_targets()
    }
    photo_times = {
        target_id: feasible_photo_times(trajectory, x, y, **PHOTO_CONSTRAINTS)
        for target_id, x, y in read_photo_targets()
    }
    return shoot_times, photo_times


if __name__ == "__main__":
    shoots, photos = calculate_candidates()
    print("\u5c04\u51fb\u5408\u6cd5\u6267\u884c\u65f6\u523b:")
    for target_id, times in shoots.items():
        print(f"{target_id}: {[round(time, 1) for time in times]}")
    print("\u62cd\u7167\u5408\u6cd5\u6267\u884c\u65f6\u523b:")
    for target_id, times in photos.items():
        print(f"{target_id}: {[round(time, 1) for time in times]}")
