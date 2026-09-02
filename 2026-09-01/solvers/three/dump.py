import csv
import math
from bisect import bisect_right
from pathlib import Path
from .define import PosData, PosWithTime
from .resolve import resolve

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "three"
HEADER = ("时间(s)", "X坐标(m)", "Y坐标(m)")


def resample_10hz(trajectory: list[PosWithTime]) -> list[PosWithTime]:
    """Resample a trajectory onto a strict 0.1-second time grid."""
    if not trajectory:
        return []
    if len(trajectory) == 1:
        return [trajectory[0]]

    times = [item.time for item in trajectory]
    start_index = math.ceil(times[0] * 10 - 1e-9)
    end_index = math.floor(times[-1] * 10 + 1e-9)
    if start_index > end_index:
        return []

    result = []
    for grid_index in range(start_index, end_index + 1):
        time = grid_index / 10
        right = bisect_right(times, time)

        if right == 0:
            continue
        if right == len(trajectory) or times[right - 1] == time:
            result.append(trajectory[right - 1])
            continue

        left = right - 1
        t1, t2 = times[left], times[right]
        ratio = (time - t1) / (t2 - t1)
        p1, p2 = trajectory[left].pos, trajectory[right].pos
        result.append(
            PosWithTime(
                time,
                PosData(
                    (1 - ratio) * p1.posx + ratio * p2.posx,
                    (1 - ratio) * p1.posy + ratio * p2.posy,
                ),
            )
        )
    return result


def dump_trajectory(
    trajectory: list[PosWithTime],
    path: str | Path,
) -> Path:
    """Write a position trajectory in the original three-column CSV format."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory = resample_10hz(trajectory)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for item in trajectory:
            writer.writerow(
                (
                    round(item.time, 10),
                    item.pos.posx,
                    item.pos.posy,
                )
            )

    return output_path


def dump_10hz_trajectory(
    path: str | Path = "result.csv",
) -> Path:
    """Solve problem three and dump its formal 10 Hz fused trajectory."""
    trajectory, _ = resolve()
    return dump_trajectory(trajectory, path)
