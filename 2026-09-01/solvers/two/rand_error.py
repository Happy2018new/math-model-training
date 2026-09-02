import copy, statistics
from .define import PosData, SensorData
from .delta_time import linear_interpolation
from .delta_offset import compute_delta_offset


def compute_rand_error() -> tuple[
    tuple[PosData, SensorData, SensorData],
    tuple[float, float],
]:
    offset, s1, s2 = compute_delta_offset()

    start = max(s1.start_time, s2.start_time)
    end = min(s1.end_time, s2.end_time)
    ptr = start

    mul_dx = []
    mul_dy = []

    while ptr <= end:
        pos1 = linear_interpolation(s1, ptr)
        pos2 = linear_interpolation(s2, ptr)
        if pos1 is None or pos2 is None:
            raise Exception("Should never happened")

        mul_dx.append(pos2.posx - pos1.posx)
        mul_dy.append(pos2.posy - pos1.posy)

        ptr += 0.1

    return (
        (offset, s1, s2),
        (statistics.pstdev(mul_dx) / (2**0.5), statistics.pstdev(mul_dy) / (2**0.5)),
    )
