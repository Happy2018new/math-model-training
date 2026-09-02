import statistics
from .delta_time import s1, s2, linear_interpolation, compute_delta_time
from .delta_offset import offset_sensor_data


def compute_rand_error() -> tuple[float, float]:
    new_s1 = s1
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    start = max(new_s1.start_time, new_s2.start_time)
    end = min(new_s1.end_time, new_s2.end_time)
    ptr = start

    mul_dx = []
    mul_dy = []

    while ptr <= end:
        pos1 = linear_interpolation(new_s1, ptr)
        pos2 = linear_interpolation(new_s2, ptr)
        if pos1 is None or pos2 is None:
            raise Exception("Should never happened")

        mul_dx.append(pos2.posx - pos1.posx)
        mul_dy.append(pos2.posy - pos1.posy)

        ptr += 0.1

    return (
        statistics.pstdev(mul_dx) / (2**0.5),
        statistics.pstdev(mul_dy) / (2**0.5),
    )
