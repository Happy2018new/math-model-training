from .define import PosData, PosWithTime
from .delta_time import s1, s2, linear_interpolation, compute_delta_time
from .delta_offset import offset_sensor_data
from .rand_error import compute_rand_error


def resolve() -> tuple[
    list[PosWithTime],
    tuple[float, float],
]:
    new_s1 = s1
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    result = []
    start = min(new_s1.start_time, new_s2.start_time)
    end = max(new_s1.end_time, new_s2.end_time)

    t = start
    while t <= end:
        pos1 = linear_interpolation(new_s1, t)
        pos2 = linear_interpolation(new_s2, t)

        if pos1 is None and pos2 is not None:
            result.append(PosWithTime(t, pos2))
        elif pos1 is not None and pos2 is None:
            result.append(PosWithTime(t, pos1))
        elif pos1 is not None and pos2 is not None:
            result.append(
                PosWithTime(
                    t,
                    PosData(
                        (pos1.posx + pos2.posx) / 2,
                        (pos1.posy + pos2.posy) / 2,
                    ),
                )
            )

        t = round(t + 0.1, 1)

    return result, compute_rand_error()
