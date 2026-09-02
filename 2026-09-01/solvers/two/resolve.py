from .define import PosData, PosWithTime
from .delta_time import linear_interpolation
from .rand_error import compute_rand_error


def resolve() -> tuple[
    list[PosWithTime],
    tuple[float, float],
]:
    ans, err = compute_rand_error()
    _, s1, s2 = ans

    result = []
    start = min(s1.start_time, s2.start_time)
    end = max(s1.end_time, s2.end_time)

    t = start
    while t <= end:
        pos1 = linear_interpolation(s1, t)
        pos2 = linear_interpolation(s2, t)

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

    return result, err
