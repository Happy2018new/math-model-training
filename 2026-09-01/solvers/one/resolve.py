import math
import copy
from .prepare import data1, data2
from .define import PosData, PosWithTime, SensorData, sensor_data_from_raw

s1 = sensor_data_from_raw(data1, 4)
s2 = sensor_data_from_raw(data2, 5)


def linear_interpolation(data: SensorData, time: float) -> PosData | None:
    if time < data.start_time or time > data.end_time:
        return None

    ordinal = int((time - data.start_time) * data.sensor_hz + 1)
    start = max(0, ordinal - 1)
    end = min(ordinal, len(data.payload) - 1)
    if start == end:
        return data.payload[start]

    t1 = data.start_time + start / data.sensor_hz
    t2 = data.start_time + end / data.sensor_hz
    ratio = (time - t1) / (t2 - t1)

    pos1 = data.payload[start]
    pos2 = data.payload[end]
    return PosData(
        (1 - ratio) * pos1.posx + ratio * pos2.posx,
        (1 - ratio) * pos1.posy + ratio * pos2.posy,
    )


def compute_distance(pos1: PosData, pos2: PosData) -> float:
    dx = pos1.posx - pos2.posx
    dy = pos1.posy - pos2.posy
    return dx**2 + dy**2


def guess_delta_time(answer: float) -> float | None:
    start = max(s1.start_time, s2.start_time - answer)
    end = min(s1.end_time, s2.end_time - answer)
    if start > end:
        return None

    total = 0.0
    count = 0
    for index, pos1 in enumerate(s1.payload):
        current_time = s1.start_time + index / s1.sensor_hz
        if current_time < start:
            continue
        if current_time > end:
            break
        pos2 = linear_interpolation(s2, current_time + answer)
        if pos2 is None:
            continue
        total += compute_distance(pos1, pos2)
        count += 1

    if count < 0.8 * len(s1.payload):
        return None
    return total / count


def offset_sensor_data(data: SensorData, delta_time: float) -> SensorData:
    data = copy.deepcopy(data)
    data.start_time += delta_time
    data.end_time += delta_time
    return data


def generate_10hz_ans(delta_time: float) -> list[PosWithTime]:
    result = []

    newS1, newS2 = s1, offset_sensor_data(s2, -delta_time)
    start = min(newS1.start_time, newS2.start_time)
    end = max(newS1.end_time, newS2.end_time)

    t = start
    while t <= end:
        pos1 = linear_interpolation(newS1, t)
        pos2 = linear_interpolation(newS2, t)

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

    return result


def resolve() -> list[PosWithTime]:
    ans = 0.0
    err = math.inf
    ptr = 0.0
    while ptr < 1500.0:
        temp = guess_delta_time(ptr)
        if temp is not None and temp < err:
            err = temp
            ans = ptr
        ptr += 0.1

    final_ans = ans
    final_err = err
    final_ptr = ans - 0.1
    while final_ptr <= ans + 0.1:
        temp = guess_delta_time(final_ptr)
        if temp is not None and temp < final_err:
            final_err = temp
            final_ans = final_ptr
        final_ptr += 0.0001

    return generate_10hz_ans(final_ans)
