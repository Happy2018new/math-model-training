import math
from typing import Callable
from .prepare import data1, data2
from .define import PosData, SensorData, sensor_data_from_raw


def moving_average(data: SensorData, window: int = 5) -> SensorData:
    if window < 1 or window % 2 == 0:
        raise Exception("Window must be a positive odd number")

    result = []
    radius = window // 2
    for index in range(len(data.payload)):
        left = max(0, index - radius)
        right = min(len(data.payload), index + radius + 1)
        points = data.payload[left:right]
        result.append(
            PosData(
                sum(point.posx for point in points) / len(points),
                sum(point.posy for point in points) / len(points),
            )
        )

    return SensorData(
        sensor_hz=data.sensor_hz,
        start_time=data.start_time,
        end_time=data.end_time,
        payload=result,
    )


s1 = sensor_data_from_raw(data1, 4)
s2 = sensor_data_from_raw(data2, 5)
new_s1 = moving_average(s1, 5)
new_s2 = moving_average(s2, 5)


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


def _compute_general(
    answer: float,
    func: Callable[[PosData, PosData], float],
) -> float | None:
    start = max(new_s1.start_time, new_s2.start_time - answer)
    end = min(new_s1.end_time, new_s2.end_time - answer)
    if start > end:
        return None

    total = 0.0
    count = 0
    for index, pos1 in enumerate(new_s1.payload):
        current_time = new_s1.start_time + index / new_s1.sensor_hz
        if current_time < start:
            continue
        if current_time > end:
            break
        pos2 = linear_interpolation(new_s2, current_time + answer)
        if pos2 is None:
            continue
        total += func(pos1, pos2)
        count += 1

    if count < 0.8 * len(new_s1.payload):
        return None
    return total / count


def guess_delta_time(answer: float) -> float | None:
    offset1 = _compute_general(answer, lambda pos1, pos2: pos2.posx - pos1.posx)
    if offset1 is None:
        return None

    offset2 = _compute_general(answer, lambda pos1, pos2: pos2.posy - pos1.posy)
    if offset2 is None:
        return None

    def func(pos1: PosData, pos2: PosData) -> float:
        dx = (pos2.posx - pos1.posx - offset1) ** 2
        dy = (pos2.posy - pos1.posy - offset2) ** 2
        return dx + dy

    return _compute_general(answer, func)


def compute_delta_time() -> float:
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

    return final_ans
