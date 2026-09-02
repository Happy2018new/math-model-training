import copy
from .define import PosData, SensorData
from .delta_time import s1, s2, linear_interpolation, compute_delta_time


def offset_sensor_data(data: SensorData, delta_time: float) -> SensorData:
    data = copy.deepcopy(data)
    data.start_time += delta_time
    data.end_time += delta_time
    return data


def offset_pos_list(
    data: SensorData, offset_posx: float, offset_posy: float
) -> SensorData:
    data = copy.deepcopy(data)
    for i in data.payload:
        i.posx += offset_posx
        i.posy += offset_posy
    return data


def compute_delta_offset() -> tuple[PosData, SensorData, SensorData]:
    new_s1 = copy.deepcopy(s1)
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    start = max(new_s1.start_time, new_s2.start_time)
    end = min(new_s1.end_time, new_s2.end_time)
    ptr = start

    dx = 0
    dy = 0
    count = 0

    while ptr <= end:
        pos1 = linear_interpolation(new_s1, ptr)
        pos2 = linear_interpolation(new_s2, ptr)
        if pos1 is None or pos2 is None:
            raise Exception("Should never happened")

        dx += pos2.posx - pos1.posx
        dy += pos2.posy - pos1.posy
        count += 1

        ptr += 0.1

    dx_mean = dx / count
    dy_mean = dy / count
    return (
        PosData(dx_mean, dy_mean),
        new_s1,
        offset_pos_list(new_s2, -dx_mean, -dy_mean),
    )
