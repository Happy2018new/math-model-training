import copy
import statistics
from .define import PosData, SensorData
from .delta_time import s1, s2, linear_interpolation, compute_delta_time


def offset_sensor_data(data: SensorData, delta_time: float) -> SensorData:
    data = copy.deepcopy(data)
    data.start_time += delta_time
    data.end_time += delta_time
    return data


def offset_pos_list(data: SensorData, delta_offset: PosData) -> SensorData:
    data = copy.deepcopy(data)
    for i in data.payload:
        i.posx += delta_offset.posx
        i.posy += delta_offset.posy
    return data


def compute_delta_offset() -> tuple[list[float], list[float]]:
    dx_list = []
    dy_list = []

    new_s1 = s1
    new_s2 = offset_sensor_data(s2, -compute_delta_time())

    start = max(new_s1.start_time, new_s2.start_time)
    end = min(new_s1.end_time, new_s2.end_time)
    ptr = start

    while ptr <= end:
        pos1 = linear_interpolation(new_s1, ptr)
        pos2 = linear_interpolation(new_s2, ptr)
        if pos1 is None or pos2 is None:
            raise Exception("Should never happened")

        dx_list.append(pos2.posx - pos1.posx)
        dy_list.append(pos2.posy - pos1.posy)

        ptr += 0.1

    return dx_list, dy_list


def _block_means(values: list[float], block_size: int) -> list[float]:
    return [
        sum(values[index : index + block_size]) / block_size
        for index in range(0, len(values) - block_size + 1, block_size)
    ]


def check_have_error(
    dx_list: list[float],
    dy_list: list[float],
    block_size: int = 20,
) -> tuple[bool, bool]:
    dx_blocks = _block_means(dx_list, block_size)
    dy_blocks = _block_means(dy_list, block_size)

    dx_mean = statistics.mean(dx_blocks)
    dy_mean = statistics.mean(dy_blocks)

    dx_std = statistics.stdev(dx_blocks)
    dy_std = statistics.stdev(dy_blocks)

    dx_se = dx_std / (len(dx_blocks) ** 0.5)
    dy_se = dy_std / (len(dy_blocks) ** 0.5)

    lower_bound = dx_mean - 1.96 * dx_se, dy_mean - 1.96 * dy_se
    upper_bound = dx_mean + 1.96 * dx_se, dy_mean + 1.96 * dy_se

    return (
        0 < lower_bound[0] or 0 > upper_bound[0],
        0 < lower_bound[1] or 0 > upper_bound[1],
    )
