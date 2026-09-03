import copy
import statistics
from .define import BlockMeanBiasEstimate, PosData, SensorData
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


def compute_delta_offset(delta_time: float | None = None) -> tuple[list[float], list[float]]:
    """返回按求解器时间对齐后，在 10 Hz 网格上的逐点空间差值。"""
    dx_list = []
    dy_list = []

    new_s1 = s1
    if delta_time is None:
        delta_time = compute_delta_time()
    new_s2 = offset_sensor_data(s2, -delta_time)

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
    if block_size < 1:
        raise ValueError("block_size must be positive")
    return [
        sum(values[index : index + block_size]) / block_size
        for index in range(0, len(values) - block_size + 1, block_size)
    ]


def estimate_block_mean_bias(
    dx_list: list[float],
    dy_list: list[float],
    block_size: int = 20,
    z_value: float = 1.96,
) -> BlockMeanBiasEstimate:
    """用分块均值的算术平均估计固定空间偏差及其置信区间。"""
    dx_blocks = _block_means(dx_list, block_size)
    dy_blocks = _block_means(dy_list, block_size)
    if len(dx_blocks) < 2 or len(dy_blocks) < 2:
        raise ValueError("at least two complete blocks are required")

    dx_mean = statistics.mean(dx_blocks)
    dy_mean = statistics.mean(dy_blocks)
    dx_se = statistics.stdev(dx_blocks) / (len(dx_blocks) ** 0.5)
    dy_se = statistics.stdev(dy_blocks) / (len(dy_blocks) ** 0.5)
    x_low, x_high = dx_mean - z_value * dx_se, dx_mean + z_value * dx_se
    y_low, y_high = dy_mean - z_value * dy_se, dy_mean + z_value * dy_se

    return BlockMeanBiasEstimate(
        x_block_means=tuple(dx_blocks),
        y_block_means=tuple(dy_blocks),
        mean_x=dx_mean,
        mean_y=dy_mean,
        x_low=x_low,
        x_high=x_high,
        y_low=y_low,
        y_high=y_high,
        has_x_bias=0 < x_low or 0 > x_high,
        has_y_bias=0 < y_low or 0 > y_high,
    )


def check_have_error(
    dx_list: list[float],
    dy_list: list[float],
    block_size: int = 20,
) -> tuple[bool, bool]:
    estimate = estimate_block_mean_bias(dx_list, dy_list, block_size)
    return estimate.has_x_bias, estimate.has_y_bias
