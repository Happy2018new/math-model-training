from dataclasses import dataclass, field


@dataclass
class PosData:
    posx: float
    posy: float


@dataclass
class PosWithTime:
    time: float
    pos: PosData


@dataclass
class SensorData:
    sensor_hz: int
    start_time: float
    end_time: float
    payload: list[PosData] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class BlockMeanBiasEstimate:
    """由等长分块均值得到的空间偏差估计及其正态近似区间。"""

    x_block_means: tuple[float, ...]
    y_block_means: tuple[float, ...]
    mean_x: float
    mean_y: float
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    has_x_bias: bool
    has_y_bias: bool


def sensor_data_from_raw(raw: list[list[float]], hz_data: int) -> SensorData:
    result = SensorData(hz_data, raw[0][0], raw[-1][0])
    for i in raw:
        result.payload.append(PosData(i[1], i[2]))
    return result
