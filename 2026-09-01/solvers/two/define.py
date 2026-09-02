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


def sensor_data_from_raw(raw: list[list[float]], hz_data: int) -> SensorData:
    result = SensorData(hz_data, raw[0][0], raw[-1][0])
    for i in raw:
        result.payload.append(PosData(i[1], i[2]))
    return result
