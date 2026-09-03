from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PosData:
    posx: float
    posy: float


@dataclass
class PosWithTime:
    time: float
    pos: PosData


@dataclass
class TrajectoryPoint:
    """包含位置、速度和加速度估计值的单个轨迹采样点。"""

    time: float
    pos: PosData
    velocity: PosData
    acceleration: PosData

    @property
    def speed(self) -> float:
        """返回该采样点的速度大小。"""
        return (self.velocity.posx**2 + self.velocity.posy**2) ** 0.5

    @property
    def acceleration_magnitude(self) -> float:
        """返回该采样点的加速度大小。"""
        return (self.acceleration.posx**2 + self.acceleration.posy**2) ** 0.5


@dataclass(frozen=True)
class TaskCandidate:
    """一个已通过可行性筛选的射击或拍照候选任务。

    每个候选已满足距离、速度、加速度和准备时段要求。调度模型只需要保留
    编号、任务类型、执行时刻和拍照方向角。``candidate_id`` 稳定且唯一，
    可直接作为整数规划变量的编号；准备开始时刻由执行时刻和任务类型推得。
    """

    candidate_id: str
    task_id: str
    task_type: Literal["shoot", "photo"]
    execute_time: float
    angle_deg: float | None = None

    @property
    def preparation_time(self) -> float:
        """返回该任务类型所需的准备时长，单位为秒。"""
        return 1.5 if self.task_type == "shoot" else 0.5

    @property
    def prepare_start(self) -> float:
        """返回该候选占用准备区间的开始时刻。"""
        return self.execute_time - self.preparation_time


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
