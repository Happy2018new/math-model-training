"""读取问题三输出的 10 Hz 融合轨迹，并检查时间间隔。"""

import csv

data: list[list[float]] = []
with open("outputs/three/result.csv", mode="r+", encoding="utf-8") as file:
    reader = csv.reader(file)
    data = [[float(j) for j in i] for i in list(reader)[1:]]


def validate_hz_time(
    data: list[list[float]], hz_time: int, errors: float = 0.001
) -> bool:
    for i in range(len(data)):
        if i == len(data) - 1:
            break

        delta = data[i + 1][0] - data[i][0]
        if abs(delta - 1 / hz_time) >= errors:
            return False

    return True


if not validate_hz_time(data, 10):
    raise ValueError("result.csv 不满足 10 Hz 时间间隔要求")
