import csv

data1: list[list[float]] = []
data2: list[list[float]] = []

with open("inputs/table_2_sensor_1.csv", mode="r+", encoding="utf-8") as file:
    reader = csv.reader(file)
    data1 = [[float(j) for j in i] for i in list(reader)[1:]]
with open("inputs/table_2_sensor_2.csv", mode="r+", encoding="utf-8") as file:
    reader = csv.reader(file)
    data2 = [[float(j) for j in i] for i in list(reader)[1:]]


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


if not validate_hz_time(data1, 4):
    raise Exception("Data1 does not validate for 4 Hz")
if not validate_hz_time(data2, 5):
    raise Exception("Data2 does not validate for 5 Hz")
