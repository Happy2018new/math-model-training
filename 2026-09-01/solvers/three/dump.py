import csv
from pathlib import Path

from .define import SensorData
from .delta_offset import offset_sensor_data
from .delta_time import compute_delta_time, s1, s2

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "three"
HEADER = ("时间(s)", "X坐标(m)", "Y坐标(m)")


def dump_sensor_data(data: SensorData, path: str | Path) -> Path:
    """Write one sensor sequence in the same three-column CSV format."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)
        for index, point in enumerate(data.payload):
            time = round(data.start_time + index / data.sensor_hz, 10)
            writer.writerow((time, point.posx, point.posy))

    return output_path


def dump_processed_sensors(
    s1_path: str | Path = "table_1.csv",
    s2_path: str | Path = "table_2.csv",
) -> tuple[Path, Path]:
    """Dump the two third-problem sensor sequences after time alignment.

    Sensor 1 is kept unchanged. Sensor 2 is shifted by ``-delta_time`` so
    both files use the sensor-1 time reference. No spatial offset is applied,
    because the block-based test found no significant fixed spatial bias.
    """
    delta_time = compute_delta_time()
    aligned_s2 = offset_sensor_data(s2, -delta_time)
    return dump_sensor_data(s1, s1_path), dump_sensor_data(aligned_s2, s2_path)
