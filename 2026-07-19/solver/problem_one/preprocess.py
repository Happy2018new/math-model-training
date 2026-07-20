"""Audit the supplier source data used by problem one."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDER_CSV_PATH = PROJECT_ROOT / "input" / "received.csv"
SUPPLY_CSV_PATH = PROJECT_ROOT / "input" / "ask.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "problems" / "one" / "preprocess"
WEEK_COLUMNS = tuple(f"W{week:03d}" for week in range(1, 241))
REQUIRED_COLUMNS = ("Provider", "Type", *WEEK_COLUMNS)
VALID_MATERIALS = {"A", "B", "C"}
PROVIDER_PATTERN = re.compile(r"^S(\d{3})$")


@dataclass(frozen=True)
class DatasetAudit:
    name: str
    path: Path
    rows: tuple[dict[str, str], ...]
    column_count: int
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    missing_cells: int
    duplicate_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]
    missing_expected_ids: tuple[str, ...]
    unexpected_ids: tuple[str, ...]
    invalid_materials: tuple[str, ...]
    non_numeric_cells: int
    negative_cells: int
    zero_cells: int
    numeric_cells: int
    positive_minimum: float | None
    positive_maximum: float | None
    positive_mean: float | None
    material_counts: tuple[tuple[str, int], ...]

    @property
    def zero_rate(self) -> float:
        return self.zero_cells / self.numeric_cells if self.numeric_cells else 0.0

    @property
    def passed(self) -> bool:
        return not any(
            (
                self.missing_columns,
                self.missing_cells,
                self.duplicate_ids,
                self.invalid_ids,
                self.missing_expected_ids,
                self.unexpected_ids,
                self.invalid_materials,
                self.non_numeric_cells,
                self.negative_cells,
            )
        )


@dataclass(frozen=True)
class PreprocessAudit:
    order: DatasetAudit
    supply: DatasetAudit
    missing_order_ids: tuple[str, ...]
    missing_supply_ids: tuple[str, ...]
    material_mismatches: tuple[str, ...]
    supply_without_order: int
    order_without_supply: int

    @property
    def passed(self) -> bool:
        return (
            self.order.passed
            and self.supply.passed
            and not self.missing_order_ids
            and not self.missing_supply_ids
            and not self.material_mismatches
            and self.supply_without_order == 0
        )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV 缺少表头：{path}")
                return list(reader.fieldnames), list(reader)
        except UnicodeDecodeError as error:
            last_error = error

    assert last_error is not None
    raise last_error


def _audit_dataset(name: str, path: Path) -> DatasetAudit:
    columns, rows = _read_csv(path)
    missing_columns = tuple(
        column for column in REQUIRED_COLUMNS if column not in columns
    )
    extra_columns = tuple(
        column for column in columns if column not in REQUIRED_COLUMNS
    )
    provider_ids = [str(row.get("Provider", "")).strip() for row in rows]
    duplicate_ids = tuple(
        sorted({value for value in provider_ids if provider_ids.count(value) > 1})
    )
    invalid_ids = tuple(
        value for value in provider_ids if PROVIDER_PATTERN.fullmatch(value) is None
    )
    expected_ids = {f"S{provider_id:03d}" for provider_id in range(1, 319)}
    observed_ids = set(provider_ids)

    materials = [str(row.get("Type", "")).strip() for row in rows]
    invalid_materials = tuple(
        sorted({value for value in materials if value not in VALID_MATERIALS})
    )
    material_counts = tuple(
        (material, materials.count(material)) for material in sorted(VALID_MATERIALS)
    )

    missing_cells = 0
    non_numeric_cells = 0
    negative_cells = 0
    zero_cells = 0
    numeric_values: list[float] = []
    positive_values: list[float] = []
    for row in rows:
        for column in WEEK_COLUMNS:
            raw_value = row.get(column)
            if raw_value is None or not str(raw_value).strip():
                missing_cells += 1
                continue
            try:
                value = float(raw_value)
            except ValueError:
                non_numeric_cells += 1
                continue
            if not math.isfinite(value):
                non_numeric_cells += 1
                continue
            numeric_values.append(value)
            if value < 0:
                negative_cells += 1
            elif value == 0:
                zero_cells += 1
            else:
                positive_values.append(value)

    return DatasetAudit(
        name=name,
        path=path,
        rows=tuple(rows),
        column_count=len(columns),
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        missing_cells=missing_cells,
        duplicate_ids=duplicate_ids,
        invalid_ids=invalid_ids,
        missing_expected_ids=tuple(sorted(expected_ids - observed_ids)),
        unexpected_ids=tuple(sorted(observed_ids - expected_ids)),
        invalid_materials=invalid_materials,
        non_numeric_cells=non_numeric_cells,
        negative_cells=negative_cells,
        zero_cells=zero_cells,
        numeric_cells=len(numeric_values),
        positive_minimum=min(positive_values) if positive_values else None,
        positive_maximum=max(positive_values) if positive_values else None,
        positive_mean=fmean(positive_values) if positive_values else None,
        material_counts=material_counts,
    )


def run_audit() -> PreprocessAudit:
    order = _audit_dataset("企业订货量", ORDER_CSV_PATH)
    supply = _audit_dataset("供应商供货量", SUPPLY_CSV_PATH)
    order_by_id = {row["Provider"]: row for row in order.rows}
    supply_by_id = {row["Provider"]: row for row in supply.rows}
    order_ids = set(order_by_id)
    supply_ids = set(supply_by_id)

    shared_ids = sorted(order_ids & supply_ids)
    material_mismatches = tuple(
        provider_id
        for provider_id in shared_ids
        if order_by_id[provider_id].get("Type") != supply_by_id[provider_id].get("Type")
    )
    supply_without_order = 0
    order_without_supply = 0
    for provider_id in shared_ids:
        order_row = order_by_id[provider_id]
        supply_row = supply_by_id[provider_id]
        for column in WEEK_COLUMNS:
            try:
                order_value = float(order_row[column])
                supply_value = float(supply_row[column])
            except (KeyError, TypeError, ValueError):
                continue
            supply_without_order += int(order_value == 0 and supply_value > 0)
            order_without_supply += int(order_value > 0 and supply_value == 0)

    return PreprocessAudit(
        order=order,
        supply=supply,
        missing_order_ids=tuple(sorted(supply_ids - order_ids)),
        missing_supply_ids=tuple(sorted(order_ids - supply_ids)),
        material_mismatches=material_mismatches,
        supply_without_order=supply_without_order,
        order_without_supply=order_without_supply,
    )


def _dataset_report(dataset: DatasetAudit) -> list[str]:
    material_text = "、".join(
        f"{material}={count}" for material, count in dataset.material_counts
    )
    return [
        f"数据集：{dataset.name}",
        f"文件：{dataset.path}",
        f"行数：{len(dataset.rows)}",
        f"列数：{dataset.column_count}",
        f"缺少必需列：{len(dataset.missing_columns)}",
        f"额外列：{len(dataset.extra_columns)}",
        f"缺失单元格：{dataset.missing_cells}",
        f"重复供应商 ID：{len(dataset.duplicate_ids)}",
        f"非法供应商 ID：{len(dataset.invalid_ids)}",
        f"缺少预期供应商 ID：{len(dataset.missing_expected_ids)}",
        f"非预期供应商 ID：{len(dataset.unexpected_ids)}",
        f"非法材料类别：{len(dataset.invalid_materials)}",
        f"非数值单元格：{dataset.non_numeric_cells}",
        f"负值单元格：{dataset.negative_cells}",
        f"零值单元格：{dataset.zero_cells}",
        f"零值比例：{dataset.zero_rate:.4%}",
        f"正值最小值：{dataset.positive_minimum}",
        f"正值最大值：{dataset.positive_maximum}",
        (
            f"正值平均值：{dataset.positive_mean:.6f}"
            if dataset.positive_mean is not None
            else "正值平均值：无"
        ),
        f"材料数量：{material_text}",
        f"数据集检查：{'通过' if dataset.passed else '未通过'}",
    ]


def build_report(audit: PreprocessAudit) -> str:
    lines = [
        "问题一原始数据预处理检查报告",
        "=" * 60,
        "",
        *_dataset_report(audit.order),
        "",
        *_dataset_report(audit.supply),
        "",
        "跨表一致性检查",
        f"仅存在于供货表的供应商：{len(audit.missing_order_ids)}",
        f"仅存在于订货表的供应商：{len(audit.missing_supply_ids)}",
        f"材料类别不一致：{len(audit.material_mismatches)}",
        f"无订货但发生供货的记录：{audit.supply_without_order}",
        f"有订货但供货为零的记录：{audit.order_without_supply}",
        "",
        "零值说明：零值代表当周未订货或未供货，是有效业务记录，不按缺失值处理。",
        f"总体检查：{'通过' if audit.passed else '未通过'}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    audit: PreprocessAudit,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preprocess_check_report.txt"
    summary_path = output_dir / "preprocess_check_summary.csv"
    report_path.write_text(build_report(audit), encoding="utf-8-sig")

    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "数据集",
                "行数",
                "列数",
                "缺失值",
                "重复ID",
                "非数值",
                "负值",
                "零值",
                "零值比例",
                "检查结果",
            )
        )
        for dataset in (audit.order, audit.supply):
            writer.writerow(
                (
                    dataset.name,
                    len(dataset.rows),
                    dataset.column_count,
                    dataset.missing_cells,
                    len(dataset.duplicate_ids),
                    dataset.non_numeric_cells,
                    dataset.negative_cells,
                    dataset.zero_cells,
                    f"{dataset.zero_rate:.8f}",
                    "通过" if dataset.passed else "未通过",
                )
            )
    return report_path, summary_path


if __name__ == "__main__":
    result = run_audit()
    for written_path in write_outputs(result):
        print(f"已生成：{written_path}")
    print(f"总体检查：{'通过' if result.passed else '未通过'}")
    raise SystemExit(0 if result.passed else 1)
