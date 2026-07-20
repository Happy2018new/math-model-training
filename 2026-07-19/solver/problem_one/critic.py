"""Determine objective indicator weights with the CRITIC method."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from .factors import (
    std_factor_all_products_sum,
    std_factor_order_match_rate,
    std_factor_order_match_stable,
    std_factor_product_contribute,
    std_factor_provide_weeks,
)
from ..utils.read import std_received

INDICATOR_NAMES = (
    "总供货量",
    "供货频率",
    "供货稳定性",
    "订单匹配度",
    "等效产品贡献量",
)


@dataclass(frozen=True)
class CriticResult:
    """Intermediate values and final results produced by CRITIC."""

    standard_deviations: tuple[float, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    conflicts: tuple[float, ...]
    information: tuple[float, ...]
    weights: tuple[float, ...]


def _validate_matrix(matrix: Sequence[Sequence[float]]) -> None:
    if len(matrix) < 2:
        raise ValueError("CRITIC 至少需要两个评价对象")
    if not matrix[0]:
        raise ValueError("CRITIC 至少需要一个评价指标")

    column_count = len(matrix[0])
    for row in matrix:
        if len(row) != column_count:
            raise ValueError("评价矩阵的每一行必须具有相同列数")
        if not all(math.isfinite(float(value)) for value in row):
            raise ValueError("评价矩阵只能包含有限数值")


def _pearson_correlation(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    first_delta = [value - first_mean for value in first]
    second_delta = [value - second_mean for value in second]

    denominator = math.sqrt(
        sum(value**2 for value in first_delta) * sum(value**2 for value in second_delta)
    )
    if denominator == 0:
        # Correlation is undefined for a constant column. Its standard
        # deviation is also zero, so CRITIC will assign it zero information.
        return 0.0

    correlation = (
        sum(left * right for left, right in zip(first_delta, second_delta))
        / denominator
    )
    return max(-1.0, min(1.0, correlation))


def calculate_critic(matrix: Sequence[Sequence[float]]) -> CriticResult:
    """Calculate CRITIC indicator weights.

    Rows represent suppliers and columns represent standardized indicators.
    Every indicator must already have the same direction: larger is better.
    """
    _validate_matrix(matrix)
    numeric_matrix = [[float(value) for value in row] for row in matrix]
    columns = [list(column) for column in zip(*numeric_matrix)]

    standard_deviations = tuple(statistics.pstdev(column) for column in columns)
    correlation_matrix = tuple(
        tuple(_pearson_correlation(left, right) for right in columns)
        for left in columns
    )
    conflicts = tuple(
        (
            sum(
                1.0 - correlation_matrix[index][other_index]
                for other_index, other_deviation in enumerate(standard_deviations)
                if other_deviation > 0
            )
            if deviation > 0
            else 0.0
        )
        for index, deviation in enumerate(standard_deviations)
    )
    information = tuple(
        deviation * conflict
        for deviation, conflict in zip(standard_deviations, conflicts)
    )

    total_information = sum(information)
    if total_information <= 0:
        raise ValueError("所有指标均无有效信息，无法计算 CRITIC 权重")

    weights = tuple(value / total_information for value in information)
    return CriticResult(
        standard_deviations=standard_deviations,
        correlation_matrix=correlation_matrix,
        conflicts=conflicts,
        information=information,
        weights=weights,
    )


STD_MATRIX = [
    list(row)
    for row in zip(
        std_factor_all_products_sum,
        std_factor_provide_weeks,
        std_factor_order_match_stable,
        std_factor_order_match_rate,
        std_factor_product_contribute,
    )
]

if len(STD_MATRIX) != len(std_received):
    raise ValueError("标准化指标数量与供应商数量不一致")

CRITIC_RESULT = calculate_critic(STD_MATRIX)
CRITIC_WEIGHTS = dict(zip(INDICATOR_NAMES, CRITIC_RESULT.weights))


def debug() -> None:
    """Print CRITIC intermediate values and final weights."""
    print("CRITIC 指标权重")
    print("-" * 68)
    print(f"{'指标':<18}{'标准差':>12}{'冲突性':>12}{'信息量':>12}{'权重':>12}")
    for name, deviation, conflict, information, weight in zip(
        INDICATOR_NAMES,
        CRITIC_RESULT.standard_deviations,
        CRITIC_RESULT.conflicts,
        CRITIC_RESULT.information,
        CRITIC_RESULT.weights,
    ):
        print(
            f"{name:<18}{deviation:>12.6f}{conflict:>12.6f}"
            f"{information:>12.6f}{weight:>12.6f}"
        )
