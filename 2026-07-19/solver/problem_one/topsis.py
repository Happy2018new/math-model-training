"""Evaluate suppliers with TOPSIS after obtaining CRITIC weights."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .critic import (
    CRITIC_RESULT,
    INDICATOR_NAMES,
    STD_MATRIX,
)
from ..utils.read import std_received


@dataclass(frozen=True)
class TopsisResult:
    """TOPSIS intermediate values and supplier evaluation results."""

    weighted_matrix: tuple[tuple[float, ...], ...]
    positive_ideal: tuple[float, ...]
    negative_ideal: tuple[float, ...]
    distance_to_positive: tuple[float, ...]
    distance_to_negative: tuple[float, ...]
    closeness: tuple[float, ...]
    ranking: tuple[int, ...]


def _validate_inputs(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> None:
    if not matrix or not matrix[0]:
        raise ValueError("TOPSIS 评价矩阵不能为空")
    if len(matrix[0]) != len(weights):
        raise ValueError("指标列数与权重数量不一致")
    if any(weight < 0 for weight in weights):
        raise ValueError("TOPSIS 权重不能为负数")
    if sum(weights) <= 0:
        raise ValueError("TOPSIS 权重之和必须大于 0")

    column_count = len(matrix[0])
    for row in matrix:
        if len(row) != column_count:
            raise ValueError("评价矩阵每一行必须具有相同列数")
        if not all(math.isfinite(float(value)) for value in row):
            raise ValueError("评价矩阵只能包含有限数值")


def calculate_topsis(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> TopsisResult:
    """Run TOPSIS on a standardized matrix where larger is always better.

    This follows the supplied formulas: weighted matrix, positive/negative
    ideal solutions, Euclidean distances, and relative closeness.
    """
    _validate_inputs(matrix, weights)
    numeric_matrix = [[float(value) for value in row] for row in matrix]
    weight_sum = sum(float(weight) for weight in weights)
    normalized_weights = [float(weight) / weight_sum for weight in weights]

    weighted_matrix = [
        [value * weight for value, weight in zip(row, normalized_weights)]
        for row in numeric_matrix
    ]
    positive_ideal = tuple(
        max(row[column] for row in weighted_matrix)
        for column in range(len(normalized_weights))
    )
    negative_ideal = tuple(
        min(row[column] for row in weighted_matrix)
        for column in range(len(normalized_weights))
    )

    distance_to_positive = tuple(
        math.sqrt(
            sum(
                (value - positive_ideal[column]) ** 2
                for column, value in enumerate(row)
            )
        )
        for row in weighted_matrix
    )
    distance_to_negative = tuple(
        math.sqrt(
            sum(
                (value - negative_ideal[column]) ** 2
                for column, value in enumerate(row)
            )
        )
        for row in weighted_matrix
    )
    closeness = tuple(
        (
            negative_distance / (positive_distance + negative_distance)
            if positive_distance + negative_distance > 0
            else 0.5
        )
        for positive_distance, negative_distance in zip(
            distance_to_positive,
            distance_to_negative,
        )
    )
    ranking = tuple(
        sorted(
            range(len(closeness)),
            key=lambda index: closeness[index],
            reverse=True,
        )
    )

    return TopsisResult(
        weighted_matrix=tuple(tuple(row) for row in weighted_matrix),
        positive_ideal=positive_ideal,
        negative_ideal=negative_ideal,
        distance_to_positive=distance_to_positive,
        distance_to_negative=distance_to_negative,
        closeness=closeness,
        ranking=ranking,
    )


TOPSIS_RESULT = calculate_topsis(STD_MATRIX, CRITIC_RESULT.weights)


def debug(top_n: int = 30) -> None:
    """Print TOPSIS scores and the highest-ranked suppliers."""
    if top_n <= 0:
        raise ValueError("top_n 必须是正整数")

    print("TOPSIS 指标权重")
    print("-" * 34)
    for name, weight in zip(INDICATOR_NAMES, CRITIC_RESULT.weights):
        print(f"{name:<18}{weight:>12.6f}")

    print(f"\nTOPSIS 综合评价前 {min(top_n, len(TOPSIS_RESULT.ranking))} 名供应商")
    print("-" * 40)
    print(f"{'排名':<8}{'供应商':<14}{'贴近度 C_i':>14}")
    for rank, row_index in enumerate(TOPSIS_RESULT.ranking[:top_n], start=1):
        provider_id = std_received[row_index].provider_id
        score = TOPSIS_RESULT.closeness[row_index]
        print(f"{rank:<8}{f'S{provider_id:03d}':<14}{score:>14.6f}")
