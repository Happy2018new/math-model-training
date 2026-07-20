"""Indicator-removal sensitivity analysis for CRITIC-TOPSIS."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Sequence

from .critic import INDICATOR_NAMES, STD_MATRIX, calculate_critic
from .topsis import TopsisResult, calculate_topsis
from ..utils.read import std_received

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "one" / "sensitivity.txt"
TOP_N = 30


@dataclass(frozen=True)
class ScenarioResult:
    removed_indices: tuple[int, ...]
    kept_indices: tuple[int, ...]
    weights: tuple[float, ...]
    topsis: TopsisResult
    top_overlap: int
    retention_rate: float
    jaccard: float
    spearman: float
    boundary_gap: float


def _select_columns(
    matrix: Sequence[Sequence[float]],
    column_indices: Sequence[int],
) -> list[list[float]]:
    return [[float(row[index]) for index in column_indices] for row in matrix]


def _rank_positions(ranking: Sequence[int]) -> list[int]:
    positions = [0] * len(ranking)
    for position, row_index in enumerate(ranking, start=1):
        positions[row_index] = position
    return positions


def _spearman_from_rankings(
    baseline_ranking: Sequence[int],
    scenario_ranking: Sequence[int],
) -> float:
    if len(baseline_ranking) != len(scenario_ranking):
        raise ValueError("两组排名包含的供应商数量不一致")

    sample_count = len(baseline_ranking)
    if sample_count < 2:
        return 1.0

    baseline_positions = _rank_positions(baseline_ranking)
    scenario_positions = _rank_positions(scenario_ranking)
    squared_difference_sum = sum(
        (left - right) ** 2
        for left, right in zip(baseline_positions, scenario_positions)
    )
    return 1.0 - (6.0 * squared_difference_sum / (sample_count * (sample_count**2 - 1)))


def _boundary_gap(result: TopsisResult, top_n: int) -> float:
    if len(result.ranking) <= top_n:
        return 0.0
    last_selected = result.ranking[top_n - 1]
    first_unselected = result.ranking[top_n]
    return result.closeness[last_selected] - result.closeness[first_unselected]


def _provider_name(row_index: int) -> str:
    return f"S{std_received[row_index].provider_id:03d}"


def analyze_indicator_removal(
    removal_sizes: Sequence[int] = (1, 2),
    top_n: int = TOP_N,
) -> tuple[TopsisResult, tuple[float, ...], list[ScenarioResult]]:
    """Enumerate indicator removals and recalculate CRITIC-TOPSIS each time."""
    indicator_count = len(INDICATOR_NAMES)
    if top_n <= 0 or top_n > len(STD_MATRIX):
        raise ValueError("top_n 必须位于供应商数量范围内")
    if any(size <= 0 or size >= indicator_count for size in removal_sizes):
        raise ValueError("删除数量必须大于 0 且小于指标总数")

    baseline_critic = calculate_critic(STD_MATRIX)
    baseline_topsis = calculate_topsis(STD_MATRIX, baseline_critic.weights)
    baseline_top = set(baseline_topsis.ranking[:top_n])

    scenarios: list[ScenarioResult] = []
    all_indices = tuple(range(indicator_count))
    for removal_size in removal_sizes:
        for removed_indices in combinations(all_indices, removal_size):
            kept_indices = tuple(
                index for index in all_indices if index not in removed_indices
            )
            scenario_matrix = _select_columns(STD_MATRIX, kept_indices)
            critic_result = calculate_critic(scenario_matrix)
            topsis_result = calculate_topsis(
                scenario_matrix,
                critic_result.weights,
            )

            scenario_top = set(topsis_result.ranking[:top_n])
            overlap = len(baseline_top & scenario_top)
            union = len(baseline_top | scenario_top)
            scenarios.append(
                ScenarioResult(
                    removed_indices=removed_indices,
                    kept_indices=kept_indices,
                    weights=critic_result.weights,
                    topsis=topsis_result,
                    top_overlap=overlap,
                    retention_rate=overlap / top_n,
                    jaccard=overlap / union,
                    spearman=_spearman_from_rankings(
                        baseline_topsis.ranking,
                        topsis_result.ranking,
                    ),
                    boundary_gap=_boundary_gap(topsis_result, top_n),
                )
            )

    return baseline_topsis, baseline_critic.weights, scenarios


def _format_provider_list(indices: Sequence[int]) -> str:
    return ", ".join(_provider_name(index) for index in indices)


def build_report(
    baseline_topsis: TopsisResult,
    baseline_weights: Sequence[float],
    scenarios: Sequence[ScenarioResult],
    top_n: int = TOP_N,
) -> str:
    """Build a UTF-8 text report containing all sensitivity results."""
    lines = [
        "第一题 CRITIC-TOPSIS 指标删减敏感性分析",
        "=" * 66,
        "",
        "一、分析设置",
        f"供应商数量：{len(STD_MATRIX)}",
        f"基准指标数量：{len(INDICATOR_NAMES)}",
        f"敏感性方案数量：{len(scenarios)}",
        f"每个方案均重新计算 CRITIC 权重和 TOPSIS 前 {top_n} 名。",
        "指标删减采用全部组合枚举：分别删除 1 个和 2 个指标。",
        "",
        "二、基准方案",
        "基准 CRITIC 权重：",
    ]
    lines.extend(
        f"  {name}：{weight:.6f}"
        for name, weight in zip(INDICATOR_NAMES, baseline_weights)
    )
    lines.extend(
        [
            f"基准第 {top_n} 名与第 {top_n + 1} 名贴近度差："
            f"{_boundary_gap(baseline_topsis, top_n):.6f}",
            f"基准前 {top_n} 家：",
            "  " + _format_provider_list(baseline_topsis.ranking[:top_n]),
            "",
            "三、各指标删减方案",
            "说明：保留率=与基准前30名重合数/30；Spearman 越接近 1，整体排名越稳定。",
        ]
    )

    for scenario_number, scenario in enumerate(scenarios, start=1):
        removed_names = [INDICATOR_NAMES[index] for index in scenario.removed_indices]
        kept_names = [INDICATOR_NAMES[index] for index in scenario.kept_indices]
        lines.extend(
            [
                "",
                f"方案 {scenario_number:02d}",
                f"删除指标：{'、'.join(removed_names)}",
                f"保留指标：{'、'.join(kept_names)}",
                "重新计算后的 CRITIC 权重：",
            ]
        )
        lines.extend(
            f"  {name}：{weight:.6f}"
            for name, weight in zip(kept_names, scenario.weights)
        )
        lines.extend(
            [
                f"前 {top_n} 名重合数：{scenario.top_overlap}/{top_n}",
                f"前 {top_n} 名保留率：{scenario.retention_rate:.2%}",
                f"Jaccard 相似系数：{scenario.jaccard:.6f}",
                f"Spearman 等级相关系数：{scenario.spearman:.6f}",
                f"第 {top_n} 名与第 {top_n + 1} 名贴近度差："
                f"{scenario.boundary_gap:.6f}",
                f"前 {top_n} 家：",
                "  " + _format_provider_list(scenario.topsis.ranking[:top_n]),
            ]
        )

    retention_rates = [scenario.retention_rate for scenario in scenarios]
    jaccard_values = [scenario.jaccard for scenario in scenarios]
    spearman_values = [scenario.spearman for scenario in scenarios]
    least_stable = min(scenarios, key=lambda scenario: scenario.retention_rate)
    least_stable_names = "、".join(
        INDICATOR_NAMES[index] for index in least_stable.removed_indices
    )

    selection_counts = [0] * len(STD_MATRIX)
    for scenario in scenarios:
        for row_index in scenario.topsis.ranking[:top_n]:
            selection_counts[row_index] += 1
    selected_indices = sorted(
        (index for index, count in enumerate(selection_counts) if count > 0),
        key=lambda index: (-selection_counts[index], index),
    )

    lines.extend(
        [
            "",
            "四、汇总结果",
            f"平均前 {top_n} 名保留率：{fmean(retention_rates):.2%}",
            f"最低前 {top_n} 名保留率：{min(retention_rates):.2%}",
            f"最高前 {top_n} 名保留率：{max(retention_rates):.2%}",
            f"平均 Jaccard 相似系数：{fmean(jaccard_values):.6f}",
            f"平均 Spearman 等级相关系数：{fmean(spearman_values):.6f}",
            f"最低 Spearman 等级相关系数：{min(spearman_values):.6f}",
            f"最敏感的删除方案：删除{least_stable_names}",
            "",
            "五、供应商在删减方案中的前 30 名入选频率",
        ]
    )
    lines.extend(
        f"{_provider_name(index)}：{selection_counts[index]}/{len(scenarios)} "
        f"({selection_counts[index] / len(scenarios):.2%})"
        for index in selected_indices
    )
    lines.extend(
        [
            "",
            "六、判定参考",
            "若平均前30名保留率不低于 90%，且平均 Spearman 系数不低于 0.95，",
            "可认为供应商评价结果对指标删减具有较好的稳健性。该阈值属于本文采用的经验判据。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sensitivity(
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Run the analysis and write the complete TXT report."""
    baseline_topsis, baseline_weights, scenarios = analyze_indicator_removal()
    report = build_report(baseline_topsis, baseline_weights, scenarios)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8-sig")
    return output_path
