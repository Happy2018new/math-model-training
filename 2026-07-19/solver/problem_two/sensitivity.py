"""第二题第二问：供货能力和转运损耗分位数敏感性分析。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence

from .step1 import resolve as resolve_step1
from .step3 import resolve as resolve_step3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity.txt"
DEFAULT_CSV_PATH = PROJECT_ROOT / "output" / "problems" / "two" / "sensitivity.csv"
DEFAULT_SUPPLY_PERCENTILES = (0.90, 0.95, 0.975, 1.00)
DEFAULT_LOSS_PERCENTILES = (0.50, 0.75, 0.90)


@dataclass(frozen=True)
class SensitivityScenario:
    """一个供货分位数与损耗分位数组合的分析结果。"""

    supply_percentile: float  # 供应商稳定供货能力所使用的分位数
    loss_percentile: float  # 转运商预测损耗率所使用的分位数
    feasible: bool  # 当前参数组合是否成功得到可行整数解
    meets_gap: bool | None  # 两阶段是否均达到设定的相对 MIP Gap
    provider_count: int | None  # 最少供应商数量；不可行时为空
    material_provider_counts: dict[str, int]  # A、B、C 各自的供应商数量
    baseline_cost: float | None  # Step 2 基础方案的 12 周总成本
    optimized_cost: float | None  # Step 3 优化方案的 12 周总成本
    cost_saving: float | None  # 相对基础方案节省的成本
    cost_saving_rate: float | None  # 相对基础方案的成本节省率
    baseline_transport_loss: float | None  # 基础方案的 12 周运输损耗
    optimized_transport_loss: float | None  # 优化方案的 12 周运输损耗
    transport_loss_reduction: float | None  # 相对基础方案减少的运输损耗
    transport_loss_reduction_rate: float | None  # 相对基础方案的损耗减少率
    primary_mip_gap: float | None  # 第一阶段经济成本 MIP Gap
    secondary_mip_gap: float | None  # 第二阶段运输损耗 MIP Gap
    primary_status: str | None  # 第一阶段求解状态
    secondary_status: str | None  # 第二阶段求解状态
    error: str | None = None  # 不可行或求解异常时的错误信息


def _validate_percentiles(values: Sequence[float], name: str) -> tuple[float, ...]:
    """检查分位数参数，并去重后保持用户给出的顺序。"""
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} 不能为空")
    if any(not 0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"{name} 必须全部位于 0 和 1 之间")
    return tuple(dict.fromkeys(result))


def _empty_scenario(
    supply_percentile: float,
    loss_percentile: float,
    error: Exception,
) -> SensitivityScenario:
    """构造不可行场景的统一结果。"""
    return SensitivityScenario(
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
        feasible=False,
        meets_gap=None,
        provider_count=None,
        material_provider_counts={},
        baseline_cost=None,
        optimized_cost=None,
        cost_saving=None,
        cost_saving_rate=None,
        baseline_transport_loss=None,
        optimized_transport_loss=None,
        transport_loss_reduction=None,
        transport_loss_reduction_rate=None,
        primary_mip_gap=None,
        secondary_mip_gap=None,
        primary_status=None,
        secondary_status=None,
        error=f"{type(error).__name__}: {error}",
    )


def analyze_sensitivity(
    supply_percentiles: Sequence[float] = DEFAULT_SUPPLY_PERCENTILES,
    loss_percentiles: Sequence[float] = DEFAULT_LOSS_PERCENTILES,
    *,
    elasticity: float = 0.10,
    level_count: int = 20,
    transport_unit_cost: float = 0.0,
    storage_unit_cost: float = 0.0,
    primary_tolerance: float = 1e-5,
    time_limit: float | None = 300.0,
    relative_gap: float = 0.005,
    solver_messages: bool = False,
    solver_backend: str = "cbc",
) -> list[SensitivityScenario]:
    """枚举供货分位数和损耗分位数的二维组合并求解每个场景。

    每个场景都会重新执行 Step 1 和 Step 3。Step 3 内部会调用 Step 2，
    因此基础方案和整数规划方案使用完全相同的参数与供应商集合。
    """
    supply_values = _validate_percentiles(supply_percentiles, "供货分位数")
    loss_values = _validate_percentiles(loss_percentiles, "损耗分位数")

    scenarios: list[SensitivityScenario] = []
    for supply_percentile in supply_values:
        for loss_percentile in loss_values:
            try:
                minimum_providers = resolve_step1(
                    supply_percentile=supply_percentile,
                    loss_percentile=loss_percentile,
                )
                result = resolve_step3(
                    minimum_providers=minimum_providers,
                    supply_percentile=supply_percentile,
                    loss_percentile=loss_percentile,
                    elasticity=elasticity,
                    level_count=level_count,
                    transport_unit_cost=transport_unit_cost,
                    storage_unit_cost=storage_unit_cost,
                    primary_tolerance=primary_tolerance,
                    time_limit=time_limit,
                    relative_gap=relative_gap,
                    solver_messages=solver_messages,
                    solver_backend=solver_backend,
                )
                selected = minimum_providers.selected_providers
                counts: dict[str, int] = {}
                for provider in selected:
                    counts[provider.product_type] = counts.get(
                        provider.product_type, 0
                    ) + 1

                baseline = result.baseline_plan
                optimized = result.optimized_plan
                loss_reduction = (
                    baseline.total_transport_loss
                    - optimized.total_transport_loss
                )
                scenarios.append(
                    SensitivityScenario(
                        supply_percentile=supply_percentile,
                        loss_percentile=loss_percentile,
                        feasible=True,
                        meets_gap=result.is_optimal,
                        provider_count=len(selected),
                        material_provider_counts=counts,
                        baseline_cost=baseline.total_cost,
                        optimized_cost=optimized.total_cost,
                        cost_saving=result.cost_saving,
                        cost_saving_rate=result.cost_saving_rate,
                        baseline_transport_loss=baseline.total_transport_loss,
                        optimized_transport_loss=optimized.total_transport_loss,
                        transport_loss_reduction=loss_reduction,
                        transport_loss_reduction_rate=(
                            loss_reduction / baseline.total_transport_loss
                            if baseline.total_transport_loss > 1e-12
                            else 0.0
                        ),
                        primary_mip_gap=result.primary_mip_gap,
                        secondary_mip_gap=result.secondary_mip_gap,
                        primary_status=result.primary_solver_status,
                        secondary_status=result.secondary_solver_status,
                    )
                )
            except (RuntimeError, ValueError, KeyError) as error:
                scenarios.append(
                    _empty_scenario(supply_percentile, loss_percentile, error)
                )
    return scenarios


def _format_number(value: float | None, digits: int = 3) -> str:
    return "不可用" if value is None else f"{value:.{digits}f}"


def _format_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "不可用"
    if abs(value) < 0.5 * 10 ** (-(digits + 2)):
        value = 0.0
    return f"{value:.{digits}%}"


def build_report(
    scenarios: Sequence[SensitivityScenario],
    *,
    time_limit: float | None = None,
    relative_gap: float | None = None,
) -> str:
    """生成适合论文核对和归档的 UTF-8 文本报告。"""
    if not scenarios:
        raise ValueError("scenarios 不能为空")

    supply_values = tuple(
        dict.fromkeys(scenario.supply_percentile for scenario in scenarios)
    )
    loss_values = tuple(
        dict.fromkeys(scenario.loss_percentile for scenario in scenarios)
    )
    lines = [
        "第二题第二问：供货预测分位数与转运损耗分位数敏感性分析",
        "=" * 72,
        "",
        "一、分析设置",
        "供货分位数决定供应商稳定供货能力；损耗分位数决定转运商预测损耗率。",
        "每个场景均重新执行 Step 1、Step 2 和 Step 3，两个分位数采用全组合分析。",
        "供货分位数：" + "、".join(f"{value:.1%}" for value in supply_values),
        "损耗分位数：" + "、".join(f"{value:.1%}" for value in loss_values),
        "单场景求解时限："
        + ("不限时" if time_limit is None else f"{time_limit:.0f} 秒"),
        "相对 MIP Gap 要求："
        + ("未指定" if relative_gap is None else f"{relative_gap:.2%}"),
        f"场景数量：{len(scenarios)}",
        "成本节约率=(基础方案成本-优化方案成本)/基础方案成本。",
        "损耗减少率=(基础方案运输损耗-优化方案运输损耗)/基础方案运输损耗。",
        "",
        "二、场景结果",
        "供货分位数 | 损耗分位数 | 可行 | Gap达标 | 供应商数(A/B/C) | 基础成本 | 优化成本 "
        "| 成本节约率 | 基础损耗 | 优化损耗 | 损耗减少率 | 主阶段Gap | 次阶段Gap",
    ]
    for scenario in scenarios:
        counts = scenario.material_provider_counts
        count_text = (
            f"{counts.get('A', 0)}/{counts.get('B', 0)}/{counts.get('C', 0)}"
            if scenario.feasible
            else "不可用"
        )
        lines.append(
            f"{scenario.supply_percentile:>10.3f} | "
            f"{scenario.loss_percentile:>10.3f} | "
            f"{'是' if scenario.feasible else '否':>2} | "
            f"{('是' if scenario.meets_gap else '否') if scenario.feasible else '不可用':>6} | "
            f"{count_text:>13} | "
            f"{_format_number(scenario.baseline_cost):>10} | "
            f"{_format_number(scenario.optimized_cost):>10} | "
            f"{_format_percent(scenario.cost_saving_rate):>9} | "
            f"{_format_number(scenario.baseline_transport_loss):>10} | "
            f"{_format_number(scenario.optimized_transport_loss):>10} | "
            f"{_format_percent(scenario.transport_loss_reduction_rate):>9} | "
            f"{_format_percent(scenario.primary_mip_gap, 3):>9} | "
            f"{_format_percent(scenario.secondary_mip_gap, 3):>9}"
        )
        if scenario.error:
            lines.append(f"  错误：{scenario.error}")

    feasible = [scenario for scenario in scenarios if scenario.feasible]
    gap_met = [scenario for scenario in feasible if scenario.meets_gap]
    lines.extend(["", "三、汇总结果"])
    if not feasible:
        lines.append("所有场景均不可行或未能得到整数解。")
        return "\n".join(lines) + "\n"

    provider_counts = [scenario.provider_count for scenario in feasible]
    optimized_costs = [scenario.optimized_cost for scenario in feasible]
    optimized_losses = [scenario.optimized_transport_loss for scenario in feasible]
    cost_rates = [scenario.cost_saving_rate for scenario in feasible]
    loss_rates = [scenario.transport_loss_reduction_rate for scenario in feasible]
    lines.extend(
        [
            f"可行场景数量：{len(feasible)}/{len(scenarios)}",
            f"达到设定 MIP Gap 的场景数量：{len(gap_met)}/{len(feasible)}",
            f"供应商数量范围：{min(provider_counts)} 至 {max(provider_counts)}",
            f"平均供应商数量：{fmean(provider_counts):.2f}",
            f"优化方案成本范围：{min(optimized_costs):.3f} 至 {max(optimized_costs):.3f}",
            f"优化方案运输损耗范围：{min(optimized_losses):.3f} 至 {max(optimized_losses):.3f}",
            f"平均成本节约率：{fmean(cost_rates):.2%}",
            f"平均运输损耗减少率：{fmean(loss_rates):.2%}",
        ]
    )

    lowest_cost = min(feasible, key=lambda scenario: scenario.optimized_cost or float("inf"))
    lowest_loss = min(
        feasible,
        key=lambda scenario: scenario.optimized_transport_loss or float("inf"),
    )
    lines.extend(
        [
            "",
            "四、极值场景",
            f"优化成本最低：供货分位数={lowest_cost.supply_percentile:.2f}，"
            f"损耗分位数={lowest_cost.loss_percentile:.2f}，"
            f"成本={lowest_cost.optimized_cost:.3f}",
            f"优化损耗最低：供货分位数={lowest_loss.supply_percentile:.2f}，"
            f"损耗分位数={lowest_loss.loss_percentile:.2f}，"
            f"损耗={lowest_loss.optimized_transport_loss:.3f}",
            "",
            "五、论文解释建议",
            "供货分位数升高表示采用更高的历史供货能力估计，通常会扩大单家供应商可用能力，"
            "但也可能改变最少供应商集合和价格弹性计算结果。",
            "损耗分位数升高表示采用更保守的运输损耗估计，通常会降低有效入库能力，"
            "并可能增加所需供应商数量、订货量或运输损耗。实际影响以场景结果为准。",
            "若某场景不可行，应报告为该参数组合下现有供应商集合和转运容量无法满足安全库存约束，"
            "而不是将其当作优化失败。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_sensitivity(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    **kwargs: object,
) -> tuple[Path, list[SensitivityScenario]]:
    """执行敏感性分析，并写入文本报告和结构化 CSV。"""
    scenarios = analyze_sensitivity(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_report(
            scenarios,
            time_limit=kwargs.get("time_limit", 300.0),
            relative_gap=kwargs.get("relative_gap", 0.005),
        ),
        encoding="utf-8-sig",
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "supply_percentile",
                "loss_percentile",
                "feasible",
                "meets_gap",
                "provider_count",
                "provider_count_a",
                "provider_count_b",
                "provider_count_c",
                "baseline_cost",
                "optimized_cost",
                "cost_saving_rate",
                "baseline_transport_loss",
                "optimized_transport_loss",
                "transport_loss_reduction_rate",
                "primary_mip_gap",
                "secondary_mip_gap",
                "error",
            ]
        )
        for scenario in scenarios:
            writer.writerow(
                [
                    scenario.supply_percentile,
                    scenario.loss_percentile,
                    scenario.feasible,
                    scenario.meets_gap,
                    scenario.provider_count,
                    scenario.material_provider_counts.get("A"),
                    scenario.material_provider_counts.get("B"),
                    scenario.material_provider_counts.get("C"),
                    scenario.baseline_cost,
                    scenario.optimized_cost,
                    scenario.cost_saving_rate,
                    scenario.baseline_transport_loss,
                    scenario.optimized_transport_loss,
                    scenario.transport_loss_reduction_rate,
                    scenario.primary_mip_gap,
                    scenario.secondary_mip_gap,
                    scenario.error,
                ]
            )
    return output_path, scenarios
