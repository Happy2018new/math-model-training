"""第三题多情景敏感性分析：逐情景求解并单独生成 TXT 报告。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..problem_two.step1 import MATERIAL_DEMANDS, resolve as resolve_step1
from ..problem_two.step2 import SAFETY_STOCKS
from .define import RobustPlanningResult
from .solver import (
    DEFAULT_LOSS_REDUCTIONS,
    DEFAULT_SUPPLY_DEVIATIONS,
    resolve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "problems" / "three" / "sensitivity.txt"
)


@dataclass(frozen=True)
class SensitivityCase:
    """一个独立敏感性情景的求解结果或失败信息。"""

    scenario_id: int
    supply_deviation: float
    loss_reduction: float
    result: RobustPlanningResult | None
    error: str | None = None


def _validate_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name}不能为空")
    if any(not -1.0 < value < 1.0 for value in result):
        raise ValueError(f"{name}必须位于 -1 和 1 之间")
    return tuple(dict.fromkeys(result))


def analyze_sensitivity(
    supply_deviations: Sequence[float] = DEFAULT_SUPPLY_DEVIATIONS,
    loss_reductions: Sequence[float] = DEFAULT_LOSS_REDUCTIONS,
    *,
    supply_percentile: float = 0.95,
    loss_percentile: float = 0.75,
    elasticity: float = 0.10,
    level_count: int = 10,
    reserve_count: int = 1,
    transport_unit_cost: float = 0.0,
    storage_unit_cost: float = 0.0,
    primary_tolerance: float = 1e-5,
    time_limit: float | None = 15.0,
    retry_time_limit: float | None = 120.0,
    relative_gap: float = 0.05,
    solver_messages: bool = False,
) -> list[SensitivityCase]:
    """分别求解每个供货—损耗组合，不在情景之间强制共享决策。

    该函数的结果用于敏感性分析；第三题基本方案请调用 solver.resolve()。
    """
    supply_values = _validate_values(supply_deviations, "供货偏差情景")
    loss_values = _validate_values(loss_reductions, "损耗下降情景")
    minimum_providers = resolve_step1(
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
    )

    cases: list[SensitivityCase] = []
    scenario_id = 0
    for supply_deviation in supply_values:
        for loss_reduction in loss_values:
            scenario_id += 1
            def solve_case(case_time_limit: float | None) -> RobustPlanningResult:
                return resolve(
                    minimum_providers=minimum_providers,
                    supply_percentile=supply_percentile,
                    loss_percentile=loss_percentile,
                    supply_deviation=supply_deviation,
                    loss_reduction=loss_reduction,
                    elasticity=elasticity,
                    level_count=level_count,
                    reserve_count=reserve_count,
                    transport_unit_cost=transport_unit_cost,
                    storage_unit_cost=storage_unit_cost,
                    primary_tolerance=primary_tolerance,
                    time_limit=case_time_limit,
                    relative_gap=relative_gap,
                    solver_messages=solver_messages,
                    output_path=None,
                )

            try:
                result = solve_case(time_limit)
                cases.append(
                    SensitivityCase(
                        scenario_id=scenario_id,
                        supply_deviation=supply_deviation,
                        loss_reduction=loss_reduction,
                        result=result,
                    )
                )
            except RuntimeError as first_error:
                if retry_time_limit is not None and retry_time_limit != time_limit:
                    try:
                        result = solve_case(retry_time_limit)
                        cases.append(
                            SensitivityCase(
                                scenario_id=scenario_id,
                                supply_deviation=supply_deviation,
                                loss_reduction=loss_reduction,
                                result=result,
                            )
                        )
                        continue
                    except (RuntimeError, ValueError) as retry_error:
                        error: Exception = retry_error
                else:
                    error = first_error
                cases.append(
                    SensitivityCase(
                        scenario_id=scenario_id,
                        supply_deviation=supply_deviation,
                        loss_reduction=loss_reduction,
                        result=None,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            except ValueError as error:
                cases.append(
                    SensitivityCase(
                        scenario_id=scenario_id,
                        supply_deviation=supply_deviation,
                        loss_reduction=loss_reduction,
                        result=None,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
    return cases


def _scenario_metrics(result: RobustPlanningResult) -> tuple[float, float, int]:
    scenario_result = result.scenario_results[0]
    minimum_margin = min(
        week.ending_inventories[material] - SAFETY_STOCKS[material]
        for week in scenario_result.weeks
        for material in MATERIAL_DEMANDS
    )
    maximum_load = max(
        max(week.transfer_loads) for week in scenario_result.weeks
    )
    active_provider_count = len({order.provider_id for order in result.orders})
    return minimum_margin, maximum_load, active_provider_count


def _build_report_without_failure_records(
    results: Sequence[RobustPlanningResult],
) -> str:
    """生成多情景敏感性分析 TXT 报告。"""
    if not results:
        raise ValueError("results 不能为空")

    first = results[0]
    lines = [
        "第三题：供货偏差与转运损耗下降多情景敏感性分析",
        "=" * 96,
        "",
        "一、分析说明",
        "本文件只用于敏感性分析，每个情景单独求解一个订货—转运方案。",
        "不同情景之间不共享订货量和转运商选择；因此本文件不代表一个跨情景共同执行的鲁棒方案。",
        "第三题基本方案请查看同目录下的 basic_result.txt。",
        f"供应能力分位数：{first.supply_percentile:.2%}",
        f"基准损耗率分位数：{first.loss_percentile:.2%}",
        f"供货档位数：{first.level_count}",
        f"情景数量：{len(results)}",
        "",
        "二、情景汇总",
        f"{'情景':>4} {'供货偏差':>10} {'损耗下降':>10} {'候选数':>8} "
        f"{'实际使用数':>10} {'总成本':>16} {'总运输损耗':>16} "
        f"{'最小库存裕量':>16} {'最大转运负载':>16} {'主Gap':>10} {'次Gap':>10} {'达标':>6}",
    ]

    for scenario_number, result in enumerate(results, start=1):
        scenario_result = result.scenario_results[0]
        scenario = scenario_result.scenario
        minimum_margin, maximum_load, active_count = _scenario_metrics(result)
        lines.append(
            f"{scenario_number:>4}"
            f"{scenario.supply_deviation:>10.1%}"
            f"{scenario.loss_reduction:>10.1%}"
            f"{len(result.providers):>8}"
            f"{active_count:>10}"
            f"{scenario_result.total_cost:>16.3f}"
            f"{scenario_result.total_transport_loss:>16.3f}"
            f"{minimum_margin:>16.3f}"
            f"{maximum_load:>16.3f}"
            f"{result.primary_mip_gap:>10.3%}"
            f"{result.secondary_mip_gap:>10.3%}"
            f"{'是' if result.is_optimal else '否':>6}"
        )

    lines.extend(["", "三、各情景求解状态"])
    for scenario_number, result in enumerate(results, start=1):
        scenario = result.scenarios[0]
        lines.extend(
            [
                f"情景 {scenario_number}：供货偏差={scenario.supply_deviation:+.1%}，"
                f"每周损耗下降={scenario.loss_reduction:.1%}",
                f"候选供应商：{len(result.providers)} 家；"
                f"12 周实际使用：{len({order.provider_id for order in result.orders})} 家",
                f"第一阶段：{result.primary_status}，MIP Gap={result.primary_mip_gap:.6%}",
                f"第二阶段：{result.secondary_status}，MIP Gap={result.secondary_mip_gap:.6%}",
                f"经济成本：{result.scenario_results[0].total_cost:.3f}；"
                f"运输损耗：{result.scenario_results[0].total_transport_loss:.3f}",
                "",
            ]
        )

    lines.extend(["四、各情景逐周结果"])
    for scenario_number, result in enumerate(results, start=1):
        scenario = result.scenarios[0]
        lines.extend(
            [
                "",
                f"情景 {scenario_number}：供货偏差={scenario.supply_deviation:+.1%}，"
                f"每周损耗下降={scenario.loss_reduction:.1%}",
                f"{'周':>3} {'A入库':>12} {'B入库':>12} {'C入库':>12} "
                f"{'A库存':>12} {'B库存':>12} {'C库存':>12} "
                f"{'本周成本':>14} {'本周损耗':>14} {'最大负载':>14}",
            ]
        )
        for week in result.scenario_results[0].weeks:
            lines.append(
                f"{week.week:>3}"
                f"{week.material_receipts['A']:>12.3f}"
                f"{week.material_receipts['B']:>12.3f}"
                f"{week.material_receipts['C']:>12.3f}"
                f"{week.ending_inventories['A']:>12.3f}"
                f"{week.ending_inventories['B']:>12.3f}"
                f"{week.ending_inventories['C']:>12.3f}"
                f"{week.total_cost:>14.3f}"
                f"{week.transport_loss:>14.3f}"
                f"{max(week.transfer_loads):>14.3f}"
            )

    feasible_results = [
        result
        for result in results
        if result.scenario_results and result.is_optimal
    ]
    lines.extend(["", "五、敏感性分析结论"])
    if feasible_results:
        lowest_cost = min(
            feasible_results,
            key=lambda result: result.scenario_results[0].total_cost,
        )
        lowest_loss = min(
            feasible_results,
            key=lambda result: result.scenario_results[0].total_transport_loss,
        )
        lowest_cost_index = results.index(lowest_cost) + 1
        lowest_loss_index = results.index(lowest_loss) + 1
        lines.extend(
            [
                f"达到当前 Gap 要求的场景数：{len(feasible_results)}/{len(results)}",
                f"达标场景最低成本：{lowest_cost.scenario_results[0].total_cost:.3f}，"
                f"情景={lowest_cost_index}",
                f"达标场景最低运输损耗：{lowest_loss.scenario_results[0].total_transport_loss:.3f}，"
                f"情景={lowest_loss_index}",
            ]
        )
    else:
        lines.append("没有场景在当前求解设置下达到 Gap 要求，请增加时限或放宽 Gap。")
    lines.extend(
        [
            "注意：敏感性分析中的每个情景独立求解，结果用于比较参数变化的影响；"
            "基本执行方案应使用 basic_result.txt。",
        ]
    )
    return "\n".join(lines) + "\n"


def _case_metrics(case: SensitivityCase) -> tuple[float, float, int] | None:
    """计算成功情景的库存裕量、最大负载和实际使用供应商数。"""
    if case.result is None:
        return None
    scenario_result = case.result.scenario_results[0]
    minimum_margin = min(
        week.ending_inventories[material] - SAFETY_STOCKS[material]
        for week in scenario_result.weeks
        for material in MATERIAL_DEMANDS
    )
    maximum_load = max(
        max(week.transfer_loads) for week in scenario_result.weeks
    )
    active_count = len({order.provider_id for order in case.result.orders})
    return minimum_margin, maximum_load, active_count


def build_report(cases: Sequence[SensitivityCase]) -> str:
    """生成多情景敏感性分析 TXT 报告，保留失败情景的错误信息。"""
    if not cases:
        raise ValueError("cases 不能为空")
    successful = [case for case in cases if case.result is not None]
    first_result = successful[0].result if successful else None
    lines = [
        "第三题：供货偏差与转运损耗下降多情景敏感性分析",
        "=" * 96,
        "",
        "一、分析说明",
        "本文件只用于敏感性分析，每个情景单独求解一个订货—转运方案。",
        "不同情景之间不共享订货量和转运商选择；因此本文件不代表一个跨情景共同执行的鲁棒方案。",
        "第三题基本方案请查看同目录下的 basic_result.txt。",
        f"供应能力分位数：{first_result.supply_percentile:.2%}" if first_result else "供应能力分位数：不可用",
        f"基准损耗率分位数：{first_result.loss_percentile:.2%}" if first_result else "基准损耗率分位数：不可用",
        f"供货档位数：{first_result.level_count}" if first_result else "供货档位数：不可用",
        f"情景数量：{len(cases)}",
        "",
        "二、情景汇总",
        f"{'情景':>4} {'供货偏差':>10} {'损耗下降':>10} {'状态':>10} "
        f"{'候选数':>8} {'实际使用数':>10} {'总成本':>16} {'总运输损耗':>16} "
        f"{'最小库存裕量':>16} {'最大转运负载':>16} {'主Gap':>10} {'次Gap':>10} {'达标':>6}",
    ]
    for case in cases:
        if case.result is None:
            lines.append(
                f"{case.scenario_id:>4}{case.supply_deviation:>10.1%}"
                f"{case.loss_reduction:>10.1%}{'未完成':>10}"
                f"{'-':>8}{'-':>10}{'-':>16}{'-':>16}{'-':>16}{'-':>16}"
                f"{'-':>10}{'-':>10}{'否':>6}"
            )
            lines.append(f"  错误：{case.error}")
            continue
        result = case.result
        scenario_result = result.scenario_results[0]
        minimum_margin, maximum_load, active_count = _case_metrics(case)  # type: ignore[misc]
        lines.append(
            f"{case.scenario_id:>4}"
            f"{case.supply_deviation:>10.1%}"
            f"{case.loss_reduction:>10.1%}{'完成':>10}"
            f"{len(result.providers):>8}{active_count:>10}"
            f"{scenario_result.total_cost:>16.3f}"
            f"{scenario_result.total_transport_loss:>16.3f}"
            f"{minimum_margin:>16.3f}{maximum_load:>16.3f}"
            f"{result.primary_mip_gap:>10.3%}{result.secondary_mip_gap:>10.3%}"
            f"{'是' if result.is_optimal else '否':>6}"
        )

    lines.extend(["", "三、各情景求解状态"])
    for case in cases:
        lines.append(
            f"情景 {case.scenario_id}：供货偏差={case.supply_deviation:+.1%}，"
            f"每周损耗下降={case.loss_reduction:.1%}"
        )
        if case.result is None:
            lines.extend([f"求解失败或限时未找到整数解：{case.error}", ""])
            continue
        result = case.result
        scenario_result = result.scenario_results[0]
        lines.extend(
            [
                f"候选供应商：{len(result.providers)} 家；"
                f"12 周实际使用：{len({order.provider_id for order in result.orders})} 家",
                f"第一阶段：{result.primary_status}，MIP Gap={result.primary_mip_gap:.6%}",
                f"第二阶段：{result.secondary_status}，MIP Gap={result.secondary_mip_gap:.6%}",
                f"经济成本：{scenario_result.total_cost:.3f}；"
                f"运输损耗：{scenario_result.total_transport_loss:.3f}",
                "",
            ]
        )

    lines.extend(["四、各情景逐周结果"])
    for case in cases:
        lines.extend(
            [
                "",
                f"情景 {case.scenario_id}：供货偏差={case.supply_deviation:+.1%}，"
                f"每周损耗下降={case.loss_reduction:.1%}",
            ]
        )
        if case.result is None:
            lines.append(f"无逐周结果：{case.error}")
            continue
        lines.append(
            f"{'周':>3} {'A入库':>12} {'B入库':>12} {'C入库':>12} "
            f"{'A库存':>12} {'B库存':>12} {'C库存':>12} "
            f"{'本周成本':>14} {'本周损耗':>14} {'最大负载':>14}"
        )
        for week in case.result.scenario_results[0].weeks:
            lines.append(
                f"{week.week:>3}"
                f"{week.material_receipts['A']:>12.3f}"
                f"{week.material_receipts['B']:>12.3f}"
                f"{week.material_receipts['C']:>12.3f}"
                f"{week.ending_inventories['A']:>12.3f}"
                f"{week.ending_inventories['B']:>12.3f}"
                f"{week.ending_inventories['C']:>12.3f}"
                f"{week.total_cost:>14.3f}"
                f"{week.transport_loss:>14.3f}"
                f"{max(week.transfer_loads):>14.3f}"
            )

    lines.extend(["", "五、敏感性分析结论"])
    optimal_cases = [
        case for case in cases if case.result is not None and case.result.is_optimal
    ]
    lines.append(f"成功找到整数解的场景数：{len(successful)}/{len(cases)}")
    lines.append(f"达到当前 Gap 要求的场景数：{len(optimal_cases)}/{len(cases)}")
    if optimal_cases:
        lowest_cost = min(
            optimal_cases,
            key=lambda case: case.result.scenario_results[0].total_cost,  # type: ignore[union-attr]
        )
        lowest_loss = min(
            optimal_cases,
            key=lambda case: case.result.scenario_results[0].total_transport_loss,  # type: ignore[union-attr]
        )
        lines.extend(
            [
                f"达标场景最低成本：{lowest_cost.result.scenario_results[0].total_cost:.3f}，"
                f"情景={lowest_cost.scenario_id}",
                f"达标场景最低运输损耗：{lowest_loss.result.scenario_results[0].total_transport_loss:.3f}，"
                f"情景={lowest_loss.scenario_id}",
            ]
        )
    lines.append(
        "注意：敏感性分析中的每个情景独立求解，结果用于比较参数变化的影响；"
        "基本执行方案应使用 basic_result.txt。"
    )
    return "\n".join(lines) + "\n"


def run_sensitivity(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    **kwargs: object,
) -> tuple[Path, list[SensitivityCase]]:
    """运行 9 个情景并写入独立敏感性分析报告。"""
    cases = analyze_sensitivity(**kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(cases), encoding="utf-8-sig")
    return output_path, cases


if __name__ == "__main__":
    path, cases = run_sensitivity()
    print(path)
    print(f"完成 {len(cases)} 个敏感性情景")
