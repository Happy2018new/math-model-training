"""第三题：给定基本情景下的两阶段订货与转运 MILP。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pulp

from ..problem_one.topsis import TOPSIS_RESULT
from ..problem_two.define import MinimumProviderResult, SelectedProviderInfo
from ..problem_two.step1 import (
    MATERIAL_DEMANDS,
    TRANSFER_CAPACITY,
    resolve as resolve_problem_two_step1,
)
from ..problem_two.step2 import EPSILON, SAFETY_STOCKS, WEEK_COUNT
from ..problem_two.step3 import BASE_PRICES
from ..utils.read import std_ask, std_received
from .define import (
    RobustPlanningResult,
    RobustProviderOrder,
    RobustScenario,
    RobustScenarioResult,
    RobustScenarioWeek,
)

DEFAULT_SUPPLY_DEVIATIONS = (-0.05, 0.0, 0.05)
DEFAULT_LOSS_REDUCTIONS = (0.01, 0.03, 0.05)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "problems"
    / "three"
    / "basic_result.txt"
)


def _parse_cbc_log(path: Path) -> tuple[str, float]:
    """从 CBC 原生日志读取终止状态和真实相对 MIP Gap。"""
    content = path.read_text(encoding="utf-8", errors="replace")
    if "Result - Optimal solution found" in content:
        status = "kOptimal"
    elif "Result - Stopped on time limit" in content:
        status = "kTimeLimit"
    else:
        status = "kUnknown"

    objectives = re.findall(r"Objective value:\s*([-+0-9.eE]+)", content)
    bounds = re.findall(r"Lower bound:\s*([-+0-9.eE]+)", content)
    if objectives and bounds:
        objective = float(objectives[-1])
        bound = float(bounds[-1])
        gap = abs(objective - bound) / max(abs(objective), EPSILON)
        return status, gap
    if status == "kOptimal":
        return status, 0.0
    return status, float("inf")


def _validate_rates(values: Sequence[float], name: str) -> tuple[float, ...]:
    """检查情景比例并去重，保留用户给出的顺序。"""
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name}不能为空")
    if any(not -1.0 < value < 1.0 for value in result):
        raise ValueError(f"{name}必须位于 -1 和 1 之间")
    return tuple(dict.fromkeys(result))


def _stable_capacity(history: Sequence[int], percentile: float) -> float:
    positive = [value for value in history if value > 0]
    return float(np.quantile(positive, percentile)) if positive else 0.0


def _fulfillment_rates() -> dict[int, float]:
    """根据历史订货量和实际供货量计算供应商履约率。"""
    return {
        supply.provider_id: sum(supply.provide_history)
        / (sum(order.provide_history) + EPSILON)
        for order, supply in zip(std_received, std_ask)
    }


def _material_max_capacities(percentile: float) -> dict[str, float]:
    """计算价格弹性模型中每类材料的最大稳定供货能力。"""
    result = {material: 0.0 for material in MATERIAL_DEMANDS}
    for provider in std_ask:
        capacity = _stable_capacity(provider.provide_history, percentile)
        result[provider.product_type] = max(result[provider.product_type], capacity)
    return result


def _all_provider_candidates(
    supply_percentile: float,
) -> dict[str, list[SelectedProviderInfo]]:
    """按照能力和 TOPSIS 得分生成各材料的候选供应商排序。"""
    candidates = {material: [] for material in MATERIAL_DEMANDS}
    for index, provider in enumerate(std_ask):
        percentile_capacity = _stable_capacity(
            provider.provide_history, supply_percentile
        )
        corrected_capacity = min(percentile_capacity, TRANSFER_CAPACITY)
        candidates[provider.product_type].append(
            SelectedProviderInfo(
                provider_id=provider.provider_id,
                product_type=provider.product_type,
                percentile_capacity=percentile_capacity,
                corrected_capacity=corrected_capacity,
                effective_capacity=corrected_capacity,
                topsis_score=TOPSIS_RESULT.closeness[index],
                cumulative_capacity=0.0,
            )
        )
    for material in candidates:
        candidates[material].sort(
            key=lambda provider: (
                -provider.corrected_capacity,
                -provider.topsis_score,
                provider.provider_id,
            )
        )
    return candidates


def _build_robust_provider_pool(
    base: MinimumProviderResult,
    supply_percentile: float,
    supply_deviations: Sequence[float],
    loss_reductions: Sequence[float],
    reserve_count: int,
) -> list[SelectedProviderInfo]:
    """从第二题集合开始，补入能够覆盖最不利入库情景的候选供应商。"""
    if reserve_count < 0:
        raise ValueError("reserve_count 不能为负数")

    candidates = _all_provider_candidates(supply_percentile)
    selected_ids = {provider.provider_id for provider in base.selected_providers}
    selected_by_material = {
        material: [
            provider
            for provider in candidates[material]
            if provider.provider_id in selected_ids
        ]
        for material in MATERIAL_DEMANDS
    }

    worst_supply_factor = 1.0 + min(supply_deviations)
    best_supply_factor = 1.0 + max(supply_deviations)
    # 名义供货量受最有利情景的能力上限约束，因此最大名义量为 U_i/best_factor。
    worst_nominal_to_actual = worst_supply_factor / best_supply_factor
    worst_first_week_loss = max(base.transfer_loss_rates) * (1.0 - min(loss_reductions))

    for material, demand in MATERIAL_DEMANDS.items():
        chosen = selected_by_material[material]

        def robust_potential() -> float:
            return sum(
                worst_nominal_to_actual
                * (1.0 - worst_first_week_loss)
                * provider.corrected_capacity
                for provider in chosen
            )

        ordered = candidates[material]
        next_candidates = [
            provider
            for provider in ordered
            if provider.provider_id not in {item.provider_id for item in chosen}
        ]
        while robust_potential() + EPSILON < demand:
            if not next_candidates:
                raise ValueError(f"{material} 类供应商在最不利情景下无法满足需求")
            chosen.append(next_candidates.pop(0))

        for _ in range(min(reserve_count, len(next_candidates))):
            chosen.append(next_candidates.pop(0))

    selected = [
        provider
        for material in MATERIAL_DEMANDS
        for provider in selected_by_material[material]
    ]
    return selected


def _build_scenarios(
    supply_deviations: Sequence[float],
    loss_reductions: Sequence[float],
) -> list[RobustScenario]:
    return [
        RobustScenario(
            scenario_id=index + 1,
            supply_deviation=supply_deviation,
            loss_reduction=loss_reduction,
        )
        for index, (supply_deviation, loss_reduction) in enumerate(
            (supply_deviation, loss_reduction)
            for supply_deviation in supply_deviations
            for loss_reduction in loss_reductions
        )
    ]


def _build_supply_levels(
    providers: Sequence[SelectedProviderInfo],
    level_count: int,
    baseline_supplies: dict[int, float],
) -> dict[int, list[float]]:
    if level_count <= 0:
        raise ValueError("level_count 必须是正整数")
    levels: dict[int, list[float]] = {}
    for provider in providers:
        values = {
            provider.corrected_capacity * level / level_count
            for level in range(1, level_count + 1)
        }
        baseline_supply = baseline_supplies.get(provider.provider_id, 0.0)
        if baseline_supply > EPSILON:
            values.add(baseline_supply)
        levels[provider.provider_id] = sorted(values)
    return levels


def _solve_robust_baseline(
    providers: Sequence[SelectedProviderInfo],
    transfer_loss_rates: Sequence[float],
    supply_deviations: Sequence[float],
    loss_reductions: Sequence[float],
) -> tuple[dict[int, float], dict[int, int]]:
    """用一周连续 MILP 构造可重复使用的鲁棒可行初始方案。"""
    provider_ids = [provider.provider_id for provider in providers]
    provider_by_id = {provider.provider_id: provider for provider in providers}
    transfers = tuple(range(len(transfer_loss_rates)))
    worst_supply_factor = 1.0 + min(supply_deviations)
    best_supply_factor = 1.0 + max(supply_deviations)
    worst_reduction = min(loss_reductions)
    worst_losses = [
        loss_rate * (1.0 - worst_reduction) for loss_rate in transfer_loss_rates
    ]

    problem = pulp.LpProblem("robust_baseline", pulp.LpMinimize)
    active = {
        provider_id: pulp.LpVariable(f"u_{provider_id}", cat=pulp.LpBinary)
        for provider_id in provider_ids
    }
    supplies = {
        provider_id: pulp.LpVariable(
            f"s_{provider_id}",
            lowBound=0.0,
            upBound=provider_by_id[provider_id].corrected_capacity / best_supply_factor,
        )
        for provider_id in provider_ids
    }
    assignments = {
        (provider_id, transfer): pulp.LpVariable(
            f"y_{provider_id}_{transfer + 1}", cat=pulp.LpBinary
        )
        for provider_id in provider_ids
        for transfer in transfers
    }
    shipments = {
        (provider_id, transfer): pulp.LpVariable(
            f"x_{provider_id}_{transfer + 1}", lowBound=0.0
        )
        for provider_id in provider_ids
        for transfer in transfers
    }

    for provider_id in provider_ids:
        capacity = provider_by_id[provider_id].corrected_capacity
        problem += (
            pulp.lpSum(assignments[provider_id, transfer] for transfer in transfers)
            == active[provider_id]
        )
        problem += supplies[provider_id] <= capacity * active[provider_id]
        problem += (
            pulp.lpSum(shipments[provider_id, transfer] for transfer in transfers)
            == supplies[provider_id]
        )
        for transfer in transfers:
            problem += (
                shipments[provider_id, transfer]
                <= capacity * assignments[provider_id, transfer]
            )

    for transfer in transfers:
        problem += (
            best_supply_factor
            * pulp.lpSum(
                shipments[provider_id, transfer] for provider_id in provider_ids
            )
            <= TRANSFER_CAPACITY
        )

    for material, demand in MATERIAL_DEMANDS.items():
        material_ids = [
            provider.provider_id
            for provider in providers
            if provider.product_type == material
        ]
        problem += (
            worst_supply_factor
            * pulp.lpSum(
                (1.0 - worst_losses[transfer]) * shipments[provider_id, transfer]
                for provider_id in material_ids
                for transfer in transfers
            )
            >= demand + 1e-5
        )

    # 先减少总名义供货量，再用极小权重减少启用供应商数量。
    problem.setObjective(
        pulp.lpSum(supplies.values()) + 1e-4 * pulp.lpSum(active.values())
    )
    status_code = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status_code] != "Optimal":
        raise ValueError("当前候选供应商无法构造鲁棒可行初始方案")

    baseline_supplies = {
        provider_id: max(0.0, float(supplies[provider_id].varValue or 0.0))
        for provider_id in provider_ids
    }
    baseline_assignments = {
        provider_id: next(
            (
                transfer + 1
                for transfer in transfers
                if float(assignments[provider_id, transfer].varValue or 0.0) > 0.5
            ),
            0,
        )
        for provider_id in provider_ids
    }
    return baseline_supplies, baseline_assignments


def resolve(
    minimum_providers: MinimumProviderResult | None = None,
    supply_percentile: float = 0.95,
    loss_percentile: float = 0.75,
    supply_deviation: float = -0.05,
    loss_reduction: float = 0.01,
    elasticity: float = 0.10,
    level_count: int = 10,
    reserve_count: int = 1,
    transport_unit_cost: float = 0.0,
    storage_unit_cost: float = 0.0,
    primary_tolerance: float = 1e-5,
    time_limit: float | None = 60.0,
    relative_gap: float = 0.05,
    solver_messages: bool = False,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
) -> RobustPlanningResult:
    """求解一个给定情景下经济成本最小、再运输损耗最小的方案。

    默认基本情景取供货下降 5%、每周损耗率下降 1%。损耗率按照
    l_jt = l_j0 * (1-delta)^t 逐周下降。多情景比较由 sensitivity.py 完成。
    默认采用 10% 标准档位和 5% MIP Gap，以控制两阶段求解时间；如需
    更精细的论文结果，可显式提高 level_count 并降低 relative_gap。
    """
    if not 0.0 <= supply_percentile <= 1.0:
        raise ValueError("supply_percentile 必须位于 0 到 1 之间")
    if not 0.0 <= loss_percentile <= 1.0:
        raise ValueError("loss_percentile 必须位于 0 到 1 之间")
    if elasticity < 0.0:
        raise ValueError("elasticity 不能为负数")
    if transport_unit_cost < 0.0 or storage_unit_cost < 0.0:
        raise ValueError("运输和库存单位成本不能为负数")
    if primary_tolerance < 0.0:
        raise ValueError("primary_tolerance 不能为负数")
    if not 0.0 <= relative_gap <= 1.0:
        raise ValueError("relative_gap 必须位于 0 到 1 之间")

    supply_values = _validate_rates((supply_deviation,), "供货偏差")
    loss_values = _validate_rates((loss_reduction,), "损耗下降比例")
    if any(value < 0.0 for value in loss_values):
        raise ValueError("损耗下降比例不能为负数")

    base = minimum_providers or resolve_problem_two_step1(
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
    )
    providers = _build_robust_provider_pool(
        base,
        supply_percentile,
        supply_values,
        loss_values,
        reserve_count,
    )
    scenarios = _build_scenarios(supply_values, loss_values)
    baseline_supplies, baseline_assignments = _solve_robust_baseline(
        providers,
        base.transfer_loss_rates,
        supply_values,
        loss_values,
    )
    levels = _build_supply_levels(providers, level_count, baseline_supplies)
    fulfillment_rates = _fulfillment_rates()
    material_max_capacities = _material_max_capacities(supply_percentile)
    provider_by_id = {provider.provider_id: provider for provider in providers}
    provider_ids = [provider.provider_id for provider in providers]
    transfer_indices = tuple(range(len(base.transfer_loss_rates)))
    week_indices = tuple(range(WEEK_COUNT))
    scenario_indices = tuple(range(len(scenarios)))

    for provider_id in provider_ids:
        if fulfillment_rates[provider_id] <= EPSILON:
            raise ValueError(f"供应商 S{provider_id:03d} 的历史履约率为 0")

    loss_rates = {
        (scenario_index, transfer, week): (
            base.transfer_loss_rates[transfer]
            * (1.0 - scenarios[scenario_index].loss_reduction) ** (week + 1)
        )
        for scenario_index in scenario_indices
        for transfer in transfer_indices
        for week in week_indices
    }

    problem = pulp.LpProblem("robust_twelve_week_plan", pulp.LpMinimize)
    level_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    transfer_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    # 运输偏差是统一比例，因此只建立共同的名义运输量；
    # 各情景运输量由该变量乘以 (1 + xi) 得到，避免重复建立 9 份运输变量。
    shipment_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    inventory_vars: dict[tuple[int, str, int], pulp.LpVariable] = {}

    for provider_id in provider_ids:
        provider = provider_by_id[provider_id]
        for week in week_indices:
            for level_index in range(len(levels[provider_id])):
                level_vars[provider_id, level_index, week] = pulp.LpVariable(
                    f"z_{provider_id}_{level_index}_{week + 1}",
                    cat=pulp.LpBinary,
                )
            for transfer in transfer_indices:
                transfer_vars[provider_id, transfer, week] = pulp.LpVariable(
                    f"y_{provider_id}_{transfer + 1}_{week + 1}",
                    cat=pulp.LpBinary,
                )
                shipment_vars[provider_id, transfer, week] = pulp.LpVariable(
                    f"x_{provider_id}_{transfer + 1}_{week + 1}",
                    lowBound=0.0,
                    upBound=provider.corrected_capacity,
                )

    for scenario_index in scenario_indices:
        for material in MATERIAL_DEMANDS:
            for week in week_indices:
                inventory_vars[scenario_index, material, week] = pulp.LpVariable(
                    f"inventory_{scenario_index + 1}_{material}_{week + 1}",
                    lowBound=SAFETY_STOCKS[material],
                )

    theta_cost = pulp.LpVariable("theta_cost", lowBound=0.0)
    theta_loss = pulp.LpVariable("theta_loss", lowBound=0.0)
    theta_cost.setInitialValue(1e9)
    theta_loss.setInitialValue(1e9)

    cost_expressions: dict[int, pulp.LpAffineExpression] = {}
    loss_expressions: dict[int, pulp.LpAffineExpression] = {}
    for scenario_index, scenario in enumerate(scenarios):
        purchase_terms: list[pulp.LpAffineExpression] = []
        for provider_id in provider_ids:
            provider = provider_by_id[provider_id]
            material = provider.product_type
            base_price = BASE_PRICES[material]
            max_capacity = material_max_capacities[material]
            for level_index, nominal_supply in enumerate(levels[provider_id]):
                actual_supply = (1.0 + scenario.supply_deviation) * nominal_supply
                unit_price = base_price * (
                    1.0 + elasticity * (1.0 - actual_supply / max_capacity)
                )
                coefficient = unit_price * actual_supply
                for week in week_indices:
                    purchase_terms.append(
                        coefficient * level_vars[provider_id, level_index, week]
                    )

        scenario_shipments = (1.0 + scenario.supply_deviation) * pulp.lpSum(
            shipment_vars.values()
        )
        scenario_inventories = pulp.lpSum(
            variable
            for (current_scenario, _, _), variable in inventory_vars.items()
            if current_scenario == scenario_index
        )
        cost_expressions[scenario_index] = (
            pulp.lpSum(purchase_terms)
            + transport_unit_cost * scenario_shipments
            + storage_unit_cost * scenario_inventories
        )
        loss_expressions[scenario_index] = pulp.lpSum(
            loss_rates[scenario_index, transfer, week]
            * (1.0 + scenario.supply_deviation)
            * variable
            for (provider_id, transfer, week), variable in shipment_vars.items()
        )
        problem += (
            cost_expressions[scenario_index] <= theta_cost
        ), f"worst_cost_{scenario_index + 1}"
        problem += (
            loss_expressions[scenario_index] <= theta_loss
        ), f"worst_loss_{scenario_index + 1}"

    max_supply_factor = 1.0 + max(supply_values)
    for provider_id in provider_ids:
        provider = provider_by_id[provider_id]
        for week in week_indices:
            chosen_levels = pulp.lpSum(
                level_vars[provider_id, level_index, week]
                for level_index in range(len(levels[provider_id]))
            )
            chosen_transfers = pulp.lpSum(
                transfer_vars[provider_id, transfer, week]
                for transfer in transfer_indices
            )
            nominal_supply = pulp.lpSum(
                levels[provider_id][level_index]
                * level_vars[provider_id, level_index, week]
                for level_index in range(len(levels[provider_id]))
            )
            problem += chosen_levels <= 1, f"one_level_{provider_id}_{week + 1}"
            problem += (
                chosen_transfers == chosen_levels
            ), f"one_transfer_{provider_id}_{week + 1}"
            problem += (
                max_supply_factor * nominal_supply <= provider.corrected_capacity
            ), f"robust_supply_capacity_{provider_id}_{week + 1}"
            nominal_shipment = pulp.lpSum(
                shipment_vars[provider_id, transfer, week]
                for transfer in transfer_indices
            )
            problem += (
                nominal_shipment == nominal_supply
            ), f"shipment_supply_{provider_id}_{week + 1}"
            for transfer in transfer_indices:
                problem += (
                    shipment_vars[provider_id, transfer, week]
                    <= provider.corrected_capacity
                    * transfer_vars[provider_id, transfer, week]
                ), f"shipment_link_{provider_id}_{transfer + 1}_{week + 1}"

    for scenario_index, scenario in enumerate(scenarios):
        supply_factor = 1.0 + scenario.supply_deviation
        for transfer in transfer_indices:
            for week in week_indices:
                problem += (
                    (
                        supply_factor
                        * pulp.lpSum(
                            shipment_vars[provider_id, transfer, week]
                            for provider_id in provider_ids
                        )
                        <= TRANSFER_CAPACITY
                    ),
                    f"transfer_capacity_{scenario_index + 1}_{transfer + 1}_{week + 1}",
                )

    for scenario_index, scenario in enumerate(scenarios):
        supply_factor = 1.0 + scenario.supply_deviation
        for material, demand in MATERIAL_DEMANDS.items():
            material_provider_ids = [
                provider.provider_id
                for provider in providers
                if provider.product_type == material
            ]
            for week in week_indices:
                receipt = pulp.lpSum(
                    supply_factor
                    * (1.0 - loss_rates[scenario_index, transfer, week])
                    * shipment_vars[provider_id, transfer, week]
                    for provider_id in material_provider_ids
                    for transfer in transfer_indices
                )
                previous_inventory: float | pulp.LpVariable = (
                    SAFETY_STOCKS[material]
                    if week == 0
                    else inventory_vars[scenario_index, material, week - 1]
                )
                problem += (
                    inventory_vars[scenario_index, material, week]
                    == previous_inventory + receipt - demand
                ), f"inventory_{scenario_index + 1}_{material}_{week + 1}"

    # 将连续鲁棒可行方案作为 MILP 起点，避免求解器长时间找不到第一组整数解。
    for provider_id in provider_ids:
        baseline_supply = baseline_supplies[provider_id]
        if baseline_supply > EPSILON:
            level_index = min(
                range(len(levels[provider_id])),
                key=lambda index: abs(levels[provider_id][index] - baseline_supply),
            )
        else:
            level_index = None
        for week in week_indices:
            if level_index is not None:
                level_vars[provider_id, level_index, week].setInitialValue(1)
            transfer_id = baseline_assignments[provider_id]
            if transfer_id > 0 and baseline_supply > EPSILON:
                transfer_vars[provider_id, transfer_id - 1, week].setInitialValue(1)
                shipment_variable = shipment_vars[
                    provider_id, transfer_id - 1, week
                ]
                shipment_variable.setInitialValue(
                    min(baseline_supply, float(shipment_variable.upBound or baseline_supply))
                )

    for scenario_index, scenario in enumerate(scenarios):
        inventories = dict(SAFETY_STOCKS)
        for week in week_indices:
            receipts = {material: 0.0 for material in MATERIAL_DEMANDS}
            for provider in providers:
                provider_id = provider.provider_id
                transfer_id = baseline_assignments[provider_id]
                if transfer_id <= 0:
                    continue
                transfer = transfer_id - 1
                actual_supply = (1.0 + scenario.supply_deviation) * baseline_supplies[
                    provider_id
                ]
                receipts[provider.product_type] += (
                    1.0 - loss_rates[scenario_index, transfer, week]
                ) * actual_supply
            for material, demand in MATERIAL_DEMANDS.items():
                inventories[material] = (
                    inventories[material] + receipts[material] - demand
                )
                inventory_vars[scenario_index, material, week].setInitialValue(
                    max(SAFETY_STOCKS[material], inventories[material])
                )

    cbc_logs: list[Path] = []

    def make_solver(phase: str) -> pulp.LpSolver:
        log_path = Path(f"{problem.name}-{phase}-cbc.log")
        cbc_logs.append(log_path)
        return pulp.PULP_CBC_CMD(
            msg=solver_messages,
            timeLimit=time_limit,
            gapRel=relative_gap,
            warmStart=True,
            keepFiles=True,
            logPath=str(log_path),
        )

    problem.setObjective(theta_cost)
    primary_code = problem.solve(make_solver("primary"))
    primary_pulp_status = pulp.LpStatus[primary_code]
    primary_status, primary_gap = _parse_cbc_log(cbc_logs[-1])
    if primary_pulp_status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"第一阶段未找到可行整数解：{primary_status}")
    primary_theta = float(pulp.value(theta_cost))  # type: ignore

    for key, variable in level_vars.items():
        chosen_value = 1 if float(variable.varValue or 0.0) > 0.5 else 0
        problem += variable == chosen_value, (
            f"fix_level_{key[0]}_{key[1]}_{key[2] + 1}"
        )
    allowed_cost = primary_theta + max(primary_tolerance, abs(primary_theta) * 1e-8)
    problem += theta_cost <= allowed_cost, "keep_robust_economic_cost"

    problem.setObjective(theta_loss)
    secondary_code = problem.solve(make_solver("secondary"))
    secondary_pulp_status = pulp.LpStatus[secondary_code]
    secondary_status, secondary_gap = _parse_cbc_log(cbc_logs[-1])
    if secondary_pulp_status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"第二阶段未找到可行整数解：{secondary_status}")

    orders: list[RobustProviderOrder] = []
    nominal_supplies: dict[tuple[int, int], float] = {}
    common_transfers: dict[tuple[int, int], int] = {}
    for week in week_indices:
        for provider_id in provider_ids:
            nominal_supply = sum(
                levels[provider_id][level_index]
                * (
                    1
                    if float(level_vars[provider_id, level_index, week].varValue or 0.0)
                    > 0.5
                    else 0
                )
                for level_index in range(len(levels[provider_id]))
            )
            nominal_supplies[provider_id, week] = nominal_supply
            transfer_id = next(
                (
                    transfer + 1
                    for transfer in transfer_indices
                    if float(transfer_vars[provider_id, transfer, week].varValue or 0.0)
                    > 0.5
                ),
                0,
            )
            common_transfers[provider_id, week] = transfer_id
            if nominal_supply <= EPSILON:
                continue
            provider = provider_by_id[provider_id]
            orders.append(
                RobustProviderOrder(
                    week=week + 1,
                    provider_id=provider_id,
                    product_type=provider.product_type,
                    nominal_supply=nominal_supply,
                    order_quantity=nominal_supply / fulfillment_rates[provider_id],
                    supply_capacity=provider.corrected_capacity,
                    transfer_id=transfer_id,
                )
            )

    scenario_results: list[RobustScenarioResult] = []
    for scenario_index, scenario in enumerate(scenarios):
        weeks: list[RobustScenarioWeek] = []
        total_purchase = 0.0
        total_transport = 0.0
        total_storage = 0.0
        total_loss = 0.0
        for week in week_indices:
            actual_supplies: dict[int, float] = {}
            transfer_loads = [0.0] * len(transfer_indices)
            receipts = {material: 0.0 for material in MATERIAL_DEMANDS}
            purchase_cost = 0.0
            transport_loss = 0.0
            for provider_id in provider_ids:
                provider = provider_by_id[provider_id]
                actual_supply = (1.0 + scenario.supply_deviation) * nominal_supplies[
                    provider_id, week
                ]
                actual_supplies[provider_id] = actual_supply
                transfer_id = common_transfers[provider_id, week]
                if actual_supply <= EPSILON or transfer_id == 0:
                    continue
                transfer = transfer_id - 1
                transfer_loads[transfer] += actual_supply
                loss_rate = loss_rates[scenario_index, transfer, week]
                received = (1.0 - loss_rate) * actual_supply
                receipts[provider.product_type] += received
                transport_loss += loss_rate * actual_supply
                base_price = BASE_PRICES[provider.product_type]
                max_capacity = material_max_capacities[provider.product_type]
                unit_price = base_price * (
                    1.0 + elasticity * (1.0 - actual_supply / max_capacity)
                )
                purchase_cost += unit_price * actual_supply

            ending_inventories = {
                material: float(
                    pulp.value(inventory_vars[scenario_index, material, week])  # type: ignore
                )
                for material in MATERIAL_DEMANDS
            }
            total_shipment = sum(actual_supplies.values())
            transport_cost = transport_unit_cost * total_shipment
            storage_cost = storage_unit_cost * sum(ending_inventories.values())
            total_cost = purchase_cost + transport_cost + storage_cost
            weeks.append(
                RobustScenarioWeek(
                    scenario_id=scenario.scenario_id,
                    week=week + 1,
                    actual_supplies=actual_supplies,
                    transfer_loads=transfer_loads,
                    material_receipts=receipts,
                    ending_inventories=ending_inventories,
                    purchase_cost=purchase_cost,
                    transport_cost=transport_cost,
                    storage_cost=storage_cost,
                    total_cost=total_cost,
                    transport_loss=transport_loss,
                )
            )
            total_purchase += purchase_cost
            total_transport += transport_cost
            total_storage += storage_cost
            total_loss += transport_loss
        scenario_results.append(
            RobustScenarioResult(
                scenario=scenario,
                total_purchase_cost=total_purchase,
                total_transport_cost=total_transport,
                total_storage_cost=total_storage,
                total_cost=total_purchase + total_transport + total_storage,
                total_transport_loss=total_loss,
                weeks=weeks,
            )
        )

    final_theta_cost = max(result.total_cost for result in scenario_results)
    final_theta_loss = max(result.total_transport_loss for result in scenario_results)
    for suffix in ("mps", "mst", "sol"):
        Path(f"{problem.name}-pulp.{suffix}").unlink(missing_ok=True)
    for log_path in cbc_logs:
        log_path.unlink(missing_ok=True)

    result = RobustPlanningResult(
        providers=provider_ids,
        scenarios=scenarios,
        orders=orders,
        scenario_results=scenario_results,
        theta_cost=final_theta_cost,
        theta_loss=final_theta_loss,
        primary_status=primary_status,
        secondary_status=secondary_status,
        primary_mip_gap=primary_gap,
        secondary_mip_gap=secondary_gap,
        is_optimal=(
            primary_status == "kOptimal"
            and secondary_status == "kOptimal"
            and primary_gap <= relative_gap + 1e-9
            and secondary_gap <= relative_gap + 1e-9
        ),
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
        level_count=level_count,
    )
    if output_path is not None:
        write_report(result, Path(output_path))
    return result


def _format_provider_supply(actual_supplies: dict[int, float]) -> str:
    """将情景中非零供应商实际供货量格式化为单行文本。"""
    values = [
        f"S{provider_id:03d}={supply:.3f}"
        for provider_id, supply in sorted(actual_supplies.items())
        if supply > EPSILON
    ]
    return ", ".join(values) if values else "无"


def build_report(result: RobustPlanningResult) -> str:
    """生成第三题鲁棒订货规划的详细 UTF-8 文本报告。"""
    lines: list[str] = [
        "第三题：供货下降 5%、每周损耗下降 1% 基本情景求解结果",
        "=" * 88,
        "",
        "一、模型设置",
        f"供应能力分位数：{result.supply_percentile:.2%}",
        f"基准损耗率分位数：{result.loss_percentile:.2%}",
        f"供货档位数：{result.level_count}",
        f"情景数量：{len(result.scenarios)}",
        "供货偏差情景："
        + ", ".join(
            f"{scenario.supply_deviation:+.1%}" for scenario in result.scenarios[::3]
        ),
        "每周损耗下降情景："
        + ", ".join(
            f"{value:.1%}"
            for value in sorted(
                {scenario.loss_reduction for scenario in result.scenarios}
            )
        ),
        "损耗率计算：l_jt = l_j0 * (1 - delta)^t。",
        "本文件只包含一个基本情景；其他情景由 sensitivity.py 单独求解和比较。",
        "",
        "二、求解状态",
        f"第一阶段状态：{result.primary_status}",
        f"第一阶段 MIP Gap：{result.primary_mip_gap:.6%}",
        f"第二阶段状态：{result.secondary_status}",
        f"第二阶段 MIP Gap：{result.secondary_mip_gap:.6%}",
        f"是否达到当前 Gap 要求：{'是' if result.is_optimal else '否'}",
        f"最坏情景经济成本：{result.theta_cost:.3f}",
        f"最坏情景运输损耗：{result.theta_loss:.3f}",
        "",
            "三、基本情景候选供应商",
        f"候选供应商总数：{len(result.providers)}",
    ]

    provider_materials = {
        provider.provider_id: provider.product_type for provider in std_ask
    }
    by_material: dict[str, list[int]] = defaultdict(list)
    for provider_id in result.providers:
        by_material[provider_materials[provider_id]].append(provider_id)
    for material in MATERIAL_DEMANDS:
        provider_ids = sorted(by_material.get(material, []))
        lines.append(
            f"{material} 类：{len(provider_ids)} 家，"
            + (", ".join(f"S{provider_id:03d}" for provider_id in provider_ids) or "无")
        )
    active_provider_ids = sorted({order.provider_id for order in result.orders})
    lines.append(
        f"12 周中至少供货一次的供应商：{len(active_provider_ids)} 家，"
        + ", ".join(f"S{provider_id:03d}" for provider_id in active_provider_ids)
    )

    lines.extend(
        [
            "",
            "四、未来 12 周共同订货与转运计划",
            "说明：该表不依赖情景，是企业事前制定的共同计划。",
            f"{'周':>3} {'供应商':>8} {'材料':>6} {'名义供货':>14} "
            f"{'订货量':>14} {'能力上限':>14} {'转运商':>10}",
        ]
    )
    for order in result.orders:
        lines.append(
            f"{order.week:>3} S{order.provider_id:03d}"
            f"{order.product_type:>6}"
            f"{order.nominal_supply:>14.3f}"
            f"{order.order_quantity:>14.3f}"
            f"{order.supply_capacity:>14.3f}"
            f"{('T' + str(order.transfer_id)) if order.transfer_id else '未分配':>10}"
        )

    lines.extend(
        [
            "",
            "五、基本情景 12 周汇总",
            f"{'情景':>4} {'供货偏差':>10} {'损耗下降':>10} {'总成本':>16} "
            f"{'总运输损耗':>16} {'最小库存裕量':>16} {'最大转运负载':>16}",
        ]
    )
    for scenario_result in result.scenario_results:
        minimum_margin = min(
            week.ending_inventories[material] - SAFETY_STOCKS[material]
            for week in scenario_result.weeks
            for material in MATERIAL_DEMANDS
        )
        maximum_load = max(max(week.transfer_loads) for week in scenario_result.weeks)
        scenario = scenario_result.scenario
        lines.append(
            f"{scenario.scenario_id:>4}"
            f"{scenario.supply_deviation:>10.1%}"
            f"{scenario.loss_reduction:>10.1%}"
            f"{scenario_result.total_cost:>16.3f}"
            f"{scenario_result.total_transport_loss:>16.3f}"
            f"{minimum_margin:>16.3f}"
            f"{maximum_load:>16.3f}"
        )

    lines.extend(["", "六、基本情景逐周库存、损耗和运输负载"])
    for scenario_result in result.scenario_results:
        scenario = scenario_result.scenario
        lines.extend(
            [
                "",
                f"情景 {scenario.scenario_id}：供货偏差={scenario.supply_deviation:+.1%}，"
                f"每周损耗下降={scenario.loss_reduction:.1%}",
                f"{'周':>3} {'A入库':>12} {'B入库':>12} {'C入库':>12} "
                f"{'A库存':>12} {'B库存':>12} {'C库存':>12} "
                f"{'本周成本':>14} {'本周损耗':>14} {'最大负载':>14}",
            ]
        )
        for week in scenario_result.weeks:
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

    lines.extend(["", "七、基本情景各供应商实际供货量"])
    for scenario_result in result.scenario_results:
        scenario = scenario_result.scenario
        lines.extend(
            [
                "",
                f"情景 {scenario.scenario_id}：供货偏差={scenario.supply_deviation:+.1%}，"
                f"每周损耗下降={scenario.loss_reduction:.1%}",
                "格式：周次：供应商=实际供货量；实际供货量为运输前数量。",
            ]
        )
        for week in scenario_result.weeks:
            lines.append(
                f"第 {week.week:02d} 周：{_format_provider_supply(week.actual_supplies)}"
            )

    lines.extend(
        [
            "",
            "八、约束核验",
            "1. 每个情景均按照共同订货方案计算实际供货量。",
            "2. 每个供应商每周最多使用一家转运商。",
            "3. 每个情景每家转运商每周运输负载均不超过 6200。",
            "4. 每个情景 A、B、C 三类材料均满足库存平衡方程。",
            "5. 每个情景每周末库存均不低于三周安全库存。",
            "6. 第一阶段最小化最坏情景经济成本，第二阶段最小化最坏情景运输损耗。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(result: RobustPlanningResult, output_path: Path) -> Path:
    """将详细结果写入 UTF-8-BOM 文本文件并返回文件路径。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(result), encoding="utf-8-sig")
    return output_path


def debug(result: RobustPlanningResult) -> None:
    """打印第三题鲁棒求解状态、共同方案和情景汇总。"""
    print(f"候选供应商数量：{len(result.providers)}")
    print(f"情景数量：{len(result.scenarios)}")
    print(f"第一阶段状态：{result.primary_status}")
    print(f"第一阶段 MIP Gap：{result.primary_mip_gap:.6%}")
    print(f"第二阶段状态：{result.secondary_status}")
    print(f"第二阶段 MIP Gap：{result.secondary_mip_gap:.6%}")
    print(f"是否达到要求的最优性间隙：{'是' if result.is_optimal else '否'}")
    print(f"最坏情景经济成本：{result.theta_cost:.3f}")
    print(f"最坏情景运输损耗：{result.theta_loss:.3f}")
    print()
    print(
        f"{'情景':>4}{'供货偏差':>12}{'每周损耗下降':>14}"
        f"{'总成本':>16}{'运输损耗':>14}{'最低库存裕量':>16}"
    )
    for scenario_result in result.scenario_results:
        minimum_margin = min(
            week.ending_inventories[material] - SAFETY_STOCKS[material]
            for week in scenario_result.weeks
            for material in MATERIAL_DEMANDS
        )
        scenario = scenario_result.scenario
        print(
            f"{scenario.scenario_id:>4}"
            f"{scenario.supply_deviation:>12.1%}"
            f"{scenario.loss_reduction:>14.1%}"
            f"{scenario_result.total_cost:>16.3f}"
            f"{scenario_result.total_transport_loss:>14.3f}"
            f"{minimum_margin:>16.3f}"
        )


if __name__ == "__main__":
    debug(resolve())
