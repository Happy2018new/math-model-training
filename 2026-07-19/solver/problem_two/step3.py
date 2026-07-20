"""使用离散供货档位求解未来 12 周 0-1 整数规划。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pulp

from ..utils.read import std_ask, std_received
from .define import (
    IntegerProgrammingResult,
    MinimumProviderResult,
    TwelveWeekOrderPlanResult,
    WeeklyMaterialInventory,
    WeeklyOrderPlan,
    WeeklyProviderOrder,
)
from .step1 import MATERIAL_DEMANDS, TRANSFER_CAPACITY, resolve as resolve_step1
from .step2 import (
    EPSILON,
    SAFETY_STOCKS,
    WEEK_COUNT,
    resolve as resolve_baseline,
)

BASE_PRICES = {"A": 1.25, "B": 1.15, "C": 1.00}


def _parse_cbc_log(path: Path) -> tuple[str, float]:
    """从 CBC 原生日志读取终止状态和真实相对 MIP Gap。"""
    content = path.read_text(encoding="utf-8", errors="replace")
    if "Result - Optimal solution found" in content:
        status = "kOptimal"
    elif "Result - Stopped on time limit" in content:
        status = "kTimeLimit"
    else:
        status = "kUnknown"

    objective_matches = re.findall(r"Objective value:\s*([-+0-9.eE]+)", content)
    bound_matches = re.findall(r"Lower bound:\s*([-+0-9.eE]+)", content)
    if objective_matches and bound_matches:
        objective = float(objective_matches[-1])
        bound = float(bound_matches[-1])
        gap = abs(objective - bound) / max(abs(objective), EPSILON)
        return status, gap
    if status == "kOptimal":
        return status, 0.0
    return status, float("inf")


def _fulfillment_rates() -> dict[int, float]:
    """计算 318 家供应商的历史履约率。"""
    return {
        supply.provider_id: sum(supply.provide_history)
        / (sum(order.provide_history) + EPSILON)
        for order, supply in zip(std_received, std_ask)
    }


def _material_max_capacities(percentile: float) -> dict[str, float]:
    """计算价格模型中 A、B、C 各自的最大稳定供货能力。"""
    result = {material: 0.0 for material in MATERIAL_DEMANDS}
    for provider in std_ask:
        positive = [value for value in provider.provide_history if value > 0]
        capacity = float(np.quantile(positive, percentile)) if positive else 0.0
        result[provider.product_type] = max(result[provider.product_type], capacity)
    return result


def _build_supply_levels(
    minimum_providers: MinimumProviderResult,
    baseline: TwelveWeekOrderPlanResult,
    level_count: int,
) -> dict[int, list[float]]:
    """生成标准供货档位，并加入基础方案中的实际供货量。"""
    if level_count <= 0:
        raise ValueError("level_count 必须是正整数")

    baseline_supplies: dict[int, set[float]] = defaultdict(set)
    for week in baseline.weeks:
        for order in week.provider_orders:
            if order.expected_supply > EPSILON:
                baseline_supplies[order.provider_id].add(order.expected_supply)

    levels: dict[int, list[float]] = {}
    for provider in minimum_providers.selected_providers:
        values = {
            provider.corrected_capacity * level / level_count
            for level in range(1, level_count + 1)
        }
        values.update(baseline_supplies[provider.provider_id])
        levels[provider.provider_id] = sorted(
            value
            for value in values
            if EPSILON < value <= provider.corrected_capacity + 1e-7
        )
    return levels


def resolve(
    minimum_providers: MinimumProviderResult | None = None,
    supply_percentile: float = 0.95,
    loss_percentile: float = 0.75,
    elasticity: float = 0.10,
    level_count: int = 20,
    transport_unit_cost: float = 0.0,
    storage_unit_cost: float = 0.0,
    primary_tolerance: float = 1e-5,
    time_limit: float | None = 300.0,
    relative_gap: float = 0.005,
    solver_messages: bool = False,
    solver_backend: str = "cbc",
) -> IntegerProgrammingResult:
    """先最小化经济成本，再在最优成本容差内最小化运输损耗。

    标准档位数默认为 20，即每档为供应能力的 5%。基础贪心方案中的供货
    数量会作为额外档位加入，以保证整数模型至少包含该已知可行方案。
    """
    if minimum_providers is None:
        minimum_providers = resolve_step1(supply_percentile, loss_percentile)
    if elasticity < 0:
        raise ValueError("elasticity 不能为负数")
    if transport_unit_cost < 0 or storage_unit_cost < 0:
        raise ValueError("运输和库存单位成本不能为负数")
    if primary_tolerance < 0:
        raise ValueError("primary_tolerance 不能为负数")
    if not 0 <= relative_gap <= 1:
        raise ValueError("relative_gap 必须位于 0 到 1 之间")
    if solver_backend not in {"highs", "cbc"}:
        raise ValueError("solver_backend 只能是 'highs' 或 'cbc'")

    baseline = resolve_baseline(
        minimum_providers=minimum_providers,
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
        elasticity=elasticity,
        transport_unit_cost=transport_unit_cost,
        storage_unit_cost=storage_unit_cost,
    )
    levels = _build_supply_levels(minimum_providers, baseline, level_count)
    fulfillment_rates = _fulfillment_rates()
    material_max_capacities = _material_max_capacities(supply_percentile)
    providers = minimum_providers.selected_providers
    provider_by_id = {provider.provider_id: provider for provider in providers}
    provider_ids = [provider.provider_id for provider in providers]
    transfer_indices = range(len(minimum_providers.transfer_loss_rates))
    week_indices = range(WEEK_COUNT)

    problem = pulp.LpProblem("twelve_week_order_plan", pulp.LpMinimize)

    level_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    transfer_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    shipment_vars: dict[tuple[int, int, int], pulp.LpVariable] = {}
    inventory_vars: dict[tuple[str, int], pulp.LpVariable] = {}

    for provider_id in provider_ids:
        for week in week_indices:
            for level_index in range(len(levels[provider_id])):
                level_vars[provider_id, level_index, week] = pulp.LpVariable(
                    f"z_{provider_id}_{level_index}_{week + 1}",
                    cat=pulp.LpBinary,
                )
                level_vars[provider_id, level_index, week].setInitialValue(0)
            for transfer in transfer_indices:
                transfer_vars[provider_id, transfer, week] = pulp.LpVariable(
                    f"y_{provider_id}_{transfer + 1}_{week + 1}",
                    cat=pulp.LpBinary,
                )
                transfer_vars[provider_id, transfer, week].setInitialValue(0)
                shipment_vars[provider_id, transfer, week] = pulp.LpVariable(
                    f"x_{provider_id}_{transfer + 1}_{week + 1}",
                    lowBound=0.0,
                    upBound=provider_by_id[provider_id].corrected_capacity,
                )
                shipment_vars[provider_id, transfer, week].setInitialValue(0.0)

    for material in MATERIAL_DEMANDS:
        for week in week_indices:
            inventory_vars[material, week] = pulp.LpVariable(
                f"inventory_{material}_{week + 1}",
                lowBound=SAFETY_STOCKS[material],
            )

    purchase_terms: list[pulp.LpAffineExpression] = []
    for provider_id in provider_ids:
        provider = provider_by_id[provider_id]
        material = provider.product_type
        base_price = BASE_PRICES[material]
        max_capacity = material_max_capacities[material]
        for level_index, supply in enumerate(levels[provider_id]):
            unit_price = base_price * (
                1 + elasticity * (1 - supply / max_capacity)
            )
            purchase_cost = unit_price * supply
            for week in week_indices:
                purchase_terms.append(
                    purchase_cost * level_vars[provider_id, level_index, week]
                )

    purchase_expression = pulp.lpSum(purchase_terms)
    transport_expression = transport_unit_cost * pulp.lpSum(shipment_vars.values())
    storage_expression = storage_unit_cost * pulp.lpSum(inventory_vars.values())
    primary_expression = (
        purchase_expression + transport_expression + storage_expression
    )
    loss_expression = pulp.lpSum(
        minimum_providers.transfer_loss_rates[transfer] * variable
        for (provider_id, transfer, week), variable in shipment_vars.items()
    )

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
            total_shipment = pulp.lpSum(
                shipment_vars[provider_id, transfer, week]
                for transfer in transfer_indices
            )
            selected_supply = pulp.lpSum(
                levels[provider_id][level_index]
                * level_vars[provider_id, level_index, week]
                for level_index in range(len(levels[provider_id]))
            )

            problem += chosen_levels <= 1, f"one_level_{provider_id}_{week + 1}"
            problem += (
                chosen_transfers == chosen_levels
            ), f"one_transfer_{provider_id}_{week + 1}"
            problem += (
                total_shipment == selected_supply
            ), f"shipment_equals_level_{provider_id}_{week + 1}"
            for transfer in transfer_indices:
                problem += (
                    shipment_vars[provider_id, transfer, week]
                    <= provider.corrected_capacity
                    * transfer_vars[provider_id, transfer, week]
                ), f"shipment_link_{provider_id}_{transfer + 1}_{week + 1}"

    for transfer in transfer_indices:
        for week in week_indices:
            problem += (
                pulp.lpSum(
                    shipment_vars[provider_id, transfer, week]
                    for provider_id in provider_ids
                )
                <= TRANSFER_CAPACITY
            ), f"transfer_capacity_{transfer + 1}_{week + 1}"

    for material, demand in MATERIAL_DEMANDS.items():
        material_provider_ids = [
            provider.provider_id
            for provider in providers
            if provider.product_type == material
        ]
        for week in week_indices:
            actual_receipt = pulp.lpSum(
                (1 - minimum_providers.transfer_loss_rates[transfer])
                * shipment_vars[provider_id, transfer, week]
                for provider_id in material_provider_ids
                for transfer in transfer_indices
            )
            previous_inventory: float | pulp.LpVariable = (
                SAFETY_STOCKS[material]
                if week == 0
                else inventory_vars[material, week - 1]
            )
            problem += (
                inventory_vars[material, week]
                == previous_inventory + actual_receipt - demand
            ), f"inventory_balance_{material}_{week + 1}"

    baseline_upper_tolerance = max(
        primary_tolerance,
        abs(baseline.total_cost) * 1e-8,
    )
    problem += (
        primary_expression <= baseline.total_cost + baseline_upper_tolerance
    ), "baseline_cost_upper_bound"

    baseline_orders = {
        (order.provider_id, week.week - 1): order
        for week in baseline.weeks
        for order in week.provider_orders
    }
    baseline_inventories = {
        (state.product_type, week.week - 1): state.ending_inventory
        for week in baseline.weeks
        for state in week.material_inventories
    }
    for provider_id in provider_ids:
        for week in week_indices:
            baseline_order = baseline_orders[provider_id, week]
            if baseline_order.expected_supply <= EPSILON:
                continue
            closest_level = min(
                range(len(levels[provider_id])),
                key=lambda index: abs(
                    levels[provider_id][index] - baseline_order.expected_supply
                ),
            )
            level_vars[provider_id, closest_level, week].setInitialValue(1)
            transfer_index = baseline_order.transfer_id - 1
            transfer_vars[provider_id, transfer_index, week].setInitialValue(1)
            shipment_vars[provider_id, transfer_index, week].setInitialValue(
                min(
                    baseline_order.expected_supply,
                    provider_by_id[provider_id].corrected_capacity,
                )
            )
    for key, value in baseline_inventories.items():
        inventory_vars[key].setInitialValue(value)

    cbc_log_paths: list[Path] = []

    def make_solver(phase: str) -> pulp.LpSolver:
        if solver_backend == "highs":
            return pulp.HiGHS(
                msg=solver_messages,
                timeLimit=time_limit,
                gapRel=relative_gap,
            )
        log_path = Path(f"{problem.name}-{phase}-cbc.log")
        cbc_log_paths.append(log_path)
        return pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=time_limit,
            gapRel=relative_gap,
            warmStart=True,
            keepFiles=True,
            logPath=str(log_path),
        )

    solver_name = "HiGHS" if solver_backend == "highs" else "CBC"

    problem.setObjective(primary_expression)
    primary_status_code = problem.solve(make_solver("primary"))
    primary_pulp_status = pulp.LpStatus[primary_status_code]
    if solver_backend == "highs":
        primary_status = str(problem.solverModel.getModelStatus()).split(".")[-1]
        primary_info = problem.solverModel.getInfo()
        primary_mip_gap = float(primary_info.mip_gap)
    else:
        primary_status, primary_mip_gap = _parse_cbc_log(cbc_log_paths[-1])
    if primary_pulp_status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"第一阶段未找到可行整数解：{primary_status}")
    primary_best_cost = float(pulp.value(primary_expression))

    # “据此制定损耗最少的转运方案”：固定第一阶段得到的订货档位，
    # 第二阶段只重新安排转运商，采购数量保持为第一阶段的经济方案。
    for key, variable in level_vars.items():
        chosen_value = 1 if float(variable.varValue or 0.0) > 0.5 else 0
        problem += variable == chosen_value, (
            f"fix_order_level_{key[0]}_{key[1]}_{key[2] + 1}"
        )

    allowed_primary_cost = primary_best_cost + max(
        primary_tolerance,
        abs(primary_best_cost) * 1e-8,
    )
    problem += (
        primary_expression <= allowed_primary_cost
    ), "fix_primary_cost_for_loss_optimization"
    problem.setObjective(loss_expression)
    secondary_status_code = problem.solve(make_solver("secondary"))
    secondary_pulp_status = pulp.LpStatus[secondary_status_code]
    if solver_backend == "highs":
        secondary_status = str(problem.solverModel.getModelStatus()).split(".")[-1]
        secondary_info = problem.solverModel.getInfo()
        secondary_mip_gap = float(secondary_info.mip_gap)
    else:
        secondary_status, secondary_mip_gap = _parse_cbc_log(cbc_log_paths[-1])
    if secondary_pulp_status not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"第二阶段未找到可行整数解：{secondary_status}")

    weeks: list[WeeklyOrderPlan] = []
    total_purchase_cost = 0.0
    total_transport_cost = 0.0
    total_storage_cost = 0.0
    total_transport_loss = 0.0
    successful_allocation_count = 0

    for week in week_indices:
        provider_orders: list[WeeklyProviderOrder] = []
        transfer_loads = [0.0] * len(minimum_providers.transfer_loss_rates)
        material_receipts = {material: 0.0 for material in MATERIAL_DEMANDS}

        for provider_id in provider_ids:
            provider = provider_by_id[provider_id]
            shipments = [
                max(0.0, float(pulp.value(shipment_vars[provider_id, transfer, week])))
                for transfer in transfer_indices
            ]
            expected_supply = sum(shipments)
            active_transfer = next(
                (
                    transfer
                    for transfer, shipment in enumerate(shipments)
                    if shipment > 1e-6
                ),
                None,
            )
            if active_transfer is None:
                transfer_id = 0
                transfer_loss_rate = 0.0
                actual_received = 0.0
            else:
                transfer_id = active_transfer + 1
                transfer_loss_rate = minimum_providers.transfer_loss_rates[
                    active_transfer
                ]
                actual_received = (1 - transfer_loss_rate) * expected_supply
                transfer_loads[active_transfer] += expected_supply
                successful_allocation_count += 1

            fulfillment_rate = fulfillment_rates[provider_id]
            order_quantity = (
                expected_supply / fulfillment_rate
                if expected_supply > EPSILON
                else 0.0
            )
            material = provider.product_type
            base_price = BASE_PRICES[material]
            unit_price = base_price * (
                1
                + elasticity
                * (
                    1
                    - expected_supply / material_max_capacities[material]
                )
            )
            purchase_cost = unit_price * expected_supply
            material_receipts[material] += actual_received
            provider_orders.append(
                WeeklyProviderOrder(
                    week=week + 1,
                    provider_id=provider_id,
                    product_type=material,
                    order_quantity=order_quantity,
                    expected_supply=expected_supply,
                    supply_capacity=provider.corrected_capacity,
                    unit_price=unit_price,
                    purchase_cost=purchase_cost,
                    transfer_id=transfer_id,
                    transfer_loss_rate=transfer_loss_rate,
                    actual_received=actual_received,
                )
            )

        material_inventories: list[WeeklyMaterialInventory] = []
        for material, demand in MATERIAL_DEMANDS.items():
            previous_inventory = (
                SAFETY_STOCKS[material]
                if week == 0
                else next(
                    state.ending_inventory
                    for state in weeks[-1].material_inventories
                    if state.product_type == material
                )
            )
            ending_inventory = float(pulp.value(inventory_vars[material, week]))
            required_receipt = max(
                0.0,
                demand + SAFETY_STOCKS[material] - previous_inventory,
            )
            material_inventories.append(
                WeeklyMaterialInventory(
                    week=week + 1,
                    product_type=material,
                    demand=demand,
                    required_receipt=required_receipt,
                    actual_received=material_receipts[material],
                    ending_inventory=ending_inventory,
                    safety_stock=SAFETY_STOCKS[material],
                )
            )

        purchase_cost = sum(order.purchase_cost for order in provider_orders)
        total_shipment = sum(order.expected_supply for order in provider_orders)
        transport_cost = transport_unit_cost * total_shipment
        storage_cost = storage_unit_cost * sum(
            state.ending_inventory for state in material_inventories
        )
        transport_loss = sum(
            order.expected_supply - order.actual_received
            for order in provider_orders
        )
        total_cost = purchase_cost + transport_cost + storage_cost
        weeks.append(
            WeeklyOrderPlan(
                week=week + 1,
                provider_orders=provider_orders,
                material_inventories=material_inventories,
                transfer_loads=transfer_loads,
                purchase_cost=purchase_cost,
                transport_cost=transport_cost,
                storage_cost=storage_cost,
                total_cost=total_cost,
                transport_loss=transport_loss,
            )
        )
        total_purchase_cost += purchase_cost
        total_transport_cost += transport_cost
        total_storage_cost += storage_cost
        total_transport_loss += transport_loss

    optimized_plan = TwelveWeekOrderPlanResult(
        elasticity=elasticity,
        weeks=weeks,
        total_purchase_cost=total_purchase_cost,
        total_transport_cost=total_transport_cost,
        total_storage_cost=total_storage_cost,
        total_cost=total_purchase_cost + total_transport_cost + total_storage_cost,
        total_transport_loss=total_transport_loss,
        successful_allocation_count=successful_allocation_count,
    )
    cost_saving = baseline.total_cost - optimized_plan.total_cost
    cost_saving_rate = (
        cost_saving / baseline.total_cost if baseline.total_cost > EPSILON else 0.0
    )
    if solver_backend == "cbc":
        for suffix in ("mps", "mst", "sol"):
            Path(f"{problem.name}-pulp.{suffix}").unlink(missing_ok=True)
        for log_path in cbc_log_paths:
            log_path.unlink(missing_ok=True)
    return IntegerProgrammingResult(
        optimized_plan=optimized_plan,
        baseline_plan=baseline,
        solver_name=solver_name,
        primary_solver_status=primary_status,
        secondary_solver_status=secondary_status,
        is_optimal=(
            primary_status in {"Optimal", "kOptimal"}
            and secondary_status in {"Optimal", "kOptimal"}
            and primary_mip_gap <= relative_gap + 1e-9
            and secondary_mip_gap <= relative_gap + 1e-9
        ),
        level_count=level_count,
        primary_best_cost=primary_best_cost,
        primary_mip_gap=primary_mip_gap,
        secondary_mip_gap=secondary_mip_gap,
        cost_saving=cost_saving,
        cost_saving_rate=cost_saving_rate,
    )


def debug(result: IntegerProgrammingResult) -> None:
    """打印整数规划状态、成本改进和逐周库存摘要。"""
    print(f"求解器：{result.solver_name}")
    print(f"第一阶段状态：{result.primary_solver_status}")
    print(f"第一阶段 MIP Gap：{result.primary_mip_gap:.6%}")
    print(f"第二阶段状态：{result.secondary_solver_status}")
    print(f"第二阶段 MIP Gap：{result.secondary_mip_gap:.6%}")
    print(f"是否达到要求的最优性间隙：{'是' if result.is_optimal else '否'}")
    print(f"标准档位数：{result.level_count}")
    print(f"基础方案成本：{result.baseline_plan.total_cost:.3f}")
    print(f"优化方案成本：{result.optimized_plan.total_cost:.3f}")
    print(f"成本节省：{result.cost_saving:.3f}")
    print(f"成本节省率：{result.cost_saving_rate:.4%}")
    print(f"优化方案运输损耗：{result.optimized_plan.total_transport_loss:.3f}")
    print(
        f"{'周':>3}{'A库存':>12}{'B库存':>12}{'C库存':>12}"
        f"{'采购成本':>14}{'运输损耗':>12}"
    )
    for week in result.optimized_plan.weeks:
        inventories = {
            state.product_type: state.ending_inventory
            for state in week.material_inventories
        }
        print(
            f"{week.week:>3}{inventories['A']:>12.3f}{inventories['B']:>12.3f}"
            f"{inventories['C']:>12.3f}{week.purchase_cost:>14.3f}"
            f"{week.transport_loss:>12.3f}"
        )


if __name__ == "__main__":
    debug(resolve())
