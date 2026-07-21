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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "problems" / "two" / "integer_programming_result.txt"
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
            unit_price = base_price * (1 + elasticity * (1 - supply / max_capacity))
            purchase_cost = unit_price * supply
            for week in week_indices:
                purchase_terms.append(
                    purchase_cost * level_vars[provider_id, level_index, week]
                )

    purchase_expression = pulp.lpSum(purchase_terms)
    transport_expression = transport_unit_cost * pulp.lpSum(shipment_vars.values())
    storage_expression = storage_unit_cost * pulp.lpSum(inventory_vars.values())
    primary_expression = purchase_expression + transport_expression + storage_expression
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
        primary_status = str(problem.solverModel.getModelStatus()).split(".")[-1]  # type: ignore
        primary_info = problem.solverModel.getInfo()  # type: ignore
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
        secondary_status = str(problem.solverModel.getModelStatus()).split(".")[-1]  # type: ignore
        secondary_info = problem.solverModel.getInfo()  # type: ignore
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
                max(0.0, float(pulp.value(shipment_vars[provider_id, transfer, week])))  # type: ignore
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
                expected_supply / fulfillment_rate if expected_supply > EPSILON else 0.0
            )
            material = provider.product_type
            base_price = BASE_PRICES[material]
            unit_price = base_price * (
                1
                + elasticity * (1 - expected_supply / material_max_capacities[material])
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
            ending_inventory = float(pulp.value(inventory_vars[material, week]))  # type: ignore
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
            order.expected_supply - order.actual_received for order in provider_orders
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
        minimum_provider_result=minimum_providers,
        supply_percentile=supply_percentile,
        loss_percentile=loss_percentile,
        transport_unit_cost=transport_unit_cost,
        storage_unit_cost=storage_unit_cost,
        primary_tolerance=primary_tolerance,
        relative_gap=relative_gap,
        time_limit=time_limit,
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


def _format_time_limit(value: float | None) -> str:
    """将求解时间上限转换为报告文本。"""
    return "无限制" if value is None else f"{value:.1f} 秒/阶段"


def _format_week_ranges(weeks: list[int]) -> str:
    """将周次压缩为 1-4, 7, 9-12 形式。"""
    if not weeks:
        return "无"
    ordered = sorted(set(weeks))
    ranges: list[str] = []
    start = previous = ordered[0]
    for week in ordered[1:]:
        if week == previous + 1:
            previous = week
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = week
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def build_detailed_report(result: IntegerProgrammingResult) -> str:
    """生成第二题 0-1 整数规划的完整 UTF-8 文本报告。"""
    optimized = result.optimized_plan
    baseline = result.baseline_plan
    minimum_providers = result.minimum_provider_result
    selected = sorted(
        minimum_providers.selected_providers,
        key=lambda provider: (provider.product_type, provider.provider_id),
    )
    provider_ids = [provider.provider_id for provider in selected]
    fulfillment_rates = _fulfillment_rates()
    loss_rates = minimum_providers.transfer_loss_rates

    orders_by_provider: dict[int, list[WeeklyProviderOrder]] = defaultdict(list)
    for week in optimized.weeks:
        for order in week.provider_orders:
            orders_by_provider[order.provider_id].append(order)

    lines = [
        "第二题第二问：未来 12 周 0-1 整数规划详细方案",
        "=" * 116,
        "",
        "一、模型与求解参数",
        f"供货能力分位数：{result.supply_percentile:.2%}",
        f"转运损耗率分位数：{result.loss_percentile:.2%}",
        f"价格弹性系数：{optimized.elasticity:.4f}",
        f"标准供货档位数：{result.level_count}（标准档距为供货能力的 {1 / result.level_count:.2%}）",
        "注：为保留已知可行基础方案，基础方案中的供货量也会作为额外可选档位。",
        f"单位运输成本：{result.transport_unit_cost:.6f}",
        f"单位库存成本：{result.storage_unit_cost:.6f}",
        f"第一阶段成本容差：{result.primary_tolerance:.8f}",
        f"目标相对 MIP Gap：{result.relative_gap:.4%}",
        f"求解时间上限：{_format_time_limit(result.time_limit)}",
        f"求解器：{result.solver_name}",
        f"材料每周需求：A={MATERIAL_DEMANDS['A']:.3f}, "
        f"B={MATERIAL_DEMANDS['B']:.3f}, C={MATERIAL_DEMANDS['C']:.3f}",
        f"三周安全库存：A={SAFETY_STOCKS['A']:.3f}, "
        f"B={SAFETY_STOCKS['B']:.3f}, C={SAFETY_STOCKS['C']:.3f}",
        f"单家转运商每周容量上限：{TRANSFER_CAPACITY:.3f}",
        "材料基准价格："
        + ", ".join(
            f"{material}={price:.4f}" for material, price in BASE_PRICES.items()
        ),
        "转运商预测损耗率："
        + ", ".join(
            f"T{index}={rate:.4%}" for index, rate in enumerate(loss_rates, 1)
        ),
        "",
        "二、求解状态与基础方案对比",
        f"第一阶段状态：{result.primary_solver_status}",
        f"第一阶段 MIP Gap：{result.primary_mip_gap:.6%}",
        f"第二阶段状态：{result.secondary_solver_status}",
        f"第二阶段 MIP Gap：{result.secondary_mip_gap:.6%}",
        f"是否达到设定 Gap 要求：{'是' if result.is_optimal else '否'}",
        "说明：Gap 达标表示在当前精度要求下得到可接受近优解，不代表零 Gap 的精确全局最优证明。",
        "",
        f"{'指标':<18}{'基础方案':>18}{'0-1优化方案':>18}{'优化-基础':>18}",
    ]
    comparison_rows = (
        ("采购成本", baseline.total_purchase_cost, optimized.total_purchase_cost),
        ("运输成本", baseline.total_transport_cost, optimized.total_transport_cost),
        ("库存成本", baseline.total_storage_cost, optimized.total_storage_cost),
        ("总经济成本", baseline.total_cost, optimized.total_cost),
        ("运输损耗", baseline.total_transport_loss, optimized.total_transport_loss),
    )
    for label, baseline_value, optimized_value in comparison_rows:
        lines.append(
            f"{label:<18}{baseline_value:>18.3f}{optimized_value:>18.3f}"
            f"{optimized_value - baseline_value:>18.3f}"
        )
    lines.extend(
        [
            f"成本节省：{result.cost_saving:.3f}",
            f"成本节省率：{result.cost_saving_rate:.4%}",
            f"运输损耗减少：{baseline.total_transport_loss - optimized.total_transport_loss:.3f}",
            "运输损耗降低率："
            f"{(baseline.total_transport_loss - optimized.total_transport_loss) / baseline.total_transport_loss:.4%}",
            f"基础方案非零供货分配次数：{baseline.successful_allocation_count}",
            f"优化方案非零供货分配次数：{optimized.successful_allocation_count}",
            "",
            "三、入选供应商基础参数与 12 周汇总",
            f"入选供应商数量：{len(selected)}",
            f"{'供应商':<8}{'材料':<6}{'分位数能力':>14}{'修正能力':>14}"
            f"{'履约率':>12}{'启用周数':>10}{'累计订货':>14}{'预计供货':>14}"
            f"{'实际入库':>14}{'运输损耗':>14}{'加权均价':>12}{'采购成本':>14}{'使用转运商':>16}",
        ]
    )
    for provider in selected:
        provider_orders = orders_by_provider[provider.provider_id]
        active_orders = [
            order for order in provider_orders if order.expected_supply > EPSILON
        ]
        total_order = sum(order.order_quantity for order in active_orders)
        total_supply = sum(order.expected_supply for order in active_orders)
        total_received = sum(order.actual_received for order in active_orders)
        total_purchase = sum(order.purchase_cost for order in active_orders)
        weighted_price = total_purchase / total_supply if total_supply > EPSILON else 0.0
        transfers = sorted(
            {order.transfer_id for order in active_orders if order.transfer_id > 0}
        )
        transfer_text = ",".join(f"T{transfer}" for transfer in transfers) or "未启用"
        lines.append(
            f"S{provider.provider_id:03d}    {provider.product_type:<6}"
            f"{provider.percentile_capacity:>14.3f}{provider.corrected_capacity:>14.3f}"
            f"{fulfillment_rates[provider.provider_id]:>12.4%}{len(active_orders):>10}"
            f"{total_order:>14.3f}{total_supply:>14.3f}{total_received:>14.3f}"
            f"{total_supply - total_received:>14.3f}{weighted_price:>12.6f}"
            f"{total_purchase:>14.3f}{transfer_text:>16}"
        )

    lines.extend(
        [
            "",
            "四、未来 12 周逐家供应商订货与转运方案",
            "说明：订货量为企业向供应商下达的数量；预计供货为考虑历史履约率后的运输前数量；"
            "实际入库为扣除转运损耗后的数量。",
            "非零行表示该供应商当周选中一个供货档位，并选中一家转运商；未启用名单中的供应商当周 0-1 选择值均为 0。",
        ]
    )
    for week in optimized.weeks:
        active_orders = sorted(
            (
                order
                for order in week.provider_orders
                if order.expected_supply > EPSILON
            ),
            key=lambda order: (order.product_type, order.provider_id),
        )
        active_ids = {order.provider_id for order in active_orders}
        inactive = [
            f"S{provider_id:03d}" for provider_id in provider_ids if provider_id not in active_ids
        ]
        lines.extend(
            [
                "",
                f"第 {week.week:02d} 周",
                f"{'供应商':<8}{'材料':<6}{'订货量':>14}{'预计供货':>14}"
                f"{'能力上限':>14}{'能力占用':>12}{'单位价格':>12}{'采购成本':>14}"
                f"{'转运商':>10}{'损耗率':>12}{'实际入库':>14}{'运输损耗':>14}",
            ]
        )
        for order in active_orders:
            capacity_rate = order.expected_supply / order.supply_capacity
            lines.append(
                f"S{order.provider_id:03d}    {order.product_type:<6}"
                f"{order.order_quantity:>14.3f}{order.expected_supply:>14.3f}"
                f"{order.supply_capacity:>14.3f}{capacity_rate:>12.2%}"
                f"{order.unit_price:>12.6f}{order.purchase_cost:>14.3f}"
                f"{f'T{order.transfer_id}':>10}{order.transfer_loss_rate:>12.4%}"
                f"{order.actual_received:>14.3f}"
                f"{order.expected_supply - order.actual_received:>14.3f}"
            )
        lines.append("本周未启用供应商：" + (", ".join(inactive) or "无"))

    lines.extend(
        [
            "",
            "五、三类材料需求、入库、库存与安全裕量的逐周变化",
            f"{'周':>4}{'材料':>6}{'生产需求':>14}{'所需入库':>14}{'实际入库':>14}"
            f"{'期末库存':>14}{'安全库存':>14}{'安全裕量':>14}",
        ]
    )
    minimum_margin = (float("inf"), 0, "")
    inventory_balance_ok = True
    previous_inventory = dict(SAFETY_STOCKS)
    for week in optimized.weeks:
        for state in sorted(
            week.material_inventories, key=lambda value: value.product_type
        ):
            margin = state.ending_inventory - state.safety_stock
            minimum_margin = min(
                minimum_margin,
                (margin, week.week, state.product_type),
            )
            expected_inventory = (
                previous_inventory[state.product_type]
                + state.actual_received
                - state.demand
            )
            inventory_balance_ok &= abs(expected_inventory - state.ending_inventory) <= 1e-4
            previous_inventory[state.product_type] = state.ending_inventory
            lines.append(
                f"{week.week:>4}{state.product_type:>6}{state.demand:>14.3f}"
                f"{state.required_receipt:>14.3f}{state.actual_received:>14.3f}"
                f"{state.ending_inventory:>14.3f}{state.safety_stock:>14.3f}"
                f"{margin:>14.3f}"
            )

    lines.extend(
        [
            "",
            "六、采购成本、运输成本、库存成本与运输损耗的逐周变化",
            f"{'周':>4}{'采购成本':>16}{'运输成本':>16}{'库存成本':>16}"
            f"{'本周总成本':>16}{'本周运输损耗':>18}{'累计总成本':>16}{'累计损耗':>16}",
        ]
    )
    cumulative_cost = 0.0
    cumulative_loss = 0.0
    for week in optimized.weeks:
        cumulative_cost += week.total_cost
        cumulative_loss += week.transport_loss
        lines.append(
            f"{week.week:>4}{week.purchase_cost:>16.3f}{week.transport_cost:>16.3f}"
            f"{week.storage_cost:>16.3f}{week.total_cost:>16.3f}"
            f"{week.transport_loss:>18.3f}{cumulative_cost:>16.3f}"
            f"{cumulative_loss:>16.3f}"
        )

    lines.extend(
        [
            "",
            "七、未来 12 周各转运商分配与负载",
            "说明：下表按周列出实际启用的转运商；未启用的转运商在每周末单独列出。",
        ]
    )
    transfer_totals = [0.0] * len(loss_rates)
    transfer_received = [0.0] * len(loss_rates)
    transfer_assignment_counts = [0] * len(loss_rates)
    transfer_active_weeks: list[set[int]] = [set() for _ in loss_rates]
    transfer_load_records: list[tuple[float, int, int]] = []
    transfer_capacity_ok = True
    for week in optimized.weeks:
        lines.extend(
            [
                "",
                f"第 {week.week:02d} 周",
                f"{'转运商':<8}{'分配供应商':<42}{'运输负载':>14}{'容量占用率':>14}"
                f"{'预测损耗率':>14}{'实际入库':>14}{'运输损耗':>14}",
            ]
        )
        active_transfers: set[int] = set()
        for transfer_id, load in enumerate(week.transfer_loads, 1):
            assigned_orders = [
                order
                for order in week.provider_orders
                if order.transfer_id == transfer_id
                and order.expected_supply > EPSILON
            ]
            if not assigned_orders:
                continue
            active_transfers.add(transfer_id)
            provider_text = ",".join(
                f"S{order.provider_id:03d}" for order in assigned_orders
            )
            received = sum(order.actual_received for order in assigned_orders)
            loss = load - received
            transfer_totals[transfer_id - 1] += load
            transfer_received[transfer_id - 1] += received
            transfer_assignment_counts[transfer_id - 1] += len(assigned_orders)
            transfer_active_weeks[transfer_id - 1].add(week.week)
            transfer_load_records.append((load, week.week, transfer_id))
            transfer_capacity_ok &= load <= TRANSFER_CAPACITY + 1e-5
            lines.append(
                f"T{transfer_id:<7}{provider_text:<42}{load:>14.3f}"
                f"{load / TRANSFER_CAPACITY:>14.2%}{loss_rates[transfer_id - 1]:>14.4%}"
                f"{received:>14.3f}{loss:>14.3f}"
            )
        idle = [
            f"T{transfer_id}"
            for transfer_id in range(1, len(loss_rates) + 1)
            if transfer_id not in active_transfers
        ]
        lines.append("本周未启用转运商：" + (", ".join(idle) or "无"))

    lines.extend(
        [
            "",
            "八、各转运商 12 周汇总",
            f"{'转运商':<8}{'预测损耗率':>14}{'启用周数':>12}{'供应商分配次数':>16}"
            f"{'累计运输负载':>16}{'累计实际入库':>16}{'累计运输损耗':>16}",
        ]
    )
    for transfer_id, rate in enumerate(loss_rates, 1):
        index = transfer_id - 1
        lines.append(
            f"T{transfer_id:<7}{rate:>14.4%}{len(transfer_active_weeks[index]):>12}"
            f"{transfer_assignment_counts[index]:>16}{transfer_totals[index]:>16.3f}"
            f"{transfer_received[index]:>16.3f}"
            f"{transfer_totals[index] - transfer_received[index]:>16.3f}"
        )

    all_orders = [order for week in optimized.weeks for order in week.provider_orders]
    single_transfer_ok = all(
        order.transfer_id in range(1, len(loss_rates) + 1)
        for order in all_orders
        if order.expected_supply > EPSILON
    )
    supply_capacity_ok = all(
        order.expected_supply <= order.supply_capacity + 1e-5
        for order in all_orders
    )
    safety_stock_ok = minimum_margin[0] >= -1e-5
    cost_total_ok = abs(cumulative_cost - optimized.total_cost) <= 1e-4
    loss_total_ok = abs(cumulative_loss - optimized.total_transport_loss) <= 1e-4
    maximum_transfer_load = max(record[0] for record in transfer_load_records)
    maximum_transfer_points = [
        (week, transfer_id)
        for load, week, transfer_id in transfer_load_records
        if abs(load - maximum_transfer_load) <= 1e-5
    ]
    maximum_weeks_by_transfer: dict[int, list[int]] = defaultdict(list)
    for week, transfer_id in maximum_transfer_points:
        maximum_weeks_by_transfer[transfer_id].append(week)
    maximum_transfer_text = ", ".join(
        f"T{transfer_id}（第 {_format_week_ranges(weeks)} 周）"
        for transfer_id, weeks in sorted(maximum_weeks_by_transfer.items())
    )
    lines.extend(
        [
            "",
            "九、约束与汇总数据核验",
            f"1. 每家供应商每周非零供货均仅分配一家转运商：{'通过' if single_transfer_ok else '未通过'}",
            f"2. 所有供应商预计供货量均不超过修正能力上限：{'通过' if supply_capacity_ok else '未通过'}",
            f"3. 所有转运商每周运输负载均不超过 {TRANSFER_CAPACITY:.3f}："
            f"{'通过' if transfer_capacity_ok else '未通过'}",
            f"   最大周负载为 {maximum_transfer_load:.3f}，"
            f"容量占用率为 {maximum_transfer_load / TRANSFER_CAPACITY:.4%}。",
            "   达到最大负载的周次与转运商：" + maximum_transfer_text,
            f"4. A、B、C 三类材料逐周库存平衡关系：{'通过' if inventory_balance_ok else '未通过'}",
            f"5. 所有周次期末库存均不低于三周安全库存：{'通过' if safety_stock_ok else '未通过'}",
            f"   最小安全库存裕量为 {minimum_margin[0]:.3f}，出现在第 {minimum_margin[1]} 周"
            f" {minimum_margin[2]} 类材料。",
            f"6. 逐周成本之和与 12 周总成本一致：{'通过' if cost_total_ok else '未通过'}",
            f"7. 逐周运输损耗之和与 12 周总损耗一致：{'通过' if loss_total_ok else '未通过'}",
            "",
            "十、关键参数变化的阅读说明",
            "1. 供应商的订货量经历史履约率修正后形成预计供货量，因此订货量通常大于预计供货量。",
            "2. 单位价格随供应商当周所选供货档位变化；供货量越接近同类材料的基准能力，价格加成越小。",
            "3. 预计供货量经对应转运商的预测损耗率修正后形成实际入库量。",
            "4. 期末库存由上周库存、本周实际入库量和生产需求共同决定；安全裕量用于衡量库存距离安全线的缓冲空间。",
            "5. 转运商的周负载为当周分配给该转运商的所有供应商预计供货量之和。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_detailed_report(
    result: IntegerProgrammingResult,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """将 0-1 整数规划详细方案写入 UTF-8 TXT。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_detailed_report(result), encoding="utf-8")
    return output_path


def run(
    output_path: Path = DEFAULT_OUTPUT_PATH,
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
    """求解第二题 0-1 整数规划并写出详细方案。"""
    result = resolve(
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
    write_detailed_report(result, output_path)
    return result


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
    solved_result = run()
    debug(solved_result)
    print(f"详细方案已写入：{DEFAULT_OUTPUT_PATH}")
