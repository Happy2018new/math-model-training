"""根据已选供应商制定未来 12 周订货、转运和库存计划。"""

from __future__ import annotations

from ..utils.define import ProviderInfo
from ..utils.read import std_ask, std_received
from .define import (
    MinimumProviderResult,
    TwelveWeekOrderPlanResult,
    WeeklyMaterialInventory,
    WeeklyOrderPlan,
    WeeklyProviderOrder,
)
from .step1 import MATERIAL_DEMANDS, TRANSFER_CAPACITY, resolve as resolve_step1
from .utils import calculate_provider_prices

SAFETY_STOCKS = {"A": 18600.0, "B": 20100.0, "C": 23400.0}
WEEK_COUNT = 12
EPSILON = 1e-8


def _fulfillment_rates() -> dict[int, float]:
    """根据 240 周历史订货量和实际供货量计算供应商履约率。"""
    return {
        supply.provider_id: sum(supply.provide_history)
        / (sum(order.provide_history) + EPSILON)
        for order, supply in zip(std_received, std_ask)
    }


def resolve(
    minimum_providers: MinimumProviderResult | None = None,
    supply_percentile: float = 0.95,
    loss_percentile: float = 0.75,
    elasticity: float = 0.10,
    transport_unit_cost: float = 0.0,
    storage_unit_cost: float = 0.0,
) -> TwelveWeekOrderPlanResult:
    """使用库存驱动的贪心算法生成未来 12 周经济订货计划。

    题目没有给出运输和库存的具体单位费用，因此二者通过参数传入，默认
    为 0。供应商在同类材料中按可稳定供货能力从大到小使用，以集中采购
    并利用价格弹性模型中的批量价格优惠。
    """
    if minimum_providers is None:
        minimum_providers = resolve_step1(supply_percentile, loss_percentile)
    if elasticity < 0:
        raise ValueError("elasticity 不能为负数")
    if transport_unit_cost < 0 or storage_unit_cost < 0:
        raise ValueError("运输和库存单位成本不能为负数")

    fulfillment_rates = _fulfillment_rates()
    selected_by_material = {
        material: sorted(
            (
                provider
                for provider in minimum_providers.selected_providers
                if provider.product_type == material
            ),
            key=lambda provider: (
                -provider.corrected_capacity,
                -provider.topsis_score,
                provider.provider_id,
            ),
        )
        for material in MATERIAL_DEMANDS
    }

    inventories = SAFETY_STOCKS.copy()
    raw_weeks: list[dict[str, object]] = []
    supply_histories = {
        provider.provider_id: [0.0] * WEEK_COUNT
        for provider in minimum_providers.selected_providers
    }

    for week_index in range(WEEK_COUNT):
        required_receipts = {
            material: max(
                0.0,
                demand + SAFETY_STOCKS[material] - inventories[material],
            )
            for material, demand in MATERIAL_DEMANDS.items()
        }
        shipments = {
            provider.provider_id: 0.0
            for provider in minimum_providers.selected_providers
        }
        actual_receipts = {material: 0.0 for material in MATERIAL_DEMANDS}

        for material, required_receipt in required_receipts.items():
            remaining = required_receipt
            for provider in selected_by_material[material]:
                if remaining <= EPSILON:
                    break
                loss_rate = provider.transfer_loss_rate
                max_received = (1 - loss_rate) * provider.corrected_capacity
                received = min(remaining, max_received)
                shipment = received / (1 - loss_rate)
                shipments[provider.provider_id] = shipment
                actual_receipts[material] += received
                remaining -= received
            if remaining > 1e-6:
                raise ValueError(
                    f"第 {week_index + 1} 周 {material} 类材料供货能力不足，"
                    f"仍缺少 {remaining:.6f}"
                )

        transfer_loads = [0.0] * len(minimum_providers.transfer_loss_rates)
        for provider in minimum_providers.selected_providers:
            shipment = shipments[provider.provider_id]
            transfer_loads[provider.transfer_id - 1] += shipment
            supply_histories[provider.provider_id][week_index] = shipment
        if any(load > TRANSFER_CAPACITY + 1e-6 for load in transfer_loads):
            raise ValueError(f"第 {week_index + 1} 周存在转运商负载超过 6200")

        ending_inventories: dict[str, float] = {}
        for material, demand in MATERIAL_DEMANDS.items():
            ending_inventory = (
                inventories[material] + actual_receipts[material] - demand
            )
            if ending_inventory + 1e-6 < SAFETY_STOCKS[material]:
                raise ValueError(f"第 {week_index + 1} 周 {material} 类库存低于安全线")
            ending_inventories[material] = ending_inventory

        raw_weeks.append(
            {
                "required_receipts": required_receipts,
                "shipments": shipments,
                "actual_receipts": actual_receipts,
                "ending_inventories": ending_inventories,
                "transfer_loads": transfer_loads,
            }
        )
        inventories = ending_inventories

    predicted_supplies = [
        ProviderInfo(
            provider_id=provider.provider_id,
            product_type=provider.product_type,
            provide_history=supply_histories[provider.provider_id],  # type: ignore[arg-type]
        )
        for provider in minimum_providers.selected_providers
    ]
    price_results = calculate_provider_prices(
        predicted_supplies,
        capacity_percentile=supply_percentile,
        elasticity=elasticity,
    )
    prices_by_provider = {result.provider_id: result for result in price_results}

    weeks: list[WeeklyOrderPlan] = []
    total_purchase_cost = 0.0
    total_transport_cost = 0.0
    total_storage_cost = 0.0
    total_transport_loss = 0.0
    successful_allocation_count = 0

    for week_index, raw_week in enumerate(raw_weeks):
        shipments = raw_week["shipments"]
        provider_orders: list[WeeklyProviderOrder] = []
        for provider in minimum_providers.selected_providers:
            shipment = shipments[provider.provider_id]  # type: ignore[index]
            fulfillment_rate = fulfillment_rates[provider.provider_id]
            if shipment > EPSILON and fulfillment_rate <= EPSILON:
                raise ValueError(f"供应商 S{provider.provider_id:03d} 的历史履约率为 0")
            order_quantity = (
                shipment / fulfillment_rate if shipment > EPSILON else 0.0
            )
            price_result = prices_by_provider[provider.provider_id]
            unit_price = price_result.unit_prices[week_index]
            purchase_cost = price_result.purchase_costs[week_index]
            actual_received = (1 - provider.transfer_loss_rate) * shipment
            if shipment > EPSILON:
                successful_allocation_count += 1
            provider_orders.append(
                WeeklyProviderOrder(
                    week=week_index + 1,
                    provider_id=provider.provider_id,
                    product_type=provider.product_type,
                    order_quantity=order_quantity,
                    expected_supply=shipment,
                    supply_capacity=provider.corrected_capacity,
                    unit_price=unit_price,
                    purchase_cost=purchase_cost,
                    transfer_id=provider.transfer_id,
                    transfer_loss_rate=provider.transfer_loss_rate,
                    actual_received=actual_received,
                )
            )

        material_inventories = [
            WeeklyMaterialInventory(
                week=week_index + 1,
                product_type=material,
                demand=MATERIAL_DEMANDS[material],
                required_receipt=raw_week["required_receipts"][material],  # type: ignore[index]
                actual_received=raw_week["actual_receipts"][material],  # type: ignore[index]
                ending_inventory=raw_week["ending_inventories"][material],  # type: ignore[index]
                safety_stock=SAFETY_STOCKS[material],
            )
            for material in MATERIAL_DEMANDS
        ]
        purchase_cost = sum(order.purchase_cost for order in provider_orders)
        total_shipment = sum(order.expected_supply for order in provider_orders)
        transport_cost = transport_unit_cost * total_shipment
        storage_cost = storage_unit_cost * sum(
            material.ending_inventory for material in material_inventories
        )
        transport_loss = sum(
            order.expected_supply - order.actual_received
            for order in provider_orders
        )
        weekly_total_cost = purchase_cost + transport_cost + storage_cost

        weeks.append(
            WeeklyOrderPlan(
                week=week_index + 1,
                provider_orders=provider_orders,
                material_inventories=material_inventories,
                transfer_loads=list(raw_week["transfer_loads"]),  # type: ignore[arg-type]
                purchase_cost=purchase_cost,
                transport_cost=transport_cost,
                storage_cost=storage_cost,
                total_cost=weekly_total_cost,
                transport_loss=transport_loss,
            )
        )
        total_purchase_cost += purchase_cost
        total_transport_cost += transport_cost
        total_storage_cost += storage_cost
        total_transport_loss += transport_loss

    return TwelveWeekOrderPlanResult(
        elasticity=elasticity,
        weeks=weeks,
        total_purchase_cost=total_purchase_cost,
        total_transport_cost=total_transport_cost,
        total_storage_cost=total_storage_cost,
        total_cost=total_purchase_cost + total_transport_cost + total_storage_cost,
        total_transport_loss=total_transport_loss,
        successful_allocation_count=successful_allocation_count,
    )


def compare_elasticities(
    minimum_providers: MinimumProviderResult | None = None,
    elasticities: tuple[float, ...] = (0.05, 0.10, 0.15),
    **kwargs: float,
) -> list[TwelveWeekOrderPlanResult]:
    """分别计算低、中、高价格弹性下的未来 12 周计划成本。"""
    if minimum_providers is None:
        minimum_providers = resolve_step1(
            kwargs.get("supply_percentile", 0.95),
            kwargs.get("loss_percentile", 0.75),
        )
    return [
        resolve(
            minimum_providers=minimum_providers,
            elasticity=elasticity,
            **kwargs,
        )
        for elasticity in elasticities
    ]


def debug(result: TwelveWeekOrderPlanResult) -> None:
    """打印未来 12 周计划的库存、运输和成本摘要。"""
    print(f"价格弹性：{result.elasticity:.2f}")
    print(
        f"{'周':>3}{'A入库':>12}{'A库存':>12}{'B入库':>12}{'B库存':>12}"
        f"{'C入库':>12}{'C库存':>12}{'采购成本':>14}{'运输损耗':>12}"
    )
    for week in result.weeks:
        states = {state.product_type: state for state in week.material_inventories}
        print(
            f"{week.week:>3}"
            f"{states['A'].actual_received:>12.3f}{states['A'].ending_inventory:>12.3f}"
            f"{states['B'].actual_received:>12.3f}{states['B'].ending_inventory:>12.3f}"
            f"{states['C'].actual_received:>12.3f}{states['C'].ending_inventory:>12.3f}"
            f"{week.purchase_cost:>14.3f}{week.transport_loss:>12.3f}"
        )
    print(f"12 周采购成本：{result.total_purchase_cost:.3f}")
    print(f"12 周运输成本：{result.total_transport_cost:.3f}")
    print(f"12 周库存成本：{result.total_storage_cost:.3f}")
    print(f"12 周总成本：{result.total_cost:.3f}")
    print(f"12 周运输总损耗：{result.total_transport_loss:.3f}")
    print(f"非零供货分配次数：{result.successful_allocation_count}")
