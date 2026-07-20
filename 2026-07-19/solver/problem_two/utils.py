"""根据历史数据预测供应商供货量和转运商损耗率。"""

import numpy as np
from ..utils.define import ProviderInfo
from ..utils.read import std_ask, std_received, std_transfer
from .define import ProviderPriceInfo


def predict_provider_supply(percentile: float) -> list[ProviderInfo]:
    """根据历史 CSV 数据预测各供应商未来 12 周的实际供货量。

    先根据 240 周历史数据计算供应商履约率，再以历史非零供货量的给定
    分位数作为稳定供货能力上限。未来订货量取过去 5 年对应周的平均值。
    ``percentile`` 位于 0 到 1 之间，例如 95% 分位数应传入 ``0.95``。
    """
    if not 0 <= percentile <= 1:
        raise ValueError("percentile 必须位于 0 到 1 之间")

    epsilon = 1e-8
    result: list[ProviderInfo] = []

    for historical_order, historical_supply in zip(std_received, std_ask):
        order_history = historical_order.provide_history
        supply_history = historical_supply.provide_history
        fulfillment_rate = sum(supply_history) / (sum(order_history) + epsilon)

        positive_supplies = [supply for supply in supply_history if supply > 0]
        stable_capacity = (
            float(np.quantile(positive_supplies, percentile))
            if positive_supplies
            else 0.0
        )

        future_orders = [
            sum(order_history[week + year * 48] for year in range(5)) / 5
            for week in range(12)
        ]
        predicted_supplies = [
            min(fulfillment_rate * order, stable_capacity) for order in future_orders
        ]
        result.append(
            ProviderInfo(
                provider_id=historical_supply.provider_id,
                product_type=historical_supply.product_type,
                provide_history=predicted_supplies,  # type: ignore[arg-type]
            )
        )

    return result


def predict_transfer_loss(percentile: float) -> list[float]:
    """返回 8 家转运商在给定分位数下的预测损耗率。

    历史数据中的 0 表示当周没有运输，因此不参与分位数计算。CSV 中的
    损耗率使用百分数形式，返回结果转换为计算所需的小数形式。例如历史
    分位数为 1.5 时，返回的预测损耗率为 0.015。列表元素依次对应
    T1 至 T8；同一结果适用于该转运商未来 12 周。
    """
    if not 0 <= percentile <= 1:
        raise ValueError("percentile 必须位于 0 到 1 之间")

    result: list[float] = []
    for transfer in std_transfer:
        positive_losses = [loss for loss in transfer.loss_rates if loss > 0]
        predicted_loss = (
            float(np.quantile(positive_losses, percentile)) / 100
            if positive_losses
            else 0.0
        )
        result.append(predicted_loss)

    return result


def calculate_provider_prices(
    predicted_supplies: list[ProviderInfo],
    capacity_percentile: float,
    elasticity: float = 0.10,
) -> list[ProviderPriceInfo]:
    """计算各供应商未来每周的单位价格和采购成本。

    ``predicted_supplies`` 为供应商未来逐周实际供货量，通常直接使用
    ``predict_provider_supply`` 的返回结果。``capacity_percentile`` 必须与
    供货预测采用的分位数一致。A、B、C 三类材料的基准价格分别为
    1.25、1.15、1.00。
    """
    if not 0 <= capacity_percentile <= 1:
        raise ValueError("capacity_percentile 必须位于 0 到 1 之间")
    if elasticity < 0:
        raise ValueError("elasticity 不能为负数")

    base_prices = {"A": 1.25, "B": 1.15, "C": 1.00}
    historical_supplies = {provider.provider_id: provider for provider in std_ask}

    stable_capacities: dict[int, float] = {}
    material_max_capacities = {material: 0.0 for material in base_prices}
    for provider in std_ask:
        positive_supplies = [
            supply for supply in provider.provide_history if supply > 0
        ]
        stable_capacity = (
            float(np.quantile(positive_supplies, capacity_percentile))
            if positive_supplies
            else 0.0
        )
        stable_capacities[provider.provider_id] = stable_capacity
        material_max_capacities[provider.product_type] = max(
            material_max_capacities[provider.product_type],
            stable_capacity,
        )

    result: list[ProviderPriceInfo] = []
    for prediction in predicted_supplies:
        if prediction.provider_id not in historical_supplies:
            raise ValueError(f"不存在供应商 S{prediction.provider_id:03d} 的历史数据")

        historical_provider = historical_supplies[prediction.provider_id]
        if prediction.product_type != historical_provider.product_type:
            raise ValueError(f"供应商 S{prediction.provider_id:03d} 的材料类别不一致")
        if any(supply < 0 for supply in prediction.provide_history):
            raise ValueError("预测供货量不能为负数")

        stable_capacity = stable_capacities[prediction.provider_id]
        material_max_capacity = material_max_capacities[prediction.product_type]
        if material_max_capacity <= 0:
            raise ValueError(f"{prediction.product_type} 类材料没有正供货能力")
        if any(
            supply > stable_capacity + 1e-9 for supply in prediction.provide_history
        ):
            raise ValueError(
                f"供应商 S{prediction.provider_id:03d} 的预测供货量超过稳定供货能力"
            )

        base_price = base_prices[prediction.product_type]
        unit_prices = [
            base_price * (1 + elasticity * (1 - supply / material_max_capacity))
            for supply in prediction.provide_history
        ]
        purchase_costs = [
            price * supply
            for price, supply in zip(unit_prices, prediction.provide_history)
        ]
        result.append(
            ProviderPriceInfo(
                provider_id=prediction.provider_id,
                product_type=prediction.product_type,
                stable_capacity=stable_capacity,
                material_max_capacity=material_max_capacity,
                unit_prices=unit_prices,
                purchase_costs=purchase_costs,
            )
        )

    return result
