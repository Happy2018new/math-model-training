"""按照材料需求和有效供货能力求解最少供应商集合。"""

import numpy as np
from ..problem_one.topsis import TOPSIS_RESULT
from ..utils.read import std_ask
from .define import MinimumProviderResult, SelectedProviderInfo
from .utils import predict_transfer_loss

MATERIAL_DEMANDS = {"A": 6200.0, "B": 6700.0, "C": 7800.0}
TRANSFER_COUNT = 8
TRANSFER_CAPACITY = 6200.0


def _stable_capacity(history: list[int], percentile: float) -> float:
    """计算历史非零供货量的给定分位数。"""
    positive_supplies = [supply for supply in history if supply > 0]
    return (
        float(np.quantile(positive_supplies, percentile)) if positive_supplies else 0.0
    )


def _assign_transfers(
    providers: list[SelectedProviderInfo],
    loss_rates: list[float],
) -> tuple[dict[int, int], list[float]] | None:
    """分配转运商并同时满足运输容量和三类材料有效入库需求。"""
    if len(loss_rates) != TRANSFER_COUNT:
        raise ValueError("转运商损耗率数量必须为 8")

    ordered = sorted(
        providers,
        key=lambda provider: provider.corrected_capacity,
        reverse=True,
    )
    loads = [0.0] * TRANSFER_COUNT
    assignments: dict[int, int] = {}
    effective_totals = {material: 0.0 for material in MATERIAL_DEMANDS}
    transfer_order = sorted(range(TRANSFER_COUNT), key=lambda index: loss_rates[index])

    remaining_best = [
        {material: 0.0 for material in MATERIAL_DEMANDS}
        for _ in range(len(ordered) + 1)
    ]
    best_loss_rate = min(loss_rates)
    for index in range(len(ordered) - 1, -1, -1):
        remaining_best[index] = remaining_best[index + 1].copy()
        provider = ordered[index]
        remaining_best[index][provider.product_type] += (
            1 - best_loss_rate
        ) * provider.corrected_capacity

    def search(index: int) -> bool:
        if index == len(ordered):
            return all(
                effective_totals[material] + 1e-9 >= demand
                for material, demand in MATERIAL_DEMANDS.items()
            )

        if any(
            effective_totals[material] + remaining_best[index][material] + 1e-9 < demand
            for material, demand in MATERIAL_DEMANDS.items()
        ):
            return False

        provider = ordered[index]
        tried_states: set[tuple[float, float]] = set()
        for transfer_index in transfer_order:
            load = loads[transfer_index]
            state = (round(load, 9), round(loss_rates[transfer_index], 12))
            if state in tried_states:
                continue
            tried_states.add(state)
            if load + provider.corrected_capacity > TRANSFER_CAPACITY + 1e-9:
                continue

            effective_capacity = (
                1 - loss_rates[transfer_index]
            ) * provider.corrected_capacity
            loads[transfer_index] += provider.corrected_capacity
            effective_totals[provider.product_type] += effective_capacity
            assignments[provider.provider_id] = transfer_index + 1
            if search(index + 1):
                return True
            loads[transfer_index] -= provider.corrected_capacity
            effective_totals[provider.product_type] -= effective_capacity
            del assignments[provider.provider_id]

        return False

    if not search(0):
        return None
    return assignments, loads


def resolve(
    supply_percentile: float = 0.95,
    loss_percentile: float = 0.75,
) -> MinimumProviderResult:
    """按有效供货能力降序选择满足 A、B、C 需求的最少供应商。

    供应能力默认使用历史非零供货量 P95。每家供应商分配给一家转运商，
    并按照该转运商自己的预测损耗率计算有效供货能力。供货能力相同时，
    优先选择 TOPSIS 得分更高的供应商。
    """
    if not 0 <= supply_percentile <= 1:
        raise ValueError("supply_percentile 必须位于 0 到 1 之间")
    if not 0 <= loss_percentile <= 1:
        raise ValueError("loss_percentile 必须位于 0 到 1 之间")

    transfer_loss_rates = predict_transfer_loss(loss_percentile)
    best_loss_rate = min(transfer_loss_rates)
    candidates: dict[str, list[SelectedProviderInfo]] = {
        material: [] for material in MATERIAL_DEMANDS
    }

    for index, provider in enumerate(std_ask):
        percentile_capacity = _stable_capacity(
            provider.provide_history,
            supply_percentile,
        )
        corrected_capacity = min(percentile_capacity, TRANSFER_CAPACITY)
        # 先用最低损耗率计算能力上界，以确定每类供应商数量的理论下界；
        # 实际有效能力在完成转运商分配后按对应转运商损耗率重新计算。
        effective_capacity = (1 - best_loss_rate) * corrected_capacity
        candidates[provider.product_type].append(
            SelectedProviderInfo(
                provider_id=provider.provider_id,
                product_type=provider.product_type,
                percentile_capacity=percentile_capacity,
                corrected_capacity=corrected_capacity,
                effective_capacity=effective_capacity,
                topsis_score=TOPSIS_RESULT.closeness[index],
                cumulative_capacity=0.0,
            )
        )

    selected: list[SelectedProviderInfo] = []
    material_totals: dict[str, float] = {}
    for material, demand in MATERIAL_DEMANDS.items():
        ordered = sorted(
            candidates[material],
            key=lambda provider: (
                -provider.effective_capacity,
                -provider.topsis_score,
                provider.provider_id,
            ),
        )
        cumulative = 0.0
        for provider in ordered:
            cumulative += provider.effective_capacity
            provider.cumulative_capacity = cumulative
            selected.append(provider)
            if cumulative >= demand:
                break
        if cumulative < demand:
            raise ValueError(f"{material} 类供应商的有效供货能力无法满足需求")
        material_totals[material] = cumulative

    transfer_result = _assign_transfers(selected, transfer_loss_rates)
    if transfer_result is None:
        raise ValueError("所选供应商的预计供货量无法分配给 8 家转运商")
    transfer_assignments, transfer_loads = transfer_result

    material_totals = {material: 0.0 for material in MATERIAL_DEMANDS}
    for provider in selected:
        transfer_id = transfer_assignments[provider.provider_id]
        transfer_loss_rate = transfer_loss_rates[transfer_id - 1]
        provider.transfer_id = transfer_id
        provider.transfer_loss_rate = transfer_loss_rate
        provider.effective_capacity = (
            1 - transfer_loss_rate
        ) * provider.corrected_capacity

    for material in MATERIAL_DEMANDS:
        cumulative = 0.0
        for provider in selected:
            if provider.product_type != material:
                continue
            cumulative += provider.effective_capacity
            provider.cumulative_capacity = cumulative
        material_totals[material] = cumulative

    return MinimumProviderResult(
        selected_providers=selected,
        transfer_loss_rates=transfer_loss_rates,
        material_effective_capacities=material_totals,
        transfer_assignments=transfer_assignments,
        transfer_loads=transfer_loads,
    )


def debug(result: MinimumProviderResult) -> None:
    """以表格形式打印最少供应商求解结果。"""
    print(
        "转运商预测损耗率："
        + ", ".join(
            f"T{index + 1}={loss_rate:.4%}"
            for index, loss_rate in enumerate(result.transfer_loss_rates)
        )
    )
    print(f"最少供应商数量：{len(result.selected_providers)}")
    print(
        f"{'供应商':<8}{'材料':<6}{'分位数能力':>14}{'修正能力':>14}"
        f"{'有效能力':>14}{'TOPSIS':>12}{'组内累计':>14}"
        f"{'转运商':>10}{'损耗率':>12}"
    )
    for provider in result.selected_providers:
        transfer_name = f"T{result.transfer_assignments[provider.provider_id]}"
        print(
            f"S{provider.provider_id:03d}    {provider.product_type:<6}"
            f"{provider.percentile_capacity:>14.3f}"
            f"{provider.corrected_capacity:>14.3f}"
            f"{provider.effective_capacity:>14.3f}"
            f"{provider.topsis_score:>12.6f}"
            f"{provider.cumulative_capacity:>14.3f}"
            f"{transfer_name:>10}"
            f"{provider.transfer_loss_rate:>11.4%}"
        )
    print(
        "材料有效能力："
        + ", ".join(
            f"{material}={capacity:.3f}"
            for material, capacity in result.material_effective_capacities.items()
        )
    )
    print(
        "转运商负载："
        + ", ".join(
            f"T{index + 1}={load:.3f}"
            for index, load in enumerate(result.transfer_loads)
        )
    )
