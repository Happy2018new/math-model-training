"""问题二使用的结果数据结构。"""

from dataclasses import dataclass


@dataclass
class ProviderPriceInfo:
    """供应商价格弹性模型的计算结果。"""

    provider_id: int  # 供应商编号
    product_type: str  # 供应商提供的材料类别（A、B 或 C）
    stable_capacity: float  # 该供应商按分位数计算的稳定供货能力 U_i
    material_max_capacity: float  # 同类材料供应商中的最大稳定供货能力 U_k^max
    unit_prices: list[float]  # 未来各周的单位采购价格 p_it
    purchase_costs: list[float]  # 未来各周的采购成本 p_it * s_it


@dataclass
class SelectedProviderInfo:
    """最少供应商求解中入选供应商的计算结果。"""

    provider_id: int  # 入选供应商编号
    product_type: str  # 入选供应商提供的材料类别
    percentile_capacity: float  # 历史非零供货量分位数得到的 U_i
    corrected_capacity: float  # 考虑单家转运能力后的 U_i' = min(U_i, 6200)
    effective_capacity: float  # 扣除计划运输损耗后的 E_i
    topsis_score: float  # 该供应商的 TOPSIS 综合得分
    cumulative_capacity: float  # 在所属材料类别中的累计有效供货能力
    transfer_id: int = 0  # 承运该供应商供货量的转运商编号
    transfer_loss_rate: float = 0.0  # 所分配转运商的预测损耗率


@dataclass
class MinimumProviderResult:
    """最少供应商集合及其转运可行性检查结果。"""

    selected_providers: list[SelectedProviderInfo]  # 最终选中的供应商明细
    transfer_loss_rates: list[float]  # T1 至 T8 各自的预测损耗率
    material_effective_capacities: dict[str, float]  # A/B/C 三类累计有效能力
    transfer_assignments: dict[int, int]  # 供应商编号到转运商编号的分配
    transfer_loads: list[float]  # 8 家转运商的预计运输负载
