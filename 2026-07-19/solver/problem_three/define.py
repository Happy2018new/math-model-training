"""第三题鲁棒订货规划的结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustScenario:
    """一个供货偏差与损耗下降比例组合成的情景。"""

    scenario_id: int  # 情景编号，从 1 开始
    supply_deviation: float  # 相对于名义供货量的偏差率 xi
    loss_reduction: float  # 每周损耗率相对于上一周的下降比例 delta


@dataclass(frozen=True)
class RobustProviderOrder:
    """共同订货方案中一家供应商某一周的计划。"""

    week: int  # 未来周次
    provider_id: int  # 供应商编号
    product_type: str  # 材料类别
    nominal_supply: float  # 名义预计实际供货量 s_bar_it
    order_quantity: float  # 根据历史履约率反推的订货量 q_it
    supply_capacity: float  # 供应商修正稳定供货能力 U_i
    transfer_id: int  # 共同计划中的转运商编号


@dataclass(frozen=True)
class RobustScenarioWeek:
    """某个情景下某一周的实际运输、入库和库存结果。"""

    scenario_id: int  # 情景编号
    week: int  # 未来周次
    actual_supplies: dict[int, float]  # 各供应商情景实际供货量
    transfer_loads: list[float]  # 各转运商情景运输负载
    material_receipts: dict[str, float]  # A/B/C 各类材料实际入库量
    ending_inventories: dict[str, float]  # A/B/C 各类材料期末库存
    purchase_cost: float  # 本周情景采购成本
    transport_cost: float  # 本周情景运输成本
    storage_cost: float  # 本周情景库存成本
    total_cost: float  # 本周情景总成本
    transport_loss: float  # 本周情景运输损耗


@dataclass(frozen=True)
class RobustScenarioResult:
    """一个情景下未来 12 周的汇总结果。"""

    scenario: RobustScenario  # 情景参数
    total_purchase_cost: float  # 12 周采购成本
    total_transport_cost: float  # 12 周运输成本
    total_storage_cost: float  # 12 周库存成本
    total_cost: float  # 12 周总经济成本
    total_transport_loss: float  # 12 周运输损耗
    weeks: list[RobustScenarioWeek]  # 逐周情景结果


@dataclass(frozen=True)
class RobustPlanningResult:
    """第三题两阶段鲁棒优化的完整结果。"""

    providers: list[int]  # 实际放入鲁棒模型的候选供应商编号
    scenarios: list[RobustScenario]  # 使用的全部不确定情景
    orders: list[RobustProviderOrder]  # 对所有情景共同的订货方案
    scenario_results: list[RobustScenarioResult]  # 各情景实际结果
    theta_cost: float  # 第一阶段最坏情景经济成本
    theta_loss: float  # 第二阶段最坏情景运输损耗
    primary_status: str  # 第一阶段求解状态
    secondary_status: str  # 第二阶段求解状态
    primary_mip_gap: float  # 第一阶段 MIP Gap
    secondary_mip_gap: float  # 第二阶段 MIP Gap
    is_optimal: bool  # 两阶段是否均达到设定的 Gap
    supply_percentile: float  # 供应能力分位数
    loss_percentile: float  # 基准损耗率分位数
    level_count: int  # 每家供应商的供货档位数
