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


@dataclass
class WeeklyProviderOrder:
    """一家供应商在某一周的订货、供货和转运结果。"""

    week: int  # 未来周编号（1 至 12）
    provider_id: int  # 供应商编号
    product_type: str  # 供应材料类别
    order_quantity: float  # 企业向供应商下达的订货量 q_it
    expected_supply: float  # 根据履约率预计的实际供货量 s_it
    supply_capacity: float  # 该供应商经单家转运限制修正后的供货能力
    unit_price: float  # 根据本周预计供货量计算的单位采购价格 p_it
    purchase_cost: float  # 本周采购成本 p_it * s_it
    transfer_id: int  # 本周负责承运的转运商编号
    transfer_loss_rate: float  # 所用转运商的预测损耗率
    actual_received: float  # 扣除运输损耗后的实际入库量


@dataclass
class WeeklyMaterialInventory:
    """某类材料在某一周的需求与库存状态。"""

    week: int  # 未来周编号（1 至 12）
    product_type: str  # 材料类别
    demand: float  # 本周生产消耗量 D_k
    required_receipt: float  # 为恢复安全库存需要的本周入库量
    actual_received: float  # 本周运输后实际入库量 R_kt
    ending_inventory: float  # 本周期末库存 I_kt
    safety_stock: float  # 该材料的三周安全库存下限


@dataclass
class WeeklyOrderPlan:
    """未来某一周的完整订货与库存计划。"""

    week: int  # 未来周编号（1 至 12）
    provider_orders: list[WeeklyProviderOrder]  # 本周各供应商订供货明细
    material_inventories: list[WeeklyMaterialInventory]  # 本周各材料库存状态
    transfer_loads: list[float]  # 本周 8 家转运商的运输负载
    purchase_cost: float  # 本周采购成本
    transport_cost: float  # 本周运输成本
    storage_cost: float  # 本周库存成本
    total_cost: float  # 本周总成本
    transport_loss: float  # 本周运输损耗总量


@dataclass
class TwelveWeekOrderPlanResult:
    """未来 12 周订货、转运、库存和成本的汇总结果。"""

    elasticity: float  # 价格弹性系数 eta
    weeks: list[WeeklyOrderPlan]  # 未来 12 周逐周计划
    total_purchase_cost: float  # 12 周采购成本
    total_transport_cost: float  # 12 周运输成本
    total_storage_cost: float  # 12 周库存成本
    total_cost: float  # 12 周总成本
    total_transport_loss: float  # 12 周运输损耗总量
    successful_allocation_count: int  # 12 周中非零供应商供货分配次数


@dataclass
class IntegerProgrammingResult:
    """离散 0-1 整数规划的求解状态与方案对比。"""

    optimized_plan: TwelveWeekOrderPlanResult  # 整数规划得到的未来 12 周方案
    baseline_plan: TwelveWeekOrderPlanResult  # 贪心算法生成的基础可行方案
    solver_name: str  # 实际使用的 MILP 求解器名称
    primary_solver_status: str  # 第一阶段经济成本最小化的求解状态
    secondary_solver_status: str  # 第二阶段运输损耗最小化的求解状态
    is_optimal: bool  # 两个阶段是否均达到用户要求的最优性间隙
    level_count: int  # 每家供应商供货能力划分的标准档位数
    primary_best_cost: float  # 第一阶段当前找到的最低经济成本
    primary_mip_gap: float  # 第一阶段上下界之间的相对最优性间隙
    secondary_mip_gap: float  # 第二阶段上下界之间的相对最优性间隙
    cost_saving: float  # 相对基础方案节省的成本
    cost_saving_rate: float  # 相对基础方案的成本节省比例
