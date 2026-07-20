import statistics
from ..utils.read import std_ask, std_received

all_products_sum = [sum(i.provide_history) for i in std_received]  # 总供货量
avg_products_sum = [
    sum(i.provide_history) / len(i.provide_history) for i in std_received
]  # 周平均供货量
provide_weeks = [
    sum([j > 0 for j in i.provide_history]) / len(i.provide_history)
    for i in std_received
]  # 供货频率
order_match_stable = [
    (
        statistics.pstdev(std_received[i].provide_history) / avg_products_sum[i]
        if avg_products_sum[i] > 0
        else 0.0
    )
    for i in range(318)
]  # 供货稳定性
order_match_rate = [
    1
    - sum(
        [
            abs(std_ask[i].provide_history[j] - std_received[i].provide_history[j])
            for j in range(240)
        ]
    )
    / sum(
        [
            abs(std_ask[i].provide_history[j] + std_received[i].provide_history[j])
            for j in range(240)
        ]
    )
    for i in range(318)
]  # 订单匹配度
product_contribute = [
    (
        all_products_sum[i] / 0.62
        if std_ask[i].product_type == "A"
        else (
            all_products_sum[i] / 0.67
            if std_ask[i].product_type == "B"
            else all_products_sum[i] / 0.78
        )
    )
    for i in range(318)
]  # 等效产品贡献量


def _apply_std(values: list[int] | list[float], positive: bool) -> list[float]:
    values = [float(value) for value in values]
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return [1.0] * len(values)
    if positive:
        return [(value - min_val) / (max_val - min_val) for value in values]

    return [(max_val - value) / (max_val - min_val) for value in values]


std_factor_all_products_sum = _apply_std(all_products_sum, True)
std_factor_provide_weeks = _apply_std(provide_weeks, True)
std_factor_order_match_stable = _apply_std(order_match_stable, False)
std_factor_order_match_rate = _apply_std(order_match_rate, True)
std_factor_product_contribute = _apply_std(product_contribute, True)
