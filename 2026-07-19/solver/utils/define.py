from dataclasses import dataclass


@dataclass
class ProviderInfo:
    provider_id: int
    product_type: str
    provide_history: list[int]


@dataclass
class TransferInfo:
    transfer_id: int
    loss_rates: list[float]
