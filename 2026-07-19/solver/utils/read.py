import csv
from pathlib import Path
from .define import ProviderInfo, TransferInfo

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
ASK_CSV_PATH = PROJECT_ROOT_PATH / "input" / "ask.csv"
RECEIVED_CSV_PATH = PROJECT_ROOT_PATH / "input" / "received.csv"
TRANSFER_CSV_PATH = PROJECT_ROOT_PATH / "input" / "transfer.csv"


with open(ASK_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    ask = list(reader)

with open(RECEIVED_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    received = list(reader)

with open(TRANSFER_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    transfer = list(reader)


std_ask: list[ProviderInfo] = []
for index, value in enumerate(ask):
    provider = ProviderInfo(0, "", [])

    provider_id = int(value["Provider"][1:])
    if provider_id != index + 1:
        raise Exception("Should never happened")
    provider.provider_id = provider_id

    for i in range(1, 241):
        temp = str(i)
        while len(temp) < 3:
            temp = "0" + temp
        temp = "W" + temp
        provider.provide_history.append(int(value[temp]))

    provider.product_type = value["Type"]
    std_ask.append(provider)

std_received: list[ProviderInfo] = []
for index, value in enumerate(received):
    provider = ProviderInfo(0, "", [])

    provider_id = int(value["Provider"][1:])
    if provider_id != index + 1:
        raise Exception("Should never happened")
    provider.provider_id = provider_id

    for i in range(1, 241):
        temp = str(i)
        while len(temp) < 3:
            temp = "0" + temp
        temp = "W" + temp
        provider.provide_history.append(int(value[temp]))

    provider.product_type = value["Type"]
    std_received.append(provider)

std_transfer: list[TransferInfo] = []
for index, value in enumerate(transfer):
    transfer = TransferInfo(0, [])

    transfer_id = int(value["Transfer"][1:])
    if transfer_id != index + 1:
        raise Exception("Should never happened")
    transfer.transfer_id = transfer_id

    for i in range(1, 241):
        temp = str(i)
        while len(temp) < 3:
            temp = "0" + temp
        temp = "W" + temp
        transfer.loss_rates.append(float(value[temp]))

    std_transfer.append(transfer)
