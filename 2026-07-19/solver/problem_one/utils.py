import csv
from pathlib import Path

PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
ASK_CSV_PATH = PROJECT_ROOT_PATH / "input" / "ask.csv"
RECEIVE_CSV_PATH = PROJECT_ROOT_PATH / "input" / "receive.csv"
TRANSFER_CSV_PATH = PROJECT_ROOT_PATH / "input" / "transfer.csv"


with open(ASK_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    ask = list(reader)

with open(RECEIVE_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    receive = list(reader)

with open(TRANSFER_CSV_PATH, "r+", encoding="utf-8") as file:
    reader = csv.DictReader(file, delimiter=",")
    transfer = list(reader)
