from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"
DESTINATION = ROOT / "data/raw"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / "creditcard.csv"
    if not path.exists():
        with urlopen(URL) as response:
            archive = BytesIO(response.read())
        with ZipFile(archive) as bundle:
            bundle.extract("creditcard.csv", DESTINATION)
    data = pd.read_csv(path, usecols=["Class"])
    if len(data) != EXPECTED_ROWS or int(data["Class"].sum()) != EXPECTED_FRAUDS:
        raise RuntimeError("Downloaded dataset failed row or fraud-count verification")
    print(f"Verified {path}: {len(data):,} rows and {int(data['Class'].sum()):,} frauds")


if __name__ == "__main__":
    main()

