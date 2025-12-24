from __future__ import annotations

import pandas as pd

from src.utils.helpers import project_root, load_yaml, ensure_dir


SP500_CSV_URL = (
    "https://raw.githubusercontent.com/"
    "datasets/s-and-p-500-companies/master/data/constituents.csv"
)


def main():
    cfg = load_yaml("src/config/equities.yaml")
    root = project_root()

    const_path = root / cfg["io"]["raw_constituents_path"]
    ensure_dir(const_path.parent)

    # Load constituents from GitHub (robust, no scraping)
    df = pd.read_csv(SP500_CSV_URL)

    # Standardize column name + yfinance ticker format
    df["Symbol"] = df["Symbol"].str.replace("-", ".", regex=False)

    df = df[["Symbol"]].drop_duplicates().sort_values("Symbol")

    df.to_parquet(const_path, index=False)
    print(f"Saved {len(df)} S&P 500 tickers to {const_path}")


if __name__ == "__main__":
    main()
