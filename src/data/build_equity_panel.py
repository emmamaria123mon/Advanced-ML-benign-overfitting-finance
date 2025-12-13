from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.helpers import project_root, load_yaml, ensure_dir


def load_all_prices(prices_dir: Path) -> pd.DataFrame:
    files = list(prices_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {prices_dir}.")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = normalize_columns(df)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    required = {"Date", "Ticker", "Adj Close", "Close", "Volume"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(
            f"Missing expected columns: {missing}\n"
            f"Columns found (sample): {list(out.columns)[:20]}"
        )

    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes produces multi-index columns (or stringified tuples).
    This converts them into clean flat names:
      ('Date','') -> Date
      ('Adj Close','AAPL') -> Adj Close
      ('Close','AAPL') -> Close
      ('Volume','AAPL') -> Volume
      ('Ticker','') -> Ticker
    """
    new_cols = []
    for c in df.columns:
        s = str(c)

        # if it looks like a tuple string "('Adj Close', 'AAPL')"
        if s.startswith("(") and s.endswith(")"):
            # remove parentheses
            s2 = s[1:-1]
            # split on comma only once: "'Adj Close', 'AAPL'"
            parts = s2.split(",", 1)
            left = parts[0].strip().strip("'").strip('"')
            new_cols.append(left)
        else:
            new_cols.append(s)

    df = df.copy()
    df.columns = new_cols
    return df

def standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have columns: Date, Ticker, Adj Close, Close, Volume
    regardless of whether Date came as an index or a column.
    """
    # If Date is in the index, bring it back
    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})

    # Sometimes yfinance names it 'Datetime'
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    return df

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = (
        df.groupby(["Ticker", "Month"])
          .agg({"Adj Close": "last", "Volume": "sum"})
          .reset_index()
          .rename(columns={"Month": "Date"})
    )

    monthly["ret"] = (
        monthly.groupby("Ticker")["Adj Close"]
        .pct_change()
    )

    return monthly


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["Ticker", "Date"])

    # Momentum features
    for k in [1, 3, 6, 12]:
        panel[f"mom_{k}m"] = (
            panel.groupby("Ticker")["ret"]
            .rolling(k)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Volatility
    panel["vol_12m"] = (
        panel.groupby("Ticker")["ret"]
        .rolling(12)
        .std()
        .reset_index(level=0, drop=True)
    )

    return panel


def build_target(panel: pd.DataFrame) -> pd.DataFrame:
    panel["ret_fwd"] = (
        panel.groupby("Ticker")["ret"].shift(-1)
    )
    panel["y"] = (panel["ret_fwd"] > 0).astype(int)
    return panel


def main():
    cfg = load_yaml("src/config/equities.yaml")
    root = project_root()

    prices_dir = root / cfg["io"]["raw_prices_dir"]
    out_path = root / cfg["io"]["processed_panel_path"]
    ensure_dir(out_path.parent)

    print("Loading raw prices...")
    df = load_all_prices(prices_dir)

    print("Aggregating to monthly...")
    panel = to_monthly(df)

    print("Building features...")
    panel = build_features(panel)

    print("Building target...")
    panel = build_target(panel)

    panel = panel.dropna().reset_index(drop=True)

    panel.to_parquet(out_path, index=False)
    print(f"Saved panel to {out_path}")
    print(panel.head())


if __name__ == "__main__":
    main()
