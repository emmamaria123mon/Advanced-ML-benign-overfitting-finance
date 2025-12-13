from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.helpers import project_root, load_yaml, ensure_dir


#WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


#def fetch_sp500_constituents() -> pd.DataFrame:
    #"""
    #Scrape the S&P 500 constituents from Wikipedia.
    #Returns a DataFrame with Symbol, Security, Sector, etc.
    #"""
    #tables = pd.read_html(WIKI_SP500_URL)
    #df = tables[0].copy()

    # yfinance uses '.' instead of '-' for tickers like BRK-B -> BRK.B
    #df["Symbol"] = df["Symbol"].astype(str).str.replace("-", ".", regex=False)

    #return df


def download_one_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance.
    We'll keep Adj Close, Close, Volume.
    """
    data = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()

    out = data[["Adj Close", "Close", "Volume"]].copy()
    out.index = pd.to_datetime(out.index)
    out = out.reset_index().rename(columns={"index": "Date"})
    out.insert(1, "Ticker", ticker)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config/equities.yaml", help="Path relative to repo root")
    parser.add_argument("--force", action="store_true", help="Redownload even if file exists")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = project_root()

    # Paths from config
    const_path = root / cfg["io"]["raw_constituents_path"]
    prices_dir = root / cfg["io"]["raw_prices_dir"]

    start = cfg["data"]["start"]
    end = cfg["data"]["end"]
    max_tickers = int(cfg["universe"].get("max_tickers", 500))

    ensure_dir(const_path.parent)
    ensure_dir(prices_dir)

    # 1) Constituents (read from file; avoid Wikipedia 403)
    if not const_path.exists():
        raise RuntimeError(
        f"Missing constituents file: {const_path}\n"
        "Run: python -m src.data.fetch_sp500_constituents"
    )

    constituents = pd.read_parquet(const_path)


    tickers = constituents["Symbol"].dropna().unique().tolist()[:max_tickers]
    print(f"Tickers to download: {len(tickers)}")

    # 2) Prices per ticker (separate files for robustness)
    n_ok, n_empty = 0, 0
    for t in tqdm(tickers, desc="Downloading prices"):
        out_fp = prices_dir / f"{t}.parquet"
        if out_fp.exists() and not args.force:
            n_ok += 1
            continue

        df = download_one_ticker(t, start=start, end=end)
        if df.empty:
            n_empty += 1
            continue
        df.to_parquet(out_fp, index=False)
        n_ok += 1

    print(f"Done. Saved/kept: {n_ok} tickers. Empty: {n_empty}.")
    print(f"Constituents saved at: {const_path}")
    print(f"Prices folder: {prices_dir}")


if __name__ == "__main__":
    main()
