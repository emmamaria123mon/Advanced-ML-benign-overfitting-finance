# src/data/download_equities.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.utils.helpers import project_root, load_yaml, ensure_dir


def download_one_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV from yfinance.
    Keep Adj Close, Close, Volume.
    Returns a flat DataFrame with columns: Date, Ticker, Adj Close, Close, Volume
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


def read_constituents(const_path: Path) -> list[str]:
    df = pd.read_parquet(const_path)
    tickers = df["Symbol"].dropna().astype(str).unique().tolist()
    # yfinance convention for e.g. BRK-B -> BRK.B
    tickers = [t.replace("-", ".") for t in tickers]
    return sorted(set(tickers))


def read_universe_tickers(universe_path: Path) -> list[str]:
    tickers = []
    for line in universe_path.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if t:
            tickers.append(t)
    return sorted(set(tickers))


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    prices_dir: Path,
    force: bool = False,
) -> tuple[int, int]:
    """
    Download and store per-ticker parquet files under prices_dir.
    Returns (n_ok, n_empty).
    """
    ensure_dir(prices_dir)

    n_ok, n_empty = 0, 0
    for t in tqdm(tickers, desc=f"Downloading {len(tickers)} tickers"):
        out_fp = prices_dir / f"{t}.parquet"
        if out_fp.exists() and not force:
            n_ok += 1
            continue

        df = download_one_ticker(t, start=start, end=end)
        if df.empty:
            n_empty += 1
            continue

        df.to_parquet(out_fp, index=False)
        n_ok += 1

    return n_ok, n_empty


def main() -> None:
    """
    Two-stage downloader (recommended):

    Stage 1 (pre): Download all constituents only up to the selection cutoff.
      - This is lightweight and used to compute the liquidity-ranked universe without lookahead.

    Stage 2 (full): Download full history only for the selected universe tickers.
      - Requires universe_tickers.txt produced by build_equity_panel.py.

    Usage:
      python -m src.data.download_equities --stage pre
      python -m src.data.build_equity_panel
      python -m src.data.download_equities --stage full
      python -m src.data.build_equity_panel
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/config/equities.yaml",
        help="Path relative to repo root",
    )
    parser.add_argument(
        "--stage",
        choices=["pre", "full", "all"],
        default="all",
        help=(
            "pre: download all tickers up to selection_end_date; "
            "full: download only selected universe for full period; "
            "all: download all tickers for full period (slowest)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if file exists",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = project_root()

    # Paths from config
    const_path = root / cfg["io"]["raw_constituents_path"]
    prices_dir = root / cfg["io"]["raw_prices_dir"]

    # Dates from config
    start_full = cfg["data"]["start"]
    end_full = cfg["data"]["end"]
    selection_end_date = cfg["universe"].get("selection_end_date", "2016-12-31")

    # Universe tickers output produced by build_equity_panel.py
    universe_txt = root / "data/processed/universe_tickers.txt"

    ensure_dir(const_path.parent)
    ensure_dir(prices_dir)

    # Constituents file must exist (no Wikipedia scraping)
    if not const_path.exists():
        raise RuntimeError(
            f"Missing constituents file: {const_path}\n"
            "Run: python -m src.data.fetch_sp500_constituents"
        )

    all_tickers = read_constituents(const_path)

    if args.stage == "pre":
        # Download all tickers but only up to selection_end_date (no lookahead)
        start = start_full
        end = selection_end_date
        tickers = all_tickers
        print(f"[Stage PRE] Downloading {len(tickers)} tickers from {start} to {end}...")

    elif args.stage == "full":
        # Download full history but only for the selected universe tickers
        if not universe_txt.exists():
            raise RuntimeError(
                f"Missing universe tickers file: {universe_txt}\n"
                "Run: python -m src.data.build_equity_panel (after stage pre) to create it."
            )
        tickers = read_universe_tickers(universe_txt)
        start = start_full
        end = end_full
        print(f"[Stage FULL] Downloading {len(tickers)} universe tickers from {start} to {end}...")

    else:  # args.stage == "all"
        tickers = all_tickers
        start = start_full
        end = end_full
        print(f"[Stage ALL] Downloading {len(tickers)} tickers from {start} to {end} (slow)...")

    n_ok, n_empty = download_prices(
        tickers=tickers,
        start=start,
        end=end,
        prices_dir=prices_dir,
        force=args.force,
    )

    print(f"Done. Saved/kept: {n_ok} tickers. Empty: {n_empty}.")
    print(f"Constituents at: {const_path}")
    print(f"Prices folder: {prices_dir}")
    if args.stage == "full":
        print(f"Universe tickers source: {universe_txt}")


if __name__ == "__main__":
    main()
