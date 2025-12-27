# src/data/build_equity_panel.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.helpers import project_root, load_yaml, ensure_dir


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle weird column names like:
      ('Date','') , ('Adj Close','AAPL') etc
    or strings that literally look like "('Date', '')".
    """
    # If MultiIndex columns (true tuples)
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for a, b in df.columns:
            if a in ["Date", "Ticker"]:
                new_cols.append(a)
            else:
                new_cols.append(a)  # "Adj Close", "Close", "Volume", etc.
        df.columns = new_cols
        return df

    # If columns are strings that look like tuples
    rename = {}
    for c in df.columns:
        if c == "('Date', '')":
            rename[c] = "Date"
        elif c == "('Ticker', '')":
            rename[c] = "Ticker"
        elif isinstance(c, str) and c.startswith("('Adj Close'"):
            rename[c] = "Adj Close"
        elif isinstance(c, str) and c.startswith("('Close'"):
            rename[c] = "Close"
        elif isinstance(c, str) and c.startswith("('Volume'"):
            rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)

    return df


def load_monthly_from_price_files(prices_dir: Path) -> pd.DataFrame:
    """
    Memory-safe: reads each ticker parquet separately, keeps minimal cols,
    aggregates to month-end, and concatenates ONLY monthly panels.
    """
    monthly_list = []

    files = sorted(prices_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No parquet files found in {prices_dir}")

    for fp in files:
        df = pd.read_parquet(fp)
        if df is None or df.empty:
            continue

        df = _normalize_columns(df)

        # Minimal required columns
        needed = ["Date", "Ticker", "Adj Close", "Volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            # skip weird/broken file
            continue

        df = df[needed].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Ticker"] = df["Ticker"].astype(str)

        # Month-end
        df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")

        # Aggregate daily -> monthly
        m = (
            df.groupby(["Ticker", "Date"], as_index=False)
              .agg({"Adj Close": "last", "Volume": "mean"})
              .sort_values(["Ticker", "Date"])
        )

        # Monthly return from Adj Close
        m["ret"] = m.groupby("Ticker")["Adj Close"].pct_change()

        monthly_list.append(m)

    if not monthly_list:
        raise ValueError("All files were empty or unusable.")

    panel = pd.concat(monthly_list, ignore_index=True)
    panel = panel.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return panel



# -----------------------
# Feature engineering
# -----------------------

def build_features(panel: pd.DataFrame, mom_windows=(1, 3, 6, 12),  vol_windows=(1, 6, 12)) -> pd.DataFrame:
    panel = panel.sort_values(["Ticker", "Date"]).copy()

    panel["log_volume"] = np.log1p(panel["Volume"])

    # Equal-weight market return
    panel["mkt_ew_ret"] = panel.groupby("Date")["ret"].transform("mean")

    # Momentum: rolling mean of returns
    for k in mom_windows:
        panel[f"mom_{k}m"] = (
            panel.groupby("Ticker")["ret"]
            .rolling(k)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Volatility: rolling std of returns (MULTI-HORIZON)
    for k in vol_windows:
        panel[f"vol_{k}m"] = (
            panel.groupby("Ticker")["ret"]
            .rolling(k)
            .std()
            .reset_index(level=0, drop=True)
        )

    return panel


# -----------------------
# Target construction
# -----------------------

def build_target(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["Ticker", "Date"]).copy()

    # Next-month forward return per ticker
    panel["ret_fwd"] = panel.groupby("Ticker")["ret"].shift(-1)

    # Unconditional label: positive next-month return
    panel["y"] = (panel["ret_fwd"] > 0).astype(int)

    # Cross-sectional label: outperform the cross-sectional median that month
    panel["y_cs"] = (
        panel.groupby("Date")["ret_fwd"]
        .transform(lambda x: (x > x.median()).astype(int))
    )

    return panel


# -----------------------
# Universe selection (no lookahead)
# -----------------------

def enforce_min_history_pre_cutoff(
    panel: pd.DataFrame,
    selection_end_date: str,
    min_months: int = 24,
) -> pd.DataFrame:
    cutoff = pd.to_datetime(selection_end_date)

    pre = panel[panel["Date"] <= cutoff]
    n_obs = pre.groupby("Ticker")["Date"].nunique()

    eligible = n_obs[n_obs >= min_months].index
    return panel[panel["Ticker"].isin(eligible)].copy()


def filter_top_liquid_stocks(
    panel: pd.DataFrame,
    n_stocks: int,
    selection_end_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Select top n_stocks by average Volume using ONLY data up to selection_end_date.
    Returns (filtered_panel, selected_tickers).
    """
    cutoff = pd.to_datetime(selection_end_date)

    panel_pre = panel[panel["Date"] <= cutoff]
    avg_volume = (
        panel_pre.groupby("Ticker")["Volume"]
        .mean()
        .sort_values(ascending=False)
    )

    selected = avg_volume.head(n_stocks).index.tolist()
    filtered = panel[panel["Ticker"].isin(selected)].copy()
    return filtered, selected


# -----------------------
# Sanity checks
# -----------------------

def print_sanity_checks(
    panel: pd.DataFrame,
    selection_end_date: str,
    selected_tickers: list[str],
) -> None:
    cutoff = pd.to_datetime(selection_end_date)

    print("\n[Sanity checks]")
    print("Universe selection cutoff:", selection_end_date)

    pre = panel[panel["Date"] <= cutoff]
    if len(pre) > 0:
        print("Pre-cutoff date range in filtered panel:", pre["Date"].min(), "->", pre["Date"].max())
        print("Tickers with pre-cutoff observations (filtered):", pre["Ticker"].nunique())
    else:
        print("WARNING: No pre-cutoff rows in filtered panel (check dates/cutoff).")

    print("Tickers selected:", len(selected_tickers))
    print("Tickers in final panel:", panel["Ticker"].nunique())

    # Coverage per month
    per_month = panel.groupby("Date")["Ticker"].nunique()
    print("Tickers per month — min/max:", int(per_month.min()), int(per_month.max()))
    print(per_month.describe())

    if "y" in panel.columns:
        print("y mean (ret_fwd>0):", float(panel["y"].mean()))
    if "y_cs" in panel.columns:
        print("y_cs mean (ret_fwd > monthly median):", float(panel["y_cs"].mean()))

    print("Final date range:", panel["Date"].min(), "->", panel["Date"].max())
    print("Rows:", len(panel))
    print("[End sanity checks]\n")


# -----------------------
# Main
# -----------------------

def main() -> None:
    cfg = load_yaml("src/config/equities.yaml")
    root = project_root()

    prices_dir = root / cfg["io"]["raw_prices_dir"]
    out_path = root / cfg["io"]["processed_panel_path"]
    ensure_dir(out_path.parent)

    # Config
    n_stocks = int(cfg.get("universe", {}).get("max_tickers", 300))
    selection_end_date = cfg.get("universe", {}).get("selection_end_date", "2016-12-31")
    min_months = int(cfg.get("panel", {}).get("min_history_months", 24))

    mom_windows = tuple(cfg.get("features", {}).get("mom_windows_months", [1, 3, 6, 12]))
    vol_windows = tuple(cfg.get("features", {}).get("vol_windows_months", [1, 6, 12]))

    print("Loading raw prices + aggregating to monthly (streaming)...")
    panel = load_monthly_from_price_files(prices_dir)

    print("Building features...")
    panel = build_features(panel, mom_windows=mom_windows, vol_windows=vol_windows)

    print("Building target...")
    panel = build_target(panel)

    # Enforce eligibility pre-cutoff (min history)
    print(f"Enforcing min history: {min_months} months pre-cutoff...")
    panel = enforce_min_history_pre_cutoff(panel, selection_end_date, min_months=min_months)

    # Liquidity selection pre-cutoff
    print(f"Filtering top liquid stocks (top {n_stocks}) using data up to {selection_end_date}...")
    panel, selected_tickers = filter_top_liquid_stocks(panel, n_stocks=n_stocks, selection_end_date=selection_end_date)

    # Drop NA only on needed columns
    feature_cols = ["ret", "log_volume", "mkt_ew_ret"] + [f"mom_{k}m" for k in mom_windows] + [f"vol_{k}m" for k in vol_windows]
    target_cols = ["ret_fwd", "y", "y_cs"]
    panel = panel.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

    # Sanity checks
    print_sanity_checks(panel, selection_end_date, selected_tickers)

    # Save panel
    panel.to_parquet(out_path, index=False)
    print(f"Saved panel to {out_path}")

    # Save universe tickers
    universe_txt = root / "data/processed/universe_tickers.txt"
    ensure_dir(universe_txt.parent)
    universe = sorted(panel["Ticker"].unique().tolist())
    universe_txt.write_text("\n".join(universe), encoding="utf-8")

    # Save metadata
    meta = {
        "n_rows": int(len(panel)),
        "n_tickers": int(panel["Ticker"].nunique()),
        "date_min": str(panel["Date"].min()),
        "date_max": str(panel["Date"].max()),
        "label_mean_y": float(panel["y"].mean()),
        "label_mean_y_cs": float(panel["y_cs"].mean()),
        "selection_end_date": selection_end_date,
        "min_history_months": min_months,
        "n_stocks_target": n_stocks,
        "mom_windows_months": list(mom_windows),
        "vol_windows_months": list(vol_windows),
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "raw_prices_dir": str(prices_dir),
        "processed_panel_path": str(out_path),
    }
    meta_path = root / "data/processed/dataset_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
5