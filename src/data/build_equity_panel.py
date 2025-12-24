from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from src.utils.helpers import project_root, load_yaml, ensure_dir
import numpy as np

# I/O + normalization helpers

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance can produce multi-index columns, and when saved to parquet
    they may come back as stringified tuples like:
        "('Adj Close', 'AAPL')" or "('Date', '')"
    This converts them into clean flat names:
        -> "Adj Close", "Close", "Volume", "Date", "Ticker"
    """
    new_cols = []
    for c in df.columns:
        s = str(c)

        # If it looks like a tuple string "('Adj Close', 'AAPL')"
        if s.startswith("(") and s.endswith(")"):
            s2 = s[1:-1]  # remove parentheses
            parts = s2.split(",", 1)  # split only once
            left = parts[0].strip().strip("'").strip('"')
            new_cols.append(left)
        else:
            new_cols.append(s)

    out = df.copy()
    out.columns = new_cols
    return out


def standardize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have columns:
      Date, Ticker, Adj Close, Close (optional), Volume
    regardless of whether Date came as an index or a column.
    """
    out = df.copy()

    # If Date is in the index, bring it back
    if "Date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "Date"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "Date"})

    # Sometimes yfinance names it 'Datetime'
    if "Date" not in out.columns and "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "Date"})

    return out


def load_all_prices(prices_dir: Path) -> pd.DataFrame:
    files = list(prices_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No parquet files found in {prices_dir}.")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = normalize_columns(df)
        df = standardize_price_df(df)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    # Close is not strictly needed (we use Adj Close + Volume),
    # but keep it if present.
    required = {"Date", "Ticker", "Adj Close", "Volume"}
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(
            f"Missing expected columns: {missing}\n"
            f"Columns found (sample): {list(out.columns)[:30]}"
        )

    return out


# Panel construction

def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # month-end timestamp
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")

    monthly = (
        df.groupby(["Ticker", "Month"])
        .agg({"Adj Close": "last", "Volume": "sum"})
        .reset_index()
        .rename(columns={"Month": "Date"})
    )

    monthly["ret"] = monthly.groupby("Ticker")["Adj Close"].pct_change()

    return monthly


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["Ticker", "Date"]).copy()

    panel["log_volume"] = np.log1p(panel["Volume"])

    panel["mkt_ew_ret"] = panel.groupby("Date")["ret"].transform("mean")

    # Momentum features: rolling mean of monthly returns
    for k in [1, 3, 6, 12]:
        panel[f"mom_{k}m"] = (
            panel.groupby("Ticker")["ret"]
            .rolling(k)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Volatility: rolling std of monthly returns
    panel["vol_12m"] = (
        panel.groupby("Ticker")["ret"]
        .rolling(12)
        .std()
        .reset_index(level=0, drop=True)
    )

    return panel


def build_target(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel = panel.sort_values(["Ticker", "Date"])

    # next-month return (already monthly)
    panel["ret_fwd"] = panel.groupby("Ticker")["ret"].shift(-1)

    # simple direction label
    panel["y"] = (panel["ret_fwd"] > 0).astype(int)

    # cross-sectional label (balanced each month)
    panel["y_cs"] = (
        panel["ret_fwd"] > panel.groupby("Date")["ret_fwd"].transform("median")
    ).astype(int)

    return panel



def filter_top_liquid_stocks(
    panel: pd.DataFrame,
    n_stocks: int,
    selection_end_date: str
) -> pd.DataFrame:
    """
    Select top n_stocks by average volume using ONLY data
    up to selection_end_date (no lookahead bias).
    """
    cutoff = pd.to_datetime(selection_end_date)

    panel_pre = panel[panel["Date"] <= cutoff]

    avg_volume = (
        panel_pre.groupby("Ticker")["Volume"]
        .mean()
        .sort_values(ascending=False)
    )

    top_tickers = avg_volume.head(n_stocks).index

    return panel[panel["Ticker"].isin(top_tickers)].copy()



# Main
def main() -> None:
    cfg = load_yaml("src/config/equities.yaml")
    root = project_root()

    universe_cfg = cfg["universe"]
    n_stocks = int(universe_cfg.get("max_tickers", 300))
    selection_end_date = universe_cfg.get("selection_end_date")
    if selection_end_date is None:
        raise ValueError(
            "Missing universe.selection_end_date in equities.yaml "
            "(needed to avoid lookahead bias)."
        )
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

    feature_cols = [
        "ret",
        "mom_1m",
        "mom_3m",
        "mom_6m",
        "mom_12m",
        "vol_12m",
        "Volume",
        "log_volume",
        "mkt_ew_ret",
    ]
    target_cols = ["y", "y_cs"]

    print(
        f"Filtering top liquid stocks (top {n_stocks}) "
        f"using data up to {selection_end_date}..."
    )
    panel = filter_top_liquid_stocks(
        panel,
        n_stocks=n_stocks,
        selection_end_date=selection_end_date,
    )

    panel = (
        panel
        .dropna(subset=feature_cols + target_cols)
        .reset_index(drop=True)
    )
    # --- Save universe tickers list ---
    tickers = sorted(panel["Ticker"].unique())
    tickers_path = out_path.parent / "universe_tickers.txt"
    tickers_path.write_text("\n".join(tickers), encoding="utf-8")

# --- Save dataset metadata ---
    meta = {
    "n_rows": int(len(panel)),
    "n_tickers": int(panel["Ticker"].nunique()),
    "date_min": str(panel["Date"].min()),
    "date_max": str(panel["Date"].max()),
    "freq": cfg.get("panel", {}).get("freq", "M"),
    "max_tickers": int(cfg.get("universe", {}).get("max_tickers", n_stocks)),
    "selection_end_date": cfg.get("universe", {}).get("selection_end_date"),
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "class_balance": panel["y"].value_counts(normalize=True).to_dict() if "y" in panel.columns else None,
}
    meta_path = out_path.parent / "dataset_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


    panel.to_parquet(out_path, index=False)
    print(f"Saved panel to {out_path}")
    print("Tickers:", panel["Ticker"].nunique())
    print("Date range:", panel["Date"].min(), "->", panel["Date"].max())
    print(panel.head())

if __name__ == "__main__":
    main()
