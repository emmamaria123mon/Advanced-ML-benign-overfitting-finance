# Benign Overfitting and Double Descent in Overparameterized Linear Regression (Equities)

This project empirically studies **benign overfitting** and the **double descent phenomenon** in overparameterized linear regression using a U.S. equities panel. Starting from a fixed set of real predictors, we synthetically increase model dimension by appending irrelevant random features, allowing us to sweep across the interpolation threshold and analyze test error, parameter norms, conditioning, implicit regularization (early stopping), and robustness to label noise.

## Repository Structure

Advanced-ML-benign-overfitting-finance/
├── data/
│ ├── processed/
│ │ ├── dataset_metadata.json
│ │ ├── equities_panel.csv
│ │ ├── equities_panel.parquet
│ │ └── universe_tickers.txt
│ └── raw/
│ └── equities/
│ ├── constituents/
│ │ └── sp500_constituents.parquet
│ └── prices/
│ ├── one parquet file per ticker
├── image/ # optional assets
├── notebooks/
│ ├── results/ # notebook-generated outputs
│ └── notebook_submission.ipynb
├── report/
│ └── AML_LUCIA_CRIADO_DEL_REL_EMMA_MONCIA.pdf #pdf report
├── results/
│ ├── benign_overfitting_equities/
│ ├── benign_overfitting_equities_regression/
│ ├── figures/
│ ├── logs/
│ └── tables/
├── src/
│ ├── config/
│ │ └── equities.yaml
│ ├── data/
│ │ ├──  **init** .py
│ │ ├── build_equity_panel.py
│ │ ├── download_equities.py
│ │ └── fetch_sp500_constituents.py
│ ├── experiments/
│ ├── features/
│ ├── models/
│ └── utils/
│ ├──  **init** .py
│ └── helpers.py
├── .gitignore
├── README.md
└── requirements.txt

---

## Setup

We recommend using a virtual environment.

```bash
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
pip install -r requirements.txt
```

2. Open and run:

```text
notebooks/notebook_submission.ipynb
```

The notebook:

* loads the processed equity panel from `data/processed/`,
* constructs controlled overparameterization via random feature augmentation,
* runs experiments (double descent, norm growth, conditioning, implicit regularization, label noise),
* saves outputs to the `results/` directory (figures, logs, tables).

## Data Procurement

We constructed the raw equity-price dataset from **publicly available U.S. equity market data** and an  **S&P 500 constituents list** .

1. **S&P 500 constituents (ticker universe).**

   We download an S&P 500 constituents table from a publicly maintained dataset hosted on GitHub:

* `datasets/s-and-p-500-companies` (`constituents.csv`), downloaded directly in `src/data/fetch_sp500_constituents.py`.

  We then **standardize tickers to the yfinance naming convention** by replacing `-` with `.` (e.g., `BRK-B → BRK.B`) before saving the list to parquet. This avoids brittle web scraping and ensures tickers match the downstream price downloader.

2. **Daily OHLCV prices.**

   We download daily price and volume data using the **`yfinance`** Python package in `src/data/download_equities.py`. For each ticker, we call:

   yf.download(tickers=ticker, start=start, end=end, interval="1d", auto_adjust=False, actions=False, threads=False)

   and retain only the columns required for our project:  **Adj Close, Close, Volume** , plus Date and Ticker.
3. **Two-stage download protocol (to avoid look-ahead in universe construction).**

   To prevent implicitly using future information when selecting the universe, we implement a **two-stage procedure** controlled by the `--stage` argument:

* **Stage `pre`:** download all constituents only up to a fixed cutoff date (`selection_end_date`). These files are used solely to compute liquidity statistics and select a stable universe without peeking beyond the cutoff.
* **Stage `full`:** once the universe tickers have been chosen and saved to `data/processed/universe_tickers.txt`, we download full-history data **only** for that selected universe.
* **Processed panel:**
  `data/processed/equities_panel.csv`, `data/processed/equities_panel.parquet`
* **Metadata:**
  `data/processed/dataset_metadata.json`
* **Universe:**
  `data/processed/universe_tickers.txt`
* **Raw inputs:**
  `data/raw/equities/` (S&P 500 constituents and per-ticker price parquet files)

All preprocessing and standardization steps use **training-only statistics** to avoid data leakage.

### Rebuild the dataset from raw files

This script in src/data/build_equity_panel.py converts the per-ticker **daily** parquet files into a **monthly equity panel** with a small set of standard predictors.

* **Streaming / memory-safe build:** we iterate over tickers one-by-one, aggregate each to month-end, then concatenate, which avoids loading the full dataset in memory.
* **Robust column handling:** `_normalize_columns` fixes occasional weird column formats from parquet/yfinance (e.g., multi-index or stringified tuples).
* **Monthly finance convention:** we resample to **month-end** dates and compute monthly returns from **Adj Close** via `pct_change()`.
* **Minimal feature set:** `log_volume`, equal-weight market return, momentum (1/3/6/12m rolling means), and volatility (6/12m rolling std + `vol_1m = |ret|`). We keep features simple because the goal is to study  **overparameterization geometry** , not maximize alpha.
* **Targets:** next-month forward return `ret_fwd` (via `shift(-1)`); binary labels (`y`, `y_cs`) are saved for extensions, but the main notebook uses the continuous regression target.
* **No-lookahead universe:** we enforce a minimum pre-cutoff history and select the most liquid tickers using data only up to `selection_end_date`.
* **Outputs:** saves the panel to `.parquet` and `.csv`, writes `universe_tickers.txt`, and stores `dataset_metadata.json` for reproducibility.

To rebuild the processed equity panel from raw parquet files:

```bash
python -m src.data.build_equity_panel
```

Paths and parameters are configured in:

```text
src/config/equities.yaml
```

## Reproducibility

* Random seeds are fixed where applicable.
* All figures and results in the report are generated directly from this repository’s code and notebooks.
* Library versions are specified in `requirements.txt`.

## Report

The LaTeX source for the written report is available at:

```text
report/Advanced-ML-benign-overfitting-finance\report\AML_LUCIA_CRIADO_DEL_REL_EMMA_MONCIA.pdf
```

## Authors

Emma Moncia
Lucía Criado del Rey Gutiérrez
