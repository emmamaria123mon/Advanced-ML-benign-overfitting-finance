# Benign Overfitting and Double Descent in Overparameterized Linear Regression (Equities)

This project empirically studies **benign overfitting** and the **double descent phenomenon** in overparameterized linear regression using a U.S. equities panel.
Starting from a fixed set of real predictors, we synthetically increase model dimension by appending irrelevant random features, allowing us to sweep across the interpolation threshold and analyze test error, parameter norms, conditioning, implicit regularization (early stopping), and robustness to label noise.

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
│ ├── A.parquet
│ ├── AAPL.parquet
│ ├── ABBV.parquet
│ └── ... (one parquet file per ticker)
├── image/ # optional assets
├── notebooks/
│ ├── results/ # notebook-generated outputs (if used)
│ └── benign_overfitting_equities_SUBMISSION_v2.ipynb
├── report/
│ └── main.tex # LaTeX report
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
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
---

The project relies on standard scientific Python libraries
(`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `jupyter`).

---

## Running the Project

### Reproduce the experiments and figures (recommended)

All experiments reported in the paper can be reproduced from the submission notebook.

1. Start Jupyter:

```bash
jupyter notebook
```

2. Open and run:

```text
notebooks/benign_overfitting_equities_SUBMISSION_v2.ipynb
```

The notebook:

* loads the processed equity panel from `data/processed/`,
* constructs controlled overparameterization via random feature augmentation,
* runs experiments (double descent, norm growth, conditioning, implicit regularization, label noise),
* saves outputs to the `results/` directory (figures, logs, tables).

---

### (Optional) Rebuild the dataset from raw files

To rebuild the processed equity panel from raw parquet files:

```bash
python -m src.data.build_equity_panel
```

Paths and parameters are configured in:

```text
src/config/equities.yaml
```

---

## Data

* **Processed panel:**
  `data/processed/equities_panel.csv`, `data/processed/equities_panel.parquet`
* **Metadata:**
  `data/processed/dataset_metadata.json`
* **Universe:**
  `data/processed/universe_tickers.txt`
* **Raw inputs:**
  `data/raw/equities/` (S&P 500 constituents and per-ticker price parquet files)

All preprocessing and standardization steps use **training-only statistics** to avoid data leakage.

---

## Reproducibility

* Random seeds are fixed where applicable.
* All figures and results in the report are generated directly from this repository’s code and notebooks.
* Library versions are specified in `requirements.txt`.

---

## Report

The LaTeX source for the written report is available at:

```text
report/Advanced-ML-benign-overfitting-finance\report\AML_LUCIA_CRIADO_DEL_REL_EMMA_MONCIA.pdf
```

---

## Authors

Emma Moncia
Lucía Criado del Rey Gutiérrez
