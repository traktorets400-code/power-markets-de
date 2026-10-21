"""Forecast evaluation: point metrics per year and the multivariate Diebold-Mariano test.

DM test as in Lago et al. (2021), "Forecasting day-ahead electricity prices: A review of
state-of-the-art algorithms, best practices and an open-access benchmark", Applied Energy 293:
the loss of a day is the mean absolute error over its hours, so one observation per day and no
intraday autocorrelation. H0: model A is not more accurate than model B (one-sided).

    python3 evaluate.py            # reads results/predictions.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
MODELS = ["naive", "lear", "lgbm_nofund", "lgbm", "ens"]


def smape(y, f):
    return 100 * np.mean(2 * np.abs(y - f) / (np.abs(y) + np.abs(f)).clip(lower=1e-6))


def metrics(p, models=MODELS):
    rows = []
    for year, g in list(p.groupby(p["date"].dt.year)) + [("all", p)]:
        mae_naive = (g["price"] - g["naive"]).abs().mean()
        for m in models:
            e = g["price"] - g[m]
            rows.append(dict(year=year, model=m, MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()),
                             sMAPE=smape(g["price"], g[m]), rMAE=e.abs().mean() / mae_naive))
    return pd.DataFrame(rows)


def dm_test(p, a, b):
    """One-sided p-value that model a has lower daily MAE than model b."""
    daily = p.assign(la=(p["price"] - p[a]).abs(), lb=(p["price"] - p[b]).abs()).groupby("date")[["la", "lb"]].mean()
    d = daily["lb"] - daily["la"]
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return 1 - norm.cdf(stat)


def dm_matrix(p, models=MODELS):
    return pd.DataFrame([[np.nan if a == b else dm_test(p, a, b) for b in models] for a in models],
                        index=models, columns=models)


def main():
    p = pd.read_parquet(OUT / "predictions.parquet").dropna(subset=MODELS + ["price"])
    p["date"] = pd.to_datetime(p["date"])
    m = metrics(p)
    m.to_csv(OUT / "forecast_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(m[m["year"] == "all"].round(3).to_string(index=False))
    dm = dm_matrix(p)
    dm.to_csv(OUT / "dm_pvalues.csv")
    print("\nDM test p-values (row more accurate than column; < 0.05 = significant)")
    print(dm.map(lambda v: "" if np.isnan(v) else f"{v:.3f}").to_string())


if __name__ == "__main__":
    main()
