"""Build the hourly modelling table from data/raw/*.parquet.

Everything is aggregated to hourly means (UTC). The SDAC moved to 15-min MTU on
2025-10-01; hourly averaging keeps one consistent target over 2019-2026.

Information set for delivery day D (auction gate closure 12:00 CET on D-1):
  * day-ahead forecasts for D (load, wind, solar, generation)   -> known
  * all DA prices of D-1 and earlier (D-1 cleared at ~12:45 on D-2) -> known
  * actual load / generation of D-1                              -> NOT fully known, not used
"""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
TZ = "Europe/Berlin"


def load_raw(name):
    files = sorted(RAW.glob(f"{name}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {name}_*.parquet in {RAW}; run download.py first")
    df = pd.concat(pd.read_parquet(f) for f in files)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.resample("1h").mean()


def hourly_table():
    parts = [load_raw(n) for n in ("price_da", "load_fc", "ws_fc", "gen_fc", "load_act")]
    df = pd.concat(parts, axis=1)
    df = df.loc[df["price"].first_valid_index():df["price"].last_valid_index()]
    df["wind_fc"] = df[["wind_on_fc", "wind_off_fc"]].sum(axis=1, min_count=1)
    df["res_fc"] = df["load_fc"] - df["wind_fc"] - df["solar_fc"]
    local = df.index.tz_convert(TZ)
    df["date"] = local.tz_localize(None).normalize()
    df["hour"] = local.hour
    return df


def same_hour_lag(out, col, lag):
    """Value of `col` at the same local hour `lag` days earlier (aligned by local date/hour, DST-safe)."""
    piv = out.pivot_table(index="date", columns="hour", values=col, aggfunc="mean").shift(lag, freq="D")
    s = piv.stack()
    idx = pd.MultiIndex.from_arrays([out["date"], out["hour"]])
    return s.reindex(idx).to_numpy()


def add_features(df):
    """Features for predicting `price` of each hour of day D using only info available at 12:00 D-1."""
    out = df.copy()
    de_hol = holidays.Germany(years=range(2018, 2028))
    dates = out["date"]
    out["dow"] = dates.dt.dayofweek
    out["month"] = dates.dt.month
    out["doy"] = dates.dt.dayofyear
    out["holiday"] = dates.dt.date.map(lambda d: d in de_hol).astype(int)
    out["holiday_bridge"] = (dates - pd.Timedelta(days=1)).dt.date.map(lambda d: d in de_hol).astype(int) \
        | (dates + pd.Timedelta(days=1)).dt.date.map(lambda d: d in de_hol).astype(int)

    for lag in (1, 2, 7):
        out[f"p_lag{lag}"] = same_hour_lag(out, "price", lag)
    daily = out.groupby("date")["price"].agg(["mean", "min", "max", "std"])
    d1 = daily.shift(1, freq="D").add_prefix("d1_")
    out = out.join(d1, on="date")
    # D-1 peak (8-20h) mean and price at D-1 hour 23 (last known point before D)
    peak = out[out["hour"].between(8, 19)].groupby("date")["price"].mean().shift(1, freq="D").rename("d1_peak")
    last = out[out["hour"] == 23].groupby("date")["price"].mean().shift(1, freq="D").rename("d1_h23")
    out = out.join(peak, on="date").join(last, on="date")

    # Fundamentals for D, plus their daily aggregates and change vs. D-1 forecasts.
    for col in ("load_fc", "solar_fc", "wind_fc", "res_fc"):
        g = out.groupby("date")[col]
        out[f"{col}_dmean"] = g.transform("mean")
        out[f"{col}_dmax"] = g.transform("max")
    out["res_fc_dmin"] = out.groupby("date")["res_fc"].transform("min")
    out["res_fc_lag1"] = same_hour_lag(out, "res_fc", 1)
    out["res_fc_chg"] = out["res_fc"] - out["res_fc_lag1"]
    out["res_share"] = out["res_fc"] / out["load_fc"]
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


FUND_FEATURES = ["load_fc", "solar_fc", "wind_fc", "res_fc", "gen_fc", "res_share",
                 "load_fc_dmean", "load_fc_dmax", "solar_fc_dmean", "solar_fc_dmax",
                 "wind_fc_dmean", "wind_fc_dmax", "res_fc_dmean", "res_fc_dmax", "res_fc_dmin",
                 "res_fc_lag1", "res_fc_chg"]
PRICE_FEATURES = ["p_lag1", "p_lag2", "p_lag7", "d1_mean", "d1_min", "d1_max", "d1_std", "d1_peak", "d1_h23"]
CAL_FEATURES = ["hour", "dow", "month", "doy", "holiday", "holiday_bridge", "hour_sin", "hour_cos"]
ALL_FEATURES = CAL_FEATURES + PRICE_FEATURES + FUND_FEATURES


def build():
    df = add_features(hourly_table())
    path = ROOT / "data" / "hourly_features.parquet"
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    d = build()
    print(d.shape, d.index[0], d.index[-1])
    print(d[["price"] + FUND_FEATURES[:5]].describe().T.round(1))
    print("missing share per feature:")
    print(d[ALL_FEATURES].isna().mean().round(4).sort_values(ascending=False).head(10))
