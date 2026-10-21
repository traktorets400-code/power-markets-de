"""Download DE-LU fundamentals + day-ahead prices from ENTSO-E Transparency.

One parquet per (dataset, year) in data/raw/, index in UTC at native resolution
(hourly until 2025-09-30, 15-min afterwards for prices). Re-running skips years
already on disk, except the current year which is always refreshed.

    python3 download.py               # 2019 .. today
    python3 download.py 2024 2026     # subset of years
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
ZONE = "DE_LU"
TZ = "Europe/Berlin"


def api_key():
    key = os.environ.get("ENTSOE_API_KEY")
    if key:
        return key.strip()
    return (ROOT / ".entsoe_key").read_text().strip()


# name -> function(client, start, end) returning a DataFrame
DATASETS = {
    "price_da": lambda c, s, e: c.query_day_ahead_prices(ZONE, start=s, end=e).to_frame("price"),
    "load_fc": lambda c, s, e: c.query_load_forecast(ZONE, start=s, end=e).rename(
        columns={"Forecasted Load": "load_fc"}),
    "load_act": lambda c, s, e: c.query_load(ZONE, start=s, end=e).rename(
        columns={"Actual Load": "load_act"}),
    "ws_fc": lambda c, s, e: c.query_wind_and_solar_forecast(ZONE, start=s, end=e).rename(
        columns={"Solar": "solar_fc", "Wind Offshore": "wind_off_fc", "Wind Onshore": "wind_on_fc"}),
    "gen_fc": lambda c, s, e: _gen_fc(c, s, e),
}


def _gen_fc(c, s, e):
    g = c.query_generation_forecast(ZONE, start=s, end=e)
    if isinstance(g, pd.DataFrame):
        g = g.iloc[:, 0]
    return g.to_frame("gen_fc")


def fetch(client, name, year):
    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz=TZ),
              pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=2))
    for attempt in range(4):
        try:
            df = DATASETS[name](client, start, end)
            break
        except NoMatchingDataError:
            return None
        except Exception as exc:  # network hiccups, 503s from the platform
            wait = 10 * (attempt + 1)
            print(f"   {name} {year}: {type(exc).__name__}: {exc} -> retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"giving up on {name} {year}")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "ts_utc"
    return df.astype("float32")


def main():
    years = list(range(2019, pd.Timestamp.now().year + 1))
    if len(sys.argv) == 3:
        years = list(range(int(sys.argv[1]), int(sys.argv[2]) + 1))
    RAW.mkdir(parents=True, exist_ok=True)
    client = EntsoePandasClient(api_key=api_key(), retry_count=3, retry_delay=10)
    this_year = pd.Timestamp.now().year
    for year in years:
        for name in DATASETS:
            out = RAW / f"{name}_{year}.parquet"
            if out.exists() and year != this_year:
                continue
            t0 = time.time()
            df = fetch(client, name, year)
            if df is None:
                print(f"{name} {year}: no data")
                continue
            df.to_parquet(out)
            print(f"{name} {year}: {len(df):6d} rows  {df.index[0]} .. {df.index[-1]}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
