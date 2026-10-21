"""Neighbouring bidding zones and generation outages from ENTSO-E Transparency.

1) For each neighbour of DE-LU: day-ahead price, load forecast, wind/solar forecast
   -> data/raw/nb_{zone}_{dataset}_{year}.parquet (UTC, hourly means)
2) Unavailability of generation units (REMIT outage messages, all revisions, with publication
   time) for FR and DE -> data/raw/outages_{country}_{yyyymm}.parquet. Publication time lets the
   features use only what was known at gate closure.

    python3 download_neighbours.py zones
    python3 download_neighbours.py outages [FR DE]
"""
import os
import sys
import time

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from download import RAW, TZ, api_key

FIRST_YEAR = int(os.environ.get("FIRST_YEAR", "2019"))
ZONES = ["FR", "NL", "BE", "AT", "PL", "CZ", "DK_1", "CH"]
DATASETS = {
    "price": lambda c, z, s, e: c.query_day_ahead_prices(z, start=s, end=e).to_frame("price"),
    "load": lambda c, z, s, e: c.query_load_forecast(z, start=s, end=e).iloc[:, :1].set_axis(["load_fc"], axis=1),
    "ws": lambda c, z, s, e: c.query_wind_and_solar_forecast(z, start=s, end=e),
}


def retry(fn, label):
    for attempt in range(5):
        try:
            return fn()
        except NoMatchingDataError:
            return None
        except Exception as exc:
            wait = 20 * (attempt + 1)
            print(f"  {label}: {type(exc).__name__}: {str(exc)[:120]} -> retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"  {label}: giving up", flush=True)
    return None


def zones(client):
    this_year = pd.Timestamp.now().year
    for year in range(FIRST_YEAR, this_year + 1):
        s = pd.Timestamp(f"{year}-01-01", tz=TZ)
        e = min(pd.Timestamp(f"{year + 1}-01-01", tz=TZ), pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=2))
        for z in ZONES:
            for name, fn in DATASETS.items():
                out = RAW / f"nb_{z}_{name}_{year}.parquet"
                if out.exists() and year != this_year:
                    continue
                df = retry(lambda: fn(client, z, s, e), f"{z} {name} {year}")
                if df is None or df.empty:
                    print(f"{z} {name} {year}: no data", flush=True)
                    continue
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.index = df.index.tz_convert("UTC")
                df = df.resample("1h").mean().astype("float32")
                df.index.name = "ts_utc"
                df.to_parquet(out)
                print(f"{z} {name} {year}: {len(df)} rows", flush=True)


def outages(client, countries):
    cols = ["created_doc_time", "docstatus", "businesstype", "mrid", "revision", "plant_type",
            "production_resource_name", "nominal_power", "avail_qty", "start", "end"]
    months = pd.date_range(f"{FIRST_YEAR}-01-01", pd.Timestamp.now().normalize() + pd.offsets.MonthBegin(1), freq="MS")
    for country in countries:
        for m in months:
            out = RAW / f"outages_{country}_{m:%Y%m}.parquet"
            current = m >= pd.Timestamp.now().normalize() - pd.offsets.MonthBegin(1)
            if out.exists() and not current:
                continue
            s, e = m.tz_localize(TZ), (m + pd.offsets.MonthBegin(1)).tz_localize(TZ)
            t0 = time.time()
            df = retry(lambda: client.query_unavailability_of_generation_units(country, start=s, end=e),
                       f"outages {country} {m:%Y-%m}")
            if df is None or df.empty:
                print(f"outages {country} {m:%Y-%m}: none", flush=True)
                continue
            df = df.reset_index().rename(columns={"index": "created_doc_time"})
            df = df[[c for c in cols if c in df.columns]]
            for c in ("created_doc_time", "start", "end"):
                df[c] = pd.to_datetime(df[c], utc=True)
            for c in ("nominal_power", "avail_qty"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["revision"] = pd.to_numeric(df["revision"], errors="coerce")
            df.astype({c: "string" for c in ("docstatus", "businesstype", "mrid", "plant_type",
                                             "production_resource_name")}).to_parquet(out)
            print(f"outages {country} {m:%Y-%m}: {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)


def main():
    client = EntsoePandasClient(api_key=api_key(), retry_count=3, retry_delay=10)
    what = sys.argv[1] if len(sys.argv) > 1 else "zones"
    if what == "zones":
        zones(client)
    else:
        outages(client, sys.argv[2:] or ["FR", "DE_LU"])


if __name__ == "__main__":
    main()
