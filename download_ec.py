"""Download hourly intraday-continuous prices, gas and CO2 for DE-LU from Energy-Charts (Fraunhofer ISE).

Source: the yearly chart-data files behind energy-charts.info "Spot market prices".
Intraday data originates from EPEX SPOT: private/internal research use only, do not redistribute.

Columns (EUR/MWh, gas EUR/MWh, CO2 EUR/t), hourly, index = delivery-hour start in UTC:
  da_ec, id_avg, id_low, id_high, id1, id3, co2_de, co2_eu, gas_the, gas_co2_cost
ID1 / ID3 = volume-weighted average of continuous trades in the last 1 / 3 hours before delivery.

    python3 download_ec.py            # 2019 .. this year (this year always refreshed)
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
URL = "https://www.energy-charts.info/charts/price_spot_market/data/de/year_{year}.json"

NAMES = {
    "Day Ahead Auction (DE-LU)": "da_ec",
    "Intraday Continuous Average Price (DE-LU)": "id_avg",
    "Intraday Continuous Low Price (DE-LU)": "id_low",
    "Intraday Continuous High Price (DE-LU)": "id_high",
    "Intraday Continuous ID1-Price (DE-LU)": "id1",
    "Intraday Continuous ID3-Price (DE-LU)": "id3",
    "CO2 Emission Allowances, Auction DE": "co2_de",
    "CO2 Emission Allowances, Auction EU": "co2_eu",
    "Gas (NCG, THE)": "gas_the",
    "2 × Gas + 0.4 × CO2 (DE)": "gas_co2_cost",
}


def series_name(s):
    n = s.get("name")
    if isinstance(n, list):
        n = n[0]
    return n.get("en") if isinstance(n, dict) else n


def fetch_year(year):
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "power-de research"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                raw = json.load(r)
            break
        except Exception as exc:
            print(f"  {year}: {exc}; retry")
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"energy-charts {year} failed")
    idx = pd.to_datetime(raw[0]["xAxisValues"], unit="ms", utc=True)
    cols = {}
    for s in raw:
        col = NAMES.get(series_name(s))
        if col and len(s.get("data") or []) == len(idx):
            cols[col] = pd.to_numeric(pd.Series(s["data"], index=idx), errors="coerce")
    df = pd.DataFrame(cols)
    df.index.name = "ts_utc"
    return df.astype("float32")


def main():
    this_year = pd.Timestamp.now().year
    years = range(int(sys.argv[1]) if len(sys.argv) > 1 else 2019, this_year + 1)
    RAW.mkdir(parents=True, exist_ok=True)
    for year in years:
        out = RAW / f"ec_prices_{year}.parquet"
        if out.exists() and year != this_year:
            continue
        df = fetch_year(year)
        df.to_parquet(out)
        print(f"ec_prices {year}: {len(df)} rows, cols={list(df.columns)}")


if __name__ == "__main__":
    main()
