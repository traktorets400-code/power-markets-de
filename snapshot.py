"""Record forecast *vintages*: what the forecasts looked like at the moment of the run.

ENTSO-E and Energy-Charts only serve the latest version of each forecast, so a
backtest on downloaded history silently uses forecasts that may have been revised
after the trading decision. Running this several times a day builds a true
point-in-time archive: data/snapshots/YYYY/MM/DD_HHMM.parquet (long format:
source, series, ts_utc, value, snapshot_utc).

Scheduled by systemd timer power-snapshot.timer (see deploy/).
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from download import ZONE, TZ, api_key

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "data" / "snapshots"
EC_URL = "https://api.energy-charts.info/v2/public_power_forecast?country=de&production_type={pt}&forecast_type={ft}&start={start}&end={end}"


def long(df, source):
    df = df.copy()
    df.index = df.index.tz_convert("UTC")
    out = df.stack().rename("value").reset_index()
    out.columns = ["ts_utc", "series", "value"]
    out["source"] = source
    return out


def entsoe_parts(start, end):
    c = EntsoePandasClient(api_key=api_key(), retry_count=3, retry_delay=10)
    jobs = {
        "load_fc_da": lambda: c.query_load_forecast(ZONE, start=start, end=end),
        "ws_fc_da": lambda: c.query_wind_and_solar_forecast(ZONE, start=start, end=end),
        "ws_fc_id": lambda: c.query_intraday_wind_and_solar_forecast(ZONE, start=start, end=end),
        "gen_fc_da": lambda: c.query_generation_forecast(ZONE, start=start, end=end),
        "price_da": lambda: c.query_day_ahead_prices(ZONE, start=start, end=end),
    }
    parts = []
    for name, job in jobs.items():
        try:
            df = job()
        except NoMatchingDataError:
            continue
        except Exception as exc:
            print(f"entsoe {name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(df, pd.Series):
            df = df.to_frame(name)
        df.columns = [f"{name}:{col}" for col in df.columns]
        parts.append(long(df, "entsoe"))
    return parts


def energy_charts_parts(start, end):
    parts = []
    for pt in ("solar", "wind_onshore", "wind_offshore", "load"):
        for ft in (("day-ahead",) if pt == "load" else ("day-ahead", "intraday", "current")):
            time.sleep(3)  # energy-charts rate limit (HTTP 429)
            url = EC_URL.format(pt=pt, ft=ft, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
            raw = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "power-de research"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        raw = json.load(r)
                    break
                except Exception as exc:
                    print(f"energy-charts {pt}/{ft}: {exc}")
                    time.sleep(20 * (attempt + 1))
            if raw is None:
                continue
            rows = raw.get("data") or []
            if not rows:
                continue
            ts = pd.to_datetime([r["timestamp"] for r in rows], utc=True)
            key = next(iter(rows[0]["values"]))
            vals = [r["values"].get(key) for r in rows]
            df = pd.DataFrame({f"{pt}:{ft}": pd.to_numeric(vals, errors="coerce")}, index=ts)
            parts.append(long(df.dropna(), "energy_charts"))
    return parts


def main():
    now = pd.Timestamp.now(tz="UTC")
    start = now.tz_convert(TZ).normalize()
    end = start + pd.Timedelta(days=2)
    t0 = time.time()
    parts = entsoe_parts(start, end) + energy_charts_parts(start, end)
    if not parts:
        raise SystemExit("snapshot: nothing fetched")
    snap = pd.concat(parts, ignore_index=True)
    snap["snapshot_utc"] = now
    snap["value"] = snap["value"].astype("float32")
    local = now.tz_convert(TZ)
    out = SNAP / f"{local:%Y}" / f"{local:%m}" / f"{local:%d_%H%M}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    snap.to_parquet(out)
    print(f"snapshot {out.relative_to(ROOT)}: {len(snap)} rows, "
          f"{snap['series'].nunique()} series, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
