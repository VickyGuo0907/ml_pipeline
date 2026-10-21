"""Stage the pjm_load_forecast weather companion file: hourly temperature,
dew point, relative humidity and wind speed for the PJM East (PJME) region,
pulled from the Open-Meteo Historical Weather API.

Why weather: regional load is driven mostly by HVAC demand (air conditioning
on hot, humid days; heating on cold, windy ones). The load series alone
can't anticipate a heat wave or cold snap; temperature can.

Why these stations: PJME spans the Philadelphia, northern New Jersey,
Baltimore and Washington DC metros, so one airport per metro (PHL, EWR, BWI,
DCA) is averaged rather than trusting a single location. Stations and their
weights live in config/pjm_load_forecast/cleaning.yaml (the clean stage reads
the same list to aggregate them), not here.

Why Open-Meteo: free, no API key, hourly back to 1940 (ERA5 reanalysis), so
it covers PJME_hourly.csv's 2002-2018 span with no gaps. Reanalysis is a
gridded model estimate rather than the airport's own sensor, which smooths
some local extremes; good enough to capture the load-temperature
relationship.

Time zone: the PJM load file uses local prevailing (Eastern) wall-clock time
with no UTC offset. Weather is requested in UTC and converted to the same
naive Eastern wall clock, so it repeats/skips the same DST hours the load
file does and both are deduplicated/filled identically by the clean stage.

The landing file is the only output. Requests are chunked by year with a
pause between calls to stay inside Open-Meteo's free-tier rate limits.

Run: uv run python -m scripts.stage_pjm_load_forecast_landing
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_forecast_cleaning_config

API_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo variable name -> landing-file column (the clean stage's contract).
HOURLY_VARIABLES: dict[str, str] = {
    "temperature_2m": "temp_f",
    "dew_point_2m": "dewpoint_f",
    "relative_humidity_2m": "rel_humidity",
    "wind_speed_10m": "wind_mph",
}
# One day of padding either side of PJME_hourly.csv (2002-01-01 .. 2018-08-03)
# so the UTC -> Eastern conversion never leaves the first/last load hours
# without weather.
DEFAULT_START = "2001-12-31"
DEFAULT_END = "2018-08-04"
DEFAULT_CONFIG_DIR = Path("config/pjm_load_forecast")
DEFAULT_DEST = Path("data/pjm_load_forecast/landing")
OUTPUT_FILENAME = "weather_hourly.csv"
LOCAL_TZ = "America/New_York"
PAUSE_SECONDS = 1.0
MAX_RETRIES = 4

Fetcher = Callable[[dict[str, Any]], dict[str, Any]]


def fetch_open_meteo(params: dict[str, Any]) -> dict[str, Any]:
    """GET the Open-Meteo archive API, retrying with backoff on HTTP 429/5xx."""
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            retryable = e.code == 429 or e.code >= 500
            if not retryable or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("unreachable")


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-year chunks (inclusive dates)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    chunks = []
    for year in range(s.year, e.year + 1):
        c_start = max(s, pd.Timestamp(year=year, month=1, day=1))
        c_end = min(e, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((c_start.strftime("%Y-%m-%d"), c_end.strftime("%Y-%m-%d")))
    return chunks


def _to_frame(payload: dict[str, Any], station: str) -> pd.DataFrame:
    """Convert one Open-Meteo JSON response to landing-file rows."""
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"Open-Meteo response for {station} has no hourly data: {str(payload)[:200]}")
    utc = pd.to_datetime(hourly["time"]).tz_localize("UTC")
    local = utc.tz_convert(LOCAL_TZ).tz_localize(None)
    frame = pd.DataFrame({"Datetime": local, "station": station})
    for api_name, column in HOURLY_VARIABLES.items():
        frame[column] = pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce").to_numpy()
    return frame


def stage_landing(
    stations: list[dict[str, Any]],
    start: str,
    end: str,
    dest: Path,
    fetcher: Fetcher = fetch_open_meteo,
    pause_seconds: float = PAUSE_SECONDS,
) -> dict[str, Any]:
    """Pull hourly weather for each station and write one long-format CSV
    (Datetime, station, temp_f, dewpoint_f, rel_humidity, wind_mph) to dest.

    Args:
        stations: [{"name", "latitude", "longitude"}, ...].
        start, end: Inclusive date range, YYYY-MM-DD.
        dest: Landing directory (created if absent).
        fetcher: Callable taking request params and returning the API's JSON
            (injectable so tests don't hit the network).
        pause_seconds: Sleep between requests (rate-limit courtesy).

    Returns:
        Dict with output_path, station_count, row_count, start, end.

    Raises:
        ValueError: If no stations are given or a response has no data.
    """
    if not stations:
        raise ValueError("No weather stations configured.")
    frames = []
    chunks = _year_chunks(start, end)
    for station in stations:
        for chunk_start, chunk_end in chunks:
            params = {
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "start_date": chunk_start,
                "end_date": chunk_end,
                "hourly": ",".join(HOURLY_VARIABLES),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "GMT",
            }
            frames.append(_to_frame(fetcher(params), station["name"]))
            if pause_seconds:
                time.sleep(pause_seconds)

    out = pd.concat(frames, ignore_index=True).sort_values(["station", "Datetime"])
    dest.mkdir(parents=True, exist_ok=True)
    output_path = dest / OUTPUT_FILENAME
    out.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "station_count": len(stations),
        "row_count": len(out),
        "start": str(out["Datetime"].min()),
        "end": str(out["Datetime"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    weather_cfg = load_forecast_cleaning_config(args.config_dir).weather
    if weather_cfg is None:
        raise SystemExit(f"No weather block in {args.config_dir}/cleaning.yaml - nothing to stage.")
    stations = [st.model_dump() for st in weather_cfg.stations]
    print(f"Fetching {len(stations)} stations x {len(_year_chunks(args.start, args.end))} years "
          f"from Open-Meteo ({args.start} to {args.end})...")
    result = stage_landing(stations, args.start, args.end, args.dest)
    print(f"Wrote {result['row_count']:,} rows for {result['station_count']} stations "
          f"({result['start']} to {result['end']}) to {result['output_path']}")


if __name__ == "__main__":
    main()
