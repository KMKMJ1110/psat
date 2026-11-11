"""Download and aggregate CAMS Total AOD 550 nm using the team's forecast policy."""

import argparse
from calendar import monthrange
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


START = date(2021, 1, 19)
END = date(2026, 5, 31)
ONE_DAY = timedelta(days=1)
KST = "Asia/Seoul"
ADS_API = "https://ads.atmosphere.copernicus.eu/api"
DATASET = "cams-global-atmospheric-composition-forecasts"
VARIABLE = "total_aerosol_optical_depth_550nm"
PARAM_ID = 210207
AREA = [39.0, 124.0, 33.0, 131.5]
LEADS = range(27, 52)
AOD_COLS = ["cams_total_aod_mean", "cams_total_aod_p90"]
AOD_PREV = [f"{col}_prev" for col in AOD_COLS]
AOD_NEXT = [f"{col}_next" for col in AOD_COLS]
AOD_FINAL = [*AOD_COLS, *AOD_PREV, *AOD_NEXT]
EXCLUDED = "cams_excluded"
FINAL_NAME = "cams_total_aod_features_final.parquet"


def init_for(day):
    return datetime.combine(day - 2 * ONE_DAY, time(12), timezone.utc)


def load_exclusions(path):
    excluded = {}
    for period in json.loads(path.read_text(encoding="utf-8-sig")):
        first = date.fromisoformat(period["init_start_utc"])
        last = date.fromisoformat(period["init_end_utc"])
        if last < first:
            raise ValueError(f"Invalid exclusion range: {first} to {last}")
        for offset in range((last - first).days + 1):
            day = first + timedelta(days=offset + 2)
            if day in excluded:
                raise ValueError(f"Duplicate exclusion for {day}")
            excluded[day] = {
                "target_day_kst": day.isoformat(),
                "init_utc": init_for(day).isoformat(),
                "scheduled_release_kst": pd.Timestamp(init_for(day) + timedelta(hours=10)).tz_convert(KST).isoformat(),
                "status": period["status"],
                "reason": period["reason"],
                "sources": period["sources"],
            }
    return excluded


def request_for(first, last):
    return {
        "variable": [VARIABLE],
        "date": [f"{init_for(first).date()}/{init_for(last).date()}"],
        "time": ["12:00"],
        "leadtime_hour": [str(lead) for lead in LEADS],
        "type": ["forecast"],
        "area": AREA,
        "data_format": "grib",
    }


def monthly_batches(first, end):
    while first <= end:
        init_day = init_for(first).date()
        last_init = init_day.replace(day=monthrange(init_day.year, init_day.month)[1])
        last = min(end, last_init + 2 * ONE_DAY)
        yield first, last
        first = last + ONE_DAY


def publish_new(staged, target):
    """Publish a complete file without replacing an existing destination."""
    if os.name == "nt":
        staged.rename(target)
    else:
        os.link(staged, target)


def download_grib(raw_dir, batches, local_only):
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    client = None
    for first, last in batches:
        target = raw_dir / f"cams_total_aod_{first:%Y%m%d}_{last:%Y%m%d}.grib"
        if not target.is_file():
            if local_only:
                raise FileNotFoundError(f"Missing cached Total AOD GRIB: {target}")
            if client is None:
                import cdsapi

                client = cdsapi.Client(url=ADS_API)
            print(f"Downloading Total AOD for KST delivery days {first} to {last}", flush=True)
            with TemporaryDirectory(dir=raw_dir) as staging:
                partial = Path(staging) / target.name
                client.retrieve(DATASET, request_for(first, last)).download(str(partial))
                with partial.open("rb") as handle:
                    if handle.read(4) != b"GRIB":
                        raise ValueError("ADS response was not an uncompressed GRIB file")
                try:
                    publish_new(partial, target)
                except FileExistsError:
                    print(f"Using concurrently downloaded {target.name}", flush=True)
        else:
            print(f"Using cached {target.name}", flush=True)
        files.append(target)
    return files


def grib_time(eccodes, message, date_key, time_key):
    ymd = eccodes.codes_get_long(message, date_key)
    hm = eccodes.codes_get_long(message, time_key)
    return datetime.strptime(f"{ymd:08d}{hm:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def read_grib_fields(files, start, end, excluded):
    import eccodes

    fields = {}
    grids = {}
    north, west, south, east = AREA
    for path in files:
        found_total = False
        # ecCodes reads from the descriptor, so the magic-byte check must not read ahead.
        with path.open("rb", buffering=0) as handle:
            if handle.read(4) != b"GRIB":
                raise ValueError(f"Expected an uncompressed GRIB file: {path}")
            handle.seek(0)
            while (message := eccodes.codes_grib_new_from_file(handle)) is not None:
                try:
                    if eccodes.codes_get_long(message, "paramId") != PARAM_ID:
                        continue
                    found_total = True
                    init = grib_time(eccodes, message, "dataDate", "dataTime")
                    day = init.date() + 2 * ONE_DAY
                    if init != init_for(day) or not start <= day <= end or day in excluded:
                        continue
                    valid = grib_time(eccodes, message, "validityDate", "validityTime")
                    lead = (valid - init).total_seconds() / 3600
                    if lead not in LEADS:
                        continue
                    lat = eccodes.codes_get_array(message, "latitudes")
                    lon = eccodes.codes_get_array(message, "longitudes")
                    keep = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
                    if not keep.any():
                        raise ValueError(f"No grid cells in the fixed Korea area: {path}")
                    lat, lon = lat[keep], lon[keep]
                    order = np.lexsort((lon, lat))
                    grid = np.column_stack((lat[order], lon[order]))
                    if day in grids and not np.array_equal(grids[day], grid):
                        raise ValueError(f"Grid differs across hours: {day}")
                    grids[day] = grid
                    values = eccodes.codes_get_values(message)[keep][order].astype("float32")
                    missing = eccodes.codes_get_double(message, "missingValue")
                    values[~np.isfinite(values) | (values < 0) | (values == missing)] = np.nan
                    key = (day, int(lead))
                    if key in fields and not np.allclose(fields[key], values, equal_nan=True):
                        raise ValueError(f"Conflicting forecasts for delivery day {day}, lead {lead:g}h")
                    fields[key] = values
                finally:
                    eccodes.codes_release(message)
        if not found_total:
            raise ValueError(f"GRIB contains no Total AOD 550 nm (paramId {PARAM_ID}): {path}")
    return fields


def summarize_grib(fields, start, end, excluded):
    rows = []
    day = start
    while day <= end:
        for lead in range(28, 52):
            ts = pd.Timestamp(init_for(day) + timedelta(hours=lead)).tz_convert(KST).tz_localize(None)
            if day in excluded:
                rows.append((ts, np.nan, np.nan, True))
                continue
            left, right = fields.get((day, lead - 1)), fields.get((day, lead))
            if left is None or right is None:
                raise ValueError(f"Missing Total AOD endpoint for {day}, hour ending {ts}. Check the cached GRIB for that month.")
            hourly = (left + right) / 2
            valid = hourly[np.isfinite(hourly)]
            if not valid.size:
                raise ValueError(f"No valid Total AOD values for {day}, hour ending {ts}")
            rows.append((ts, valid.mean(), np.quantile(valid, 0.9), False))
        day += ONE_DAY
    table = pd.DataFrame(rows, columns=["ts", *AOD_COLS, EXCLUDED])
    delivery_day = (table["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    grouped = table.groupby(delivery_day, sort=False)[AOD_COLS]
    table[AOD_PREV] = grouped.shift(1)
    table[AOD_NEXT] = grouped.shift(-1)
    table[AOD_FINAL] = table[AOD_FINAL].astype("float32")
    return table[["ts", *AOD_FINAL, EXCLUDED]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Team data directory")
    parser.add_argument("--exclude-file", type=Path, default=Path(__file__).with_name("cams_delay_exclusions.json"))
    parser.add_argument("--start", type=date.fromisoformat, default=START, help="First KST delivery day")
    parser.add_argument("--end", type=date.fromisoformat, default=END, help="Last KST delivery day, inclusive")
    parser.add_argument("--dry-run", action="store_true", help="Print request without calling ADS")
    parser.add_argument("--local-only", action="store_true", help="Build using the saved Total AOD GRIB batches")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must be on or after --start")
    batches = list(monthly_batches(args.start, args.end))
    if args.dry_run:
        print(json.dumps({"dataset": DATASET, "batches": len(batches), "first_request": request_for(*batches[0])}, indent=2))
        return

    excluded = load_exclusions(args.exclude_file)
    external = args.root / "output" / "external"
    filename = FINAL_NAME if (args.start, args.end) == (START, END) else f"cams_total_aod_features_{args.start:%Y%m%d}_{args.end:%Y%m%d}.parquet"
    target = external / filename
    metadata = target.with_suffix(".provenance.json")
    if target.exists() or metadata.exists():
        raise FileExistsError(f"Total AOD output already exists: {target} or {metadata}")
    files = download_grib(external / "cams_raw_total_aod", batches, args.local_only)
    fields = read_grib_fields(files, args.start, args.end, excluded)
    table = summarize_grib(fields, args.start, args.end, excluded)
    omitted = [excluded[day] for day in sorted(excluded) if args.start <= day <= args.end]
    provenance = {
        "source": f"https://ads.atmosphere.copernicus.eu/datasets/{DATASET}",
        "doi": "10.24381/04a0b097",
        "license": "CC BY 4.0",
        "attribution": "Contains modified Copernicus Atmosphere Monitoring Service information",
        "variable": VARIABLE,
        "param_id": PARAM_ID,
        "request_parameters": request_for(args.start, args.end),
        "cycle": "D-2 12:00 UTC; scheduled dissemination D-2 22:00 UTC",
        "timestamp": "KST, hour ending; mean of the two forecast endpoints at each grid cell",
        "neighbor_policy": "Previous/next hour from the same forecast cycle; day-edge neighbors are null",
        "feature_columns": AOD_FINAL,
        "raw_files": [file.name for file in files],
        "excluded_cycles": omitted,
        "excluded_cams_hours": int(table[EXCLUDED].sum()),
        "complete_cams_hours": int((~table[EXCLUDED]).sum()),
    }
    with TemporaryDirectory(dir=external) as staging:
        staged_table = Path(staging) / target.name
        staged_metadata = Path(staging) / metadata.name
        table.to_parquet(staged_table, index=False)
        staged_metadata.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        publish_new(staged_metadata, metadata)
        try:
            publish_new(staged_table, target)
        except BaseException:
            metadata.unlink()
            raise
    print(f"Saved {target}: {len(table):,} hours, {provenance['excluded_cams_hours']:,} excluded")
    print(f"Saved {metadata}")


if __name__ == "__main__":
    main()
