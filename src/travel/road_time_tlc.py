"""Road and ridehail travel-time layer using TLC records when readable."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

from src.travel.utils import ensure_dir, haversine_meters

PICKUP_COLS = ["pickup_datetime", "request_datetime", "on_scene_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"]
DROPOFF_COLS = ["dropoff_datetime", "tpep_dropoff_datetime", "lpep_dropoff_datetime"]


def departure_period(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if 7 <= hour < 10:
        return "morning_peak"
    if 10 <= hour < 16:
        return "midday"
    if 16 <= hour < 19:
        return "evening_peak"
    if hour >= 22 or hour < 6:
        return "night"
    return "other"


def _find(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _parquet_supported() -> bool:
    return bool(importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet"))


def inspect_travel_files(raw_travel_dir: str | Path = "data/raw/nyc/real/travel", output_dir: str | Path = "outputs") -> dict[str, Any]:
    raw_path = Path(raw_travel_dir)
    output = ensure_dir(output_dir)
    files = sorted(raw_path.glob("*.parquet"))
    lookup = raw_path / "taxi_zone_lookup.csv"
    zips = sorted(raw_path.glob("*.zip"))
    parquet_supported = _parquet_supported()
    lines = ["# Step 11 travel data inspection", "", f"- Parquet engine available: {parquet_supported}", ""]
    lines.append("## Parquet trip files")
    summaries = []
    for path in files:
        item = {"file": path.name, "readable": False}
        lines.append(f"### {path.name}")
        if parquet_supported:
            try:
                df = pd.read_parquet(path)
                item.update({"readable": True, "rows": len(df), "columns": list(df.columns)})
                pickup = _find(list(df.columns), PICKUP_COLS)
                dropoff = _find(list(df.columns), DROPOFF_COLS)
                item.update({"pickup_datetime_column": pickup, "dropoff_datetime_column": dropoff, "pickup_zone_column": _find(list(df.columns), ["PULocationID"]), "dropoff_zone_column": _find(list(df.columns), ["DOLocationID"])})
                lines.append(f"- Rows: {len(df):,}")
                lines.append(f"- Columns: {', '.join(map(str, df.columns))}")
                lines.append(f"- Usable datetime columns: pickup={pickup}, dropoff={dropoff}")
                lines.append(f"- Usable zone columns: pickup={item['pickup_zone_column']}, dropoff={item['dropoff_zone_column']}")
            except Exception as exc:
                item["error"] = str(exc)
                lines.append(f"- Read failed: {exc}")
        else:
            item["error"] = "pyarrow/fastparquet unavailable"
            lines.append("- Read skipped: pyarrow/fastparquet is unavailable.")
        summaries.append(item)
    lines.extend(["", "## Taxi zone lookup", ""])
    if lookup.exists():
        df = pd.read_csv(lookup)
        lines.append(f"- File: `{lookup}`")
        lines.append(f"- Rows: {len(df):,}")
        lines.append(f"- Columns: {', '.join(df.columns)}")
    else:
        lines.append("- Missing taxi_zone_lookup.csv")
    lines.extend(["", "## Taxi zone geometry", ""])
    lines.append(f"- Zip files: {', '.join(p.name for p in zips) if zips else 'none'}")
    lines.append("- Taxi zone geometry can be read: no, because geopandas/pyshp/shapely are unavailable in this environment.")
    lines.extend(["", "## Known limitations", "", "- TLC observed OD times require a Parquet engine such as pyarrow or fastparquet.", "- Taxi-zone polygon assignment requires a geospatial reader; fallback uses distance-based road estimates."])
    (output / "step11_travel_data_inspection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"parquet_supported": parquet_supported, "files": summaries, "taxi_lookup_exists": lookup.exists(), "taxi_geometry_readable": False}


def aggregate_tlc_road_times(raw_travel_dir: str | Path, processed_dir: str | Path, output_dir: str | Path = "outputs", max_trip_files: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = ensure_dir(processed_dir)
    report_dir = ensure_dir(output_dir)
    raw_path = Path(raw_travel_dir)
    files = sorted(raw_path.glob("fhvhv_tripdata_*.parquet")) + sorted(raw_path.glob("yellow_tripdata_*.parquet")) + sorted(raw_path.glob("green_tripdata_*.parquet"))
    if max_trip_files:
        files = files[:max_trip_files]
    processed = []
    records = []
    if not _parquet_supported():
        summary = {"status": "blocked_missing_parquet_engine", "files_seen": len(files), "rows_read": 0, "rows_retained": 0, "od_period_records": 0}
        pd.DataFrame(columns=["origin_zone_id", "destination_zone_id", "departure_period", "median_road_time_minutes", "p25_time_minutes", "p75_time_minutes", "trip_count", "source"]).to_csv(output / "road_travel_times.csv", index=False)
        _write_road_summary(report_dir, summary, processed)
        return pd.DataFrame(), summary
    for path in files:
        try:
            df = pd.read_parquet(path)
            pickup = _find(list(df.columns), PICKUP_COLS)
            dropoff = _find(list(df.columns), DROPOFF_COLS)
            if not pickup or not dropoff or "PULocationID" not in df.columns or "DOLocationID" not in df.columns:
                processed.append({"file": path.name, "status": "missing_required_columns", "rows_read": len(df), "rows_retained": 0})
                continue
            start = pd.to_datetime(df[pickup], errors="coerce")
            end = pd.to_datetime(df[dropoff], errors="coerce")
            duration = (end - start).dt.total_seconds() / 60.0
            keep = duration.gt(0) & duration.le(180) & df["PULocationID"].notna() & df["DOLocationID"].notna()
            tmp = pd.DataFrame({"origin_zone_id": df.loc[keep, "PULocationID"].astype(int), "destination_zone_id": df.loc[keep, "DOLocationID"].astype(int), "departure_period": start.loc[keep].map(departure_period), "duration": duration.loc[keep]})
            records.append(tmp)
            processed.append({"file": path.name, "status": "processed", "rows_read": len(df), "rows_retained": len(tmp)})
        except Exception as exc:
            processed.append({"file": path.name, "status": "failed", "error": str(exc), "rows_read": 0, "rows_retained": 0})
    if not records:
        summary = {"status": "no_usable_trip_rows", "files_seen": len(files), "rows_read": sum(x.get("rows_read", 0) for x in processed), "rows_retained": 0, "od_period_records": 0}
        pd.DataFrame().to_csv(output / "road_travel_times.csv", index=False)
        _write_road_summary(report_dir, summary, processed)
        return pd.DataFrame(), summary
    trips = pd.concat(records, ignore_index=True)
    grouped = trips.groupby(["origin_zone_id", "destination_zone_id", "departure_period"])["duration"]
    result = grouped.agg(median_road_time_minutes="median", p25_time_minutes=lambda s: s.quantile(0.25), p75_time_minutes=lambda s: s.quantile(0.75), trip_count="count").reset_index()
    result["source"] = "TLC_OBSERVED"
    result.to_csv(output / "road_travel_times.csv", index=False)
    summary = {"status": "generated", "files_seen": len(files), "rows_read": sum(x.get("rows_read", 0) for x in processed), "rows_retained": len(trips), "od_period_records": len(result)}
    _write_road_summary(report_dir, summary, processed, result)
    return result, summary


def _write_road_summary(output_dir: Path, summary: dict[str, Any], processed: list[dict[str, Any]], result: pd.DataFrame | None = None) -> None:
    lines = ["# Step 11 road time summary", "", f"- Status: {summary.get('status')}", f"- Files seen: {summary.get('files_seen', 0)}", f"- Rows read: {summary.get('rows_read', 0)}", f"- Rows retained: {summary.get('rows_retained', 0)}", f"- OD-period records generated: {summary.get('od_period_records', 0)}", "", "## Files processed", ""]
    for item in processed:
        lines.append(f"- {item.get('file')}: {item.get('status')} | read={item.get('rows_read', 0)} | retained={item.get('rows_retained', 0)} {item.get('error', '')}")
    if result is not None and not result.empty:
        desc = result["median_road_time_minutes"].describe().to_dict()
        lines.extend(["", "## Median duration distribution", ""])
        for key, value in desc.items():
            lines.append(f"- {key}: {value:.3f}")
    lines.extend(["", "## Limitations", "", "- TLC observed road times are only used when local Parquet support is available.", "- Without taxi-zone polygon assignment, area-resource road times use distance-based fallback."])
    (output_dir / "step11_road_time_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_area_resource_road_fallback(areas: pd.DataFrame, resources: pd.DataFrame, processed_dir: str | Path, car_speed_kmh: float = 22.0) -> pd.DataFrame:
    output = ensure_dir(processed_dir)
    rows = []
    for _, area in areas.iterrows():
        for _, resource in resources.iterrows():
            distance = haversine_meters(area["lon"], area["lat"], resource["lon"], resource["lat"])
            minutes = (distance / 1000.0) / car_speed_kmh * 60.0 * 1.35 + 4.0
            for period in ["morning_peak", "midday", "evening_peak", "night", "other"]:
                factor = {"morning_peak": 1.2, "evening_peak": 1.25, "night": 0.85, "midday": 1.0, "other": 1.05}[period]
                rows.append({"area_id": area["area_id"], "resource_id": resource["resource_id"], "activity_type": resource["activity_type"], "departure_period": period, "mode": "ridehail", "travel_time_minutes": round(minutes * factor, 3), "trip_count": 0, "source": "ROAD_DISTANCE_FALLBACK"})
    result = pd.DataFrame(rows)
    result.to_csv(output / "area_resource_road_travel_times.csv", index=False)
    return result
