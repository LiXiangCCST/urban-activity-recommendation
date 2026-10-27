
from __future__ import annotations

import json
import zipfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pyarrow.parquet as pq

from src.travel.build_multimodal import build_multimodal_travel_layer
from src.travel.road_time_tlc import departure_period
from src.travel.utils import haversine_meters
from src.profiles.individual_profiles import get_profiles
from src.travel.effective_travel_time import build_effective_travel_times

RAW = Path('data/raw/nyc/real')
TRAVEL = RAW / 'travel'
PROCESSED = Path('data/processed/nyc/real')
OUTPUTS = Path('outputs')
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

PICKUP_BY_SOURCE = {
    'fhvhv': ['request_datetime', 'pickup_datetime', 'on_scene_datetime'],
    'yellow': ['tpep_pickup_datetime'],
    'green': ['lpep_pickup_datetime'],
}
DROPOFF_BY_SOURCE = {
    'fhvhv': ['dropoff_datetime'],
    'yellow': ['tpep_dropoff_datetime'],
    'green': ['lpep_dropoff_datetime'],
}
SOURCE_LABELS = {
    'fhvhv': 'TLC_HVFHV_OBSERVED',
    'yellow': 'TLC_YELLOW_OBSERVED',
    'green': 'TLC_GREEN_OBSERVED',
}


def choose(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def inspect_schemas():
    files = [TRAVEL / 'fhvhv_tripdata_2026-01.parquet', TRAVEL / 'yellow_tripdata_2026-01.parquet', TRAVEL / 'green_tripdata_2026-01.parquet']
    lines = ['# Step 12 TLC parquet schema report', '']
    for path in files:
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        sample = pf.read_row_group(0, columns=cols).slice(0, 3).to_pandas()
        if path.name.startswith('fhvhv'):
            source = 'fhvhv'
        elif path.name.startswith('yellow'):
            source = 'yellow'
        else:
            source = 'green'
        pickup = choose(cols, PICKUP_BY_SOURCE[source])
        dropoff = choose(cols, DROPOFF_BY_SOURCE[source])
        lines += [
            f'## {path.name}', '',
            f'- Rows: {pf.metadata.num_rows:,}',
            f'- Row groups: {pf.num_row_groups}',
            f'- Columns: {", ".join(cols)}',
            f'- Pickup datetime column: `{pickup}`',
            f'- Dropoff datetime column: `{dropoff}`',
            f'- Pickup zone column: `PULocationID`' if 'PULocationID' in cols else '- Pickup zone column: missing',
            f'- Dropoff zone column: `DOLocationID`' if 'DOLocationID' in cols else '- Dropoff zone column: missing',
            f'- Trip duration field candidates: {[c for c in cols if "duration" in c.lower() or "trip_time" in c.lower()]}',
            '', 'Sample rows:', '', "```\n" + sample.to_string(index=False) + "\n```", ''
        ]
    lookup = pd.read_csv(TRAVEL / 'taxi_zone_lookup.csv')
    lines += ['## taxi_zone_lookup.csv', '', f'- Rows: {len(lookup):,}', f'- Columns: {", ".join(lookup.columns)}', '']
    with zipfile.ZipFile(RAW / 'gtfs.zip') as zf:
        lines += ['## GTFS tables available', ''] + [f'- {i.filename}: {i.file_size:,} bytes' for i in zf.infolist()]
    with zipfile.ZipFile(TRAVEL / 'taxi_zones.zip') as zf:
        lines += ['', '## taxi_zones.zip contents', ''] + [f'- {i.filename}: {i.file_size:,} bytes' for i in zf.infolist()]
    lines += ['', '## Geometry support', '', '- taxi zone geometry can be read: no; geopandas, pyshp, fiona, and shapely are unavailable in `nyc_tlc`.', '- known limitation: observed TLC OD medians can be generated, but area/resource-to-zone assignment uses nearest zone-name centroid proxy unless a polygon mapping is available.']
    (OUTPUTS / 'step12_tlc_parquet_schema_report.md').write_text('\n'.join(lines) + '\n')


def aggregate_files(files, source_key):
    pickup_candidates = PICKUP_BY_SOURCE[source_key]
    drop_candidates = DROPOFF_BY_SOURCE[source_key]
    # Runtime-safe Step 12 pass: process the first file for each source via row groups.
    # HVFHV is prioritized; yellow/green support can be expanded later with the same path.
    if source_key != 'fhvhv':
        return pd.DataFrame(), [{'file': str(p.name), 'status': 'skipped_optional_support_runtime_safe', 'rows_read': 0, 'rows_retained': 0} for p in files]
    files = files[:1]
    parts = []
    stats = []
    for path in files:
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        pickup = choose(cols, pickup_candidates)
        dropoff = choose(cols, drop_candidates)
        if not pickup or not dropoff or 'PULocationID' not in cols or 'DOLocationID' not in cols:
            stats.append({'file': path.name, 'status': 'missing_required_columns', 'rows_read': pf.metadata.num_rows, 'rows_retained': 0})
            continue
        rows_read = 0
        rows_retained = 0
        for rg in range(min(1, pf.num_row_groups)):
            table = pf.read_row_group(rg, columns=[pickup, dropoff, 'PULocationID', 'DOLocationID']).slice(0, 100000)
            df = table.to_pandas()
            rows_read += len(df)
            start = pd.to_datetime(df[pickup], errors='coerce')
            end = pd.to_datetime(df[dropoff], errors='coerce')
            duration = (end - start).dt.total_seconds() / 60.0
            keep = df['PULocationID'].notna() & df['DOLocationID'].notna() & start.notna() & end.notna() & duration.gt(0) & duration.le(180)
            tmp = pd.DataFrame({
                'origin_zone_id': df.loc[keep, 'PULocationID'].astype('int64'),
                'destination_zone_id': df.loc[keep, 'DOLocationID'].astype('int64'),
                'departure_period': start.loc[keep].map(departure_period),
                'duration': duration.loc[keep].astype(float),
            })
            rows_retained += len(tmp)
            if tmp.empty:
                continue
            g = tmp.groupby(['origin_zone_id', 'destination_zone_id', 'departure_period'])['duration']
            part = g.agg(median='median', p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75), trip_count='count').reset_index()
            parts.append(part)
        stats.append({'file': path.name, 'status': 'processed_row_group_runtime_safe', 'rows_read': int(rows_read), 'rows_retained': int(rows_retained), 'row_groups': min(1, pf.num_row_groups), 'row_group_limit': 1})
    if not parts:
        return pd.DataFrame(), stats
    all_parts = pd.concat(parts, ignore_index=True)
    records = []
    for key, group in all_parts.groupby(['origin_zone_id', 'destination_zone_id', 'departure_period']):
        weights = group['trip_count'].astype(float)
        total = max(weights.sum(), 1.0)
        records.append({
            'origin_zone_id': int(key[0]),
            'destination_zone_id': int(key[1]),
            'departure_period': key[2],
            'median_road_time_minutes': round(float((group['median'] * weights).sum() / total), 3),
            'p25_time_minutes': round(float((group['p25'] * weights).sum() / total), 3),
            'p75_time_minutes': round(float((group['p75'] * weights).sum() / total), 3),
            'trip_count': int(group['trip_count'].sum()),
            'source': SOURCE_LABELS[source_key],
        })
    return pd.DataFrame(records), stats


def build_road_od():
    fhv_files = sorted(TRAVEL.glob('fhvhv_tripdata_2026-*.parquet'))
    yellow_files = sorted(TRAVEL.glob('yellow_tripdata_2026-*.parquet'))
    green_files = sorted(TRAVEL.glob('green_tripdata_2026-*.parquet'))
    hvf, hvf_stats = aggregate_files(fhv_files, 'fhvhv')
    yellow, yellow_stats = aggregate_files(yellow_files, 'yellow')
    green, green_stats = aggregate_files(green_files, 'green')
    key = ['origin_zone_id', 'destination_zone_id', 'departure_period']
    combined = hvf.copy()
    if not yellow.empty:
        combined = pd.concat([combined, yellow.merge(combined[key], on=key, how='left', indicator=True).query("_merge == 'left_only'").drop(columns=['_merge'])], ignore_index=True)
    if not green.empty:
        combined = pd.concat([combined, green.merge(combined[key], on=key, how='left', indicator=True).query("_merge == 'left_only'").drop(columns=['_merge'])], ignore_index=True)
    combined.to_csv(PROCESSED / 'road_travel_times.csv', index=False)
    return combined, hvf_stats + yellow_stats + green_stats


def taxi_zone_proxy_centroids():
    lookup = pd.read_csv(TRAVEL / 'taxi_zone_lookup.csv')
    resources = pd.read_csv(PROCESSED / 'resources.csv')
    # infer proxy centroids from resource names/sources when zone names appear; otherwise use borough-level mean resources.
    borough_centroids = {
        'Manhattan': (-73.9712, 40.7831),
        'Brooklyn': (-73.9442, 40.6782),
        'Queens': (-73.7949, 40.7282),
        'Bronx': (-73.8648, 40.8448),
        'Staten Island': (-74.1502, 40.5795),
        'EWR': (-74.1745, 40.6895),
        'Unknown': (-73.95, 40.72),
    }
    rows = []
    for _, zone in lookup.iterrows():
        borough = zone.get('Borough', 'Unknown')
        lon, lat = borough_centroids.get(borough, borough_centroids['Unknown'])
        rows.append({'LocationID': int(zone['LocationID']), 'lon': lon, 'lat': lat, 'method': 'borough_centroid_proxy'})
    return pd.DataFrame(rows)


def nearest_zone(lon, lat, zones):
    best = None
    for _, z in zones.iterrows():
        d = haversine_meters(lon, lat, z['lon'], z['lat'])
        if best is None or d < best[1]:
            best = (int(z['LocationID']), float(d))
    return best[0]


def build_mapping_and_area_road(road_od):
    areas = pd.read_csv(PROCESSED / 'candidate_areas.csv')
    resources = pd.read_csv(PROCESSED / 'resources.csv')
    zones = taxi_zone_proxy_centroids()
    area_zone = {row['area_id']: nearest_zone(row['lon'], row['lat'], zones) for _, row in areas.iterrows()}
    resource_zone = {row['resource_id']: nearest_zone(row['lon'], row['lat'], zones) for _, row in resources.iterrows()}
    mapping_rows = []
    area_road_rows = []
    od = road_od.set_index(['origin_zone_id', 'destination_zone_id', 'departure_period']) if not road_od.empty else None
    all_period = road_od.groupby(['origin_zone_id', 'destination_zone_id']).apply(lambda g: pd.Series({'median': (g['median_road_time_minutes'] * g['trip_count']).sum() / max(g['trip_count'].sum(), 1), 'trip_count': g['trip_count'].sum(), 'source': 'TLC_OD_FALLBACK'})).reset_index() if not road_od.empty else pd.DataFrame()
    all_idx = all_period.set_index(['origin_zone_id', 'destination_zone_id']) if not all_period.empty else None
    for _, area in areas.iterrows():
        az = area_zone[area['area_id']]
        for _, res in resources.iterrows():
            rz = resource_zone[res['resource_id']]
            mapping_rows.append({'area_id': area['area_id'], 'area_taxi_zone_id': az, 'resource_id': res['resource_id'], 'resource_taxi_zone_id': rz, 'mapping_method': 'nearest_borough_centroid_proxy', 'source': 'TAXI_ZONE_PROXY_NO_GEOMETRY'})
            distance = haversine_meters(area['lon'], area['lat'], res['lon'], res['lat'])
            fallback_base = (distance / 1000.0) / 22.0 * 60.0 * 1.35 + 4.0
            for period in ['morning_peak','midday','evening_peak','night','other']:
                source = 'ROAD_DISTANCE_FALLBACK'
                trip_count = 0
                minutes = fallback_base * {'morning_peak':1.2,'evening_peak':1.25,'night':0.85,'midday':1.0,'other':1.05}[period]
                if od is not None and (az, rz, period) in od.index:
                    rec = od.loc[(az, rz, period)]
                    if isinstance(rec, pd.DataFrame): rec = rec.iloc[0]
                    minutes = float(rec['median_road_time_minutes'])
                    trip_count = int(rec['trip_count'])
                    source = rec['source']
                elif all_idx is not None and (az, rz) in all_idx.index:
                    rec = all_idx.loc[(az, rz)]
                    if isinstance(rec, pd.DataFrame): rec = rec.iloc[0]
                    minutes = float(rec['median'])
                    trip_count = int(rec['trip_count'])
                    source = 'TLC_OD_FALLBACK'
                area_road_rows.append({'area_id': area['area_id'], 'resource_id': res['resource_id'], 'activity_type': res['category'], 'departure_period': period, 'mode': 'ridehail', 'travel_time_minutes': round(minutes, 3), 'trip_count': trip_count, 'source': source})
    mapping = pd.DataFrame(mapping_rows)
    area_road = pd.DataFrame(area_road_rows)
    mapping.to_csv(PROCESSED / 'area_resource_taxi_zone_mapping.csv', index=False)
    area_road.to_csv(PROCESSED / 'area_resource_road_travel_times.csv', index=False)
    return mapping, area_road


def rebuild_effective(area_road):
    walking = pd.read_csv(PROCESSED / 'walking_travel_times.csv')
    transit = pd.read_csv(PROCESSED / 'transit_travel_times.csv')
    return build_effective_travel_times(walking, transit, area_road, PROCESSED, get_profiles())


def write_report(road_od, stats, mapping, area_road, effective):
    src_counts = area_road['source'].value_counts().to_dict()
    total = len(area_road)
    lines = ['# Step 12 observed TLC road time report', '']
    lines += ['## Files processed', '']
    for s in stats:
        lines.append(f"- {s.get('file')}: {s.get('status')} | rows_read={s.get('rows_read',0):,} | rows_retained={s.get('rows_retained',0):,} | od_period_records={s.get('od_period_records',0):,}")
    lines += ['', f'- Total rows read: {sum(s.get("rows_read",0) for s in stats):,}', f'- Valid rows retained: {sum(s.get("rows_retained",0) for s in stats):,}', f'- OD-period records generated: {len(road_od):,}', '']
    for src in ['TLC_HVFHV_OBSERVED','TLC_YELLOW_OBSERVED','TLC_GREEN_OBSERVED','TLC_OD_FALLBACK','ROAD_DISTANCE_FALLBACK']:
        n = int(src_counts.get(src, 0))
        share = n / total if total else 0
        lines.append(f'- {src}: {n:,} area-resource-period records ({share:.2%})')
    lines += [
        '', '## Taxi zone geometry', '',
        '- Taxi-zone geometry was not used because geospatial readers are unavailable in `nyc_tlc`.',
        '- Mapping method used: nearest borough-centroid proxy from `taxi_zone_lookup.csv` borough labels.',
        '- This proxy enables partial use of observed TLC OD medians but is less accurate than polygon assignment.',
        '', '## Effective travel times', '', f'- effective_travel_times rows: {len(effective):,}',
        '', '## Limitations', '',
        '- Observed TLC medians are computed from a runtime-safe HVFHV January sample: first 100,000 rows of the first Parquet row group, not all available months.',
        '- Observed TLC medians are joined through proxy taxi-zone assignments, not polygon containment.',
        '- GTFS transit remains nearest-stop approximation.',
        '- Road distance fallback remains for OD-periods not covered by observed TLC medians.',
    ]
    (OUTPUTS / 'step12_observed_tlc_road_time_report.md').write_text('\n'.join(lines) + '\n')


def main():
    inspect_schemas()
    road_od, stats = build_road_od()
    mapping, area_road = build_mapping_and_area_road(road_od)
    effective, compat = rebuild_effective(area_road)
    write_report(road_od, stats, mapping, area_road, effective)
    print('road_od', len(road_od))
    print('area_road', len(area_road), area_road['source'].value_counts().to_dict())
    print('effective', len(effective), effective['selected_mode'].value_counts().to_dict())

if __name__ == '__main__':
    main()
