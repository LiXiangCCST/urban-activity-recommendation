"""Shared helpers for multimodal travel-time generation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import pandas as pd


def haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371000.0
    phi1, phi2 = math.radians(float(lat1)), math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def load_standard_areas(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lon_col = "lon" if "lon" in df.columns else "centroid_x"
    lat_col = "lat" if "lat" in df.columns else "centroid_y"
    result = df[["area_id", lon_col, lat_col]].rename(columns={lon_col: "lon", lat_col: "lat"}).copy()
    result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
    result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
    return result.dropna(subset=["lon", "lat"])


def load_standard_resources(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lon_col = "lon" if "lon" in df.columns else "x"
    lat_col = "lat" if "lat" in df.columns else "y"
    cols = ["resource_id", "category", lon_col, lat_col]
    if "name" in df.columns:
        cols.append("name")
    result = df[cols].rename(columns={lon_col: "lon", lat_col: "lat", "category": "activity_type"}).copy()
    result["lon"] = pd.to_numeric(result["lon"], errors="coerce")
    result["lat"] = pd.to_numeric(result["lat"], errors="coerce")
    return result.dropna(subset=["lon", "lat"])


def period_for_profile(profile_id: str) -> str:
    return "night" if profile_id == "night_shift_worker" else "morning_peak"
