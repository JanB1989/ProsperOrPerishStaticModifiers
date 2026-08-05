from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import rasterio

from prosper_or_perish_static_modifiers.crops import HECTARES_PER_KM2
from prosper_or_perish_static_modifiers.eu5_population import build_eu5_population_layers
from prosper_or_perish_static_modifiers.external_layers import (
    PILOT_LAYERS,
    resolve_layer_raster,
)
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.sample import (
    _land_mask_expr,
    _lon_lat_columns,
    _sample_weight_expr,
    sample_raster_at_footprint_points,
)

# Mean Earth radius conversion: degrees of latitude ≈ 111.32 km.
_KM_PER_DEG = 111.32


def _aggregate_layer_to_locations(
    sample_frame: pl.DataFrame,
    *,
    value_column: str,
) -> pl.DataFrame:
    valid = (
        pl.col(value_column).is_finite()
        & (pl.col(value_column) > 0)
        & pl.col("sample_is_land").cast(pl.Boolean, strict=False).fill_null(True)
    )
    # Europe suitability can be 0 as meaningful "unsuitable"; still treat <=0 as missing for agg.
    return (
        sample_frame.group_by(LOCATION_TAG)
        .agg(
            (
                (pl.col(value_column) * pl.col("sample_weight")).filter(valid).sum()
                / pl.col("sample_weight").filter(valid).sum()
            ).alias(value_column),
        )
        .with_columns(pl.col(value_column).fill_nan(None))
    )


def _people_per_pixel_to_density(
    sampled: pl.DataFrame,
    sample_points: pl.DataFrame,
    *,
    value_column: str,
    raster_path: Path,
) -> pl.DataFrame:
    """Convert WorldPop people/pixel counts to people/km² using cell area at sample latitude."""
    with rasterio.open(raster_path) as dataset:
        cellsize_deg = abs(float(dataset.transform.a))
    lon_expr, lat_expr = _lon_lat_columns(sample_points)
    with_lat = sampled.join(
        sample_points.select(
            LOCATION_TAG,
            "sample_index",
            lat_expr.alias("_sample_lat"),
        ),
        on=[LOCATION_TAG, "sample_index"],
        how="left",
    )
    lat = with_lat["_sample_lat"].to_numpy()
    values = with_lat[value_column].to_numpy()
    # Spherical rectangle: height constant, width shrinks with cos(latitude).
    height_km = cellsize_deg * _KM_PER_DEG
    width_km = cellsize_deg * _KM_PER_DEG * np.cos(np.deg2rad(lat))
    area_km2 = height_km * width_km
    density = np.full(values.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(area_km2) & (area_km2 > 0)
    density[ok] = values[ok] / area_km2[ok]
    return sampled.with_columns(
        pl.Series(value_column, density, dtype=pl.Float64).fill_nan(None)
    )


def build_external_wide(
    *,
    sample_points_path: Path,
    cache_dir: Path,
    geometry_path: Path,
    output_path: Path,
    start_pops_path: Path | None = None,
    location_area_path: Path | None = None,
) -> Path:
    """Sample pilot external rasters and emit one wide row per location."""

    sample_points = pl.read_parquet(sample_points_path)
    if "sample_index" not in sample_points.columns:
        sample_points = sample_points.with_row_index("sample_index")

    geometry = pl.read_parquet(geometry_path)
    base = geometry.select(LOCATION_TAG).unique(subset=[LOCATION_TAG], keep="first")

    land = _land_mask_expr(sample_points)
    weight = _sample_weight_expr(sample_points)
    base_pts = sample_points.select(
        LOCATION_TAG,
        "sample_index",
        weight.alias("sample_weight"),
        land.alias("sample_is_land"),
    )

    out = base
    for layer in PILOT_LAYERS:
        if layer.source == "eu5":
            continue
        raster_path = resolve_layer_raster(cache_dir, layer)
        band = layer.raster_band or 1
        sampled = sample_raster_at_footprint_points(
            sample_points,
            raster_path,
            value_column=layer.layer_id,
            band_index=band,
        )
        if layer.count_to_density:
            sampled = _people_per_pixel_to_density(
                sampled,
                sample_points,
                value_column=layer.layer_id,
                raster_path=raster_path,
            )
        if layer.per_ha_to_per_km2:
            sampled = sampled.with_columns(
                (pl.col(layer.layer_id) * HECTARES_PER_KM2).alias(layer.layer_id)
            )
        if layer.ha_to_km2:
            sampled = sampled.with_columns(
                (pl.col(layer.layer_id) / HECTARES_PER_KM2).alias(layer.layer_id)
            )
        sampled = sampled.join(base_pts, on=[LOCATION_TAG, "sample_index"], how="left")
        loc = _aggregate_layer_to_locations(sampled, value_column=layer.layer_id)
        out = out.join(loc, on=LOCATION_TAG, how="left")

    eu5_layers = [layer for layer in PILOT_LAYERS if layer.source == "eu5"]
    if eu5_layers:
        if start_pops_path is None or location_area_path is None:
            raise ValueError(
                "EU5 population layers require start_pops_path and location_area_path"
            )
        eu5 = build_eu5_population_layers(
            start_pops_path=start_pops_path,
            location_area_path=location_area_path,
        )
        for layer in eu5_layers:
            column = layer.table_column or layer.layer_id
            if column not in eu5.columns:
                raise ValueError(f"EU5 population frame missing column {column}")
            out = out.join(
                eu5.select(LOCATION_TAG, pl.col(column).alias(layer.layer_id)),
                on=LOCATION_TAG,
                how="left",
            )

    out = out.sort(LOCATION_TAG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return output_path
