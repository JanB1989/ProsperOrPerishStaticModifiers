from __future__ import annotations

from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.external_layers import (
    PILOT_LAYERS,
    resolve_layer_raster,
)
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.sample import (
    _land_mask_expr,
    _sample_weight_expr,
    sample_raster_at_footprint_points,
)


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


def build_external_wide(
    *,
    sample_points_path: Path,
    cache_dir: Path,
    geometry_path: Path,
    output_path: Path,
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
        raster_path = resolve_layer_raster(cache_dir, layer)
        band = layer.raster_band or 1
        sampled = sample_raster_at_footprint_points(
            sample_points,
            raster_path,
            value_column=layer.layer_id,
            band_index=band,
        ).join(base_pts, on=[LOCATION_TAG, "sample_index"], how="left")
        loc = _aggregate_layer_to_locations(sampled, value_column=layer.layer_id)
        out = out.join(loc, on=LOCATION_TAG, how="left")

    out = out.sort(LOCATION_TAG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return output_path
