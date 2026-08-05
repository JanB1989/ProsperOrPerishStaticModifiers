from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from prosper_or_perish_static_modifiers.crops import (
    HECTARES_PER_KM2,
    WATER_MODES,
    crop_variant,
    raster_specs,
    selected_crops,
)
from prosper_or_perish_static_modifiers.fetch import resolve_raster_path
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

PRODUCTION_DENSITY_COLUMN = "production_density_kg_dm_total_km2"
YIELD_COLUMN = "yield_kg_dm_suitable_km2"


def _lon_lat_columns(sample_points: pl.DataFrame) -> tuple[pl.Expr, pl.Expr]:
    if "physical_sample_lon" in sample_points.columns:
        lon = pl.coalesce([pl.col("physical_sample_lon"), pl.col("calibrated_lon")])
        lat = pl.coalesce([pl.col("physical_sample_lat"), pl.col("calibrated_lat")])
    else:
        lon = pl.col("calibrated_lon")
        lat = pl.col("calibrated_lat")
    return lon.cast(pl.Float64), lat.cast(pl.Float64)


def sample_raster_at_footprint_points(
    sample_points: pl.DataFrame,
    source_path: Path,
    *,
    value_column: str,
    band_index: int = 1,
) -> pl.DataFrame:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform

    lon_expr, lat_expr = _lon_lat_columns(sample_points)
    points = sample_points.select(
        LOCATION_TAG,
        "sample_index",
        lon_expr.alias("lookup_lon"),
        lat_expr.alias("lookup_lat"),
    )
    longitude = points["lookup_lon"].to_list()
    latitude = points["lookup_lat"].to_list()
    with rasterio.open(source_path) as dataset:
        if band_index < 1 or band_index > dataset.count:
            raise ValueError(
                f"band {band_index} out of range for {source_path} (count={dataset.count})"
            )
        if dataset.crs is None:
            raise ValueError(f"raster has no CRS: {source_path}")
        if dataset.crs.to_epsg() != 4326:
            longitude, latitude = transform("EPSG:4326", dataset.crs, longitude, latitude)
        rows, columns = rowcol(dataset.transform, longitude, latitude, op=np.floor)
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        in_bounds = (
            (rows >= 0)
            & (rows < dataset.height)
            & (columns >= 0)
            & (columns < dataset.width)
        )
        values = np.full(points.height, np.nan, dtype=np.float64)
        band = dataset.read(band_index, masked=True).astype(np.float64).filled(np.nan)
        values[in_bounds] = band[rows[in_bounds], columns[in_bounds]]
    return points.select(LOCATION_TAG, "sample_index").with_columns(
        pl.Series(value_column, values, dtype=pl.Float64).fill_nan(None)
    )


def _sample_weight_expr(sample_points: pl.DataFrame) -> pl.Expr:
    if "sample_weight" in sample_points.columns:
        return pl.col("sample_weight").cast(pl.Float64)
    return pl.lit(1.0) / pl.len().over(LOCATION_TAG)


def _land_mask_expr(sample_points: pl.DataFrame) -> pl.Expr:
    if "sample_is_land" in sample_points.columns:
        return pl.col("sample_is_land").cast(pl.Boolean)
    return pl.lit(True)


def sample_crop_water_mode(
    *,
    sample_points: pl.DataFrame,
    cache_dir: Path,
    crop: str,
    water_mode: str,
    crop_code: str,
) -> pl.DataFrame:
    specs = {
        spec.variable: spec
        for spec in raster_specs(
            crops=selected_crops([crop]),
            water_modes=(water_mode,),
        )
        if spec.crop_code == crop_code
    }
    yxx = resolve_raster_path(cache_dir, specs["RES05-YXX"])
    ylx = resolve_raster_path(cache_dir, specs["RES05-YLX"])
    sx3 = resolve_raster_path(cache_dir, specs["RES05-SX3"])

    # YXX/YLX native units are kg DM / ha; convert to per km² for publish.
    yield_samples = sample_raster_at_footprint_points(
        sample_points, yxx, value_column="_yxx_kg_dm_ha"
    )
    density_samples = sample_raster_at_footprint_points(
        sample_points, ylx, value_column="_ylx_kg_dm_total_ha"
    )
    suit_samples = sample_raster_at_footprint_points(
        sample_points, sx3, value_column="suitable_area_share_raw"
    )

    land = _land_mask_expr(sample_points)
    frame = (
        yield_samples.join(density_samples, on=[LOCATION_TAG, "sample_index"], how="inner")
        .join(suit_samples, on=[LOCATION_TAG, "sample_index"], how="inner")
        .join(
            sample_points.select(
                LOCATION_TAG,
                "sample_index",
                _sample_weight_expr(sample_points).alias("sample_weight"),
                land.alias("sample_is_land"),
            ),
            on=[LOCATION_TAG, "sample_index"],
            how="left",
        )
        .with_columns(
            (pl.col("suitable_area_share_raw") / 10_000.0).alias("suitable_fraction"),
            (pl.col("_yxx_kg_dm_ha") * HECTARES_PER_KM2).alias(YIELD_COLUMN),
            (
                pl.col("_ylx_kg_dm_total_ha") * HECTARES_PER_KM2
            ).alias(PRODUCTION_DENSITY_COLUMN),
            pl.lit(crop).alias("crop"),
            pl.lit(water_mode).alias("water_mode"),
            pl.lit(crop_code).alias("crop_code"),
            pl.lit(crop_variant(crop, crop_code)).alias("crop_variant"),
        )
        .drop("_yxx_kg_dm_ha", "_ylx_kg_dm_total_ha")
        .with_columns(
            (
                pl.col(YIELD_COLUMN) * pl.col("sample_is_land").cast(pl.Float64)
            ).alias(YIELD_COLUMN),
            (
                pl.col(PRODUCTION_DENSITY_COLUMN)
                * pl.col("sample_is_land").cast(pl.Float64)
            ).alias(PRODUCTION_DENSITY_COLUMN),
            (
                pl.col("suitable_fraction") * pl.col("sample_is_land").cast(pl.Float64)
            ).alias("suitable_fraction"),
        )
    )
    return frame


def combine_taro_variants(frames: list[pl.DataFrame]) -> pl.DataFrame:
    """Per sample, keep the TAROD/TAROW variant with highest finite YLX density."""

    variants = pl.concat(frames, how="vertical_relaxed").with_columns(
        (
            pl.col("yield_kg_dm_suitable_km2").is_finite()
            & pl.col("production_density_kg_dm_total_km2").is_finite()
            & pl.col("suitable_fraction").is_finite()
        ).alias("_finite")
    )
    return (
        variants.sort(
            [
                LOCATION_TAG,
                "sample_index",
                "_finite",
                "production_density_kg_dm_total_km2",
                "yield_kg_dm_suitable_km2",
                "crop_variant",
            ],
            descending=[False, False, True, True, True, False],
            nulls_last=True,
        )
        .unique(subset=[LOCATION_TAG, "sample_index"], keep="first", maintain_order=True)
        .drop("_finite")
    )


def aggregate_samples_to_locations(sample_frame: pl.DataFrame) -> pl.DataFrame:
    finite = (
        pl.col("yield_kg_dm_suitable_km2").is_finite()
        & pl.col("production_density_kg_dm_total_km2").is_finite()
        & pl.col("suitable_fraction").is_finite()
    )
    return (
        sample_frame.group_by(LOCATION_TAG)
        .agg(
            (
                (pl.col("yield_kg_dm_suitable_km2") * pl.col("sample_weight")).filter(finite).sum()
                / pl.col("sample_weight").filter(finite).sum()
            ).alias("yield_kg_dm_suitable_km2"),
            (
                (pl.col("production_density_kg_dm_total_km2") * pl.col("sample_weight"))
                .filter(finite)
                .sum()
                / pl.col("sample_weight").filter(finite).sum()
            ).alias("production_density_kg_dm_total_km2"),
            (
                (pl.col("suitable_fraction") * pl.col("sample_weight")).filter(finite).sum()
                / pl.col("sample_weight").filter(finite).sum()
            ).alias("suitable_fraction"),
            pl.col("crop").first(),
            pl.col("water_mode").first(),
            pl.col("sample_weight").filter(finite).sum().alias("sample_coverage"),
        )
        .with_columns(
            pl.col("yield_kg_dm_suitable_km2").fill_nan(None),
            pl.col("production_density_kg_dm_total_km2").fill_nan(None),
            pl.col("suitable_fraction").fill_nan(None),
        )
    )


def build_samples(
    *,
    sample_points_path: Path,
    cache_dir: Path,
    output_dir: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
) -> list[Path]:
    sample_points = pl.read_parquet(sample_points_path)
    if "sample_index" not in sample_points.columns:
        sample_points = sample_points.with_row_index("sample_index")
    crop_defs = selected_crops(crops)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for crop in crop_defs:
        for water_mode in water_modes:
            variant_frames: list[pl.DataFrame] = []
            for crop_code in crop.gaez_codes:
                frame = sample_crop_water_mode(
                    sample_points=sample_points,
                    cache_dir=cache_dir,
                    crop=crop.crop,
                    water_mode=water_mode,
                    crop_code=crop_code,
                )
                variant_frames.append(frame)
            if crop.crop == "taro":
                samples = combine_taro_variants(variant_frames)
            else:
                samples = variant_frames[0]
            path = output_dir / f"{crop.crop}_{water_mode}.parquet"
            samples.write_parquet(path)
            written.append(path)
            locations = aggregate_samples_to_locations(samples)
            loc_path = output_dir / f"{crop.crop}_{water_mode}_locations.parquet"
            locations.write_parquet(loc_path)
            written.append(loc_path)
    return written


def aggregate_calendar_from_samples(
    samples_dir: Path,
    *,
    crops: list[str] | None = None,
    water_modes: tuple[str, ...] | list[str] = WATER_MODES,
) -> pl.DataFrame:
    """Weighted location means for crop calendar fields from constructor sample parquets."""

    crop_names = {crop.crop for crop in selected_crops(crops)}
    modes = set(water_modes)
    pieces: list[pl.DataFrame] = []
    for path in sorted(samples_dir.glob("*.parquet")):
        if path.name.endswith("_locations.parquet"):
            continue
        crop, water_mode = path.stem.rsplit("_", 1)
        if crop not in crop_names or water_mode not in modes:
            continue
        schema = pl.scan_parquet(path).collect_schema().names()
        if "gaez_crop_cycle_start_doy" not in schema:
            continue
        start = pl.col("gaez_crop_cycle_start_doy")
        length = pl.col("gaez_crop_cycle_length_days")
        valid = (
            start.is_finite()
            & length.is_finite()
            & (start > 0)
            & (length > 0)
            & pl.col("sample_is_land").cast(pl.Boolean, strict=False).fill_null(True)
        )
        weight = (
            pl.col("sample_weight").cast(pl.Float64)
            if "sample_weight" in schema
            else pl.lit(1.0)
        )
        agg = (
            pl.scan_parquet(path)
            .group_by(LOCATION_TAG)
            .agg(
                ((start * weight).filter(valid).sum() / weight.filter(valid).sum()).alias(
                    "crop_cycle_start_doy"
                ),
                ((length * weight).filter(valid).sum() / weight.filter(valid).sum()).alias(
                    "crop_cycle_length_days"
                ),
            )
            .with_columns(
                pl.lit(crop).alias("crop"),
                pl.lit(water_mode).alias("water_mode"),
                pl.col("crop_cycle_start_doy").fill_nan(None),
                pl.col("crop_cycle_length_days").fill_nan(None),
            )
            .collect()
        )
        pieces.append(agg)
    if not pieces:
        return pl.DataFrame(
            schema={
                LOCATION_TAG: pl.Utf8,
                "crop": pl.Utf8,
                "water_mode": pl.Utf8,
                "crop_cycle_start_doy": pl.Float64,
                "crop_cycle_length_days": pl.Float64,
            }
        )
    return pl.concat(pieces, how="vertical_relaxed")


def _aggregate_six_to_locations(sample_frame: pl.DataFrame) -> pl.DataFrame:
    valid = (
        pl.col("suitability_index").is_finite()
        & (pl.col("suitability_index") > 0)
        & pl.col("sample_is_land").cast(pl.Boolean, strict=False).fill_null(True)
    )
    return (
        sample_frame.group_by(LOCATION_TAG)
        .agg(
            (
                (pl.col("suitability_index") * pl.col("sample_weight")).filter(valid).sum()
                / pl.col("sample_weight").filter(valid).sum()
            ).alias("suitability_index"),
            pl.col("crop").first(),
            pl.col("water_mode").first(),
        )
        .with_columns(pl.col("suitability_index").fill_nan(None))
    )


def build_six_location_metrics(
    *,
    sample_points_path: Path,
    cache_dir: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
) -> pl.DataFrame:
    """Fetch/sample RES05-SIX suitability-class rasters and aggregate to locations."""

    from prosper_or_perish_static_modifiers.fetch import fetch_gaez, resolve_raster_path

    sample_points = pl.read_parquet(sample_points_path)
    if "sample_index" not in sample_points.columns:
        sample_points = sample_points.with_row_index("sample_index")

    fetch_gaez(
        cache_dir,
        crops=crops,
        water_modes=tuple(water_modes),
        variables=("RES05-SIX",),
    )

    crop_defs = selected_crops(crops)
    rows: list[pl.DataFrame] = []
    land = _land_mask_expr(sample_points)
    weight = _sample_weight_expr(sample_points)
    base_pts = sample_points.select(
        LOCATION_TAG,
        "sample_index",
        weight.alias("sample_weight"),
        land.alias("sample_is_land"),
    )

    for crop in crop_defs:
        for water_mode in water_modes:
            variant_frames: list[pl.DataFrame] = []
            for crop_code in crop.gaez_codes:
                specs = {
                    spec.crop_code: spec
                    for spec in raster_specs(
                        crops=selected_crops([crop.crop]),
                        water_modes=(water_mode,),
                        variables=("RES05-SIX",),
                    )
                }
                path = resolve_raster_path(cache_dir, specs[crop_code])
                sampled = sample_raster_at_footprint_points(
                    sample_points, path, value_column="suitability_index"
                ).join(base_pts, on=[LOCATION_TAG, "sample_index"], how="left").with_columns(
                    pl.lit(crop.crop).alias("crop"),
                    pl.lit(water_mode).alias("water_mode"),
                    pl.lit(crop_variant(crop.crop, crop_code)).alias("crop_variant"),
                )
                variant_frames.append(sampled)
            if crop.crop == "taro":
                # Lower class number is better; keep the best finite class per sample.
                combined = (
                    pl.concat(variant_frames, how="vertical_relaxed")
                    .with_columns(
                        (
                            pl.col("suitability_index").is_finite()
                            & (pl.col("suitability_index") > 0)
                        ).alias("_finite")
                    )
                    .sort(
                        [LOCATION_TAG, "sample_index", "_finite", "suitability_index"],
                        descending=[False, False, True, False],
                        nulls_last=True,
                    )
                    .unique(
                        subset=[LOCATION_TAG, "sample_index"],
                        keep="first",
                        maintain_order=True,
                    )
                    .drop("_finite")
                )
            else:
                combined = variant_frames[0]
            rows.append(_aggregate_six_to_locations(combined))
    return pl.concat(rows, how="vertical_relaxed")

