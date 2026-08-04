from __future__ import annotations

from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.crops import (
    METRICS,
    WATER_MODES,
    metric_column,
    selected_crops,
)
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.sample import aggregate_samples_to_locations


IDENTITY_COLUMNS = (
    LOCATION_TAG,
    "map_color_int",
    "map_color_rgb",
    "centroid_x",
    "centroid_y",
    "approx_lon",
    "approx_lat",
    "pixel_count",
    "geometry_status",
)


def pivot_location_metrics(
    long_frame: pl.DataFrame,
    *,
    crops: list[str] | None = None,
    water_modes: tuple[str, ...] | list[str] = WATER_MODES,
) -> pl.DataFrame:
    """Pivot long (location, crop, water_mode, metrics) into one wide row per location."""

    crop_defs = selected_crops(crops)
    pieces: list[pl.DataFrame] = []
    keys = long_frame.select(LOCATION_TAG).unique().sort(LOCATION_TAG)
    for crop in crop_defs:
        for water_mode in water_modes:
            subset = long_frame.filter(
                (pl.col("crop") == crop.crop) & (pl.col("water_mode") == water_mode)
            )
            renamed = subset.select(
                LOCATION_TAG,
                *[
                    pl.col(suffix).alias(metric_column(crop.crop, water_mode, suffix))
                    for _metric_id, suffix, _label, _unit in METRICS
                ],
            )
            pieces.append(renamed)
    out = keys
    for piece in pieces:
        out = out.join(piece, on=LOCATION_TAG, how="left")
    return out.sort(LOCATION_TAG)


def build_wide_dataframe(
    *,
    geometry_path: Path,
    samples_dir: Path,
    output_path: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
) -> Path:
    geometry = pl.read_parquet(geometry_path)
    identity_cols = [col for col in IDENTITY_COLUMNS if col in geometry.columns]
    base = geometry.select(identity_cols).unique(subset=[LOCATION_TAG], keep="first")

    crop_defs = selected_crops(crops)
    long_rows: list[pl.DataFrame] = []
    for crop in crop_defs:
        for water_mode in water_modes:
            loc_path = samples_dir / f"{crop.crop}_{water_mode}_locations.parquet"
            sample_path = samples_dir / f"{crop.crop}_{water_mode}.parquet"
            if loc_path.is_file():
                frame = pl.read_parquet(loc_path)
            elif sample_path.is_file():
                frame = aggregate_samples_to_locations(pl.read_parquet(sample_path))
            else:
                raise FileNotFoundError(
                    f"missing samples for {crop.crop}/{water_mode}; run build-samples"
                )
            long_rows.append(frame)

    long_frame = pl.concat(long_rows, how="vertical_relaxed")
    wide_metrics = pivot_location_metrics(
        long_frame, crops=crops, water_modes=water_modes
    )
    wide = base.join(wide_metrics, on=LOCATION_TAG, how="left").sort(LOCATION_TAG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(output_path)
    return output_path


def build_wide_from_labels(
    *,
    geometry_path: Path,
    labels_path: Path,
    output_path: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
    engine: str = "gaez_v5",
) -> Path:
    """Pivot a long crop_mode_labels parquet into the publish wide dataframe."""

    geometry = pl.read_parquet(geometry_path)
    identity_cols = [col for col in IDENTITY_COLUMNS if col in geometry.columns]
    base = geometry.select(identity_cols).unique(subset=[LOCATION_TAG], keep="first")

    labels = pl.read_parquet(labels_path)
    if "engine" in labels.columns:
        labels = labels.filter(pl.col("engine") == engine)
    crop_defs = selected_crops(crops)
    crop_names = [crop.crop for crop in crop_defs]
    labels = labels.filter(
        pl.col("crop").is_in(crop_names) & pl.col("water_mode").is_in(list(water_modes))
    )
    needed = [
        LOCATION_TAG,
        "crop",
        "water_mode",
        "yield_kg_dm_ha",
        "production_density_kg_dm_total_ha",
        "suitable_fraction",
    ]
    missing = [col for col in needed if col not in labels.columns]
    if missing:
        raise ValueError(f"labels parquet missing columns: {missing}")

    wide_metrics = pivot_location_metrics(
        labels.select(needed),
        crops=crops,
        water_modes=water_modes,
    )
    wide = base.join(wide_metrics, on=LOCATION_TAG, how="left").sort(LOCATION_TAG)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(output_path)
    return output_path


def wide_metric_columns(
    crops: list[str] | None = None,
    water_modes: tuple[str, ...] | list[str] = WATER_MODES,
) -> list[str]:
    names: list[str] = []
    for crop in selected_crops(crops):
        for water_mode in water_modes:
            for _metric_id, suffix, _label, _unit in METRICS:
                names.append(metric_column(crop.crop, water_mode, suffix))
    return names
