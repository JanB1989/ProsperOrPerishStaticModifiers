from __future__ import annotations

from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.crops import (
    HECTARES_PER_KM2,
    WATER_MODES,
    iter_metrics,
    metric_column,
    selected_crops,
)
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.sample import (
    aggregate_calendar_from_samples,
    aggregate_samples_to_locations,
    build_six_location_metrics,
)


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
    present = set(long_frame.columns)
    for crop in crop_defs:
        for water_mode in water_modes:
            subset = long_frame.filter(
                (pl.col("crop") == crop.crop) & (pl.col("water_mode") == water_mode)
            )
            exprs: list[pl.Expr] = []
            for metric in iter_metrics():
                suffix = metric["suffix"]
                alias = metric_column(crop.crop, water_mode, suffix)
                if suffix in present:
                    exprs.append(pl.col(suffix).alias(alias))
                else:
                    exprs.append(pl.lit(None).cast(pl.Float64).alias(alias))
            renamed = subset.select(LOCATION_TAG, *exprs)
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


def _enrich_long_metrics(
    labels: pl.DataFrame,
    *,
    crop_samples_dir: Path | None,
    sample_points_path: Path | None,
    gaez_cache_dir: Path | None,
    crops: list[str] | None,
    water_modes: tuple[str, ...] | list[str],
) -> pl.DataFrame:
    """Attach calendar (from samples) and SIX (sampled from GAEZ) onto labels."""

    frame = labels
    if crop_samples_dir is not None and crop_samples_dir.is_dir():
        calendar = aggregate_calendar_from_samples(
            crop_samples_dir,
            crops=crops,
            water_modes=water_modes,
        )
        if calendar.height:
            frame = frame.join(
                calendar,
                on=[LOCATION_TAG, "crop", "water_mode"],
                how="left",
            )

    if "suitability_index" in frame.columns:
        need_six = frame["suitability_index"].null_count() == frame.height
    else:
        need_six = True
    if (
        need_six
        and sample_points_path is not None
        and sample_points_path.is_file()
        and gaez_cache_dir is not None
    ):
        six = build_six_location_metrics(
            sample_points_path=sample_points_path,
            cache_dir=gaez_cache_dir,
            crops=crops,
            water_modes=water_modes,
        )
        keep = [LOCATION_TAG, "crop", "water_mode", "suitability_index"]
        frame = frame.drop("suitability_index", strict=False).join(
            six.select(keep),
            on=[LOCATION_TAG, "crop", "water_mode"],
            how="left",
        )
    return frame


def build_wide_from_labels(
    *,
    geometry_path: Path,
    labels_path: Path,
    output_path: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
    engine: str = "gaez_v5",
    crop_samples_dir: Path | None = None,
    sample_points_path: Path | None = None,
    gaez_cache_dir: Path | None = None,
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

    required = [
        LOCATION_TAG,
        "crop",
        "water_mode",
        "yield_kg_dm_ha",
        "suitable_fraction",
    ]
    density_src = None
    if "production_density_kg_dm_total_km2" in labels.columns:
        density_src = "production_density_kg_dm_total_km2"
    elif "production_density_kg_dm_total_ha" in labels.columns:
        density_src = "production_density_kg_dm_total_ha"
    else:
        raise ValueError(
            "labels parquet missing production density column "
            "(production_density_kg_dm_total_km2 or production_density_kg_dm_total_ha)"
        )
    missing = [col for col in required if col not in labels.columns]
    if missing:
        raise ValueError(f"labels parquet missing columns: {missing}")

    optional = [
        "net_irrigation_requirement_mm",
        "crop_cycle_start_doy",
        "crop_cycle_length_days",
        "suitability_index",
    ]
    keep = required + [density_src] + [col for col in optional if col in labels.columns]
    labels = labels.select(keep)
    if density_src == "production_density_kg_dm_total_ha":
        labels = labels.with_columns(
            (pl.col("production_density_kg_dm_total_ha") * HECTARES_PER_KM2).alias(
                "production_density_kg_dm_total_km2"
            )
        ).drop("production_density_kg_dm_total_ha")

    labels = _enrich_long_metrics(
        labels,
        crop_samples_dir=crop_samples_dir,
        sample_points_path=sample_points_path,
        gaez_cache_dir=gaez_cache_dir,
        crops=crops,
        water_modes=water_modes,
    )

    wide_metrics = pivot_location_metrics(
        labels,
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
            for metric in iter_metrics():
                names.append(metric_column(crop.crop, water_mode, metric["suffix"]))
    return names