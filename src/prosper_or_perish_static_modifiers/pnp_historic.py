"""Build location-level wheat yield labels from historic BAHS observations.

Pipeline intent:
1. Take BAHS medieval wheat yield-per-seed observations (EU5 ``wheat`` family).
2. Convert to kg/ha using Pretty (1990) Winchester seed-rate calibration
   (515 kg/ha gross ÷ 4.0 YPS ≈ 128.75 kg seed/ha).
3. Join each observation to arable-ish locations in its mapped EU5 province.
4. Train attribute → yield on that set; predict the rest of the world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

# Pretty 1990 Table 2: gross wheat 515 kg/ha at mean YPS 4.0 → implied seed rate.
PRETTY_WHEAT_GROSS_KG_HA = 515.0
PRETTY_WHEAT_YPS = 4.0
WHEAT_SEED_RATE_KG_HA = PRETTY_WHEAT_GROSS_KG_HA / PRETTY_WHEAT_YPS

ARABLE_VEGETATION = ("farmland", "grasslands", "woods", "forest", "sparse")


@dataclass(frozen=True)
class HistoricPaths:
    bahs_observations: Path
    pretty_benchmarks: Path


def default_historic_paths(repo: Path | None = None) -> HistoricPaths:
    root = repo or Path(__file__).resolve().parents[2]
    base = (
        root
        / "../ProsperOrPerishConstructor/artifacts/data/population_capacity/historical_yields"
    ).resolve()
    return HistoricPaths(
        bahs_observations=base / "bahs_medieval_yield_observations.parquet",
        pretty_benchmarks=base / "pretty_1990_absolute_yield_benchmarks.parquet",
    )


def load_bahs_wheat_kg_ha(
    observations_path: Path,
    *,
    year_min: int = 1211,
    year_max: int = 1450,
    seed_rate_kg_ha: float = WHEAT_SEED_RATE_KG_HA,
) -> pl.DataFrame:
    """BAHS wheat observations converted to gross kg/ha."""

    if not observations_path.is_file():
        raise FileNotFoundError(f"missing BAHS observations: {observations_path}")
    obs = pl.read_parquet(observations_path)
    wheat = obs.filter(
        (pl.col("model_crop") == "wheat")
        & (~pl.col("is_mixture").fill_null(False))
        & (~pl.col("is_derived_aggregate").fill_null(False))
        & pl.col("gross_yield_per_seed_ratio").is_not_null()
        & (pl.col("harvest_year") >= year_min)
        & (pl.col("harvest_year") <= year_max)
        & pl.col("eu5_province").is_not_null()
    )
    return wheat.with_columns(
        (pl.col("gross_yield_per_seed_ratio") * seed_rate_kg_ha).alias("yield_kg_ha"),
        pl.lit(seed_rate_kg_ha).alias("seed_rate_kg_ha"),
        pl.lit("bahs_yps_to_kg_ha").alias("yield_unit_method"),
    )


def expand_historic_yields_to_locations(
    bahs_kg_ha: pl.DataFrame,
    geometry: pl.DataFrame,
    *,
    arable_only: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Attach each historic observation to locations in its EU5 province.

    Each observation keeps its own yield (full BAHS variance). Weight is
    ``1 / n_locations_in_province`` so one harvest contributes total mass 1.
    """

    loc_cols = [LOCATION_TAG, "province", "vegetation", "climate"]
    locs = geometry.select([c for c in loc_cols if c in geometry.columns])
    if arable_only and "vegetation" in locs.columns:
        locs = locs.filter(pl.col("vegetation").is_in(list(ARABLE_VEGETATION)))

    province_n = locs.group_by("province").len().rename({"len": "n_locs_in_province"})
    locs = locs.join(province_n, on="province", how="left")

    expanded = (
        bahs_kg_ha.select(
            [
                "eu5_province",
                "eu5_location_tag",
                "manor_key",
                "harvest_year",
                "gross_yield_per_seed_ratio",
                "yield_kg_ha",
                "seed_rate_kg_ha",
                "yield_unit_method",
                "county_manor_code",
            ]
        )
        .join(locs, left_on="eu5_province", right_on="province", how="inner")
        .with_columns(
            # Prefer exact location match when present; still keep province peers
            # but up-weight the exact tag.
            pl.when(pl.col("eu5_location_tag") == pl.col(LOCATION_TAG))
            .then(pl.lit(4.0))
            .otherwise(pl.lit(1.0))
            .alias("location_match_boost"),
            (pl.lit(1.0) / pl.col("n_locs_in_province").cast(pl.Float64)).alias(
                "province_share"
            ),
        )
        .with_columns(
            (pl.col("province_share") * pl.col("location_match_boost")).alias(
                "sample_weight"
            ),
            pl.lit("bahs_historic").alias("label_source"),
            pl.lit(True).alias("label_hard"),
        )
    )

    # Collapse to one row per (location, harvest observation identity) already unique;
    # for modeling we aggregate to location-level mean/std with total weight.
    by_loc = (
        expanded.group_by(LOCATION_TAG)
        .agg(
            pl.col("yield_kg_ha").mean().alias("historic_yield_mean_kg_ha"),
            pl.col("yield_kg_ha").median().alias("historic_yield_median_kg_ha"),
            pl.col("yield_kg_ha").std().alias("historic_yield_std_kg_ha"),
            pl.col("yield_kg_ha").count().alias("historic_n_obs"),
            pl.col("sample_weight").sum().alias("historic_weight"),
            pl.col("manor_key").n_unique().alias("historic_n_manors"),
            pl.col("eu5_province").first().alias("historic_province"),
        )
        .with_columns(
            pl.col("historic_yield_median_kg_ha").alias("label_yield_kg_ha"),
            pl.col("historic_weight").alias("sample_weight_yield"),
            pl.lit(True).alias("label_hard"),
            pl.lit("bahs_historic").alias("label_source"),
        )
    )

    meta = {
        "n_bahs_observations": int(bahs_kg_ha.height),
        "n_expanded_rows": int(expanded.height),
        "n_labeled_locations": int(by_loc.height),
        "yield_kg_ha_mean": float(bahs_kg_ha["yield_kg_ha"].mean()),
        "yield_kg_ha_median": float(bahs_kg_ha["yield_kg_ha"].median()),
        "yield_kg_ha_std": float(bahs_kg_ha["yield_kg_ha"].std()),
        "yield_kg_ha_p10": float(bahs_kg_ha["yield_kg_ha"].quantile(0.10)),
        "yield_kg_ha_p90": float(bahs_kg_ha["yield_kg_ha"].quantile(0.90)),
        "seed_rate_kg_ha": float(WHEAT_SEED_RATE_KG_HA),
        "pretty_wheat_gross_kg_ha": PRETTY_WHEAT_GROSS_KG_HA,
        "n_provinces": int(bahs_kg_ha["eu5_province"].n_unique()),
    }
    return by_loc, meta


def hostile_zero_locations(geometry: pl.DataFrame) -> pl.DataFrame:
    """Locations where wheat historically/physically cannot grow — y=0 anchors."""

    g = geometry
    if "has_river" not in g.columns:
        g = g.with_columns(pl.lit(False).alias("has_river"))
    arctic = pl.col("climate") == "arctic"
    tropical_jungle = (pl.col("climate") == "tropical") & (
        pl.col("vegetation").is_in(["jungle", "woods"])
    )
    desert_no_river = (pl.col("vegetation") == "desert") & (
        ~pl.col("has_river").cast(pl.Boolean).fill_null(False)
    )
    mask = arctic | tropical_jungle | desert_no_river
    return (
        g.filter(mask)
        .select(LOCATION_TAG)
        .unique()
        .with_columns(
            pl.lit(0.0).alias("label_yield_kg_ha"),
            pl.lit(1.5).alias("sample_weight_yield"),
            pl.lit(True).alias("label_hard"),
            pl.lit("hostile_zero").alias("label_source"),
            pl.lit(0.0).alias("historic_yield_median_kg_ha"),
            pl.lit(0).alias("historic_n_obs"),
            pl.lit(1.5).alias("historic_weight"),
        )
    )


def build_historic_location_labels(
    *,
    geometry_path: Path,
    observations_path: Path | None = None,
    year_min: int = 1211,
    year_max: int = 1450,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = default_historic_paths()
    obs_path = observations_path or paths.bahs_observations
    geometry = pl.read_parquet(geometry_path)
    bahs = load_bahs_wheat_kg_ha(obs_path, year_min=year_min, year_max=year_max)
    historic_locs, hist_meta = expand_historic_yields_to_locations(bahs, geometry)
    zeros = hostile_zero_locations(geometry)
    # Historic wins over hostile if a location somehow appears in both.
    labels = (
        pl.concat(
            [
                historic_locs.select(
                    LOCATION_TAG,
                    "label_yield_kg_ha",
                    "sample_weight_yield",
                    "label_hard",
                    "label_source",
                    "historic_yield_median_kg_ha",
                    "historic_n_obs",
                    "historic_weight",
                ),
                zeros.select(
                    LOCATION_TAG,
                    "label_yield_kg_ha",
                    "sample_weight_yield",
                    "label_hard",
                    "label_source",
                    "historic_yield_median_kg_ha",
                    "historic_n_obs",
                    "historic_weight",
                ),
            ],
            how="diagonal",
        )
        .sort("label_source")  # bahs_historic before hostile_zero for keep='first' if we unique
        .unique(subset=[LOCATION_TAG], keep="first")
    )
    meta = {
        **hist_meta,
        "n_hostile_zero_locations": int(zeros.height),
        "n_training_locations": int(labels.height),
        "year_min": year_min,
        "year_max": year_max,
    }
    return labels, meta


def write_label_audit(labels: pl.DataFrame, meta: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pnp_wheat_historic_labels.parquet"
    labels.write_parquet(path)
    (out_dir / "pnp_wheat_historic_labels_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return path
