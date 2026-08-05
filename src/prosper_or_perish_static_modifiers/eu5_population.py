from __future__ import annotations

from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

# EU5 stores pop size in thousands of people (1 game unit = 1000 people).
PEOPLE_PER_GAME_POPULATION_UNIT = 1_000.0

DEFAULT_EU5_VANILLA_RELATIVE = (
    "../ProsperOrPerishConstructor/artifacts/data/food_building_startup/"
    "derived_food_balance_by_location.parquet"
)


def _area_km2_expr(frame: pl.DataFrame) -> pl.Expr:
    if "area_km2" in frame.columns:
        return pl.col("area_km2").cast(pl.Float64)
    if "area_jacobian_km2" in frame.columns:
        return pl.col("area_jacobian_km2").cast(pl.Float64)
    raise ValueError(
        "location area parquet needs area_km2 or area_jacobian_km2 "
        f"(columns={frame.columns[:20]})"
    )


def _location_key_expr(frame: pl.DataFrame) -> pl.Expr:
    if LOCATION_TAG in frame.columns:
        return pl.col(LOCATION_TAG).cast(pl.String)
    if "slug" in frame.columns:
        return pl.col("slug").cast(pl.String)
    raise ValueError(
        f"EU5 Vanilla source needs {LOCATION_TAG} or slug "
        f"(columns={frame.columns[:20]})"
    )


def load_constructor_eu5_vanilla_frame(path: Path) -> pl.DataFrame:
    """Load Constructor game-start location demographics (pop + development)."""

    frame = pl.read_parquet(path) if path.suffix.lower() == ".parquet" else pl.read_csv(path)
    location = _location_key_expr(frame).alias(LOCATION_TAG)

    if "eu5_start_population" in frame.columns:
        population = pl.col("eu5_start_population").cast(pl.Float64)
    elif "total_population" in frame.columns:
        population = pl.col("total_population").cast(pl.Float64) * PEOPLE_PER_GAME_POPULATION_UNIT
    else:
        raise ValueError(
            f"EU5 Vanilla source missing population column in {path} "
            "(need total_population or eu5_start_population)"
        )

    if "development" not in frame.columns:
        raise ValueError(f"EU5 Vanilla source missing development column in {path}")

    return (
        frame.select(
            location,
            population.alias("eu5_population_total"),
            pl.col("development").cast(pl.Float64).alias("eu5_development"),
        )
        .unique(subset=[LOCATION_TAG], keep="first")
        .sort(LOCATION_TAG)
    )


def build_eu5_vanilla_layers(
    *,
    constructor_locations_path: Path,
    location_area_path: Path,
) -> pl.DataFrame:
    """Return location_tag + pop total/density + development from Constructor data."""

    vanilla = load_constructor_eu5_vanilla_frame(constructor_locations_path)
    area = pl.read_parquet(location_area_path)
    if LOCATION_TAG not in area.columns:
        raise ValueError(f"missing {LOCATION_TAG} in {location_area_path}")
    area = area.select(LOCATION_TAG, _area_km2_expr(area).alias("_area_km2"))
    return (
        vanilla.join(area, on=LOCATION_TAG, how="left")
        .with_columns(
            pl.when(pl.col("_area_km2").is_not_null() & (pl.col("_area_km2") > 0))
            .then(pl.col("eu5_population_total") / pl.col("_area_km2"))
            .otherwise(None)
            .alias("eu5_population_density"),
        )
        .drop("_area_km2")
    )


# Back-compat alias used by older call sites / tests.
def build_eu5_population_layers(
    *,
    constructor_locations_path: Path,
    location_area_path: Path,
) -> pl.DataFrame:
    return build_eu5_vanilla_layers(
        constructor_locations_path=constructor_locations_path,
        location_area_path=location_area_path,
    )
