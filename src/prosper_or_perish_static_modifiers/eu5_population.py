from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

# EU5 stores pop size in thousands of people (1 game unit = 1000 people).
PEOPLE_PER_GAME_POPULATION_UNIT = 1_000.0

_LOCATION_OPEN = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*\{")
_SIZE = re.compile(r"size\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def default_start_pops_path(vanilla_root: Path) -> Path:
    return (
        vanilla_root
        / "game"
        / "main_menu"
        / "setup"
        / "start"
        / "06_pops.txt"
    )


def parse_eu5_start_population(path: Path) -> pl.DataFrame:
    """Parse vanilla 06_pops.txt into people counts per location_tag."""

    text = path.read_text(encoding="utf-8", errors="replace")
    totals: dict[str, float] = {}
    depth = 0
    in_locations = False
    current: str | None = None

    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].strip()
        if not stripped:
            continue
        opens = stripped.count("{")
        closes = stripped.count("}")

        if not in_locations:
            if re.match(r"^locations\s*=", stripped):
                in_locations = True
                depth += opens - closes
            continue

        if depth == 1:
            match = _LOCATION_OPEN.match(stripped)
            if match:
                current = match.group(1)
                totals.setdefault(current, 0.0)

        if current is not None and depth >= 2:
            for size_match in _SIZE.finditer(stripped):
                totals[current] += (
                    float(size_match.group(1)) * PEOPLE_PER_GAME_POPULATION_UNIT
                )

        depth += opens - closes
        if depth <= 0:
            break
        if depth < 2:
            current = None

    if not totals:
        raise ValueError(f"no location populations parsed from {path}")

    return (
        pl.DataFrame(
            {
                LOCATION_TAG: list(totals.keys()),
                "eu5_population_total": list(totals.values()),
            }
        )
        .with_columns(pl.col("eu5_population_total").cast(pl.Float64))
        .sort(LOCATION_TAG)
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


def build_eu5_population_layers(
    *,
    start_pops_path: Path,
    location_area_path: Path,
) -> pl.DataFrame:
    """Return location_tag + total people + people/km² for exploration mapmodes."""

    pops = parse_eu5_start_population(start_pops_path)
    area = pl.read_parquet(location_area_path)
    if LOCATION_TAG not in area.columns:
        raise ValueError(f"missing {LOCATION_TAG} in {location_area_path}")
    area = area.select(LOCATION_TAG, _area_km2_expr(area).alias("_area_km2"))
    return (
        pops.join(area, on=LOCATION_TAG, how="left")
        .with_columns(
            pl.when(pl.col("_area_km2").is_not_null() & (pl.col("_area_km2") > 0))
            .then(pl.col("eu5_population_total") / pl.col("_area_km2"))
            .otherwise(None)
            .alias("eu5_population_density")
        )
        .drop("_area_km2")
    )
