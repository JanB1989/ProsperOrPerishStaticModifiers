from __future__ import annotations

from pathlib import Path

import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.pnp_model import train_and_predict_pnp_wheat


def build_pnp_wide(
    *,
    candidates_path: Path,
    pyaez_yields_path: Path,
    gaez_wide_path: Path,
    geometry_path: Path,
    output_path: Path,
    model_dir: Path,
    require_validation: bool = True,
) -> Path:
    """Train the 1337 wheat model and emit one wide row per location."""

    wide, report, _card = train_and_predict_pnp_wheat(
        candidates_path=candidates_path,
        pyaez_yields_path=pyaez_yields_path,
        gaez_wide_path=gaez_wide_path,
        model_dir=model_dir,
        geometry_path=geometry_path,
    )
    if require_validation and not report.passed:
        failed = [k for k, v in report.checks.items() if not v["passed"]]
        raise RuntimeError(
            "P&P wheat validation gate failed: "
            + ", ".join(failed)
            + f" details={report.checks}"
        )

    geometry = pl.read_parquet(geometry_path)
    if LOCATION_TAG not in geometry.columns:
        raise ValueError(f"missing {LOCATION_TAG} in {geometry_path}")
    base = geometry.select(LOCATION_TAG).unique(subset=[LOCATION_TAG], keep="first")
    out = (
        base.join(wide, on=LOCATION_TAG, how="left")
        .with_columns(
            pl.col("pnp_wheat_production_density").fill_null(0.0),
            pl.col("pnp_wheat_yield").fill_null(0.0),
            pl.col("pnp_wheat_suitable_fraction").fill_null(0.0),
            pl.col("pnp_wheat_suitability_class").fill_null(9.0),
        )
        .sort(LOCATION_TAG)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    return output_path
