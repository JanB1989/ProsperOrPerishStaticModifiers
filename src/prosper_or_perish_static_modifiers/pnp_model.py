from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from prosper_or_perish_static_modifiers.crops import HECTARES_PER_KM2
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

# BAHS / Pretty 1990 Bishop of Winchester manors 1283–1349 (gross wheat).
BAHS_WHEAT_GROSS_KG_HA = 515.0

# Wheat agronomic optimum used for engineered temperature distance (°C).
WHEAT_OPTIMUM_TEMP_C = 12.0

NUMERIC_FEATURES: tuple[str, ...] = (
    "chelsa_annual_mean_temperature_target",
    "chelsa_annual_precipitation_target",
    "chelsa_precipitation_seasonality_target",
    "calibrated_lat",
    "calibrated_lon",
    "abs_lat",
    "aridity_proxy",
    "temp_distance_from_optimum",
    "severe_winter",
    "soil_quality_ord",
    "gaez_wheat_yield_km2",
    "gaez_wheat_suitable_fraction",
    "gaez_barley_yield_km2",
    "gaez_rye_yield_km2",
    "gaez_oats_yield_km2",
)

# One-hot / ordinal categorical expansions appended in assemble_training_frame.
CATEGORICAL_PREFIXES: tuple[str, ...] = (
    "topo_",
    "veg_",
    "clim_",
)

SOIL_QUALITY_ORD: dict[str, float] = {
    "soil_barren": 0.0,
    "soil_permafrost": 0.5,
    "soil_awful": 1.0,
    "soil_verypoor": 2.0,
    "soil_poor": 3.0,
    "soil_average": 4.0,
    "soil_good": 5.0,
    "soil_excellent": 6.0,
}

# Monotonic constraints aligned with NUMERIC_FEATURES order (+ categoricals unconstrained).
#  1 = increasing, -1 = decreasing, 0 = none
NUMERIC_MONOTONIC: tuple[int, ...] = (
    0,   # temperature (handled via temp_distance)
    1,   # precipitation
    0,   # seasonality
    0,   # lat
    0,   # lon
    0,   # abs_lat
    1,   # aridity_proxy (higher precip/(T+10) better for drylands)
    -1,  # temp distance from optimum
    -1,  # severe winter
    1,   # soil quality
    1,   # modern gaez wheat yield
    1,   # modern gaez wheat suitability
    1,   # barley analogue
    1,   # rye analogue
    1,   # oats analogue
)


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: dict[str, dict[str, object]]
    metrics: dict[str, float]
    tech_scale: float


def _encode_categories(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for col, prefix, values in (
        ("topography", "topo_", sorted(frame["topography"].drop_nulls().unique().to_list())),
        ("vegetation", "veg_", sorted(frame["vegetation"].drop_nulls().unique().to_list())),
        ("climate", "clim_", sorted(frame["climate"].drop_nulls().unique().to_list())),
    ):
        for value in values:
            safe = str(value).replace(" ", "_")
            out = out.with_columns(
                (pl.col(col).cast(pl.String) == value).cast(pl.Float64).alias(f"{prefix}{safe}")
            )
    return out


def assemble_training_frame(
    *,
    candidates_path: Path,
    pyaez_yields_path: Path,
    gaez_wide_path: Path,
) -> pl.DataFrame:
    """Join location features + GAEZ analogues + calibrated 1337 wheat targets."""

    cand_cols = [
        LOCATION_TAG,
        "chelsa_annual_mean_temperature_target",
        "chelsa_annual_precipitation_target",
        "chelsa_precipitation_seasonality_target",
        "calibrated_lat",
        "calibrated_lon",
        "topography",
        "vegetation",
        "climate",
        "climate_winter",
        "soil_quality",
    ]
    cand_schema = set(pl.scan_parquet(candidates_path).collect_schema().names())
    candidates = pl.read_parquet(candidates_path).select(
        [c for c in cand_cols if c in cand_schema]
    )

    gaez_wanted = {
        "wheat_rainfed_yield_kg_dm_suitable_km2": "gaez_wheat_yield_km2",
        "wheat_rainfed_suitable_fraction": "gaez_wheat_suitable_fraction",
        "barley_rainfed_yield_kg_dm_suitable_km2": "gaez_barley_yield_km2",
        "rye_rainfed_yield_kg_dm_suitable_km2": "gaez_rye_yield_km2",
        "oats_rainfed_yield_kg_dm_suitable_km2": "gaez_oats_yield_km2",
    }
    gaez_raw = pl.read_parquet(gaez_wide_path)
    gaez_select = [LOCATION_TAG] + [c for c in gaez_wanted if c in gaez_raw.columns]
    gaez = gaez_raw.select(gaez_select)
    rename = {src: dst for src, dst in gaez_wanted.items() if src in gaez.columns}
    gaez = gaez.rename(rename)
    for dst in gaez_wanted.values():
        if dst not in gaez.columns:
            gaez = gaez.with_columns(pl.lit(0.0).alias(dst))
        else:
            gaez = gaez.with_columns(pl.col(dst).fill_null(0.0))

    wheat = (
        pl.read_parquet(pyaez_yields_path)
        .filter(pl.col("crop") == "wheat")
        .select(
            LOCATION_TAG,
            "water_mode",
            pl.col("yield_kg_dm_ha").cast(pl.Float64),
            pl.col("suitable_fraction").cast(pl.Float64),
            pl.col("production_density_p50_kg_dm_total_ha").cast(pl.Float64),
        )
    )
    rainfed = wheat.filter(pl.col("water_mode") == "rainfed").select(
        LOCATION_TAG,
        pl.col("yield_kg_dm_ha").alias("rf_yield"),
        pl.col("suitable_fraction").alias("rf_suit"),
        pl.col("production_density_p50_kg_dm_total_ha").alias("rf_prod"),
    )
    irrigated = wheat.filter(pl.col("water_mode") == "irrigated").select(
        LOCATION_TAG,
        pl.col("yield_kg_dm_ha").alias("ir_yield"),
        pl.col("suitable_fraction").alias("ir_suit"),
        pl.col("production_density_p50_kg_dm_total_ha").alias("ir_prod"),
    )
    # Best-achievable prefers rainfed; irrigated only where rainfed is already viable
    # (historically-plausible irrigation), not in barren hyper-arid interiors.
    modes = rainfed.join(irrigated, on=LOCATION_TAG, how="full", coalesce=True).with_columns(
        pl.when(pl.col("rf_yield").fill_null(0.0) > 50.0)
        .then(pl.max_horizontal("rf_yield", "ir_yield"))
        .otherwise(pl.col("rf_yield").fill_null(0.0))
        .alias("target_yield_kg_ha"),
        pl.when(pl.col("rf_yield").fill_null(0.0) > 50.0)
        .then(pl.max_horizontal("rf_suit", "ir_suit"))
        .otherwise(pl.col("rf_suit").fill_null(0.0))
        .alias("target_suitable_fraction"),
        pl.when(pl.col("rf_yield").fill_null(0.0) > 50.0)
        .then(pl.max_horizontal("rf_prod", "ir_prod"))
        .otherwise(pl.col("rf_prod").fill_null(0.0))
        .alias("target_production_density_kg_ha"),
    )
    best = modes.select(
        LOCATION_TAG,
        "target_yield_kg_ha",
        "target_suitable_fraction",
        "target_production_density_kg_ha",
    )

    positive = best.filter(pl.col("target_yield_kg_ha") > 0)["target_yield_kg_ha"]
    if positive.len() == 0:
        raise ValueError("no positive wheat yields in pyaez_1337_yields")
    median_pos = float(positive.median())
    tech_scale = BAHS_WHEAT_GROSS_KG_HA / median_pos if median_pos > 0 else 1.0

    best = best.with_columns(
        (pl.col("target_yield_kg_ha") * tech_scale).alias("target_yield_kg_ha"),
        (pl.col("target_production_density_kg_ha") * tech_scale).alias(
            "target_production_density_kg_ha"
        ),
        pl.lit(tech_scale).alias("tech_scale"),
    )

    frame = (
        candidates.join(best, on=LOCATION_TAG, how="inner")
        .join(gaez, on=LOCATION_TAG, how="left")
        .with_columns(
            pl.col("calibrated_lat").abs().alias("abs_lat"),
            (
                pl.col("chelsa_annual_precipitation_target")
                / (pl.col("chelsa_annual_mean_temperature_target") + 10.0)
            ).alias("aridity_proxy"),
            (pl.col("chelsa_annual_mean_temperature_target") - WHEAT_OPTIMUM_TEMP_C)
            .abs()
            .alias("temp_distance_from_optimum"),
            (pl.col("climate_winter").cast(pl.String) == "severe")
            .cast(pl.Float64)
            .alias("severe_winter"),
            pl.col("soil_quality")
            .cast(pl.String)
            .replace_strict(SOIL_QUALITY_ORD, default=3.0)
            .cast(pl.Float64)
            .alias("soil_quality_ord"),
        )
    )
    frame = _encode_categories(frame)
    return frame


def feature_columns(frame: pl.DataFrame) -> list[str]:
    cols = list(NUMERIC_FEATURES)
    for col in frame.columns:
        if any(col.startswith(prefix) for prefix in CATEGORICAL_PREFIXES):
            cols.append(col)
    return cols


def _matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"training frame missing features: {missing[:8]}")
    data = frame.select(columns).fill_null(0.0).fill_nan(0.0).to_numpy()
    return np.asarray(data, dtype=np.float64)


def _monotonic_for(columns: list[str]) -> list[int]:
    mono = list(NUMERIC_MONOTONIC)
    # Pad categorical one-hots with 0.
    while len(mono) < len(columns):
        mono.append(0)
    return mono[: len(columns)]


def suitability_class_from_fraction(fraction: np.ndarray) -> np.ndarray:
    """Map suitable fraction → GAEZ-style class 1 (best) … 9 (worst)."""

    f = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    # Invert: high suitability → low class number.
    raw = 1.0 + (1.0 - f) * 8.0
    classes = np.rint(raw).astype(np.float64)
    classes[(f <= 0) | ~np.isfinite(f)] = 9.0
    return np.clip(classes, 1.0, 9.0)


def train_pnp_wheat_models(
    frame: pl.DataFrame,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, list[str], dict]:
    columns = feature_columns(frame)
    x = _matrix(frame, columns)
    y_yield = frame["target_yield_kg_ha"].fill_null(0.0).to_numpy().astype(np.float64)
    y_suit = (
        frame["target_suitable_fraction"].fill_null(0.0).to_numpy().astype(np.float64)
    )
    y_suit = np.clip(y_suit, 0.0, 1.0)
    mono = _monotonic_for(columns)

    yield_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        l2_regularization=0.1,
        random_state=1337,
        monotonic_cst=mono,
    )
    suit_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=180,
        l2_regularization=0.1,
        random_state=1337,
        monotonic_cst=mono,
    )
    yield_model.fit(x, y_yield)
    suit_model.fit(x, y_suit)

    yield_cv = float(
        np.mean(
            cross_val_score(
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=120,
                    l2_regularization=0.1,
                    random_state=1337,
                    monotonic_cst=mono,
                ),
                x,
                y_yield,
                cv=3,
                scoring="r2",
            )
        )
    )
    suit_cv = float(
        np.mean(
            cross_val_score(
                HistGradientBoostingRegressor(
                    max_depth=5,
                    learning_rate=0.08,
                    max_iter=100,
                    l2_regularization=0.1,
                    random_state=1337,
                    monotonic_cst=mono,
                ),
                x,
                y_suit,
                cv=3,
                scoring="r2",
            )
        )
    )
    tech_scale = float(frame["tech_scale"][0])
    meta = {
        "feature_columns": columns,
        "tech_scale": tech_scale,
        "bahs_wheat_gross_kg_ha": BAHS_WHEAT_GROSS_KG_HA,
        "cv_r2_yield": yield_cv,
        "cv_r2_suitable_fraction": suit_cv,
        "n_rows": int(frame.height),
    }
    return yield_model, suit_model, columns, meta


def predict_pnp_wheat(
    frame: pl.DataFrame,
    *,
    yield_model: HistGradientBoostingRegressor,
    suit_model: HistGradientBoostingRegressor,
    feature_cols: list[str],
) -> pl.DataFrame:
    x = _matrix(frame, feature_cols)
    yield_ha = np.clip(yield_model.predict(x), 0.0, None)
    suit = np.clip(suit_model.predict(x), 0.0, 1.0)
    # Soft zero-out where climate / modern GAEZ imply wheat is not viable.
    if "gaez_wheat_suitable_fraction" in frame.columns and "climate" in frame.columns:
        gaez_sf = frame["gaez_wheat_suitable_fraction"].fill_null(0.0).to_numpy()
        climate = frame["climate"].cast(pl.String).to_numpy()
        vegetation = (
            frame["vegetation"].cast(pl.String).to_numpy()
            if "vegetation" in frame.columns
            else np.array([""] * len(climate))
        )
        precip = (
            frame["chelsa_annual_precipitation_target"].fill_null(0.0).to_numpy()
            if "chelsa_annual_precipitation_target" in frame.columns
            else np.zeros(len(climate))
        )
        arctic = climate == "arctic"
        hyper_arid = (climate == "arid") & (vegetation == "desert") & (precip < 150)
        hostile = (arctic & (gaez_sf <= 0.05)) | hyper_arid
        yield_ha = np.where(hostile, 0.0, yield_ha)
        suit = np.where(hostile, 0.0, suit)

    prod_ha = yield_ha * suit
    return pl.DataFrame(
        {
            LOCATION_TAG: frame[LOCATION_TAG],
            "pnp_wheat_yield": yield_ha * HECTARES_PER_KM2,
            "pnp_wheat_suitable_fraction": suit,
            "pnp_wheat_production_density": prod_ha * HECTARES_PER_KM2,
            "pnp_wheat_suitability_class": suitability_class_from_fraction(suit),
            "pred_yield_kg_ha": yield_ha,
        }
    )


def validate_pnp_predictions(
    frame: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    tech_scale: float,
    cv_r2_yield: float,
) -> ValidationReport:
    joined = frame.select(
        LOCATION_TAG,
        "climate",
        "vegetation",
        "chelsa_annual_precipitation_target",
        "calibrated_lat",
        "calibrated_lon",
        "abs_lat",
    ).join(predictions, on=LOCATION_TAG, how="inner")

    checks: dict[str, dict[str, object]] = {}

    arctic = joined.filter(
        (pl.col("climate") == "arctic") & (pl.col("abs_lat") > 60)
    )
    arctic_mean = float(arctic["pred_yield_kg_ha"].mean() or 0.0)
    checks["A1_arctic_near_zero"] = {
        "passed": arctic_mean < 80.0,
        "arctic_mean_kg_ha": arctic_mean,
        "n": arctic.height,
    }

    arid = joined.filter(
        (pl.col("climate") == "arid")
        & (pl.col("vegetation") == "desert")
        & (pl.col("chelsa_annual_precipitation_target") < 150)
    )
    arid_mean = float(arid["pred_yield_kg_ha"].mean() or 0.0)
    checks["A2_hyper_arid_low"] = {
        "passed": arid_mean < 200.0,
        "arid_mean_kg_ha": arid_mean,
        "n": arid.height,
    }

    tropical = joined.filter(
        (pl.col("climate") == "tropical") & (pl.col("vegetation") == "jungle")
    )
    temperate = joined.filter(pl.col("climate").is_in(["oceanic", "continental", "mediterranean"]))
    trop_mean = float(tropical["pred_yield_kg_ha"].mean() or 0.0)
    temp_mean = float(temperate["pred_yield_kg_ha"].mean() or 0.0)
    checks["A3_tropical_below_temperate"] = {
        "passed": trop_mean < temp_mean * 0.85,
        "tropical_mean_kg_ha": trop_mean,
        "temperate_mean_kg_ha": temp_mean,
    }

    nw = joined.filter(
        pl.col("climate").is_in(["oceanic", "continental"])
        & (pl.col("calibrated_lat") > 45)
        & (pl.col("calibrated_lat") < 60)
        & (pl.col("calibrated_lon") > -10)
        & (pl.col("calibrated_lon") < 30)
    )
    nw_mean = float(nw["pred_yield_kg_ha"].mean() or 0.0)
    checks["A4_nw_europe_high"] = {
        "passed": nw_mean > max(arid_mean, arctic_mean) * 1.5 and nw_mean > 150,
        "nw_europe_mean_kg_ha": nw_mean,
        "n": nw.height,
    }

    positive = joined.filter(pl.col("pred_yield_kg_ha") > 0)["pred_yield_kg_ha"]
    median_pos = float(positive.median()) if positive.len() else 0.0
    checks["A5_bahs_median_band"] = {
        "passed": 515.0 * 0.6 <= median_pos <= 515.0 * 1.4,
        "positive_median_kg_ha": median_pos,
        "target_kg_ha": BAHS_WHEAT_GROSS_KG_HA,
    }

    checks["A6_model_fit"] = {
        "passed": cv_r2_yield > 0.35,
        "cv_r2_yield": cv_r2_yield,
    }

    passed = all(bool(v["passed"]) for v in checks.values())
    return ValidationReport(
        passed=passed,
        checks=checks,
        metrics={
            "tech_scale": tech_scale,
            "cv_r2_yield": cv_r2_yield,
            "positive_median_kg_ha": median_pos,
            "nw_europe_mean_kg_ha": nw_mean,
            "arctic_mean_kg_ha": arctic_mean,
        },
        tech_scale=tech_scale,
    )


def train_and_predict_pnp_wheat(
    *,
    candidates_path: Path,
    pyaez_yields_path: Path,
    gaez_wide_path: Path,
    model_dir: Path,
) -> tuple[pl.DataFrame, ValidationReport, dict]:
    """Full train → predict → validate pipeline; persist model card JSON."""

    frame = assemble_training_frame(
        candidates_path=candidates_path,
        pyaez_yields_path=pyaez_yields_path,
        gaez_wide_path=gaez_wide_path,
    )
    yield_model, suit_model, columns, train_meta = train_pnp_wheat_models(frame)
    predictions = predict_pnp_wheat(
        frame,
        yield_model=yield_model,
        suit_model=suit_model,
        feature_cols=columns,
    )
    report = validate_pnp_predictions(
        frame,
        predictions,
        tech_scale=float(train_meta["tech_scale"]),
        cv_r2_yield=float(train_meta["cv_r2_yield"]),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    card = {
        **train_meta,
        "validation_passed": report.passed,
        "validation_checks": report.checks,
        "validation_metrics": report.metrics,
    }
    (model_dir / "pnp_wheat_model_card.json").write_text(
        json.dumps(card, indent=2),
        encoding="utf-8",
    )
    wide = predictions.select(
        LOCATION_TAG,
        "pnp_wheat_production_density",
        "pnp_wheat_yield",
        "pnp_wheat_suitable_fraction",
        "pnp_wheat_suitability_class",
    )
    return wide, report, card
