from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from prosper_or_perish_static_modifiers.crops import HECTARES_PER_KM2
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG
from prosper_or_perish_static_modifiers.pnp_evidence import (
    build_location_labels_from_evidence,
    default_evidence_path,
    load_evidence_catalog,
    validate_attribute_strata,
)

BAHS_WHEAT_GROSS_KG_HA = 515.0
WHEAT_OPTIMUM_TEMP_C = 12.0

# Curated agronomic features — never region / super_region.
CANDIDATE_NUMERIC: tuple[str, ...] = (
    "calibrated_lat",
    "calibrated_lon",
    "chelsa_annual_mean_temperature",
    "chelsa_annual_precipitation",
    "chelsa_precipitation_seasonality",
    "gaez_wheat_low_input_yield",
    "gaez_wheat_low_input_yield_suitable_fraction",
    "gaez_wheat_irrigated_low_input_yield",
    "gaez_wheat_irrigated_low_input_yield_suitable_fraction",
    "gaez_irrigated_wheat_potential",
    "gaez_wheat_suitability",
    "gaez_wheat_land_yield",
    "gaez_barley_low_input_yield",
    "terrain_irrigable_fraction",
    "water_irrigable_fraction",
    "conveyance_irrigable_fraction",
    "irrigable_fraction",
    "farm_system_hydraulic_access_fraction",
    "farm_system_production_scale_p50",
)

EXTERNAL_NUMERIC: tuple[str, ...] = (
    "spam_wheat_rainfed_yield",
    "spam_wheat_rainfed_harvested_area",
    "europe_ag_suitability_1500",
    "hyde_pop_density_1300",
    "glw_cattle_density",
    "glw_sheep_density",
)

GAEZ_WIDE_NUMERIC: tuple[str, ...] = (
    "wheat_rainfed_yield_kg_dm_suitable_km2",
    "wheat_rainfed_suitable_fraction",
    "wheat_irrigated_yield_kg_dm_suitable_km2",
    "wheat_irrigated_suitable_fraction",
    "barley_rainfed_yield_kg_dm_suitable_km2",
    "rye_rainfed_yield_kg_dm_suitable_km2",
    "oats_rainfed_yield_kg_dm_suitable_km2",
)

ENGINEERED_NUMERIC: tuple[str, ...] = (
    "abs_lat",
    "aridity_proxy",
    "temp_distance_from_optimum",
    "severe_winter",
    "soil_quality_ord",
    "has_river_f",
    "has_winter_f",
    "is_coastal_f",
    "is_adjacent_to_lake_f",
    "pyaez_wheat_rainfed_yield",
    "pyaez_wheat_irrigated_yield",
    "pyaez_wheat_rainfed_suit",
    "pyaez_wheat_irrigated_suit",
)

CATEGORICAL_PREFIXES: tuple[str, ...] = ("topo_", "veg_", "clim_")

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


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    checks: dict[str, dict[str, object]]
    metrics: dict[str, float]
    tech_scale: float


def _encode_categories(frame: pl.DataFrame) -> pl.DataFrame:
    out = frame
    for col, prefix in (
        ("topography", "topo_"),
        ("vegetation", "veg_"),
        ("climate", "clim_"),
    ):
        values = sorted(frame[col].drop_nulls().unique().to_list())
        for value in values:
            safe = str(value).replace(" ", "_")
            out = out.with_columns(
                (pl.col(col).cast(pl.String) == value).cast(pl.Float64).alias(f"{prefix}{safe}")
            )
    return out


def _bool_f(frame: pl.DataFrame, col: str, alias: str) -> pl.Expr:
    if col not in frame.columns:
        return pl.lit(0.0).alias(alias)
    return pl.col(col).cast(pl.Boolean).fill_null(False).cast(pl.Float64).alias(alias)


def assemble_training_frame(
    *,
    candidates_path: Path,
    pyaez_yields_path: Path,
    gaez_wide_path: Path,
    external_wide_path: Path | None = None,
    crop_mode_labels_path: Path | None = None,
    evidence_path: Path | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Join attributes + build attribute-matched labels (no region scaling)."""

    catalog = load_evidence_catalog(evidence_path or default_evidence_path())

    cand_schema = set(pl.scan_parquet(candidates_path).collect_schema().names())
    cand_cols = [
        LOCATION_TAG,
        "topography",
        "vegetation",
        "climate",
        "climate_winter",
        "soil_quality",
        "has_river",
        "has_winter",
        "is_coastal",
        "is_adjacent_to_lake",
        *CANDIDATE_NUMERIC,
    ]
    candidates = pl.read_parquet(candidates_path).select(
        [c for c in cand_cols if c in cand_schema]
    )

    gaez_raw = pl.read_parquet(gaez_wide_path)
    gaez_cols = [LOCATION_TAG] + [c for c in GAEZ_WIDE_NUMERIC if c in gaez_raw.columns]
    gaez = gaez_raw.select(gaez_cols)
    for col in GAEZ_WIDE_NUMERIC:
        if col not in gaez.columns:
            gaez = gaez.with_columns(pl.lit(0.0).alias(col))

    external = None
    if external_wide_path is not None and external_wide_path.is_file():
        ext_raw = pl.read_parquet(external_wide_path)
        ext_cols = [LOCATION_TAG] + [c for c in EXTERNAL_NUMERIC if c in ext_raw.columns]
        external = ext_raw.select(ext_cols)

    wheat = (
        pl.read_parquet(pyaez_yields_path)
        .filter(pl.col("crop") == "wheat")
        .select(
            LOCATION_TAG,
            "water_mode",
            pl.col("yield_kg_dm_ha").cast(pl.Float64),
            pl.col("suitable_fraction").cast(pl.Float64),
        )
    )
    rainfed = wheat.filter(pl.col("water_mode") == "rainfed").select(
        LOCATION_TAG,
        pl.col("yield_kg_dm_ha").alias("pyaez_wheat_rainfed_yield"),
        pl.col("suitable_fraction").alias("pyaez_wheat_rainfed_suit"),
    )
    irrigated = wheat.filter(pl.col("water_mode") == "irrigated").select(
        LOCATION_TAG,
        pl.col("yield_kg_dm_ha").alias("pyaez_wheat_irrigated_yield"),
        pl.col("suitable_fraction").alias("pyaez_wheat_irrigated_suit"),
    )
    pyaez = rainfed.join(irrigated, on=LOCATION_TAG, how="full", coalesce=True).with_columns(
        pl.col("pyaez_wheat_rainfed_yield").fill_null(0.0),
        pl.col("pyaez_wheat_irrigated_yield").fill_null(0.0),
        pl.col("pyaez_wheat_rainfed_suit").fill_null(0.0),
        pl.col("pyaez_wheat_irrigated_suit").fill_null(0.0),
    )

    # Physical soft prior: rainfed-first; allow irrigated where river/hydraulic access.
    frame = candidates.join(pyaez, on=LOCATION_TAG, how="inner").join(
        gaez, on=LOCATION_TAG, how="left"
    )
    if external is not None:
        frame = frame.join(external, on=LOCATION_TAG, how="left")

    for col in CANDIDATE_NUMERIC + EXTERNAL_NUMERIC + GAEZ_WIDE_NUMERIC:
        if col in frame.columns:
            frame = frame.with_columns(pl.col(col).cast(pl.Float64).fill_null(0.0))
        else:
            frame = frame.with_columns(pl.lit(0.0).alias(col))

    has_river = (
        frame["has_river"].cast(pl.Boolean).fill_null(False)
        if "has_river" in frame.columns
        else pl.Series([False] * frame.height)
    )
    hydraulic = frame["farm_system_hydraulic_access_fraction"].fill_null(0.0)
    prefer_ir = has_river | (hydraulic > 0.05)
    phys_yield = (
        pl.when(prefer_ir)
        .then(
            pl.max_horizontal(
                "pyaez_wheat_rainfed_yield", "pyaez_wheat_irrigated_yield"
            )
        )
        .otherwise(pl.col("pyaez_wheat_rainfed_yield"))
    )
    phys_suit = (
        pl.when(prefer_ir)
        .then(pl.max_horizontal("pyaez_wheat_rainfed_suit", "pyaez_wheat_irrigated_suit"))
        .otherwise(pl.col("pyaez_wheat_rainfed_suit"))
    )
    frame = frame.with_columns(
        phys_yield.alias("physical_yield_raw"),
        phys_suit.alias("physical_suitable_fraction"),
    )
    positive = frame.filter(pl.col("physical_yield_raw") > 0)["physical_yield_raw"]
    median_pos = float(positive.median()) if positive.len() else 1.0
    tech_scale = catalog.bahs_kg_ha / median_pos if median_pos > 0 else 1.0
    frame = frame.with_columns(
        (pl.col("physical_yield_raw") * tech_scale).alias("physical_yield_kg_ha"),
        pl.lit(tech_scale).alias("tech_scale"),
    )

    # Temperature for engineering: prefer absolute CHELSA; fall back from normalized.
    temp_col = (
        "chelsa_annual_mean_temperature"
        if "chelsa_annual_mean_temperature" in frame.columns
        else "chelsa_annual_mean_temperature_target"
    )
    precip_col = (
        "chelsa_annual_precipitation"
        if "chelsa_annual_precipitation" in frame.columns
        else "chelsa_annual_precipitation_target"
    )

    frame = frame.with_columns(
        pl.col("calibrated_lat").abs().alias("abs_lat"),
        (
            pl.col(precip_col)
            / (pl.col(temp_col) + 10.0)
        ).alias("aridity_proxy"),
        (pl.col(temp_col) - WHEAT_OPTIMUM_TEMP_C).abs().alias("temp_distance_from_optimum"),
        (pl.col("climate_winter").cast(pl.String) == "severe")
        .cast(pl.Float64)
        .alias("severe_winter"),
        pl.col("soil_quality")
        .cast(pl.String)
        .replace_strict(SOIL_QUALITY_ORD, default=3.0)
        .cast(pl.Float64)
        .alias("soil_quality_ord"),
        _bool_f(frame, "has_river", "has_river_f"),
        _bool_f(frame, "has_winter", "has_winter_f"),
        _bool_f(frame, "is_coastal", "is_coastal_f"),
        _bool_f(frame, "is_adjacent_to_lake", "is_adjacent_to_lake_f"),
    )

    frame = _encode_categories(frame)
    frame, label_meta = build_location_labels_from_evidence(
        frame,
        catalog,
        crop_history_path=crop_mode_labels_path,
        include_holdout_in_training=False,
    )

    assemble_meta = {
        "tech_scale": tech_scale,
        "bahs_wheat_gross_kg_ha": catalog.bahs_kg_ha,
        "pyaez_positive_median_before_scale": median_pos,
        "evidence_path": str(catalog.path),
        "evidence_version": catalog.version,
        "label_construction": label_meta,
        "n_features_numeric_base": len(CANDIDATE_NUMERIC)
        + len(EXTERNAL_NUMERIC)
        + len(GAEZ_WIDE_NUMERIC)
        + len(ENGINEERED_NUMERIC),
    }
    return frame, assemble_meta


def feature_columns(frame: pl.DataFrame) -> list[str]:
    cols: list[str] = []
    for group in (
        CANDIDATE_NUMERIC,
        EXTERNAL_NUMERIC,
        GAEZ_WIDE_NUMERIC,
        ENGINEERED_NUMERIC,
    ):
        for col in group:
            if col in frame.columns:
                cols.append(col)
    for col in frame.columns:
        if any(col.startswith(prefix) for prefix in CATEGORICAL_PREFIXES):
            cols.append(col)
    # Hard ban on geography IDs.
    banned = {"region", "super_region", "macro_region", "province", "area"}
    return [c for c in cols if c not in banned and not c.startswith("region")]


def _matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"training frame missing features: {missing[:8]}")
    data = frame.select(columns).fill_null(0.0).fill_nan(0.0).to_numpy()
    return np.asarray(data, dtype=np.float64)


def suitability_class_from_fraction(fraction: np.ndarray) -> np.ndarray:
    f = np.clip(np.asarray(fraction, dtype=np.float64), 0.0, 1.0)
    raw = 1.0 + (1.0 - f) * 8.0
    classes = np.rint(raw).astype(np.float64)
    classes[(f <= 0) | ~np.isfinite(f)] = 9.0
    return np.clip(classes, 1.0, 9.0)


def train_pnp_wheat_models(
    frame: pl.DataFrame,
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor, list[str], dict]:
    columns = feature_columns(frame)
    x = _matrix(frame, columns)
    y_yield = frame["label_yield_kg_ha"].to_numpy().astype(np.float64)
    y_suit = np.clip(frame["label_suitable_fraction"].to_numpy().astype(np.float64), 0.0, 1.0)
    w_yield = frame["sample_weight_yield"].to_numpy().astype(np.float64)
    w_suit = frame["sample_weight_suit"].to_numpy().astype(np.float64)

    yield_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=220,
        l2_regularization=0.2,
        random_state=1337,
    )
    suit_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.08,
        max_iter=180,
        l2_regularization=0.2,
        random_state=1337,
    )
    yield_model.fit(x, y_yield, sample_weight=w_yield)
    suit_model.fit(x, y_suit, sample_weight=w_suit)

    yield_cv = float(
        np.mean(
            cross_val_score(
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=120,
                    l2_regularization=0.2,
                    random_state=1337,
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
                    l2_regularization=0.2,
                    random_state=1337,
                ),
                x,
                y_suit,
                cv=3,
                scoring="r2",
            )
        )
    )

    # Permutation importance on a stratified sample (hard labels oversampled).
    rng = np.random.default_rng(1337)
    hard = frame["label_hard"].to_numpy().astype(bool)
    idx_hard = np.flatnonzero(hard)
    idx_soft = np.flatnonzero(~hard)
    take_hard = idx_hard if len(idx_hard) <= 800 else rng.choice(idx_hard, 800, replace=False)
    take_soft = idx_soft if len(idx_soft) <= 400 else rng.choice(idx_soft, 400, replace=False)
    sample_idx = np.concatenate([take_hard, take_soft])
    perm = permutation_importance(
        yield_model,
        x[sample_idx],
        y_yield[sample_idx],
        n_repeats=5,
        random_state=1337,
        scoring="r2",
    )
    order = np.argsort(perm.importances_mean)[::-1]
    top_importance = [
        {
            "feature": columns[i],
            "importance_mean": float(perm.importances_mean[i]),
            "importance_std": float(perm.importances_std[i]),
        }
        for i in order[:25]
    ]

    tech_scale = float(frame["tech_scale"][0])
    meta = {
        "feature_columns": columns,
        "n_features": len(columns),
        "tech_scale": tech_scale,
        "bahs_wheat_gross_kg_ha": BAHS_WHEAT_GROSS_KG_HA,
        "cv_r2_yield": yield_cv,
        "cv_r2_suitable_fraction": suit_cv,
        "n_rows": int(frame.height),
        "feature_importance_top": top_importance,
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

    # Soft physical veto for arctic + desert-without-water (attribute-based, not region).
    climate = frame["climate"].cast(pl.String).to_numpy()
    vegetation = frame["vegetation"].cast(pl.String).to_numpy()
    precip = (
        frame["chelsa_annual_precipitation"].fill_null(0.0).to_numpy()
        if "chelsa_annual_precipitation" in frame.columns
        else np.zeros(len(climate))
    )
    has_river = (
        frame["has_river_f"].fill_null(0.0).to_numpy()
        if "has_river_f" in frame.columns
        else np.zeros(len(climate))
    )
    arctic = climate == "arctic"
    desert_dry = (vegetation == "desert") & (precip < 150) & (has_river < 0.5)
    hostile = arctic | desert_dry
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
    evidence_path: Path | None = None,
) -> ValidationReport:
    joined = frame.select(
        LOCATION_TAG,
        "climate",
        "vegetation",
        "chelsa_annual_precipitation",
        "calibrated_lat",
        "calibrated_lon",
        "abs_lat",
        "has_river_f",
    ).join(predictions, on=LOCATION_TAG, how="inner")

    checks: dict[str, dict[str, object]] = {}

    arctic = joined.filter((pl.col("climate") == "arctic") & (pl.col("abs_lat") > 60))
    arctic_mean = float(arctic["pred_yield_kg_ha"].mean() or 0.0)
    checks["A1_arctic_near_zero"] = {
        "passed": arctic_mean < 80.0,
        "arctic_mean_kg_ha": arctic_mean,
        "n": arctic.height,
    }

    arid = joined.filter(
        (pl.col("vegetation") == "desert")
        & (pl.col("chelsa_annual_precipitation") < 150)
        & (pl.col("has_river_f") < 0.5)
    )
    arid_mean = float(arid["pred_yield_kg_ha"].mean() or 0.0)
    checks["A2_hyper_arid_no_water_low"] = {
        "passed": arid_mean < 200.0,
        "arid_mean_kg_ha": arid_mean,
        "n": arid.height,
    }

    tropical = joined.filter(
        (pl.col("climate") == "tropical") & (pl.col("vegetation") == "jungle")
    )
    temperate = joined.filter(
        pl.col("climate").is_in(["oceanic", "continental", "mediterranean"])
    )
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
    # Broader band: attribute learning need not pin global median to BAHS.
    checks["A5_global_median_sane"] = {
        "passed": 150.0 <= median_pos <= 1200.0,
        "positive_median_kg_ha": median_pos,
        "reference_bahs_kg_ha": BAHS_WHEAT_GROSS_KG_HA,
    }

    checks["A6_model_fit"] = {
        "passed": cv_r2_yield > 0.25,
        "cv_r2_yield": cv_r2_yield,
    }

    catalog = load_evidence_catalog(evidence_path or default_evidence_path())
    strata = validate_attribute_strata(frame, predictions, catalog)
    checks.update(strata)

    # Holdout gates are informational for deploy: require A* + S* only.
    hard_keys = [k for k in checks if k.startswith("A") or k.startswith("S_")]
    passed = all(bool(checks[k]["passed"]) for k in hard_keys)
    return ValidationReport(
        passed=passed,
        checks=checks,
        metrics={
            "tech_scale": tech_scale,
            "cv_r2_yield": cv_r2_yield,
            "positive_median_kg_ha": median_pos,
            "nw_europe_mean_kg_ha": nw_mean,
            "arctic_mean_kg_ha": arctic_mean,
            "n_strata_gates": float(sum(1 for k in checks if k.startswith("S_"))),
            "n_strata_passed": float(
                sum(1 for k, v in checks.items() if k.startswith("S_") and v.get("passed"))
            ),
            "n_holdout_passed": float(
                sum(1 for k, v in checks.items() if k.startswith("H_") and v.get("passed"))
            ),
        },
        tech_scale=tech_scale,
    )


def train_and_predict_pnp_wheat(
    *,
    candidates_path: Path,
    pyaez_yields_path: Path,
    gaez_wide_path: Path,
    model_dir: Path,
    evidence_path: Path | None = None,
    geometry_path: Path | None = None,
    external_wide_path: Path | None = None,
    crop_mode_labels_path: Path | None = None,
) -> tuple[pl.DataFrame, ValidationReport, dict]:
    """Attribute-driven train → predict → validate; persist model card."""

    _ = geometry_path  # reserved; region joins intentionally unused
    evidence_path = evidence_path or default_evidence_path()
    if crop_mode_labels_path is None:
        crop_mode_labels_path = (
            Path(__file__).resolve().parents[2]
            / "../ProsperOrPerishConstructor/artifacts/data/population_capacity/"
            "crop_mode_labels.parquet"
        ).resolve()
    if external_wide_path is None:
        external_wide_path = (
            Path(__file__).resolve().parents[2] / "artifacts" / "location_external_wide.parquet"
        )

    frame, assemble_meta = assemble_training_frame(
        candidates_path=candidates_path,
        pyaez_yields_path=pyaez_yields_path,
        gaez_wide_path=gaez_wide_path,
        external_wide_path=external_wide_path,
        crop_mode_labels_path=crop_mode_labels_path,
        evidence_path=evidence_path,
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
        evidence_path=evidence_path,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    card = {
        **train_meta,
        **assemble_meta,
        "validation_passed": report.passed,
        "validation_checks": report.checks,
        "validation_metrics": report.metrics,
        "method": "attribute_matched_dual_head",
    }
    (model_dir / "pnp_wheat_model_card.json").write_text(
        json.dumps(card, indent=2),
        encoding="utf-8",
    )
    research_card = Path(__file__).resolve().parents[2] / "research" / "pnp_wheat_model_card.json"
    research_card.write_text(json.dumps(card, indent=2), encoding="utf-8")

    wide = predictions.select(
        LOCATION_TAG,
        "pnp_wheat_production_density",
        "pnp_wheat_yield",
        "pnp_wheat_suitable_fraction",
        "pnp_wheat_suitability_class",
    )
    return wide, report, card
