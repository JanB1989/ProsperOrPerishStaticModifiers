"""Successive historical evidence for P&P wheat calibration.

Evidence lives in ``research/pnp_wheat_evidence.json``. Append new anchors and
retrain — the pipeline soft-scales regional training targets, optionally flips
irrigation policy for irrigated corridors, then learns a feature residual that
generalizes sparse regional scales to similar climates worldwide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "high": 0.75,
    "medium": 0.55,
    "low": 0.35,
}

DEFAULT_PRIOR_STRENGTH = 1.0
DEFAULT_SCALE_CLIP = (0.25, 4.0)


@dataclass(frozen=True)
class EvidenceCatalog:
    path: Path
    version: int
    good: str
    method: dict[str, Any]
    global_anchor: dict[str, Any]
    evidence: list[dict[str, Any]]

    @property
    def prior_strength(self) -> float:
        return float(self.method.get("prior_strength", DEFAULT_PRIOR_STRENGTH))

    @property
    def scale_clip(self) -> tuple[float, float]:
        raw = self.method.get("scale_clip", list(DEFAULT_SCALE_CLIP))
        return float(raw[0]), float(raw[1])

    def weight_for(self, entry: dict[str, Any]) -> float:
        if "weight" in entry and entry["weight"] is not None:
            return float(entry["weight"])
        conf = str(entry.get("confidence", "medium")).lower()
        overrides = self.method.get("confidence_weights") or CONFIDENCE_WEIGHTS
        return float(overrides.get(conf, CONFIDENCE_WEIGHTS["medium"]))


def default_evidence_path(repo: Path | None = None) -> Path:
    root = repo or Path(__file__).resolve().parents[2]
    return root / "research" / "pnp_wheat_evidence.json"


def load_evidence_catalog(path: Path | None = None) -> EvidenceCatalog:
    evidence_path = path or default_evidence_path()
    if not evidence_path.is_file():
        raise FileNotFoundError(f"missing evidence catalog: {evidence_path}")
    raw = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    return EvidenceCatalog(
        path=evidence_path,
        version=int(raw.get("version", 1)),
        good=str(raw.get("good", "wheat")),
        method=dict(raw.get("method") or {}),
        global_anchor=dict(raw.get("global_anchor") or {}),
        evidence=list(raw.get("evidence") or []),
    )


def irrigated_prefer_regions(catalog: EvidenceCatalog) -> set[str]:
    out: set[str] = set()
    for entry in catalog.evidence:
        if entry.get("irrigation_policy") == "prefer_irrigated":
            out.update(str(r) for r in entry.get("eu5_regions") or [])
    return out


def _region_mask(regions: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    wanted = {str(r) for r in entry.get("eu5_regions") or []}
    if not wanted:
        return np.zeros(len(regions), dtype=bool)
    return np.isin(regions, list(wanted))


def apply_evidence_target_scales(
    frame: pl.DataFrame,
    catalog: EvidenceCatalog,
    *,
    yield_col: str = "target_yield_kg_ha",
    region_col: str = "region",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Soft-scale training targets toward regional historical medians.

    For each evidence entry, compute ``ratio = target / positive_median`` on
    matching locations, then apply a confidence-weighted log-scale with a prior
    toward 1 (Empirical-Bayes style). Overlapping regions accumulate weighted
    log-ratios.
    """

    if region_col not in frame.columns:
        raise ValueError(f"frame missing {region_col} for evidence matching")
    if yield_col not in frame.columns:
        raise ValueError(f"frame missing {yield_col}")

    y = frame[yield_col].fill_null(0.0).to_numpy().astype(np.float64)
    regions = frame[region_col].cast(pl.String).fill_null("").to_numpy()
    log_num = np.zeros(len(y), dtype=np.float64)
    w_sum = np.zeros(len(y), dtype=np.float64)
    applied: list[dict[str, Any]] = []
    lo, hi = catalog.scale_clip
    prior = catalog.prior_strength

    for entry in catalog.evidence:
        mask = _region_mask(regions, entry)
        n = int(mask.sum())
        if n == 0:
            applied.append({"id": entry.get("id"), "skipped": "no_locations"})
            continue
        w = catalog.weight_for(entry)
        kind = str(entry.get("kind", "positive_median"))
        if kind == "near_zero":
            # Soft pull toward zero: scale ≈ (1 - w_eff) with prior.
            w_eff = w / (w + prior)
            ratio = max(1.0 - w_eff, lo)
            log_r = float(np.log(ratio))
            log_num[mask] += w * log_r
            w_sum[mask] += w
            applied.append(
                {
                    "id": entry.get("id"),
                    "kind": kind,
                    "n": n,
                    "weight": w,
                    "implied_ratio": ratio,
                }
            )
            continue

        target = float(entry.get("target_kg_ha") or 0.0)
        if target <= 0:
            applied.append({"id": entry.get("id"), "skipped": "non_positive_target"})
            continue
        pos = mask & (y > 1.0)
        if int(pos.sum()) < 5:
            applied.append({"id": entry.get("id"), "skipped": "too_few_positive"})
            continue
        med = float(np.median(y[pos]))
        if med <= 0:
            applied.append({"id": entry.get("id"), "skipped": "zero_median"})
            continue
        ratio = float(np.clip(target / med, lo, hi))
        log_num[mask] += w * float(np.log(ratio))
        w_sum[mask] += w
        applied.append(
            {
                "id": entry.get("id"),
                "kind": kind,
                "n": n,
                "n_positive": int(pos.sum()),
                "weight": w,
                "pre_median_kg_ha": med,
                "target_kg_ha": target,
                "raw_ratio": target / med,
                "clipped_ratio": ratio,
            }
        )

    scale = np.ones(len(y), dtype=np.float64)
    touched = w_sum > 0
    scale[touched] = np.exp(log_num[touched] / (w_sum[touched] + prior))
    scale = np.clip(scale, lo, hi)

    out = frame.with_columns(
        (pl.Series(yield_col, y * scale)).alias(yield_col),
        pl.Series("evidence_target_scale", scale),
    )
    if "target_production_density_kg_ha" in out.columns:
        out = out.with_columns(
            (pl.col("target_production_density_kg_ha") * pl.col("evidence_target_scale")).alias(
                "target_production_density_kg_ha"
            )
        )
    meta = {
        "n_touched": int(touched.sum()),
        "scale_mean": float(scale.mean()),
        "scale_p05": float(np.quantile(scale, 0.05)),
        "scale_p95": float(np.quantile(scale, 0.95)),
        "entries": applied,
    }
    return out, meta


def fit_evidence_residual_calibrator(
    frame: pl.DataFrame,
    predictions: pl.DataFrame,
    catalog: EvidenceCatalog,
    feature_cols: list[str],
    *,
    region_col: str = "region",
) -> tuple[HistGradientBoostingRegressor | None, dict[str, Any]]:
    """Learn feature→log(scale) from sparse regional evidence residuals.

    Locations in evidence regions receive labels ``log(target/pred_median)``.
    The calibrator is then applied globally so irrigated-arid corridors similar
    to Egypt can inherit correction even outside named regions.
    """

    joined = frame.select(
        [LOCATION_TAG, region_col] + [c for c in feature_cols if c in frame.columns]
    ).join(
        predictions.select(LOCATION_TAG, "pred_yield_kg_ha"),
        on=LOCATION_TAG,
        how="inner",
    )
    regions = joined[region_col].cast(pl.String).fill_null("").to_numpy()
    pred = joined["pred_yield_kg_ha"].fill_null(0.0).to_numpy().astype(np.float64)
    lo, hi = catalog.scale_clip

    log_label = np.zeros(len(pred), dtype=np.float64)
    sample_w = np.zeros(len(pred), dtype=np.float64)
    used: list[dict[str, Any]] = []

    for entry in catalog.evidence:
        mask = _region_mask(regions, entry)
        if int(mask.sum()) == 0:
            continue
        w = catalog.weight_for(entry)
        kind = str(entry.get("kind", "positive_median"))
        if kind == "near_zero":
            mean_pred = float(pred[mask].mean()) if mask.any() else 0.0
            # Pull means toward ~0 without forcing exact zeros on every cell.
            target_mean = float(entry.get("max_mean_kg_ha", 20.0)) * 0.25
            if mean_pred <= 1.0:
                ratio = 1.0
            else:
                ratio = float(np.clip(target_mean / mean_pred, lo, hi))
            log_label[mask] = np.log(ratio)
            sample_w[mask] = w
            used.append(
                {
                    "id": entry.get("id"),
                    "kind": kind,
                    "pred_mean": mean_pred,
                    "ratio": ratio,
                }
            )
            continue

        target = float(entry.get("target_kg_ha") or 0.0)
        pos = mask & (pred > 1.0)
        if int(pos.sum()) < 5 or target <= 0:
            continue
        med = float(np.median(pred[pos]))
        if med <= 0:
            continue
        ratio = float(np.clip(target / med, lo, hi))
        log_label[mask] = np.log(ratio)
        sample_w[mask] = w
        used.append(
            {
                "id": entry.get("id"),
                "kind": kind,
                "pred_median": med,
                "target": target,
                "ratio": ratio,
            }
        )

    train_mask = sample_w > 0
    if int(train_mask.sum()) < 50:
        return None, {"skipped": "too_few_calibration_rows", "n": int(train_mask.sum()), "used": used}

    x_cols = [c for c in feature_cols if c in joined.columns]
    x = (
        joined.select(x_cols)
        .fill_null(0.0)
        .fill_nan(0.0)
        .to_numpy()
        .astype(np.float64)
    )
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.06,
        max_iter=120,
        l2_regularization=1.0,
        random_state=1337,
    )
    model.fit(x[train_mask], log_label[train_mask], sample_weight=sample_w[train_mask])
    meta = {
        "n_train": int(train_mask.sum()),
        "feature_columns": x_cols,
        "used": used,
        "label_log_mean": float(log_label[train_mask].mean()),
    }
    return model, meta


def apply_residual_calibrator(
    frame: pl.DataFrame,
    predictions: pl.DataFrame,
    calibrator: HistGradientBoostingRegressor,
    feature_cols: list[str],
    *,
    scale_clip: tuple[float, float] = DEFAULT_SCALE_CLIP,
) -> pl.DataFrame:
    from prosper_or_perish_static_modifiers.crops import HECTARES_PER_KM2

    x_cols = [c for c in feature_cols if c in frame.columns]
    x = frame.select(x_cols).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float64)
    log_scale = calibrator.predict(x)
    lo, hi = scale_clip
    scale = np.clip(np.exp(log_scale), lo, hi)

    yield_ha = predictions["pred_yield_kg_ha"].to_numpy().astype(np.float64) * scale
    suit = predictions["pnp_wheat_suitable_fraction"].to_numpy().astype(np.float64)
    # Keep near-zero cells near zero after calibration.
    yield_ha = np.where(predictions["pred_yield_kg_ha"].to_numpy() <= 0.5, 0.0, yield_ha)
    yield_ha = np.clip(yield_ha, 0.0, None)
    prod_ha = yield_ha * suit

    from prosper_or_perish_static_modifiers.pnp_model import suitability_class_from_fraction

    return pl.DataFrame(
        {
            LOCATION_TAG: predictions[LOCATION_TAG],
            "pnp_wheat_yield": yield_ha * HECTARES_PER_KM2,
            "pnp_wheat_suitable_fraction": suit,
            "pnp_wheat_production_density": prod_ha * HECTARES_PER_KM2,
            "pnp_wheat_suitability_class": suitability_class_from_fraction(suit),
            "pred_yield_kg_ha": yield_ha,
            "evidence_residual_scale": scale,
        }
    )


def validate_evidence_gates(
    frame: pl.DataFrame,
    predictions: pl.DataFrame,
    catalog: EvidenceCatalog,
    *,
    region_col: str = "region",
) -> dict[str, dict[str, object]]:
    """Per-evidence validation gates (E*)."""

    joined = frame.select(LOCATION_TAG, region_col).join(
        predictions.select(LOCATION_TAG, "pred_yield_kg_ha"),
        on=LOCATION_TAG,
        how="inner",
    )
    regions = joined[region_col].cast(pl.String).fill_null("").to_numpy()
    pred = joined["pred_yield_kg_ha"].fill_null(0.0).to_numpy().astype(np.float64)
    checks: dict[str, dict[str, object]] = {}

    for entry in catalog.evidence:
        eid = str(entry.get("id", "unknown"))
        mask = _region_mask(regions, entry)
        n = int(mask.sum())
        kind = str(entry.get("kind", "positive_median"))
        if n == 0:
            checks[f"E_{eid}"] = {"passed": False, "reason": "no_locations", "n": 0}
            continue
        if kind == "near_zero":
            mean_v = float(pred[mask].mean())
            max_mean = float(entry.get("max_mean_kg_ha", 40.0))
            checks[f"E_{eid}"] = {
                "passed": mean_v <= max_mean,
                "kind": kind,
                "n": n,
                "mean_kg_ha": mean_v,
                "max_mean_kg_ha": max_mean,
            }
            continue

        target = float(entry.get("target_kg_ha") or 0.0)
        tol = float(entry.get("tolerance", 0.45))
        pos = mask & (pred > 1.0)
        if int(pos.sum()) < 3:
            # Irrigated corridors must have positive yields.
            checks[f"E_{eid}"] = {
                "passed": False,
                "kind": kind,
                "n": n,
                "n_positive": int(pos.sum()),
                "reason": "too_few_positive",
                "target_kg_ha": target,
            }
            continue
        med = float(np.median(pred[pos]))
        lo = target * (1.0 - tol)
        hi = target * (1.0 + tol)
        checks[f"E_{eid}"] = {
            "passed": lo <= med <= hi,
            "kind": kind,
            "n": n,
            "n_positive": int(pos.sum()),
            "positive_median_kg_ha": med,
            "target_kg_ha": target,
            "band": [lo, hi],
            "confidence": entry.get("confidence"),
        }
    return checks
