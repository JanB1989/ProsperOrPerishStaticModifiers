"""Attribute-matched historical evidence for P&P wheat.

Evidence recipes in ``research/pnp_wheat_evidence.json`` assign location-level
labels by agronomic attribute rules (never by EU5 region multipliers). The
model must learn X→y from those labels plus crop-history presence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG


@dataclass(frozen=True)
class EvidenceCatalog:
    path: Path
    version: int
    good: str
    method: dict[str, Any]
    evidence: list[dict[str, Any]]
    validation_strata: list[dict[str, Any]]

    @property
    def bahs_kg_ha(self) -> float:
        return float(self.method.get("bahs_wheat_gross_kg_ha", 515.0))

    @property
    def physical_soft_weight(self) -> float:
        return float(self.method.get("physical_soft_weight", 0.15))

    @property
    def history_base_weight(self) -> float:
        return float(self.method.get("history_base_weight", 1.0))


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
        version=int(raw.get("version", 2)),
        good=str(raw.get("good", "wheat")),
        method=dict(raw.get("method") or {}),
        evidence=list(raw.get("evidence") or []),
        validation_strata=list(raw.get("validation_strata") or []),
    )


def _eval_clause(frame: pl.DataFrame, clause: dict[str, Any]) -> pl.Expr:
    if "any" in clause:
        parts = [_eval_clause(frame, c) for c in clause["any"]]
        expr = parts[0]
        for p in parts[1:]:
            expr = expr | p
        return expr.fill_null(False)
    if "all" in clause:
        parts = [_eval_clause(frame, c) for c in clause["all"]]
        expr = parts[0]
        for p in parts[1:]:
            expr = expr & p
        return expr.fill_null(False)

    col = str(clause["col"])
    if col not in frame.columns:
        return pl.lit(False)

    series_expr = pl.col(col)
    if "eq" in clause:
        return (series_expr == clause["eq"]).fill_null(False)
    if "ne" in clause:
        return (series_expr != clause["ne"]).fill_null(False)
    if "in" in clause:
        return series_expr.is_in(list(clause["in"])).fill_null(False)
    if "gt" in clause:
        return (series_expr.cast(pl.Float64) > float(clause["gt"])).fill_null(False)
    if "gte" in clause:
        return (series_expr.cast(pl.Float64) >= float(clause["gte"])).fill_null(False)
    if "lt" in clause:
        return (series_expr.cast(pl.Float64) < float(clause["lt"])).fill_null(False)
    if "lte" in clause:
        return (series_expr.cast(pl.Float64) <= float(clause["lte"])).fill_null(False)
    raise ValueError(f"unsupported match clause: {clause}")


def match_mask(frame: pl.DataFrame, match: dict[str, Any] | None) -> np.ndarray:
    if not match:
        return np.zeros(frame.height, dtype=bool)
    expr = _eval_clause(frame, match)
    return frame.select(expr.alias("_m"))["_m"].to_numpy().astype(bool)


def load_crop_history_wheat_presence(crop_mode_labels_path: Path) -> pl.DataFrame:
    """Return location_tag rows with known wheat availability (presence positives)."""

    if not crop_mode_labels_path.is_file():
        raise FileNotFoundError(f"missing crop mode labels: {crop_mode_labels_path}")
    return (
        pl.scan_parquet(crop_mode_labels_path)
        .filter(
            (pl.col("crop") == "wheat")
            & (pl.col("historical_availability_state") == "known_available")
        )
        .select(LOCATION_TAG)
        .unique()
        .collect()
        .with_columns(pl.lit(True).alias("crop_history_wheat_present"))
    )


def build_location_labels_from_evidence(
    frame: pl.DataFrame,
    catalog: EvidenceCatalog,
    *,
    crop_history_path: Path | None = None,
    include_holdout_in_training: bool = False,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Assign yield / suitability / sample-weight columns from attribute recipes.

    Priority when recipes overlap: higher ``weight`` wins for hard labels.
    Crop-history presence raises suitability without overriding hard yield labels.
    Physical soft labels (``physical_yield_kg_ha`` / ``physical_suitable_fraction``
    on the frame) fill remaining gaps at low weight.
    """

    n = frame.height
    y_yield = np.full(n, np.nan, dtype=np.float64)
    y_suit = np.full(n, np.nan, dtype=np.float64)
    w_yield = np.zeros(n, dtype=np.float64)
    w_suit = np.zeros(n, dtype=np.float64)
    hard = np.zeros(n, dtype=bool)
    recipe_id = np.array([""] * n, dtype=object)
    recipe_meta: list[dict[str, Any]] = []

    # Sort by weight ascending so higher weight overwrites.
    recipes = sorted(
        catalog.evidence,
        key=lambda e: float(e.get("weight", catalog.history_base_weight)),
    )
    for entry in recipes:
        if entry.get("holdout") and not include_holdout_in_training:
            mask = match_mask(frame, entry.get("match"))
            recipe_meta.append(
                {
                    "id": entry.get("id"),
                    "holdout": True,
                    "n_match": int(mask.sum()),
                    "used_in_training": False,
                }
            )
            continue
        mask = match_mask(frame, entry.get("match"))
        n_match = int(mask.sum())
        if n_match == 0:
            recipe_meta.append({"id": entry.get("id"), "n_match": 0, "skipped": "no_match"})
            continue
        weight = float(entry.get("weight", catalog.history_base_weight))
        kind = str(entry.get("kind", "yield_intensity"))
        target = float(entry.get("target_kg_ha", 0.0))
        suit = float(entry.get("suitable_fraction", 0.0 if kind == "near_zero" else 0.8))
        y_yield[mask] = target
        y_suit[mask] = suit
        w_yield[mask] = weight
        w_suit[mask] = weight
        hard[mask] = True
        recipe_id[mask] = str(entry.get("id"))
        recipe_meta.append(
            {
                "id": entry.get("id"),
                "kind": kind,
                "n_match": n_match,
                "target_kg_ha": target,
                "weight": weight,
                "holdout": False,
                "used_in_training": True,
            }
        )

    # Crop-history presence → suitability positives (do not clobber hard zeros).
    presence_n = 0
    if crop_history_path is not None and crop_history_path.is_file():
        presence = load_crop_history_wheat_presence(crop_history_path)
        joined = frame.select(LOCATION_TAG).join(presence, on=LOCATION_TAG, how="left")
        present = (
            joined["crop_history_wheat_present"].fill_null(False).to_numpy().astype(bool)
        )
        presence_n = int(present.sum())
        # Raise suitability where present and not hard-zeroed near_zero.
        raise_suit = present & ~(hard & (y_yield <= 0.0))
        y_suit[raise_suit] = np.fmax(
            np.nan_to_num(y_suit[raise_suit], nan=0.0),
            0.85,
        )
        w_suit[raise_suit] = np.fmax(w_suit[raise_suit], catalog.history_base_weight * 1.5)
        # If yield still unlabeled, leave for physical soft fill.

    # Physical soft labels where still unlabeled.
    soft_w = catalog.physical_soft_weight
    soft_n = 0
    if "physical_yield_kg_ha" in frame.columns and soft_w > 0:
        phys_y = frame["physical_yield_kg_ha"].fill_null(0.0).to_numpy().astype(np.float64)
        phys_s = (
            frame["physical_suitable_fraction"].fill_null(0.0).to_numpy().astype(np.float64)
            if "physical_suitable_fraction" in frame.columns
            else np.clip(phys_y / 800.0, 0.0, 1.0)
        )
        need_y = ~hard & np.isnan(y_yield)
        y_yield[need_y] = phys_y[need_y]
        w_yield[need_y] = soft_w
        need_s = np.isnan(y_suit)
        y_suit[need_s] = phys_s[need_s]
        w_suit[need_s] = np.where(hard[need_s], w_suit[need_s], soft_w)
        soft_n = int(need_y.sum())

    # Remaining NaNs → 0 with tiny weight so matrix is dense for sklearn.
    still_y = np.isnan(y_yield)
    y_yield[still_y] = 0.0
    w_yield[still_y] = np.fmax(w_yield[still_y], 0.05)
    still_s = np.isnan(y_suit)
    y_suit[still_s] = 0.0
    w_suit[still_s] = np.fmax(w_suit[still_s], 0.05)

    out = frame.with_columns(
        pl.Series("label_yield_kg_ha", y_yield),
        pl.Series("label_suitable_fraction", np.clip(y_suit, 0.0, 1.0)),
        pl.Series("sample_weight_yield", w_yield),
        pl.Series("sample_weight_suit", w_suit),
        pl.Series("label_hard", hard),
        pl.Series("label_recipe_id", recipe_id),
    )
    meta = {
        "n_hard": int(hard.sum()),
        "n_soft_physical": soft_n,
        "n_crop_history_presence": presence_n,
        "recipes": recipe_meta,
        "yield_label_mean": float(np.average(y_yield, weights=w_yield)),
        "suit_label_mean": float(np.average(y_suit, weights=w_suit)),
    }
    return out, meta


def validate_attribute_strata(
    frame: pl.DataFrame,
    predictions: pl.DataFrame,
    catalog: EvidenceCatalog,
) -> dict[str, dict[str, object]]:
    """Attribute-strata gates (S*) and holdout recipe recovery (H*)."""

    joined = frame.join(
        predictions.select(LOCATION_TAG, "pred_yield_kg_ha"),
        on=LOCATION_TAG,
        how="inner",
    )
    checks: dict[str, dict[str, object]] = {}

    for stratum in catalog.validation_strata:
        sid = str(stratum.get("id", "unknown"))
        mask = match_mask(joined, stratum.get("match"))
        n = int(mask.sum())
        pred = joined["pred_yield_kg_ha"].to_numpy()[mask]
        expect = str(stratum.get("expect", "mean_lte"))
        thr = float(stratum.get("threshold_kg_ha", 0.0))
        if n == 0:
            checks[sid] = {"passed": False, "reason": "no_locations", "n": 0}
            continue
        mean_v = float(pred.mean())
        pos = pred[pred > 1.0]
        med_pos = float(np.median(pos)) if len(pos) else 0.0
        if expect == "mean_lte":
            passed = mean_v <= thr
        elif expect == "positive_median_gte":
            passed = med_pos >= thr
        elif expect == "positive_median_between":
            lo, hi = thr, float(stratum.get("threshold_high_kg_ha", thr))
            passed = lo <= med_pos <= hi
        else:
            passed = False
        checks[sid] = {
            "passed": passed,
            "n": n,
            "mean_kg_ha": mean_v,
            "positive_median_kg_ha": med_pos,
            "expect": expect,
            "threshold_kg_ha": thr,
        }

    for entry in catalog.evidence:
        if not entry.get("holdout"):
            continue
        hid = f"H_{entry.get('id')}"
        mask = match_mask(joined, entry.get("match"))
        n = int(mask.sum())
        target = float(entry.get("target_kg_ha", 0.0))
        tol = float(entry.get("tolerance", 0.5))
        if n < 5:
            checks[hid] = {"passed": False, "n": n, "reason": "too_few"}
            continue
        pred = joined["pred_yield_kg_ha"].to_numpy()[mask]
        pos = pred[pred > 1.0]
        med = float(np.median(pos)) if len(pos) else float(pred.mean())
        lo, hi = target * (1.0 - tol), target * (1.0 + tol)
        checks[hid] = {
            "passed": lo <= med <= hi,
            "n": n,
            "n_positive": int(len(pos)),
            "positive_median_kg_ha": med,
            "target_kg_ha": target,
            "band": [lo, hi],
            "holdout": True,
        }
    return checks
