"""Historical breadbasket assumption tests for P&P wheat.

Catalog: ``research/pnp_wheat_historical_assumptions.json`` (~25–30 tests).
These are post-hoc validation only — never training labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG


def default_assumptions_path(repo: Path | None = None) -> Path:
    root = repo or Path(__file__).resolve().parents[2]
    return root / "research" / "pnp_wheat_historical_assumptions.json"


def load_assumption_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or default_assumptions_path()
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing assumption catalog: {catalog_path}")
    return json.loads(catalog_path.read_text(encoding="utf-8-sig"))


def _apply_selector(frame: pl.DataFrame, sel: dict[str, Any]) -> pl.DataFrame:
    out = frame
    if "region" in sel and "region" in out.columns:
        out = out.filter(pl.col("region") == sel["region"])
    if "bbox" in sel:
        la0, la1, lo0, lo1 = (float(x) for x in sel["bbox"])
        out = out.filter(
            (pl.col("calibrated_lat") >= la0)
            & (pl.col("calibrated_lat") <= la1)
            & (pl.col("calibrated_lon") >= lo0)
            & (pl.col("calibrated_lon") <= lo1)
        )
    if "has_river" in sel and "has_river_f" in out.columns:
        if bool(sel["has_river"]):
            out = out.filter(pl.col("has_river_f") > 0.5)
        else:
            out = out.filter(pl.col("has_river_f") <= 0.5)
    if "vegetation" in sel and "vegetation" in out.columns:
        out = out.filter(pl.col("vegetation") == sel["vegetation"])
    if "vegetation_in" in sel and "vegetation" in out.columns:
        out = out.filter(pl.col("vegetation").is_in(list(sel["vegetation_in"])))
    if "climate" in sel and "climate" in out.columns:
        out = out.filter(pl.col("climate") == sel["climate"])
    if sel.get("positive_only"):
        out = out.filter(pl.col("pred_yield_kg_ha") > 1.0)
    return out


def _median(df: pl.DataFrame) -> float | None:
    if df.height == 0:
        return None
    return float(df["pred_yield_kg_ha"].median())


def _mean(df: pl.DataFrame) -> float:
    if df.height == 0:
        return 0.0
    return float(df["pred_yield_kg_ha"].mean() or 0.0)


def run_historical_assumption_tests(
    joined: pl.DataFrame,
    *,
    catalog_path: Path | None = None,
) -> dict[str, dict[str, object]]:
    """Evaluate the dedicated historical assumption catalog against predictions."""

    catalog = load_assumption_catalog(catalog_path)
    tests_out: dict[str, dict[str, object]] = {}

    global_pos = joined.filter(pl.col("pred_yield_kg_ha") > 1.0)
    global_med = float(global_pos["pred_yield_kg_ha"].median()) if global_pos.height else 0.0
    old_world = global_pos.filter(pl.col("calibrated_lon") > -25.0)
    ow_q25 = float(old_world["pred_yield_kg_ha"].quantile(0.25)) if old_world.height else 0.0
    ow_q50 = float(old_world["pred_yield_kg_ha"].quantile(0.50)) if old_world.height else 0.0
    ow_q75 = float(old_world["pred_yield_kg_ha"].quantile(0.75)) if old_world.height else 0.0

    for entry in catalog.get("tests") or []:
        tid = str(entry["id"])
        expect = str(entry["expect"])
        result: dict[str, object] = {
            "group": entry.get("group"),
            "description": entry.get("description"),
            "expect": expect,
        }

        if expect in {"median_a_gt_median_b", "median_a_ge_median_b"}:
            a = _apply_selector(joined, entry["a"])
            b = _apply_selector(joined, entry["b"])
            a_med = _median(a)
            b_med = _median(b)
            if expect == "median_a_gt_median_b":
                passed = a_med is not None and b_med is not None and a_med > b_med
            else:
                passed = a_med is not None and b_med is not None and a_med >= b_med
            result.update(
                {
                    "passed": bool(passed),
                    "a_name": entry["a"].get("name"),
                    "b_name": entry["b"].get("name"),
                    "a_median_kg_ha": a_med,
                    "b_median_kg_ha": b_med,
                    "n_a": a.height,
                    "n_b": b.height,
                }
            )
        elif expect in {
            "median_ge_global_median",
            "median_ge_old_world_q25",
            "median_ge_old_world_q50",
            "median_ge_old_world_q75",
            "median_lt_old_world_q75",
            "mean_lt",
        }:
            sample = _apply_selector(joined, entry["sample"])
            med = _median(sample)
            mean_v = _mean(sample)
            if expect == "median_ge_global_median":
                passed = med is not None and med >= global_med
                result["threshold_kg_ha"] = global_med
            elif expect == "median_ge_old_world_q25":
                passed = med is not None and med >= ow_q25
                result["threshold_kg_ha"] = ow_q25
            elif expect == "median_ge_old_world_q50":
                passed = med is not None and med >= ow_q50
                result["threshold_kg_ha"] = ow_q50
            elif expect == "median_ge_old_world_q75":
                passed = med is not None and med >= ow_q75
                result["threshold_kg_ha"] = ow_q75
            elif expect == "median_lt_old_world_q75":
                passed = med is not None and med < ow_q75
                result["threshold_kg_ha"] = ow_q75
            else:  # mean_lt
                thr = float(entry.get("threshold_kg_ha", 80.0))
                passed = mean_v < thr
                result["threshold_kg_ha"] = thr
            result.update(
                {
                    "passed": bool(passed),
                    "sample_name": entry["sample"].get("name"),
                    "pos_median_kg_ha": med,
                    "mean_kg_ha": mean_v,
                    "n": sample.height,
                    "global_pos_median_kg_ha": global_med,
                    "old_world_q25_kg_ha": ow_q25,
                    "old_world_q75_kg_ha": ow_q75,
                }
            )
        else:
            result["passed"] = False
            result["error"] = f"unknown expect: {expect}"

        tests_out[tid] = result

    return tests_out


def assumption_catalog_stats(path: Path | None = None) -> dict[str, object]:
    catalog = load_assumption_catalog(path)
    tests = list(catalog.get("tests") or [])
    groups: dict[str, int] = {}
    for t in tests:
        g = str(t.get("group") or "ungrouped")
        groups[g] = groups.get(g, 0) + 1
    return {
        "path": str(path or default_assumptions_path()),
        "version": catalog.get("version"),
        "n_tests": len(tests),
        "groups": groups,
        "ids": [t["id"] for t in tests],
    }
