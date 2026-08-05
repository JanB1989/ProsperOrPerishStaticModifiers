from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prosper_or_perish_static_modifiers.crops import (
    CROPS,
    GAEZ_V5_YLX_SHA256,
    GAEZ_V5_YXX_SHA256,
    iter_metrics,
    metric_column,
    raster_specs,
    sha256_lock_count,
)
from prosper_or_perish_static_modifiers.publish import publish_docs
from prosper_or_perish_static_modifiers.wide import pivot_location_metrics


def test_sha256_lock_covers_all_standard_crops() -> None:
    locks = sha256_lock_count()
    assert locks["crops"] == 23
    assert locks["yxx"] == 44  # 22 non-taro crops × 2 modes
    assert locks["ylx"] == 44
    assert ("wheat", "rainfed") in GAEZ_V5_YXX_SHA256
    assert ("wheat", "rainfed") in GAEZ_V5_YLX_SHA256


def test_raster_specs_include_ylx_density() -> None:
    specs = raster_specs(crops=tuple(c for c in CROPS if c.crop == "wheat"))
    variables = {spec.variable for spec in specs}
    assert variables == {"RES05-YXX", "RES05-YLX", "RES05-SX3"}
    assert any("YLX" in spec.filename for spec in specs)
    six = raster_specs(
        crops=tuple(c for c in CROPS if c.crop == "wheat"),
        variables=("RES05-SIX",),
    )
    assert len(six) == 2
    assert all(spec.variable == "RES05-SIX" for spec in six)


def test_wide_pivot_column_naming() -> None:
    long = pl.DataFrame(
        {
            "location_tag": ["a", "a", "b", "b"],
            "crop": ["wheat", "wheat", "wheat", "wheat"],
            "water_mode": ["rainfed", "irrigated", "rainfed", "irrigated"],
            "yield_kg_dm_suitable_km2": [1.0, 2.0, 3.0, 4.0],
            "production_density_kg_dm_total_km2": [10.0, 20.0, 30.0, 40.0],
            "suitable_fraction": [0.1, 0.2, 0.3, 0.4],
            "net_irrigation_requirement_mm": [0.0, 50.0, 0.0, 80.0],
            "crop_cycle_start_doy": [None, 60.0, None, 90.0],
            "crop_cycle_length_days": [None, 120.0, None, 140.0],
            "suitability_index": [4.0, 3.0, 5.0, 2.0],
        }
    )
    wide = pivot_location_metrics(long, crops=["wheat"])
    assert metric_column("wheat", "rainfed", "production_density_kg_dm_total_km2") in wide.columns
    assert metric_column("wheat", "irrigated", "yield_kg_dm_suitable_km2") in wide.columns
    assert metric_column("wheat", "irrigated", "net_irrigation_requirement_mm") in wide.columns
    assert metric_column("wheat", "irrigated", "suitability_index") in wide.columns
    row_a = wide.filter(pl.col("location_tag") == "a").to_dicts()[0]
    assert row_a["wheat_rainfed_production_density_kg_dm_total_km2"] == 10.0
    assert row_a["wheat_irrigated_suitable_fraction"] == 0.2
    assert row_a["wheat_irrigated_net_irrigation_requirement_mm"] == 50.0


def test_wide_from_labels_converts_ha_density_to_km2(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.crops import HECTARES_PER_KM2
    from prosper_or_perish_static_modifiers.wide import build_wide_from_labels

    geom = pl.DataFrame({"location_tag": ["a"], "pixel_count": [1]})
    geom_path = tmp_path / "geom.parquet"
    geom.write_parquet(geom_path)
    labels = pl.DataFrame(
        {
            "location_tag": ["a", "a"],
            "crop": ["wheat", "wheat"],
            "water_mode": ["rainfed", "irrigated"],
            "engine": ["gaez_v5", "gaez_v5"],
            "yield_kg_dm_ha": [1.0, 2.0],
            "production_density_kg_dm_total_ha": [3.0, 4.0],
            "suitable_fraction": [0.1, 0.2],
        }
    )
    labels_path = tmp_path / "labels.parquet"
    labels.write_parquet(labels_path)
    out = tmp_path / "wide.parquet"
    build_wide_from_labels(
        geometry_path=geom_path,
        labels_path=labels_path,
        output_path=out,
        crops=["wheat"],
    )
    wide = pl.read_parquet(out)
    assert wide["wheat_rainfed_production_density_kg_dm_total_km2"][0] == pytest.approx(
        3.0 * HECTARES_PER_KM2
    )
    assert wide["wheat_rainfed_yield_kg_dm_suitable_km2"][0] == pytest.approx(
        1.0 * HECTARES_PER_KM2
    )


def test_publish_pack_round_trip(tmp_path: Path) -> None:
    tags = ["loc_a", "loc_b"]
    data: dict[str, object] = {"location_tag": tags}
    for water in ("rainfed", "irrigated"):
        for metric in iter_metrics():
            col = metric_column("wheat", water, metric["suffix"])
            data[col] = [1.5, 0.0] if water == "rainfed" else [2.5, 1.0]
    wide = pl.DataFrame(data)
    wide_path = tmp_path / "wide.parquet"
    wide.write_parquet(wide_path)

    width, height = 4, 2
    id_map = np.array([1, 1, 0, 2, 2, 0, 0, 0], dtype=np.uint16)
    id_map_path = tmp_path / "location_id_map.bin.gz"
    with gzip.open(id_map_path, "wb") as handle:
        handle.write(id_map.tobytes())
    meta_path = tmp_path / "location_id_map.meta.json"
    meta_path.write_text(
        json.dumps({"width": width, "height": height, "dtype": "uint16"}),
        encoding="utf-8",
    )
    order_path = tmp_path / "location_row_order.json"
    order_path.write_text(json.dumps(tags), encoding="utf-8")

    docs_dir = tmp_path / "docs"
    publish_docs(
        wide_path=wide_path,
        location_id_map_path=id_map_path,
        location_id_meta_path=meta_path,
        location_row_order_path=order_path,
        docs_dir=docs_dir,
        crops=["wheat"],
    )

    meta = json.loads((docs_dir / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["default_metric"] == "production_density"
    assert meta["location_count"] == 2
    assert any(col.endswith("production_density_kg_dm_total_km2") for col in meta["attribute_columns"])
    assert {m["id"] for m in meta["metrics"]} >= {"irrigation_need", "cycle_start", "suitability_index"}
    assert all("group" in m and "zero_is_missing" in m and "water_modes" in m for m in meta["metrics"])
    irrig = next(m for m in meta["metrics"] if m["id"] == "irrigation_need")
    assert irrig["water_modes"] == ["irrigated"]

    with gzip.open(docs_dir / "data" / "attributes.bin.gz", "rb") as handle:
        attrs = np.frombuffer(handle.read(), dtype=np.float32)
    assert attrs.shape[0] == 2 * len(meta["attribute_columns"])
    assert (docs_dir / "index.html").is_file()


def test_publish_exploration_pack(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.external_layers import (
        EU5_LAYERS,
        EXPLORATION_LAYERS,
        PILOT_LAYERS,
    )

    tags = ["loc_a", "loc_b"]
    data: dict[str, object] = {"location_tag": tags}
    for water in ("rainfed", "irrigated"):
        for metric in iter_metrics():
            col = metric_column("wheat", water, metric["suffix"])
            data[col] = [1.0, 0.0]
    wide_path = tmp_path / "wide.parquet"
    pl.DataFrame(data).write_parquet(wide_path)

    ext = {"location_tag": tags}
    for layer in PILOT_LAYERS:
        ext[layer.layer_id] = [2.0, 0.5]
    ext_path = tmp_path / "external.parquet"
    pl.DataFrame(ext).write_parquet(ext_path)

    id_map_path = tmp_path / "location_id_map.bin.gz"
    with gzip.open(id_map_path, "wb") as handle:
        handle.write(np.array([1, 2], dtype=np.uint16).tobytes())
    meta_path = tmp_path / "location_id_map.meta.json"
    meta_path.write_text(json.dumps({"width": 2, "height": 1, "dtype": "uint16"}), encoding="utf-8")
    order_path = tmp_path / "location_row_order.json"
    order_path.write_text(json.dumps(tags), encoding="utf-8")

    docs_dir = tmp_path / "docs"
    publish_docs(
        wide_path=wide_path,
        location_id_map_path=id_map_path,
        location_id_meta_path=meta_path,
        location_row_order_path=order_path,
        docs_dir=docs_dir,
        crops=["wheat"],
        external_wide_path=ext_path,
    )
    meta = json.loads((docs_dir / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["exploration"] is not None
    assert meta["eu5"] is not None
    assert len(meta["exploration"]["layers"]) == len(EXPLORATION_LAYERS)
    assert len(meta["eu5"]["layers"]) == len(EU5_LAYERS)
    assert (docs_dir / "data" / "exploration_attributes.bin.gz").is_file()
    assert (docs_dir / "data" / "eu5_attributes.bin.gz").is_file()
    assert any((docs_dir / "data").glob("exploration_attributes.*.bin.gz"))
    assert any((docs_dir / "data").glob("eu5_attributes.*.bin.gz"))
    assert meta["eu5"]["assets"]["attributes"].startswith("data/eu5_attributes.")
    assert meta["eu5"]["assets"]["attributes"].endswith(".bin.gz")
    assert "eu5_development" in meta["eu5"]["attribute_columns"]
    assert "spam_cotton_rainfed_yield" in meta["exploration"]["attribute_columns"]
    assert "glw_cattle_density" in meta["exploration"]["attribute_columns"]
    assert "europe_ag_suitability_1500" in meta["exploration"]["attribute_columns"]
    assert "eu5_population_total" not in meta["exploration"]["attribute_columns"]
    assert "eu5_population_total" in meta["eu5"]["attribute_columns"]
    assert "eu5_population_density" in meta["eu5"]["attribute_columns"]
    assert "eu5_development" in meta["eu5"]["attribute_columns"]


def test_parse_eu5_start_population_scales_game_units(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.eu5_population import (
        PEOPLE_PER_GAME_POPULATION_UNIT,
        build_eu5_vanilla_layers,
        load_constructor_eu5_vanilla_frame,
    )

    source_path = tmp_path / "derived_food_balance_by_location.parquet"
    pl.DataFrame(
        {
            "slug": ["stockholm", "empty_lake"],
            "total_population": [21.816, 0.0],
            "development": [22.75, -1.0],
        }
    ).write_parquet(source_path)

    vanilla = load_constructor_eu5_vanilla_frame(source_path)
    stockholm = vanilla.filter(pl.col("location_tag") == "stockholm").to_dicts()[0]
    assert stockholm["eu5_population_total"] == pytest.approx(
        21.816 * PEOPLE_PER_GAME_POPULATION_UNIT
    )
    assert stockholm["eu5_development"] == pytest.approx(22.75)
    empty = vanilla.filter(pl.col("location_tag") == "empty_lake").to_dicts()[0]
    assert empty["eu5_population_total"] == 0.0
    assert empty["eu5_development"] == pytest.approx(-1.0)

    area_path = tmp_path / "area.parquet"
    pl.DataFrame(
        {
            "location_tag": ["stockholm", "empty_lake"],
            "area_jacobian_km2": [2000.0, 100.0],
        }
    ).write_parquet(area_path)
    wide = build_eu5_vanilla_layers(
        constructor_locations_path=source_path,
        location_area_path=area_path,
    )
    row = wide.filter(pl.col("location_tag") == "stockholm").to_dicts()[0]
    assert row["eu5_population_density"] == pytest.approx(
        (21.816 * PEOPLE_PER_GAME_POPULATION_UNIT) / 2000.0
    )
    assert row["eu5_development"] == pytest.approx(22.75)


def test_pilot_layer_catalog_covers_plan() -> None:
    from prosper_or_perish_static_modifiers.external_layers import PILOT_LAYERS

    ids = {layer.layer_id for layer in PILOT_LAYERS}
    assert "spam_wheat_rainfed_yield" in ids
    assert "spam_maize_rainfed_yield" in ids
    assert "spam_cotton_rainfed_yield" in ids
    assert "glw_cattle_density" in ids
    assert "glw_sheep_density" in ids
    assert "europe_ag_suitability_1500" in ids
    assert "worldpop_pop_density_2020" in ids
    assert "hyde_pop_density_1300" in ids
    assert "hyde_pop_density_1400" in ids
    assert "hyde_pop_density_1500" in ids
    assert "eu5_population_total" in ids
    assert "eu5_population_density" in ids
    assert "eu5_development" in ids
    groups = {layer.group for layer in PILOT_LAYERS}
    assert groups == {
        "MapSPAM observed",
        "Livestock GLW",
        "Historical Europe",
        "Modern population",
        "Historical population",
        "Vanilla start",
    }
    from prosper_or_perish_static_modifiers.external_layers import (
        EU5_LAYERS,
        EXPLORATION_LAYERS,
        PNP_LAYERS,
    )

    assert {layer.layer_id for layer in EU5_LAYERS} == {
        "eu5_population_total",
        "eu5_population_density",
        "eu5_development",
    }
    assert all(layer.source != "eu5" for layer in EXPLORATION_LAYERS)
    assert {layer.layer_id for layer in PNP_LAYERS} == {
        "pnp_wheat_production_density",
        "pnp_wheat_yield",
        "pnp_wheat_suitable_fraction",
        "pnp_wheat_suitability_class",
    }


def test_evidence_catalog_and_soft_scaling(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.pnp_evidence import (
        apply_evidence_target_scales,
        load_evidence_catalog,
    )

    catalog = load_evidence_catalog()
    assert catalog.good == "wheat"
    assert catalog.global_anchor["target_positive_median_kg_ha"] == 515.0
    ids = {e["id"] for e in catalog.evidence}
    assert "egypt_nile_mamluk" in ids
    assert "indonesia_near_zero" in ids

    frame = pl.DataFrame(
        {
            "location_tag": ["a", "b", "c", "d", "e", "f"],
            "region": [
                "egypt_region",
                "egypt_region",
                "egypt_region",
                "egypt_region",
                "egypt_region",
                "egypt_region",
            ],
            "target_yield_kg_ha": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "target_production_density_kg_ha": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
        }
    )
    out, meta = apply_evidence_target_scales(frame, catalog)
    assert meta["n_touched"] == 6
    assert float(out["target_yield_kg_ha"].mean()) > float(frame["target_yield_kg_ha"].mean())
    assert "evidence_target_scale" in out.columns


def test_suitability_class_and_pnp_publish(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.external_layers import PNP_LAYERS
    from prosper_or_perish_static_modifiers.pnp_model import suitability_class_from_fraction

    classes = suitability_class_from_fraction([1.0, 0.5, 0.0])
    assert classes[0] == 1.0
    assert classes[2] == 9.0
    assert 4.0 <= classes[1] <= 6.0

    tags = ["loc_a", "loc_b"]
    data: dict[str, object] = {"location_tag": tags}
    for water in ("rainfed", "irrigated"):
        for metric in iter_metrics():
            col = metric_column("wheat", water, metric["suffix"])
            data[col] = [1.0, 0.0]
    wide_path = tmp_path / "wide.parquet"
    pl.DataFrame(data).write_parquet(wide_path)

    pnp = {
        "location_tag": tags,
        "pnp_wheat_production_density": [1000.0, 0.0],
        "pnp_wheat_yield": [2000.0, 0.0],
        "pnp_wheat_suitable_fraction": [0.5, 0.0],
        "pnp_wheat_suitability_class": [5.0, 9.0],
    }
    pnp_path = tmp_path / "pnp.parquet"
    pl.DataFrame(pnp).write_parquet(pnp_path)

    id_map_path = tmp_path / "location_id_map.bin.gz"
    with gzip.open(id_map_path, "wb") as handle:
        handle.write(np.array([1, 2], dtype=np.uint16).tobytes())
    meta_path = tmp_path / "location_id_map.meta.json"
    meta_path.write_text(json.dumps({"width": 2, "height": 1, "dtype": "uint16"}), encoding="utf-8")
    order_path = tmp_path / "location_row_order.json"
    order_path.write_text(json.dumps(tags), encoding="utf-8")

    docs_dir = tmp_path / "docs"
    publish_docs(
        wide_path=wide_path,
        location_id_map_path=id_map_path,
        location_id_meta_path=meta_path,
        location_row_order_path=order_path,
        docs_dir=docs_dir,
        crops=["wheat"],
        pnp_wide_path=pnp_path,
    )
    meta = json.loads((docs_dir / "data" / "meta.json").read_text(encoding="utf-8"))
    assert meta["pnp"] is not None
    assert len(meta["pnp"]["layers"]) == len(PNP_LAYERS)
    assert meta["pnp"]["goods"] == ["wheat"]
    assert "pnp_wheat_yield" in meta["pnp"]["attribute_columns"]
    assert any((docs_dir / "data").glob("pnp_attributes.*.bin.gz"))
    assert "P&P" in (docs_dir / "index.html").read_text(encoding="utf-8")
