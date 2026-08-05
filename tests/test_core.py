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
    from prosper_or_perish_static_modifiers.external_layers import PILOT_LAYERS

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
    assert len(meta["exploration"]["layers"]) == len(PILOT_LAYERS)
    assert (docs_dir / "data" / "exploration_attributes.bin.gz").is_file()
    assert "spam_cotton_rainfed_yield" in meta["exploration"]["attribute_columns"]
    assert "glw_cattle_density" in meta["exploration"]["attribute_columns"]
    assert "europe_ag_suitability_1500" in meta["exploration"]["attribute_columns"]
    assert "eu5_population_total" in meta["exploration"]["attribute_columns"]
    assert "eu5_population_density" in meta["exploration"]["attribute_columns"]


def test_parse_eu5_start_population_scales_game_units(tmp_path: Path) -> None:
    from prosper_or_perish_static_modifiers.eu5_population import (
        PEOPLE_PER_GAME_POPULATION_UNIT,
        build_eu5_population_layers,
        parse_eu5_start_population,
    )

    pops_path = tmp_path / "06_pops.txt"
    pops_path.write_text(
        """
locations={
stockholm = {
	define_pop = {	type = burghers	size = 1.5	culture = swedish	religion = catholic }
	define_pop = {	type = peasants	size = 20.316	culture = swedish	religion = catholic }
}
empty_lake = {
}
}
""",
        encoding="utf-8",
    )
    pops = parse_eu5_start_population(pops_path)
    stockholm = pops.filter(pl.col("location_tag") == "stockholm").to_dicts()[0]
    assert stockholm["eu5_population_total"] == pytest.approx(
        (1.5 + 20.316) * PEOPLE_PER_GAME_POPULATION_UNIT
    )
    empty = pops.filter(pl.col("location_tag") == "empty_lake").to_dicts()[0]
    assert empty["eu5_population_total"] == 0.0

    area_path = tmp_path / "area.parquet"
    pl.DataFrame(
        {
            "location_tag": ["stockholm", "empty_lake"],
            "area_jacobian_km2": [2000.0, 100.0],
        }
    ).write_parquet(area_path)
    wide = build_eu5_population_layers(
        start_pops_path=pops_path,
        location_area_path=area_path,
    )
    row = wide.filter(pl.col("location_tag") == "stockholm").to_dicts()[0]
    assert row["eu5_population_density"] == pytest.approx(
        ((1.5 + 20.316) * PEOPLE_PER_GAME_POPULATION_UNIT) / 2000.0
    )


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
    groups = {layer.group for layer in PILOT_LAYERS}
    assert groups == {
        "MapSPAM observed",
        "Livestock GLW",
        "Historical Europe",
        "Modern population",
        "Historical population",
        "EU5 population",
    }
