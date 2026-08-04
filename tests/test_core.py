from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

from prosper_or_perish_static_modifiers.crops import (
    CROPS,
    GAEZ_V5_YLX_SHA256,
    GAEZ_V5_YXX_SHA256,
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


def test_wide_pivot_column_naming() -> None:
    long = pl.DataFrame(
        {
            "location_tag": ["a", "a", "b", "b"],
            "crop": ["wheat", "wheat", "wheat", "wheat"],
            "water_mode": ["rainfed", "irrigated", "rainfed", "irrigated"],
            "yield_kg_dm_ha": [1.0, 2.0, 3.0, 4.0],
            "production_density_kg_dm_total_ha": [10.0, 20.0, 30.0, 40.0],
            "suitable_fraction": [0.1, 0.2, 0.3, 0.4],
        }
    )
    wide = pivot_location_metrics(long, crops=["wheat"])
    assert metric_column("wheat", "rainfed", "production_density_kg_dm_total_ha") in wide.columns
    assert metric_column("wheat", "irrigated", "yield_kg_dm_ha") in wide.columns
    row_a = wide.filter(pl.col("location_tag") == "a").to_dicts()[0]
    assert row_a["wheat_rainfed_production_density_kg_dm_total_ha"] == 10.0
    assert row_a["wheat_irrigated_suitable_fraction"] == 0.2


def test_publish_pack_round_trip(tmp_path: Path) -> None:
    tags = ["loc_a", "loc_b"]
    wide = pl.DataFrame(
        {
            "location_tag": tags,
            "wheat_rainfed_production_density_kg_dm_total_ha": [1.5, 0.0],
            "wheat_rainfed_yield_kg_dm_ha": [100.0, 0.0],
            "wheat_rainfed_suitable_fraction": [0.25, 0.0],
            "wheat_irrigated_production_density_kg_dm_total_ha": [2.5, 1.0],
            "wheat_irrigated_yield_kg_dm_ha": [120.0, 50.0],
            "wheat_irrigated_suitable_fraction": [0.4, 0.1],
        }
    )
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
    assert any(col.endswith("production_density_kg_dm_total_ha") for col in meta["attribute_columns"])

    with gzip.open(docs_dir / "data" / "attributes.bin.gz", "rb") as handle:
        attrs = np.frombuffer(handle.read(), dtype=np.float32)
    assert attrs.shape[0] == 2 * len(meta["attribute_columns"])
    assert (docs_dir / "index.html").is_file()
