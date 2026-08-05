from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl

from prosper_or_perish_static_modifiers.crops import (
    DEFAULT_METRIC,
    WATER_MODES,
    iter_metrics,
    metric_column,
    selected_crops,
)
from prosper_or_perish_static_modifiers.external_layers import (
    EU5_LAYERS,
    EXPLORATION_LAYERS,
    PNP_LAYERS,
)
from prosper_or_perish_static_modifiers.geometry import LOCATION_TAG


def load_viewer_html(*, asset_version: str | None = None) -> str:
    html = (Path(__file__).with_name("viewer.html")).read_text(encoding="utf-8")
    version = asset_version or str(int(time.time()))
    return html.replace("__ASSET_VERSION__", version)


def _finite_or_nan(values: list[float | None]) -> np.ndarray:
    out = np.empty(len(values), dtype=np.float32)
    for i, value in enumerate(values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out[i] = np.nan
        else:
            out[i] = float(value)
    return out


def _pack_layers_from_frame(
    *,
    ordered: pl.DataFrame,
    layers: tuple,
    asset_stem: str,
    data_dir: Path,
) -> dict[str, object]:
    columns = [layer.layer_id for layer in layers]
    missing = [col for col in columns if col not in ordered.columns]
    if missing:
        raise ValueError(f"wide missing columns: {missing[:8]}")
    packs = [
        _finite_or_nan(ordered.get_column(column).to_list()) for column in columns
    ]
    attributes = np.concatenate(packs).astype(np.float32, copy=False)
    digest = hashlib.sha256(attributes.tobytes()).hexdigest()[:12]
    asset_name = f"{asset_stem}.{digest}.bin.gz"
    path = data_dir / asset_name
    with gzip.open(path, "wb") as handle:
        handle.write(attributes.tobytes())
    alias = data_dir / f"{asset_stem}.bin.gz"
    alias.write_bytes(path.read_bytes())
    return {
        "default_layer": columns[0],
        "attribute_columns": columns,
        "layers": [
            {
                "id": layer.layer_id,
                "label": layer.label,
                "group": layer.group,
                "unit": layer.unit,
                "zero_is_missing": layer.zero_is_missing,
                "eu5_goods": list(layer.eu5_goods),
            }
            for layer in layers
        ],
        "goods": sorted({g for layer in layers for g in layer.eu5_goods}),
        "assets": {"attributes": f"data/{asset_name}"},
    }


def publish_docs(
    *,
    wide_path: Path,
    location_id_map_path: Path,
    location_id_meta_path: Path,
    location_row_order_path: Path,
    docs_dir: Path,
    crops: list[str] | None = None,
    water_modes: list[str] | tuple[str, ...] = WATER_MODES,
    external_wide_path: Path | None = None,
    pnp_wide_path: Path | None = None,
) -> Path:
    wide = pl.read_parquet(wide_path)
    order = json.loads(location_row_order_path.read_text(encoding="utf-8"))
    id_meta = json.loads(location_id_meta_path.read_text(encoding="utf-8"))

    ordered = (
        pl.DataFrame({LOCATION_TAG: order})
        .join(wide, on=LOCATION_TAG, how="left")
    )
    if ordered.height != len(order):
        raise ValueError("wide dataframe does not cover all location id-map rows")

    crop_defs = selected_crops(crops)
    columns: list[str] = []
    for crop in crop_defs:
        for water_mode in water_modes:
            for metric in iter_metrics():
                columns.append(metric_column(crop.crop, water_mode, metric["suffix"]))

    missing = [col for col in columns if col not in ordered.columns]
    if missing:
        raise ValueError(f"wide dataframe missing columns: {missing[:8]}")

    packs: list[np.ndarray] = []
    for column in columns:
        packs.append(_finite_or_nan(ordered.get_column(column).to_list()))
    attributes = np.concatenate(packs).astype(np.float32, copy=False)

    data_dir = docs_dir / "data"
    assets_dir = docs_dir / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    attr_path = data_dir / "attributes.bin.gz"
    with gzip.open(attr_path, "wb") as handle:
        handle.write(attributes.tobytes())

    loc_path = data_dir / "locations.json.gz"
    with gzip.open(loc_path, "wb") as handle:
        handle.write(json.dumps(order).encode("utf-8"))

    # Copy id map into docs assets.
    docs_id_map = assets_dir / "location_id_map.bin.gz"
    docs_id_map.write_bytes(location_id_map_path.read_bytes())

    exploration = None
    eu5 = None
    pnp = None
    if external_wide_path is not None and external_wide_path.is_file():
        external = pl.read_parquet(external_wide_path)
        ordered_ext = (
            pl.DataFrame({LOCATION_TAG: order})
            .join(external, on=LOCATION_TAG, how="left")
        )
        if EXPLORATION_LAYERS:
            exploration = _pack_layers_from_frame(
                ordered=ordered_ext,
                layers=EXPLORATION_LAYERS,
                asset_stem="exploration_attributes",
                data_dir=data_dir,
            )
        if EU5_LAYERS:
            eu5 = _pack_layers_from_frame(
                ordered=ordered_ext,
                layers=EU5_LAYERS,
                asset_stem="eu5_attributes",
                data_dir=data_dir,
            )

    if pnp_wide_path is not None and pnp_wide_path.is_file() and PNP_LAYERS:
        pnp_frame = pl.read_parquet(pnp_wide_path)
        ordered_pnp = (
            pl.DataFrame({LOCATION_TAG: order})
            .join(pnp_frame, on=LOCATION_TAG, how="left")
        )
        pnp = _pack_layers_from_frame(
            ordered=ordered_pnp,
            layers=PNP_LAYERS,
            asset_stem="pnp_attributes",
            data_dir=data_dir,
        )

    meta = {
        "title": "GAEZ Crop Mapmodes",
        "default_metric": DEFAULT_METRIC,
        "location_count": len(order),
        "attribute_columns": columns,
        "crops": [{"id": crop.crop, "label": crop.label} for crop in crop_defs],
        "water_modes": list(water_modes),
        "metrics": [
            {
                "id": metric["id"],
                "suffix": metric["suffix"],
                "label": metric["label"],
                "unit": metric["unit"],
                "group": metric["group"],
                "zero_is_missing": metric["zero_is_missing"],
                "water_modes": metric["water_modes"],
            }
            for metric in iter_metrics()
        ],
        "map": {
            "width": id_meta["width"],
            "height": id_meta["height"],
            "dtype": id_meta["dtype"],
        },
        "assets": {
            "location_id_map": "assets/location_id_map.bin.gz",
            "attributes": "data/attributes.bin.gz",
            "locations": "data/locations.json.gz",
        },
        "climate_scope": "GAEZ v5 RES05 HP8100 (1981-2000) modern diagnostic",
        "exploration": exploration,
        "eu5": eu5,
        "pnp": pnp,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (docs_dir / "index.html").write_text(
        load_viewer_html(asset_version=str(int(time.time()))),
        encoding="utf-8",
    )
    return docs_dir / "index.html"
