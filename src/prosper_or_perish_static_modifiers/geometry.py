from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

LOCATION_TAG = "location_tag"
LOCATION_HEX = "location_hex"


@dataclass
class PixelStats:
    color: int
    pixel_count: int
    sum_x: float
    sum_y: float
    min_x: int
    max_x: int
    min_y: int
    max_y: int

    @property
    def centroid_x(self) -> float:
        return self.sum_x / self.pixel_count

    @property
    def centroid_y(self) -> float:
        return self.sum_y / self.pixel_count


def _hex_to_rgb_int(value: str) -> int:
    text = value.lower().removeprefix("#")
    if not text or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"invalid location hex: {value}")
    if len(text) > 6:
        raise ValueError(f"invalid location hex: {value}")
    text = text.zfill(6)
    return int(text, 16)


def _rgb_int_to_hex(color: int) -> str:
    return f"#{color:06x}"


def scan_location_pixels(
    locations_png_path: Path,
    *,
    target_colors: set[int],
    chunk_rows: int = 128,
) -> dict[int, PixelStats]:
    Image.MAX_IMAGE_PIXELS = None
    target_set = set(target_colors)
    accum: dict[int, dict[str, float | int]] = {}
    with Image.open(locations_png_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        for y0 in range(0, height, chunk_rows):
            y1 = min(height, y0 + chunk_rows)
            chunk = np.asarray(image.crop((0, y0, width, y1)), dtype=np.uint32)
            packed = (chunk[:, :, 0] << 16) | (chunk[:, :, 1] << 8) | chunk[:, :, 2]
            flat = packed.reshape(-1)
            unique, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
            for idx, color in enumerate(unique.tolist()):
                color_i = int(color)
                if color_i not in target_set:
                    continue
                mask = inverse == idx
                indices = np.flatnonzero(mask)
                xs = indices % width
                ys = indices // width + y0
                entry = accum.get(color_i)
                if entry is None:
                    accum[color_i] = {
                        "pixel_count": int(counts[idx]),
                        "sum_x": float(xs.sum()),
                        "sum_y": float(ys.sum()),
                        "min_x": int(xs.min()),
                        "max_x": int(xs.max()),
                        "min_y": int(ys.min()),
                        "max_y": int(ys.max()),
                    }
                else:
                    entry["pixel_count"] = int(entry["pixel_count"]) + int(counts[idx])
                    entry["sum_x"] = float(entry["sum_x"]) + float(xs.sum())
                    entry["sum_y"] = float(entry["sum_y"]) + float(ys.sum())
                    entry["min_x"] = min(int(entry["min_x"]), int(xs.min()))
                    entry["max_x"] = max(int(entry["max_x"]), int(xs.max()))
                    entry["min_y"] = min(int(entry["min_y"]), int(ys.min()))
                    entry["max_y"] = max(int(entry["max_y"]), int(ys.max()))
    return {
        color: PixelStats(
            color=color,
            pixel_count=int(values["pixel_count"]),
            sum_x=float(values["sum_x"]),
            sum_y=float(values["sum_y"]),
            min_x=int(values["min_x"]),
            max_x=int(values["max_x"]),
            min_y=int(values["min_y"]),
            max_y=int(values["max_y"]),
        )
        for color, values in accum.items()
    }



def build_location_geometry(
    *,
    baseline_path: Path,
    locations_png_path: Path,
    equator_y: int,
) -> pl.DataFrame:
    baseline = pl.read_parquet(baseline_path)
    if LOCATION_HEX not in baseline.columns:
        for alias in ("named_location_hex", "map_color_rgb", "location_color_hex"):
            if alias in baseline.columns:
                baseline = baseline.with_columns(
                    pl.col(alias).cast(pl.Utf8).str.to_lowercase().alias(LOCATION_HEX)
                )
                break
    required = {LOCATION_TAG, LOCATION_HEX}
    missing = required - set(baseline.columns)
    if missing:
        raise ValueError(f"baseline missing columns: {sorted(missing)}")

    location_rows = baseline.filter(pl.col(LOCATION_HEX).is_not_null())
    colors_by_hex = {
        str(row[LOCATION_HEX]).lower().removeprefix("#"): _hex_to_rgb_int(
            str(row[LOCATION_HEX])
        )
        for row in location_rows.select(LOCATION_HEX).unique().to_dicts()
    }
    stats_by_color = scan_location_pixels(
        locations_png_path,
        target_colors=set(colors_by_hex.values()),
    )
    with Image.open(locations_png_path) as image:
        width, _height = image.size

    geometry_rows: list[dict[str, object]] = []
    for hex_value, color in colors_by_hex.items():
        stats = stats_by_color.get(color)
        if stats is None:
            geometry_rows.append(
                {
                    LOCATION_HEX: hex_value,
                    "map_color_int": color,
                    "map_color_rgb": _rgb_int_to_hex(color),
                    "geometry_status": "missing_color",
                }
            )
            continue
        lon = (stats.centroid_x / width * 360.0) - 180.0
        lat = (float(equator_y) - stats.centroid_y) / (width / 360.0)
        geometry_rows.append(
            {
                LOCATION_HEX: hex_value,
                "map_color_int": color,
                "map_color_rgb": _rgb_int_to_hex(color),
                "geometry_status": "ok",
                "pixel_count": stats.pixel_count,
                "centroid_x": stats.centroid_x,
                "centroid_y": stats.centroid_y,
                "approx_lon": max(-180.0, min(180.0, lon)),
                "approx_lat": max(-90.0, min(90.0, lat)),
            }
        )

    geometry = pl.DataFrame(geometry_rows)
    return (
        baseline.with_columns(pl.col(LOCATION_HEX).str.to_lowercase().str.strip_prefix("#"))
        .join(geometry, on=LOCATION_HEX, how="left")
        .sort(LOCATION_TAG)
    )


def build_location_id_map(
    *,
    geometry: pl.DataFrame,
    locations_png_path: Path,
    output_bin_gz: Path,
    output_meta: Path,
) -> dict[str, object]:
    """Pack a pixel→row-index map. Index 0 = sea/unmapped; rows are 1-based."""

    ordered = (
        geometry.filter(pl.col("geometry_status") == "ok")
        .select(LOCATION_TAG, "map_color_int")
        .unique(subset=[LOCATION_TAG], keep="first")
        .sort(LOCATION_TAG)
        .with_row_index("row_index", offset=1)
    )
    color_to_row = {
        int(row["map_color_int"]): int(row["row_index"])
        for row in ordered.select("map_color_int", "row_index").to_dicts()
    }
    tags = ordered.get_column(LOCATION_TAG).to_list()

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(locations_png_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        # Downscale for Pages size if huge; keep native for correctness unless > 8M px.
        scale = 1
        if width * height > 8_000_000:
            scale = 2
        if scale > 1:
            image = image.resize((width // scale, height // scale), Image.NEAREST)
            width, height = image.size
        rgb = np.asarray(image, dtype=np.uint32)
        packed = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]

    id_map = np.zeros(width * height, dtype=np.uint16)
    flat = packed.reshape(-1)
    # Vectorized fill via unique colors present.
    unique_colors = np.unique(flat)
    for color in unique_colors.tolist():
        row = color_to_row.get(int(color))
        if row is None:
            continue
        id_map[flat == color] = row

    output_bin_gz.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_bin_gz, "wb") as handle:
        handle.write(id_map.tobytes())

    meta = {
        "width": width,
        "height": height,
        "dtype": "uint16",
        "scale": scale,
        "location_count": len(tags),
        "sea_index": 0,
        "row_offset": 1,
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Persist row order alongside geometry artifact consumers.
    order_path = output_bin_gz.with_name("location_row_order.json")
    order_path.write_text(json.dumps(tags), encoding="utf-8")
    return {**meta, "tags": tags, "order_path": str(order_path)}
