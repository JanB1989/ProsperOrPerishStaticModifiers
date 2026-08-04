from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from prosper_or_perish_static_modifiers.crops import RasterSpec, raster_specs, selected_crops


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _download(
    url: str,
    target: Path,
    *,
    expected_sha256: str,
    source_id: str,
) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        digest = sha256_file(target)
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(
                f"cached {target.name} sha256 mismatch: got {digest}, expected {expected_sha256}"
            )
        return {
            "source": source_id,
            "url": url,
            "path": str(target),
            "status": "cached",
            "sha256": digest,
        }

    tmp = target.with_suffix(target.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)
    digest = sha256_file(tmp)
    if expected_sha256 and digest != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"download {target.name} sha256 mismatch: got {digest}, expected {expected_sha256}"
        )
    tmp.replace(target)
    return {
        "source": source_id,
        "url": url,
        "path": str(target),
        "status": "downloaded",
        "sha256": digest,
    }


def fetch_gaez(
    cache_dir: Path,
    *,
    crops: list[str] | None = None,
    water_modes: tuple[str, ...] | list[str] = ("rainfed", "irrigated"),
    variables: tuple[str, ...] | None = None,
) -> dict[str, object]:
    crop_defs = selected_crops(crops)
    kwargs = {"crops": crop_defs, "water_modes": tuple(water_modes)}
    if variables is not None:
        kwargs["variables"] = tuple(variables)
    specs = raster_specs(**kwargs)
    rows: list[dict[str, object]] = []
    for spec in specs:
        source_id = (
            f"gaez_v5_{spec.variable.removeprefix('RES05-').lower()}"
            f"_{spec.crop}_{spec.crop_variant}_{spec.water_mode}"
        )
        rows.append(
            _download(
                spec.url,
                cache_dir / spec.cache_relpath,
                expected_sha256=spec.expected_sha256,
                source_id=source_id,
            )
        )
    variables = sorted({spec.variable for spec in specs})
    manifest = {
        "engine": "gaez_v5",
        "period": "HP8100 1981-2000",
        "variables": variables,
        "source_count": len(rows),
        "sources": rows,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "source_manifest.json"
    if manifest_path.is_file() and variables != ["RES05-SX3", "RES05-YLX", "RES05-YXX"]:
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        prev_sources = {
            (row.get("source"), row.get("path")): row
            for row in previous.get("sources", [])
            if isinstance(row, dict)
        }
        for row in rows:
            prev_sources[(row.get("source"), row.get("path"))] = row
        merged = list(prev_sources.values())
        manifest = {
            "engine": "gaez_v5",
            "period": "HP8100 1981-2000",
            "variables": sorted(
                set(previous.get("variables", [])) | set(variables)
            ),
            "source_count": len(merged),
            "sources": merged,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def resolve_raster_path(cache_dir: Path, spec: RasterSpec) -> Path:
    path = cache_dir / spec.cache_relpath
    if not path.is_file():
        raise FileNotFoundError(
            f"missing GAEZ raster {path}; run `uv run posm fetch-gaez` first"
        )
    if spec.expected_sha256:
        digest = sha256_file(path)
        if digest != spec.expected_sha256:
            raise ValueError(
                f"raster {path.name} sha256 mismatch: got {digest}, expected {spec.expected_sha256}"
            )
    return path
