from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from prosper_or_perish_static_modifiers.external_layers import (
    PILOT_LAYERS,
    glw_density_relpath,
    spam_tif_name,
    spam_zip_name,
)


USER_AGENT = (
    "ProsperOrPerishStaticModifiers/0.1 "
    "(+https://github.com/JanB1989/ProsperOrPerishStaticModifiers)"
)

# Harvard Dataverse file ids for MapSPAM 2010 v2.0 GeoTIFF bundles (doi:10.7910/DVN/PRFF8V).
# Note: MapSPAM 2020 (SWPENT) requires an interactive Dataverse guestbook response;
# 2010 is used for the automated pilot and remains GAEZ-comparable observed production.
SPAM_DATAVERSE_FILE_IDS = {
    "Y": 3985012,  # yield geotiff zip
    "H": 3985008,  # harvested area geotiff zip
}

EUROPE_SUIT_DATAVERSE_FILE_ID = 10695119  # doi:10.7910/DVN/ECWMZS suit.tif

GLW_GCS_BASE = "https://storage.googleapis.com/fao-gismgr-glw-data"


def _download(url: str, target: Path) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {"url": url, "path": str(target), "status": "cached"}
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=600) as response:
        data = response.read()
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_bytes(data)
    tmp.replace(target)
    return {"url": url, "path": str(target), "status": "downloaded", "bytes": len(data)}


def _dataverse_file_url(file_id: int) -> str:
    return f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"


def _extract_spam_members(zip_path: Path, dest_dir: Path, members: list[str]) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        tif_names = [n for n in names if n.lower().endswith(".tif")]
        for member in members:
            preferred = Path(member).name
            hits = [n for n in tif_names if Path(n).name.lower() == preferred.lower()]
            if not hits:
                # spam2010V2r0_global_Y_WHEA_R.tif → tokens Y, WHEA, R
                stem = Path(member).stem
                parts = stem.split("_")
                crop = parts[-2]
                system = parts[-1]
                variable = parts[-3]
                hits = [
                    n
                    for n in tif_names
                    if crop in Path(n).name
                    and Path(n).stem.upper().endswith(f"_{system.upper()}")
                    and f"_{variable.upper()}_" in Path(n).stem.upper()
                ]
            if not hits:
                sample = [Path(n).name for n in tif_names[:12]]
                raise FileNotFoundError(
                    f"{member} not found in {zip_path.name}; sample={sample}"
                )
            name = hits[0]
            target = dest_dir / Path(name).name
            if not target.is_file():
                target.write_bytes(archive.read(name))
            extracted.append(str(target))
    return extracted


def fetch_mapspam_pilot(cache_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    needed: dict[str, set[str]] = {}
    for layer in PILOT_LAYERS:
        if layer.source != "spam":
            continue
        assert layer.spam_variable and layer.spam_crop_code and layer.spam_system
        needed.setdefault(layer.spam_variable, set()).add(
            spam_tif_name(layer.spam_variable, layer.spam_crop_code, layer.spam_system)
        )

    for variable, members in needed.items():
        file_id = SPAM_DATAVERSE_FILE_IDS[variable]
        zip_path = cache_dir / "mapspam" / "zips" / spam_zip_name(variable)
        rows.append(_download(_dataverse_file_url(file_id), zip_path))
        dest = cache_dir / "mapspam" / variable
        extracted = _extract_spam_members(zip_path, dest, sorted(members))
        rows.append(
            {
                "source": f"mapspam_{variable}",
                "status": "extracted",
                "files": extracted,
            }
        )
    return rows


def fetch_glw_pilot(cache_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dest_dir = cache_dir / "glw"
    for layer in PILOT_LAYERS:
        if layer.source != "glw":
            continue
        assert layer.glw_species_code
        rel = glw_density_relpath(layer.glw_species_code)
        url = f"{GLW_GCS_BASE}/{rel}"
        target = dest_dir / Path(rel).name
        rows.append(_download(url, target))
    return rows


def fetch_europe_suit_pilot(cache_dir: Path) -> list[dict[str, object]]:
    target = cache_dir / "europe_suit" / "suit.tif"
    return [
        _download(
            _dataverse_file_url(EUROPE_SUIT_DATAVERSE_FILE_ID),
            target,
        )
    ]


def fetch_external_pilots(cache_dir: Path) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(fetch_mapspam_pilot(cache_dir))
    rows.extend(fetch_glw_pilot(cache_dir))
    rows.extend(fetch_europe_suit_pilot(cache_dir))
    manifest = {
        "engine": "external_pilots",
        "layer_count": len(PILOT_LAYERS),
        "layers": [layer.layer_id for layer in PILOT_LAYERS],
        "sources": rows,
    }
    path = cache_dir / "external_source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
