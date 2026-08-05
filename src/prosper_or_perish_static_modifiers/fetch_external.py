from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from rasterio.crs import CRS
from zipfile_deflate64.deflate64 import Deflate64

from prosper_or_perish_static_modifiers.external_layers import (
    PILOT_LAYERS,
    glw_density_relpath,
    hyde_popd_member,
    hyde_popd_tif_name,
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

# WorldPop unconstrained global mosaic (people per 1 km pixel), Global1 archive.
WORLDPOP_2020_1KM_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/0_Mosaicked/"
    "ppp_2020_1km_Aggregated.tif"
)

# HYDE 3.2.1 baseline zip on DANS (doi:10.17026/dans-25g-gez3); supports HTTP Range.
HYDE_DANS_FILE_ID = 5490328
HYDE_BASELINE_URL = (
    f"https://archaeology.datastations.nl/api/access/datafile/{HYDE_DANS_FILE_ID}"
)


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


def _download_streaming(url: str, target: Path) -> dict[str, object]:
    """Stream large files to disk (WorldPop ~870 MB)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {"url": url, "path": str(target), "status": "cached"}
    req = Request(url, headers={"User-Agent": USER_AGENT})
    tmp = target.with_suffix(target.suffix + ".partial")
    total = 0
    with urlopen(req, timeout=600) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
    tmp.replace(target)
    return {"url": url, "path": str(target), "status": "downloaded", "bytes": total}


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


def fetch_worldpop_pilot(cache_dir: Path) -> list[dict[str, object]]:
    if not any(layer.source == "worldpop" for layer in PILOT_LAYERS):
        return []
    target = cache_dir / "worldpop" / "ppp_2020_1km_Aggregated.tif"
    return [_download_streaming(WORLDPOP_2020_1KM_URL, target)]


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": f"bytes={start}-{end}",
        },
    )
    with urlopen(req, timeout=300) as response:
        return response.read()


def _hyde_content_length(url: str) -> int:
    # Some Dataverse redirects reject HEAD; probe with a 1-byte Range GET instead.
    req = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Range": "bytes=0-0",
        },
    )
    with urlopen(req, timeout=120) as response:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            total = content_range.rsplit("/", 1)[-1]
            if total.isdigit():
                return int(total)
        length = response.headers.get("Content-Length")
        if length and length.isdigit():
            # Fallback if the server ignored Range.
            return int(length)
        raise RuntimeError("HYDE baseline zip size could not be determined")


def _parse_zip64_central_directory(
    url: str, size: int
) -> dict[str, tuple[int, int, int, int]]:
    """Return member_name -> (local_header_offset, compressed_size, uncompressed_size, method)."""
    tail = _fetch_range(url, max(0, size - 120_000), size - 1)
    loc_idx = tail.rfind(b"PK\x06\x07")
    if loc_idx < 0:
        raise RuntimeError("ZIP64 end-of-central-directory locator not found")
    _disk_cd, z64_eocd_offset, _disks = struct.unpack_from("<IQI", tail, loc_idx + 4)
    z64 = _fetch_range(url, z64_eocd_offset, z64_eocd_offset + 100)
    if z64[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 EOCD signature missing")
    (
        _z64_size,
        _ver_made,
        _ver_need,
        _disk,
        _disk_cd2,
        _entries_here,
        _entries_total,
        cd_size,
        cd_offset,
    ) = struct.unpack_from("<QHHIIQQQQ", z64, 4)
    cd = _fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    members: dict[str, tuple[int, int, int, int]] = {}
    off = 0
    while off + 46 <= len(cd):
        if cd[off : off + 4] != b"PK\x01\x02":
            break
        (
            _ver_made,
            _ver_need,
            _flag,
            method,
            _time,
            _date,
            _crc,
            csize,
            usize,
            namelen,
            extralen,
            commentlen,
            _disk_start,
            _int_attr,
            _ext_attr,
            local_off,
        ) = struct.unpack_from("<HHHHHHIIIHHHHHII", cd, off + 4)
        name = cd[off + 46 : off + 46 + namelen].decode("utf-8", "replace")
        extra = cd[off + 46 + namelen : off + 46 + namelen + extralen]
        real_csize, real_usize, real_local = csize, usize, local_off
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or local_off == 0xFFFFFFFF:
            eoff = 0
            while eoff + 4 <= len(extra):
                eid, esz = struct.unpack_from("<HH", extra, eoff)
                edata = extra[eoff + 4 : eoff + 4 + esz]
                pos = 0
                if eid == 1:
                    if usize == 0xFFFFFFFF:
                        real_usize = struct.unpack_from("<Q", edata, pos)[0]
                        pos += 8
                    if csize == 0xFFFFFFFF:
                        real_csize = struct.unpack_from("<Q", edata, pos)[0]
                        pos += 8
                    if local_off == 0xFFFFFFFF:
                        real_local = struct.unpack_from("<Q", edata, pos)[0]
                        pos += 8
                eoff += 4 + esz
        members[name] = (real_local, real_csize, real_usize, method)
        off += 46 + namelen + extralen + commentlen
    return members


def _extract_hyde_member_bytes(url: str, local_off: int, csize: int) -> bytes:
    header = _fetch_range(url, local_off, local_off + 8192)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("HYDE local file header signature missing")
    namelen, extralen = struct.unpack_from("<HH", header, 26)
    data_start = local_off + 30 + namelen + extralen
    payload = _fetch_range(url, data_start, data_start + csize - 1)
    return Deflate64().decompress(payload)


def _asc_to_geotiff(asc_bytes: bytes, tif_path: Path) -> None:
    import rasterio

    tif_path.parent.mkdir(parents=True, exist_ok=True)
    asc_path = tif_path.with_suffix(".asc")
    asc_path.write_bytes(asc_bytes)
    try:
        with rasterio.open(asc_path) as src:
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                compress="deflate",
                predictor=2,
                crs=CRS.from_epsg(4326),
            )
            data = src.read()
            nodata = src.nodata
            if nodata is not None:
                profile["nodata"] = nodata
            tmp = tif_path.with_suffix(".tif.partial")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data)
            tmp.replace(tif_path)
    finally:
        if asc_path.is_file():
            asc_path.unlink()


def fetch_hyde_pilot(cache_dir: Path) -> list[dict[str, object]]:
    years = sorted(
        {
            layer.hyde_year
            for layer in PILOT_LAYERS
            if layer.source == "hyde" and layer.hyde_year is not None
        }
    )
    if not years:
        return []

    dest_dir = cache_dir / "hyde"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    needed = [year for year in years if not (dest_dir / hyde_popd_tif_name(year)).is_file()]
    if not needed:
        return [
            {
                "source": "hyde",
                "status": "cached",
                "years": list(years),
                "files": [str(dest_dir / hyde_popd_tif_name(y)) for y in years],
            }
        ]

    size = _hyde_content_length(HYDE_BASELINE_URL)
    members = _parse_zip64_central_directory(HYDE_BASELINE_URL, size)
    for year in needed:
        member = hyde_popd_member(year)
        if member not in members:
            raise FileNotFoundError(f"{member} not in HYDE baseline zip")
        local_off, csize, _usize, _method = members[member]
        asc_bytes = _extract_hyde_member_bytes(HYDE_BASELINE_URL, local_off, csize)
        tif_path = dest_dir / hyde_popd_tif_name(year)
        _asc_to_geotiff(asc_bytes, tif_path)
        rows.append(
            {
                "source": "hyde",
                "year": year,
                "member": member,
                "status": "extracted",
                "path": str(tif_path),
                "asc_bytes": len(asc_bytes),
            }
        )
    return rows


def fetch_external_pilots(cache_dir: Path) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(fetch_mapspam_pilot(cache_dir))
    rows.extend(fetch_glw_pilot(cache_dir))
    rows.extend(fetch_europe_suit_pilot(cache_dir))
    rows.extend(fetch_worldpop_pilot(cache_dir))
    rows.extend(fetch_hyde_pilot(cache_dir))
    manifest = {
        "engine": "external_pilots",
        "layer_count": len(PILOT_LAYERS),
        "layers": [layer.layer_id for layer in PILOT_LAYERS],
        "sources": rows,
    }
    path = cache_dir / "external_source_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
