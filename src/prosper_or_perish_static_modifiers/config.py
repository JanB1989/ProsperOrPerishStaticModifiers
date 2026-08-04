from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw: str | Path, *, base: Path | None = None) -> Path:
    """Resolve config paths, including Windows drive letters under WSL/Linux."""

    text = str(raw).strip().strip('"').strip("'")
    if not text:
        raise ValueError("empty path")
    base = base or repo_root()

    if text.startswith("/") and not text.startswith("//"):
        return Path(text).resolve()

    if text.startswith("\\\\wsl$\\") or text.startswith("//wsl$/"):
        normalized = text.replace("\\", "/")
        parts = normalized.split("/")
        if len(parts) >= 4:
            return Path("/" + "/".join(parts[3:])).resolve()

    if len(text) >= 3 and text[1] == ":" and text[0].isalpha():
        drive = text[0].lower()
        rest = text[2:].replace("\\", "/").lstrip("/")
        try:
            release = os.uname().release.lower()
        except AttributeError:
            release = ""
        if sys.platform.startswith("linux") or "microsoft" in release:
            return Path(f"/mnt/{drive}/{rest}").resolve()
        return Path(text).resolve()

    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectConfig:
    repo: Path
    vanilla_root: Path
    locations_png: Path
    labeling_baseline: Path
    sample_points: Path
    gaez_cache_dir: Path
    artifacts_dir: Path
    docs_dir: Path
    labels_long: Path | None
    crop_samples_dir: Path | None
    crops: list[str]
    water_modes: list[str]
    equator_y: int

    @property
    def geometry_path(self) -> Path:
        return self.artifacts_dir / "location_geometry.parquet"

    @property
    def location_id_map_path(self) -> Path:
        return self.artifacts_dir / "location_id_map.bin.gz"

    @property
    def location_id_meta_path(self) -> Path:
        return self.artifacts_dir / "location_id_map.meta.json"

    @property
    def samples_dir(self) -> Path:
        return self.artifacts_dir / "crop_mode_samples"

    @property
    def wide_path(self) -> Path:
        return self.artifacts_dir / "location_gaez_wide.parquet"

    @property
    def source_manifest_path(self) -> Path:
        return self.gaez_cache_dir / "source_manifest.json"


def load_config(path: str | Path | None = None) -> ProjectConfig:
    import tomllib

    repo = repo_root()
    config_path = resolve_path(path, base=repo) if path else (repo / "config.local.toml")
    if not config_path.is_file():
        example = repo / "config.example.toml"
        if not example.is_file():
            raise FileNotFoundError(
                f"missing {config_path}; copy config.example.toml to config.local.toml"
            )
        config_path = example

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    pipeline = data.get("pipeline", {})

    vanilla_root = resolve_path(paths["vanilla_root"], base=repo)
    locations_png = resolve_path(
        paths.get(
            "locations_png",
            str(vanilla_root / "game" / "in_game" / "map_data" / "locations.png"),
        ),
        base=repo,
    )
    labels_raw = paths.get("labels_long")
    samples_raw = paths.get("crop_samples_dir")
    return ProjectConfig(
        repo=repo,
        vanilla_root=vanilla_root,
        locations_png=locations_png,
        labeling_baseline=resolve_path(paths["labeling_baseline"], base=repo),
        sample_points=resolve_path(paths["sample_points"], base=repo),
        gaez_cache_dir=resolve_path(paths.get("gaez_cache_dir", "artifacts/gaez_cache"), base=repo),
        artifacts_dir=resolve_path(paths.get("artifacts_dir", "artifacts"), base=repo),
        docs_dir=resolve_path(paths.get("docs_dir", "docs"), base=repo),
        labels_long=resolve_path(labels_raw, base=repo) if labels_raw else None,
        crop_samples_dir=resolve_path(samples_raw, base=repo) if samples_raw else None,
        crops=list(pipeline.get("crops") or []),
        water_modes=list(pipeline.get("water_modes") or ["rainfed", "irrigated"]),
        equator_y=int(pipeline.get("equator_y", 1024)),
    )