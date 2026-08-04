# Prosper or Perish Static Modifiers

Standalone **uv / Python** ETL that samples FAO GAEZ v5 crop rasters onto Europa Universalis V locations and publishes a GitHub Pages map browser.

## What it does

1. Downloads locked GAEZ v5 RES05 **YXX** (yield), **YLX** (production density), and **SX3** (suitable area) rasters.
2. Samples them at game-authoritative EU5 footprint points.
3. Emits one wide parquet: one row per location, with rain-fed and irrigated density / yield / suitable columns for 23 crops.
4. Publishes a compact client-side map under `docs/` (crop + water mode + metric dropdowns, hover tooltips).

Production **density** is the default map metric so sparse suitability does not light up the map as if it were strong production.

GAEZ HP8100 covers **1981–2000** (modern diagnostic climate), not circa-1337 reconstruction.

## Setup

```bash
cd ProsperOrPerishStaticModifiers
cp config.example.toml config.local.toml
# edit paths: vanilla_root, labeling_baseline, sample_points
uv sync --extra dev
uv run posm info
```

`config.local.toml` is gitignored. Keep real machine paths there; commit only `config.example.toml`.

## Pipeline

```bash
uv run posm fetch-gaez
uv run posm build-geometry
uv run posm build-samples
uv run posm build-wide
uv run posm publish
uv run posm serve
```

If you already have a long `crop_mode_labels.parquet` from the constructor population-capacity pipeline, you can skip sampling and pivot it directly:

```bash
# set [paths].labels_long in config.local.toml
uv run posm build-geometry
uv run posm build-wide --from-labels
uv run posm publish
```

Open `http://127.0.0.1:8000/`.

### Outputs

| Path | Role |
|------|------|
| `artifacts/gaez_cache/` | Downloaded GeoTIFFs + source manifest |
| `artifacts/location_geometry.parquet` | EU5 location geometry |
| `artifacts/location_id_map.bin.gz` | Pixel → location row index |
| `artifacts/crop_mode_samples/` | Per crop/mode sample + location aggregates |
| `artifacts/location_gaez_wide.parquet` | **Publish dataframe** (one row per location) |
| `docs/` | GitHub Pages site (viewer + packed attributes) |

## GitHub Pages

Enable Pages for this repo with source **Deploy from a branch** → `main` → `/docs`.

The committed `docs/` tree must include the generated attribute pack and location id map. Vanilla `locations.png` is never committed.

## Tests

```bash
uv run pytest
```

## License / data notes

- EU5 `locations.png` and labeling baseline are local inputs (game install / sibling repos).
- GAEZ rasters are downloaded from FAO public cloud storage under their terms.
- This repo does not redistribute vanilla map assets.
