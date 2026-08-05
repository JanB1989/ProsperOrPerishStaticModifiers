from __future__ import annotations

import argparse
import http.server
import json
import socketserver

from prosper_or_perish_static_modifiers.config import load_config
from prosper_or_perish_static_modifiers.crops import sha256_lock_count
from prosper_or_perish_static_modifiers.build_external import build_external_wide
from prosper_or_perish_static_modifiers.fetch import fetch_gaez
from prosper_or_perish_static_modifiers.fetch_external import fetch_external_pilots
from prosper_or_perish_static_modifiers.geometry import (
    build_location_geometry,
    build_location_id_map,
)
from prosper_or_perish_static_modifiers.publish import publish_docs
from prosper_or_perish_static_modifiers.sample import build_samples
from prosper_or_perish_static_modifiers.wide import (
    build_wide_dataframe,
    build_wide_from_labels,
)


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.local.toml (default: repo config.local.toml or example)",
    )


def cmd_fetch_gaez(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    manifest = fetch_gaez(
        cfg.gaez_cache_dir,
        crops=cfg.crops or None,
        water_modes=tuple(cfg.water_modes),
    )
    print(
        json.dumps(
            {
                "source_count": manifest["source_count"],
                "manifest": str(cfg.source_manifest_path),
            },
            indent=2,
        )
    )
    return 0


def cmd_build_geometry(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    geometry = build_location_geometry(
        baseline_path=cfg.labeling_baseline,
        locations_png_path=cfg.locations_png,
        equator_y=cfg.equator_y,
    )
    geometry.write_parquet(cfg.geometry_path)
    meta = build_location_id_map(
        geometry=geometry,
        locations_png_path=cfg.locations_png,
        output_bin_gz=cfg.location_id_map_path,
        output_meta=cfg.location_id_meta_path,
    )
    print(
        json.dumps(
            {
                "geometry": str(cfg.geometry_path),
                "locations": geometry.height,
                "id_map": str(cfg.location_id_map_path),
                "width": meta["width"],
                "height": meta["height"],
            },
            indent=2,
        )
    )
    return 0


def cmd_build_samples(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    written = build_samples(
        sample_points_path=cfg.sample_points,
        cache_dir=cfg.gaez_cache_dir,
        output_dir=cfg.samples_dir,
        crops=cfg.crops or None,
        water_modes=tuple(cfg.water_modes),
    )
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


def cmd_build_wide(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.from_labels:
        if cfg.labels_long is None or not cfg.labels_long.is_file():
            raise FileNotFoundError(
                "labels_long path missing; set [paths].labels_long in config"
            )
        path = build_wide_from_labels(
            geometry_path=cfg.geometry_path,
            labels_path=cfg.labels_long,
            output_path=cfg.wide_path,
            crops=cfg.crops or None,
            water_modes=tuple(cfg.water_modes),
            crop_samples_dir=cfg.crop_samples_dir,
            sample_points_path=cfg.sample_points,
            gaez_cache_dir=cfg.gaez_cache_dir,
        )
    else:
        path = build_wide_dataframe(
            geometry_path=cfg.geometry_path,
            samples_dir=cfg.samples_dir,
            output_path=cfg.wide_path,
            crops=cfg.crops or None,
            water_modes=tuple(cfg.water_modes),
        )
    print(json.dumps({"wide": str(path)}, indent=2))
    return 0


def cmd_fetch_external(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    manifest = fetch_external_pilots(cfg.external_cache_dir)
    print(
        json.dumps(
            {
                "layer_count": manifest["layer_count"],
                "cache": str(cfg.external_cache_dir),
                "manifest": str(cfg.external_cache_dir / "external_source_manifest.json"),
            },
            indent=2,
        )
    )
    return 0


def cmd_build_external(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    path = build_external_wide(
        sample_points_path=cfg.sample_points,
        cache_dir=cfg.external_cache_dir,
        geometry_path=cfg.geometry_path,
        output_path=cfg.external_wide_path,
        start_pops_path=cfg.start_pops_path,
        location_area_path=cfg.location_area_path,
    )
    print(json.dumps({"external_wide": str(path)}, indent=2))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    order_path = cfg.artifacts_dir / "location_row_order.json"
    external_path = cfg.external_wide_path if cfg.external_wide_path.is_file() else None
    index = publish_docs(
        wide_path=cfg.wide_path,
        location_id_map_path=cfg.location_id_map_path,
        location_id_meta_path=cfg.location_id_meta_path,
        location_row_order_path=order_path,
        docs_dir=cfg.docs_dir,
        crops=cfg.crops or None,
        water_modes=tuple(cfg.water_modes),
        external_wide_path=external_path,
    )
    print(
        json.dumps(
            {
                "index": str(index),
                "exploration": external_path is not None,
            },
            indent=2,
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = cfg.docs_dir
    port = int(args.port)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

    print(f"Serving {root} at http://127.0.0.1:{port}/")
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        httpd.serve_forever()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(
        json.dumps(
            {
                "repo": str(cfg.repo),
                "vanilla_root": str(cfg.vanilla_root),
                "locations_png": str(cfg.locations_png),
                "labeling_baseline": str(cfg.labeling_baseline),
                "sample_points": str(cfg.sample_points),
                "gaez_cache_dir": str(cfg.gaez_cache_dir),
                "start_pops": str(cfg.start_pops_path),
                "location_area": str(cfg.location_area_path),
                "labels_long": str(cfg.labels_long) if cfg.labels_long else None,
                "sha256_locks": sha256_lock_count(),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posm",
        description="Prosper or Perish Static Modifiers — GAEZ ETL and map browser",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch-gaez", help="Download locked GAEZ v5 YXX/YLX/SX3 rasters")
    _add_config_arg(fetch)
    fetch.set_defaults(func=cmd_fetch_gaez)

    fetch_ext = sub.add_parser(
        "fetch-external",
        help="Download MapSPAM / GLW4 / Europe-1500 / population pilot rasters",
    )
    _add_config_arg(fetch_ext)
    fetch_ext.set_defaults(func=cmd_fetch_external)

    geom = sub.add_parser("build-geometry", help="Build location geometry + id map")
    _add_config_arg(geom)
    geom.set_defaults(func=cmd_build_geometry)

    samples = sub.add_parser("build-samples", help="Sample GAEZ rasters at footprint points")
    _add_config_arg(samples)
    samples.set_defaults(func=cmd_build_samples)

    wide = sub.add_parser("build-wide", help="Emit location_gaez_wide.parquet")
    _add_config_arg(wide)
    wide.add_argument(
        "--from-labels",
        action="store_true",
        help="Pivot an existing long crop_mode_labels parquet (paths.labels_long)",
    )
    wide.set_defaults(func=cmd_build_wide)

    external = sub.add_parser(
        "build-external",
        help="Sample pilot external rasters onto locations",
    )
    _add_config_arg(external)
    external.set_defaults(func=cmd_build_external)

    publish = sub.add_parser("publish", help="Write GitHub Pages assets under docs/")
    _add_config_arg(publish)
    publish.set_defaults(func=cmd_publish)

    serve = sub.add_parser("serve", help="Serve docs/ locally")
    _add_config_arg(serve)
    serve.add_argument("--port", default="8000")
    serve.set_defaults(func=cmd_serve)

    info = sub.add_parser("info", help="Show resolved config paths")
    _add_config_arg(info)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
