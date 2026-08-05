from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalLayerSpec:
    """One raster column sampled onto EU5 locations for exploration mapmodes."""

    layer_id: str
    label: str
    group: str
    unit: str
    source: str  # spam | glw | europe_suit | worldpop | hyde | eu5 | pnp
    zero_is_missing: bool = True
    # Source-specific fields
    spam_variable: str | None = None  # Y | H
    spam_crop_code: str | None = None  # WHEA | MAIZ | COTT
    spam_system: str | None = None  # R | I | A
    glw_species_code: str | None = None  # CTX | SHX
    raster_band: int | None = None  # 1-based GDAL band; Europe suit year 1500 = band 1
    hyde_year: int | None = None  # CE year for HYDE popd (e.g. 1300)
    # WorldPop mosaic is people/pixel; convert to people/km² using cell area at sample lat.
    count_to_density: bool = False
    # Native MapSPAM/GAEZ-style per-hectare rates → per km² (×100).
    per_ha_to_per_km2: bool = False
    # Native area in hectares → km² (÷100), e.g. MapSPAM harvested ha/cell.
    ha_to_km2: bool = False
    # Table-backed layers (no raster sample); column is layer_id on the EU5 pop frame.
    table_column: str | None = None
    eu5_goods: tuple[str, ...] = ()


# Pilot set from research/dataset_catalog.md
PILOT_LAYERS: tuple[ExternalLayerSpec, ...] = (
    ExternalLayerSpec(
        layer_id="spam_wheat_rainfed_yield",
        label="Wheat yield (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="t/km²",
        source="spam",
        spam_variable="Y",
        spam_crop_code="WHEA",
        spam_system="R",
        per_ha_to_per_km2=True,
        eu5_goods=("wheat",),
    ),
    ExternalLayerSpec(
        layer_id="spam_maize_rainfed_yield",
        label="Maize yield (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="t/km²",
        source="spam",
        spam_variable="Y",
        spam_crop_code="MAIZ",
        spam_system="R",
        per_ha_to_per_km2=True,
        eu5_goods=("maize",),
    ),
    ExternalLayerSpec(
        layer_id="spam_cotton_rainfed_yield",
        label="Cotton yield (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="t/km²",
        source="spam",
        spam_variable="Y",
        spam_crop_code="COTT",
        spam_system="R",
        per_ha_to_per_km2=True,
        eu5_goods=("cotton",),
    ),
    ExternalLayerSpec(
        layer_id="spam_wheat_rainfed_harvested_area",
        label="Wheat harvested area (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="km²/cell",
        source="spam",
        spam_variable="H",
        spam_crop_code="WHEA",
        spam_system="R",
        ha_to_km2=True,
        eu5_goods=("wheat",),
    ),
    ExternalLayerSpec(
        layer_id="spam_maize_rainfed_harvested_area",
        label="Maize harvested area (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="km²/cell",
        source="spam",
        spam_variable="H",
        spam_crop_code="MAIZ",
        spam_system="R",
        ha_to_km2=True,
        eu5_goods=("maize",),
    ),
    ExternalLayerSpec(
        layer_id="spam_cotton_rainfed_harvested_area",
        label="Cotton harvested area (MapSPAM rainfed)",
        group="MapSPAM observed",
        unit="km²/cell",
        source="spam",
        spam_variable="H",
        spam_crop_code="COTT",
        spam_system="R",
        ha_to_km2=True,
        eu5_goods=("cotton",),
    ),
    ExternalLayerSpec(
        layer_id="glw_cattle_density",
        label="Cattle density (GLW4)",
        group="Livestock GLW",
        unit="heads/km²",
        source="glw",
        glw_species_code="CTX",
        eu5_goods=("livestock",),
    ),
    ExternalLayerSpec(
        layer_id="glw_sheep_density",
        label="Sheep density (GLW4)",
        group="Livestock GLW",
        unit="heads/km²",
        source="glw",
        glw_species_code="SHX",
        eu5_goods=("livestock", "wool"),
    ),
    ExternalLayerSpec(
        layer_id="europe_ag_suitability_1500",
        label="Europe ag suitability (1500)",
        group="Historical Europe",
        unit="index 0–1",
        source="europe_suit",
        raster_band=1,  # year 1500 is first band
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="worldpop_pop_density_2020",
        label="Population density (WorldPop 2020)",
        group="Modern population",
        unit="people/km²",
        source="worldpop",
        count_to_density=True,
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="hyde_pop_density_1300",
        label="Population density (HYDE 1300)",
        group="Historical population",
        unit="people/km²",
        source="hyde",
        hyde_year=1300,
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="hyde_pop_density_1400",
        label="Population density (HYDE 1400)",
        group="Historical population",
        unit="people/km²",
        source="hyde",
        hyde_year=1400,
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="hyde_pop_density_1500",
        label="Population density (HYDE 1500)",
        group="Historical population",
        unit="people/km²",
        source="hyde",
        hyde_year=1500,
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="eu5_population_total",
        label="Population",
        group="Vanilla start",
        unit="people",
        source="eu5",
        table_column="eu5_population_total",
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="eu5_population_density",
        label="Population density",
        group="Vanilla start",
        unit="people/km²",
        source="eu5",
        table_column="eu5_population_density",
        zero_is_missing=True,
        eu5_goods=(),
    ),
    ExternalLayerSpec(
        layer_id="eu5_development",
        label="Development",
        group="Vanilla start",
        unit="development",
        source="eu5",
        table_column="eu5_development",
        zero_is_missing=False,
        eu5_goods=(),
    ),
)

EXPLORATION_LAYERS: tuple[ExternalLayerSpec, ...] = tuple(
    layer for layer in PILOT_LAYERS if layer.source != "eu5"
)
EU5_LAYERS: tuple[ExternalLayerSpec, ...] = tuple(
    layer for layer in PILOT_LAYERS if layer.source == "eu5"
)

# Prosper-or-Perish 1337 reconstructed goods mapmodes (table-backed, model output).
PNP_LAYERS: tuple[ExternalLayerSpec, ...] = (
    ExternalLayerSpec(
        layer_id="pnp_wheat_production_density",
        label="Production density",
        group="Wheat (1337)",
        unit="kg DM / km²",
        source="pnp",
        table_column="pnp_wheat_production_density",
        zero_is_missing=True,
        eu5_goods=("wheat",),
    ),
    ExternalLayerSpec(
        layer_id="pnp_wheat_yield",
        label="Yield",
        group="Wheat (1337)",
        unit="kg DM / suitable km²",
        source="pnp",
        table_column="pnp_wheat_yield",
        zero_is_missing=True,
        eu5_goods=("wheat",),
    ),
    ExternalLayerSpec(
        layer_id="pnp_wheat_suitable_fraction",
        label="Suitable fraction",
        group="Wheat (1337)",
        unit="fraction",
        source="pnp",
        table_column="pnp_wheat_suitable_fraction",
        zero_is_missing=True,
        eu5_goods=("wheat",),
    ),
    ExternalLayerSpec(
        layer_id="pnp_wheat_suitability_class",
        label="Suitability class",
        group="Wheat (1337)",
        unit="class 1 best → 9 worst",
        source="pnp",
        table_column="pnp_wheat_suitability_class",
        zero_is_missing=False,
        eu5_goods=("wheat",),
    ),
)


def layer_by_id(layer_id: str) -> ExternalLayerSpec:
    for layer in PILOT_LAYERS:
        if layer.layer_id == layer_id:
            return layer
    raise KeyError(layer_id)


def spam_tif_name(variable: str, crop_code: str, system: str) -> str:
    # Preferred basename; resolver also accepts flexible globs across SPAM releases.
    return f"spam2010V2r0_global_{variable}_{crop_code}_{system}.tif"


def spam_zip_name(variable: str) -> str:
    return {
        "Y": "spam2010v2r0_global_yield.geotiff.zip",
        "H": "spam2010v2r0_global_harv_area.geotiff.zip",
    }[variable]


def glw_density_relpath(species_code: str) -> str:
    return f"DATA/GLW/MAPSET/D-AW/GLW.D-AW.{species_code}.tif"


def hyde_popd_member(year: int) -> str:
    return f"baseline/asc/{year}AD_pop/popd_{year}AD.asc"


def hyde_popd_tif_name(year: int) -> str:
    return f"hyde_popd_{year}AD.tif"


def resolve_layer_raster(cache_dir: Path, layer: ExternalLayerSpec) -> Path:
    if layer.source == "spam":
        assert layer.spam_variable and layer.spam_crop_code and layer.spam_system
        folder = cache_dir / "mapspam" / layer.spam_variable
        exact = folder / spam_tif_name(
            layer.spam_variable, layer.spam_crop_code, layer.spam_system
        )
        if exact.is_file():
            return exact
        # Flexible match for minor naming drift across SPAM releases.
        pattern = (
            f"*_{layer.spam_variable}_{layer.spam_crop_code}_{layer.spam_system}.tif"
        )
        hits = sorted(folder.glob(pattern))
        if not hits:
            hits = sorted(
                folder.glob(
                    f"*{layer.spam_crop_code}*{layer.spam_system}*.tif"
                )
            )
        if not hits:
            raise FileNotFoundError(
                f"missing MapSPAM raster for {layer.layer_id} under {folder}"
            )
        return hits[0]

    if layer.source == "glw":
        assert layer.glw_species_code
        path = cache_dir / "glw" / Path(glw_density_relpath(layer.glw_species_code)).name
        if not path.is_file():
            raise FileNotFoundError(f"missing GLW raster {path}")
        return path

    if layer.source == "europe_suit":
        path = cache_dir / "europe_suit" / "suit.tif"
        if not path.is_file():
            raise FileNotFoundError(f"missing Europe suitability raster {path}")
        return path

    if layer.source == "worldpop":
        path = cache_dir / "worldpop" / "ppp_2020_1km_Aggregated.tif"
        if not path.is_file():
            raise FileNotFoundError(f"missing WorldPop raster {path}")
        return path

    if layer.source == "hyde":
        assert layer.hyde_year is not None
        path = cache_dir / "hyde" / hyde_popd_tif_name(layer.hyde_year)
        if not path.is_file():
            raise FileNotFoundError(f"missing HYDE raster {path}")
        return path

    if layer.source == "eu5":
        raise ValueError(
            f"layer {layer.layer_id} is table-backed (source=eu5); no raster path"
        )

    raise ValueError(f"unknown layer source: {layer.source}")
