# Dataset catalog for EU5 location mapping

Living inventory of global (or Old-World) datasets that can reuse the ProsperOrPerishStaticModifiers pattern:

`raster → sample at EU5 footprint points → aggregate to location_tag → wide parquet → Pages mapmodes`

Later these become **ML features** (biophysical potential / climate / soil) and **labels/priors** (observed production, historical proxies) for mapping into EU5 farming goods (plants + animals).

## Roles (do not mix in one UI mode blindly)

| Role | ML use | Notes |
|------|--------|-------|
| Biophysical potential | Features | Climate+soil fitness / attainable yield |
| Observed modern production | Soft labels / calibration | Modern census downscaled; **not** 1337 truth |
| Livestock / pasture | Features + soft labels | Densities and grazing potential |
| Historical / era proxies | Labels toward 1337 | Coarser, Europe-weighted, or presence-only |
| Climate / soil primitives | Features | Building blocks, not goods mapmodes |

## EU5 farming goods gap lens

| Coverage | Goods |
|----------|-------|
| Strong via GAEZ physics | wheat, maize, rice, millet, potato/roots, legumes (partial), banana/tropical staples |
| Weak / missing plant physics | coffee, cocoa, tea, sugar, cotton, tobacco, olives, wine/fruit, fiber_crops, silk, beeswax, spices |
| Animals | livestock, wool, horses (elephants/wild_game are rural, not farming method) |

---

## Tier A — priority for map browser pilots

| Name | Role | Resolution | Crops / species | EU5 goods | License | Fetch | Mapable like GAEZ? | Priority |
|------|------|------------|-----------------|-----------|---------|-------|--------------------|----------|
| **GAEZ v5 RES05** (in repo) | Potential | 5′ | 23 contracted crops × rainfed/irrigated | staples / legumes / roots | FAO public GCS | `posm fetch-gaez` | Yes (baseline) | Done |
| **MapSPAM 2020 v2.2** | Observed modern | ~10 km / 5′ | 46 crops × rainfed/irrigated/total | maize, wheat, rice, potato, legumes, **cotton**, **sugar**, **tobacco**, **coffee**, **cocoa**, **tea** | IFPRI / Dataverse | doi:[10.7910/DVN/SWPENT](https://doi.org/10.7910/DVN/SWPENT) | Yes (guestbook required) | A1 |
| **MapSPAM 2010 v2.0** (pilot fetch) | Observed modern | ~10 km / 5′ | same family; wheat/maize/cotton rainfed Y+H | wheat, maize, cotton | IFPRI / Dataverse | doi:[10.7910/DVN/PRFF8V](https://doi.org/10.7910/DVN/PRFF8V) | Yes (automated) | **A1 pilot** |
| **FAO GLW4 (2020)** | Livestock density | 5′ | cattle, buffalo, sheep, goats, pigs, chickens (+ horses in some packs) | livestock, wool; horses weaker | CC BY 4.0 | GCS `fao-gismgr-glw-data` `DATA/GLW/MAPSET/D-AW/GLW.D-AW.{CTX,SHX,...}.tif` | Yes | **A2** |
| **Global Pasture Watch** | Pasture / livestock land potential | 1 km | potential livestock land + systems 2000–2022 | livestock, wool | Zenodo | [zenodo.14933679](https://zenodo.org/records/14933679) | Yes (large) | A2 pair |
| **EarthStat / Monfreda (~2000)** | Observed modern | 5′ | **175** crops harvested area + yield | olives, grapes→wine proxy, spices, fiber, specialty plants | Academic / EarthStat terms | [earthstat.org](http://www.earthstat.org/) | Yes | A3 |
| **Zabel et al. crop suitability v3** | Potential (independent of GAEZ) | 30″ | ~23 food/feed/fiber + overall; rainfed/irrigated | staples + fiber | Zenodo | [zenodo.5982577](https://zenodo.org/records/5982577) | Yes | A4 |
| **Europe ag suitability 1500–2000** | Historical / era proxy | 0.5° | Not crop-specific; annual index 0–1, bands 1500–2000 | general ag potential (Europe) | Harvard Dataverse | doi:[10.7910/DVN/ECWMZS](https://doi.org/10.7910/DVN/ECWMZS) (`suit.tif`) | Yes (Europe-only) | **A5** |
| **WorldPop 2020 (1 km mosaic)** | Modern population density | ~1 km | people/pixel → people/km² at sample lat | settlement / demand prior | CC BY 4.0 | `data.worldpop.org` Global_2000_2020 mosaic | Yes | **A6** |
| **HYDE 3.2.1 popd** | Historical population density | 5′ | people/km² for 1300 / 1400 / 1500 CE | era population prior (not 1337 truth) | CC BY 3.0 | DANS doi:[10.17026/dans-25g-gez3](https://doi.org/10.17026/dans-25g-gez3) (Range-extract `popd_*AD.asc`) | Yes | **A6** |

### Pilot layers implemented in this repo (`posm fetch-external` / `build-external`)

| Layer id | Source | Metric | Notes |
|----------|--------|--------|-------|
| `spam_wheat_rainfed_yield` | MapSPAM 2010 | yield t/km² | Observed rainfed wheat (native t/ha ×100) |
| `spam_maize_rainfed_yield` | MapSPAM 2010 | yield t/km² | Observed rainfed maize |
| `spam_cotton_rainfed_yield` | MapSPAM 2010 | yield t/km² | Observed rainfed cotton (GAEZ gap) |
| `spam_wheat_rainfed_harvested_area` | MapSPAM 2010 | km²/cell | Harvested area (native ha/cell ÷100) |
| `spam_maize_rainfed_harvested_area` | MapSPAM 2010 | km²/cell | |
| `spam_cotton_rainfed_harvested_area` | MapSPAM 2010 | km²/cell | |
| `glw_cattle_density` | GLW4 D-AW CTX | heads/km² | Livestock |
| `glw_sheep_density` | GLW4 D-AW SHX | heads/km² | Wool proxy |
| `europe_ag_suitability_1500` | ECWMZS suit.tif band 1500 | index 0–1 | Closest year to EU5 start in dataset |
| `worldpop_pop_density_2020` | WorldPop 2020 1 km | people/km² | Modern; separate UI group |
| `hyde_pop_density_1300` | HYDE 3.2.1 popd | people/km² | Historical; nearest century before 1337 |
| `hyde_pop_density_1400` | HYDE 3.2.1 popd | people/km² | Historical |
| `hyde_pop_density_1500` | HYDE 3.2.1 popd | people/km² | Historical; pairs with Europe 1500 suitability |
| `eu5_population_total` | Constructor `derived_food_balance_by_location` | people | EU5 Vanilla; `total_population` × 1000 |
| `eu5_population_density` | Constructor pop / calibrated area | people/km² | Uses Constructor `area_jacobian_km2` |
| `eu5_development` | Constructor `derived_food_balance_by_location` | development | Numeric game-start development score |

---

## Tier B — high value, slightly harder or narrower

| Name | Role | Resolution | Notes | Fetch |
|------|------|------------|-------|-------|
| **GYGA global yield potential** | Potential | 30″ | Maize/wheat/rice only; high quality | WUR / Nature Food 2024 |
| **MIRCA2000** | Irrigated vs rainfed monthly areas | 5′ | Seasonality; pairs with GMIA | Uni Frankfurt / Zenodo |
| **GMIA** | Irrigated extent | ~5′ | Already used in Constructor inventory | FAO |
| **GDHY** | Modern yield time series | crop grids | Heavier ETL; Constructor has `gdhy_v1_2_v1_3` cache tree | NASA/UTokyo lineage |
| **CROPGRIDS (~2020)** | Observed | ~0.05° | Newer Monfreda-style 173 crops if license OK | literature / authors |
| **FGGD pasture suitability** | Pasture potential | 5′ | Older FAO/IIASA pasture index | FAO catalog |
| **FAO tsetse atlas (2024)** | Livestock disease constraint | points→raster | Constructor already has tsetse contract features | FAO PAAT |
| **Specialty climate suitability** | Potential for plantation goods | varies | Coffee/cocoa/tea/sugar niche models; patchwork | papers / Ecocrop-style |

---

## Tier C — supporting primitives (features, not goods mapmodes)

| Name | Role | Notes | Status in PoP |
|------|------|-------|---------------|
| SoilGrids / HWSD | Soil features | texture, pH, SOC, depth | GAEZ uses HWSD internally |
| CHELSA / WorldClim | Climate | modern climatologies | — |
| **CHELSA-TraCE21k (~1300)** | Paleoclimate near start | Era-relevant climate features | Constructor inventory `chelsa_trace21k_1300` |
| **HYDE 3.2** | Historical cropland/pasture / population | Diagnostic land use; popd pilots in this repo | Constructor `hyde_diagnostic_only`; StaticModifiers exploration layers |
| **ArchaeoGLOBE / crop history registry** | Presence evidence | Pre-1337 crop availability | `population_capacity_crop_history_registry.toml` |

---

## MapSPAM crop codes used in pilots

| EU5-relevant name | SPAM code in GeoTIFF names |
|-------------------|----------------------------|
| wheat | `WHEA` |
| maize | `MAIZ` |
| cotton | `COTT` |

Production systems in filenames: `_R` rainfed, `_I` irrigated, `_A` all/total.

Variables: `Y` yield, `H` harvested area, `A` physical area, `P` production.

Example: `spam2010V2r0_global_Y_WHEA_R.tif` (pilot). MapSPAM 2020 uses a similar `spam2020_*` pattern but currently requires a Dataverse guestbook for bulk GeoTIFF zips.

## GLW4 species codes (GCS)

| Species | Code | Pilot? |
|---------|------|--------|
| cattle | CTX | yes |
| sheep | SHX | yes |
| goats | GTX | — |
| pigs | PGX | — |
| chickens | CHX | — |
| buffalo | BFX | — |
| horses | HOX | later (horses good) |
| ducks | DKX | — |

Density mapset: `D-AW` (animals per km²). Absolute counts: `AWX`.

---

## Explicit caveats for ML later

1. **MapSPAM / EarthStat / GLW are modern** — use as calibration priors, not 1337 labels alone.  
2. **GAEZ HP8100 is 1981–2000 climate** — modern diagnostic potential, not medieval.  
3. **Europe 1500 suitability** starts at 1500 (not 1337) and is not crop-specific.  
4. **WorldPop 2020** is modern observed settlement density — calibration prior only, never a 1337 label.  
5. **HYDE popd** is century-step reconstruction (1300/1400/1500); useful as an era prior, not census truth.  
6. Keep **potential vs observed vs historical vs population** as separate viewer groups so the UI stays honest.

## Recommended next steps after pilots

1. Expand MapSPAM to coffee/cocoa/sugar/tobacco rainfed+irrigated.  
2. Add GLW horses + Global Pasture Watch potential land.  
3. Add EarthStat olives/grapes for wine/olive gaps.  
4. Defer ML training until ≥2 potential + 1 observed + 1 livestock + 1 historical layer are location-aligned (pilots below satisfy this bar once published).
