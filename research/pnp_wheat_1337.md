# P&P wheat — 1337 technology mapmodes

## What EU5 means by wheat

Vanilla EU5 (`game/in_game/common/goods/03_food.txt`):

- `wheat` is a `farming` `raw_material`
- `food = 8.0`, staple demand across peasants/laborers/soldiers
- `origin_in_old_world = yes`

Constructor maps the good 1:1 to GAEZ crop code **WHE** / Module II **WHEA**.

For 1337 taxa: *Triticum aestivum, durum, spelta, dicoccum, monococcum*.

## Scientific approach (attribute-driven)

**Goal:** learn where wheat could/did grow well from location attributes — not overwrite regions.

Catalog: [`research/pnp_wheat_evidence.json`](pnp_wheat_evidence.json) (v2).

```
historical recipes + crop-history presence
        │
        ▼
 attribute match rules ──► location-level labels (yield + suitability + weights)
        │
        ▼
 curated features X (climate, soil, water access, GAEZ, HYDE, …) ──► HGB dual heads
        │
        ▼
 global map + feature importances (what pattern the model found)
```

Critical rules:

- **Never** use `region` / `super_region` as features or as uniform target multipliers.
- Historical geography enters only by labeling locations that match **agronomic attribute strata** (e.g. Nile: lat/lon band + `has_river` + arid/mediterranean).
- Hold out Anatolia + Maghreb recipes to test whether the feature model recovers them.

### Label sources

| Kind | Source |
|------|--------|
| Intensity recipes | BAHS 515, Mediterranean ~420, Song N. China ~650, Indo-Gangetic ~1100, Nile ~1090, Crescent ~900 |
| Presence | `crop_mode_labels.parquet` wheat `known_available` (~1,348 locations) |
| Near-zero | arctic; tropical+jungle; desert without river/hydraulic access |
| Soft physical prior | PyAEZ 1337 wheat (low sample weight) where history is silent |

### Features (curated)

From `location_candidates` + `location_gaez_wide` + `location_external_wide` + PyAEZ as features:

- CHELSA temperature / precipitation / seasonality
- Water access: `has_river`, lake adjacency, irrigable fractions, hydraulic access
- GAEZ wheat rainfed + irrigated potentials / suitability; barley/rye/oats analogues
- External: MapSPAM wheat, Europe ag suitability 1500, HYDE pop 1300, livestock densities
- EU5 topo / vegetation / climate one-hots; soil ordinal; winter flags
- Engineered: aridity proxy, temp distance from 12 °C, abs lat

### Outputs

| Layer id | Label | Unit |
|----------|-------|------|
| `pnp_wheat_production_density` | Production density | kg DM / km² |
| `pnp_wheat_yield` | Yield | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | Suitable fraction | fraction |
| `pnp_wheat_suitability_class` | Suitability class | 1 best → 9 worst |

### Validation

- **A\*** physical gates (arctic, desert-without-water, tropics, NW Europe, global median sane, CV R²)
- **S\*** attribute-strata gates (Nile+river high; desert no river low; tropical jungle low; NW farmland mid-high)
- **H\*** holdout recipe recovery (Anatolia, Maghreb) — reported; do not block deploy alone

### How to add evidence

Append a recipe with `match` (`all` / `any` + `eq`/`in`/`gt`/… on attribute columns), `target_kg_ha`, `weight`, optional `holdout: true`. Then `posm build-pnp` + publish.

## Latest run

See `artifacts/pnp_models/pnp_wheat_model_card.json` (mirrored under `research/`).
