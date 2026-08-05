# P&P wheat — 1337 technology mapmodes

## What EU5 means by wheat

Vanilla EU5 (`game/in_game/common/goods/03_food.txt`):

- `wheat` is a `farming` `raw_material`
- `food = 8.0`, staple demand across peasants/laborers/soldiers
- `origin_in_old_world = yes`
- Localization: staple vegetable foodstuff for humans and animals; later joined by New World crops (tomato, maize, potato)

Constructor maps the good 1:1 to GAEZ crop code **WHE** / Module II **WHEA**.

For 1337 taxa (Constructor crop-history registry, `target_year = 1337`):

- *Triticum aestivum, durum, spelta, dicoccum, monococcum*
- Cool-season temperate cereal (winter + spring wheat)

## 1337 technology

Manorial low-input agriculture: three-field rotation, ox traction, organic manure only, no synthetic fertilizer or mechanization.

**Absolute yield anchor** (Pretty 1990 / BAHS Bishop of Winchester manors 1283–1349):

| Crop  | Gross kg/ha | Net kg/ha |
|-------|-------------|-----------|
| Wheat | **515**     | 385       |

Source: `ProsperOrPerishPopulationCapacityPipeline/.../historical_yields.py`.

## Successive evidence (how to add more)

Catalog: [`research/pnp_wheat_evidence.json`](pnp_wheat_evidence.json).

**Scientific method** — hierarchical soft anchors + feature residual:

1. Assemble PyAEZ 1337 wheat targets.
2. **Irrigation policy overrides** from evidence (`prefer_irrigated` regions such as Egypt / Hindustan / Crescent take `max(rainfed, irrigated)` so floodplain cells are not killed by arid rainfed zeros).
3. Global BAHS tech scale so positive median ≈ 515 kg/ha.
4. **Soft regional target scales**: for each evidence entry, compute `ratio = target / regional_positive_median`, apply confidence-weighted log-scale with an Empirical-Bayes prior toward 1 (overlapping regions accumulate weights). Near-zero entries soft-pull tropical belts down.
5. Train two `HistGradientBoostingRegressor`s on adjusted targets (yield + suitable_fraction).
6. **Feature residual calibrator**: learn `features → log(scale)` from evidence-covered locations only; apply globally so similar climates inherit corrections.
7. Validate agronomic gates **A\*** and per-evidence gates **E\***.

To add evidence later: append a JSON object with `id`, `kind` (`positive_median` | `near_zero`), `eu5_regions`, `target_kg_ha`, `confidence`, `tolerance`, optional `irrigation_policy`, and source metadata — then `posm build-pnp` + publish.

Confidence weights (default): high 0.75 · medium 0.55 · low 0.35. Prior strength 1.0. Scale clip [0.25, 4.0].

## Features

- 1337 climate: CHELSA-TraCE `*_target`
- EU5 statics: `soil_quality`, `topography`, `vegetation`, `climate`, `climate_winter`
- Geography: calibrated lat/lon, absolute latitude
- Modern GAEZ rainfed potentials for wheat + cool-cereal analogues
- Engineered: aridity proxy, temperature distance from ~12 °C, severe-winter flag
- `prefer_irrigated` corridor flag from evidence

## Assumptions (validation gate)

| ID | Assumption | Gate |
|----|------------|------|
| A1 | Near-zero wheat potential in Antarctica / polar ice | mean yield ≈ 0 |
| A2 | Near-zero in hyper-arid desert interiors (excl. irrigated corridors) | mean yield low |
| A3 | Near-zero in hot-wet equatorial lowlands | mean yield low vs temperate |
| A4 | High potential in NW Europe oceanic / continental temperate belt | mean yield ≫ arid/arctic |
| A5 | Global positive-yield median near BAHS 515 kg/ha (±40%) | distribution check |
| A6 | Model fit adequate | CV R² yield > 0.35 |
| E\* | Each evidence entry within its tolerance band (or near-zero max mean) | per catalog |

## Outputs

| Layer id | Label | Unit |
|----------|-------|------|
| `pnp_wheat_production_density` | Production density | kg DM / km² |
| `pnp_wheat_yield` | Yield | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | Suitable fraction | fraction |
| `pnp_wheat_suitability_class` | Suitability class | 1 best → 9 worst |

Dataset selector: **P&P** · Good: **wheat** (v1).

## Latest training run

See `artifacts/pnp_models/pnp_wheat_model_card.json` (mirrored to `research/pnp_wheat_model_card.json`).
