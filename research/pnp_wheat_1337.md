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

## Modeling approach

1. **Supervised target** — best-achievable 1337 wheat ceiling per location from
   `pyaez_1337_yields.parquet` (low-input wheat, rainfed ∪ irrigated → max).
2. **Technology discount** — scale absolute yield so the median of *positive*
   fertile yields matches BAHS 515 kg/ha:
   `scale = 515 / median(yield_kg_dm_ha | yield > 0)`.
3. **Features** (location attributes we know):
   - 1337 climate: CHELSA-TraCE `*_target` (mean temperature, precipitation, seasonality)
   - EU5 statics: `soil_quality`, `topography`, `vegetation`, `climate`, `climate_winter`
   - Geography: calibrated lat/lon, absolute latitude
   - Modern GAEZ rainfed potentials for wheat + cool-cereal analogues (barley, rye, oats)
   - Engineered: aridity proxy `precip / (temp_C + 10)`, temperature distance from ~12 °C wheat optimum, severe-winter flag
4. **Model** — two `HistGradientBoostingRegressor`s (yield, suitable_fraction) with
   monotonic constraints on agronomic features; then:
   - `production_density = yield × suitable_fraction`
   - `suitability_class` = 1…9 bins from suitable_fraction (1 = best)
5. Publish units match GAEZ mapmodes: **kg DM / km²** (×100 from kg/ha).

## Assumptions (validation gate)

| ID | Assumption | Gate |
|----|------------|------|
| A1 | Near-zero wheat potential in Antarctica / polar ice (`climate=arctic` + high |lat|) | mean yield ≈ 0 |
| A2 | Near-zero in hyper-arid desert interiors (`climate=arid` + desert vegetation + low precip) | mean yield low |
| A3 | Near-zero in hot-wet equatorial lowlands (`climate=tropical` + jungle) | mean yield low vs temperate |
| A4 | High potential in NW Europe oceanic / continental temperate belt | mean yield ≫ arid/arctic |
| A5 | Global positive-yield median near BAHS 515 kg/ha (±40%) after calibration | distribution check |
| A6 | Suitability falls as temperature leaves the wheat window | correlation / monotonic |

## Outputs

| Layer id | Label | Unit |
|----------|-------|------|
| `pnp_wheat_production_density` | Production density | kg DM / km² |
| `pnp_wheat_yield` | Yield | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | Suitable fraction | fraction |
| `pnp_wheat_suitability_class` | Suitability class | 1 best → 9 worst |

Dataset selector: **P&P** · Good: **wheat** (v1).

## Latest training run (artifacts/pnp_models/pnp_wheat_model_card.json)

- Technology scale: **0.324** (BAHS 515 / median positive PyAEZ rainfed-priority yield)
- CV R² yield: **0.94**
- Validation: **all gates passed**
  - A1 arctic mean ≈ 18 kg/ha
  - A2 hyper-arid desert mean = 0 kg/ha
  - A3 tropical jungle ≪ temperate
  - A4 NW Europe mean ≈ 572 kg/ha
  - A5 positive median ≈ 455 kg/ha (within ±40% of 515)
  - A6 model fit OK
