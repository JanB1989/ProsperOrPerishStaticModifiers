# P&P wheat — 1337 historic-supervised mapmodes

## Objective

1. Merge historic yield data that matches EU5’s wheat definition.
2. Convert / scale onto a medieval kg/ha footing (Pretty–BAHS calibration).
3. Train a model of **yield from location attributes**.
4. Deploy that function to the rest of the world.

No EU5 region multipliers. No constant “recipe plateaus.”

## Historic training set

Source: BAHS *Three centuries of English crop yields* (`bahs_medieval_yield_observations.parquet`).

| Step | Detail |
|------|--------|
| Filter | `model_crop == wheat`, not mixtures/aggregates, years 1211–1450 |
| Unit | Yield-per-seed → kg/ha via Pretty (1990) seed rate: **515 ÷ 4.0 = 128.75 kg seed/ha** |
| Geography | Each observation joins to arable vegetation locations in its mapped `eu5_province` (~18 English provinces, ~69 locations) |
| Negatives | Arctic, tropical jungle, desert-without-river → y = 0 |
| Soft prior | PyAEZ wheat (irrigated allowed where `has_river`) **scaled** so median on BAHS locations matches historic median — continuous global coverage at medieval intensity |

Hard BAHS rows are high weight; soft physical rows are low weight. No lat/lon, no Europe-only layers, no climate one-hots (those memorized England).

Audit artifacts: `artifacts/pnp_models/pnp_wheat_historic_labels.parquet`.

## Features

Curated attributes from `location_candidates` + GAEZ wide + external + PyAEZ-as-feature (climate/soil/water/GAEZ analogues). **Never** `region` / `super_region`.

## Model

Two `HistGradientBoostingRegressor`s trained **only** on historic (+ hostile-zero) rows; then predict all locations.

- `production_density = yield × suitable_fraction`
- Hostile climates soft-zeroed at predict time

## Validation

- A*: arctic / desert-no-river / tropics / NW Europe / BAHS median band / CV R²
- H*: fit on labeled historic locations
- D*: enough distinct 10 kg/ha bins (continuous distribution check)

## Outputs

| Layer | Unit |
|-------|------|
| `pnp_wheat_production_density` | kg DM / km² |
| `pnp_wheat_yield` | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | fraction |
| `pnp_wheat_suitability_class` | 1 best → 9 worst |

## Limits

Historic labels are **England-centric**. The global map is an attribute extrapolation from medieval English wheat yields plus hostile zeros. Enriching with more non-English absolute yield series (when mappable to `location_tag`) is the way to improve coverage — append observations, retrain.
