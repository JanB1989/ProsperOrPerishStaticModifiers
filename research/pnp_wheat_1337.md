# P&P wheat — 1337 historic + GAEZ-shape mapmodes

## Objective

1. Use historic yield observations (BAHS) as **hard labels**.
2. Use modern GAEZ low-input wheat as the **soft geographic shape**, scaled to medieval absolute intensity.
3. Train yield from location **attributes** (GAEZ/PyAEZ/climate/soil/water as features).
4. Treat historical breadbasket beliefs as **post-hoc assumption tests**, not training recipes.

No EU5 region multipliers. No constant “recipe plateaus.” No hard-coded Nile/Punjab kg/ha targets.

## Historic hard labels

Source: BAHS *Three centuries of English crop yields* (`bahs_medieval_yield_observations.parquet`).

| Step | Detail |
|------|--------|
| Filter | `model_crop == wheat`, not mixtures/aggregates, years 1211–1450 |
| Unit | Yield-per-seed → kg/ha via Pretty (1990) seed rate: **515 ÷ 4.0 = 128.75 kg seed/ha** |
| Geography | Each observation joins to arable vegetation locations in its mapped `eu5_province` |
| Negatives | Arctic, tropical jungle, desert-without-river → y = 0 |

## Soft prior (data shape, not lore)

GAEZ wheat yield (published kg/suitable-km² ÷ 100 → kg/ha):

- no river → rainfed
- `has_river` → `max(rainfed, irrigated)`

Scale so median on BAHS locations matches the historic BAHS median. Soft weight ~0.12; BAHS hard rows stay high weight. **PyAEZ is a feature only**, not the soft teacher.

Suitability soft targets are irrig-aware the same way.

## Features

Curated attributes from `location_candidates` + GAEZ wide + external + PyAEZ-as-feature. **Never** `region` / `super_region` as model inputs (`region` may be carried for assumption tests only).

## Validation

- **A\*/H\*/D\*** physical gates gate the build (arctic / desert / tropics / NW Europe / BAHS band / CV / historic fit / continuity).
- **T\*** assumption tests report whether the map matches ~1300 breadbasket expectations (Nile > Britain farmland; France/steppes/Sicily upper half; Maghreb/Punjab not bottom Old-World quartile; Indonesia near zero). They do **not** write labels and do **not** fail the build.

## Outputs

| Layer | Unit |
|-------|------|
| `pnp_wheat_production_density` | kg DM / km² |
| `pnp_wheat_yield` | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | fraction |
| `pnp_wheat_suitability_class` | 1 best → 9 worst |

## Limits

Hard absolute yields are still England-centric. Global rank-order comes from GAEZ shape at medieval scale. Failed T\* tests mean the data teacher disagrees with historical lore — fix by adding mappable non-English absolute yields, not by baking lore into labels.
