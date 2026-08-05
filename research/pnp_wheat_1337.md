# P&P wheat — global historic anchors + GAEZ shape

## Objective

Train wheat yield from **all available historic absolute data** (not England alone), using attributes as features, and treat breadbasket lore as gates that must pass.

## Hard labels

1. **BAHS** English observation-level wheat YPS → kg/ha (Pretty seed rate).
2. **Worldwide absolute evidence** from [`research/pnp_wheat_evidence.json`](pnp_wheat_evidence.json) — Nile, Indo-Gangetic, Maghreb, Anatolia, N. China, Mediterranean, Fertile Crescent, temperate demesne, plus near-zero hostiles — matched by agronomic attributes (never EU5 region multipliers).
3. Within each evidence stratum, reshape irrig-aware GAEZ so the stratum median equals the published absolute target (continuous within stratum).

## Soft prior

GAEZ low-input wheat (irrigated where river), scaled to BAHS median on BAHS sites, low weight, only where hard evidence is absent.

## Features

Climate / soil / water / GAEZ / PyAEZ attributes. No `region` as a model feature.

## Validation (all must pass)

- Physical A\*/H\*/D\* gates
- Assumption T\* gates (Nile > Britain farmland; France/steppes/Sicily upper half; Maghreb/Punjab not bottom quartile; Indonesia near zero)

## Outputs

| Layer | Unit |
|-------|------|
| `pnp_wheat_production_density` | kg DM / km² |
| `pnp_wheat_yield` | kg DM / suitable km² |
| `pnp_wheat_suitable_fraction` | fraction |
| `pnp_wheat_suitability_class` | 1 best → 9 worst |
