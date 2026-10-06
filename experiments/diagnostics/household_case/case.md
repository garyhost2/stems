# E0 -- the household-benefit case

Generated 2026-10-06 16:17:16. Scenario `tx_travis_8b__year-insample__refn8__cap300-80`, window 0-8759 (8759 transitions), 8 households, seed 0.

## The case, pinned

- Comfort band: +/-1.0 degC around the CityLearn set points, occupied hours only.
- District import cap: 300.0 kW.
- EV penetration: None.

### Tariffs

- **`citylearn_tou`** -- pricing.csv shipped with the CityLearn citylearn_challenge_2022_phase_all dataset (Nweye, Sankaranarayanan & Nagy 2023, doi:10.18738/T8/0YLJ6Q). Time-of-use, five distinct prices spanning 0.21-0.54 USD/kWh, peak 16:00-20:00 local. The neighbourhood is in Fontana, California (Nweye et al., NeurIPS 2022 Competition Track, PMLR 220:85-103), NOT Travis County, Texas.

- **`austin_standard`** -- City of Austin Electric Tariff effective 2025-11-01 (FY2026), https://austinenergy.com/-/media/project/websites/shared/pdfs/rates/tariff.pdf, retrieved 2026-10-06; Residential Service Inside City Limits (p. 4), Regulatory Charge Secondary energy basis (p. 27). Customer charge 16.50 USD/mo; tiers 4.640/5.138/7.525/10.884 c/kWh at 300/900/2000 kWh; flat riders PSA 4.118 + CBC 1.275 + Regulatory 1.338 c/kWh. No time-of-use component.

- **`austin_tou_pilot`** -- City of Austin Electric Tariff effective 2025-11-01 (FY2026), https://austinenergy.com/-/media/project/websites/shared/pdfs/rates/tariff.pdf, retrieved 2026-10-06; Residential Service Inside City Limits (p. 4) with the Pilot Programs Residential Time-of-Use power supply charges (p. 37) in lieu of the standard PSA: off-peak 2.677, mid-peak 4.118, on-peak 8.442 c/kWh; weekday on-peak 15:00-18:00, mid-peak 07:00-15:00 and 18:00-22:00, off-peak 22:00-07:00, weekends off-peak all day. Pilot limited to 100 individual meters (p. 36).

### Emission factors

- **`citylearn_hourly`** -- carbon_intensity.csv shipped with citylearn_challenge_2022_phase_all (doi:10.18738/T8/0YLJ6Q); hourly, mean 0.1565 kg CO2/kWh over 8760 h. California grid, NOT ERCOT.

- **`erct_annual`** -- US EPA, Greenhouse Gas Equivalencies Calculator - Calculations and References, eGRID Regions table, ERCT 'ERCOT All' total output (baseload) rate, eGRID2022 (year 2022 data) [771.1 lb CO2/MWh, delivered basis]

## Price signal, by tariff

| tariff | mean marginal USD/kWh | min | max | max/min | time-varying |
|---|---:|---:|---:|---:|:--:|
| `citylearn_tou` | 0.27314 | 0.21000 | 0.54000 | 2.571 | yes |
| `austin_standard` | 0.11371 | 0.11371 | 0.11371 | 1.000 | no |
| `austin_tou_pilot` | 0.10961 | 0.09930 | 0.15695 | 1.581 | yes |

## Annual household outcome, by arm and case

| arm | tariff | emissions | bill USD | CO2 kg | comfort in band | own peak kW | coincident peak kW |
|---|---|---|---:|---:|---:|---:|---:|
| `idle` | `citylearn_tou` | `citylearn_hourly` | 2872.63 | 1523.4 | 85.5% | 11.44 | 5.24 |
| `idle` | `citylearn_tou` | `erct_annual` | 2872.63 | 3518.3 | 85.5% | 11.44 | 5.24 |
| `idle` | `austin_standard` | `citylearn_hourly` | 1345.69 | 1523.4 | 85.5% | 11.44 | 5.24 |
| `idle` | `austin_standard` | `erct_annual` | 1345.69 | 3518.3 | 85.5% | 11.44 | 5.24 |
| `idle` | `austin_tou_pilot` | `citylearn_hourly` | 1286.58 | 1523.4 | 85.5% | 11.44 | 5.24 |
| `idle` | `austin_tou_pilot` | `erct_annual` | 1286.58 | 3518.3 | 85.5% | 11.44 | 5.24 |
| `rbc` | `citylearn_tou` | `citylearn_hourly` | 2346.62 | 1442.8 | 85.5% | 12.31 | 8.01 |
| `rbc` | `citylearn_tou` | `erct_annual` | 2346.62 | 3372.6 | 85.5% | 12.31 | 8.01 |
| `rbc` | `austin_standard` | `citylearn_hourly` | 1296.58 | 1442.8 | 85.5% | 12.31 | 8.01 |
| `rbc` | `austin_standard` | `erct_annual` | 1296.58 | 3372.6 | 85.5% | 12.31 | 8.01 |
| `rbc` | `austin_tou_pilot` | `citylearn_hourly` | 1230.83 | 1442.8 | 85.5% | 12.31 | 8.01 |
| `rbc` | `austin_tou_pilot` | `erct_annual` | 1230.83 | 3372.6 | 85.5% | 12.31 | 8.01 |

## Austin Energy with the Value-of-Solar rider

Gross-billed, production-credited: billable usage is on-site photovoltaic self-consumption plus delivered energy, and the credit is on gross photovoltaic output. Net metering is not available.

| arm | tariff | gross-billed USD | VoS credit USD | net bill USD |
|---|---|---:|---:|---:|
| `idle` | `austin_standard` | 2199.79 | 1243.02 | 956.77 |
| `idle` | `austin_tou_pilot` | 2126.62 | 1243.02 | 883.60 |
| `rbc` | `austin_standard` | 2340.81 | 1243.02 | 1097.79 |
| `rbc` | `austin_tou_pilot` | 2254.85 | 1243.02 | 1011.83 |

## Is the district cap binding?

Scenario default cap: **300.0 kW**. Measured district import peak, reference arm `idle`: **41.88 kW** (14.0% of the cap), mean 8.72 kW, hours above the cap **0** of 8759.

A cap of 300.0 kW cannot bind on this neighbourhood, so no grid-constraint arm can be distinguished from an unconstrained one under it. A cap that binds for this stock is about **34 kW** (80% of the uncontrolled peak).

## Saving of each arm against the reference, per household per year

Reference arm: `idle`.

| arm | tariff | emissions | bill saved USD | bill saved % | CO2 avoided kg | import avoided kWh | export forgone kWh | gross load saved kWh | coincident peak cut kW |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rbc` | `citylearn_tou` | `citylearn_hourly` | 526.02 | 17.90% | 80.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
| `rbc` | `citylearn_tou` | `erct_annual` | 526.02 | 17.90% | 145.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
| `rbc` | `austin_standard` | `citylearn_hourly` | 49.10 | 3.50% | 80.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
| `rbc` | `austin_standard` | `erct_annual` | 49.10 | 3.50% | 145.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
| `rbc` | `austin_tou_pilot` | `citylearn_hourly` | 55.75 | 4.19% | 80.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
| `rbc` | `austin_tou_pilot` | `erct_annual` | 55.75 | 4.19% | 145.7 | 395.4 | 1376.8 | -981.4 | -2.774 |
