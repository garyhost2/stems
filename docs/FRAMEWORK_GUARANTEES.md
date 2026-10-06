# What the framework guarantees, and what it does not

*Section for `docs/FRAMEWORK.md`; kept in a separate file on branch `framework/guarantees`
so it does not collide with the abstraction track, and merged from there.*

A framework is worth the guarantees it can state precisely and survive being attacked on.
This section states each one as a conditional — *this constraint is hard provided X* — names
the calibration X depends on, and gives the **measured** residual when X fails, from
`experiments/diagnostics/guarantee_residuals.py`. Nothing below is asserted from the design;
every number is copied from that script's output on branch `framework/guarantees`.

## S0. Notation

Symbols are the ones used in the code and in `docs/REPORT_2026-10.md`; they are defined once
here and not redefined later.

| symbol | meaning | unit |
| --- | --- | --- |
| `b` | building index, `b = 1..B` | — |
| `t` | control step; `dt = 1 h` throughout | — |
| `x_b` | state of charge of a store in building `b` | dimensionless, `[0, 1]` |
| `x_min, x_max` | state-of-charge band the shield must keep (`CBFConfig.SOC_min/SOC_max`, 0.1 / 0.9) | dimensionless |
| `a_b` | commanded action of building `b` | dimensionless, `[-1, 1]` |
| `C_b` | store capacity of building `b` | kWh |
| `p_nom_b` | store nominal power of building `b` | kW |
| `r_b` | one-step rate, `r_b = p_nom_b · dt / C_b` | 1/step |
| `P_cap` | district import cap (`FleetShield.cap`, `CBFConfig.P_grid_max`) | kW |
| `e_b(t)` | realised base (non-vehicle) import of building `b` | kW |
| `ê_b(t)` | the forecaster's prediction of `e_b(t)` | kW |
| `m(t)` | forecast safety margin: 95th percentile of the district one-step forecast error over the last 168 steps (`BaseLoadForecaster.margin`) | kW |
| `I(t)` | realised district import, `Σ_b max(e_b + d_b, 0)` | kW |
| `d_b(t)` | realised charger draw of building `b` | kW |
| `R(t)` | **cap residual**, `max(I(t) − P_cap, 0)` | kW |
| `s_b` | vehicle state of charge at a charger | dimensionless |
| `s*_b` | required state of charge at departure | dimensionless |
| `n_b` | steps remaining until departure | steps |
| `p_min_b, p_max_b` | charger minimum / maximum charging power | kW |

A *residual* is always the amount by which the realised quantity violated the constraint, in
the constraint's own units. A guarantee measured at exactly zero over an exhaustive sweep is
reported as **hard on that sweep**, never as hard in general.

The public surface the guarantees are stated over is `CBFShield.project`,
`FleetShield.project`, `DeadlineStorageBarrier.project`, and the three module functions
`schedule`, `schedule_executable`, `apply_dead_band`. Internals may move; these may not.

## S1. The battery state-of-charge band — hard, conditional on the plant model

**Promise.** For any commanded `a_b` and any `x_b ∈ [x_min, x_max]`, the state of charge after
the simulator's own battery step stays in `[x_min, x_max]`.

**Condition.** The shield is constructed with the environment's own battery
(`CBFShield(battery_model=STEMSEnvironment.battery_model())`), which on the real environment
is `BatteryModel.from_citylearn` — the exact nonlinear `citylearn.energy_model.Battery`,
including the power-efficiency curve, the capacity-power curve and the self-discharge
coefficient. This is what `experiments/controllers.py` passes (lines 495 and 769). The
projection is a 24-iteration bisection on that model (`BatteryModel.safe_interval`), not a
linearisation of it.

**Measured residual.** Exhaustive sweep over the 8 autosized `tx_travis_8b` batteries
(`C_b = [10.8, 6.6, 5.0, 9.7, 13.5, 5.4, 16.2, 16.0]` kWh, true
`r_b = [0.267, 0.244, 0.500, 0.516, 0.244, 0.533, 0.178, 0.208]` per step), 81 commanded
actions × 81 in-band start states × 8 buildings:

| shield's plant model | cases | breaches | rate | max below `x_min` | max above `x_max` |
| --- | --- | --- | --- | --- | --- |
| exact inverse of the plant | 52 488 | **0** | 0.0000 | 0.00000 | 0.00000 |
| uniform `r_b = 0.1` surrogate | 52 488 | 4 829 | 0.0920 | 0.10000 | 0.09390 |
| surrogate at half the true `r_b` | 52 488 | 1 223 | 0.0233 | 0.10000 | 0.05998 |

So the band is hard *because* the inverse is exact, and for no other reason. The audit's
hypothetical uniform `r_b = 0.1` — which understates the true rate by 1.8× to 5.3× on these
devices — breaches the band in 9.2 % of states, driving the floor all the way to `x = 0`. The
live code does **not** use that surrogate: `STEMSEnvironment._extract_battery_info` reads
`nominal_power/capacity` per building, and on the mock environment `r_b = 0.1` happens to be
the mock plant's exact rate (`_MockBuilding.MOCK_SOC_RATE = 0.1`). The surrogate rows above
are the cost of getting the calibration wrong, not a current defect.

**What is not promised.** From a start *outside* the band the shield returns the best one-step
recovery action, and the residual is whatever the plant cannot undo in one step. Over the full
`[0, 1]` sweep the exact inverse shows 972/65 448 = 1.49 % breaches, all of them
`x_0`-inherited, with max excess 0.050 above `x_max` and 0.000 below `x_min`. Recovery is
measured: from `x_0 = 0.00`, 0 of 8 buildings are still outside after one step; from
`x_0 = 1.00`, 5 of 8 are still outside after one step and 0 of 8 after two.

## S2. The district import cap — hard up to forecast error; residual measured

**Promise.** `FleetShield` plans the fleet against `P_cap − m(t)`, so the realised district
import respects `P_cap` whenever the one-step base-load forecast error stays inside the
margin `m(t)` the forecaster has calibrated.

**Condition / calibration.** `m(t)` is the 95th percentile of the district one-step forecast
error over a 168-step window (`BaseLoadForecaster(quantile=0.95, window=168)`); it is exactly
0 for the first 24 observations and whenever a replay forecast is used. By construction the
cap can therefore be exceeded in roughly 5 % of steps even when nothing is wrong — that is the
quantile, not a bug. The *observed* part of the base load (`non_shiftable_load` minus
`solar_generation` at the current step) carries no forecast error at all; the error is entirely
in the endogenous part (thermal, hot water) and in the later horizon steps.

**Measured residual.** 4 houses, `P_cap = 30` kW, 378 scored steps after a 126-step burn-in,
5 seeds, rule `lp`, synthetic base load with an unobserved AR(1) thermal component
(`guarantee_residuals.py cap`). "rate" is the fraction of steps with `R(t) > 0`; "worst max"
is the largest single-step residual over all 5 seeds.

| configuration | rate | Σ R [kWh] | mean max R [kW] | worst max R [kW] | mean `m` [kW] |
| --- | --- | --- | --- | --- | --- |
| margin on | 0.0048 | 0.48 | 0.352 | 0.660 | 1.943 |
| margin off | 0.0582 | 16.31 | 2.212 | 2.461 | 0.000 |
| margin on, step +1 kW/house at `t = 336` | 0.0058 | 0.61 | 0.422 | 0.698 | 1.969 |
| margin on, step +2 kW/house | 0.0090 | 0.93 | 0.452 | 0.698 | 1.969 |
| margin on, step +3 kW/house | 0.0153 | 2.02 | 0.928 | 1.741 | 1.969 |
| margin on, step +6 kW/house | 0.4481 | 1202.98 | 11.099 | 11.564 | 1.969 |
| margin on, innovation spread ×2 | 0.0074 | 1.22 | 0.674 | 1.836 | 2.385 |
| margin on, innovation spread ×4 | 0.0143 | 8.26 | 3.207 | 5.660 | 3.364 |
| margin on, innovation spread ×8 | 0.0804 | 164.44 | 16.685 | 22.381 | 5.221 |

Three things to read off this table.

1. **The margin earns its place.** It cuts the exceedance frequency 12× and the exceeded
   energy 34×, at a worst-case residual of 0.66 kW = 2.2 % of the cap.
2. **A persistent error is absorbed, an unpredictable one is not.** A step change in the
   unobserved load is tracked by the forecaster's persistence correction within one step, so
   +12 kW district (`+3` kW/house, six times the calibrated margin) still only lifts the
   exceedance rate from 0.5 % to 1.5 % and the worst residual to 1.74 kW. What the margin
   cannot absorb is a change in the *innovation*: multiplying the innovation spread by 8
   raises the worst residual to 22.4 kW = 75 % of the cap, and the margin — which adapts, to
   5.22 kW — lags it the whole way.
3. **There is a regime the shield cannot reach at all.** At `+6` kW/house the base load alone
   (≈ 38 kW) exceeds the 30 kW cap. There is no charging left to shed; the residual is 11.1 kW
   and it is not the shield's to fix. The shield does not say so — see S5.3.

In the saturated regime `P_cap = 16` kW, where the deadline set is infeasible every night and
the fleet is permanently pressed against the cap, the same comparison gives rate 0.0735 /
Σ 19.5 kWh / worst 2.82 kW with the margin, against 0.2725 / 98.0 kWh / 3.93 kW without it.

**Statement.** The district cap is **soft with a quantified residual**, not hard. With the
calibrated margin and a stationary error process the residual is ≤ 0.66 kW on a 30 kW cap
(2.2 %) and occurs on 0.5 % of steps. Any paper sentence claiming the cap is enforced must
carry those numbers.

## S3. Vehicle departure state of charge — hard when the joint set is non-empty, reported when not

**Promise.** On rule `lp`, `FleetShield.project` solves a joint linear programme over all
connected vehicles and the whole remaining horizon. When a schedule exists that brings every
vehicle to `s*_b` by `n_b` under the shared cap, the LP finds it. When none exists, the LP
admits a per-vehicle shortfall `d_b` penalised by big-M (`big = 1e4`, weighted by `C_b`) plus
a max-share term, and returns `feasible = False`, `total_shortfall_kwh` and a `worst_share`.
Shortfall is **spread**, not dumped on one vehicle.

**Condition.** `rule == "lp"`. The six myopic rules do not report feasibility at all — S5.1.

**Measured.** 3 vehicles, each owing 0.4 SOC of a 50 kWh battery, deadlines at 2/3/4 steps,
`p_max = 10` kW, one shared cap. Each vehicle is individually reachable at every cap in the
table; only the joint set fails.

| `P_cap` [kW] | feasible | shortfall [kWh] | worst share | Σ now [kW] |
| --- | --- | --- | --- | --- |
| 30.0 | True | 0.000 | 0.0000 | 30.000 |
| 20.0 | True | 0.000 | 0.0000 | 20.000 |
| 17.5 | True | 0.000 | 0.0000 | 17.500 |
| 15.0 | False | 5.000 | 0.0833 | 15.000 |
| 10.0 | False | 20.000 | 0.3333 | 10.000 |
| 5.0 | False | 40.000 | 0.6667 | 5.000 |

The transition is at 17.5 kW and the shortfall below it is exactly the missing energy; the LP
keeps charging at the full cap in every infeasible case rather than giving up. **This is the
right design** and it is why the deadline constraint must be described as *hard conditional on
feasibility*: the deadline is not a constraint the solver can always satisfy, so it is one the
solver must always *report* on.

**How often is the set empty?** It depends entirely on the sizing, so a single rate is
meaningless. In the synthetic rollout of S2 at `P_cap = 30` kW the LP reports infeasible on
13.0 % of steps (mean over 5 seeds, range 11.4–15.1 %) and the vehicles depart with a mean
total shortfall of 0.017 SOC; at `P_cap = 16` kW it reports infeasible on 57.7 % of steps and
every vehicle misses, by a mean 0.278 SOC. At `P_cap = 45` kW: 2 of 286 connected steps
flagged, 0 of 80 departures missed. The framework's claim is the *correspondence* between the
flag and the outcome, not a rate.

## S4. Comfort — deliberately soft on CityLearn

Comfort is a quadratic penalty with weight 0.4 in the reward, and the hard-comfort barrier
(`stems/comfort.py`) is gated off on CityLearn. This is a decision, not an omission, and the
reason is that CityLearn's learned thermal model is not physical: a cooling action of −0.25
drops an indoor temperature by 4–13 °C in one hour on these buildings (audit §4A, CHANGELOG
step 3). A hard barrier on an indoor temperature that the simulator moves by 13 °C per step
would be enforcing an invariant of a model nobody believes, and the resulting
"zero comfort violations" would be an artefact of the thermal model rather than evidence about
the controller.

**Statement.** Comfort is soft on CityLearn by design. The comfort barrier exists, is tested,
and is the mechanism intended for real building data, where an RC model with a measured time
constant makes a one-step reachability claim meaningful. Until then, comfort is **scored, not
enforced**, and every comfort number in the paper must be labelled as such.

## S5. Where the framework does *not* surface a breach

These are the findings of the adversarial suite (`tests/test_guarantees.py`). Each is a case
where the constraint is broken and the caller is not told. They are listed here rather than
fixed quietly, because the cost of a silent breach in a safety framework is the claim, not the
kilowatt.

### S5.1 The six myopic rules never report infeasibility — and a published metric is therefore identically zero

`FleetShield.project` writes `feasible` and `shortfall_kwh` into `self.last` **only** on rule
`lp`. On `independent`, `static`, `proportional`, `edf`, `llf`, `sllf` those keys are absent.
Measured on two vehicles each owing 25 kWh in one step under a 5 kW cap — 50 kWh owed, 5 kWh
available, as infeasible as a set can be:

| rule | `feasible` | `shortfall_kwh` | `binding` | predicted import [kW] |
| --- | --- | --- | --- | --- |
| independent | ABSENT | ABSENT | False | 20.000 |
| static | ABSENT | ABSENT | True | 5.000 |
| proportional | ABSENT | ABSENT | True | 5.000 |
| edf | ABSENT | ABSENT | True | 5.000 |
| llf | ABSENT | ABSENT | True | 5.000 |
| sllf | ABSENT | ABSENT | True | 5.000 |
| lp | **False** | **40.0** | True | 5.000 |

`experiments/ev_coupling.py:135` counts infeasible hours as
`int(shield.last.get("feasible") is False)`. With the key absent this evaluates to `0` for
every myopic rule at every step, so the reported `infeasible_hour_rate` is **structurally
0.000 for six of the seven rules** — not a measurement, an artefact of a missing dictionary
key. It is currently tabulated only for `lp` (`experiments/ev_report.py:123`), which is the
only reason this has not already produced a wrong number in a table, but
`experiments/diagnostics/replay_stored_runs.py:19` lists it among the keys it prints for any
arm. **Any future table that puts `infeasible_hour_rate` next to a myopic rule will report a
perfect score for a rule that cannot fail by construction.**

### S5.2 `apply_dead_band` rounds an urgent charger *up*, past the cap, and says nothing

`EVFleetModel.applied_kw` clips a non-zero action into `[p_min_b, p_max_b]`: a charger is off
or at least `p_min_b`, with nothing in between. `apply_dead_band` resolves an allocation inside
that gap by rounding **up to `p_min_b` for an urgent vehicle** (laxity ≤ 0) and down to zero
otherwise. The round-up can exceed the cap the allocator just respected, and
`apply_dead_band` has no report channel — it returns an array.

Isolated on the public functions, 2 vehicles, `p_min = 4` kW, `P_cap = 2` kW, vehicle 1 urgent:

```
allocate        -> [2.0, 0.0]  sum 2.000 kW <= P_cap 2.0 kW
apply_dead_band -> [4.0, 0.0]  sum 4.000 kW vs P_cap 2.0 kW, residual 2.000 kW
```

Through `FleetShield.project` the same step gives a predicted import of 4.000 kW against a
2.0 kW cap — a residual of 2.000 kW, **100 % of the cap** — on all five coordinating myopic
rules, each reporting `binding: True`, no `feasible` key, and no breach flag. The LP refuses
instead: 0.000 kW drawn, `feasible: False`.

Two mitigations, both real: `predicted_import_kw` does carry the post-round number (4.000), so
the information exists in `self.last` — nothing compares it to `self.cap`; and
`experiments/ev_coupling.py` measures `cap_exceed_kwh` from the realised net import, so the
breach is visible *ex post* in the experiment even though the shield never admits it. On the
`lp` rule the round-up is unreachable in practice: over 400 random fleets
(`guarantee_residuals.py converge`), `schedule_executable`'s dead-band second pass converged
every time, leaving **0** sub-`p_min` allocations for `apply_dead_band` to act on.

The deeper point is not the kilowatt, it is that two rules in the same framework resolve the
same deadline-versus-cap conflict in **opposite directions** — myopic rules favour the
deadline and break the cap, the LP favours the cap and misses the deadline — with no record of
the choice anywhere. A framework that claims a priority ordering has to state which one wins.

### S5.3 When the cap is below the inflexible load, the shield reports `feasible: True`

`FleetShield._shed_storage_charging` bisects the house battery and tank charging down to the
state-of-charge recovery floor, then returns. If the inflexible load alone still exceeds the
cap, nothing says so. One house, 8 kW inflexible load, 5 kW battery charge requested, no
vehicle:

| `P_cap` [kW] | predicted import [kW] | residual [kW] | shed [kW] | `feasible` | `binding` |
| --- | --- | --- | --- | --- | --- |
| 10.0 | 10.000 | 0.000 | 3.000 | True | False |
| 6.0 | 8.000 | **2.000** | 5.000 | True | False |
| 3.0 | 8.000 | **5.000** | 5.000 | True | False |
| 1.0 | 8.000 | **7.000** | 5.000 | True | False |

This is the worst of the three, because it is a *positive false claim* rather than an absence:
the report asserts `feasible: True` and `binding: False` while predicting an import 7× the
cap. `feasible: True` here comes from `schedule`'s `"nothing to charge"` early return — no
vehicle is connected, so the LP never runs, and the key means "the vehicle deadlines are
satisfiable", which is vacuously true. The key is correct for what it measures and badly
wrong for what a caller will read it as.

The same shape appears on the heat-pump guard. `CBFShield._apply_hvac_power_guard` bisects a
common shed factor to zero and stops; with 4 buildings at 40 kW of base load each against a
95 kW derated cap the import stays at 160 kW, a residual of 65 kW, and `CBFShield.project`
returns only an action array — it has no `last`, no report attribute, nothing.

### S5.4 On the `CBFShield` path, deadline infeasibility is never surfaced at all

`CBFShield.feasibility_report` computes exactly the right thing — `coupled_feasibility` plus a
`prioritise` allocation naming which device misses by how much — and **no live code path calls
it**. The only callers in the repository are `tests/test_deadline.py`. So the shield used by
every RL arm (`experiments/controllers.py:495, 769`) carries a working infeasibility detector
that never runs.

Worse, the two coordination modes silently override the barrier they are supposed to
coordinate. Three stores each owing 0.7 SOC of 50 kWh in one step at 6 kW, `P_grid_max` 4 kW
(3.8 kW after the 5 % derate):

| `coordination` | barrier demands `a` | shield returns `a` | draw [kW] | grid cap [kW] |
| --- | --- | --- | --- | --- |
| `independent` (the live default) | [1.000, 1.000, 1.000] | [1.000, 1.000, 1.000] | 18.00 | 3.80 |
| `proportional` | [1.000, 1.000, 1.000] | [0.211, 0.211, 0.211] | 3.80 | 3.80 |
| `edf` | [1.000, 1.000, 1.000] | [0.633, 0.000, 0.000] | 3.80 | 3.80 |

Under `independent` — which is what `experiments/controllers.py` constructs, since it never
passes `coordination` — the district cap is **not applied to deadline-barrier actions at all**:
18 kW drawn against a 3.8 kW cap. Under the other two the cap is enforced by scaling the
barrier's forced floor down by 4.7×, i.e. the hard deadline constraint is overridden, with no
record. Meanwhile `feasibility_report()` on the same state returns
`feasible=False, required=105.0 kWh, available=3.8 kWh, shortfall=101.2 kWh` — the whole
answer, uncalled.

### S5.5 `reserve_hours` clips an unreachable target to 1.0 and then reports success

`FleetShield.state` inflates the required departure state of charge to compensate for the idle
self-discharge over the reserved hours, `target = min(s* / (1−loss)^idle, 1.0)`. When the
compensated target exceeds 1.0 the clip is physically correct — the battery cannot go above
full — but the resulting *unavoidable* shortfall at departure is not reported: the LP meets
the clipped target of 1.0 and returns `feasible = True`. One vehicle, 5 %/step self-discharge,
`s* = 0.90`, 23 steps to departure:

| `reserve_hours` | slots | target set | target needed | clipped | departure `s` | residual SOC |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 24 | 0.9000 | 0.9000 | False | 0.9000 | 0.0000 |
| 2 | 22 | 0.9972 | 0.9972 | False | 0.9000 | 0.0000 |
| 4 | 20 | 1.0000 | 1.1050 | True | 0.8145 | **0.0855** |
| 6 | 18 | 1.0000 | 1.2243 | True | 0.7351 | **0.1649** |
| 12 | 12 | 1.0000 | 1.6656 | True | 0.5404 | **0.3596** |

This bites only when the vehicle battery has a non-zero `loss_coefficient` *and*
`reserve_hours > 0`; `experiments/ev_coupling.py`'s `reserve` stage sets the latter. The
`tx_travis_8b_ev` vehicle batteries carry no explicit `loss_coefficient` in the schema, so the
residual on the shipped scenario is zero — but the clip is unconditional and nothing warns.

### S5.6 A plant model mis-calibrated against the simulator breaks the cap silently

`HouseStorage.draw_kw` predicts the house draw with the shield's own `TankModel`. If that
model's heater efficiency does not match the plant's, the prediction is wrong by the
difference and the cap is missed by that amount, with `predicted_import_kw` reporting exactly
the cap. 8 kW inflexible load, tank charge action 1.0, `P_cap = 12` kW, true heater efficiency
0.85:

| shield's `η_heater` | predicted [kW] | realised [kW] | error [kW] | residual [kW] |
| --- | --- | --- | --- | --- |
| 0.60 | 12.000 | 10.824 | −1.176 | 0.000 |
| 0.85 | 12.000 | 12.000 | 0.000 | 0.000 |
| 1.00 | 12.000 | 12.706 | +0.706 | **0.706** |

An optimistic model (η too high) under-predicts the draw and breaches by 0.706 kW = 5.9 % of
the cap while reporting the import as exactly 12.000 kW. A pessimistic one is safe and
wasteful. The guarantee therefore depends on `TankModel.from_citylearn` /
`BatteryModel.from_citylearn` being exact — the same calibration S1 rests on, stated once and
relied on twice.

## S6. What is enforced, what is only scored, what does not exist

| constraint | status | mechanism | residual |
| --- | --- | --- | --- |
| battery state-of-charge band | **hard** | exact inverse of `citylearn.energy_model.Battery` in `BatteryModel.safe_interval` | 0 / 52 488 in-band cases (S1) |
| district import cap, `lp` rule | **soft, quantified** | joint LP against `P_cap − m(t)` | ≤ 0.66 kW on 30 kW, 0.5 % of steps (S2) |
| district import cap, myopic rules | **soft, unreported** | `allocate` respects it, `apply_dead_band` can undo it | up to 100 % of cap (S5.2) |
| district import cap, CBF path | **not enforced** by default | `coordination="independent"` ignores it | 18 kW on a 3.8 kW cap (S5.4) |
| EV departure state of charge | **hard when feasible, reported when not** (`lp` only) | big-M shortfall in the LP | exact: shortfall = missing energy (S3) |
| EV departure state of charge, myopic rules | **best-effort, unreported** | priority heuristics | no feasibility key at all (S5.1) |
| hot-water / Legionella deadline (CBF path) | **best-effort, unreported** | `DeadlineStorageBarrier.project` forces a floor; nothing checks it was kept | detector written and uncalled (S5.4) |
| thermal comfort | **soft by design** | quadratic reward penalty, weight 0.4 | not applicable — scored, not enforced (S4) |
| per-building power limit | **partially enforced** | `CBFShield._apply_power_guard` clips the battery action against `P_building_max`; the heat-pump stage is reached only when a `cop_model` is passed, which `experiments/controllers.py` does not do | not measured here |
| battery degradation | **scored only** | `stems/degradation.py` computes a capacity-fade cost; no projection, no barrier | — |
| transformer / thermal rating limits | **does not exist** | — | — |
| grid voltage, phase balance, network constraints | **does not exist** | — | — |

## S7. Reproducing the numbers

```sh
export XDG_CACHE_HOME=$PWD/.citylearn_cache
python experiments/diagnostics/guarantee_residuals.py            # every section
python experiments/diagnostics/guarantee_residuals.py soc cap    # named sections
python -m pytest tests/test_guarantees.py -q                     # the adversarial suite
```

Sections: `soc`, `cap`, `deadline`, `report`, `cbf`, `converge`, `tank`. The `soc` and `cap`
sections take a few minutes each; `--quick` subsamples them for a smoke run and must not be
used for quoted numbers.
