# The STEMS flexibility framework

*One abstraction, four loads, one cap. Written 2026-10-06 on branch `framework/abstraction`.*

This document states the object the repository is built around. It is not a summary of
results; `docs/REPORT_2026-10.md` holds those. Every symbol below is the name the code
uses. Where a symbol appears in a figure, a test or a paper, it appears under this name.

---

## 1. The object

> **A household flexible load is a store with a rate limit that must reach a required
> level by a deadline. A neighbourhood is a set of such loads under one shared import
> cap.**

That sentence is the whole framework. Everything else here is its arithmetic.

For device `i` at control step `t`, with step length `dt` hours:

```
                      soc[i,t]  in  [0, 1]                               the store
       soc[i,t+1] - soc[i,t]  <=  rate[i]                          the rate limit
                      soc[i,t]  >=  required_soc[i,t]   whenever  active[i,t]   and
                                                        steps_to_deadline[i,t] = 0
                                                             the required level and deadline
   sum_i  import_kw[i,t]  <=  cap                                   the shared cap
```

The first three lines are `stems.deadline.DeadlineStorageBarrier`. The fourth is
`stems.fleet.schedule` (a linear programme) or `stems.cbf.CBFShield` (a priority rule).

Two things follow from writing it this way, and they are the reason to write it this
way at all.

**One device type.** An electric vehicle, a hot-water tank, the weekly Legionella
disinfection cycle and a house battery differ in what fills their store, what sets
`required_soc`, and when `steps_to_deadline` hits zero. They do not differ in the
constraint, in the projection, or in how they contend for the cap. Section 5 lists the
four, and each is a set of arguments to one constructor, not a code path.

**The deadline is the flexibility.** `slack = steps_to_deadline - steps_needed` is the
number of steps the controller may still choose freely. While `slack > 0` the barrier
is inert and the policy decides; at `slack <= 0` the barrier forces the minimum
charging that still meets the deadline. A constraint stated as a deadline therefore
*hands the policy an explicit budget of discretion*, which a per-step penalty does not.

---

## 2. Symbols

Units are the device-interface units CityLearn reports and the repository carries
throughout: energy in kWh, power in kW, time in hours, temperature in degC, state of
charge dimensionless. One kWh is 3.6 MJ exactly; the conversion is applied in
`stems/comfort.py`, which works in SI base units (J/K, W/K, s) because an RC envelope
has no natural kWh scale. No other module mixes the two.

| symbol | code | meaning | unit |
|---|---|---|---|
| `soc` | `soc`, `soc_fn(obs_list)` | state of charge of the store | — (in [0, 1]) |
| `capacity` | `capacity` | energy the store holds at `soc = 1` | kWh |
| `rate` | `rate`, `current_rate(obs_list)` | state of charge gained in one step at the full action | per step |
| `action_bound` | `action_bound` | largest value of this load's action column | — |
| `action_index` | `action_index` | which column of the action matrix this load owns | — |
| `efficiency` | `efficiency` | energy into the store per unit of energy drawn | — |
| `dt` | `dt`, `hours_per_step` | control step length | h |
| `required_soc` | `required_soc` | level owed at the deadline, `soc_req + margin`, clipped to `soc_cap` | — |
| `margin` | `margin` | safety addition to the required level | — |
| `soc_cap` | `soc_cap` | ceiling on the required level (a store need not be filled to serve) | — |
| `active` | `active` | the requirement applies at this step (vehicle plugged in, zone occupied) | bool |
| `steps_to_deadline` | `steps_to_deadline` | steps remaining until the level is owed | steps |
| `gap` | `gap` | `max(required_soc - soc, 0)`, what is still owed | — |
| `steps_needed` | `steps_needed` | `ceil(gap / rate)`, steps of full charging to close the gap | steps |
| `slack` | `slack` | `steps_to_deadline - steps_needed`, discretion left | steps |
| `deficit_kwh` | `deficit_kwh` | `gap * capacity`, the owed energy in the store | kWh |
| — | `energy_still_required_kwh` | `gap * capacity / efficiency`, the owed energy at the meter | kWh |
| `cap` | `cap`, `CBFConfig.P_grid_max` | shared import cap over the set of loads | kW |
| `base` | `base` | forecast import of everything that is not this load, per step ahead | kW |
| `slots` | `slots` | charging opportunities left before the deadline | steps |
| `target` | `target` | `required_soc` corrected for standing loss over idle steps | — |
| `now_kw` | `now_kw` | power the cap shield grants this load this step | kW |
| `p_max`, `p_min` | `p_max`, `p_min` | charger power limits; **the attainable set is `{0} u [p_min, p_max]`** | kW |
| `SOC_min`, `SOC_max` | `CBFConfig.SOC_min/SOC_max` | band the store must stay inside at all times | — |
| `P_building_max` | `CBFConfig.P_building_max` | per-building import cap | kW |

---

## 3. Two constraint faces, one store

A store carries two different constraints and the framework keeps them apart, because
they are enforced by different code and fail in different ways.

**The band**, `SOC_min <= soc <= SOC_max`, holds at *every* step. It is enforced by
inverting the plant exactly: `BatteryModel.safe_interval` returns the interval of
actions whose one-step successor state lands inside the band, and the shield clips the
policy's action into it. There is no deadline and nothing is owed.

**The deadline**, `soc >= required_soc` at `steps_to_deadline = 0`, holds at *one*
step. It is enforced by `DeadlineStorageBarrier.project`, which raises the action to
`a_min` once `slack <= 0` and leaves it alone otherwise.

A house battery has only a band. An electric vehicle has both. The hot-water tank and
the Legionella cycle have deadlines on the same store, with different periods. Writing
the two faces separately is what lets the battery be the *degenerate* member of the
family — `active` false everywhere — rather than a fifth code path.

### Exactness, and where it stops

The projection is only a safety guarantee if the inverse it uses is the plant's own.
Measured on the real CityLearn simulator by
`experiments/plant_projection_error.py` (written to
`experiments/diagnostics/plant_projection/projection_error.json`):

| plant model | forward error vs simulator | inverse round-trip | samples |
|---|---|---|---|
| `BatteryModel` | 2.98e-08 soc (max) | 5.34e-13 soc (max) | 938 steps, 70.2% unsaturated |
| `TankModel` | 3.27e-07 kWh/step (max) | 9.49e-13 soc (max) | 1912 |
| `EVFleetModel` | 9.78e-05 soc (max); 7.36e-03 kW on the draw | 6.55e-07 kW (max) | 981 |

Three caveats belong with those numbers, not under them.

1. The battery figure excludes saturated steps — 29.8% of the rollout, where the
   device clips at its depth-of-discharge floor or at `soc = 0.95`. There the model
   does not agree with the plant; it is *conservative in the protected direction*,
   which `tests/test_battery_model.py::test_model_errs_toward_the_bound_being_protected`
   pins and which is the property a safety projection needs.
2. The charger's minimum power makes the EV action set non-convex: targets inside
   `(0, p_min)` are unattainable, and asking for them misses by up to 1.038 kW on this
   schema (`p_min = 1.4 kW`). That is not solver error. `stems.fleet.apply_dead_band`
   resolves it — round up to `p_min` when the deadline is binding, down to zero
   otherwise — and `schedule_executable` re-solves the cap programme under that choice.
3. **`DeadlineStorageBarrier` does not currently call these inverses.** Its `a_min`
   comes from `action_for_soc_gain`, a linear map `gap / rate * action_bound`, and
   `EVReadinessBarrier` derates `rate` by `rate_derate = 0.85` to stay on the safe side
   of it. The exact inverses above are used by the cap shield and the model-predictive
   controller. `CHANGELOG.md`, abstraction track step 2.2, records the seam that closes
   this (`DeadlineStorageBarrier(plant=...)`, off by default) and why it was not turned
   on in a refactor.

---

## 4. The shared cap

A set of loads under one cap is not the same problem as each load under its own share.
Two devices can each be feasible alone and jointly infeasible, and the only way to know
is to solve them together.

`coupled_feasibility(barriers, obs_list, power_cap_kw, dt_hours)` answers the question
cheaply and conservatively: total `energy_still_required_kwh` against
`power_cap_kw * dt_hours * horizon_steps`, where `horizon_steps` is the *earliest*
deadline anyone owes. It returns `feasible`, `shortfall_kwh` and `per_device`. When it
reports infeasible, `prioritise` allocates the budget earliest-deadline-first and names
who is `missed` — a declared triage rule rather than whichever device the loop reached
first.

`stems.fleet.schedule` is the exact version: a linear programme over `K` steps and the
connected devices, with the plant's piecewise-linear power curve as constraints, the
cap as a budget row per step, and a slack variable `shortfall_soc` per device so the
programme is always feasible and reports the shortfall instead of failing. Its dual on
the cap row, `cap_shadow_price`, is the marginal value of one more kW of cap — the
quantity a network operator would price. `fleet_power_bounds` runs the same programme
twice, minimising and then maximising this step's total, which brackets the
neighbourhood's flexibility interval `[u_min, u_max]` in kW.

The cap itself is set with a forecast margin: `cap - forecaster.margin`, where `margin`
is the 95th percentile of the recent one-step base-load forecast error
(`BaseLoadForecaster.margin`, 168-step window). A cap enforced against a perfect
forecast is not a cap.

---

## 5. The instantiations

Four loads, plus the building envelope, which is in the family and gated off on this
testbed (last row). Each row is one constructor call; nothing in the table is a
separate enforcement path.

| load | store | `rate` is set by | `required_soc` | deadline | plant model |
|---|---|---|---|---|---|
| Electric vehicle | traction battery | charger `p_max * eta_c / capacity`, derated | `required_soc_departure` from the observation | departure hour | `EVFleetModel` |
| Hot-water tank | tank thermal store | heater `nominal_power * efficiency / capacity` | forecast draw over the next `horizon` steps | now (every step) | `TankModel` |
| Legionella cycle | a `ShadowTank` over the same store | `P_hp`, less standing loss and `draw_margin_kwh` | `soc_legionella` = `(T_legionella - T_cold)/(T_rated - T_cold)` | end of the `period_hours` window, recurring | `ShadowTank` |
| House battery | electrical storage | `nominal_power * dt / capacity` | — (`active` false) | — | `BatteryModel` |
| *(gated off on CityLearn)* building envelope | zone thermal mass | `temperature_gain_k(phi_max_w) / SPAN_K` | comfort band edge | now, when occupied | `RCThermalModel` |

### A limitation of the deadline, found by instantiating the fourth load

`DeadlineStorageBarrier` starts forcing when `steps_to_deadline` falls to
`steps_needed = ceil(gap / rate)`. That is a **just-in-time schedule with no
recourse**: it commands the fewest steps that would suffice if nothing else happened.
Three of the four loads tolerate it. The hot-water store does not, because — unlike a
vehicle, which is not driven while plugged in — it is *drawn from while it is being
charged*, and the draw is bursty.

Measured by `experiments/legionella_demo.py` over 336 steps × 8 buildings (16
building-windows) on the real hot-water series:

| source size | draw reserve | cycles completed | windows missed |
|---|---|---|---|
| sizing rule | median draw (= 0.0 kWh on this data) | 1 / 16 | 15 |
| sizing rule | 90th percentile of own draw | **10 / 16** | 6 |
| sizing rule | maximum draw | refused: reserve ≥ source output per step | — |
| 2 × sizing rule | median draw | 0 / 16 | 16 |

A **larger source completes fewer cycles**, which is not a bug: it raises `rate`, so
`steps_needed` falls to 1 for six of eight buildings, and the whole cycle must then
succeed inside one hour. The median hourly draw is 0.0 kWh and the per-building
maximum ranges from 1.613 to 6.302 kWh, so a median reserve is no reserve. The lever that works is the reserve, not the
source. Closing the gap properly needs a forecast of the draw during the forced hours;
`draw_margin_kwh` is the declared stand-in and defaults to zero, which guarantees the
deadline only against a zero draw.

The envelope row is in the family and is **off by default**, because CityLearn's
learned temperature model drops a house 4–13 degC in one hour under a −0.25 cooling
action (`docs/REPORT_2026-10.md` §2). A hard band enforced against that model is
satisfied in simulation and says nothing about a building. It turns on when a
calibrated model is identified from real measurements — which is what
`RCThermalModel.identify` is for, and what the sim-to-real seam in §6 is for.

---

## 6. The environment seam

The framework claims to outlive CityLearn, which is only true if nothing above imports
it. The controller and the shield talk to `stems.protocols.Environment`:

```python
reset() -> (obs_list, info)          step(actions) -> (obs_list, rewards, term, trunc, info)
num_buildings  obs_dim  action_dim  obs_names  action_names  index_of(name)  executed_actions
```

and, when they need to project exactly, to two optional extensions —
`PlantProvider` (the `BatteryModel`/`TankModel` for this site and the action columns
they own) and `EVProvider` (the charger fleet). A real-building adapter implements
`Environment` always, `PlantProvider` when the plant is characterised, `EVProvider`
only if there are vehicles. `STEMSEnvironment` implements all three;
`stems.replay.CSVReplayEnvironment` implements the first two from logged data.

Demonstrated, not declared. `experiments/replay_roundtrip.py` exports a 72-step
CityLearn rollout to the logged-data format, loads it back through the replay adapter,
and projects the *same* `CBFShield` and the *same* `FlexibilityPortfolio` over both:

| quantity, over 71 steps × 8 buildings | max absolute difference |
|---|---|
| observation vector, simulator vs replay | 0.000e+00 |
| projected action, with the adapter's assumed flat plant curves | 3.155e-02 |
| projected action, with the simulator's own plant curves | 0.000e+00 |

The interface costs nothing; the *plant characterisation* costs 3.155e-02 of action.
That is the number a deployment needs, because it says how much of the shield's
guarantee rests on fitting the site's devices rather than on plumbing its data. A
logged export does not come with measured efficiency and capacity-power curves, so the
adapter assumes them flat and declares the assumption in `plant_notes`.

What the replay adapter is for, stated plainly: it is **open loop**. Recorded
observations do not respond to the actions a controller takes, so it can check that a
controller runs, that the observation vector is laid out as `selected_obs_names`
expects, and that the shield's projections are feasible against the recorded state —
and it cannot evaluate control. Closed-loop evaluation on real data needs an
identified model (`RCThermalModel.identify`, `BatteryModel`, `TankModel` fitted to the
site), validated against held-out measurements first. That is the next seam, not this
one.

---

## 7. Assumed, measured, unsourced

| statement | status |
|---|---|
| Battery, tank and EV one-step models reproduce the simulator to the errors in §3 | **measured**, `experiments/plant_projection_error.py` |
| CityLearn exposes no building geometry, so the graph is feature-based | **measured** (five ways), `CHANGELOG.md` step 6 |
| CityLearn's thermal model is not physical; hard comfort stays off there | **measured**, `docs/REPORT_2026-10.md` §2 |
| The cap margin covers 95% of recent one-step forecast errors | **measured** on the rollout, by construction of the quantile |
| `rate` for the EV barrier is conservative by `rate_derate = 0.85` | **assumption**, chosen not fitted |
| `soc_cap = 0.95` for the tank, `margin = 0.05` | **assumption**, chosen not fitted |
| Heat-pump source power `P_hp`, `T_normal`, `T_legionella`, `T_cold`, `T_rated`, `T_hp,max` | **unsourced** — required arguments with no default; `LegionellaSpec.unsourced` lists them at runtime |
| `cop_legionella`, the CoP at the disinfection sink | **unsourced**, and the step at the heat-pump ceiling is a modelling choice: a real unit's CoP falls continuously with sink temperature |
| `soc` is linear in store temperature | **assumption**: a fully mixed tank. A real cylinder stratifies, so the energy to disinfect the whole volume is understated whenever a draw has left a cold bottom layer |
| Unmet hot water is zero at the design note's sizing rule on this schema | **measured**, `experiments/legionella_demo.py`; it appears (4.17 kWh) only at a quarter of the rule |
| A just-in-time deadline does not survive a bursty disturbance | **measured**, same script; 1 of 16 windows met with a median reserve, 10 of 16 with a 90th-percentile reserve |
| A weekly Legionella cycle has not previously been posed as a deadline constraint in reinforcement learning | **plausible, not established.** `docs/LITERATURE.md` found no such work; that is an absence of evidence from one search, not a proven absence. [engelbrecht2021optimal] is a confirmed field study of 77 water heaters (median saving 21.9% → 16.2% with Legionella prevention) but its method is A\* search, not RL; the second citation the earlier report leaned on was struck — [reyespremer2025model] does not mention Legionella. See `stems/legionella.py` |
