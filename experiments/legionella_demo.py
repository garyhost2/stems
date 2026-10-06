#!/usr/bin/env python3
"""The disinfection cycle against the simulator's real hot-water draws.

Drives ``stems.legionella.LegionellaStack`` alongside ``STEMSEnvironment`` on the real
hot-water demand series, with the cycle barrier on and off, and reports the two things
the design note says the model exists to produce: **unmet hot water** (the service
failure a power-limited source makes possible and CityLearn's oversized heater cannot)
and the **electricity the weekly cycle costs**.

This is a diagnostic, not an experiment. One schema, one deterministic rule, no policy,
no seeds, no comparison table. Its job is to show the fourth load running against real
draws and to put the sensitivity to the unsourced source power on the record.

Sizing, and why there is no single number here
----------------------------------------------
``P_hp`` has no sourced value in this repository, so the script does not pick one. It
sweeps a **sizing rule** instead -- the design note's "the capacity that covers the
mean daily demand in ~4 h" -- at several multipliers, computing the mean daily demand
from the schema's own series. The rule is a rule of thumb from the design note, not a
product specification; the multipliers show how hard the conclusion depends on it.

The tank capacity ``C`` is read from the schema (``env.dhw_info()["capacity"]``) and is
therefore sourced. The four temperatures are not sourced anywhere, and the values used
here are stated in ``TEMPERATURES`` below as assumptions of this diagnostic.

What it found, so the numbers are not read backwards
----------------------------------------------------
A **larger** heat source completes **fewer** cycles (0 of 16 building-windows at twice
the sizing rule, against 1 of 16 at the rule). That looks like a bug and is not; the
mechanism was traced and is a property of the deadline formulation.

``DeadlineStorageBarrier`` starts forcing when ``steps_to_deadline`` falls to
``steps_needed = ceil(gap / rate)``. That is a *just-in-time* schedule with no
recourse: it commands the minimum number of steps that would suffice if nothing else
happened. A bigger source raises ``rate``, so ``steps_needed`` falls -- to **1** for
six of the eight buildings at twice the sizing rule -- and the whole cycle then has to
succeed inside a single hour. Hot-water draws on this schema are bursty: the **median
hourly draw is 0.0 kWh** while the per-building maximum ranges from 1.613 to 6.302 kWh.
So a median-quantile reserve is no reserve at all, and one burst in the single forced
hour defeats the deadline.

The lever that works is therefore the reserve, not the source. Holding the 90th
percentile of each building's own draw takes completions from 1 of 16 to **10 of 16**
at unchanged source size. Holding the maximum is *refused* by ``LegionellaSpec``,
because for at least one building the peak draw exceeds everything the source can
deliver in a step, which would make the constraint unsatisfiable by construction.

Two consequences worth carrying forward, both negative results about this testbed:

* **Unmet hot water is zero at the design note's sizing rule** and at every multiple
  above it; it appears (4.17 kWh over 336 steps × 8 buildings) only at a quarter of the
  rule. On this schema, with the schema's own tank capacities, the power limit does not
  bind. The service-failure KPI the design note wants exists and measures zero here.
* **A just-in-time deadline barrier does not guarantee a deadline under disturbance.**
  Closing that needs either a forecast of the draw during the forced hours or a
  declared reserve, and this repository has neither sourced. It is a finding about the
  framework, not about a controller, and no controller was run.

    PYTHONPATH=. python -B experiments/legionella_demo.py

Writes ``experiments/diagnostics/legionella/demo.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.environment import STEMSEnvironment
from stems.legionella import HOURS_PER_WEEK, LegionellaSpec, LegionellaStack, ShadowTank
from stems.observations import obs_index

OUT = REPO / "experiments" / "diagnostics" / "legionella" / "demo.json"
SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
STEPS = 336                       # two disinfection windows
IDX_DHW_DEMAND = obs_index("dhw_demand")

#: Assumptions of this diagnostic, not sourced values. A European domestic cylinder is
#: commonly run near 50 degC with a ~60 degC disinfection set point and a cold inlet
#: around 10 degC, and the heat pump alone is assumed to reach 55 degC. Every one of
#: these is exactly the kind of number the design note lists as an open question; they
#: are here so the script runs, they are recorded as unsourced in the output, and they
#: must not be carried into a paper without a source.
TEMPERATURES = {"t_cold_c": 10.0, "t_rated_c": 60.0, "t_normal_c": 50.0,
                "t_legionella_c": 60.0, "t_heat_pump_max_c": 55.0,
                "cop_legionella": 2.5, "element_efficiency": 1.0}

#: Multiples of the design note's sizing rule, `mean daily demand / 4 h`.
SIZING_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)

#: Quantiles of each building's own draw used as `draw_margin_kwh`, the reserve the
#: barrier holds against hot-water being drawn *while the disinfection cycle runs*.
#: Two one-dimensional sweeps are reported, not a grid: sizing at the **median**
#: reserve (quantile 0.5), and reserve quantile at the nominal sizing. The median, not
#: the mean: an earlier draft of this script reserved `demand.mean(axis=0)` and the
#: label was not updated when it became a quantile. The distinction is load-bearing
#: here rather than pedantic -- hot-water draws are bursty, so on this schema the
#: median hourly draw is 0.0 kWh while the mean is not, and "a median reserve is no
#: reserve" is the finding below.
RESERVE_QUANTILES = (0.5, 0.9, 1.0)


def demand_series(env: STEMSEnvironment, steps: int) -> np.ndarray:
    """The schema's own hot-water draw, kWh per step, shape (steps, buildings)."""
    obs, _ = env.reset()
    rows = [np.array([float(o[IDX_DHW_DEMAND]) for o in obs])]
    zero = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    for _ in range(steps - 1):
        obs, _r, term, trunc, _i = env.step(zero)
        rows.append(np.array([float(o[IDX_DHW_DEMAND]) for o in obs]))
        if term or trunc:
            break
    return np.stack(rows)


def run(spec: LegionellaSpec, demand: np.ndarray, cycle_on: bool) -> Dict[str, Any]:
    """Serve the draws with a thermostat; add the cycle barrier when ``cycle_on``.

    The thermostat is the no-policy reference: command exactly the heat that brings the
    store back to its normal set point, no more. It is deliberately not smart -- the
    point is to isolate what the *deadline* does, not to propose a controller -- but it
    must not be bang-bang. A full-power command whenever the store is below the set
    point overshoots to the ceiling and parks there, which in this model means holding
    the tank permanently at the disinfection temperature: the cycle is then satisfied
    by accident at every step and the constraint measures nothing. (That is also the
    real practice [reyespremer2025model] describes and beats: constant 60 degC storage.)
    """
    stack = LegionellaStack.build(spec, action_index=0)
    B = spec.B
    per_step = spec.heat_source_kw * spec.dt_hours
    for t in range(demand.shape[0]):
        soc = stack.tank.soc
        need_kwh = np.maximum(spec.soc_normal - soc, 0.0) * spec.capacity_kwh + demand[t]
        thermostat = np.clip(need_kwh / np.maximum(per_step, 1e-9), 0.0, 1.0)
        actions = np.zeros((B, 1), dtype=np.float32)
        actions[:, 0] = thermostat.astype(np.float32)
        if cycle_on:
            actions = stack.load.project(actions, [np.zeros(40, dtype=np.float32)] * B)
        stack.observe(actions[:, 0], demand[t])
    report = stack.report()
    report["cycle_enforced"] = cycle_on
    return report


def main() -> None:
    env = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, STEPS)]})
    capacity = np.asarray(env.dhw_info()["capacity"], dtype=np.float64)
    demand = demand_series(env, STEPS)
    mean_daily = demand.mean(axis=0) * 24.0
    base_p_hp = mean_daily / 4.0          # the design note's sizing rule

    loss = np.asarray(env.dhw_info()["loss_coefficient"], dtype=np.float64)

    def make(multiplier: float, quantile: float) -> LegionellaSpec:
        spec = LegionellaSpec(
            capacity_kwh=capacity, heat_source_kw=base_p_hp * multiplier,
            loss_coefficient=loss,
            draw_margin_kwh=np.quantile(demand, quantile, axis=0),
            period_hours=HOURS_PER_WEEK, **TEMPERATURES)
        spec.provenance["capacity_kwh"] = f"schema {SCHEMA}, dhw_storage.capacity"
        spec.provenance["loss_coefficient"] = f"schema {SCHEMA}, dhw_storage"
        return spec

    rows: List[Dict[str, Any]] = []
    for m in SIZING_MULTIPLIERS:                       # sweep A: source size
        spec = make(m, 0.5)
        for cycle_on in (False, True):
            r = run(spec, demand, cycle_on)
            r.update(sweep="sizing", sizing_multiplier=m, reserve_quantile=0.5,
                     heat_source_kw=[round(float(p), 4) for p in spec.heat_source_kw],
                     draw_margin_kwh=[round(float(d), 4) for d in spec.draw_margin_kwh])
            rows.append(r)
    for q in RESERVE_QUANTILES:                        # sweep B: the draw reserve
        try:
            spec = make(1.0, q)
        except ValueError as exc:
            # Reserving a draw the source cannot outrun makes the cycle unreachable by
            # construction, and `LegionellaSpec` refuses rather than running a
            # constraint that can never be met. That refusal is the result for this
            # row, not an error to work around.
            rows.append({"sweep": "reserve", "sizing_multiplier": 1.0,
                         "reserve_quantile": q, "cycle_enforced": True,
                         "refused": str(exc),
                         "service": {"unmet_kwh": float("nan"),
                                     "electricity_kwh": float("nan")},
                         "cycle": {"cycles_completed": [], "missed_windows": []}})
            continue
        r = run(spec, demand, True)
        r.update(sweep="reserve", sizing_multiplier=1.0, reserve_quantile=q,
                 heat_source_kw=[round(float(p), 4) for p in spec.heat_source_kw],
                 draw_margin_kwh=[round(float(d), 4) for d in spec.draw_margin_kwh])
        rows.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"schema": SCHEMA, "steps": int(demand.shape[0]),
         "buildings": int(demand.shape[1]),
         "mean_daily_demand_kwh": [round(float(d), 4) for d in mean_daily],
         "sizing_rule": "mean daily demand / 4 h (docs/dhw_legionella_design.md)",
         "temperatures_are_assumptions_of_this_diagnostic": TEMPERATURES,
         "runs": rows}, indent=2) + "\n")

    print(f"{demand.shape[0]} steps, {demand.shape[1]} buildings, "
          f"total draw {demand.sum():.1f} kWh, "
          f"{demand.shape[1] * demand.shape[0] / HOURS_PER_WEEK:.0f} building-windows")
    print(f"mean daily demand per building, kWh: "
          f"{[round(float(d), 2) for d in mean_daily]}")
    head = (f"{'sweep':>9}{'P_hp x':>8}{'reserve q':>11}{'cycle':>7}{'unmet kWh':>11}"
            f"{'done':>6}{'missed':>8}{'elec kWh':>10}")
    for label in ("sizing", "reserve"):
        print("\n" + head)
        print("-" * len(head))
        for r in (x for x in rows if x["sweep"] == label):
            s, c = r["service"], r["cycle"]
            lead = (f"{r['sweep']:>9}{r['sizing_multiplier']:>8}"
                    f"{r['reserve_quantile']:>11}{str(r['cycle_enforced']):>7}")
            if "refused" in r:
                print(lead + "   REFUSED: reserve >= source output per step")
                continue
            print(lead + f"{s['unmet_kwh']:>11.3f}{sum(c['cycles_completed']):>6}"
                  f"{sum(c['missed_windows']):>8}{s['electricity_kwh']:>10.2f}")
    print(f"\nunsourced parameters carried by every row: "
          f"{make(1.0, 0.5).unsourced}")
    print(f"written to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
