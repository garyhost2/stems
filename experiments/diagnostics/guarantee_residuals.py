"""Measure the residual on every constraint the framework claims to enforce.

This script produces the numbers quoted in ``docs/FRAMEWORK_GUARANTEES.md``. It calls
only the public projection interface (``CBFShield.project``, ``FleetShield.project``,
``schedule``/``schedule_executable``/``apply_dead_band``), so it survives a refactor of
the barrier internals.

Notation, identical to ``docs/FRAMEWORK_GUARANTEES.md`` and to the code:

  b              building index, b = 1..B
  x_b            state of charge of a store in building b, dimensionless in [0, 1]
  x_min, x_max   the state-of-charge band the shield must keep, dimensionless
                 (``CBFConfig.SOC_min``, ``CBFConfig.SOC_max``)
  a_b            commanded action of building b, dimensionless in [-1, 1]
  r_b            one-step state-of-charge rate of building b, = p_nom_b*dt/C_b, 1/step
  C_b            usable store capacity of building b, kWh
  p_nom_b        store nominal power of building b, kW
  P_cap          district import cap, kW
  e_b(t)         realised base (non-vehicle) import of building b at step t, kW
  ehat_b(t)      the forecaster's prediction of e_b(t), kW
  m(t)           forecast safety margin at step t, kW; the 95th percentile of the
                 district one-step forecast error over the last 168 steps
  I(t)           realised district import, = sum_b max(e_b(t) + d_b(t), 0), kW
  d_b(t)         realised vehicle charger draw of building b, kW
  R(t)           cap residual, = max(I(t) - P_cap, 0), kW
  s_b            vehicle state of charge of building b at a charger, dimensionless
  s*_b           required state of charge at departure, dimensionless
  n_b            steps remaining until departure, steps
  p_min_b        charger minimum charging power, kW (zero or [p_min, p_max], no values
                 in between -- ``EVFleetModel.applied_kw`` clips into the dead band)

Energy is kWh, power kW, time steps of dt = 1 h throughout. "Residual" always means the
amount by which the realised quantity violated the constraint, in the constraint's own
units; a guarantee with a measured residual of exactly zero over an exhaustive sweep is
reported as hard *on that sweep*, never as hard in general.

Usage (from the repository root):
    python experiments/diagnostics/guarantee_residuals.py             # every section
    python experiments/diagnostics/guarantee_residuals.py soc cap     # named sections
Sections: soc, cap, deadline, report, tank.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from stems.battery import BatteryModel, TankModel
from stems.cbf import CBFShield
from stems.config import CBFConfig
from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.fleet import (RULES, BaseLoadForecaster, EVFleetModel, FleetShield, FleetState,
                         HouseStorage, allocate, apply_dead_band, schedule,
                         schedule_executable)
from stems.observations import obs_index

CFG = CBFConfig()
IDX_SOC = obs_index("electrical_storage_soc")
IDX_LOAD = obs_index("non_shiftable_load")
IDX_NET = obs_index("net_electricity_consumption")
OBS_LEN = 30
FLAT = np.array([[0.0, 1.0], [1.0, 1.0]])

# Synthetic-vehicle observation layout used by every fleet section below. The four
# fields are the ones ``FleetShield`` reads; their positions are arbitrary because the
# shield is told where they are.
LAYOUT = {"connected_state": 0, "soc": 1, "required_soc_departure": 2, "departure_time": 3}


def ev_obs(connected: float, soc: float, target: float, countdown: float,
           load_kw: float = 0.0) -> np.ndarray:
    o = np.zeros(OBS_LEN)
    o[0], o[1], o[2], o[3] = connected, soc, target, countdown
    o[IDX_LOAD] = load_kw
    return o


def ev_fleet(B: int, p_max: float = 7.0, p_min: float = 1.4, capacity: float = 60.0,
             eta: float = 0.95) -> EVFleetModel:
    battery = BatteryModel([capacity] * B, [p_max] * B, [0.0] * B, [FLAT] * B, [FLAT] * B)
    return EVFleetModel([True] * B, [p_max] * B, [p_min] * B, [eta] * B, battery)


class ZeroMargin(BaseLoadForecaster):
    """The same forecaster with the safety margin disabled, to isolate the margin."""

    @property
    def margin(self) -> float:
        return 0.0


# ----------------------------------------------------------------------------- soc ---

def real_battery() -> BatteryModel:
    """The simulator's own eight batteries, resolved through CityLearn's autosizing."""
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(seed=0, heat_pump=True, env_kwargs={"episode_time_steps": [(0, 23)]})
    return env.battery_model()


def soc_sweep(shield: CBFShield, plant: BatteryModel, socs: np.ndarray,
              actions: np.ndarray) -> dict:
    """Project every (x_b, a_b) pair, step the plant, and measure the band residual."""
    B = plant.B
    below = above = 0.0
    breaches = cases = 0
    for x in socs:
        states = [np.zeros(OBS_LEN) for _ in range(B)]
        for s in states:
            s[IDX_SOC] = x
        for a in actions:
            cmd = np.zeros((B, 3), dtype=np.float32)
            cmd[:, 1] = a
            safe = shield.project(cmd, states)
            nxt = plant.next_soc(np.full(B, x), safe[:, 1].astype(np.float64))
            below = max(below, float(np.max(CFG.SOC_min - nxt)))
            above = max(above, float(np.max(nxt - CFG.SOC_max)))
            breaches += int(np.sum((nxt < CFG.SOC_min - 1e-9) | (nxt > CFG.SOC_max + 1e-9)))
            cases += B
    return {"cases": cases, "breaches": breaches,
            "rate": breaches / max(cases, 1), "below": below, "above": above}


def section_soc(quick: bool = False) -> None:
    plant = real_battery()
    B = plant.B
    rate = plant.nominal_power * plant.dt / plant.capacity
    step = 4 if quick else 1
    socs = np.linspace(0.0, 1.0, 101)[::step]
    acts = np.linspace(-1.0, 1.0, 81)[::step]
    in_band = socs[(socs >= CFG.SOC_min - 1e-12) & (socs <= CFG.SOC_max + 1e-12)]

    print(f"\n[soc] battery state-of-charge band [{CFG.SOC_min}, {CFG.SOC_max}], "
          f"{B} CityLearn buildings")
    print("      true one-step rate r_b = p_nom_b*dt/C_b  : "
          + np.array2string(np.round(rate, 4), separator=", "))
    print("      C_b [kWh]                                : "
          + np.array2string(np.round(plant.capacity, 1), separator=", "))

    variants = [
        ("exact inverse of the plant", CBFShield(num_buildings=B, battery_model=plant, elec_idx=1)),
        ("uniform r_b = 0.1 surrogate", CBFShield(num_buildings=B, soc_rate=np.full(B, 0.1), elec_idx=1)),
        ("surrogate at half the true r_b", CBFShield(num_buildings=B, soc_rate=rate * 0.5, elec_idx=1)),
    ]
    head = f"      {'shield plant model':<32}{'cases':>8}{'breaches':>10}{'rate':>9}{'max below':>11}{'max above':>11}"
    print("\n      starts inside the band:")
    print(head)
    for label, shield in variants:
        r = soc_sweep(shield, plant, in_band, acts)
        print(f"      {label:<32}{r['cases']:>8}{r['breaches']:>10}{r['rate']:>9.4f}"
              f"{r['below']:>11.5f}{r['above']:>11.5f}")
    print("\n      starts anywhere in [0, 1] (includes states already outside the band):")
    print(head)
    for label, shield in variants:
        r = soc_sweep(shield, plant, socs, acts)
        print(f"      {label:<32}{r['cases']:>8}{r['breaches']:>10}{r['rate']:>9.4f}"
              f"{r['below']:>11.5f}{r['above']:>11.5f}")

    exact = variants[0][1]
    print("\n      recovery from an out-of-band start, exact inverse, worst building:")
    for x0, cmd in ((0.0, 1.0), (1.0, -1.0)):
        x = np.full(B, x0)
        traj = []
        for _ in range(4):
            states = [np.zeros(OBS_LEN) for _ in range(B)]
            for j, s in enumerate(states):
                s[IDX_SOC] = x[j]
            A = np.zeros((B, 3), dtype=np.float32)
            A[:, 1] = cmd
            x = plant.next_soc(x, exact.project(A, states)[:, 1].astype(np.float64))
            traj.append(int(np.sum((x < CFG.SOC_min - 1e-9) | (x > CFG.SOC_max + 1e-9))))
        print(f"      x0 = {x0:.2f}: buildings still outside the band after 1..4 steps = {traj}")


# ----------------------------------------------------------------------------- cap ---

def cap_rollout(T: int = 504, B: int = 4, cap_kw: float = 30.0, seed: int = 0,
                rule: str = "lp", margin: bool = True, target: float = 0.7,
                level_shock_kw: float = 0.0, spread_shock: float = 1.0,
                shock_at: int = 336) -> dict:
    """Drive ``FleetShield`` over a synthetic base load and measure the cap residual.

    The realised base load of building b is e_b(t) = u_b(t) + h_b(t), where u_b is the
    *observed* part (``non_shiftable_load`` minus ``solar_generation``, which the
    forecaster reads directly from the observation) and h_b is an unobserved
    autoregressive component standing for the thermal and hot-water draw. h_b is the
    only source of forecast error, which is the structure of the error in CityLearn:
    the shield observes this step's exogenous load exactly and must predict the
    endogenous part. ``level_shock_kw`` adds a step change to e_b from ``shock_at``;
    ``spread_shock`` multiplies the innovation scale of h_b from ``shock_at`` while
    holding its mean, so the margin calibrated on the first ``shock_at`` steps is too
    small for what follows.
    """
    rng = np.random.default_rng(seed)
    model = ev_fleet(B)
    forecaster = (BaseLoadForecaster if margin else ZeroMargin)(B)
    shield = FleetShield(model, LAYOUT, 0, cap_kw, rule, forecaster)
    soc = np.full(B, 0.4)
    connected = np.zeros(B, dtype=bool)
    countdown = np.zeros(B, dtype=int)
    eps = np.zeros(B)
    hidden = np.zeros(B)
    mean_innovation = 2.0 * 0.35          # gamma(shape=2, scale=0.35) mean, kW
    residual, imports, margins, flags, departures = [], [], [], [], []
    for t in range(T):
        hour = t % 24
        shocked = t >= shock_at
        eps = 0.6 * eps + rng.normal(0.0, 0.2, B)
        observed = np.maximum(0.7 + 0.5 * np.sin(2 * np.pi * (hour - 7) / 24) ** 2 + eps, 0.0)
        scale = spread_shock if shocked else 1.0
        hidden = np.maximum(0.7 * hidden + scale * rng.gamma(2.0, 0.35, B)
                            - (scale - 1.0) * mean_innovation, 0.0)
        base = observed + hidden + (level_shock_kw if shocked else 0.0)
        if hour == 18:
            connected[:] = True
            countdown[:] = 13
            soc[:] = rng.uniform(0.3, 0.45, B)
        obs = [ev_obs(float(connected[i]), soc[i], target, float(max(countdown[i], 0)),
                      observed[i]) for i in range(B)]
        action = shield.project(np.zeros((B, 1), dtype=np.float32), obs)
        draw = model.draw_kw(soc, action[:, 0].astype(np.float64))
        soc = np.where(connected, model.next_soc(soc, action[:, 0].astype(np.float64)), soc)
        imp = float(np.maximum(base + draw, 0.0).sum())
        imports.append(imp)
        residual.append(max(imp - cap_kw, 0.0))
        margins.append(float(shield.last.get("margin_kw", 0.0)))
        flags.append(shield.last.get("feasible"))
        next_obs = []
        for i in range(B):
            o = np.zeros(OBS_LEN)
            o[IDX_NET] = base[i] + draw[i]
            next_obs.append(o)
        shield.observe(next_obs, draw)
        if hour == 7 and connected.any():
            departures.append(float(np.maximum(target - soc, 0.0).sum()))
            connected[:] = False
        countdown = np.maximum(countdown - 1, 0)
    burn = T // 4                      # the margin needs 168 steps of history to exist
    R = np.array(residual[burn:])
    return {"residual_rate": float((R > 1e-6).mean()), "residual_kwh": float(R.sum()),
            "residual_max_kw": float(R.max()), "peak_kw": float(max(imports[burn:])),
            "margin_kw": float(np.mean(margins[burn:])),
            "infeasible_rate": float(np.mean([f is False for f in flags[burn:]])),
            "departure_shortfall_soc": float(np.sum(departures))}


def _cap_row(label: str, seeds: int = 5, **kw) -> None:
    runs = [cap_rollout(seed=s, **kw) for s in range(seeds)]
    get = lambda k: np.array([r[k] for r in runs])
    print(f"      {label:<42}"
          f"{get('residual_rate').mean():>10.4f}"
          f"{get('residual_kwh').mean():>11.2f}"
          f"{get('residual_max_kw').mean():>11.3f}"
          f"{get('residual_max_kw').max():>11.3f}"
          f"{get('margin_kw').mean():>10.3f}")


def section_cap(quick: bool = False) -> None:
    seeds = 2 if quick else 5
    cap = 30.0
    print(f"\n[cap] district import cap P_cap = {cap} kW, 4 houses, 378 scored steps, "
          f"{seeds} seeds, rule='lp'")
    print(f"      {'configuration':<42}{'rate':>10}{'sum R [kWh]':>11}{'mean max':>11}"
          f"{'worst max':>11}{'m [kW]':>10}")
    _cap_row("margin on", seeds, cap_kw=cap, margin=True)
    _cap_row("margin off", seeds, cap_kw=cap, margin=False)
    for kw_house in (1.0, 2.0, 3.0, 6.0):
        _cap_row(f"margin on, step +{kw_house:.0f} kW/house at t=336", seeds,
                 cap_kw=cap, margin=True, level_shock_kw=kw_house)
    for spread in (2.0, 4.0, 8.0):
        _cap_row(f"margin on, innovation spread x{spread:.0f} at t=336", seeds,
                 cap_kw=cap, margin=True, spread_shock=spread)
    print(f"\n      saturated regime, P_cap = 16 kW (the deadline set is infeasible "
          f"every night):")
    _cap_row("margin on", seeds, cap_kw=16.0, margin=True)
    _cap_row("margin off", seeds, cap_kw=16.0, margin=False)

    print("\n      cap below the inflexible house load alone "
          "(8 kW load, 5 kW battery charge requested, no vehicle):")
    print(f"      {'P_cap [kW]':>11}{'predicted import':>18}{'residual [kW]':>15}"
          f"{'shed [kW]':>11}{'feasible':>10}{'binding':>9}")
    house_model = BatteryModel([20.0], [5.0], [0.0], [FLAT], [FLAT])
    for cap_kw in (10.0, 6.0, 3.0, 1.0):
        house = HouseStorage(house_model, battery_action=1, battery_soc=19,
                             soc_lo=0.1, soc_hi=0.9, tank_action=2)
        shield = FleetShield(ev_fleet(1), LAYOUT, 0, cap_kw, "lp",
                             BaseLoadForecaster(1), house=house)
        o = ev_obs(0, 0, 0, 0, load_kw=8.0)
        o[19] = 0.5
        shield.project(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), [o])
        pred = shield.last["predicted_import_kw"]
        print(f"      {cap_kw:>11.1f}{pred:>18.3f}{max(pred - cap_kw, 0.0):>15.3f}"
              f"{shield.last.get('storage_shed_kw', 0.0):>11.3f}"
              f"{str(shield.last.get('feasible', 'absent')):>10}"
              f"{str(shield.last.get('binding')):>9}")


# ------------------------------------------------------------------------ deadline ---

def section_deadline(quick: bool = False) -> None:
    print("\n[deadline] vehicle departure state of charge")
    model = ev_fleet(3, p_max=10.0, p_min=1.0, capacity=50.0, eta=1.0)

    print("\n      infeasible joint set: 3 vehicles, each individually reachable, "
          "one shared cap")
    print(f"      {'P_cap [kW]':>11}{'feasible':>10}{'shortfall [kWh]':>17}"
          f"{'worst share':>13}{'now sum [kW]':>14}")
    for cap_kw in (30.0, 20.0, 17.5, 15.0, 10.0, 5.0):
        state = FleetState([True] * 3, [0.4] * 3, [0.8] * 3, [2, 3, 4],
                           np.zeros((8, 3)), cap_kw)
        sol = schedule(model, state, np.array([10.0, 10.0, 10.0]))
        print(f"      {cap_kw:>11.1f}{str(sol['feasible']):>10}"
              f"{sol['total_shortfall_kwh']:>17.3f}{sol['worst_shortfall_share']:>13.4f}"
              f"{sol['now_kw'].sum():>14.3f}")

    print("\n      charger minimum power above what the cap leaves "
          "(1 vehicle, P_cap = 2 kW, owes 0.5 SOC in 3 steps)")
    print(f"      {'p_min [kW]':>11}{'drawn [kW]':>12}{'feasible':>10}"
          f"{'shortfall [kWh]':>17}{'predicted [kW]':>16}")
    for p_min in (1.0, 2.0, 4.0, 8.0, 10.0):
        m = ev_fleet(1, p_max=10.0, p_min=p_min, capacity=50.0, eta=1.0)
        shield = FleetShield(m, LAYOUT, 0, 2.0, "lp", BaseLoadForecaster(1))
        a = shield.project(np.ones((1, 1), dtype=np.float32), [ev_obs(1, 0.4, 0.9, 3)])
        drawn = float(m.draw_kw(np.array([0.4]), a[:, 0].astype(np.float64))[0])
        print(f"      {p_min:>11.1f}{drawn:>12.4f}{str(shield.last.get('feasible')):>10}"
              f"{shield.last.get('shortfall_kwh', 0.0):>17.3f}"
              f"{shield.last['predicted_import_kw']:>16.3f}")

    print("\n      vehicle departs the step after it arrives (countdown = 0), P_cap = 100 kW")
    print(f"      {'s':>6}{'s*':>6}{'action':>9}{'next s':>9}{'residual soc':>14}"
          f"{'feasible':>10}{'shortfall [kWh]':>17}")
    m = ev_fleet(1, p_max=10.0, p_min=1.0, capacity=50.0, eta=1.0)
    for s0, target in ((0.20, 0.90), (0.70, 0.90), (0.89, 0.90), (0.95, 0.90)):
        shield = FleetShield(m, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1))
        a = shield.project(np.zeros((1, 1), dtype=np.float32), [ev_obs(1, s0, target, 0)])
        nxt = float(m.next_soc(np.array([s0]), a[:, 0].astype(np.float64))[0])
        print(f"      {s0:>6.2f}{target:>6.2f}{float(a[0, 0]):>9.4f}{nxt:>9.4f}"
              f"{max(target - nxt, 0.0):>14.4f}{str(shield.last.get('feasible')):>10}"
              f"{shield.last.get('shortfall_kwh', 0.0):>17.3f}")

    print("\n      reserve_hours inflates the target by the idle self-discharge and "
          "clips it at 1.0")
    print("      (battery self-discharge 5 %/step, s* = 0.90, 23 steps to departure)")
    print(f"      {'reserve':>8}{'slots':>7}{'target set':>12}{'target needed':>15}"
          f"{'clipped':>9}{'departure s':>13}{'residual soc':>14}")
    lossy = EVFleetModel([True], [10.0], [1.0], [1.0],
                         BatteryModel([50.0], [10.0], [0.05], [FLAT], [FLAT]))
    keep = 1.0 - 0.05
    for reserve in (0, 2, 4, 6, 12):
        shield = FleetShield(lossy, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1),
                             reserve_hours=reserve)
        state = shield.state([ev_obs(1, 0.30, 0.90, 23)])
        slots = int(state.slots[0])
        idle = max(24 - slots, 0)
        needed = 0.90 / keep ** idle
        arrives = float(state.target[0]) * keep ** idle
        print(f"      {reserve:>8}{slots:>7}{float(state.target[0]):>12.4f}{needed:>15.4f}"
              f"{str(needed > 1.0):>9}{arrives:>13.4f}{max(0.90 - arrives, 0.0):>14.4f}")


# -------------------------------------------------------------------------- report ---

def section_report(quick: bool = False) -> None:
    print("\n[report] what each allocation rule tells the caller when the deadline set "
          "cannot be met")
    print("      2 vehicles, each owing 0.5 SOC of a 50 kWh battery in 1 step, "
          "P_cap = 5 kW -> 50 kWh owed, 5 kWh available")
    print(f"      {'rule':<14}{'feasible':>10}{'shortfall [kWh]':>17}{'binding':>9}"
          f"{'predicted [kW]':>16}")
    model = ev_fleet(2, p_max=10.0, p_min=1.0, capacity=50.0, eta=1.0)
    obs = [ev_obs(1, 0.4, 0.9, 1), ev_obs(1, 0.4, 0.9, 1)]
    for rule in RULES:
        shield = FleetShield(model, LAYOUT, 0, 5.0, rule, BaseLoadForecaster(2))
        shield.project(np.zeros((2, 1), dtype=np.float32), obs)
        print(f"      {rule:<14}{str(shield.last.get('feasible', 'ABSENT')):>10}"
              f"{str(shield.last.get('shortfall_kwh', 'ABSENT')):>17}"
              f"{str(shield.last.get('binding')):>9}"
              f"{shield.last['predicted_import_kw']:>16.3f}")

    print("\n      charger dead band rounds an urgent vehicle UP, past the cap")
    print("      vehicle 1 urgent and owing, vehicle 2 relaxed, p_min = 4 kW, P_cap = 2 kW")
    print(f"      {'rule':<14}{'predicted [kW]':>16}{'residual [kW]':>15}{'feasible':>10}"
          f"{'binding':>9}")
    model = ev_fleet(2, p_max=10.0, p_min=4.0, capacity=50.0, eta=1.0)
    obs = [ev_obs(1, 0.4, 0.9, 0), ev_obs(1, 0.4, 0.5, 9)]
    for rule in RULES:
        shield = FleetShield(model, LAYOUT, 0, 2.0, rule, BaseLoadForecaster(2))
        shield.project(np.zeros((2, 1), dtype=np.float32), obs)
        pred = shield.last["predicted_import_kw"]
        print(f"      {rule:<14}{pred:>16.3f}{max(pred - 2.0, 0.0):>15.3f}"
              f"{str(shield.last.get('feasible', 'ABSENT')):>10}"
              f"{str(shield.last.get('binding')):>9}")

    print("\n      the same step isolated on the public functions: allocate respects the "
          "cap, apply_dead_band undoes it")
    state = FleetState([True, True], [0.4, 0.4], [0.9, 0.5], [1, 9], np.zeros((9, 2)), 2.0)
    granted = allocate(model, state, np.array([2.0, 0.0]), "proportional")
    banded = apply_dead_band(model, state, granted)
    print(f"      allocate        -> {granted.tolist()}  sum {granted.sum():.3f} kW "
          f"<= P_cap {state.cap:.1f} kW")
    print(f"      apply_dead_band -> {banded.tolist()}  sum {banded.sum():.3f} kW "
          f"vs P_cap {state.cap:.1f} kW, residual {max(banded.sum() - state.cap, 0.0):.3f} kW")

    print("\n      schedule_executable: does the dead-band second pass still report the "
          "shortfall it creates?")
    print(f"      {'p_min [kW]':>11}{'pass 1 now [kW]':>17}{'pass 1 short':>14}"
          f"{'final now [kW]':>16}{'final short':>13}{'feasible':>10}")
    for p_min in (1.0, 3.0, 6.0, 9.0):
        m = ev_fleet(1, p_max=10.0, p_min=p_min, capacity=50.0, eta=1.0)
        st = FleetState([True], [0.40], [0.90], [1], np.zeros((1, 1)), 2.0)
        first = schedule(m, st, np.array([10.0]))
        final = schedule_executable(m, st, np.array([10.0]))
        print(f"      {p_min:>11.1f}{first['now_kw'][0]:>17.4f}"
              f"{first['total_shortfall_kwh']:>14.3f}{final['now_kw'][0]:>16.4f}"
              f"{final['total_shortfall_kwh']:>13.3f}{str(final['feasible']):>10}")


# ---------------------------------------------------------------------------- tank ---

def section_tank(quick: bool = False) -> None:
    print("\n[tank] hot-water tank model deliberately mis-calibrated against the plant")
    print("      the shield predicts the house draw with its own TankModel; the plant "
          "uses the true one")
    print("      8 kW inflexible load, tank charge action 1.0, P_cap = 12 kW, "
          "true heater efficiency 0.85")
    print(f"      {'shield eta_heater':>19}{'predicted [kW]':>16}{'realised [kW]':>15}"
          f"{'error [kW]':>12}{'residual [kW]':>15}")
    cap_kw, true_eta, demand = 12.0, 0.85, np.array([0.0])
    battery = BatteryModel([20.0], [5.0], [0.0], [FLAT], [FLAT])
    plant_tank = TankModel([10.0], [8.0], [true_eta], [1.0], [0.0])
    for eta in (0.60, 0.85, 1.00):
        shield_tank = TankModel([10.0], [8.0], [eta], [1.0], [0.0])
        house = HouseStorage(battery, battery_action=1, battery_soc=19, soc_lo=0.1,
                             soc_hi=0.9, tank=shield_tank, tank_action=2)
        shield = FleetShield(ev_fleet(1), LAYOUT, 0, cap_kw, "lp",
                             BaseLoadForecaster(1), house=house)
        o = ev_obs(0, 0, 0, 0, load_kw=8.0)
        o[19] = 0.5
        a = shield.project(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), [o])
        soc_tank = np.array([float(o[obs_index("dhw_storage_soc")])])
        realised = 8.0 + float(plant_tank.drawn_kwh(soc_tank, a[:, 2].astype(np.float64),
                                                    demand)[0]) / plant_tank.dt
        pred = shield.last["predicted_import_kw"]
        print(f"      {eta:>19.2f}{pred:>16.3f}{realised:>15.3f}{realised - pred:>12.3f}"
              f"{max(realised - cap_kw, 0.0):>15.3f}")


# ----------------------------------------------------------------------------- cbf ---

def _dhw_like_barrier(B: int, action_index: int = 0, p_charge_kw: float = 6.0,
                      rate: float = 0.12) -> DeadlineStorageBarrier:
    """A deadline barrier on a synthetic store, with the store state in spare columns.

    Columns 24..27 of the observation hold (state of charge, required state of charge,
    steps to deadline, active). Nothing in ``stems`` reads those positions, so this is a
    free scratch area for a barrier whose only job here is to demand a charge.
    """
    barrier = DeadlineStorageBarrier(
        rate=np.full(B, rate), action_bound=np.ones(B), action_index=action_index,
        capacity=np.full(B, 50.0), efficiency=np.ones(B),
        requirement_fn=lambda ol: DeadlineRequirement(
            soc=np.array([float(o[25]) for o in ol]),
            steps_to_deadline=np.array([float(o[26]) for o in ol]),
            active=np.array([o[27] > 0.5 for o in ol])),
        soc_fn=lambda ol: np.array([float(o[24]) for o in ol]), name="deadline_store")
    barrier._p_charge = np.full(B, p_charge_kw)
    return barrier


def section_cbf(quick: bool = False) -> None:
    from stems.thermal import CoPModel

    B, p_charge, g_cap = 3, 6.0, 4.0
    print("\n[cbf] CBFShield: a deadline barrier and the district cap disagree")
    print(f"      {B} stores each owing 0.7 SOC of 50 kWh in 1 step at {p_charge} kW; "
          f"P_grid_max = {g_cap} kW")
    print(f"      {'coordination':<16}{'barrier demands a':>20}{'shield returns a':>42}"
          f"{'draw [kW]':>11}{'grid cap':>10}")
    states = []
    for _ in range(B):
        o = np.zeros(OBS_LEN)
        o[IDX_SOC], o[IDX_NET] = 0.5, 0.0
        o[24], o[25], o[26], o[27] = 0.2, 0.9, 1.0, 1.0
        states.append(o)
    for coordination in ("independent", "proportional", "edf"):
        shield = CBFShield(num_buildings=B, soc_rate=np.full(B, 0.12), elec_idx=1,
                           deadline_barriers=[_dhw_like_barrier(B)],
                           coordination=coordination,
                           config=CBFConfig(P_grid_max=g_cap), enforce_soc=False)
        demanded = _dhw_like_barrier(B).project(
            np.zeros((B, 3), dtype=np.float32), states)[:, 0]
        got = shield.project(np.zeros((B, 3), dtype=np.float32), states)[:, 0]
        fmt = lambda v: "[" + ", ".join(f"{float(x):.3f}" for x in v) + "]"
        print(f"      {coordination:<16}{fmt(demanded):>20}{fmt(got):>42}"
              f"{float((got * p_charge).sum()):>11.2f}{shield.grid_cap():>10.2f}")
    report = CBFShield(num_buildings=B, soc_rate=np.full(B, 0.12), elec_idx=1,
                       deadline_barriers=[_dhw_like_barrier(B)],
                       config=CBFConfig(P_grid_max=g_cap),
                       enforce_soc=False).feasibility_report(states)
    print(f"      feasibility_report(): feasible={report['feasible']}, "
          f"required={report['energy_required_kwh']:.1f} kWh, "
          f"available={report['energy_available_kwh']:.1f} kWh, "
          f"shortfall={report['shortfall_kwh']:.1f} kWh")
    print("      -- and no live code path calls it; see docs/FRAMEWORK_GUARANTEES.md S5.")

    print("\n      heat-pump guard with the base load alone already over the cap "
          "(P_grid_max = 100 kW, derate 5 %)")
    print(f"      {'net/house [kW]':>16}{'base alone [kW]':>17}{'after guard [kW]':>18}"
          f"{'residual [kW]':>15}")
    ones = np.ones(4, dtype=np.float32)
    cop = CoPModel(efficiency_heat=0.3 * ones, target_heat=45.0 * ones,
                   efficiency_cool=0.3 * ones, target_cool=7.0 * ones,
                   nominal_power_heat=10.0 * ones, nominal_power_cool=10.0 * ones)
    idx_t = obs_index("outdoor_dry_bulb_temperature")
    for net in (5.0, 20.0, 40.0, 80.0):
        shield = CBFShield(config=CBFConfig(P_grid_max=100.0, P_building_max=80.0),
                           num_buildings=4, soc_rate=np.full(4, 0.2), elec_idx=1,
                           cop_model=cop, hvac_idx=2, enforce_soc=False)
        st = []
        for _ in range(4):
            o = np.zeros(OBS_LEN)
            o[IDX_NET], o[idx_t] = net, 5.0
            st.append(o)
        cmd = np.zeros((4, 3), dtype=np.float32)
        cmd[:, 2] = 1.0
        safe = shield.project(cmd, st)
        p_nom = np.where(safe[:, 2] > 0, cop.p_h, cop.p_c)
        imp = float(np.maximum(np.full(4, net) + np.abs(safe[:, 2]) * p_nom, 0.0).sum())
        print(f"      {net:>16.1f}{4 * net:>17.1f}{imp:>18.2f}"
              f"{max(imp - shield.grid_cap(), 0.0):>15.2f}")


# ----------------------------------------------------------------------- converge ---

def section_converge(quick: bool = False) -> None:
    """How often does schedule_executable leave a sub-p_min allocation for the band?"""
    trials = 100 if quick else 400
    rng = np.random.default_rng(0)
    solved = survivors = 0
    for _ in range(trials):
        n = int(rng.integers(2, 6))
        p_max = rng.uniform(4.0, 11.0, n)
        p_min = p_max * rng.uniform(0.1, 0.6, n)
        capacity = rng.uniform(30.0, 70.0, n)
        model = EVFleetModel([True] * n, p_max, p_min, np.full(n, 0.95),
                             BatteryModel(capacity, p_max, np.zeros(n),
                                          [FLAT] * n, [FLAT] * n))
        state = FleetState([True] * n, rng.uniform(0.2, 0.6, n), rng.uniform(0.6, 0.95, n),
                           rng.integers(1, 6, n), np.zeros((6, n)),
                           float(rng.uniform(3.0, 25.0)))
        try:
            sol = schedule_executable(model, state, rng.uniform(0.0, p_max, n))
        except RuntimeError:
            continue
        solved += 1
        survivors += int(((sol["now_kw"] > 1e-6)
                          & (sol["now_kw"] < model.p_min - 1e-9)).any())
    print(f"\n[converge] schedule_executable, {solved} random fleets solved of {trials} drawn")
    print(f"      allocations still inside the charger dead band after max_rounds=4: "
          f"{survivors} ({survivors / max(solved, 1):.4f})")
    print("      -- so apply_dead_band is a no-op on the 'lp' rule in this sample; the "
          "round-up measured in [report] is reachable only on the six myopic rules.")


SECTIONS = {"soc": section_soc, "cap": section_cap, "deadline": section_deadline,
            "report": section_report, "cbf": section_cbf, "converge": section_converge,
            "tank": section_tank}


def main(argv: list) -> None:
    quick = "--quick" in argv
    names = [a for a in argv if not a.startswith("-")] or list(SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in names:
        SECTIONS[name](quick=quick)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
