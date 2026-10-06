#!/usr/bin/env python3
"""Does the model-predictive horizon matter, and where does it stop mattering?

``docs/LITERATURE.md`` section 10 records that no retrieved source states a horizon
length for building MPC or a convention for choosing one. So ``stems.mpc`` cannot defer
to the field, and ``DEFAULT_HORIZON_H`` has to be our choice, justified by measurement
on this testbed and presented as ours.

This script sweeps the horizon on a fixed evaluation window and reports, for each H:
the electricity cost, the district peak import, the mean solve time per control step,
and the state-of-charge band violations. The horizon to adopt is the smallest one past
which cost stops improving -- buying foresight costs solve time, and an MPC that is too
slow to run in a control loop is not a baseline a deployment could use.

Both forecasters are swept, because the answer can differ: perfect foresight has more
to gain from a longer horizon than a seasonal-naive forecast whose error grows with the
lead time.

    python -B experiments/mpc_horizon.py --days 7 --horizons 2 4 6 12 24 48

Writes a JSON record to ``experiments/diagnostics/mpc_horizon/`` and prints a
table. It deliberately does not write under ``results/``: that tree holds evaluation
run records produced by ``experiments/runner.py``, and a diagnostic sweep is not one. Nothing here
trains; it is an experiment about the optimiser, not about a policy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.forecast import CausalForecaster, record_idle_episode
from stems.mpc import StorageMPC
from stems.observations import obs_indices
from stems.utils import set_seed

from experiments.scenario import TX_SCHEMA

IDX_PRICE, IDX_NET, IDX_SOC = obs_indices(
    "electricity_pricing", "net_electricity_consumption", "electrical_storage_soc")


def make_env(schema: str, steps: int, seed: int = 0) -> STEMSEnvironment:
    return STEMSEnvironment(schema=schema, seed=seed, heat_pump=True,
                            env_kwargs={"episode_time_steps": [(0, steps - 1)]},
                            hvac_control="setpoint")


def roll(controller, env, steps: int, cfg: STEMSConfig) -> Dict[str, Any]:
    """One evaluation pass. ``controller=None`` is the do-nothing reference."""
    obs, _ = env.reset()
    cost = 0.0
    peak = 0.0
    solve_times: List[float] = []
    violations = 0
    n = 0
    for _ in range(steps):
        if controller is None:
            actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        else:
            t0 = time.perf_counter()
            actions = controller.select_action(obs)
            solve_times.append(time.perf_counter() - t0)
        price = float(obs[0][IDX_PRICE])
        nxt, _r, term, trunc, _i = env.step(actions)
        if controller is not None:
            controller.notify_executed(env.executed_actions)
            controller.observe(nxt)
        imports = np.maximum(np.array([o[IDX_NET] for o in nxt], dtype=float), 0.0)
        cost += price * float(imports.sum())
        peak = max(peak, float(imports.sum()))
        soc = np.array([o[IDX_SOC] for o in nxt], dtype=float)
        violations += int(((soc < cfg.cbf.SOC_min - 1e-3)
                           | (soc > cfg.cbf.SOC_max + 1e-3)).sum())
        obs = nxt
        n += 1
        if term or trunc:
            break
    return {"cost": cost, "peak_kw": peak, "steps": n,
            "soc_violations": violations,
            "median_solve_ms": (float(np.median(solve_times)) * 1e3
                                if solve_times else 0.0),
            "p95_solve_ms": (float(np.percentile(solve_times, 95)) * 1e3
                             if solve_times else 0.0)}


def build(env, cfg, horizon: int, oracle: bool, steps: int) -> StorageMPC:
    battery, tank = env.battery_model(), env.dhw_tank_model()
    if oracle:
        tape_env = make_env(env_schema, steps)
        forecaster = record_idle_episode(tape_env, steps)
    else:
        forecaster = CausalForecaster(env.num_buildings, battery, tank,
                                      env.electrical_storage_action_index,
                                      env.dhw_action_index)
    return StorageMPC(env.num_buildings, env.action_dim, battery, tank, forecaster,
                      elec_idx=env.electrical_storage_action_index,
                      dhw_idx=env.dhw_action_index, hvac_idx=env.hvac_action_index,
                      hvac_control=env.hvac_control, horizon=horizon,
                      soc_min=cfg.cbf.SOC_min, soc_max=cfg.cbf.SOC_max,
                      dhw_action_bound=env.dhw_info()["action_bound"],
                      p_grid_max=cfg.cbf.P_grid_max,
                      p_building_max=cfg.cbf.P_building_max)


def main() -> None:
    global env_schema
    ap = argparse.ArgumentParser(description="MPC horizon sweep")
    ap.add_argument("--schema", default=TX_SCHEMA)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--horizons", nargs="+", type=int,
                    default=[2, 4, 6, 12, 24, 48])
    ap.add_argument("--grid-cap", type=float, default=300.0)
    ap.add_argument("--building-cap", type=float, default=80.0)
    ap.add_argument("--out", default="experiments/diagnostics/mpc_horizon")
    args = ap.parse_args()

    env_schema = args.schema
    steps = args.days * 24
    cfg = STEMSConfig()
    cfg.cbf.P_grid_max = args.grid_cap
    cfg.cbf.P_building_max = args.building_cap

    set_seed(0)
    rows: List[Dict[str, Any]] = []
    idle = roll(None, make_env(args.schema, steps), steps, cfg)
    idle.update(horizon=0, forecast="none", arm="idle")
    rows.append(idle)
    print(f"idle                       cost {idle['cost']:9.2f}  peak {idle['peak_kw']:7.2f} kW")

    for oracle in (False, True):
        label = "oracle" if oracle else "causal"
        for H in args.horizons:
            env = make_env(args.schema, steps)
            mpc = build(env, cfg, H, oracle, steps)
            row = roll(mpc, env, steps, cfg)
            row.update(horizon=H, forecast=label, arm="mpc-oracle" if oracle else "mpc",
                       saving_pct=100.0 * (idle["cost"] - row["cost"]) / idle["cost"])
            rows.append(row)
            print(f"{label:6s} H={H:3d}  cost {row['cost']:9.2f}  "
                  f"saving {row['saving_pct']:5.2f}%  peak {row['peak_kw']:7.2f} kW  "
                  f"solve {row['median_solve_ms']:6.1f} ms  "
                  f"soc-violations {row['soc_violations']}", flush=True)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"horizon_sweep_{args.days}d.json"
    path.write_text(json.dumps(
        {"schema": args.schema, "days": args.days, "steps": steps,
         "grid_cap_kw": args.grid_cap, "building_cap_kw": args.building_cap,
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
