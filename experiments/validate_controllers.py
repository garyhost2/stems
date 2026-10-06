#!/usr/bin/env python3
"""Eligibility gate: no controller enters a comparison table until it passes this.

Two things are checked on the *real* CityLearn environment, for every arm:

**Actuator evidence.** ``experiments/runner.py::ActuatorEvidence`` correlates what the
controller commanded with what the devices actually did -- charging commands against
state-of-charge rises, heat-pump commands against heating and cooling electricity. A
controller whose commands do not move the plant is not controlling anything, and its
cost number would be the uncontrolled building's cost wearing a label. The verdict is
``True`` (every device checked responds), ``False`` (one demonstrably does not) or
``None`` (not enough contrast in the commands to tell).

Note what ``None`` means and does not mean. A controller that has not been trained
emits whatever its initialisation emits; if that is nearly constant, the check cannot
separate "the actuator is dead" from "the command never varied". This script therefore
drives every learning arm with exploration on, which is the condition under which its
commands vary, and reports the verdict per device rather than only in aggregate.

**Computation time per control step.** Reported because this literature reports it:
[maier2023approximating] matches a rule-based controller at 15% of an MPC's compute,
and for a model-predictive baseline the solve time is part of the result, not an
implementation footnote. Median and 95th percentile over the rollout, in milliseconds,
measured around ``select_action`` only -- the environment step and the metric
bookkeeping are excluded, because they are the same for every arm.

    python -B experiments/validate_controllers.py --steps 168

Writes ``experiments/diagnostics/controller_validation/validation.json`` and prints a
table. It does not train: the learning arms are evaluated from their initialisation,
which is the point -- this gate asks whether a controller is wired to the plant, not
whether it is any good.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.utils import HistoryBuffer, set_seed

from experiments.controllers import ARMS, COMPARISON_POLICIES, build_controller
from experiments.runner import ActuatorEvidence, make_config
from experiments.scenario import Scenario, TX_SCHEMA

#: Arms validated here: the nine comparison controllers, plus the three incumbents they
#: will be compared against, so the gate is calibrated against arms already in use.
DEFAULT_ARMS = tuple(sorted(COMPARISON_POLICIES)) + ("rbc+calibrated", "rl+calibrated",
                                                     "idle+calibrated")


def validate_one(name: str, steps: int, scenario: Scenario, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=seed,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=scenario.hvac_control)
    config = make_config(scenario)
    if env.hvac_action_index < 0:
        config.reward.lambda_indoor = 0.0
    t_build = time.perf_counter()
    controller = build_controller(ARMS[name], env, config)
    build_s = time.perf_counter() - t_build

    evidence = ActuatorEvidence(env)
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    # Exploration on: a deterministic untrained policy can emit a near-constant command,
    # and the actuator check then returns "cannot tell" for a reason that has nothing to
    # do with the actuators.
    explore = bool(ARMS[name].learns)
    times: List[float] = []
    n = 0
    for _ in range(steps):
        window = hist.get()
        t0 = time.perf_counter()
        actions = controller.select_action(obs, window, explore=explore)
        times.append(time.perf_counter() - t0)
        nxt, _r, term, trunc, _i = env.step(actions)
        if hasattr(controller, "observe"):
            controller.observe(nxt, env.ev_draw_kwh)
        evidence.add(obs, env.executed_actions, nxt)
        hist.update(nxt)
        obs = nxt
        n += 1
        if term or trunc:
            break

    summary = evidence.summary()
    ms = np.asarray(times) * 1e3
    return {"arm": name, "steps": n, "explore": explore,
            "verified": summary["verified"],
            "battery": summary["battery"], "dhw": summary["dhw"], "hvac": summary["hvac"],
            "median_ms_per_step": float(np.median(ms)),
            "p95_ms_per_step": float(np.percentile(ms, 95)),
            "mean_ms_per_step": float(ms.mean()),
            "build_seconds": round(build_s, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Controller eligibility gate")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--steps", type=int, default=168, help="one week at hourly steps")
    ap.add_argument("--schema", default=TX_SCHEMA)
    ap.add_argument("--season", default="winter")
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/diagnostics/controller_validation")
    args = ap.parse_args()

    scenario = Scenario(schema=args.schema, season=args.season, days=args.days)
    rows: List[Dict[str, Any]] = []
    for name in args.arms:
        try:
            row = validate_one(name, args.steps, scenario, args.seed)
        except Exception as exc:
            row = {"arm": name, "verified": "error", "error": repr(exc),
                   "traceback": traceback.format_exc()}
        rows.append(row)
        verdict = row.get("verified")
        if verdict == "error":
            print(f"{name:16s} ERROR {row['error'][:90]}", flush=True)
        else:
            dev = lambda k: ("-" if row[k] is None else str(row[k].get("responds")))
            print(f"{name:16s} verified={str(verdict):5s} "
                  f"battery={dev('battery'):5s} dhw={dev('dhw'):5s} hvac={dev('hvac'):5s} "
                  f"median={row['median_ms_per_step']:8.3f} ms  "
                  f"p95={row['p95_ms_per_step']:8.3f} ms", flush=True)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "validation.json"
    path.write_text(json.dumps({"scenario": scenario.describe(), "steps": args.steps,
                                "seed": args.seed, "rows": rows}, indent=2,
                               default=str), encoding="utf-8")
    eligible = [r["arm"] for r in rows if r.get("verified") is True]
    blocked = [r["arm"] for r in rows if r.get("verified") is not True]
    print(f"\neligible for a comparison table: {eligible}")
    print(f"not eligible:                    {blocked}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
