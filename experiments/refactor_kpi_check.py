#!/usr/bin/env python3
"""Behavioural fingerprint of the shield stack, for before/after comparison.

A refactor that claims not to change behaviour has to be checked against behaviour, not
against the test suite alone: the suite pins the properties someone thought to assert,
and a refactor can preserve every one of them and still move a trajectory. This script
runs deterministic rollouts on the real simulator and writes every key-performance
indicator the repository computes, so the two JSON files can be diffed exactly.

Three arms, chosen because between them they drive every projection in the stack:

``idle+calibrated``   the shield alone -- no policy, so any difference is the shield's.
``rbc+calibrated``    the rule, the electric-vehicle rule, and the LP cap shield.
``rl+calibrated``     the untrained agent: the control-barrier shield, the deadline
                      barriers and the fleet shield, driven by a seeded network.

The electric-vehicle schema is used throughout, because the fleet cap shield is the
piece most at risk in this refactor and it does not exist without chargers. Rollouts
are deterministic: ``set_seed(seed)`` before construction, ``explore=False``, and
CityLearn is never seeded from the run seed (see ``tests/test_provenance.py``).

    PYTHONPATH=. python -B experiments/refactor_kpi_check.py --tag before
    PYTHONPATH=. python -B experiments/refactor_kpi_check.py --tag after
    PYTHONPATH=. python -B experiments/refactor_kpi_check.py --compare before after

Writes ``experiments/diagnostics/refactor_kpi/<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.environment import STEMSEnvironment
from stems.utils import HistoryBuffer, set_seed

from experiments.controllers import ARMS, build_controller
from experiments.runner import make_config
from experiments.scenario import Scenario

OUT = REPO / "experiments" / "diagnostics" / "refactor_kpi"
EV_SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"
DEFAULT_ARMS = ("idle+calibrated", "rbc+calibrated", "rl+calibrated")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 10)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def rollout(name: str, steps: int, seed: int) -> Dict[str, Any]:
    from stems.metrics import MetricsCalculator

    set_seed(seed)
    scenario = Scenario(schema=EV_SCHEMA, n_buildings=8, days=7)
    env = STEMSEnvironment(schema=EV_SCHEMA, seed=seed, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, steps - 1)]},
                           hvac_control="setpoint")
    config = make_config(scenario)
    controller = build_controller(ARMS[name], env, config)

    metrics = MetricsCalculator(env.num_buildings, config.cbf,
                                soc_rate=env.battery_info()["soc_rate"],
                                heating_setpoint_idx=env.heating_setpoint_idx,
                                hvac_idx=env.hvac_action_index)
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    # The executed action stream is the sharpest fingerprint available: a projection
    # that changed by one bisection iteration shows here before it shows in a KPI.
    action_sum = np.zeros(env.action_dim, dtype=np.float64)
    action_absum = np.zeros(env.action_dim, dtype=np.float64)
    n = 0
    for _ in range(steps):
        actions = controller.select_action(obs, hist.get(), explore=False)
        raw = getattr(controller, "_last_nominal_actions",
                      getattr(controller, "_last_raw_actions", None))
        nxt, _r, term, trunc, _i = env.step(actions)
        if hasattr(controller, "observe"):
            controller.observe(nxt, env.ev_draw_kwh)
        executed = env.executed_actions
        metrics.add_step(obs, actions, nxt, raw_actions=raw, device_actions=executed)
        metrics.add_ev_departures(env.ev_departures)
        action_sum += executed.sum(axis=0)
        action_absum += np.abs(executed).sum(axis=0)
        hist.update(nxt)
        obs = nxt
        n += 1
        if term or trunc:
            break
    return {"arm": name, "steps": n, "seed": seed,
            "kpis": _jsonable(metrics.compute_all()),
            "executed_action_sum": _jsonable(action_sum),
            "executed_action_abs_sum": _jsonable(action_absum)}


def compare(a: Path, b: Path, tol: float = 0.0) -> int:
    left, right = json.loads(a.read_text()), json.loads(b.read_text())
    rows, worst = [], 0.0
    def walk(x, y, path):
        nonlocal worst
        if isinstance(x, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    rows.append((f"{path}.{k}", "missing", "missing"))
                else:
                    walk(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, list):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            d = abs(float(x) - float(y))
            worst = max(worst, d)
            if d > tol:
                rows.append((path, x, y))
        elif x != y:
            rows.append((path, x, y))
    walk(left, right, "")
    print(f"compared {a.name} against {b.name}: "
          f"{len(rows)} differing field(s), largest absolute difference {worst:.3e}")
    for path, x, y in rows[:40]:
        print(f"  {path}: {x} -> {y}")
    return 1 if rows else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=None, help="name of the fingerprint to write")
    ap.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    ap.add_argument("--steps", type=int, default=168)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        raise SystemExit(compare(OUT / f"{args.compare[0]}.json",
                                 OUT / f"{args.compare[1]}.json", args.tol))
    if not args.tag:
        ap.error("--tag is required unless --compare is given")
    rows = [rollout(a, args.steps, args.seed) for a in args.arms]
    path = OUT / f"{args.tag}.json"
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    for r in rows:
        k = r["kpis"]
        print(f"{r['arm']:<18} steps={r['steps']:<4} "
              + "  ".join(f"{n}={k[n]:.6f}" for n in sorted(k) if isinstance(k[n], float))[:160])
    print(f"\nwritten to {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
