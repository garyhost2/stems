"""A second implementation of ``stems.protocols.Environment``, backed by logged CSV data.

A protocol with one implementation has not been shown to abstract anything. This is the
second one, and it is deliberately the *cheapest* adapter that a real site can supply:
one CSV per building of time-stamped measurements, plus a small JSON description of the
plant. If the framework's interface can be served from that, it can be served from a
building management system export.

What this adapter is for
------------------------
Three things, all of which are real work and none of which is control evaluation.

1. **Data conformance.** It fails loudly when a column is missing, a timestamp is out
   of order, a step is not the declared length, or a unit looks wrong -- at
   construction, not at step 4000 of a run.
2. **Interface conformance.** It proves a controller and a shield built against
   ``Environment``/``PlantProvider`` run without importing CityLearn.
3. **Offline feasibility checking.** A shield can be projected against the recorded
   state at every step, which answers "would this constraint have been violated, and
   would the projection have been feasible, on real data" -- a question worth asking
   before any deployment.

**What it is NOT, stated first so nobody reads a number off it.** It is *open loop*.
The recorded observations do not respond to the actions the controller takes: the
indoor temperature at step t+1 is what the real building did under the real controller,
not what it would have done under this one. Every quantity that depends on the
closed loop -- cost, peak, comfort violation under *this* policy -- is therefore
counterfactual and this adapter cannot produce it. ``step`` returns a reward of 0.0 for
that reason rather than a plausible-looking number, and ``info["open_loop"]`` is True
at every step. Closed-loop evaluation on real data needs an identified model
(``stems.comfort.RCThermalModel.identify``, ``BatteryModel``/``TankModel`` fitted to
the site) validated against held-out measurements first. That is the next seam.

Data contract
-------------
``manifest.json``::

    {
      "dt_hours": 1.0,
      "heat_pump": true,
      "buildings": [
        {"name": "b1", "csv": "b1.csv",
         "battery": {"capacity_kwh": 10.8, "nominal_power_kw": 2.88,
                     "loss_coefficient": 0.0},
         "dhw": {"capacity_kwh": 4.0, "heater_power_kw": 4.0, "heater_efficiency": 0.9,
                 "storage_efficiency": 1.0, "loss_coefficient": 0.008}}
      ],
      "action_names": ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]
    }

Each CSV carries a ``timestamp`` column (ISO 8601, strictly increasing, spacing equal to
``dt_hours``) and one column per observation name in
``stems.observations.selected_obs_names(heat_pump, ev_slots=0)``. Units are those of
``stems/observations.py``: kW, kWh, degC, dimensionless state of charge. Missing columns
are an error, not a zero-fill: a silently zero-filled observation is the single most
expensive kind of bug in this repository's experience, because the shield reads
observations by position and a zero is a legal value everywhere.

The plant blocks are optional. Without them the adapter still implements
``Environment``; it does not implement ``PlantProvider``, and
``stems.protocols.missing_capabilities`` says which methods are absent rather than
returning a model with invented parameters.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.battery import BatteryModel, TankModel
from stems.observations import index_in, selected_obs_names

__all__ = ["CSVReplayEnvironment", "ReplayManifestError", "write_manifest_template"]

DEFAULT_ACTION_NAMES = ("dhw_storage", "electrical_storage", "cooling_or_heating_device")


class ReplayManifestError(ValueError):
    """The logged data does not meet the contract. Raised at construction."""


@dataclass
class _BuildingSpec:
    name: str
    battery: Optional[Dict[str, float]]
    dhw: Optional[Dict[str, float]]


class CSVReplayEnvironment:
    """Replay logged building measurements behind ``stems.protocols.Environment``.

    Parameters
    ----------
    manifest
        Path to the JSON described in the module docstring.
    strict
        When True (the default) every audit below is fatal. Set False only to inspect
        a broken export; a non-strict adapter must not be used for anything reported.
    """

    def __init__(self, manifest: str | Path, strict: bool = True) -> None:
        self.manifest_path = Path(manifest)
        self.strict = bool(strict)
        doc = json.loads(self.manifest_path.read_text())
        self.dt_hours = float(doc.get("dt_hours", 1.0))
        self._heat_pump = bool(doc.get("heat_pump", True))
        self._action_names = list(doc.get("action_names", DEFAULT_ACTION_NAMES))
        self._obs_names = selected_obs_names(heat_pump=self._heat_pump, ev_slots=0)
        specs = doc.get("buildings") or []
        if not specs:
            raise ReplayManifestError(f"{self.manifest_path}: no buildings listed")
        self._specs = [_BuildingSpec(name=s["name"], battery=s.get("battery"),
                                     dhw=s.get("dhw")) for s in specs]
        self._series, self.audit = self._load(specs)
        self._B = len(self._specs)
        self._T = self._series.shape[1]
        self._t = 0
        self._executed = np.zeros((self._B, len(self._action_names)), dtype=np.float32)

    # ---------------------------------------------------------------- loading --

    def _load(self, specs: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        root = self.manifest_path.parent
        rows, lengths, problems = [], [], []
        for spec in specs:
            path = root / spec["csv"]
            if not path.is_file():
                raise ReplayManifestError(f"{path}: listed in the manifest, not on disk")
            with path.open(newline="") as fh:
                table = list(csv.DictReader(fh))
            if not table:
                raise ReplayManifestError(f"{path}: no rows")
            missing = [n for n in self._obs_names if n not in table[0]]
            if missing:
                raise ReplayManifestError(
                    f"{path}: missing observation column(s) {missing}. The adapter does "
                    "not zero-fill: the shield reads observations by position and zero "
                    "is a legal value for every one of them, so a fill would be a "
                    "silent corruption rather than a missing value.")
            if "timestamp" not in table[0]:
                raise ReplayManifestError(f"{path}: no 'timestamp' column")
            problems += self._audit_time(path, [r["timestamp"] for r in table])
            arr = np.empty((len(table), len(self._obs_names)), dtype=np.float32)
            for i, row in enumerate(table):
                for j, name in enumerate(self._obs_names):
                    raw = row[name]
                    if raw is None or raw.strip() == "":
                        problems.append(f"{path.name} row {i}: '{name}' is empty")
                        arr[i, j] = np.nan
                    else:
                        arr[i, j] = float(raw)
            problems += self._audit_values(path, arr)
            rows.append(arr)
            lengths.append(len(table))
        if len(set(lengths)) != 1:
            raise ReplayManifestError(
                f"buildings have different numbers of rows {lengths}; a neighbourhood "
                "under one shared cap has to be on one clock")
        if problems and self.strict:
            raise ReplayManifestError(
                f"{len(problems)} data problem(s); first ten:\n  "
                + "\n  ".join(problems[:10]))
        return np.stack(rows), {"problems": problems, "rows": lengths[0],
                                "buildings": len(rows)}

    def _audit_time(self, path: Path, stamps: Sequence[str]) -> List[str]:
        out: List[str] = []
        try:
            times = [datetime.fromisoformat(s) for s in stamps]
        except ValueError as exc:
            return [f"{path.name}: timestamp not ISO 8601 ({exc})"]
        step = self.dt_hours * 3600.0
        for i in range(1, len(times)):
            gap = (times[i] - times[i - 1]).total_seconds()
            if gap <= 0:
                out.append(f"{path.name} row {i}: timestamps not increasing")
            elif abs(gap - step) > 1.0:
                out.append(f"{path.name} row {i}: step is {gap:.0f} s, manifest says "
                           f"{step:.0f} s")
        return out[:50]

    def _audit_values(self, path: Path, arr: np.ndarray) -> List[str]:
        """Unit and range checks on the columns where a wrong unit is detectable."""
        out: List[str] = []
        checks = {"electrical_storage_soc": (0.0, 1.0), "dhw_storage_soc": (0.0, 1.0),
                  "hour": (1.0, 24.0), "day_type": (1.0, 8.0),
                  "outdoor_dry_bulb_temperature": (-60.0, 60.0),
                  "indoor_dry_bulb_temperature": (-10.0, 60.0)}
        for name, (lo, hi) in checks.items():
            if name not in self._obs_names:
                continue
            col = arr[:, index_in(self._obs_names, name)]
            bad = int(np.sum((col < lo) | (col > hi)))
            if bad:
                out.append(f"{path.name}: '{name}' outside [{lo}, {hi}] on {bad} row(s) "
                           f"(min {np.nanmin(col):.3g}, max {np.nanmax(col):.3g}) -- "
                           "check the unit")
        nans = int(np.isnan(arr).sum())
        if nans:
            out.append(f"{path.name}: {nans} non-numeric or empty cell(s)")
        return out

    # ------------------------------------------------------------ Environment --

    @property
    def num_buildings(self) -> int:
        return self._B

    @property
    def obs_dim(self) -> int:
        return len(self._obs_names)

    @property
    def action_dim(self) -> int:
        return len(self._action_names)

    @property
    def obs_names(self) -> List[str]:
        return list(self._obs_names)

    @property
    def action_names(self) -> List[str]:
        return list(self._action_names)

    @property
    def executed_actions(self) -> np.ndarray:
        """What the plant did.

        On replayed data the plant did whatever the *recorded* controller did, which
        this adapter does not know. Returning the commanded action would be a lie that
        every downstream metric would believe, so this returns zeros and
        ``info["executed_actions_known"]`` is False at every step.
        """
        return np.zeros_like(self._executed)

    @property
    def time_step(self) -> int:
        return self._t

    @property
    def num_steps(self) -> int:
        return self._T

    def index_of(self, name: str) -> int:
        return index_in(self._obs_names, name)

    def _obs(self) -> List[np.ndarray]:
        return [self._series[b, self._t].copy() for b in range(self._B)]

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        self._t = 0
        self._executed = np.zeros((self._B, len(self._action_names)), dtype=np.float32)
        return self._obs(), {"open_loop": True, "source": str(self.manifest_path)}

    def step(self, actions: np.ndarray
             ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]:
        self._executed = np.asarray(actions, dtype=np.float32).reshape(
            self._B, len(self._action_names)).copy()
        self._t += 1
        terminated = self._t >= self._T - 1
        self._t = min(self._t, self._T - 1)
        info = {"open_loop": True, "executed_actions_known": False,
                "commanded_actions": self._executed.copy()}
        # Reward 0.0, deliberately: a reward computed from observations the action did
        # not influence is a number that looks like a result and is not one.
        return self._obs(), [0.0] * self._B, bool(terminated), False, info

    # ----------------------------------------------------------- PlantProvider --

    def _has_plant(self, key: str) -> bool:
        return all(getattr(s, key) is not None for s in self._specs)

    @property
    def electrical_storage_action_index(self) -> int:
        return (self._action_names.index("electrical_storage")
                if "electrical_storage" in self._action_names else -1)

    @property
    def dhw_action_index(self) -> int:
        return (self._action_names.index("dhw_storage")
                if "dhw_storage" in self._action_names else -1)

    @property
    def hvac_action_index(self) -> int:
        return (self._action_names.index("cooling_or_heating_device")
                if "cooling_or_heating_device" in self._action_names else -1)

    def battery_model(self) -> BatteryModel:
        self._require("battery")
        flat = np.array([[0.0, 1.0], [1.0, 1.0]])
        b = [s.battery for s in self._specs]
        # A flat efficiency and power curve, because a logged export does not come with
        # a measured one. This is an assumption, declared here and in `plant_notes`,
        # not a calibration: fit the curves from the site's own charge/discharge logs
        # before any projection built on this is reported.
        return BatteryModel(capacity=[x["capacity_kwh"] for x in b],
                            nominal_power=[x["nominal_power_kw"] for x in b],
                            loss=[x.get("loss_coefficient", 0.0) for x in b],
                            eta_curves=[flat] * self._B, power_curves=[flat] * self._B,
                            hours_per_step=self.dt_hours)

    def battery_info(self) -> Dict[str, np.ndarray]:
        self._require("battery")
        b = [s.battery for s in self._specs]
        cap = np.array([x["capacity_kwh"] for x in b], dtype=np.float32)
        p = np.array([x["nominal_power_kw"] for x in b], dtype=np.float32)
        return {"capacity": cap, "nominal_power": p,
                "efficiency": np.ones(self._B, dtype=np.float32),
                "soc_rate": (p * self.dt_hours / np.maximum(cap, 1e-9)).astype(np.float32)}

    def dhw_tank_model(self) -> Optional[TankModel]:
        if not self._has_plant("dhw"):
            return None
        d = [s.dhw for s in self._specs]
        return TankModel(capacity=[x["capacity_kwh"] for x in d],
                         heater_power=[x["heater_power_kw"] for x in d],
                         heater_efficiency=[x["heater_efficiency"] for x in d],
                         storage_efficiency=[x.get("storage_efficiency", 1.0) for x in d],
                         loss=[x.get("loss_coefficient", 0.0) for x in d],
                         hours_per_step=self.dt_hours)

    def dhw_info(self) -> Dict[str, np.ndarray]:
        self._require("dhw")
        d = [s.dhw for s in self._specs]
        f = lambda k, default=None: np.array(
            [x[k] if default is None else x.get(k, default) for x in d], dtype=np.float32)
        return {"capacity": f("capacity_kwh"), "nominal_power": f("heater_power_kw"),
                "efficiency": f("heater_efficiency"),
                "loss_coefficient": f("loss_coefficient", 0.0),
                "action_bound": np.ones(self._B, dtype=np.float32)}

    def _require(self, key: str) -> None:
        if not self._has_plant(key):
            absent = [s.name for s in self._specs if getattr(s, key) is None]
            raise ReplayManifestError(
                f"no '{key}' block for building(s) {absent}. The adapter will not "
                "invent plant parameters: a projection against a made-up model is "
                "worse than no projection, because it reports a guarantee. Supply the "
                f"'{key}' block, or use this adapter as an Environment only.")

    @property
    def plant_notes(self) -> List[str]:
        """Assumptions a caller must carry into any result built on this adapter."""
        notes = ["open loop: recorded observations do not respond to the action",
                 "executed actions are unknown and reported as zero",
                 "reward is 0.0 by construction, not computed"]
        if self._has_plant("battery"):
            notes.append("battery efficiency and capacity-power curves assumed flat; "
                         "not fitted to this site")
        return notes


def write_manifest_template(path: str | Path, buildings: Sequence[str],
                            heat_pump: bool = True, dt_hours: float = 1.0) -> Path:
    """Write a manifest skeleton listing exactly the columns the adapter requires."""
    path = Path(path)
    doc = {
        "dt_hours": dt_hours,
        "heat_pump": heat_pump,
        "action_names": list(DEFAULT_ACTION_NAMES),
        "required_csv_columns": ["timestamp"] + selected_obs_names(heat_pump, 0),
        "buildings": [{"name": n, "csv": f"{n}.csv",
                       "battery": {"capacity_kwh": None, "nominal_power_kw": None,
                                   "loss_coefficient": 0.0},
                       "dhw": {"capacity_kwh": None, "heater_power_kw": None,
                               "heater_efficiency": None, "storage_efficiency": 1.0,
                               "loss_coefficient": 0.0}}
                      for n in buildings],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
