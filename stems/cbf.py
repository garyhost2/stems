from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from stems.battery import BatteryModel
from stems.config import CBFConfig, SafetyConfig
from stems.deadline import DeadlineStorageBarrier
from stems.flexibility import FlexibilityPortfolio
from stems.observations import obs_indices
from stems.thermal import CoPModel, DHWReadinessBarrier

_IDX_T_OUT, _IDX_SOC_ELEC, _IDX_NET = obs_indices(
    "outdoor_dry_bulb_temperature", "electrical_storage_soc",
    "net_electricity_consumption")


class CBFShield:
    def __init__(
        self,
        config: Optional[CBFConfig] = None,
        num_buildings: int = 8,
        soc_rate: Optional[np.ndarray] = None,
        nominal_power: Optional[np.ndarray] = None,
        action_scale: float = 1.0,
        elec_idx: int = 1,
        safety_cfg: Optional[SafetyConfig] = None,
        enforce_soc: bool = True,
        dhw_barrier: Optional[DHWReadinessBarrier] = None,
        cop_model: Optional[CoPModel] = None,
        hvac_idx: int = -1,
        deadline_barriers: Optional[List[DeadlineStorageBarrier]] = None,
        coordination: str = "independent",
        battery_model: Optional[BatteryModel] = None,
    ) -> None:
        self.cfg = config or CBFConfig()
        self.safety = safety_cfg or SafetyConfig()
        self.B = num_buildings
        self.action_scale = float(action_scale)
        self.elec_idx = int(elec_idx)
        self.enforce_soc = bool(enforce_soc)
        if battery_model is None:
            if soc_rate is None:
                if self.enforce_soc:
                    raise ValueError("CBFShield needs battery_model or soc_rate to enforce "
                                     "the state-of-charge band")
                soc_rate = np.zeros(num_buildings, dtype=np.float32)
            battery_model = BatteryModel.linear(np.asarray(soc_rate, dtype=np.float64))
        self.battery = battery_model
        self.soc_rate = (battery_model.nominal_power * battery_model.dt
                         / battery_model.capacity).astype(np.float32)
        self.nominal_power = (np.asarray(nominal_power, dtype=np.float32).reshape(-1)
                              if nominal_power is not None else None)
        self.grid_guard = True
        self.dhw_barrier = dhw_barrier
        self.cop_model = cop_model
        self.hvac_idx = int(hvac_idx)
        if coordination not in ("independent", "proportional", "edf"):
            raise ValueError("coordination must be 'independent', 'proportional' "
                             f"or 'edf', got {coordination!r}")
        self.coordination = coordination
        self.deadline_barriers: List[DeadlineStorageBarrier] = list(
            deadline_barriers or [])
        if dhw_barrier is not None and dhw_barrier not in self.deadline_barriers:
            self.deadline_barriers.insert(0, dhw_barrier)

    @property
    def portfolio(self) -> FlexibilityPortfolio:
        """The deadline loads this shield enforces, as the framework object.

        Built per call from ``self.deadline_barriers`` rather than cached, because that
        list is a public attribute and callers may append to it after construction --
        ``tests/test_flexibility.py`` does, and relies on the appended barrier being
        enforced. (``experiments/controllers.py`` does *not*: it passes
        ``deadline_barriers=`` as a constructor argument and never mutates the list.)
        A cached portfolio would silently enforce a stale set, which is the failure
        mode a safety layer least wants.

        The cap is passed at projection time, not stored here: ``grid_cap()`` depends
        on the safety configuration's derate, which is also mutable.
        """
        return FlexibilityPortfolio.from_barriers(
            self.deadline_barriers, cap_kw=self.grid_cap(),
            coordination=self.coordination, net_index=_IDX_NET)


    def enforced_soc_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        margin = self.safety.soc_margin if self.safety.robust_margins else 0.0
        margin += self.safety.soc_tolerance
        lo = np.full(self.B, self.cfg.SOC_min + margin, dtype=np.float32)
        hi = np.full(self.B, self.cfg.SOC_max - margin, dtype=np.float32)
        if self.safety.anticipatory:
            buf = 0.5 * self.soc_rate * float(self.safety.invariance_horizon)
            lo = lo + buf
            hi = hi - buf
        mid = 0.5 * (self.cfg.SOC_min + self.cfg.SOC_max)
        lo = np.minimum(lo, mid - 1e-3)
        hi = np.maximum(hi, mid + 1e-3)
        return lo, hi

    def power_cap(self) -> float:
        derate = self.safety.power_derate if self.safety.robust_margins else 0.0
        return float(self.cfg.P_building_max * (1.0 - derate))

    def grid_cap(self) -> float:
        derate = self.safety.power_derate if self.safety.robust_margins else 0.0
        return float(self.cfg.P_grid_max * (1.0 - derate))


    def project(self, actions: np.ndarray, states: List[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        B, action_dim = actions.shape
        safe = actions.copy()
        if not self.enforce_soc:
            safe = self._apply_deadline_barriers(safe, states)
            return self._apply_hvac_power_guard(safe, states)
        soc = np.array([float(s[_IDX_SOC_ELEC]) for s in states], dtype=np.float32)
        lo, hi = self.enforced_soc_bounds()
        a_lo, a_hi = self.battery.safe_interval(soc, lo, hi, a_max=self.action_scale)
        safe[:, self.elec_idx] = np.clip(actions[:, self.elec_idx], a_lo, a_hi)

        if self.nominal_power is not None:
            safe = self._apply_power_guard(safe, states, a_lo)
        safe = self._apply_deadline_barriers(safe, states)
        return self._apply_hvac_power_guard(safe, states)

    def _apply_power_guard(self, safe: np.ndarray, states: List[np.ndarray],
                           a_lo: np.ndarray) -> np.ndarray:
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        nom = self.nominal_power
        pred_import = net + np.maximum(safe[:, self.elec_idx], 0.0) * nom
        p_cap = self.power_cap()
        over = pred_import > p_cap
        for i in np.where(over)[0]:
            allowed = max(0.0, (p_cap - net[i]) / max(nom[i], 1e-6))
            safe[i, self.elec_idx] = float(np.clip(allowed, min(a_lo[i], safe[i, self.elec_idx]),
                                                   safe[i, self.elec_idx]))
        if not self.grid_guard:
            return safe
        charge = np.maximum(safe[:, self.elec_idx], 0.0) * nom
        g_cap = self.grid_cap()
        imported = lambda s: float(np.maximum(net + s * charge, 0.0).sum())
        if imported(1.0) > g_cap and charge.sum() > 1e-9:
            lo_s, hi_s = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if imported(mid) <= g_cap else (lo_s, mid)
            charging = safe[:, self.elec_idx] > 0
            keep = np.minimum(np.maximum(a_lo, 0.0), safe[:, self.elec_idx])
            scaled = np.maximum(safe[:, self.elec_idx] * lo_s, keep)
            safe[charging, self.elec_idx] = scaled[charging]
        return safe


    def _apply_deadline_barriers(self, safe: np.ndarray,
                                 states: List[np.ndarray]) -> np.ndarray:
        """Every deadline load, then the shared cap.

        The body of this method moved to
        ``stems.flexibility.FlexibilityPortfolio.project_deadlines`` and
        ``.allocate_under_cap``, unchanged: the shield used to be the only place that
        knew a set of deadline loads contends for one cap, which meant a new load could
        not join the set without editing the battery shield. Deliberately *not* a
        change of behaviour -- see CHANGELOG.md, abstraction track step 2, for the
        before/after key-performance-indicator comparison that checks it.
        """
        return self.portfolio.project_deadlines(safe, states, cap_kw=self.grid_cap())

    def feasibility_report(self, states: List[np.ndarray]) -> Optional[dict]:
        return self.portfolio.feasibility(states, cap_kw=self.grid_cap())


    def _apply_hvac_power_guard(self, safe: np.ndarray,
                                states: List[np.ndarray]) -> np.ndarray:
        """Trim the heat-pump action to the per-building and district import caps.

        Two stages, both on the *achieved* import rather than on the action:

        1. per building, clip ``|a_hvac|`` so that ``net_b + |a_b|*p_nom_b`` stays under
           a cap derated by the coefficient-of-performance shortfall at the current
           outdoor temperature;
        2. district-wide, bisect one common shed factor ``s`` in [0, 1] applied to every
           building's heat-pump draw until ``sum_b max(net_b + s*|a_b|*p_nom_b, 0)``
           meets the grid cap.

        Stage 2 used to be ``safe[:, hvac] *= g_cap / total`` (audit B1). That does not
        bring the import to the cap, because ``total`` contains the uncontrollable
        ``net`` term, which the rescale cannot touch: with ``net`` alone above the cap no
        factor works, and with ``net`` below it the factor over-sheds. It is the same
        arithmetic error that was found and corrected in the battery grid guard
        (``_apply_power_guard``); this is now the same bisection.

        Dimensional caveat: ``safe[:, hvac_idx]`` is a power fraction only when the
        environment runs with ``hvac_control="power"``, which is ``STEMSEnvironment``'s
        own parameter default. Under ``hvac_control="setpoint"`` -- which is what
        ``experiments/scenario.py::Scenario`` defaults to, and therefore what every run
        driven through a Scenario uses -- that column is a +/-1.5 degC set-point offset,
        the integral thermostat issues the power command, and multiplying the offset by
        ``p_nom`` is meaningless. The guard is reached only when a ``cop_model`` is
        passed, which ``experiments/controllers.py`` does not do; see CHANGELOG.md for
        that decision.
        """
        if self.cop_model is None or self.hvac_idx < 0:
            return safe
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        t_out = np.array([float(s[_IDX_T_OUT]) for s in states], dtype=np.float32)
        heating = safe[:, self.hvac_idx] > 0.0

        p_nom = np.where(heating, self.cop_model.p_h, self.cop_model.p_c)
        cop_h = self.cop_model.cop(t_out, heating=True)
        cop_c = self.cop_model.cop(t_out, heating=False)
        cop = np.where(heating, cop_h, cop_c)
        cop_ref = np.maximum(
            np.where(heating,
                     self.cop_model.cop(np.full_like(t_out, 10.0), heating=True),
                     self.cop_model.cop(np.full_like(t_out, 30.0), heating=False)),
            1e-3)
        shortfall = np.clip(1.0 - cop / cop_ref, 0.0, 0.5)

        a_hvac = safe[:, self.hvac_idx]
        draw = np.abs(a_hvac) * p_nom
        p_cap = self.power_cap() * (1.0 - shortfall)
        pred = net + draw
        over = pred > p_cap
        for i in np.where(over)[0]:
            allowed = max(0.0, (p_cap[i] - net[i]) / max(p_nom[i], 1e-6))
            safe[i, self.hvac_idx] = float(np.sign(a_hvac[i]) *
                                           min(abs(a_hvac[i]), allowed))
        a_hvac_now = safe[:, self.hvac_idx].copy()
        g_cap = self.grid_cap()

        def shed(s: float) -> np.ndarray:
            """The action array the guard would actually store for shed factor s."""
            return (a_hvac_now * np.float32(s)).astype(np.float32)

        def imported(s: float) -> float:
            return float(np.maximum(net + np.abs(shed(s)) * p_nom, 0.0).sum())

        if imported(1.0) > g_cap and float(np.abs(a_hvac_now).sum()) > 1e-9:
            lo_s, hi_s = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo_s + hi_s)
                lo_s, hi_s = (mid, hi_s) if imported(mid) <= g_cap else (lo_s, mid)
            # `shed` is evaluated on the float32 values that get written back, so the
            # invariant the bisection established is the one the caller observes.
            safe[:, self.hvac_idx] = shed(lo_s)
        return safe


    def predicted_constraint_costs(self, actions: np.ndarray,
                                   states: List[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        B = self.B
        soc = np.array([float(s[_IDX_SOC_ELEC]) for s in states], dtype=np.float32)
        net = np.array([float(s[_IDX_NET]) for s in states], dtype=np.float32)
        rate = np.maximum(self.soc_rate, 1e-6)
        next_soc = soc + actions[:, self.elec_idx] * rate

        costs = np.zeros((B, 3), dtype=np.float32)
        costs[:, 0] = ((next_soc < self.cfg.SOC_min) | (next_soc > self.cfg.SOC_max)).astype(np.float32)
        if self.nominal_power is not None:
            pred = net + np.maximum(actions[:, self.elec_idx], 0.0) * self.nominal_power
            costs[:, 1] = (np.abs(pred) > self.cfg.P_building_max).astype(np.float32)
            costs[:, 2] = float(np.maximum(pred, 0.0).sum() > self.cfg.P_grid_max)
        else:
            costs[:, 1] = (np.abs(net) > self.cfg.P_building_max).astype(np.float32)
            costs[:, 2] = float(np.maximum(net, 0.0).sum() > self.cfg.P_grid_max)
        return costs


# `NeuralSafetyFilter` -- an untrained, uncalled network that regressed a nominal action
# onto the shield's output -- was removed here. The argument is in CHANGELOG.md under
# step 3: it would have been a *learned approximation to an exact, closed-form
# projection*, so it can only lose on the one axis this repository reports (constraint
# violation rate) while buying nothing on the axis that would justify it (solve time:
# the exact inverse is a 24-iteration bisection on a scalar, and the fleet LP is the
# only optimisation in the loop). Its declared ensemble (`num_ensemble=5`) and
# uncertainty gate (`uncertainty_threshold`) were parameters of a mechanism the class
# did not implement -- one trunk, one head, no ensemble, no gate.
