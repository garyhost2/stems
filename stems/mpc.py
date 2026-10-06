r"""Receding-horizon economic model-predictive control over the storage devices.

One controller class, ``StorageMPC``. The two arms it serves differ only in which
forecaster is handed to it (``stems.forecast``): the receding-horizon arm gets
``CausalForecaster``, which sees exactly the reinforcement-learning agent's information
set, and the oracle arm gets ``OracleForecaster``, which replays the realised series.
Everything downstream of the forecast -- the plant models, the constraints, the solver,
the horizon -- is identical, so a difference between the two arms is a forecast effect
and nothing else.

Decision variables, per building :math:`b` and horizon step :math:`k`
(all dimensionless and in :math:`[0, 1]`; one hour per step, SI elsewhere):

    :math:`c_{b,k}`   battery charge command                     :math:`[0, 1]`
    :math:`d_{b,k}`   battery discharge command                  :math:`[0, 1]`
    :math:`h_{b,k}`   hot-water tank charge command              :math:`[0, 1]`
    :math:`g_{b,k}`   hot-water tank discharge command           :math:`[0, 1]`
    :math:`m_{b,k}`   grid import                                :math:`\ge 0` [kW]

The applied action is :math:`a^{\mathrm{batt}} = c - d \in [-1, 1]` and
:math:`a^{\mathrm{dhw}} = \bar{a}_b h - g`, where :math:`\bar{a}_b` is the schema's
own upper bound on the hot-water action (``STEMSEnvironment.dhw_info()["action_bound"]``,
0.54-0.87 on ``tx_travis_8b``). ``c + d <= 1`` and ``h + g <= 1`` keep the pair inside
the physical action range.

**Objective** (currency over the horizon):

.. math::
    \min \; \sum_{k=0}^{H-1} \pi_k \Delta t \sum_b m_{b,k}
      \;+\; \rho \Big( \textstyle\sum_k \sigma^{\mathrm{grid}}_k
      + \sum_{b,k} \sigma^{\mathrm{bld}}_{b,k} + \sum_{b,k} \sigma^{\mathrm{soc}}_{b,k}
      + \sum_b \sigma^{\mathrm{term}}_b \Big)

with :math:`\pi_k` the import price [currency/kWh] and :math:`\sigma` the slacks on the
constraints that an exogenous base load can make infeasible on its own. The slacks exist
so the optimisation *always* returns an action: a solver failure in a control loop is a
silent fall-back to some default, which is how a baseline becomes a strawman. Slack use
is recorded and reported (``StorageMPC.diagnostics``), never swallowed.

**Constraints.** Import definition :math:`m_{b,k} \ge \ell_{b,k} + e^{\mathrm{batt}}_{b,k}
+ e^{\mathrm{dhw}}_{b,k}` together with :math:`m \ge 0` and a positive price makes
:math:`m = \max(\cdot, 0)` at the optimum, which is the import, exported surplus earning
nothing. Per-building cap :math:`m_{b,k} \le \bar{P}_{\mathrm{bld}} +
\sigma^{\mathrm{bld}}`; district cap :math:`\sum_b m_{b,k} \le \bar{P}_{\mathrm{grid}} +
\sigma^{\mathrm{grid}}`; state-of-charge band and a terminal condition, below.

**Plant model.** The exact ``BatteryModel`` and ``TankModel`` inverses, not a constant
rate. They are nonlinear (efficiency and capacity-power curves), so each solve uses a
*secant* linearisation that is exact at the three corners the controller actually asks
for -- idle, full charge, full discharge --

.. math::
    \mathrm{soc}_{k+1} = (1-\lambda_b)\,\mathrm{soc}_k + o_{b,k}
      + \gamma^{+}_{b,k} c_{b,k} - \gamma^{-}_{b,k} d_{b,k},

where :math:`\lambda_b` is the device's own loss coefficient, :math:`\gamma^{\pm}` are
the state-of-charge changes the exact model reports at :math:`c=1` and :math:`d=1`
evaluated at the linearisation state :math:`\hat{s}_{b,k}`, and :math:`o_{b,k} =
\mathrm{idle}(\hat{s}_{b,k}) - (1-\lambda_b)\hat{s}_{b,k}` carries the offset. The energy
coefficients :math:`e^{\mathrm{batt}} = p^{+} c - p^{-} d` come from
``BatteryModel.accepted_kwh`` at the same corners. The whole problem is then re-solved
with the linearisation re-evaluated along the trajectory the previous solve chose
(``linearisations`` iterations, default 2); the first-order error this leaves is
measured, not assumed -- see ``tests/test_mpc.py::test_the_linearisation_tracks_the_exact_plant``.

**Terminal condition.** :math:`\mathrm{soc}_{b,H} \ge \mathrm{soc}_{b,0} -
\sigma^{\mathrm{term}}_b`: the battery must be handed back at least as charged as it was
found. Without it a finite-horizon economic MPC empties every store on the last step of
every horizon -- the textbook end effect -- and the measured cost then depends on
:math:`H` through an artefact rather than through foresight. The hot-water tank gets no
terminal condition because its content is worth nothing once the day's draws are served.

**Horizon.** :math:`H` is *our* choice and is stated as such. ``docs/LITERATURE.md``
section 10 records that no retrieved source gives a horizon length or a convention for
choosing one, so there is nothing to defer to.

It was chosen by measurement, and the measurement overruled the argument. The argument
said 24 h: the tariff on ``citylearn_schemas/tx_travis_8b`` is a three-level
time-of-use schedule with a 24 h period, and the storage traverses its usable band in
1.5-4.5 h, so a horizon shorter than the tariff period should miss the
off-peak-to-peak arbitrage. The sweep (``experiments/mpc_horizon.py``, eight buildings,
seven days, cost relative to doing nothing) says the gain is already exhausted at 12 h:

====  ==============  ==============  ===========  ===========  ============
H[h]  causal saving   oracle saving   causal peak  oracle peak  solve [ms]
====  ==============  ==============  ===========  ===========  ============
2      -0.77%           2.62%          39.7 kW      39.7 kW       19.1
4       0.78%           9.62%          57.5 kW      44.6 kW       29.3
6       4.67%          15.61%          62.6 kW      50.8 kW       40.2
12      8.07%          21.07%          62.5 kW      46.5 kW       72.3
24      7.96%          21.29%          61.6 kW      45.4 kW      137.4
48      7.87%          21.34%          62.0 kW      45.4 kW      268.4
====  ==============  ==============  ===========  ===========  ============

Doing nothing costs the same window 753.13 and peaks at 39.63 kW. Both forecasters
saturate at 12 h: the causal arm is *worse* at 24 h and 48 h than at 12, and the oracle
gains 0.22 and 0.27 percentage points for 1.9 and 3.7 times the solve time. Solve time
matters here, because it is part of the result for an MPC baseline
([maier2023approximating] reports an approximate MPC matching a rule-based controller
at 15% of the MPC's compute). ``DEFAULT_HORIZON_H`` is therefore 12, not the 24 the
tariff-period argument predicted. State-of-charge band violations were zero at every
horizon, for both forecasters. Raw record in
``experiments/diagnostics/mpc_horizon/``.

**Read the peak column before using this controller.** Its objective is the import
bill and nothing else, and with the district cap at its default 300 kW -- which these
eight buildings never approach -- minimising the bill *raises* the peak, by up to 58%
over doing nothing for the causal arm at 6 h. That is not a defect in the optimiser; it
is what a cost-only objective does when the constraint that would stop it is slack. Two
consequences. First, any comparison that reports peak demand must set the cap near the
uncontrolled peak, or this arm will look bad for a reason that has nothing to do with
model-predictive control. Second, the horizon chosen above is the cost-optimal horizon
for a non-binding cap, and the horizon that is right once the cap binds has not been
measured.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from stems.environment import thermostat_step
from stems.observations import obs_indices

__all__ = ["StorageMPC", "DEFAULT_HORIZON_H"]

#: Horizon in hours. Measured, not assumed: cost saturates here on ``tx_travis_8b`` for
#: both forecasters, and 24 h costs 1.9x the solve time for no gain. See the module
#: docstring for the sweep and CHANGELOG.md step 2.
DEFAULT_HORIZON_H = 12

_IDX_SOC_ELEC, _IDX_SOC_DHW, _IDX_DHW_DEMAND = obs_indices(
    "electrical_storage_soc", "dhw_storage_soc", "dhw_demand")
_IDX_T_IN, _IDX_T_COOL, _IDX_T_HEAT = obs_indices(
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point")


class StorageMPC:
    """Economic MPC over the battery and the hot-water tank.

    The heating/cooling device is not a decision variable. In ``hvac_control="setpoint"``
    -- the scenario default -- the environment overwrites the controller's heat-pump
    column with a thermostat loop anyway (``STEMSEnvironment.step``), so the only lever
    would be a set-point shift; and predicting its effect needs an indoor-temperature
    model, which on this testbed means CityLearn's learned dynamics. ``docs/REPORT_2026-10.md``
    section 2 shows those dynamics are not physical (an hour of -0.25 cooling moves a
    house by 4-13 degC, and one house does not respond to cooling at all), so an MPC that
    optimised against them would report a comfort result that is an artefact of the
    surrogate. The comfort band is therefore held by the same proportional thermostat
    every other arm uses, and the heat pump's electricity enters the optimisation inside
    the forecast base load. This is a limitation of the testbed, not of the formulation;
    on an RC model fitted to real data the heat pump becomes a decision variable and the
    band becomes a constraint.
    """

    def __init__(
        self,
        num_buildings: int,
        action_dim: int,
        battery,
        tank,
        forecaster,
        *,
        elec_idx: int = 1,
        dhw_idx: int = 0,
        hvac_idx: int = -1,
        hvac_control: str = "setpoint",
        horizon: int = DEFAULT_HORIZON_H,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        dhw_soc_cap: float = 0.95,
        dhw_action_bound: Optional[np.ndarray] = None,
        p_grid_max: float = 300.0,
        p_building_max: float = 80.0,
        hours_per_step: float = 1.0,
        linearisations: int = 2,
        penalty_scale: float = 1000.0,
        terminal_soc: str = "hold",
        solver: Optional[str] = None,
    ) -> None:
        if terminal_soc not in ("hold", "none"):
            raise ValueError(f"terminal_soc must be 'hold' or 'none', got {terminal_soc!r}")
        if hvac_control not in ("setpoint", "power"):
            raise ValueError(f"hvac_control must be 'setpoint' or 'power', got {hvac_control!r}")
        if battery is None:
            raise ValueError("StorageMPC needs the exact BatteryModel; pass "
                             "STEMSEnvironment.battery_model()")
        self.B = int(num_buildings)
        self.action_dim = int(action_dim)
        self.battery = battery
        self.tank = tank
        self.forecaster = forecaster
        self.elec_idx = int(elec_idx)
        self.dhw_idx = int(dhw_idx) if tank is not None else -1
        self.hvac_idx = int(hvac_idx)
        self.hvac_control = hvac_control
        self.H = max(1, int(horizon))
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.dhw_soc_cap = float(dhw_soc_cap)
        self.p_grid_max, self.p_building_max = float(p_grid_max), float(p_building_max)
        self.dt = float(hours_per_step)
        self.linearisations = max(1, int(linearisations))
        self.penalty_scale = float(penalty_scale)
        self.terminal_soc = terminal_soc
        self._solver = solver
        self.a_dhw = (np.ones(self.B) if dhw_action_bound is None
                      else np.asarray(dhw_action_bound, dtype=np.float64).reshape(-1))
        self._batt_keep = 1.0 - np.asarray(battery.loss, dtype=np.float64).reshape(-1)
        self._tank_keep = (np.ones(self.B) if tank is None
                           else 1.0 - np.asarray(tank.loss, dtype=np.float64).reshape(-1))
        self._problem = None
        self._coeff0: Dict[str, np.ndarray] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.reset()

    # -- controller protocol ---------------------------------------------------
    def reset(self) -> None:
        self._u_hvac = np.zeros(self.B, dtype=np.float32)
        self._last_plan: Optional[Dict[str, np.ndarray]] = None
        self._last_obs: Optional[List[np.ndarray]] = None
        if hasattr(self.forecaster, "reset"):
            self.forecaster.reset()

    def notify_executed(self, executed: np.ndarray, hvac_idx: Optional[int] = None) -> None:
        """The action that was actually applied, after any outer safety layer.

        The base-load decomposition in ``stems.forecast`` subtracts what the storage
        actually drew, so it has to be told the applied action, not the requested one.
        """
        if self._last_obs is not None:
            self.forecaster.note_applied(self._last_obs, np.asarray(executed, dtype=np.float64))
        if self.hvac_control == "power" and self.hvac_idx >= 0:
            self._u_hvac = np.asarray(executed, dtype=np.float32)[:, self.hvac_idx].copy()

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        self.forecaster.observe(next_obs_list)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        obs_list = [np.asarray(o, dtype=np.float64) for o in obs_list]
        self._last_obs = obs_list
        forecast = self.forecaster.predict(obs_list, self.H)
        plan = self._solve(obs_list, forecast)
        self._last_plan = plan

        # The plan is a state-of-charge trajectory; the command that realises its first
        # step comes from inverting the exact plant, so what the battery does is what
        # the optimiser planned and not a linear approximation of it.
        coeff = self._coeff0
        soc_b0 = np.clip(np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list]), 0.0, 1.0)
        idle_b = self.battery.next_soc(soc_b0, np.zeros(self.B))
        target_b = idle_b + coeff["gc_b"][0] * plan["c"][0] - coeff["gd_b"][0] * plan["d"][0]
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        actions[:, self.elec_idx] = np.clip(
            self.battery.action_for_soc(soc_b0, target_b), -1.0, 1.0)
        if self.dhw_idx >= 0 and self.tank is not None:
            soc_t0 = np.clip(np.array([float(o[_IDX_SOC_DHW]) for o in obs_list]), 0.0, 1.0)
            demand0 = np.maximum(np.asarray(forecast.dhw_demand_kwh[0], dtype=np.float64), 0.0)
            idle_t = self.tank.next_soc(soc_t0, np.zeros(self.B), demand0)
            target_t = (idle_t + coeff["gc_t"][0] * plan["h"][0]
                        - coeff["gd_t"][0] * plan["g"][0])
            actions[:, self.dhw_idx] = np.clip(
                self.tank.action_for_soc(soc_t0, target_t, demand0, self.a_dhw), -1.0, 1.0)
        if self.hvac_idx >= 0:
            actions[:, self.hvac_idx] = self._hvac_action(obs_list)
        # If no outer layer calls notify_executed, the forecaster still needs the
        # command it is about to see executed.
        self.forecaster.note_applied(obs_list, actions.astype(np.float64))
        return actions

    def _hvac_action(self, obs_list) -> np.ndarray:
        """Zero set-point shift, or the thermostat command when the arm drives power.

        In ``setpoint`` mode the environment runs the thermostat itself and a zero shift
        means "leave the band where the schema put it". In ``power`` mode the controller
        owns the heat-pump column, and emitting zero would mean no heating at all; it
        runs the identical thermostat loop (``stems.environment.thermostat_step``) that
        the rule-based arm and the environment's own set-point mode use, so the heat
        pump behaves the same under every arm and the comparison is about storage.
        """
        if self.hvac_control == "setpoint":
            return np.zeros(self.B, dtype=np.float32)
        col = lambda i: np.array([float(o[i]) for o in obs_list], dtype=np.float32)
        if any(len(o) <= _IDX_T_HEAT for o in obs_list):
            return np.zeros(self.B, dtype=np.float32)
        self._u_hvac = thermostat_step(self._u_hvac, col(_IDX_T_IN), col(_IDX_T_HEAT),
                                       col(_IDX_T_COOL), np.zeros(self.B, dtype=np.float32))
        return self._u_hvac

    # -- the optimisation ------------------------------------------------------
    def _battery_corners(self, soc: np.ndarray):
        """Exact state-of-charge and energy changes at idle, full charge, full discharge."""
        ones = np.ones_like(soc)
        idle = self.battery.next_soc(soc, 0.0 * ones)
        up = self.battery.next_soc(soc, ones)
        down = self.battery.next_soc(soc, -ones)
        e_up = self.battery.accepted_kwh(soc, ones)
        e_down = self.battery.accepted_kwh(soc, -ones)
        return (idle, np.maximum(up - idle, 0.0), np.maximum(idle - down, 0.0),
                np.maximum(e_up, 0.0), np.maximum(-e_down, 0.0))

    def _tank_corners(self, soc: np.ndarray, demand: np.ndarray):
        r"""Idle state, charge gain, discharge *rate*, and the energies of each.

        The discharge coefficient is deliberately **not** read at ``soc``. The tank
        model's one-step drop is :math:`\min(q/\eta_r,\; e_{\mathrm{init}})/C` -- the
        smaller of a *rate* limit set by the hot-water draw :math:`q` and an *energy*
        limit set by what is currently stored. Reading both at the current state
        conflates them, and the conflation has a fixed point: an empty tank reports
        that discharging is worth nothing, so the optimiser never charges it, so it
        stays empty and keeps reporting the same thing. Measured, before this was
        separated: ``gd_t`` was identically zero at every step of a 200-step rollout
        even with 1.4 kWh of forecast draw in the horizon, and the hot-water command
        was exactly 0.000 throughout (``CHANGELOG.md`` step 4).

        So the rate is evaluated at a reference state that holds enough energy to
        sustain it, and the energy limit is left to the linear programme, where it
        already lives as :math:`\mathrm{soc}_{t} \ge 0`. The charge coefficient *is*
        read at the current state, because saturation as the tank fills is a real
        state dependence and the constraint :math:`\mathrm{soc}_t \le` cap does not
        capture the taper.
        """
        ones = np.ones_like(soc)
        idle = self.tank.next_soc(soc, 0.0 * ones, demand)
        up = self.tank.next_soc(soc, self.a_dhw, demand)
        e_up = self.tank.drawn_kwh(soc, self.a_dhw, demand)
        ref = np.maximum(soc, self.dhw_soc_cap)
        idle_ref = self.tank.next_soc(ref, 0.0 * ones, demand)
        down_ref = self.tank.next_soc(ref, -ones, demand)
        e_down = self.tank.drawn_kwh(ref, -ones, demand)
        return (idle, np.maximum(up - idle, 0.0), np.maximum(idle_ref - down_ref, 0.0),
                np.maximum(e_up, 0.0), np.maximum(-e_down, 0.0))

    def _coefficients(self, soc_b_traj: np.ndarray, soc_t_traj: np.ndarray,
                      demand: np.ndarray, plan: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, np.ndarray]:
        """Affine coefficients of the one-step maps, evaluated along a trajectory.

        The state-of-charge coefficients are the exact reachable range at each state --
        idle, full charge, full discharge -- so the planned trajectory is *realisable*
        by construction: ``select_action`` inverts the plant to find the command that
        hits the planned state (``BatteryModel.action_for_soc``), rather than applying a
        normalised action and hoping the secant was right.

        The energy coefficients are the part that stays approximate, because cost is a
        nonlinear function of the command. On the first pass they are the corner
        secants. On later passes, where ``plan`` gives an operating point, they are
        re-fitted so that ``pc * c - pd * d`` equals the energy the exact model reports
        at the command that realises the planned state -- sequential linear programming,
        exact at the point the optimiser actually chose.
        """
        H, B = self.H, self.B
        out = {k: np.zeros((H, B)) for k in
               ("off_b", "gc_b", "gd_b", "pc_b", "pd_b",
                "off_t", "gc_t", "gd_t", "pc_t", "pd_t")}
        tol = 1e-4
        for k in range(H):
            idle, gc, gd, pc, pd = self._battery_corners(soc_b_traj[k])
            out["off_b"][k] = idle - self._batt_keep * soc_b_traj[k]
            out["gc_b"][k], out["gd_b"][k] = gc, gd
            out["pc_b"][k], out["pd_b"][k] = pc, pd
            if plan is not None:
                c, d = plan["c"][k], plan["d"][k]
                target = idle + gc * c - gd * d
                e = self.battery.accepted_kwh(
                    soc_b_traj[k], self.battery.action_for_soc(soc_b_traj[k], target))
                out["pc_b"][k] = np.where(c > tol, np.maximum(e, 0.0) / np.maximum(c, tol), pc)
                out["pd_b"][k] = np.where(d > tol, np.maximum(-e, 0.0) / np.maximum(d, tol), pd)
            if self.tank is not None:
                idle, gc, gd, pc, pd = self._tank_corners(soc_t_traj[k], demand[k])
                out["off_t"][k] = idle - self._tank_keep * soc_t_traj[k]
                out["gc_t"][k], out["gd_t"][k] = gc, gd
                out["pc_t"][k], out["pd_t"][k] = pc, pd
                if plan is not None:
                    h, g = plan["h"][k], plan["g"][k]
                    target = idle + gc * h - gd * g
                    a = self.tank.action_for_soc(soc_t_traj[k], target, demand[k], self.a_dhw)
                    e = self.tank.drawn_kwh(soc_t_traj[k], a, demand[k])
                    out["pc_t"][k] = np.where(h > tol, np.maximum(e, 0.0) / np.maximum(h, tol), pc)
                    out["pd_t"][k] = np.where(g > tol, np.maximum(-e, 0.0) / np.maximum(g, tol), pd)
        return out

    def _build(self):
        import cvxpy as cp

        H, B = self.H, self.B
        v = {n: cp.Variable((H, B), nonneg=True) for n in ("c", "d", "h", "g", "imp")}
        v["soc_b"] = cp.Variable((H + 1, B))
        v["soc_t"] = cp.Variable((H + 1, B))
        v["s_soc"] = cp.Variable((H + 1, B), nonneg=True)
        v["s_tank"] = cp.Variable((H + 1, B), nonneg=True)
        v["s_bld"] = cp.Variable((H, B), nonneg=True)
        v["s_grid"] = cp.Variable(H, nonneg=True)
        v["s_term"] = cp.Variable(B, nonneg=True)

        p = {n: cp.Parameter((H, B)) for n in
             ("base", "off_b", "off_t")}
        for n in ("gc_b", "gd_b", "pc_b", "pd_b", "gc_t", "gd_t", "pc_t", "pd_t"):
            p[n] = cp.Parameter((H, B), nonneg=True)
        p["price"] = cp.Parameter(H, nonneg=True)
        p["soc_b0"] = cp.Parameter(B)
        p["soc_t0"] = cp.Parameter(B)
        p["rho"] = cp.Parameter(nonneg=True)
        p["tank_on"] = cp.Parameter(nonneg=True)

        keep_b = np.ascontiguousarray(np.broadcast_to(self._batt_keep, (H, B)))
        keep_t = np.ascontiguousarray(np.broadcast_to(self._tank_keep, (H, B)))
        e_batt = cp.multiply(p["pc_b"], v["c"]) - cp.multiply(p["pd_b"], v["d"])
        e_tank = cp.multiply(p["pc_t"], v["h"]) - cp.multiply(p["pd_t"], v["g"])

        cons = [
            v["soc_b"][0] == p["soc_b0"],
            v["soc_t"][0] == p["soc_t0"],
            v["soc_b"][1:] == cp.multiply(keep_b, v["soc_b"][:-1]) + p["off_b"]
            + cp.multiply(p["gc_b"], v["c"]) - cp.multiply(p["gd_b"], v["d"]),
            v["soc_t"][1:] == cp.multiply(keep_t, v["soc_t"][:-1]) + p["off_t"]
            + cp.multiply(p["gc_t"], v["h"]) - cp.multiply(p["gd_t"], v["g"]),
            v["c"] + v["d"] <= 1.0,
            v["h"] + v["g"] <= 1.0,
            v["soc_b"] >= self.soc_min - v["s_soc"],
            v["soc_b"] <= self.soc_max + v["s_soc"],
            v["soc_t"] >= -v["s_tank"],
            v["soc_t"] <= self.dhw_soc_cap + v["s_tank"],
            v["imp"] >= p["base"] + e_batt + e_tank,
            v["imp"] <= self.p_building_max + v["s_bld"],
            cp.sum(v["imp"], axis=1) <= self.p_grid_max + v["s_grid"],
        ]
        if self.terminal_soc == "hold":
            cons.append(v["soc_b"][H] >= p["soc_b0"] - v["s_term"])
        else:
            cons.append(v["s_term"] == 0.0)
        # A parameter switch rather than two compiled problems: with no tank the
        # hot-water commands are pinned to zero and the tank state is held constant.
        cons += [v["h"] <= p["tank_on"], v["g"] <= p["tank_on"]]

        energy_cost = (p["price"] @ cp.sum(v["imp"], axis=1)) * self.dt
        slack = (cp.sum(v["s_grid"]) + cp.sum(v["s_bld"]) + cp.sum(v["s_soc"])
                 + cp.sum(v["s_tank"]) + cp.sum(v["s_term"]))
        problem = cp.Problem(cp.Minimize(energy_cost + p["rho"] * slack), cons)
        self._problem = (problem, v, p)
        return self._problem

    def _solve(self, obs_list, forecast) -> Dict[str, np.ndarray]:
        import cvxpy as cp

        H, B = self.H, self.B
        problem, v, p = self._problem or self._build()

        soc_b0 = np.clip(np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list]), 0.0, 1.0)
        soc_t0 = (np.clip(np.array([float(o[_IDX_SOC_DHW]) for o in obs_list]), 0.0, 1.0)
                  if self.tank is not None else np.zeros(B))
        demand = np.asarray(forecast.dhw_demand_kwh, dtype=np.float64)
        base = np.asarray(forecast.base_kw, dtype=np.float64)
        price = np.maximum(np.asarray(forecast.price, dtype=np.float64), 0.0)

        p["base"].value = base
        p["price"].value = price
        p["soc_b0"].value = soc_b0
        p["soc_t0"].value = soc_t0
        p["rho"].value = float(self.penalty_scale * max(price.max(), 1e-3))
        p["tank_on"].value = 0.0 if self.tank is None else 1.0

        soc_b_traj = np.broadcast_to(soc_b0, (H, B)).copy()
        soc_t_traj = np.broadcast_to(soc_t0, (H, B)).copy()
        plan = {k: np.zeros((H, B)) for k in ("c", "d", "h", "g")}
        last_plan = None
        status = "not solved"
        for _ in range(self.linearisations):
            coeffs = self._coefficients(soc_b_traj, soc_t_traj, demand, last_plan)
            for name, value in coeffs.items():
                p[name].value = value
            try:
                problem.solve(solver=self._solver or cp.SCIPY,
                              scipy_options={"method": "highs"})
            except Exception:
                problem.solve()
            status = problem.status
            if v["c"].value is None:
                break
            for key in ("c", "d", "h", "g"):
                plan[key] = np.clip(np.asarray(v[key].value, dtype=np.float64), 0.0, 1.0)
            soc_b_traj = np.asarray(v["soc_b"].value, dtype=np.float64)[:H]
            soc_t_traj = np.asarray(v["soc_t"].value, dtype=np.float64)[:H]
            last_plan = plan
        self._coeff0 = coeffs

        self.diagnostics = {
            "status": status,
            "grid_slack_kwh": (float(np.sum(v["s_grid"].value))
                               if v["s_grid"].value is not None else float("nan")),
            "soc_slack": (float(np.sum(v["s_soc"].value))
                          if v["s_soc"].value is not None else float("nan")),
            "objective": (float(problem.value) if problem.value is not None
                          else float("nan")),
            "linearisations": self.linearisations,
        }
        return plan

    def planned_soc(self) -> Optional[np.ndarray]:
        """The battery state-of-charge trajectory the last solve committed to, ``(H+1, B)``.

        This, not the normalised commands, is what the controller plans: the command
        applied at each step is whatever ``BatteryModel.action_for_soc`` says reaches
        the next planned state. Exposed because "the plan stays inside the band" is a
        statement about this array.
        """
        if self._problem is None:
            return None
        value = self._problem[1]["soc_b"].value
        return None if value is None else np.asarray(value, dtype=np.float64)

    # -- the rest of the controller protocol -----------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """MPC does not learn. Returns an empty stat dict, as the rule arms do."""
        return {}

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None
