"""Forecasts a model-predictive controller is allowed to use.

Two forecasters behind one interface, so that the *only* difference between the two
MPC arms is which of them is plugged in:

``CausalForecaster``
    Sees exactly what the reinforcement-learning agent sees at the same instant: the
    current observation vector and a rolling window of past observations the same
    length as the agent's temporal-transformer window (``TransformerConfig.window_size``,
    24 steps = 24 h). Nothing else. It never reads a future value.

``OracleForecaster``
    Replays a recording of the *same* environment, so every quantity it returns for a
    future step is the value that step will actually take. This is the perfect-foresight
    upper reference. It is **our** choice of reference point, not a field convention:
    `docs/LITERATURE.md` section 10 records that no retrieved source establishes a
    perfect-foresight oracle as standard practice in building MPC.

Symbols and units (SI throughout; the time step is one hour, so kW and kWh/step
coincide numerically and are kept distinct in the names anyway):

    :math:`H`                horizon, number of future steps the MPC plans over      [-]
    :math:`B`                number of buildings                                     [-]
    :math:`\\pi_k`            electricity import price at step :math:`k`      [currency/kWh]
    :math:`\\ell_{b,k}`       *base load* of building :math:`b` at step :math:`k`      [kW]
    :math:`q_{b,k}`          domestic-hot-water draw of building :math:`b`     [kWh/step]

The **base load** is the quantity the MPC cannot change: everything in the building's
net electricity consumption except the two storage devices the MPC commands. It is
defined by subtraction from a measured quantity rather than by summing components,

.. math::  \\ell_{b,t} = n_{b,t} - e^{\\mathrm{batt}}_{b,t} - e^{\\mathrm{dhw}}_{b,t},

where :math:`n` is ``net_electricity_consumption`` as observed, and the two storage
terms are evaluated with the *exact* plant models (``stems.battery.BatteryModel`` and
``stems.battery.TankModel``) at the pre-step states of charge and the action that was
actually applied. Defining it this way, rather than as
``non_shiftable_load - solar_generation + dhw_demand/efficiency + heating_electricity``,
was a measured choice: see ``CHANGELOG.md`` step 2 for the two residuals.

Price lead times are not guessed. ``electricity_pricing_predicted_1`` and
``electricity_pricing_predicted_2`` were checked against the realised series on
``citylearn_schemas/tx_travis_8b`` over 300 steps and reproduce the price at
:math:`t+6\\,\\mathrm{h}` and :math:`t+12\\,\\mathrm{h}` with a mean absolute error of
0.00000 currency/kWh; every other lead in {1, 2, 3, 6, 12, 24} h is off by >= 0.07.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from stems.observations import obs_indices

__all__ = ["Forecast", "CausalForecaster", "OracleForecaster", "PRICE_PRED_LEAD_H",
           "record_idle_episode"]

_IDX_PRICE, _IDX_PRICE_1, _IDX_PRICE_2 = obs_indices(
    "electricity_pricing", "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2")
_IDX_DHW_DEMAND = obs_indices("dhw_demand")[0]
_IDX_NET = obs_indices("net_electricity_consumption")[0]
_IDX_SOC_ELEC = obs_indices("electrical_storage_soc")[0]
_IDX_SOC_DHW = obs_indices("dhw_storage_soc")[0]

#: Lead times, in hours, of ``electricity_pricing_predicted_{1,2}``. Measured, not
#: assumed; see the module docstring.
PRICE_PRED_LEAD_H = (6, 12)

#: One day at the environment's one-hour step. The seasonal-naive lag.
DAY = 24


@dataclass(frozen=True)
class Forecast:
    """What the MPC is handed for the next ``H`` steps.

    ``price`` is (H,), shared across buildings because the schema gives every included
    building the same pricing file (``experiments/scenario.py`` refuses a schema whose
    included buildings disagree on ``pricing``). ``base_kw`` and ``dhw_demand_kwh`` are
    (H, B).
    """

    price: np.ndarray
    base_kw: np.ndarray
    dhw_demand_kwh: np.ndarray

    def __post_init__(self) -> None:
        H = self.price.shape[0]
        if self.base_kw.shape[0] != H or self.dhw_demand_kwh.shape[0] != H:
            raise ValueError(
                f"forecast horizons disagree: price {self.price.shape}, "
                f"base {self.base_kw.shape}, dhw {self.dhw_demand_kwh.shape}")

    @property
    def horizon(self) -> int:
        return int(self.price.shape[0])


class _BaseDecomposition:
    """Turns an observed transition into the base load, using the exact plant models.

    The controller tells it which action was *applied* (after any shield), and it
    recovers :math:`\\ell_{b,t}` by subtracting the two storage terms the exact models
    say those actions drew.
    """

    def __init__(self, battery, tank, elec_idx: int, dhw_idx: int) -> None:
        self.battery = battery
        self.tank = tank
        self.elec_idx = int(elec_idx)
        self.dhw_idx = int(dhw_idx)

    def __call__(self, obs_list: Sequence[np.ndarray], applied: Optional[np.ndarray],
                 next_obs_list: Sequence[np.ndarray]) -> np.ndarray:
        net = np.array([float(o[_IDX_NET]) for o in next_obs_list], dtype=np.float64)
        if applied is None:
            return net
        applied = np.asarray(applied, dtype=np.float64)
        storage = np.zeros_like(net)
        if self.battery is not None and self.elec_idx >= 0:
            soc = np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list], dtype=np.float64)
            storage += self.battery.accepted_kwh(soc, applied[:, self.elec_idx])
        if self.tank is not None and self.dhw_idx >= 0:
            soc = np.array([float(o[_IDX_SOC_DHW]) for o in obs_list], dtype=np.float64)
            demand = np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list], dtype=np.float64)
            storage += self.tank.drawn_kwh(soc, applied[:, self.dhw_idx], demand)
        return net - storage


class CausalForecaster:
    """Seasonal-naive forecaster restricted to the agent's own information set.

    *Price.* Exact at leads 0, 6 and 12 h, because the observation carries those three
    values (see the module docstring). Elsewhere: the value 24 h earlier once a day of
    history exists, and a zero-order hold between the known anchors before that. A
    zero-order hold rather than linear interpolation because the tariff is a step
    function of time -- three levels on ``tx_travis_8b`` -- and interpolating a step
    function invents prices that the tariff never charges.

    *Base load and hot-water draw.* Seasonal naive with a level correction,

    .. math:: \\hat{\\ell}_{t+k} = \\ell_{t+k-24} + (\\ell_{t-1} - \\ell_{t-25}),

    falling back to persistence (:math:`\\hat{\\ell}_{t+k} = \\ell_{t-1}`) until a day of
    history has accumulated. This is the standard naive benchmark for hourly building
    load; it is deliberately *not* a trained model, because a trained forecaster would
    have to be fitted on data and would then need its own train/test split to avoid
    leaking the evaluation window.

    The most recent base load this can know at decision time :math:`t` is
    :math:`\\ell_{t-1}`: :math:`\\ell_t` is only measurable after the step is taken.
    """

    def __init__(self, num_buildings: int, battery=None, tank=None,
                 elec_idx: int = 1, dhw_idx: int = 0, memory: int = DAY) -> None:
        self.B = int(num_buildings)
        self.memory = max(int(memory), DAY)
        self._decompose = _BaseDecomposition(battery, tank, elec_idx, dhw_idx)
        self.reset()

    def reset(self) -> None:
        self._price: List[float] = []
        self._base: List[np.ndarray] = []
        self._demand: List[np.ndarray] = []
        self._pending: Optional[tuple] = None
        self._total = 0

    # -- information intake -------------------------------------------------------
    def note_applied(self, obs_list: Sequence[np.ndarray], applied: np.ndarray) -> None:
        """Record the action that was actually applied at the current observation."""
        self._pending = ([np.asarray(o, dtype=np.float64) for o in obs_list],
                         np.asarray(applied, dtype=np.float64))

    def observe(self, next_obs_list: Sequence[np.ndarray]) -> None:
        """Close the step: measure the base load that the applied action implies."""
        if self._pending is None:
            return
        obs_list, applied = self._pending
        base = self._decompose(obs_list, applied, next_obs_list)
        self._price.append(float(obs_list[0][_IDX_PRICE]))
        self._base.append(base)
        self._demand.append(np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list],
                                     dtype=np.float64))
        self._total += 1
        # Only a day of history plus a margin is ever read (the seasonal lag), so the
        # buffers are capped; `steps_observed` still counts every step the forecaster
        # was shown, which is what a caller asking "is it warmed up?" means.
        cap = self.memory + 2
        if len(self._base) > cap:
            del self._price[:-cap], self._base[:-cap], self._demand[:-cap]
        self._pending = None

    @property
    def steps_observed(self) -> int:
        return self._total

    # -- forecasting --------------------------------------------------------------
    def _seasonal(self, series: List[np.ndarray], k: int, fallback: np.ndarray
                  ) -> np.ndarray:
        """Value 24 h before step ``t + k``, level-corrected, or ``fallback``."""
        n = len(series)
        idx = n - DAY + k
        if idx < 0 or idx >= n or n < DAY + 1:
            return fallback
        correction = series[-1] - series[n - 1 - DAY]
        return series[idx] + correction

    def _price_at(self, obs: np.ndarray, k: int) -> float:
        anchors = {0: float(obs[_IDX_PRICE]),
                   PRICE_PRED_LEAD_H[0]: float(obs[_IDX_PRICE_1]),
                   PRICE_PRED_LEAD_H[1]: float(obs[_IDX_PRICE_2])}
        if k in anchors:
            return anchors[k]
        n = len(self._price)
        idx = n - DAY + k
        if 0 <= idx < n and n >= DAY:
            return float(self._price[idx])
        held = max(a for a in anchors if a <= k) if any(a <= k for a in anchors) else 0
        return anchors[held]

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int) -> Forecast:
        H = max(1, int(horizon))
        obs_list = [np.asarray(o, dtype=np.float64) for o in obs_list]
        price = np.array([self._price_at(obs_list[0], k) for k in range(H)],
                         dtype=np.float64)
        now_demand = np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list],
                              dtype=np.float64)
        last_base = (self._base[-1] if self._base
                     else np.array([float(o[_IDX_NET]) for o in obs_list], dtype=np.float64))
        base = np.stack([self._seasonal(self._base, k, last_base) for k in range(H)])
        demand = np.stack([now_demand if k == 0
                           else self._seasonal(self._demand, k, now_demand)
                           for k in range(H)])
        return Forecast(price=price, base_kw=base,
                        dhw_demand_kwh=np.maximum(demand, 0.0))


class OracleForecaster:
    """Perfect foresight, by replaying a recording of the identical environment.

    Price and hot-water draw are exogenous, so the recording gives them exactly. The
    base load is the *idle counterfactual*: what the buildings consume when neither
    storage device is commanded. It is not exactly the base load the MPC's own
    trajectory will produce -- discharging the tank changes how much the heater has to
    make up on later steps -- and that gap was measured rather than assumed: over a
    120-step window on ``tx_travis_8b`` with eight buildings, predicting the realised
    net consumption of a random-action rollout as ``idle base + exact battery term +
    exact tank term`` left a residual of mean 0.055 kW, standard deviation 0.623 kW,
    against a mean absolute net consumption of 2.993 kW (:math:`R^2 = 0.964`). See
    ``CHANGELOG.md`` step 2.

    So this is a perfect-foresight reference for the *exogenous* signals, not an exact
    oracle of the closed loop. The honest name for the gap is 0.62 kW of unmodelled
    hot-water coupling, and it is reported rather than hidden.
    """

    def __init__(self, price: np.ndarray, base_kw: np.ndarray,
                 dhw_demand_kwh: np.ndarray) -> None:
        self.price = np.asarray(price, dtype=np.float64).reshape(-1)
        self.base = np.asarray(base_kw, dtype=np.float64)
        self.demand = np.asarray(dhw_demand_kwh, dtype=np.float64)
        if self.base.shape[0] != self.price.shape[0] or \
                self.demand.shape[0] != self.price.shape[0]:
            raise ValueError(
                f"recording lengths disagree: price {self.price.shape}, "
                f"base {self.base.shape}, demand {self.demand.shape}")
        self.T, self.B = self.base.shape
        self.reset()

    def reset(self) -> None:
        self.t = 0

    def note_applied(self, obs_list: Sequence[np.ndarray], applied: np.ndarray) -> None:
        return None

    def observe(self, next_obs_list: Sequence[np.ndarray]) -> None:
        self.t = min(self.t + 1, self.T)

    @property
    def steps_observed(self) -> int:
        return self.t

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int) -> Forecast:
        H = max(1, int(horizon))
        idx = np.clip(np.arange(self.t, self.t + H), 0, self.T - 1)
        return Forecast(price=self.price[idx].copy(),
                        base_kw=self.base[idx].copy(),
                        dhw_demand_kwh=np.maximum(self.demand[idx].copy(), 0.0))


def record_idle_episode(env, max_steps: int) -> OracleForecaster:
    """Roll ``env`` once with the zero action and record what the oracle may know.

    The zero action leaves both storage devices alone -- ``BatteryModel.accepted_kwh``
    and ``TankModel.drawn_kwh`` both return exactly 0 at action 0, checked in
    ``tests/test_mpc.py::test_the_idle_action_moves_no_energy`` -- so the recorded
    ``net_electricity_consumption`` *is* the base load, with nothing to subtract.

    ``env`` is consumed: it is reset here and stepped to the end of the window, so pass
    a throwaway environment, not the one the controller will run on.
    """
    obs, _ = env.reset()
    price: List[float] = []
    base: List[np.ndarray] = []
    demand: List[np.ndarray] = []
    for _ in range(int(max_steps)):
        price.append(float(obs[0][_IDX_PRICE]))
        demand.append(np.array([float(o[_IDX_DHW_DEMAND]) for o in obs], dtype=np.float64))
        actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        nxt, _, term, trunc, _ = env.step(actions)
        base.append(np.array([float(o[_IDX_NET]) for o in nxt], dtype=np.float64))
        obs = nxt
        if term or trunc:
            break
    return OracleForecaster(np.array(price), np.stack(base), np.stack(demand))
