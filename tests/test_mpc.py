"""Known-answer tests for the model-predictive pair and its forecasters.

An MPC baseline is only worth reporting if three things are true, and each gets a test
with an answer known in advance rather than read off a run:

1. its plant model is the real plant (``stems.battery``), not a rate constant;
2. given a price profile whose cheap and expensive hours are known, it charges in the
   cheap ones and discharges in the expensive ones;
3. the causal arm cannot see the future and the oracle arm can -- and the oracle's
   advantage comes only from the forecast, because everything else is shared.

Most of these run on the synthetic mock environment so the suite stays fast. The two
that need the real plant curves and the real tariff are marked and skip when the
CityLearn dataset is not reachable.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel, TankModel
from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.forecast import (DAY, PRICE_PRED_LEAD_H, CausalForecaster, Forecast,
                            OracleForecaster, record_idle_episode)
from stems.mpc import DEFAULT_HORIZON_H, StorageMPC
from stems.observations import obs_index
from stems.utils import set_seed

SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
IDX_SOC = obs_index("electrical_storage_soc")
IDX_SOC_DHW = obs_index("dhw_storage_soc")
IDX_PRICE = obs_index("electricity_pricing")
IDX_NET = obs_index("net_electricity_consumption")
IDX_DHW_DEMAND = obs_index("dhw_demand")


def _real_env(span=(0, 167), **kwargs):
    try:
        return STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True,
                                env_kwargs={"episode_time_steps": [span]},
                                hvac_control="setpoint", **kwargs)
    except Exception as exc:  # pragma: no cover - depends on the local dataset
        pytest.skip(f"CityLearn dataset unavailable: {exc!r}")


def _toy_battery(B=2, capacity=10.0, power=2.5):
    """A flat-efficiency battery, so the secant linearisation is exact and the known
    answer is arithmetic rather than a tolerance."""
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    return BatteryModel(capacity=np.full(B, capacity), nominal_power=np.full(B, power),
                        loss=np.zeros(B), eta_curves=[flat] * B, power_curves=[flat] * B)


def _lossy_battery(B=2, capacity=10.0, power=2.5, eta=0.81):
    """Flat but lossy: a round-trip efficiency of ``eta`` makes cycling strictly cost."""
    curve = np.array([[0.0, 1.0], [eta, eta]])
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    return BatteryModel(capacity=np.full(B, capacity), nominal_power=np.full(B, power),
                        loss=np.zeros(B), eta_curves=[curve] * B, power_curves=[flat] * B)


def _mpc(battery, forecaster, B=2, **kwargs):
    kwargs.setdefault("horizon", 12)
    kwargs.setdefault("p_grid_max", 1e6)
    kwargs.setdefault("p_building_max", 1e6)
    return StorageMPC(B, 3, battery, None, forecaster, elec_idx=1, dhw_idx=-1,
                      hvac_idx=2, hvac_control="power", **kwargs)


class _FixedForecast:
    """A forecaster that returns a profile the test wrote, so the optimum is known."""

    def __init__(self, price, base, B):
        self.price, self.base, self.B = np.asarray(price, float), float(base), B
        self.t = 0

    def reset(self):
        self.t = 0

    def note_applied(self, obs_list, applied):
        return None

    def observe(self, next_obs_list):
        self.t += 1

    def predict(self, obs_list, horizon):
        idx = np.clip(np.arange(self.t, self.t + horizon), 0, len(self.price) - 1)
        return Forecast(price=self.price[idx],
                        base_kw=np.full((horizon, self.B), self.base),
                        dhw_demand_kwh=np.zeros((horizon, self.B)))


def _obs(B, soc, price=0.3, net=5.0):
    o = np.zeros((B, 30), dtype=np.float32)
    o[:, IDX_SOC] = soc
    o[:, IDX_PRICE] = price
    o[:, IDX_NET] = net
    return [o[i] for i in range(B)]


# ---------------------------------------------------------------------------
# 1. The plant model is the real plant
# ---------------------------------------------------------------------------
def test_the_idle_action_moves_no_energy():
    """Both exact models return exactly zero at action zero.

    ``stems.forecast.record_idle_episode`` relies on this: it treats the net consumption
    of a zero-action rollout as the base load with nothing subtracted.
    """
    env = _real_env()
    battery, tank = env.battery_model(), env.dhw_tank_model()
    B = env.num_buildings
    soc = np.full(B, 0.5)
    demand = np.full(B, 2.0)
    assert np.allclose(battery.accepted_kwh(soc, np.zeros(B)), 0.0)
    assert np.allclose(tank.drawn_kwh(soc, np.zeros(B), demand), 0.0)


def test_the_tank_state_and_the_tank_electricity_come_from_one_simulation():
    """``TankModel.next_soc`` must agree with ``drawn_kwh``: charging raises the state of
    charge and costs electricity, discharging lowers it and saves electricity."""
    env = _real_env()
    tank = env.dhw_tank_model()
    B = env.num_buildings
    soc = np.full(B, 0.4)
    demand = np.full(B, 1.0)
    bound = env.dhw_info()["action_bound"].astype(np.float64)
    up_soc = tank.next_soc(soc, bound, demand)
    up_kwh = tank.drawn_kwh(soc, bound, demand)
    down_soc = tank.next_soc(soc, -np.ones(B), demand)
    down_kwh = tank.drawn_kwh(soc, -np.ones(B), demand)
    assert (up_soc > tank.next_soc(soc, np.zeros(B), demand)).all()
    assert (up_kwh > 0).all()
    assert (down_soc <= tank.next_soc(soc, np.zeros(B), demand) + 1e-12).all()
    assert (down_kwh <= 0).all()
    assert (up_soc <= 1.0 + 1e-9).all() and (down_soc >= -1e-9).all()


def test_the_plant_inverse_returns_the_idle_command_on_a_flat_region():
    """A target that idling already achieves must map to the action zero.

    ``next_soc`` is flat in the action wherever the device cannot move energy. The hot
    water tank is the acute case: it can only discharge into an actual draw, so with no
    draw its state of charge is the same for *every* non-positive action. A bisection
    over the whole interval then converges to an endpoint, and the controller issues a
    full discharge command at every step for a tank that cannot discharge.

    This was not hypothetical. It is what the first version did, and it got through the
    known-answer tests because it changes nothing in the model -- only
    ``experiments/validate_controllers.py`` saw it, as a hot-water command that was
    -1.000 at every one of 960 steps. See CHANGELOG.md step 4.
    """
    env = _real_env()
    tank, battery = env.dhw_tank_model(), env.battery_model()
    B = env.num_buildings
    soc = np.full(B, 0.5)
    bound = env.dhw_info()["action_bound"].astype(np.float64)

    no_draw = np.zeros(B)
    idle_target = tank.next_soc(soc, np.zeros(B), no_draw)
    assert np.allclose(tank.action_for_soc(soc, idle_target, no_draw, bound), 0.0), (
        "with no hot-water draw the tank cannot discharge, so holding its state of "
        "charge is the zero command")
    # And the flat region really is flat, which is why the naive bisection failed.
    assert np.allclose(tank.next_soc(soc, -np.ones(B), no_draw), idle_target)

    # Same contract for the battery at its floor, where discharging is impossible.
    floor = np.full(B, 1e-6)
    idle_floor = battery.next_soc(floor, np.zeros(B))
    assert np.allclose(battery.action_for_soc(floor, idle_floor), 0.0, atol=1e-6)
    # And in the ordinary, strictly monotone case the inverse still finds the action.
    mid = np.full(B, 0.5)
    for frac in (-0.5, 0.0, 0.5):
        target = battery.next_soc(mid, np.full(B, frac))
        assert np.allclose(battery.next_soc(mid, battery.action_for_soc(mid, target)),
                           target, atol=1e-9)


def test_the_mpc_does_not_hold_the_hot_water_command_at_an_extreme():
    """Regression on the bug above, at the level the controller is used.

    A hot-water command pinned to one extreme for a whole rollout is the signature of
    an inverse that returned the end of a flat region, and it is invisible to every
    test that only looks at the state of charge.
    """
    from experiments.controllers import ARMS, build_controller

    env = _real_env(span=(0, 71))
    controller = build_controller(ARMS["mpc"], env, STEMSConfig())
    obs, _ = env.reset()
    commands = []
    for _ in range(48):
        actions = controller.select_action(obs)
        commands.append(actions[:, env.dhw_action_index].copy())
        obs = env.step(actions)[0]
        controller.observe(obs)
    commands = np.stack(commands)
    assert not np.allclose(commands, -1.0), "hot-water command pinned to full discharge"
    assert not np.allclose(commands, commands[0]), "hot-water command never moves"
    assert float(np.abs(commands).mean()) < 0.95, (
        f"hot-water command sits at an extreme (mean magnitude "
        f"{np.abs(commands).mean():.3f})")


def test_the_planned_state_of_charge_is_reachable_exactly():
    """The controller plans a state of charge and inverts the plant to command it.

    So the state trajectory it optimises over is not an approximation: whatever target
    the linear programme picks inside the reachable range, ``action_for_soc`` returns a
    command that the *exact* model takes to that target. This is the property the band
    constraint rests on, checked across all eight real batteries and a grid of states.
    """
    env = _real_env()
    battery = env.battery_model()
    B = env.num_buildings
    worst = 0.0
    for soc0 in (0.15, 0.35, 0.5, 0.7, 0.85):
        soc = np.full(B, soc0)
        idle = battery.next_soc(soc, np.zeros(B))
        gc = battery.next_soc(soc, np.ones(B)) - idle
        gd = idle - battery.next_soc(soc, -np.ones(B))
        for frac in (-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0):
            target = idle + gc * max(frac, 0.0) - gd * max(-frac, 0.0)
            achieved = battery.next_soc(soc, battery.action_for_soc(soc, target))
            worst = max(worst, float(np.abs(achieved - target).max()))
    assert worst < 1e-6, f"planned state of charge missed by {worst:.2e}"


def test_the_secant_state_of_charge_model_would_not_have_been_good_enough():
    """Why the inversion above is needed, as a measured number rather than an assertion.

    A secant between the idle and full-charge corners is what the first version of this
    controller used. The battery saturates as it fills, so the true state of charge is
    concave in the command and the secant sits below it: planning on the secant lets a
    trajectory leave the band it was constrained to. This records how far.
    """
    env = _real_env()
    battery = env.battery_model()
    B = env.num_buildings
    worst = 0.0
    for soc0 in (0.15, 0.35, 0.5, 0.7, 0.85):
        soc = np.full(B, soc0)
        idle = battery.next_soc(soc, np.zeros(B))
        gc = battery.next_soc(soc, np.ones(B)) - idle
        for u in (0.25, 0.5, 0.75):
            secant = idle + gc * u
            worst = max(worst, float(np.abs(secant - battery.next_soc(soc, np.full(B, u))).max()))
    assert worst > 0.05, (
        "the secant is now accurate, so the inversion may no longer be needed; "
        f"measured worst-case gap {worst:.4f}")


def test_the_energy_coefficients_converge_on_the_operating_point():
    """Cost stays a linearisation, and the second pass makes it exact where it matters.

    After the sequential-linear-programming refinement the energy the optimiser charged
    itself for the first step must equal the energy the exact model reports for the
    command it is about to apply.
    """
    env = _real_env()
    battery, tank = env.battery_model(), env.dhw_tank_model()
    B = env.num_buildings
    mpc = StorageMPC(B, env.action_dim, battery, tank,
                     CausalForecaster(B, battery, tank, 1, 0),
                     elec_idx=1, dhw_idx=0, hvac_idx=2, hvac_control="setpoint",
                     horizon=12, dhw_action_bound=env.dhw_info()["action_bound"],
                     p_grid_max=300.0, p_building_max=80.0, linearisations=3)
    obs, _ = env.reset()
    actions = mpc.select_action(obs)
    plan, coeff = mpc._last_plan, mpc._coeff0
    soc = np.array([float(o[IDX_SOC]) for o in obs], dtype=np.float64)
    modelled = coeff["pc_b"][0] * plan["c"][0] - coeff["pd_b"][0] * plan["d"][0]
    actual = battery.accepted_kwh(soc, actions[:, 1].astype(np.float64))
    # Sequential linear programming converges to a fixed point rather than landing on
    # one: each pass re-fits the cost at the previous pass's operating point and the
    # next solve moves slightly. What matters is that the residual is small against the
    # energy involved, so the optimiser is not pricing a different battery from the one
    # it commands. The bound is 1% of the energy actually drawn.
    residual = float(np.abs(modelled - actual).max())
    assert residual < 0.01 * float(np.abs(actual).max()), (
        f"the optimiser priced {modelled} kWh but the plant draws {actual} kWh "
        f"(residual {residual:.5f} kWh)")


# ---------------------------------------------------------------------------
# 2. Known-answer: it arbitrages a price profile the test wrote
# ---------------------------------------------------------------------------
def test_it_charges_in_the_cheap_hour_and_discharges_in_the_expensive_one():
    r"""The defining known answer for an economic MPC.

    Price is 0.1 for six hours then 1.0 for six hours, the base load is a constant
    5 kW, there is no cap and the battery is lossless. The cost-minimising plan is
    unambiguous: fill the battery while it is cheap, empty it while it is dear. The
    controller is stepped through a static state so the test reads the *plan*, not a
    simulation.
    """
    B = 2
    battery = _toy_battery(B)
    price = np.concatenate([np.full(6, 0.1), np.full(6, 1.0)])
    mpc = _mpc(battery, _FixedForecast(price, 5.0, B), B=B, horizon=12,
               soc_min=0.1, soc_max=0.9, terminal_soc="none")
    mpc.select_action(_obs(B, soc=0.5))
    plan = mpc._last_plan
    # The known answer is arithmetic, not a threshold. From soc = 0.5 with a band of
    # [0.1, 0.9] and a 10 kWh lossless store, there are 4.0 kWh of headroom; at 2.5 kW
    # that is 1.6 full-power steps, so the six cheap hours must deliver a total charge
    # command of 1.6 and the six dear hours a total discharge of at least as much.
    headroom_steps = (0.9 - 0.5) * 10.0 / 2.5
    cheap_charge = float(plan["c"][:6].sum(0).mean())
    cheap_discharge = float(plan["d"][:6].sum(0).mean())
    dear_discharge = float(plan["d"][6:].sum(0).mean())
    assert cheap_charge == pytest.approx(headroom_steps, abs=0.02), (
        f"charged {cheap_charge:.3f} full-power steps in the cheap block, "
        f"expected {headroom_steps:.3f}")
    assert cheap_discharge < 1e-3, "discharged while electricity was cheap"
    assert dear_discharge > headroom_steps - 0.02, (
        f"did not discharge in the dear block (d={dear_discharge:.3f})")


def test_a_flat_price_gives_no_arbitrage():
    """With one price for the whole horizon, cycling a *lossy* store strictly loses
    money, so the optimum is to leave it alone.

    The battery here has a round-trip efficiency below one on purpose. With a lossless
    store the linear programme is genuinely indifferent between cycling and not, every
    vertex is optimal, and the test would be measuring which vertex the simplex happens
    to return -- a property of the solver, not of the controller.
    """
    B = 2
    mpc = _mpc(_lossy_battery(B), _FixedForecast(np.full(12, 0.3), 5.0, B), B=B,
               horizon=12, soc_min=0.1, soc_max=0.9, terminal_soc="hold")
    mpc.select_action(_obs(B, soc=0.5))
    plan = mpc._last_plan
    assert float(plan["c"].sum() + plan["d"].sum()) < 1e-2, (
        "cycled the battery with no price signal to exploit")


def test_the_terminal_condition_stops_the_end_of_horizon_dump():
    """Without a terminal condition a finite-horizon economic MPC empties the store on
    the last step of every horizon. With ``terminal_soc='hold'`` it hands the battery
    back where it found it."""
    B = 2
    price = np.full(12, 0.5)
    battery = _toy_battery(B)
    free = _mpc(battery, _FixedForecast(price, 5.0, B), B=B, horizon=12,
                soc_min=0.1, soc_max=0.9, terminal_soc="none")
    held = _mpc(battery, _FixedForecast(price, 5.0, B), B=B, horizon=12,
                soc_min=0.1, soc_max=0.9, terminal_soc="hold")
    free.select_action(_obs(B, soc=0.8))
    held.select_action(_obs(B, soc=0.8))
    end_free = 0.8 + float((free._last_plan["c"] - free._last_plan["d"]).sum(0).mean()) * 0.25
    end_held = 0.8 + float((held._last_plan["c"] - held._last_plan["d"]).sum(0).mean()) * 0.25
    assert end_free < end_held + 1e-6
    assert float(held._last_plan["d"].sum()) <= float(free._last_plan["d"].sum()) + 1e-6


def test_it_keeps_the_state_of_charge_inside_the_band():
    """The band is a constraint of the optimisation, not a projection afterwards.

    Two things have to hold and both are checked. The planned trajectory must sit in
    the band, and the *exact plant* must be able to follow it -- a plan the battery
    cannot execute is not a constraint satisfied, it is a constraint deferred to
    whatever clips the action afterwards.
    """
    B = 2
    price = np.concatenate([np.full(6, 0.1), np.full(6, 1.0)])
    mpc = _mpc(_toy_battery(B), _FixedForecast(price, 5.0, B), B=B, horizon=12,
               soc_min=0.2, soc_max=0.8, terminal_soc="none")
    mpc.select_action(_obs(B, soc=0.5))
    planned = mpc.planned_soc()
    assert planned.max() <= 0.8 + 1e-6 and planned.min() >= 0.2 - 1e-6, (
        f"the planned trajectory leaves the band: [{planned.min():.4f}, {planned.max():.4f}]")
    worst = 0.0
    for k in range(12):
        soc = planned[k]
        reached = mpc.battery.next_soc(soc, mpc.battery.action_for_soc(soc, planned[k + 1]))
        worst = max(worst, float(np.abs(reached - planned[k + 1]).max()))
    assert worst < 1e-6, f"the exact plant cannot follow the plan; worst gap {worst:.2e}"


def test_reading_the_plan_as_a_raw_action_would_leave_the_band():
    """The reason ``planned_soc`` exists, as a measured number.

    Treating the linear programme's charge and discharge fractions as normalised
    actions -- which is what a controller that linearised the *action* map would do --
    walks outside the band, because the coefficients were evaluated at the previous
    pass's trajectory. This records the violation that the state-of-charge planning
    avoids, so the design choice is justified by a number rather than an argument.
    """
    B = 2
    price = np.concatenate([np.full(6, 0.1), np.full(6, 1.0)])
    mpc = _mpc(_toy_battery(B), _FixedForecast(price, 5.0, B), B=B, horizon=12,
               soc_min=0.2, soc_max=0.8, terminal_soc="none")
    mpc.select_action(_obs(B, soc=0.5))
    plan = mpc._last_plan
    soc, worst = np.full(B, 0.5), 0.0
    for k in range(12):
        soc = mpc.battery.next_soc(soc, np.clip(plan["c"][k] - plan["d"][k], -1, 1))
        worst = max(worst, float(max(soc.max() - 0.8, 0.2 - soc.min())))
    assert worst > 1e-3, (
        "the naive reading no longer violates the band, so the state-of-charge "
        f"planning may be unnecessary; measured violation {worst:.4f}")


def test_the_district_cap_binds_when_it_is_tight():
    """A cap below the base load is infeasible, and the controller must say so through
    its slack rather than fail. A cap the storage can respect must be respected."""
    B = 2
    price = np.concatenate([np.full(6, 0.1), np.full(6, 1.0)])
    loose = _mpc(_toy_battery(B), _FixedForecast(price, 5.0, B), B=B, horizon=12,
                 p_grid_max=1e6, soc_min=0.1, soc_max=0.9, terminal_soc="none")
    loose.select_action(_obs(B, soc=0.5))
    peak_loose = float((5.0 * B + (loose._last_plan["c"] - loose._last_plan["d"]).sum(1)
                        * 2.5).max())
    tight = _mpc(_toy_battery(B), _FixedForecast(price, 5.0, B), B=B, horizon=12,
                 p_grid_max=11.0, soc_min=0.1, soc_max=0.9, terminal_soc="none")
    tight.select_action(_obs(B, soc=0.5))
    peak_tight = float((5.0 * B + (tight._last_plan["c"] - tight._last_plan["d"]).sum(1)
                        * 2.5).max())
    assert peak_loose > 11.0, "the loose case has to breach the cap for this to mean anything"
    assert peak_tight <= 11.0 + 1e-3, f"district import {peak_tight:.3f} kW exceeds the 11 kW cap"
    assert tight.diagnostics["status"] == "optimal"


def test_an_infeasible_cap_reports_slack_rather_than_failing():
    """A base load above the cap cannot be fixed by storage. The controller still has to
    return an action -- a control loop that raises is a control loop that falls back to
    something unexamined."""
    B = 2
    mpc = _mpc(_toy_battery(B), _FixedForecast(np.full(12, 0.3), 20.0, B), B=B,
               horizon=12, p_grid_max=5.0, soc_min=0.1, soc_max=0.9)
    actions = mpc.select_action(_obs(B, soc=0.5))
    assert actions.shape == (B, 3) and np.isfinite(actions).all()
    assert mpc.diagnostics["grid_slack_kwh"] > 1.0, mpc.diagnostics


# ---------------------------------------------------------------------------
# 3. The forecasters: one causal, one not
# ---------------------------------------------------------------------------
def test_the_causal_forecaster_never_reads_the_future():
    """Its whole state is what it was told. Feeding it a prefix of a series and then the
    whole series must give the same prediction at the point where the prefix ended."""
    B = 2
    f = CausalForecaster(B)
    rng = np.random.default_rng(0)
    obs = [_obs(B, 0.5, price=0.3, net=float(rng.uniform(1, 9))) for _ in range(40)]
    for t in range(30):
        f.note_applied(obs[t], np.zeros((B, 3)))
        f.observe(obs[t + 1])
    early = f.predict(obs[30], 6)
    for t in range(30, 39):
        f.note_applied(obs[t], np.zeros((B, 3)))
        f.observe(obs[t + 1])
    assert f.steps_observed == 39, "every observed step is counted, not just the kept ones"
    # The prediction made at t=30 used only data up to t=30 and cannot change later.
    f2 = CausalForecaster(B)
    for t in range(30):
        f2.note_applied(obs[t], np.zeros((B, 3)))
        f2.observe(obs[t + 1])
    again = f2.predict(obs[30], 6)
    assert np.allclose(early.base_kw, again.base_kw)
    assert np.allclose(early.price, again.price)


def test_the_causal_forecaster_uses_the_price_leads_it_is_given():
    """Leads 0, 6 and 12 h are exact because the observation carries them."""
    B = 1
    f = CausalForecaster(B)
    o = _obs(B, 0.5)
    o[0][obs_index("electricity_pricing")] = 0.22
    o[0][obs_index("electricity_pricing_predicted_1")] = 0.54
    o[0][obs_index("electricity_pricing_predicted_2")] = 0.40
    out = f.predict(o, 13)
    assert out.price[0] == pytest.approx(0.22)
    assert out.price[PRICE_PRED_LEAD_H[0]] == pytest.approx(0.54)
    assert out.price[PRICE_PRED_LEAD_H[1]] == pytest.approx(0.40)
    # Between the anchors it holds the last known level, never interpolating a price the
    # tariff does not charge.
    assert set(np.round(out.price, 6)) <= {0.22, 0.54, 0.40}


def test_the_causal_forecaster_recovers_the_base_load_from_the_exact_plant():
    """base = net - (battery terminal energy) - (tank electricity), both exact."""
    env = _real_env()
    battery, tank = env.battery_model(), env.dhw_tank_model()
    B = env.num_buildings
    f = CausalForecaster(B, battery, tank, env.electrical_storage_action_index,
                         env.dhw_action_index)
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    for _ in range(5):
        a = rng.uniform(-0.6, 0.6, size=(B, 3)).astype(np.float32)
        a[:, 2] = 0.0
        soc = np.array([float(o[IDX_SOC]) for o in obs])
        socd = np.array([float(o[IDX_SOC_DHW]) for o in obs])
        dem = np.array([float(o[IDX_DHW_DEMAND]) for o in obs])
        f.note_applied(obs, a.astype(np.float64))
        nxt = env.step(a)[0]
        executed = env.executed_actions
        f.note_applied(obs, executed.astype(np.float64))
        f.observe(nxt)
        expected = (np.array([float(o[IDX_NET]) for o in nxt])
                    - battery.accepted_kwh(soc, executed[:, 1])
                    - tank.drawn_kwh(socd, executed[:, 0], dem))
        assert np.allclose(f._base[-1], expected, atol=1e-6)
        obs = nxt


def test_the_oracle_forecaster_returns_the_realised_series():
    T, B = 20, 3
    price = np.linspace(0.1, 0.5, T)
    base = np.arange(T * B, dtype=float).reshape(T, B)
    demand = np.zeros((T, B))
    oracle = OracleForecaster(price, base, demand)
    for t in range(5):
        out = oracle.predict(_obs(B, 0.5), 4)
        assert np.allclose(out.price, price[t:t + 4])
        assert np.allclose(out.base_kw, base[t:t + 4])
        oracle.observe(_obs(B, 0.5))
    # Past the end of the tape it holds the last value rather than reading out of range.
    oracle.t = T - 1
    assert np.allclose(oracle.predict(_obs(B, 0.5), 3).price, price[-1])


def test_resetting_after_the_oracle_tape_restores_the_environment():
    """The oracle's tape is recorded by rolling the evaluation environment itself.

    That is only sound if ``STEMSEnvironment.reset`` restarts the CityLearn episode from
    the window's first step, so that the observations the controller then sees are the
    ones it would have seen with no recording at all. This compares the first five
    observations before and after a full idle recording, bit for bit.
    """
    env = _real_env(span=(0, 71))
    before = [np.array(o, copy=True) for o in env.reset()[0]]
    oracle = record_idle_episode(env, 72)
    # A window of (0, 71) is 72 time steps and therefore 71 transitions; the tape has
    # one row per transition. `OracleForecaster.predict` clamps past the end, so a
    # controller that runs one step further reads the last row rather than failing.
    assert oracle.base.shape[0] == 71
    after = [np.array(o, copy=True) for o in env.reset()[0]]
    for b, (x, y) in enumerate(zip(before, after)):
        assert np.array_equal(x, y), f"building {b}'s reset observation moved"


# ---------------------------------------------------------------------------
# 4. End to end on the real environment
# ---------------------------------------------------------------------------
def test_the_two_mpc_arms_differ_only_in_the_forecaster():
    """Same class, same horizon, same bounds, same plant -- one forecaster each."""
    from experiments.controllers import ARMS, build_controller

    env = _real_env(span=(0, 47))
    causal = build_controller(ARMS["mpc"], env, STEMSConfig())
    oracle = build_controller(ARMS["mpc-oracle"], env, STEMSConfig())
    a, b = causal.base, oracle.base
    assert type(a) is type(b) is StorageMPC
    assert (a.H, a.soc_min, a.soc_max, a.terminal_soc) == (b.H, b.soc_min, b.soc_max,
                                                           b.terminal_soc)
    assert np.allclose(a.battery.capacity, b.battery.capacity)
    assert type(a.forecaster).__name__ == "CausalForecaster"
    assert type(b.forecaster).__name__ == "OracleForecaster"
    assert a.H == DEFAULT_HORIZON_H


def test_the_mpc_beats_doing_nothing_on_a_real_window():
    """The weakest possible claim, and the one a broken MPC fails: with a real
    three-level tariff and a real battery, shifting import out of the expensive hours
    must cost less than not shifting it.

    Short window, one seed: this is a smoke test of the controller, not a result. The
    comparison table is produced centrally.
    """
    set_seed(0)
    env = _real_env(span=(0, 119))
    battery, tank = env.battery_model(), env.dhw_tank_model()
    B = env.num_buildings

    def cost_of(controller):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(96):
            a = (np.zeros((B, env.action_dim), dtype=np.float32) if controller is None
                 else controller.select_action(obs))
            price = float(obs[0][IDX_PRICE])
            nxt, _r, term, trunc, _i = env.step(a)
            if controller is not None:
                controller.notify_executed(env.executed_actions)
                controller.observe(nxt)
            total += price * float(np.maximum(
                np.array([o[IDX_NET] for o in nxt]), 0.0).sum())
            obs = nxt
            if term or trunc:
                break
        return total

    idle_cost = cost_of(None)
    mpc = StorageMPC(B, env.action_dim, battery, tank,
                     CausalForecaster(B, battery, tank,
                                      env.electrical_storage_action_index,
                                      env.dhw_action_index),
                     elec_idx=env.electrical_storage_action_index,
                     dhw_idx=env.dhw_action_index, hvac_idx=env.hvac_action_index,
                     hvac_control=env.hvac_control, horizon=DEFAULT_HORIZON_H,
                     dhw_action_bound=env.dhw_info()["action_bound"],
                     p_grid_max=300.0, p_building_max=80.0)
    mpc_cost = cost_of(mpc)
    assert mpc_cost < idle_cost, (
        f"MPC cost {mpc_cost:.2f} did not beat doing nothing {idle_cost:.2f}")
