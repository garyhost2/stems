"""Adversarial tests for every guarantee in ``docs/FRAMEWORK_GUARANTEES.md``.

Each test attacks one promise and asserts *what the framework does*, not merely that it
survives. Tests whose name begins with ``test_documented_gap_`` pin a case where a
constraint is broken and the caller is **not** told: they encode the current, wrong,
behaviour on purpose, so that closing the gap fails them loudly and forces
``docs/FRAMEWORK_GUARANTEES.md`` S5 to be updated in the same commit. Each one names the
section it pins and what the assertion should become once it is fixed.

Everything here goes through the public surface -- ``CBFShield.project``,
``FleetShield.project``, ``DeadlineStorageBarrier.project``, ``schedule``,
``schedule_executable``, ``apply_dead_band`` -- so the suite survives a refactor of the
barrier internals.

Notation follows ``docs/FRAMEWORK_GUARANTEES.md`` S0: x_b state of charge, a_b action,
r_b one-step rate, P_cap district cap, m(t) forecast margin, p_min_b charger minimum
power, s*_b required departure state of charge.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel, TankModel
from stems.cbf import CBFShield
from stems.config import CBFConfig
from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.environment import STEMSEnvironment
from stems.fleet import (RULES, BaseLoadForecaster, EVFleetModel, FleetShield, FleetState,
                         HouseStorage, allocate, apply_dead_band, schedule,
                         schedule_executable)
from stems.observations import obs_index

CFG = CBFConfig()
IDX_SOC = obs_index("electrical_storage_soc")
IDX_LOAD = obs_index("non_shiftable_load")
IDX_NET = obs_index("net_electricity_consumption")
IDX_T_OUT = obs_index("outdoor_dry_bulb_temperature")
IDX_BATT_SOC = 19
OBS_LEN = 30
FLAT = np.array([[0.0, 1.0], [1.0, 1.0]])
DAY = {"episode_time_steps": [(0, 23)]}

LAYOUT = {"connected_state": 0, "soc": 1, "required_soc_departure": 2, "departure_time": 3}
MYOPIC = tuple(r for r in RULES if r != "lp")


def ev_obs(connected, soc, target, countdown, load_kw=0.0, batt_soc=0.5):
    o = np.zeros(OBS_LEN)
    o[0], o[1], o[2], o[3] = connected, soc, target, countdown
    o[IDX_LOAD], o[IDX_BATT_SOC] = load_kw, batt_soc
    return o


def ev_fleet(B, p_max=10.0, p_min=1.0, capacity=50.0, eta=1.0):
    battery = BatteryModel([capacity] * B, [p_max] * B, [0.0] * B, [FLAT] * B, [FLAT] * B)
    return EVFleetModel([True] * B, [p_max] * B, [p_min] * B, [eta] * B, battery)


@pytest.fixture(scope="module")
def plant():
    """The simulator's own eight batteries, resolved through CityLearn's autosizing."""
    return STEMSEnvironment(seed=0, heat_pump=True, env_kwargs=DAY).battery_model()


# ------------------------------------------------- S1: the state-of-charge band ---

def _band_sweep(shield, plant, socs, actions):
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
            nxt = plant.next_soc(np.full(B, x),
                                 shield.project(cmd, states)[:, 1].astype(np.float64))
            below = max(below, float(np.max(CFG.SOC_min - nxt)))
            above = max(above, float(np.max(nxt - CFG.SOC_max)))
            breaches += int(np.sum((nxt < CFG.SOC_min - 1e-9) | (nxt > CFG.SOC_max + 1e-9)))
            cases += B
    return breaches, cases, below, above


def test_soc_band_is_hard_from_every_in_band_state_with_the_exact_inverse(plant):
    """S1: with the plant's own model the band is never left. 0/5832 here, 0/52488 in the
    full sweep in docs/FRAMEWORK_GUARANTEES.md S1."""
    socs = np.linspace(CFG.SOC_min, CFG.SOC_max, 27)
    actions = np.linspace(-1.0, 1.0, 27)
    shield = CBFShield(num_buildings=plant.B, battery_model=plant, elec_idx=1)
    breaches, cases, below, above = _band_sweep(shield, plant, socs, actions)
    assert cases == 27 * 27 * plant.B
    assert breaches == 0, f"{breaches}/{cases} breaches, {below=}, {above=}"
    assert below == 0.0 and above == 0.0


def test_a_mis_calibrated_plant_model_breaks_the_soc_band_wide_open(plant):
    """S1: the audit's uniform r_b = 0.1 surrogate understates the true rate by 1.8x-5.3x
    on these devices and drives the state of charge to the floor."""
    rate = plant.nominal_power * plant.dt / plant.capacity
    assert rate.min() > 0.1, "this attack assumes 0.1 understates every building's rate"
    socs = np.linspace(CFG.SOC_min, CFG.SOC_max, 27)
    actions = np.linspace(-1.0, 1.0, 27)
    shield = CBFShield(num_buildings=plant.B, soc_rate=np.full(plant.B, 0.1), elec_idx=1)
    breaches, cases, below, above = _band_sweep(shield, plant, socs, actions)
    assert breaches > 0.05 * cases, f"expected a wide breach, got {breaches}/{cases}"
    assert below == pytest.approx(CFG.SOC_min, abs=1e-6), (
        "the floor should be driven all the way to state of charge 0")
    assert above > 0.05


def test_an_out_of_band_start_recovers_monotonically_rather_than_being_clamped(plant):
    """S1: the band is not promised from outside it; what is promised is recovery."""
    for x0, command in ((0.0, 1.0), (1.0, -1.0)):
        shield = CBFShield(num_buildings=plant.B, battery_model=plant, elec_idx=1)
        x = np.full(plant.B, x0)
        outside = []
        for _ in range(3):
            states = [np.zeros(OBS_LEN) for _ in range(plant.B)]
            for j, s in enumerate(states):
                s[IDX_SOC] = x[j]
            cmd = np.zeros((plant.B, 3), dtype=np.float32)
            cmd[:, 1] = command
            x = plant.next_soc(x, shield.project(cmd, states)[:, 1].astype(np.float64))
            outside.append(int(np.sum((x < CFG.SOC_min - 1e-9) | (x > CFG.SOC_max + 1e-9))))
        assert outside == sorted(outside, reverse=True), f"not monotone from {x0}: {outside}"
        assert outside[-1] == 0, f"did not recover within 3 steps from {x0}: {outside}"


# ------------------------------------------------ S3: the joint deadline set ---

def test_an_infeasible_deadline_set_reports_the_exact_missing_energy():
    """S3: 3 vehicles each individually reachable, jointly not. The LP charges at the full
    cap anyway and reports the shortfall rather than refusing or hiding it."""
    model = ev_fleet(3)
    owed_kwh = 3 * (0.8 - 0.4) * 50.0
    previous = -1.0
    for cap_kw, expect_feasible in ((30.0, True), (17.5, True), (15.0, False),
                                    (10.0, False), (5.0, False)):
        state = FleetState([True] * 3, [0.4] * 3, [0.8] * 3, [2, 3, 4],
                           np.zeros((8, 3)), cap_kw)
        sol = schedule(model, state, np.array([10.0, 10.0, 10.0]))
        assert sol["feasible"] is expect_feasible, (cap_kw, sol["total_shortfall_kwh"])
        assert sol["now_kw"].sum() == pytest.approx(min(cap_kw, 30.0), abs=1e-5), (
            "an infeasible set must still charge at the full cap")
        if not expect_feasible:
            # the deadline binding at 2 steps carries the shortfall: owed minus delivered
            assert sol["total_shortfall_kwh"] > previous, "shortfall must grow as the cap falls"
            assert sol["total_shortfall_kwh"] <= owed_kwh + 1e-6
            assert 0.0 < sol["worst_shortfall_share"] <= 1.0
            previous = sol["total_shortfall_kwh"]
    tight = schedule(model, FleetState([True] * 3, [0.4] * 3, [0.8] * 3, [2, 3, 4],
                                       np.zeros((8, 3)), 5.0), np.array([10.0] * 3))
    assert tight["total_shortfall_kwh"] == pytest.approx(40.0, abs=1e-3)


def test_the_shortfall_is_spread_over_the_fleet_not_dumped_on_one_vehicle():
    """S3: three identical vehicles with identical deadlines share the loss equally."""
    model = ev_fleet(3)
    sol = schedule(model, FleetState([True] * 3, [0.4] * 3, [0.8] * 3, [2, 2, 2],
                                     np.zeros((4, 3)), 15.0), np.array([10.0] * 3))
    assert not sol["feasible"]
    np.testing.assert_allclose(sol["shortfall_soc"], [0.2, 0.2, 0.2], atol=1e-4)
    assert sol["worst_shortfall_share"] == pytest.approx(0.5, abs=1e-4)


def test_a_vehicle_that_departs_the_step_after_it_arrives_is_reported_not_hidden():
    """S3: one step, a 0.7 SOC gap. Nothing can fix it; the LP must say so and still charge."""
    model = ev_fleet(1)
    shield = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1))
    action = shield.project(np.zeros((1, 1), dtype=np.float32), [ev_obs(1, 0.20, 0.90, 0)])
    assert shield.last["feasible"] is False
    assert shield.last["shortfall_kwh"] == pytest.approx(25.0, abs=1e-3)
    assert float(model.draw_kw(np.array([0.20]), action[:, 0].astype(np.float64))[0]) == \
        pytest.approx(10.0, abs=1e-6), "it must still charge at full power"


def test_a_vehicle_already_at_its_required_state_of_charge_is_left_alone():
    """S3: the dual of the previous test -- no forced charge, no false infeasibility."""
    model = ev_fleet(1)
    for soc in (0.90, 0.95):
        idle = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1))
        action = idle.project(np.zeros((1, 1), dtype=np.float32), [ev_obs(1, soc, 0.90, 0)])
        assert idle.last["feasible"] is True
        assert idle.last.get("shortfall_kwh", 0.0) == 0.0
        assert float(idle.last["forced_kw"][0]) == 0.0, "nothing is owed, nothing is forced"
        assert float(action[0, 0]) == 0.0
        # the shield is a projection, not a policy: a request it has no reason to refuse
        # is passed through untouched.
        asked = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1))
        asked.project(np.ones((1, 1), dtype=np.float32), [ev_obs(1, soc, 0.90, 0)])
        assert float(asked.last["cut_kw"][0]) == pytest.approx(0.0, abs=1e-9)
        assert float(asked.last["forced_kw"][0]) == pytest.approx(0.0, abs=1e-9)


def test_a_charger_whose_minimum_exceeds_the_cap_headroom_refuses_and_reports():
    """S3 / S5.2: the LP would rather draw nothing than break the cap, and says why."""
    for p_min, expect_draw in ((1.0, 2.0), (2.0, 2.0), (4.0, 0.0), (10.0, 0.0)):
        model = ev_fleet(1, p_min=p_min)
        shield = FleetShield(model, LAYOUT, 0, 2.0, "lp", BaseLoadForecaster(1))
        action = shield.project(np.ones((1, 1), dtype=np.float32), [ev_obs(1, 0.4, 0.9, 3)])
        drawn = float(model.draw_kw(np.array([0.4]), action[:, 0].astype(np.float64))[0])
        assert drawn == pytest.approx(expect_draw, abs=1e-4), p_min
        assert drawn <= 2.0 + 1e-6, "the cap must hold even when the charger cannot comply"
        assert shield.last["feasible"] is False
        assert shield.last["shortfall_kwh"] > 0.0


# ------------------------------------ S5.0: where infeasibility reaches the caller ---

def test_the_dead_band_second_pass_still_reports_the_shortfall_it_creates():
    """S5 / infeasibility surfacing: switching a charger off to respect the dead band
    makes the shortfall *larger*, and schedule_executable re-solves and reports the
    larger value rather than carrying the first pass's optimistic one."""
    for p_min, off in ((1.0, False), (3.0, True), (6.0, True), (9.0, True)):
        model = ev_fleet(1, p_min=p_min)
        state = FleetState([True], [0.40], [0.90], [1], np.zeros((1, 1)), 2.0)
        first = schedule(model, state, np.array([10.0]))
        final = schedule_executable(model, state, np.array([10.0]))
        assert first["feasible"] is False and final["feasible"] is False
        assert first["total_shortfall_kwh"] == pytest.approx(23.0, abs=1e-3)
        if off:
            assert final["now_kw"][0] == 0.0
            assert final["total_shortfall_kwh"] == pytest.approx(25.0, abs=1e-3), (
                "the second pass must report the shortfall of the plan it actually chose")
            assert final["total_shortfall_kwh"] > first["total_shortfall_kwh"]
        else:
            assert final["now_kw"][0] == pytest.approx(2.0, abs=1e-4)
            assert final["total_shortfall_kwh"] == pytest.approx(first["total_shortfall_kwh"],
                                                                 abs=1e-3)


def test_schedule_executable_converges_so_the_dead_band_has_nothing_left_to_round():
    """S5.2: over 120 random fleets the two-pass dead-band handling always converged, so
    apply_dead_band is a no-op on the 'lp' rule and the round-up of S5.2 cannot be
    reached there."""
    rng = np.random.default_rng(0)
    solved = survivors = 0
    for _ in range(120):
        n = int(rng.integers(2, 6))
        p_max = rng.uniform(4.0, 11.0, n)
        p_min = p_max * rng.uniform(0.1, 0.6, n)
        capacity = rng.uniform(30.0, 70.0, n)
        model = EVFleetModel([True] * n, p_max, p_min, np.full(n, 0.95),
                             BatteryModel(capacity, p_max, np.zeros(n), [FLAT] * n, [FLAT] * n))
        state = FleetState([True] * n, rng.uniform(0.2, 0.6, n), rng.uniform(0.6, 0.95, n),
                           rng.integers(1, 6, n), np.zeros((6, n)),
                           float(rng.uniform(3.0, 25.0)))
        try:
            sol = schedule_executable(model, state, rng.uniform(0.0, p_max, n))
        except RuntimeError:
            continue
        solved += 1
        survivors += int(((sol["now_kw"] > 1e-6) & (sol["now_kw"] < model.p_min - 1e-9)).any())
    assert solved >= 100, f"only {solved}/120 fleets solved; the sample is too thin"
    assert survivors == 0, f"{survivors}/{solved} fleets left an allocation inside the dead band"


def test_an_lp_failure_in_the_dead_band_probe_does_not_silently_lose_the_shortfall():
    """S5: forcing a charger on at p_min can make the programme infeasible for the solver.
    schedule_executable catches that and falls back to the off branch -- whose own
    shortfall is reported, so nothing is lost."""
    model = ev_fleet(1, p_min=6.0)
    state = FleetState([True], [0.40], [0.90], [1], np.zeros((1, 1)), 2.0)
    with pytest.raises(RuntimeError):
        schedule(model, state, np.array([10.0]), now_min=np.array([6.0]),
                 now_max=np.array([10.0]))
    sol = schedule_executable(model, state, np.array([10.0]))
    assert sol["feasible"] is False
    assert sol["now_kw"][0] == 0.0
    assert sol["total_shortfall_kwh"] == pytest.approx(25.0, abs=1e-3)


# ---------------------------------------------------------- S5: documented gaps ---

def test_documented_gap_myopic_rules_never_report_infeasibility():
    """S5.1. Two vehicles owe 50 kWh in one step under a 5 kW cap; only 'lp' says so.

    experiments/ev_coupling.py:135 counts `last.get("feasible") is False`, so
    infeasible_hour_rate is structurally 0.000 for the six myopic rules.
    WHEN FIXED: every rule should write `feasible` and `shortfall_kwh`, and this test
    should assert `is False` for all of RULES and drop the ABSENT branch.
    """
    model = ev_fleet(2)
    obs = [ev_obs(1, 0.4, 0.9, 1), ev_obs(1, 0.4, 0.9, 1)]
    for rule in MYOPIC:
        shield = FleetShield(model, LAYOUT, 0, 5.0, rule, BaseLoadForecaster(2))
        shield.project(np.zeros((2, 1), dtype=np.float32), obs)
        assert "feasible" not in shield.last, rule
        assert "shortfall_kwh" not in shield.last, rule
        assert (shield.last.get("feasible") is False) is False, (
            f"{rule}: ev_coupling would score this hour as feasible")
    lp = FleetShield(model, LAYOUT, 0, 5.0, "lp", BaseLoadForecaster(2))
    lp.project(np.zeros((2, 1), dtype=np.float32), obs)
    assert lp.last["feasible"] is False
    assert lp.last["shortfall_kwh"] == pytest.approx(40.0, abs=1e-3)


def test_documented_gap_the_dead_band_rounds_an_urgent_charger_past_the_cap():
    """S5.2. allocate respects the cap; apply_dead_band rounds an urgent vehicle up to
    p_min and doubles the import, with no report channel of any kind.

    WHEN FIXED: apply_dead_band should either return the breach alongside the array or
    refuse the round-up, and this test should assert banded.sum() <= state.cap.
    """
    model = ev_fleet(2, p_min=4.0)
    state = FleetState([True, True], [0.4, 0.4], [0.9, 0.5], [1, 9], np.zeros((9, 2)), 2.0)
    granted = allocate(model, state, np.array([2.0, 0.0]), "proportional")
    assert granted.sum() == pytest.approx(2.0, abs=1e-9)
    assert granted.sum() <= state.cap + 1e-9
    banded = apply_dead_band(model, state, granted)
    assert banded.sum() == pytest.approx(4.0, abs=1e-9)
    assert banded.sum() > state.cap, "the round-up is the documented gap"
    assert banded.sum() - state.cap == pytest.approx(2.0, abs=1e-9)


def test_documented_gap_the_shield_predicts_an_over_cap_import_and_calls_it_binding():
    """S5.2, through the public FleetShield.project. The five coordinating myopic rules
    end at twice the cap and report only `binding: True`; 'lp' refuses instead.

    WHEN FIXED: `last` should carry a breach flag whenever predicted_import_kw exceeds
    self.cap, and this test should assert that flag rather than its absence.
    """
    model = ev_fleet(2, p_min=4.0)
    obs = [ev_obs(1, 0.4, 0.9, 0), ev_obs(1, 0.4, 0.5, 9)]
    for rule in ("static", "proportional", "edf", "llf", "sllf"):
        shield = FleetShield(model, LAYOUT, 0, 2.0, rule, BaseLoadForecaster(2))
        shield.project(np.zeros((2, 1), dtype=np.float32), obs)
        assert shield.last["predicted_import_kw"] == pytest.approx(4.0, abs=1e-3), rule
        assert shield.last["predicted_import_kw"] > shield.cap
        assert shield.last["binding"] is True
        assert not any(k for k in shield.last if "breach" in k or "exceed" in k), rule
    lp = FleetShield(model, LAYOUT, 0, 2.0, "lp", BaseLoadForecaster(2))
    lp.project(np.zeros((2, 1), dtype=np.float32), obs)
    assert lp.last["predicted_import_kw"] == 0.0 and lp.last["feasible"] is False


def test_documented_gap_a_cap_below_the_inflexible_load_still_reports_feasible():
    """S5.3. 8 kW of inflexible load, caps of 6/3/1 kW. Everything sheddable is shed and
    the report asserts `feasible: True` while predicting a 7 kW breach.

    WHEN FIXED: `feasible` should be False (or a separate `cap_feasible` key added) when
    predicted_import_kw exceeds the cap, and this test should assert that.
    """
    house_model = BatteryModel([20.0], [5.0], [0.0], [FLAT], [FLAT])
    for cap_kw, expect_pred, expect_shed in ((10.0, 10.0, 3.0), (6.0, 8.0, 5.0),
                                             (3.0, 8.0, 5.0), (1.0, 8.0, 5.0)):
        house = HouseStorage(house_model, battery_action=1, battery_soc=IDX_BATT_SOC,
                             soc_lo=0.1, soc_hi=0.9, tank_action=2)
        shield = FleetShield(ev_fleet(1), LAYOUT, 0, cap_kw, "lp",
                             BaseLoadForecaster(1), house=house)
        shield.project(np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
                       [ev_obs(0, 0, 0, 0, load_kw=8.0)])
        assert shield.last["predicted_import_kw"] == pytest.approx(expect_pred, abs=1e-3)
        assert shield.last["storage_shed_kw"] == pytest.approx(expect_shed, abs=1e-3)
        assert shield.last["feasible"] is True, "the documented false positive"
        assert shield.last["binding"] is False
    assert expect_pred - 1.0 == pytest.approx(7.0), "worst residual in the sweep, kW"


def test_documented_gap_the_heat_pump_guard_sheds_everything_and_reports_nothing():
    """S5.3. With the base load alone over the grid cap the guard zeroes the heat pump and
    the import is still 160 kW against 95 kW. CBFShield.project has no report channel.

    WHEN FIXED: CBFShield should expose a `last` report with the predicted import and a
    breach flag, and this test should assert it.
    """
    from stems.thermal import CoPModel

    ones = np.ones(4, dtype=np.float32)
    cop = CoPModel(efficiency_heat=0.3 * ones, target_heat=45.0 * ones,
                   efficiency_cool=0.3 * ones, target_cool=7.0 * ones,
                   nominal_power_heat=10.0 * ones, nominal_power_cool=10.0 * ones)
    shield = CBFShield(config=CBFConfig(P_grid_max=100.0, P_building_max=80.0),
                       num_buildings=4, soc_rate=np.full(4, 0.2), elec_idx=1,
                       cop_model=cop, hvac_idx=2, enforce_soc=False)
    states = []
    for _ in range(4):
        o = np.zeros(OBS_LEN)
        o[IDX_NET], o[IDX_T_OUT] = 40.0, 5.0
        states.append(o)
    cmd = np.zeros((4, 3), dtype=np.float32)
    cmd[:, 2] = 1.0
    safe = shield.project(cmd, states)
    assert np.allclose(safe[:, 2], 0.0), "the guard must shed everything it controls"
    imported = float(np.maximum(np.full(4, 40.0), 0.0).sum())
    assert imported - shield.grid_cap() == pytest.approx(65.0, abs=1e-6)
    assert not hasattr(shield, "last"), "the documented absence of a report channel"


def _deadline_store(B, action_index=0, p_charge_kw=6.0, rate=0.12):
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


def _deadline_states(B):
    states = []
    for _ in range(B):
        o = np.zeros(OBS_LEN)
        o[IDX_SOC], o[IDX_NET] = 0.5, 0.0
        o[24], o[25], o[26], o[27] = 0.2, 0.9, 1.0, 1.0
        states.append(o)
    return states


def test_documented_gap_the_cbf_path_never_calls_its_own_feasibility_report():
    """S5.4. feasibility_report computes the whole answer and no live path calls it: the
    only callers in the repository are tests. Under the live default coordination the
    district cap is not applied to deadline-barrier actions at all.

    WHEN FIXED: FleetShield-style reporting should be wired into CBFShield.project, and
    this test should assert the report is produced by project() itself.
    """
    B, p_charge = 3, 6.0
    states = _deadline_states(B)
    independent = CBFShield(num_buildings=B, soc_rate=np.full(B, 0.12), elec_idx=1,
                            deadline_barriers=[_deadline_store(B)],
                            config=CBFConfig(P_grid_max=4.0), enforce_soc=False)
    assert independent.coordination == "independent", "the live controllers' default"
    got = independent.project(np.zeros((B, 3), dtype=np.float32), states)[:, 0]
    assert np.allclose(got, 1.0)
    assert float((got * p_charge).sum()) == pytest.approx(18.0, abs=1e-5)
    assert float((got * p_charge).sum()) > independent.grid_cap()

    report = independent.feasibility_report(states)
    assert report["feasible"] is False
    assert report["shortfall_kwh"] == pytest.approx(101.2, abs=1e-1)
    assert report["priority"]["missed"], "it even knows which store misses by how much"


def test_documented_gap_shared_allocation_overrides_the_deadline_floor_silently():
    """S5.4. 'proportional' and 'edf' enforce the cap by scaling the barrier's *forced*
    floor down, i.e. the hard deadline constraint loses, with no record.

    WHEN FIXED: the override should be reported (which constraint was relaxed, by how
    much), and this test should assert that record.
    """
    B, p_charge, cap = 3, 6.0, 4.0
    states = _deadline_states(B)
    demanded = _deadline_store(B).project(np.zeros((B, 3), dtype=np.float32), states)[:, 0]
    assert np.allclose(demanded, 1.0), "the barrier demands a full charge"
    for coordination, expected in (("proportional", [0.211, 0.211, 0.211]),
                                   ("edf", [0.633, 0.0, 0.0])):
        shield = CBFShield(num_buildings=B, soc_rate=np.full(B, 0.12), elec_idx=1,
                           deadline_barriers=[_deadline_store(B)],
                           coordination=coordination,
                           config=CBFConfig(P_grid_max=cap), enforce_soc=False)
        got = shield.project(np.zeros((B, 3), dtype=np.float32), states)[:, 0]
        np.testing.assert_allclose(got, expected, atol=2e-3)
        assert float((got * p_charge).sum()) == pytest.approx(shield.grid_cap(), abs=1e-3)
        assert np.all(got < demanded - 1e-3), "the forced floor was overridden"


def test_documented_gap_reserve_hours_clips_an_unreachable_target_and_reports_success():
    """S5.5. The idle-loss-compensated target is clipped at 1.0; the unavoidable departure
    shortfall that creates is never reported.

    WHEN FIXED: the clip should surface a shortfall (the vehicle cannot make its departure
    state of charge whatever the controller does), and this test should assert it.
    """
    loss = 0.05
    model = EVFleetModel([True], [10.0], [1.0], [1.0],
                         BatteryModel([50.0], [10.0], [loss], [FLAT], [FLAT]))
    keep = 1.0 - loss
    seen = []
    for reserve, clipped in ((0, False), (2, False), (4, True), (6, True), (12, True)):
        shield = FleetShield(model, LAYOUT, 0, 100.0, "lp", BaseLoadForecaster(1),
                             reserve_hours=reserve)
        obs = [ev_obs(1, 0.30, 0.90, 23)]
        state = shield.state(obs)
        idle = max(24 - int(state.slots[0]), 0)
        needed = 0.90 / keep ** idle
        assert (needed > 1.0) is clipped, reserve
        assert float(state.target[0]) == pytest.approx(min(needed, 1.0), abs=1e-4)
        arrives = float(state.target[0]) * keep ** idle
        seen.append(round(max(0.90 - arrives, 0.0), 4))
        if clipped:
            shield.project(np.zeros((1, 1), dtype=np.float32), obs)
            assert shield.last["feasible"] is True, (
                "the documented false positive: it meets the clipped target and declares "
                "success while the vehicle cannot reach its real departure requirement")
    assert seen == [0.0, 0.0, 0.0855, 0.1649, 0.3596], seen


def test_documented_gap_a_mis_calibrated_tank_breaches_the_cap_while_reporting_it_met():
    """S5.6. An optimistic heater efficiency under-predicts the house draw; the shield
    reports the import as exactly the cap and the plant draws 0.706 kW more.

    WHEN FIXED: nothing in the shield can detect this -- the fix is upstream (the tank
    model must come from TankModel.from_citylearn). This test pins the magnitude so the
    calibration's value is on record.
    """
    cap_kw, true_eta, demand = 12.0, 0.85, np.array([0.0])
    battery = BatteryModel([20.0], [5.0], [0.0], [FLAT], [FLAT])
    plant_tank = TankModel([10.0], [8.0], [true_eta], [1.0], [0.0])
    residuals = {}
    for eta in (0.60, 0.85, 1.00):
        house = HouseStorage(battery, battery_action=1, battery_soc=IDX_BATT_SOC,
                             soc_lo=0.1, soc_hi=0.9,
                             tank=TankModel([10.0], [8.0], [eta], [1.0], [0.0]),
                             tank_action=2)
        shield = FleetShield(ev_fleet(1), LAYOUT, 0, cap_kw, "lp",
                             BaseLoadForecaster(1), house=house)
        obs = [ev_obs(0, 0, 0, 0, load_kw=8.0)]
        action = shield.project(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), obs)
        soc_tank = np.array([float(obs[0][obs_index("dhw_storage_soc")])])
        realised = 8.0 + float(plant_tank.drawn_kwh(soc_tank, action[:, 2].astype(np.float64),
                                                    demand)[0]) / plant_tank.dt
        assert shield.last["predicted_import_kw"] == pytest.approx(cap_kw, abs=1e-3)
        residuals[eta] = round(max(realised - cap_kw, 0.0), 4)
    assert residuals[0.85] == 0.0, "an exact model holds the cap exactly"
    assert residuals[0.60] == 0.0, "a pessimistic model is safe and wasteful"
    assert residuals[1.00] == pytest.approx(0.706, abs=1e-3), (
        "an optimistic model breaches by the modelling error, reporting the cap as met")


# ---------------------------------------------------------- S2: the cap residual ---

class _ZeroMargin(BaseLoadForecaster):
    @property
    def margin(self) -> float:
        return 0.0


def _cap_rollout(T=336, B=4, cap_kw=16.0, seed=0, margin=True, spread_shock=1.0,
                 shock_at=240):
    """A compact version of guarantee_residuals.cap_rollout: the realised base load
    carries a component the observation does not reveal, which is the only source of
    forecast error, as in CityLearn."""
    rng = np.random.default_rng(seed)
    model = ev_fleet(B, p_max=7.0, p_min=1.4, capacity=60.0, eta=0.95)
    forecaster = (BaseLoadForecaster if margin else _ZeroMargin)(B)
    shield = FleetShield(model, LAYOUT, 0, cap_kw, "lp", forecaster)
    soc = np.full(B, 0.4)
    connected = np.zeros(B, dtype=bool)
    countdown = np.zeros(B, dtype=int)
    eps = np.zeros(B)
    hidden = np.zeros(B)
    mean_innovation = 2.0 * 0.35
    residual, margins = [], []
    for t in range(T):
        hour = t % 24
        eps = 0.6 * eps + rng.normal(0.0, 0.2, B)
        observed = np.maximum(0.7 + 0.5 * np.sin(2 * np.pi * (hour - 7) / 24) ** 2 + eps, 0.0)
        scale = spread_shock if t >= shock_at else 1.0
        hidden = np.maximum(0.7 * hidden + scale * rng.gamma(2.0, 0.35, B)
                            - (scale - 1.0) * mean_innovation, 0.0)
        base = observed + hidden
        if hour == 18:
            connected[:] = True
            countdown[:] = 13
            soc[:] = rng.uniform(0.3, 0.45, B)
        obs = [ev_obs(float(connected[i]), soc[i], 0.7, float(max(countdown[i], 0)),
                      observed[i]) for i in range(B)]
        action = shield.project(np.zeros((B, 1), dtype=np.float32), obs)
        draw = model.draw_kw(soc, action[:, 0].astype(np.float64))
        soc = np.where(connected, model.next_soc(soc, action[:, 0].astype(np.float64)), soc)
        residual.append(max(float(np.maximum(base + draw, 0.0).sum()) - cap_kw, 0.0))
        margins.append(float(shield.last.get("margin_kw", 0.0)))
        nxt = []
        for i in range(B):
            o = np.zeros(OBS_LEN)
            o[IDX_NET] = base[i] + draw[i]
            nxt.append(o)
        shield.observe(nxt, draw)
        if hour == 7:
            connected[:] = False
        countdown = np.maximum(countdown - 1, 0)
    R = np.array(residual[T // 2:])
    return {"rate": float((R > 1e-6).mean()), "kwh": float(R.sum()),
            "max_kw": float(R.max()), "margin_kw": float(np.mean(margins[T // 2:]))}


def test_the_forecast_margin_shrinks_the_cap_residual_but_does_not_remove_it():
    """S2: the cap is soft up to forecast error, measured in the saturated regime
    (P_cap = 16 kW, the fleet permanently pressed against the cap). The margin is worth a
    large factor and is not a guarantee; both halves of that sentence are asserted here."""
    cap_kw = 16.0
    on = [_cap_rollout(seed=s, margin=True) for s in range(3)]
    off = [_cap_rollout(seed=s, margin=False) for s in range(3)]
    rate_on = float(np.mean([r["rate"] for r in on]))
    rate_off = float(np.mean([r["rate"] for r in off]))
    assert rate_off > 2.0 * rate_on, (rate_on, rate_off)
    assert float(np.mean([r["kwh"] for r in off])) > 2.0 * float(np.mean([r["kwh"] for r in on]))
    assert all(r["rate"] > 0.0 for r in on), (
        "a 95th-percentile margin cannot make the cap hard; it is a quantile, not a bound")
    assert max(r["max_kw"] for r in on) < 0.25 * cap_kw, (
        "with a stationary error process the residual must stay well under the cap")
    assert all(r["margin_kw"] > 0.5 for r in on)
    assert all(r["margin_kw"] == 0.0 for r in off)


def test_a_forecast_error_beyond_its_calibrated_margin_degrades_the_cap_monotonically():
    """S2: multiply the innovation spread the margin was calibrated on. The residual must
    grow with the shock and the margin must chase it, neither silently nor catastrophically
    at small shocks."""
    out = {}
    for spread in (1.0, 4.0, 8.0):
        runs = [_cap_rollout(seed=s, margin=True, spread_shock=spread) for s in range(3)]
        out[spread] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    assert out[1.0]["max_kw"] < out[4.0]["max_kw"] < out[8.0]["max_kw"], out
    assert out[1.0]["kwh"] < out[4.0]["kwh"] < out[8.0]["kwh"], out
    assert out[1.0]["margin_kw"] < out[8.0]["margin_kw"], (
        "the margin must adapt upward -- it lags, it does not ignore")
    assert out[8.0]["max_kw"] > 2.0 * out[1.0]["max_kw"], (
        "an error beyond the calibrated margin must show up as a real breach")
