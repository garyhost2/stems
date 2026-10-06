"""The weekly disinfection cycle as the framework's fourth load.

Three things are under test: that the power-limited store makes unmet hot water a real
service failure (it cannot in CityLearn, which is the whole reason this model exists),
that the cycle is a deadline on the *same* object as the vehicle and the battery with
no new enforcement path, and that no unsourced parameter can be acquired by accident.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import CBFConfig
from stems.deadline import DeadlineStorageBarrier
from stems.flexibility import FlexibilityPortfolio, FlexibleLoad
from stems.legionella import (HOURS_PER_WEEK, LegionellaCycleBarrier, LegionellaSpec,
                              LegionellaStack, ShadowTank)
from stems.observations import obs_index

IDX_NET = obs_index("net_electricity_consumption")
OBS_WIDTH = 40

#: Parameters for the tests only. Every one of the design note's open questions is
#: given a value *here*, in a test, precisely so that no value is shipped in the
#: module: these are arithmetic fixtures, not recommended settings, and the spec's
#: `provenance` records them as such.
TEST_SPEC = dict(
    capacity_kwh=[4.0, 4.0], heat_source_kw=[1.5, 1.5],
    t_cold_c=15.0, t_rated_c=60.0, t_normal_c=50.0, t_legionella_c=60.0,
    cop_legionella=2.0, element_efficiency=1.0, t_heat_pump_max_c=55.0,
    loss_coefficient=[0.0, 0.0],
    provenance={"heat_source_kw": "test fixture, not a recommended value"})


def spec(**over):
    kw = dict(TEST_SPEC)
    kw.update(over)
    return LegionellaSpec(**kw)


def obs(n=2):
    out = []
    for _ in range(n):
        o = np.zeros(OBS_WIDTH, dtype=np.float32)
        o[IDX_NET] = 1.0
        out.append(o)
    return out


# ------------------------------------------------- nothing sourced by accident --

def test_every_open_question_without_provenance_is_reported_as_unsourced():
    s = spec()
    assert "heat_source_kw" not in s.unsourced, "provenance was recorded for this one"
    for name in ("t_cold_c", "t_rated_c", "t_normal_c", "t_legionella_c",
                 "cop_legionella", "element_efficiency", "t_heat_pump_max_c"):
        assert name in s.unsourced, f"{name} has no provenance and must be reported"
    assert s.summary()["unsourced_parameters"] == s.unsourced


def test_the_spec_has_no_default_for_any_open_question():
    """A default here would end up in a paper. There must not be one."""
    import dataclasses

    required = {f.name for f in dataclasses.fields(LegionellaSpec)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}
    for name in LegionellaSpec.OPEN_QUESTIONS:
        assert name in required, (
            f"{name} is one of the design note's open questions and has acquired a "
            "default; it must stay a required argument")


def test_the_only_defaulted_interval_is_the_conservative_end_of_the_stated_range():
    assert HOURS_PER_WEEK == 168.0
    assert spec().period_hours == HOURS_PER_WEEK


def test_a_disinfection_set_point_that_normal_operation_already_meets_is_refused():
    with pytest.raises(ValueError, match="satisfied by doing nothing"):
        spec(t_normal_c=60.0, t_legionella_c=60.0)


def test_a_rated_temperature_below_the_inlet_is_refused():
    with pytest.raises(ValueError, match="must exceed t_cold_c"):
        spec(t_cold_c=60.0, t_rated_c=50.0)


def test_a_store_with_no_source_is_refused():
    with pytest.raises(ValueError, match="never reach its required level"):
        spec(heat_source_kw=[0.0, 0.0])


# ----------------------------------------------- the levels the temperatures set --

def test_the_required_level_is_the_temperature_expressed_as_a_state_of_charge():
    s = spec()
    # (60 - 15) / (60 - 15) = 1.0 and (50 - 15) / (60 - 15) = 0.7778.
    assert s.soc_legionella == pytest.approx(1.0)
    assert s.soc_normal == pytest.approx(35.0 / 45.0, abs=1e-6)
    assert s.soc_heat_pump_max == pytest.approx(40.0 / 45.0, abs=1e-6)


def test_the_rate_is_the_source_power_over_the_capacity():
    s = spec()
    np.testing.assert_allclose(s.rate, 1.5 / 4.0)
    # From normal (0.7778) to disinfection (1.0) is 0.2222 of state at 0.375 per hour.
    np.testing.assert_allclose(s.summary()["hours_to_disinfect_from_normal"], 0.593,
                               atol=1e-3)


# -------------------------------------- the power limit makes a draw fail for real --

def test_a_draw_larger_than_the_store_and_the_source_goes_unmet():
    """In CityLearn it cannot: the heater is sized so every draw is served."""
    tank = ShadowTank(spec(), soc0=np.array([0.0, 0.0]))
    out = tank.step(tank.heat_for_action(np.ones(2)), np.array([3.0, 0.1]))
    # Empty store, 1.5 kWh of heat available this hour.
    np.testing.assert_allclose(out["unmet_kwh"], [1.5, 0.0], atol=1e-9)
    np.testing.assert_allclose(out["served_kwh"], [1.5, 0.1], atol=1e-9)
    r = tank.report()
    assert r["unmet_kwh"] == pytest.approx(1.5)
    assert r["unmet_draws"] == 1 and r["draws"] == 2
    assert r["unmet_share_of_draws"] == pytest.approx(0.5)


def test_a_full_store_serves_the_same_draw_without_failure():
    tank = ShadowTank(spec(), soc0=np.array([1.0, 1.0]))
    out = tank.step(np.zeros(2), np.array([3.0, 0.1]))
    np.testing.assert_allclose(out["unmet_kwh"], 0.0, atol=1e-9)
    np.testing.assert_allclose(tank.soc, [(4.0 - 3.0) / 4.0, (4.0 - 0.1) / 4.0])


def test_the_source_power_limit_binds_whatever_the_action_asks_for():
    tank = ShadowTank(spec(), soc0=np.array([0.0, 0.0]))
    out = tank.step(np.array([99.0, 99.0]), np.zeros(2))
    np.testing.assert_allclose(out["heat_in_kwh"], 1.5)


def test_heat_the_store_cannot_hold_is_reported_as_spill_not_lost():
    tank = ShadowTank(spec(), soc0=np.array([1.0, 1.0]))
    out = tank.step(tank.heat_for_action(np.ones(2)), np.zeros(2))
    np.testing.assert_allclose(out["spill_kwh"], 1.5)
    np.testing.assert_allclose(tank.soc, 1.0)


def test_the_energy_balance_closes():
    """heat in = draw served + spill + standing loss + change in storage."""
    s = spec(loss_coefficient=[0.01, 0.02])
    tank = ShadowTank(s, soc0=np.array([0.6, 0.3]))
    e0 = tank.E.copy()
    rng = np.random.default_rng(0)
    heat = served = spill = loss = 0.0
    for _ in range(100):
        before = tank.E.copy()
        out = tank.step(tank.heat_for_action(rng.uniform(0, 1, 2)),
                        rng.uniform(0, 1.2, 2))
        loss += float((before * s.loss_coefficient).sum())
        heat += float(out["heat_in_kwh"].sum())
        served += float(out["served_kwh"].sum())
        spill += float(out["spill_kwh"].sum())
    stored = float((tank.E - e0).sum())
    assert heat - (served + spill + loss + stored) == pytest.approx(0.0, abs=1e-9)


def test_the_electric_element_is_charged_for_the_lift_above_the_heat_pump_ceiling():
    s = spec()                      # ceiling at soc 0.8889, cop 2.0, element 1.0
    tank = ShadowTank(s, soc0=np.array([0.8889, 0.0]))
    # Building 0 starts at the ceiling: every kWh is the element's, at efficiency 1.
    # Building 1 is far below it: every kWh is the heat pump's, at a CoP of 2.
    e = tank.electricity_for_heat(np.array([1.0, 1.0]), np.array([0.8889, 0.0]))
    np.testing.assert_allclose(e, [1.0, 0.5], atol=1e-3)


def test_the_propane_variant_reaches_the_set_point_on_the_heat_pump_alone():
    s = spec(t_heat_pump_max_c=60.0)
    tank = ShadowTank(s, soc0=np.array([0.0, 0.0]))
    np.testing.assert_allclose(
        tank.electricity_for_heat(np.array([2.0, 2.0]), np.array([0.0, 0.0])), 1.0)


# ------------------------------------------- the cycle is the same object as the EV --

def test_the_cycle_barrier_is_a_deadline_storage_barrier():
    stack = LegionellaStack.build(spec(), action_index=0)
    assert isinstance(stack.barrier, DeadlineStorageBarrier)
    assert isinstance(stack.load, FlexibleLoad) and stack.load.kind == "legionella"


def test_the_barrier_is_inert_while_the_week_has_slack():
    stack = LegionellaStack.build(spec(), action_index=0)
    out = stack.load.project(np.zeros((2, 2), dtype=np.float32), obs())
    np.testing.assert_allclose(out[:, 0], 0.0)
    u = stack.barrier.urgency(obs())
    np.testing.assert_allclose(u["steps_to_deadline"], 168.0)
    assert (u["slack"] > 160).all(), "a week-long window is a large block of discretion"


def test_the_barrier_forces_the_cycle_when_the_window_is_about_to_close():
    stack = LegionellaStack.build(spec(), action_index=0)
    # Run the clock to one step before the deadline without ever disinfecting.
    stack.barrier.steps_since_cycle[:] = HOURS_PER_WEEK - 1.0
    u = stack.barrier.urgency(obs())
    assert (u["slack"] <= 0).all()
    out = stack.load.project(np.zeros((2, 2), dtype=np.float32), obs())
    assert (out[:, 0] > 0.5).all(), f"the cycle was not forced: {out[:, 0]}"


def test_reaching_the_level_resets_the_window_and_counts_a_cycle():
    stack = LegionellaStack.build(spec(), action_index=0)
    stack.barrier.steps_since_cycle[:] = 100.0
    stack.tank.E[:] = stack.spec.capacity_kwh          # soc 1.0 = soc_legionella
    out = stack.barrier.observe()
    assert out["completed"].all()
    np.testing.assert_allclose(stack.barrier.steps_since_cycle, 0.0)
    np.testing.assert_array_equal(stack.barrier.cycles_completed, [1, 1])
    np.testing.assert_array_equal(stack.barrier.missed_windows, [0, 0])


def test_a_window_that_closes_without_the_cycle_is_counted_as_missed():
    stack = LegionellaStack.build(spec(), action_index=0)
    stack.barrier.steps_since_cycle[:] = HOURS_PER_WEEK - 1.0
    stack.tank.E[:] = 0.5 * stack.spec.capacity_kwh
    out = stack.barrier.observe()
    assert out["expired"].all() and not out["completed"].any()
    np.testing.assert_array_equal(stack.barrier.missed_windows, [1, 1])
    np.testing.assert_allclose(stack.barrier.steps_since_cycle, 0.0)


def _run_week(stack, draw_hi=0.25, seed=0, steps=None):
    rng = np.random.default_rng(seed)
    for _ in range(int(steps if steps is not None else HOURS_PER_WEEK)):
        a = stack.load.project(np.zeros((2, 2), dtype=np.float32), obs())
        stack.observe(a[:, 0], rng.uniform(0.0, draw_hi, 2))
    return stack.report()


def test_the_barrier_drives_the_cycle_to_completion_over_a_week():
    """End to end: with the draw reserved, the deadline alone reaches 60 degC in time."""
    stack = LegionellaStack.build(
        spec(loss_coefficient=[0.005, 0.005], draw_margin_kwh=[0.25, 0.25]),
        action_index=0)
    rep = _run_week(stack)
    assert rep["cycle"]["missed_windows"] == [0, 0], rep["cycle"]
    assert rep["cycle"]["cycles_completed"] == [1, 1], (
        "exactly one cycle per window is the rule; more is wasted energy")


def test_without_a_reserve_the_concurrent_draw_can_make_the_cycle_miss():
    """The limitation, pinned rather than tuned away.

    ``steps_needed = ceil(gap / rate)`` in the base class assumes the source's whole
    output reaches the store. A hot-water tank is drawn from while it is being
    disinfected -- unlike a vehicle, which is not driven while plugged in -- so with a
    zero reserve the barrier starts forcing too late and arrives short. Same draw
    sequence as the test above; the only difference is the reserve.
    """
    stack = LegionellaStack.build(spec(loss_coefficient=[0.005, 0.005]), action_index=0)
    rep = _run_week(stack)
    assert rep["cycle"]["missed_windows"] == [1, 1], rep["cycle"]
    assert stack.spec.unsourced.count("draw_margin_kwh") == 0, (
        "a zero reserve is a declaration, not an unsourced number")


def test_an_undersized_source_misses_the_window_and_says_so_rather_than_pretending():
    """The constraint can be infeasible, and an infeasible constraint must be visible."""
    tiny = spec(heat_source_kw=[0.02, 0.02], loss_coefficient=[0.05, 0.05])
    stack = LegionellaStack.build(tiny, action_index=0)
    for _ in range(int(HOURS_PER_WEEK) + 1):
        a = stack.load.project(np.zeros((2, 2), dtype=np.float32), obs())
        stack.observe(a[:, 0], np.zeros(2))
    assert stack.barrier.report()["missed_windows"] == [1, 1]


# ---------------------------------------- it joins the portfolio with no new path --

def test_the_cycle_joins_the_same_portfolio_as_the_other_loads():
    from stems.battery import BatteryModel

    stack = LegionellaStack.build(spec(), action_index=0)
    battery = FlexibleLoad(
        name="battery", kind="battery", action_index=1,
        plant=BatteryModel.linear(np.full(2, 0.2)),
        soc_fn=lambda o: np.full(2, 0.5, dtype=np.float32), band=(0.1, 0.9))
    pf = FlexibilityPortfolio(loads=[battery, stack.load], cap_kw=10.0,
                              net_index=IDX_NET)
    assert [l.kind for l in pf.loads] == ["battery", "legionella"]
    assert pf.barriers == [stack.barrier]

    stack.barrier.steps_since_cycle[:] = HOURS_PER_WEEK - 1.0
    out = pf.project(np.zeros((2, 2), dtype=np.float32), obs())
    assert (out[:, 0] > 0.5).all(), "the cycle was not enforced through the portfolio"

    report = pf.feasibility(obs())
    assert report is not None and report["per_device"].keys() == {"legionella"}


def test_the_cycle_contends_for_the_shared_cap_like_any_other_load():
    stack = LegionellaStack.build(spec(), action_index=0)
    pf = FlexibilityPortfolio(loads=[stack.load], cap_kw=2.0,
                              coordination="proportional", net_index=IDX_NET)
    # Two buildings asking 1.5 kW each against a 2 kW cap with 1 kW of base load each.
    out = pf.allocate_under_cap(np.ones((2, 2), dtype=np.float32), obs())
    assert (out[:, 0] < 1.0).all(), "the cap did not bind on the disinfection load"


def test_the_load_carries_its_unsourced_parameters_into_its_description():
    stack = LegionellaStack.build(spec(), action_index=0)
    note = stack.load.describe()["notes"]
    assert "unsourced parameters" in note
    assert "cop_legionella" in note and "t_legionella_c" in note
