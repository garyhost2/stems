"""The framework object: one load type, two constraint faces, one shared cap.

These tests check the three claims ``docs/FRAMEWORK.md`` makes that could silently stop
being true: that the four loads are one object, that promoting the shield's inner loop
into ``FlexibilityPortfolio`` did not change what the shield does, and that the exact
plant inverse is available behind the deadline barrier but off by default.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel, TankModel
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.flexibility import LOAD_KINDS, FlexibilityPortfolio, FlexibleLoad
from stems.observations import obs_index
from stems.protocols import (Environment, EVProvider, PlantModel, PlantProvider,
                             missing_capabilities)

IDX_SOC = obs_index("electrical_storage_soc")
IDX_NET = obs_index("net_electricity_consumption")
OBS_WIDTH = 40


def obs(net=0.0, soc=0.5, n=3):
    out = []
    for _ in range(n):
        o = np.zeros(OBS_WIDTH, dtype=np.float32)
        o[IDX_SOC], o[IDX_NET] = soc, net
        out.append(o)
    return out


def toy_barrier(name="ev", action_index=0, rate=0.2, steps=1, required=0.8, soc=0.3,
                n=3, power_kw=None):
    """A deadline load with a constant requirement, for arithmetic that is checkable by hand."""
    b = DeadlineStorageBarrier(
        rate=np.full(n, rate), action_bound=np.ones(n), action_index=action_index,
        capacity=np.full(n, 10.0), efficiency=np.ones(n),
        requirement_fn=lambda _o: DeadlineRequirement(
            soc=np.full(n, required), steps_to_deadline=np.full(n, steps),
            active=np.ones(n, dtype=bool)),
        soc_fn=lambda _o: np.full(n, soc), soc_cap=1.0, name=name)
    if power_kw is not None:
        b._p_charge = np.full(n, power_kw)
    return b


# ------------------------------------------------------- the object itself --

def test_a_load_with_neither_face_is_refused():
    with pytest.raises(ValueError, match="constrains nothing"):
        FlexibleLoad(name="ghost", kind="battery", action_index=0)


def test_a_barrier_projecting_a_different_column_is_refused():
    with pytest.raises(ValueError, match="action column"):
        FlexibleLoad(name="ev", kind="ev", action_index=2,
                     barrier=toy_barrier(action_index=0))


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown load kind"):
        FlexibleLoad(name="x", kind="microwave", action_index=0,
                     barrier=toy_barrier())
    assert set(LOAD_KINDS) == {"battery", "dhw", "legionella", "ev", "envelope"}


def test_the_battery_is_the_band_only_member_of_the_family():
    """The point of the abstraction: a house battery is a degenerate flexible load."""
    model = BatteryModel.linear(np.full(3, 0.2))
    load = FlexibleLoad(name="battery", kind="battery", action_index=0, plant=model,
                        soc_fn=lambda o: np.array([float(x[IDX_SOC]) for x in o]),
                        band=(0.1, 0.9))
    assert load.has_band and not load.has_deadline
    # soc 0.85, rate 0.2 per step: only +0.25 of the action keeps it under 0.9.
    out = load.project(np.ones((3, 2), dtype=np.float32), obs(soc=0.85))
    np.testing.assert_allclose(out[:, 0], 0.25, atol=1e-6)


def test_the_deadline_face_forces_only_when_the_slack_runs_out():
    load = FlexibleLoad(name="ev", kind="ev", action_index=0,
                        barrier=toy_barrier(steps=10, soc=0.3, required=0.8, rate=0.2))
    idle = load.project(np.zeros((3, 2), dtype=np.float32), obs())
    np.testing.assert_allclose(idle[:, 0], 0.0)

    # gap 0.5 at rate 0.2 needs ceil(2.5) = 3 steps; with 3 left the slack is zero.
    urgent = FlexibleLoad(name="ev", kind="ev", action_index=0,
                          barrier=toy_barrier(steps=3, soc=0.3, required=0.8, rate=0.2))
    out = urgent.project(np.zeros((3, 2), dtype=np.float32), obs())
    np.testing.assert_allclose(out[:, 0], 1.0, atol=1e-6)


# ----------------------------------------------------------- the exact seam --

def test_the_deadline_barrier_is_linear_by_default_and_says_so():
    b = toy_barrier()
    assert b.plant is None and b.exact_projection is False


def test_the_exact_plant_inverse_can_be_switched_in_behind_the_same_barrier():
    """``plant=`` makes the forced action the plant's own inverse, not ``gap/rate``.

    The property that matters is not that the exact command is larger or smaller -- the
    sign depends on whether ``rate`` over- or under-states the plant, and every shipped
    barrier deliberately understates it. The property is that the exact command *lands
    on the required level* and the linear one does not.
    """
    n = 3
    flat = np.array([[0.0, 1.0], [1.0, 1.0]])
    curve = np.array([[0.0, 0.3, 1.0], [0.8, 0.9, 0.9]])
    plant = BatteryModel([10.0] * n, [5.0] * n, [0.0] * n, [curve] * n, [flat] * n)
    kw = dict(rate=0.4, steps=1, required=0.6, soc=0.3, n=n)
    linear = toy_barrier(**kw)
    exact = toy_barrier(**kw)
    exact.plant = plant
    assert exact.exact_projection is True

    a_lin = linear.project(np.zeros((n, 2), dtype=np.float32), obs())[:, 0]
    a_exact = exact.project(np.zeros((n, 2), dtype=np.float32), obs())[:, 0]
    # The linear map asks for gap/rate = 0.3/0.4 = 0.75 of the bound.
    np.testing.assert_allclose(a_lin, 0.75, atol=1e-6)
    # The exact inverse asks for the command that really lands on soc 0.6.
    reached_exact = plant.next_soc(np.full(n, 0.3), a_exact.astype(np.float64))
    np.testing.assert_allclose(reached_exact, 0.6, atol=1e-6)
    # The linear command does not: on this plant it overshoots, because `rate` = 0.4
    # understates a device that moves 0.5 * sqrt(eta) of state per full step.
    reached_linear = plant.next_soc(np.full(n, 0.3), a_lin.astype(np.float64))
    assert np.abs(reached_linear - 0.6).min() > 1e-3, (
        "the linear map happened to land on the target, so this case does not "
        f"separate the two paths: reached {reached_linear}")
    assert not np.allclose(a_exact, a_lin)


def test_switching_the_plant_in_is_off_in_every_shipped_barrier():
    """A refactor must not turn the seam on by accident."""
    from stems.ev import EVChargerSpec, EVObsLayout, EVReadinessBarrier

    layout = EVObsLayout(connected_state=0, departure_time=1,
                         required_soc_departure=2, soc=3, battery_capacity=4)
    spec = EVChargerSpec(max_charging_power_kw=np.full(3, 7.0), efficiency=np.full(3, 0.9),
                         action_bound=np.ones(3), action_index=0)
    assert EVReadinessBarrier(layout, spec).exact_projection is False


# ------------------------------------------------------------- the portfolio --

def test_the_portfolio_refuses_two_loads_with_the_same_name():
    with pytest.raises(ValueError, match="duplicate load names"):
        FlexibilityPortfolio(loads=[
            FlexibleLoad("ev", "ev", 0, barrier=toy_barrier(action_index=0)),
            FlexibleLoad("ev", "ev", 1, barrier=toy_barrier(action_index=1))])


def test_the_portfolio_runs_every_deadline_in_turn():
    pf = FlexibilityPortfolio(loads=[
        FlexibleLoad("dhw", "dhw", 0, barrier=toy_barrier("dhw", 0, steps=1)),
        FlexibleLoad("ev", "ev", 1, barrier=toy_barrier("ev", 1, steps=1))])
    out = pf.project_deadlines(np.zeros((3, 2), dtype=np.float32), obs())
    assert (out[:, 0] > 0).all() and (out[:, 1] > 0).all()


def test_the_cap_cuts_back_only_the_loads_that_draw_power():
    """A load with no ``power_kw`` is skipped: its action is not a power fraction."""
    pf = FlexibilityPortfolio(
        loads=[FlexibleLoad("ev", "ev", 0, barrier=toy_barrier("ev", 0, power_kw=10.0),
                            power_kw=np.full(3, 10.0)),
               FlexibleLoad("comfort", "envelope", 1,
                            barrier=toy_barrier("comfort", 1))],
        cap_kw=15.0, coordination="proportional", net_index=IDX_NET)
    actions = np.ones((3, 2), dtype=np.float32)
    out = pf.allocate_under_cap(actions.copy(), obs(net=0.0))
    # 3 x 10 kW requested against a 15 kW cap: scaled by 0.5.
    np.testing.assert_allclose(out[:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(out[:, 1], 1.0, atol=1e-6)


def test_the_portfolio_reports_joint_infeasibility_the_devices_cannot_see_alone():
    pf = FlexibilityPortfolio(
        loads=[FlexibleLoad("ev", "ev", 0, barrier=toy_barrier("ev", 0, soc=0.0,
                                                               required=1.0, steps=1)),
               FlexibleLoad("dhw", "dhw", 1, barrier=toy_barrier("dhw", 1, soc=0.0,
                                                                 required=1.0, steps=1))],
        cap_kw=10.0, net_index=IDX_NET)
    report = pf.feasibility(obs())
    # Six devices owing 10 kWh each against one hour of a 10 kW cap.
    assert report["energy_required_kwh"] == pytest.approx(60.0)
    assert report["energy_available_kwh"] == pytest.approx(10.0)
    assert not report["feasible"]
    assert report["shortfall_kwh"] == pytest.approx(50.0)
    assert "priority" in report


def test_an_empty_portfolio_reports_nothing_rather_than_a_false_all_clear():
    assert FlexibilityPortfolio(loads=[]).feasibility(obs()) is None


# ------------------------------------------------- the shield still agrees --

def _shield(coordination, barriers):
    return CBFShield(CBFConfig(P_grid_max=15.0), num_buildings=3,
                     soc_rate=np.full(3, 0.2), nominal_power=np.full(3, 5.0),
                     elec_idx=1, safety_cfg=SafetyConfig(anticipatory=False,
                                                         robust_margins=False),
                     deadline_barriers=barriers, coordination=coordination)


@pytest.mark.parametrize("coordination", ["independent", "proportional", "edf"])
def test_the_shield_delegates_to_a_portfolio_holding_its_own_barriers(coordination):
    barriers = [toy_barrier("ev", 0, power_kw=10.0), toy_barrier("dhw", 2)]
    shield = _shield(coordination, barriers)
    pf = shield.portfolio
    assert [l.name for l in pf.loads] == ["ev", "dhw"]
    assert pf.barriers == barriers
    assert pf.coordination == coordination
    np.testing.assert_allclose(pf.loads[0].power_kw, 10.0)
    assert pf.loads[1].power_kw is None


def test_appending_a_barrier_after_construction_still_reaches_the_shield():
    """The portfolio is rebuilt per call because ``deadline_barriers`` is public."""
    shield = _shield("independent", [])
    assert len(shield.portfolio) == 0
    shield.deadline_barriers.append(toy_barrier("ev", 0, steps=1))
    out = shield.project(np.zeros((3, 3), dtype=np.float32), obs())
    assert (out[:, 0] > 0).all(), "a barrier added after construction was not enforced"


def test_the_shield_feasibility_report_is_the_portfolio_feasibility():
    shield = _shield("independent", [toy_barrier("ev", 0, soc=0.0, required=1.0, steps=1)])
    a = shield.feasibility_report(obs())
    b = shield.portfolio.feasibility(obs(), cap_kw=shield.grid_cap())
    assert a["energy_required_kwh"] == b["energy_required_kwh"]
    assert a["shortfall_kwh"] == b["shortfall_kwh"]


# ------------------------------------------------------------- the protocols --

def test_the_plant_models_satisfy_the_plant_protocol():
    assert isinstance(BatteryModel.linear(np.array([0.1])), PlantModel)
    assert isinstance(TankModel([1.0], [1.0], [1.0], [1.0], [0.0]), PlantModel)


def test_missing_capabilities_names_the_attribute_not_just_the_verdict():
    class Half:
        num_buildings = 1
        obs_dim = 2
        action_dim = 1

    missing = missing_capabilities(Half())
    assert "Environment" in missing
    assert "step" in missing["Environment"] and "reset" in missing["Environment"]
    assert not isinstance(Half(), Environment)
