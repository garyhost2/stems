"""Constraints track step 2: battery and electric-vehicle capacity fade.

What these pin:

* the throughput term reproduces CityLearn's own ``Battery.degrade`` expression
  arithmetically, so the reward prices exactly what the plant does;
* ``kappa`` is read off the simulator's batteries and lies in CityLearn's documented
  range, rather than being a number this repository made up;
* the depth term reduces to the throughput term at ``p = 0`` (the default), charges
  each unit of throughput exactly once, and is not defeated by a jittery policy;
* fade is reported next to ``battery_equivalent_full_cycles``, which the repository
  previously counted and never priced;
* the constraint channel is an indicator of the same shape as the other three, which
  is what lets the same cost critics carry it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import DegradationConfig, STEMSConfig, constraint_channel_names
from stems.degradation import (CITYLEARN_KAPPA_RANGE, DegradationAccountant,
                               DegradationModel)

B = 4
KAPPA = 5e-5
CAP = 10.0


def model(mode="throughput", p=0.0, price=0.0, calendar=None, threshold=0.05):
    return DegradationModel(kappa=np.full(B, KAPPA), capacity_kwh=np.full(B, CAP),
                            mode=mode, dod_exponent=p, price_per_kwh=price,
                            calendar_loss_per_second=calendar,
                            reversal_threshold=threshold, source="test fixture")


def citylearn_degrade(kappa, rated_kwh, energy_kwh, current_kwh):
    """The literal expression in citylearn/energy_model.py::Battery.degrade."""
    return kappa * rated_kwh * abs(energy_kwh) / (2.0 * max(current_kwh, 1e-9))


# ---------------------------------------------------------------------------------
# the throughput term IS CityLearn's
# ---------------------------------------------------------------------------------

def test_the_throughput_term_reproduces_citylearns_own_expression():
    acc = DegradationAccountant(model())
    soc = np.full(B, 0.5)
    expected_capacity = np.full(B, CAP)
    for d in (0.2, -0.35, 0.1, -0.05):
        nxt = soc + d
        fade = acc.step(soc, nxt)
        energy = abs(d) * expected_capacity
        want = np.array([citylearn_degrade(KAPPA, CAP, e, c)
                         for e, c in zip(energy, expected_capacity)])
        assert np.allclose(fade, want, rtol=1e-12)
        expected_capacity = expected_capacity - want
        soc = nxt
    assert np.allclose(acc.capacity_kwh, expected_capacity)


def test_one_equivalent_full_cycle_costs_kappa_of_rated_capacity():
    """2 * Q_rated of throughput is one equivalent full cycle, so it costs kappa."""
    acc = DegradationAccountant(model())
    # two full sweeps of the state of charge = 2 * Q_rated through the cell
    for _ in range(1):
        acc.step(np.zeros(B), np.ones(B))
        acc.step(np.ones(B), np.zeros(B))
    assert np.allclose(acc.cumulative_loss_kwh / CAP, KAPPA, rtol=2e-4)


def test_kappa_comes_from_the_simulator_and_is_in_citylearns_documented_range():
    lo, hi = CITYLEARN_KAPPA_RANGE
    assert (lo, hi) == (1e-5, 1e-4)

    class _Env:
        def battery_degradation_info(self):
            return {"capacity_loss_coefficient": np.full(B, 6.1e-5),
                    "depth_of_discharge": np.full(B, 0.9),
                    "capacity": np.full(B, CAP),
                    "seconds_per_time_step": 3600.0,
                    "source": "citylearn.energy_model.Battery.capacity_loss_coefficient"}

    m = DegradationModel.from_environment(_Env(), DegradationConfig(mode="throughput"))
    assert np.all((m.kappa >= lo) & (m.kappa <= hi))
    assert "citylearn" in m.source


def test_mode_none_charges_nothing():
    acc = DegradationAccountant(model(mode="none"))
    assert np.allclose(acc.step(np.zeros(B), np.ones(B)), 0.0)
    assert np.allclose(acc.cumulative_loss_kwh, 0.0)


# ---------------------------------------------------------------------------------
# the depth-of-discharge term
# ---------------------------------------------------------------------------------

def test_p_zero_is_exactly_the_throughput_model():
    """The honest default asserts no depth effect, and the arithmetic agrees."""
    path = [0.5, 0.9, 0.3, 0.8, 0.1, 0.6]
    plain, dod = DegradationAccountant(model()), DegradationAccountant(model("throughput+dod", p=0.0))
    for a, b in zip(path, path[1:]):
        plain.step(np.full(B, a), np.full(B, b))
        dod.step(np.full(B, a), np.full(B, b))
    dod.flush()
    assert np.allclose(plain.cumulative_loss_kwh, dod.cumulative_loss_kwh, rtol=1e-9)


def test_every_unit_of_throughput_is_committed_exactly_once():
    """With p = 0 the depth scale is 1, so the committed total must equal the throughput
    total -- no half cycle may be dropped or counted twice."""
    rng = np.random.default_rng(0)
    soc = np.full(B, 0.5)
    plain, dod = DegradationAccountant(model()), DegradationAccountant(model("throughput+dod", p=0.0))
    for _ in range(400):
        nxt = np.clip(soc + rng.normal(0.0, 0.15, size=B), 0.05, 0.95)
        plain.step(soc, nxt)
        dod.step(soc, nxt)
        soc = nxt
    dod.flush()
    assert np.allclose(plain.cumulative_loss_kwh, dod.cumulative_loss_kwh, rtol=1e-9)


def test_a_deep_swing_costs_more_per_kwh_than_a_shallow_one():
    p = 1.0
    deep, shallow = (DegradationAccountant(model("throughput+dod", p=p)) for _ in range(2))
    # same total throughput: one 0.8-deep round trip vs eight 0.1-deep round trips
    for a, b in ((0.1, 0.9), (0.9, 0.1)):
        deep.step(np.full(B, a), np.full(B, b))
    deep.flush()
    soc = 0.4
    for _ in range(8):
        shallow.step(np.full(B, soc), np.full(B, soc + 0.1))
        shallow.step(np.full(B, soc + 0.1), np.full(B, soc))
    shallow.flush()
    # Throughput is measured against the *current* capacity, which shrinks as fade
    # accumulates, so the two totals agree only to the size of that fade.
    assert np.allclose(deep.throughput_kwh, shallow.throughput_kwh, rtol=1e-5)
    assert np.all(deep.cumulative_loss_kwh > shallow.cumulative_loss_kwh)


def test_the_range_gate_stops_jitter_defeating_the_depth_term():
    """Without the hysteresis filter, a policy that jitters closes micro cycles of
    depth ~1e-4 and, with p > 0, is charged almost nothing -- the depth term would
    reward jitter. The gate makes the jittery and the smooth traversal comparable."""
    p = 1.0
    gated = DegradationAccountant(model("throughput+dod", p=p, threshold=0.05))
    soc = 0.1
    # a slow climb from 0.1 to 0.9 with 0.01-amplitude jitter riding on it
    for _ in range(80):
        gated.step(np.full(B, soc), np.full(B, soc + 0.02))
        soc += 0.02
        gated.step(np.full(B, soc), np.full(B, soc - 0.01))
        soc -= 0.01
    gated.flush()
    assert int(gated.half_cycles.max()) <= 2, "the gate must not close a cycle per wiggle"
    assert float(gated.mean_half_cycle_depth.max()) > 0.3 or int(gated.half_cycles.max()) == 0


def test_a_negative_depth_exponent_is_refused():
    with pytest.raises(ValueError, match="non-negative"):
        model("throughput+dod", p=-0.5)


# ---------------------------------------------------------------------------------
# calendar, price, constraint
# ---------------------------------------------------------------------------------

def test_calendar_ageing_is_off_unless_a_rate_is_supplied():
    assert DegradationConfig().calendar is False
    assert DegradationConfig().calendar_loss_per_second is None
    idle = DegradationAccountant(model())
    assert np.allclose(idle.step(np.full(B, 0.5), np.full(B, 0.5)), 0.0)


def test_calendar_ageing_charges_per_second_of_elapsed_time():
    c_cal = 1e-9
    acc = DegradationAccountant(model(calendar=c_cal))
    fade = acc.step(np.full(B, 0.5), np.full(B, 0.5))   # no throughput at all
    assert np.allclose(fade, c_cal * CAP * 3600.0)


def test_the_price_defaults_to_zero_so_fade_is_reported_not_priced():
    assert DegradationConfig().price_eur_per_kwh == 0.0
    acc = DegradationAccountant(model())
    assert np.allclose(acc.reward_penalty(acc.step(np.zeros(B), np.ones(B))), 0.0)


def test_pricing_fade_gives_a_currency_amount_additive_with_the_bill():
    acc = DegradationAccountant(model(price=150.0))
    fade = acc.step(np.zeros(B), np.ones(B))
    assert np.allclose(acc.reward_penalty(fade), 150.0 * fade)
    assert acc.kpis()["battery_degradation_cost"] == pytest.approx(
        150.0 * float(acc.cumulative_loss_kwh.sum()))


def test_the_constraint_channel_is_a_per_step_indicator_like_the_other_three():
    acc = DegradationAccountant(model(), limit_kwh_per_episode=1e-4, episode_steps=100)
    assert acc.budget_per_step_kwh == pytest.approx(1e-6)
    small = acc.constraint_cost(np.full(B, 1e-9))
    big = acc.constraint_cost(np.full(B, 1e-3))
    assert small.shape == (B,) and set(np.unique(small)) <= {0.0, 1.0}
    assert np.allclose(small, 0.0) and np.allclose(big, 1.0)


def test_no_budget_means_reported_and_not_enforced():
    acc = DegradationAccountant(model())
    assert acc.budget_per_step_kwh is None
    assert np.allclose(acc.constraint_cost(np.full(B, 1e3)), 0.0)


def test_the_cost_channel_appears_only_when_the_budget_does():
    cfg = STEMSConfig()
    assert constraint_channel_names(cfg) == ("soc_band", "building_power_cap",
                                             "district_import_cap")
    cfg.degradation.mode = "throughput"
    assert len(constraint_channel_names(cfg)) == 3, "priced but unconstrained"
    cfg.degradation.limit_kwh_per_episode = 0.01
    assert constraint_channel_names(cfg)[-1] == "battery_degradation"


# ---------------------------------------------------------------------------------
# reported next to the count it replaces
# ---------------------------------------------------------------------------------

def test_fade_is_reported_in_kwh_alongside_the_cycle_count():
    from stems.metrics import MetricsCalculator
    from stems.observations import obs_index

    idx_soc = obs_index("electrical_storage_soc")
    n_obs = 32
    calc = MetricsCalculator(B, degradation_model=model(price=150.0))
    rng = np.random.default_rng(3)
    soc = np.full(B, 0.5)
    for _ in range(50):
        nxt = np.clip(soc + rng.normal(0.0, 0.2, size=B), 0.1, 0.9)
        mk = lambda v: [np.eye(1, n_obs, idx_soc, dtype=np.float32).ravel() * v[i]
                        for i in range(B)]
        calc.add_step(mk(soc), np.zeros((B, 3), dtype=np.float32), mk(nxt))
        soc = nxt
    k = calc.compute_all()
    assert "battery_equivalent_full_cycles" in k
    for name in ("battery_capacity_loss_kwh", "battery_capacity_loss_fraction",
                 "battery_throughput_kwh", "battery_degradation_cost",
                 "battery_capacity_loss_kwh_worst_building"):
        assert name in k, f"{name} must sit next to the cycle count"
    assert k["battery_capacity_loss_kwh"] > 0.0
    assert k["battery_degradation_cost"] == pytest.approx(
        150.0 * k["battery_capacity_loss_kwh"])


def test_the_kpis_are_absent_when_no_model_is_wired():
    from stems.metrics import MetricsCalculator

    calc = MetricsCalculator(B)
    assert calc._accountant is None
