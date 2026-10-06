"""Constraints track step 1: thermal comfort as a deadline-storage constraint.

What these pin:

* the 1R1C identification recovers parameters it was generated from, and refuses a
  non-physical fit rather than clipping it;
* the barrier is the *same object* as the hot-water and electric-vehicle barriers --
  a ``DeadlineStorageBarrier`` -- so the framework claim is structural, not a figure
  of speech;
* the projected action is the smallest one that reaches the band under the identified
  model, in both directions, and the two directions can never bind together;
* the barrier is OFF by default and refuses, with a reason, every configuration in
  which it could not actually bound a temperature.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.comfort import (SPAN_K, T_HIGH_C, T_ZERO_C, RCThermalModel,
                           ThermalComfortBarrier, build_comfort_barriers)
from stems.config import ComfortConfig, STEMSConfig
from stems.deadline import DeadlineStorageBarrier
from stems.observations import obs_index, obs_indices

IDX_T_IN, IDX_T_OUT = obs_indices("indoor_dry_bulb_temperature",
                                  "outdoor_dry_bulb_temperature")
IDX_T_COOL = obs_index("indoor_dry_bulb_temperature_cooling_set_point")
IDX_OCC = obs_index("occupant_count")
IDX_T_HEAT = obs_index("indoor_dry_bulb_temperature_heating_set_point")
OBS_DIM = IDX_T_HEAT + 2

B = 3
DT = 3600.0


def rc(capacitance=2.0e7, ua=200.0, gain=0.0, b=B):
    return RCThermalModel(np.full(b, capacitance), np.full(b, ua), np.full(b, gain),
                          dt_s=DT, provenance="test fixture")


def obs(t_in, t_out=5.0, t_heat=20.0, t_cool=26.0, occupied=1.0, b=B):
    out = []
    for i in range(b):
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[IDX_T_IN] = np.atleast_1d(t_in)[i % np.size(t_in)]
        o[IDX_T_OUT] = t_out
        o[IDX_T_HEAT] = t_heat
        o[IDX_T_COOL] = t_cool
        o[IDX_OCC] = occupied
        out.append(o)
    return out


# ---------------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------------

def test_identification_recovers_the_parameters_it_was_generated_from():
    truth = rc(capacitance=1.5e7, ua=180.0, gain=400.0)
    n = 500
    rng = np.random.default_rng(0)
    t_out = 5.0 + 5.0 * np.sin(np.arange(n) / 12.0)
    phi = rng.uniform(-4000.0, 4000.0, size=(n, B))
    t_in = np.zeros((n, B))
    t_in[0] = 21.0
    t_out_mat = np.repeat(t_out[:, None], B, axis=1)
    for t in range(n - 1):
        t_in[t + 1] = truth.next_temperature_c(t_in[t], t_out_mat[t], phi[t])
    model, fit = RCThermalModel.identify(t_in, t_out_mat, phi, dt_s=DT)
    assert np.allclose(model.C, truth.C, rtol=1e-6)
    assert np.allclose(model.UA, truth.UA, rtol=1e-6)
    assert np.allclose(model.Phi_g, truth.Phi_g, rtol=1e-4)
    assert float(np.min(fit.r2)) > 0.999
    assert float(np.max(fit.rmse_k)) < 1e-8


def test_identification_refuses_a_non_physical_fit_instead_of_clipping_it():
    """A dataset whose temperature falls when heat is added has no 1R1C envelope."""
    n = 200
    rng = np.random.default_rng(1)
    phi = rng.uniform(0.0, 5000.0, size=(n, B))
    t_out = np.full((n, B), 5.0)
    t_in = 21.0 - 1e-3 * np.cumsum(phi, axis=0) / 100.0
    with pytest.raises(ValueError, match="not physical"):
        RCThermalModel.identify(t_in, t_out, phi, dt_s=DT)


def test_the_model_refuses_parameters_that_are_not_a_building():
    with pytest.raises(ValueError, match="thermal capacitance"):
        RCThermalModel(np.full(B, 1.0), np.full(B, 200.0), np.zeros(B))
    with pytest.raises(ValueError, match="UA"):
        RCThermalModel(np.full(B, 2.0e7), np.full(B, 0.01), np.zeros(B))


def test_the_time_constant_is_c_over_ua_in_hours():
    model = rc(capacitance=1.8e7, ua=250.0)
    assert np.allclose(model.tau_h, 1.8e7 / 250.0 / 3600.0)


# ---------------------------------------------------------------------------------
# the barrier is the framework's object, not a lookalike
# ---------------------------------------------------------------------------------

def heating_barrier(model=None, phi_w=6000.0, cop=3.0, theta=2.0):
    return ThermalComfortBarrier(model or rc(), +1, action_index=2,
                                 thermal_power_w=np.full(B, phi_w),
                                 cop=np.full(B, cop), action_bound=np.ones(B),
                                 tolerance_k=theta, name="comfort_heating")


def cooling_barrier(model=None, phi_w=6000.0, cop=3.0, theta=2.0):
    return ThermalComfortBarrier(model or rc(), -1, action_index=2,
                                 thermal_power_w=np.full(B, phi_w),
                                 cop=np.full(B, cop), action_bound=np.ones(B),
                                 tolerance_k=theta, name="comfort_cooling")


def test_the_comfort_barrier_is_a_deadline_storage_barrier():
    assert isinstance(heating_barrier(), DeadlineStorageBarrier)


def test_the_store_is_the_zones_heat_and_the_deficit_is_in_kwh():
    """gap * capacity is the sensible heat missing from the zone, C * dT / 3.6e6."""
    model = rc(capacitance=2.0e7, ua=1.0)   # negligible envelope loss
    bar = heating_barrier(model)
    o = obs(t_in=np.full(B, 15.0), t_out=15.0, t_heat=20.0)
    u = bar.urgency(o)
    expected_k = (20.0 - 2.0) - 15.0          # floor is T_heat - theta
    assert np.allclose(u["gap"] * SPAN_K, expected_k, atol=1e-3)
    thermal_kwh = 2.0e7 * expected_k / 3.6e6
    # energy_still_required_kwh divides by the efficiency, here the coefficient of
    # performance, so the number is electrical.
    assert np.allclose(bar.energy_still_required_kwh(o), thermal_kwh / 3.0, rtol=1e-3)


def test_the_deadline_is_now_and_an_empty_house_is_inactive():
    bar = heating_barrier()
    o = obs(t_in=np.full(B, 15.0))
    assert np.all(bar.urgency(o)["steps_to_deadline"] == 0.0)
    assert np.all(bar.urgency(o)["active"])
    assert not np.any(bar.urgency(obs(t_in=np.full(B, 15.0), occupied=0.0))["active"])


# ---------------------------------------------------------------------------------
# the projection
# ---------------------------------------------------------------------------------

def test_heating_projects_to_the_smallest_action_that_reaches_the_floor():
    model = rc(capacitance=2.0e7, ua=200.0)
    phi = 6000.0
    bar = heating_barrier(model, phi_w=phi)
    o = obs(t_in=np.full(B, 17.5), t_out=10.0, t_heat=20.0)
    a = bar.project(np.zeros((B, 3), dtype=np.float32), o)[:, 2]

    drift = model.drift_k(np.full(B, 17.5), np.full(B, 10.0))
    needed_k = (20.0 - 2.0) - (17.5 + drift)       # floor minus the free-running state
    expected = needed_k / model.temperature_gain_k(np.full(B, phi))
    assert np.all(expected < 1.0), "fixture must leave the band reachable"
    assert np.allclose(a, expected, atol=1e-5)
    # and it is the *minimum*: the achieved temperature lands on the floor, not above.
    reached = model.next_temperature_c(np.full(B, 17.5), np.full(B, 10.0), a * phi)
    assert np.allclose(reached, 18.0, atol=1e-4)


def test_the_barrier_only_raises_a_heating_action_never_lowers_it():
    bar = heating_barrier()
    o = obs(t_in=np.full(B, 17.0), t_out=2.0)
    a0 = np.full((B, 3), 0.9, dtype=np.float32)
    assert np.all(bar.project(a0, o)[:, 2] >= 0.9 - 1e-6)


def test_a_warm_house_is_not_touched_by_the_heating_barrier():
    bar = heating_barrier()
    o = obs(t_in=np.full(B, 24.0), t_out=20.0, t_heat=20.0)
    a0 = np.full((B, 3), -0.4, dtype=np.float32)
    assert np.allclose(bar.project(a0, o)[:, 2], -0.4)


def test_cooling_projects_downward_to_the_ceiling():
    model = rc(capacitance=2.0e7, ua=200.0)
    phi = 6000.0
    bar = cooling_barrier(model, phi_w=phi)
    o = obs(t_in=np.full(B, 28.5), t_out=30.0, t_cool=26.0)
    a = bar.project(np.zeros((B, 3), dtype=np.float32), o)[:, 2]
    assert np.all(a < 0.0)
    reached = model.next_temperature_c(np.full(B, 28.5), np.full(B, 30.0), a * phi)
    assert np.allclose(reached, 28.0, atol=1e-4)   # ceiling = T_cool + theta


def test_the_two_sides_can_never_bind_at_the_same_step():
    """Both binding would need T_heat - theta > T_cool + theta, i.e. an inverted band."""
    heat, cool = heating_barrier(), cooling_barrier()
    rng = np.random.default_rng(2)
    for _ in range(200):
        t_in = rng.uniform(-5.0, 45.0, size=B)
        t_heat = rng.uniform(15.0, 22.0)
        o = obs(t_in=t_in, t_out=rng.uniform(-15.0, 40.0), t_heat=t_heat,
                t_cool=t_heat + rng.uniform(0.0, 8.0))
        h = heat.urgency(o)
        c = cool.urgency(o)
        both = (h["gap"] > 0) & (c["gap"] > 0)
        assert not np.any(both)


def test_the_action_is_clipped_to_the_bound_when_the_band_is_unreachable():
    bar = heating_barrier(rc(capacitance=5.0e7, ua=600.0), phi_w=1000.0)
    o = obs(t_in=np.full(B, -5.0), t_out=-20.0, t_heat=22.0)
    a = bar.project(np.zeros((B, 3), dtype=np.float32), o)[:, 2]
    assert np.allclose(a, 1.0)


def test_the_rate_is_the_per_step_gain_at_full_action():
    model = rc(capacitance=2.0e7)
    bar = heating_barrier(model, phi_w=6000.0)
    assert np.allclose(bar.rate, (3600.0 / 2.0e7) * 6000.0 / SPAN_K)


# ---------------------------------------------------------------------------------
# the flag, and every refusal
# ---------------------------------------------------------------------------------

class _Env:
    def __init__(self, hvac_control="power", hvac_action_index=2,
                 heating_setpoint_idx=IDX_T_HEAT, num_buildings=B):
        self.hvac_control = hvac_control
        self.hvac_action_index = hvac_action_index
        self.heating_setpoint_idx = heating_setpoint_idx
        self.num_buildings = num_buildings
        self.action_names = ["dhw_storage", "electrical_storage",
                             "cooling_or_heating_device"]

    def heat_pump_info(self):
        f = lambda v: np.full(self.num_buildings, v, dtype=np.float32)
        return {"efficiency_heat": f(0.25), "target_heat": f(45.0),
                "nominal_power_heat": f(5.0), "efficiency_cool": f(0.25),
                "target_cool": f(8.0), "nominal_power_cool": f(5.0)}

    def _action_bounds(self):
        return np.ones((self.num_buildings, 3), dtype=np.float32)


def test_the_barrier_is_off_by_default():
    assert STEMSConfig().comfort.enabled is False
    assert build_comfort_barriers(_Env(), ComfortConfig(), rc=rc()) == []


def test_enabling_it_builds_both_sides():
    bars = build_comfort_barriers(_Env(), ComfortConfig(enabled=True), rc=rc())
    assert [b.name for b in bars] == ["comfort_heating", "comfort_cooling"]
    assert [b.direction for b in bars] == [1, -1]


def test_it_refuses_the_set_point_mode_with_the_dimensional_reason():
    with pytest.raises(RuntimeError, match="hvac_control='power'"):
        build_comfort_barriers(_Env(hvac_control="setpoint"),
                               ComfortConfig(enabled=True), rc=rc())


def test_it_refuses_to_invent_an_envelope():
    with pytest.raises(RuntimeError, match="no default envelope"):
        build_comfort_barriers(_Env(), ComfortConfig(enabled=True), rc=None)


def test_it_refuses_an_environment_with_no_heat_pump_action():
    with pytest.raises(RuntimeError, match="heat-pump action"):
        build_comfort_barriers(_Env(hvac_action_index=-1),
                               ComfortConfig(enabled=True), rc=rc())


def test_it_refuses_an_environment_with_no_heating_set_point():
    with pytest.raises(RuntimeError, match="heating set point"):
        build_comfort_barriers(_Env(heating_setpoint_idx=None),
                               ComfortConfig(enabled=True), rc=rc())


def test_it_refuses_an_envelope_of_the_wrong_width():
    with pytest.raises(ValueError, match="covers"):
        build_comfort_barriers(_Env(), ComfortConfig(enabled=True), rc=rc(b=B + 1))


def test_cooling_can_be_left_off():
    bars = build_comfort_barriers(_Env(), ComfortConfig(enabled=True, enforce_cooling=False),
                                  rc=rc())
    assert [b.name for b in bars] == ["comfort_heating"]
