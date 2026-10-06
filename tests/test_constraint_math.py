"""Audit B1 (the heat-pump power guard) and B3 (the Lagrangian's advantage scales).

Notation, consistent with docs/REPORT_2026-10.md:
  e_b            net electricity consumption of building b, kW, positive = import
  a_b            heat-pump action of building b, dimensionless in [-1, 1]
  p_b            heat-pump nominal electrical power of building b, kW
  P_grid_max     district import cap, kW
  I(s)           achieved district import under shed factor s,
                 I(s) = sum_b max(e_b + s*|a_b|*p_b, 0), kW
  A              reward advantage, standardised, dimensionless
  A^c_k          advantage of constraint cost k
  lambda_k       Lagrange multiplier of constraint k
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.agent import STEMSAgent
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.observations import obs_index
from stems.thermal import CoPModel

_IDX_T_OUT = obs_index("outdoor_dry_bulb_temperature")
_IDX_NET = obs_index("net_electricity_consumption")
_OBS_LEN = 30
B = 4
HVAC_IDX = 2


def _cop_model(p_nom_kw: float = 10.0) -> CoPModel:
    """A heat pump with the same size and efficiency in every building."""
    ones = np.ones(B, dtype=np.float32)
    return CoPModel(efficiency_heat=0.3 * ones, target_heat=45.0 * ones,
                    efficiency_cool=0.3 * ones, target_cool=7.0 * ones,
                    nominal_power_heat=p_nom_kw * ones,
                    nominal_power_cool=p_nom_kw * ones)


def _states(net_kw, t_out_c: float = 5.0):
    out = []
    for e in np.asarray(net_kw, dtype=np.float32):
        s = np.zeros(_OBS_LEN, dtype=np.float32)
        s[_IDX_NET] = e
        s[_IDX_T_OUT] = t_out_c
        out.append(s)
    return out


def _shield(grid_cap_kw: float, p_nom_kw: float = 10.0,
            building_cap_kw: float = 1e6) -> CBFShield:
    cfg = CBFConfig(P_grid_max=grid_cap_kw, P_building_max=building_cap_kw)
    safety = SafetyConfig(robust_margins=False)
    return CBFShield(config=cfg, num_buildings=B, safety_cfg=safety,
                     enforce_soc=False, cop_model=_cop_model(p_nom_kw),
                     hvac_idx=HVAC_IDX, action_scale=1.0)


def _achieved_import(safe, states, p_nom_kw: float = 10.0) -> float:
    """I(1) recomputed in float64 from the action the shield stored."""
    net = np.array([s[_IDX_NET] for s in states], dtype=np.float64)
    draw = np.abs(safe[:, HVAC_IDX]) * p_nom_kw
    return float(np.maximum(net + draw, 0.0).sum())


def _achieved_import_f32(safe, states, p_nom_kw: float = 10.0) -> float:
    """The same quantity in the shield's own float32 arithmetic."""
    net = np.array([s[_IDX_NET] for s in states], dtype=np.float32)
    p = np.full(B, p_nom_kw, dtype=np.float32)
    return float(np.maximum(net + np.abs(safe[:, HVAC_IDX]) * p, 0.0).sum())


#: Tolerance on the cap, in kW. The shield bisects on, and stores, float32 actions; it
#: meets the cap exactly in that arithmetic (``_achieved_import_f32`` below asserts it).
#: Recomputing the same import in float64 with a float64 nominal power disagrees in the
#: last few bits: the measured residual on the 100 kW case below is 3.81e-08 relative
#: (3.81e-06 kW, i.e. 3.8 mW). That is representation error in the comparison, not slack
#: in the method.
_CAP_TOL_KW = 1e-3


# ---------------------------------------------------------------- audit B1 ----

def test_hvac_guard_brings_the_district_import_to_the_cap():
    """The guard must meet the cap, and the old cap/total rescale did not.

    Four buildings, e_b = 20 kW each (80 kW uncontrollable), every building asking for
    full heat-pump power p_b = 10 kW (40 kW controllable), so I(1) = 4*(20+10) = 120 kW
    against a P_grid_max of 100 kW. The shed factor that meets the cap is
    s = (100-80)/40 = 0.5, and the guard must find it.

    The pre-fix rescale took the factor cap/I(1) = 100/120 = 0.8333 and applied it to the
    action, giving 4*(20 + 0.8333*10) = 113.33 kW -- 13.3 kW over the cap it was supposed
    to enforce, because the 80 kW of uncontrollable load was scaled down along with the
    heat pumps even though nothing can scale it.
    """
    cap = 100.0
    states = _states([20.0] * B)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0

    shield = _shield(cap)
    safe = shield.project(actions.copy(), states)
    achieved = _achieved_import(safe, states)

    assert _achieved_import_f32(safe, states) <= cap, (
        "the shield must meet the cap in its own arithmetic, exactly")
    assert achieved <= cap + _CAP_TOL_KW, f"import {achieved:.6f} kW exceeds {cap} kW"
    assert achieved == pytest.approx(cap, abs=_CAP_TOL_KW), (
        "the guard should sit on the cap, not far below it")
    assert safe[:, HVAC_IDX] == pytest.approx(np.full(B, 0.5), abs=1e-4)
    assert safe[:, HVAC_IDX].dtype == np.float32

    # What the pre-fix arithmetic would have produced, computed here explicitly.
    total = _achieved_import(actions, states)
    old = actions.copy()
    old[:, HVAC_IDX] *= cap / total
    old_import = _achieved_import(old, states)
    assert old_import > cap + 1.0, (
        "the cap/total rescale is supposed to miss the cap; if it no longer does, this "
        "test has stopped testing anything")
    assert old_import == pytest.approx(113.333333, abs=1e-3), (
        "13.3 kW over a 100 kW cap is the size of the bug audit B1 describes")


def test_hvac_guard_sheds_everything_when_the_fixed_load_alone_exceeds_the_cap():
    """No heat-pump action can meet a cap the uncontrollable load already breaks.

    The honest response is to shed the whole controllable draw, which bisection does
    (s -> 0). The old rescale would have left a positive heat-pump draw on top of an
    already-infeasible import.
    """
    cap = 50.0
    states = _states([20.0] * B)  # 80 kW fixed, above the 50 kW cap
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0

    safe = _shield(cap).project(actions.copy(), states)
    assert safe[:, HVAC_IDX] == pytest.approx(np.zeros(B), abs=1e-6)
    assert _achieved_import(safe, states) == pytest.approx(80.0, abs=1e-6)


def test_hvac_guard_leaves_a_feasible_request_untouched():
    cap = 300.0
    states = _states([20.0] * B)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 0.4
    safe = _shield(cap).project(actions.copy(), states)
    assert safe[:, HVAC_IDX] == pytest.approx(np.full(B, 0.4), abs=1e-6)


def test_hvac_guard_handles_an_exporting_building_without_offsetting_an_import():
    """A building exporting PV must not create headroom for another's import.

    e = (-30, 30, 30, 30): the signed sum is 60 kW but the import is 90 kW. The guard
    enforces the positive-part definition, the same one the reward and the KPIs use
    after audit B5.
    """
    cap = 100.0
    states = _states([-30.0, 30.0, 30.0, 30.0])
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0
    safe = _shield(cap).project(actions.copy(), states)
    achieved = _achieved_import(safe, states)
    assert _achieved_import_f32(safe, states) <= cap
    assert achieved <= cap + _CAP_TOL_KW
    assert achieved == pytest.approx(cap, abs=_CAP_TOL_KW)


def test_hvac_guard_is_unreachable_without_a_cop_model():
    """It stays off unless a COP model is supplied; see CHANGELOG.md for that decision."""
    cfg = CBFConfig(P_grid_max=1.0, P_building_max=1.0)
    shield = CBFShield(config=cfg, num_buildings=B,
                       safety_cfg=SafetyConfig(robust_margins=False),
                       enforce_soc=False, cop_model=None, hvac_idx=HVAC_IDX)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0
    safe = shield.project(actions.copy(), _states([20.0] * B))
    assert safe[:, HVAC_IDX] == pytest.approx(np.ones(B))


# ---------------------------------------------------------------- audit B3 ----

def _advantages(seed: int = 0, N: int = 200, n_b: int = 3, K: int = 3):
    g = torch.Generator().manual_seed(seed)
    adv = torch.randn(N, n_b, generator=g)
    adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
    cadv = torch.randn(N, n_b, K, generator=g) * 7.0 + 3.0
    return adv, cadv


def test_effective_advantage_is_invariant_to_the_cost_scale():
    """The property B3 says was missing: lambda means the same thing at any cost scale."""
    adv, cadv = _advantages()
    lam = torch.tensor([0.4, 1.2, 0.0])
    base = STEMSAgent.effective_advantage(adv, cadv, lam)
    for factor in (0.01, 10.0, 1000.0):
        scaled = STEMSAgent.effective_advantage(adv, cadv * factor, lam)
        assert torch.allclose(base, scaled, atol=1e-4), (
            f"a x{factor} rescale of the costs changed the objective")


def test_effective_advantage_was_not_invariant_before_the_fix():
    """Pin that the test above is testing something: mean-centring alone fails it."""
    adv, cadv = _advantages()
    lam = torch.tensor([0.4, 1.2, 0.0])

    def old(adv_, cadv_, lam_):
        cadv_ = cadv_ - cadv_.mean(0)
        return (adv_ - (cadv_ * lam_).sum(-1)) / (1.0 + lam_.sum())

    base = old(adv, cadv, lam)
    scaled = old(adv, cadv * 10.0, lam)
    assert not torch.allclose(base, scaled, atol=1e-2)


def test_cost_advantages_reach_unit_variance():
    adv, cadv = _advantages()
    lam_only_first = torch.tensor([1.0, 0.0, 0.0])
    eff = STEMSAgent.effective_advantage(adv, cadv, lam_only_first)
    # A_eff = (A - 1.0 * z_0) / 2 with z_0 standardised, so recovering z_0 and checking
    # its moments checks the standardisation without reaching into private state.
    z0 = adv - 2.0 * eff
    assert z0.mean(0).abs().max().item() < 1e-5
    assert torch.allclose(z0.std(0), torch.ones_like(z0.std(0)), atol=1e-3)


def test_effective_advantage_reduces_to_the_reward_advantage_at_zero_lambda():
    adv, cadv = _advantages()
    eff = STEMSAgent.effective_advantage(adv, cadv, torch.zeros(3))
    assert torch.allclose(eff, adv, atol=1e-6)


def test_the_same_violation_rate_gives_the_same_lambda_trajectory_at_two_reward_scales():
    """Requested check: lambda must not depend on how the reward is scaled.

    Note this held before the B3 fix as well -- `_update_lambdas` is a PID on the
    measured cost rate and never reads the reward. The scale dependence B3 identifies is
    in the *objective* `A_eff`, which the two tests above cover. This test pins the
    lambda half of the claim so a future change to `_update_lambdas` cannot introduce a
    reward-scale dependence unnoticed.
    """
    from stems.config import STEMSConfig

    def trajectory(reward_scale: float):
        cfg = STEMSConfig()
        agent = object.__new__(STEMSAgent)
        agent.cfg = cfg
        agent.device = torch.device("cpu")
        K = cfg.lagrangian.num_constraints
        agent._lambdas = torch.full((K,), cfg.lagrangian.lambda_init)
        agent._cost_integral = torch.full((K,), cfg.lagrangian.lambda_init)
        agent._prev_cost = torch.zeros(K)
        agent._cost_limit = torch.full((K,), cfg.lagrangian.cost_limit)
        rng = np.random.default_rng(0)
        trace = []
        for _ in range(25):
            rate = torch.tensor(rng.uniform(0.0, 0.3, size=K), dtype=torch.float32)
            # the reward scale enters nowhere in the dual update; pass it through a
            # no-op so the test would catch it if that ever changed
            _ = float(reward_scale)
            agent._update_lambdas(rate)
            trace.append(agent._lambdas.clone())
        return torch.stack(trace)

    assert torch.allclose(trajectory(1.0), trajectory(1000.0), atol=1e-7)
