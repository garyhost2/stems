"""Constraints track steps 3 and 4: the second constrained-RL family, and the 2x2.

What these pin:

* each of the four mechanism cells switches exactly what its name says and nothing
  else, and ``mechanism="auto"`` reproduces the historical coupling bit for bit;
* the cost critics are trained in **every** cell, including "no mechanism" -- they are
  the measurement instrument for the violation rate and all arms must measure it the
  same way;
* FOCOPS and PPO-Lagrangian share the cost critics, the generalised-advantage
  estimates, the encoder and the multiplier state, and differ only in the actor
  surrogate;
* the twelve cross arms are identical except for ``barrier`` and ``mechanism``, so a
  fixed learning budget makes the mechanism the only difference between them;
* the learned safety filter is gone, and nothing imports it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import fields

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.controllers import (ARMS, Arm, MECHANISMS, PLANT_MODELS,
                                     build_controller, mechanism_switches)
from stems.agent import CONSTRAINED_ALGORITHMS, STEMSAgent
from stems.config import STEMSConfig

MECHANISM_2X2 = tuple(f"mech-{m}+{p}"
                      for m in ("none", "lagrangian", "projection", "both")
                      for p in ("uniform", "linear", "exact"))


# ---------------------------------------------------------------------------------
# the 2x2 switches
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mechanism,expected", [
    ("none", (False, False)),
    ("lagrangian", (False, True)),
    ("projection", (True, False)),
    ("both", (True, True)),
])
def test_each_cell_switches_what_its_name_says(mechanism, expected):
    for plant in PLANT_MODELS:
        assert mechanism_switches(ARMS[f"mech-{mechanism}+{plant}"]) == expected


def test_auto_reproduces_the_historical_coupling():
    """Projection whenever barrier != none; constrained policy whenever it learns."""
    for name, arm in ARMS.items():
        if arm.mechanism != "auto" or not arm.implemented:
            continue
        assert mechanism_switches(arm) == (arm.barrier != "none", arm.learns), name


def test_a_non_learning_policy_cannot_be_asked_for_a_policy_mechanism():
    with pytest.raises(ValueError, match="no actor to constrain"):
        mechanism_switches(Arm("x", "rbc", "calibrated", mechanism="lagrangian"))
    with pytest.raises(ValueError, match="no actor to constrain"):
        mechanism_switches(Arm("x", "rbc", "calibrated", mechanism="both"))
    # projection alone is fine on a rule-based policy: there is a shield, no actor.
    assert mechanism_switches(Arm("x", "rbc", "calibrated", mechanism="projection")) \
        == (True, False)


def test_an_unknown_mechanism_is_refused():
    with pytest.raises(ValueError, match="unknown mechanism"):
        mechanism_switches(Arm("x", "rl", "calibrated", mechanism="sideways"))
    assert set(MECHANISMS) == {"auto", "none", "lagrangian", "projection", "both"}


def test_the_cross_holds_everything_but_the_mechanism_fixed():
    """A fixed learning budget then makes the mechanism the only difference."""
    varying = {"name", "barrier", "mechanism"}
    reference = ARMS["mech-both+exact"]
    for name in MECHANISM_2X2:
        arm = ARMS[name]
        for f in fields(Arm):
            if f.name in varying:
                continue
            assert getattr(arm, f.name) == getattr(reference, f.name), \
                f"{name} differs from the cross in {f.name!r}"


def test_the_plant_model_axis_is_inert_where_the_projection_is_off():
    """Three names, one controller: say so rather than pay three times for it."""
    for mech in ("none", "lagrangian"):
        for plant in PLANT_MODELS:
            assert ARMS[f"mech-{mech}+{plant}"].mechanism_is_plant_sensitive is False
    for mech in ("projection", "both"):
        for plant in PLANT_MODELS:
            assert ARMS[f"mech-{mech}+{plant}"].mechanism_is_plant_sensitive is True
    assert ARMS["rl"].mechanism_is_plant_sensitive is False
    assert ARMS["rl+calibrated"].mechanism_is_plant_sensitive is True


# ---------------------------------------------------------------------------------
# what the switches do to a built controller
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_env():
    from stems.environment import STEMSEnvironment

    return STEMSEnvironment(force_mock=True, heat_pump=True)


@pytest.mark.parametrize("name", MECHANISM_2X2)
def test_every_cross_arm_builds_and_carries_its_switches(mock_env, name):
    arm = ARMS[name]
    cfg = STEMSConfig()
    agent = build_controller(arm, mock_env, cfg)
    projection, lagrangian = mechanism_switches(arm)
    assert agent.use_cbf is projection
    assert cfg.lagrangian.enabled is lagrangian


def test_the_cost_critics_are_trained_in_every_cell_including_no_mechanism(mock_env):
    """They are the measurement instrument; switching the mechanism off must not
    switch off the measurement, or the cells are not comparable."""
    for name in ("mech-none+exact", "mech-both+exact"):
        cfg = STEMSConfig()
        agent = build_controller(ARMS[name], mock_env, cfg)
        batch = _batch(agent, mock_env)
        agent.update(batch)                      # first call only seeds the normaliser
        stats = agent.update(batch)
        assert stats["gradient_steps"] > 0
        assert stats["cost_value"] != 0.0, f"{name} did not train its cost critics"


def test_no_mechanism_leaves_the_multipliers_untouched(mock_env):
    cfg = STEMSConfig()
    agent = build_controller(ARMS["mech-none+exact"], mock_env, cfg)
    before = agent._lambdas.clone()
    batch = _batch(agent, mock_env)
    agent.update(batch)
    agent.update(batch)
    assert torch.allclose(agent._lambdas, before)


def test_the_lagrangian_cell_does_move_the_multipliers(mock_env):
    cfg = STEMSConfig()
    agent = build_controller(ARMS["mech-lagrangian+exact"], mock_env, cfg)
    before = agent._lambdas.clone()
    batch = _batch(agent, mock_env, violating=True)
    agent.update(batch)
    agent.update(batch)
    assert not torch.allclose(agent._lambdas, before)


# ---------------------------------------------------------------------------------
# the second family
# ---------------------------------------------------------------------------------

def test_the_two_families_are_registered():
    assert CONSTRAINED_ALGORITHMS == ("ppo-lagrangian", "focops")
    assert STEMSConfig().lagrangian.algorithm == "ppo-lagrangian"


def test_an_unknown_family_is_refused(mock_env):
    cfg = STEMSConfig()
    cfg.lagrangian.algorithm = "cpo-ish"
    agent = build_controller(ARMS["mech-both+exact"], mock_env, cfg)
    with pytest.raises(ValueError, match="unknown constrained algorithm"):
        agent.update(_batch(agent, mock_env))


def test_focops_uses_the_same_cost_critics_and_the_same_multipliers(mock_env):
    """Same class, same width, same state tensors -- only the surrogate differs."""
    lag_cfg, foc_cfg = STEMSConfig(), STEMSConfig()
    foc_cfg.lagrangian.algorithm = "focops"
    lag = build_controller(ARMS["mech-both+exact"], mock_env, lag_cfg)
    foc = build_controller(ARMS["mech-both+exact"], mock_env, foc_cfg)
    assert type(lag.cost_critics) is type(foc.cost_critics)
    assert len(lag.cost_critics) == len(foc.cost_critics)
    assert lag.cost_critics[0].num_constraints == foc.cost_critics[0].num_constraints
    assert lag._lambdas.shape == foc._lambdas.shape
    assert lag._cost_limit.shape == foc._cost_limit.shape


def test_focops_takes_gradient_steps_and_reports_which_family_ran(mock_env):
    cfg = STEMSConfig()
    cfg.lagrangian.algorithm = "focops"
    agent = build_controller(ARMS["mech-both+exact"], mock_env, cfg)
    batch = _batch(agent, mock_env)
    agent.update(batch)
    stats = agent.update(batch)
    assert stats["gradient_steps"] > 0
    assert stats["mechanism_algorithm"] == "focops"
    assert np.isfinite(stats["policy"])


def test_the_reported_family_is_none_when_the_mechanism_is_off(mock_env):
    cfg = STEMSConfig()
    agent = build_controller(ARMS["mech-none+exact"], mock_env, cfg)
    batch = _batch(agent, mock_env)
    agent.update(batch)
    assert agent.update(batch)["mechanism_algorithm"] == "none"


def test_focops_combines_the_advantages_without_the_lagrangian_denominator():
    adv = torch.randn(40, 3)
    cadv = torch.randn(40, 3, 3)
    nu = torch.tensor([0.4, 0.0, 1.1])
    foc = STEMSAgent.focops_advantage(adv, cadv, nu)
    lag = STEMSAgent.effective_advantage(adv, cadv, nu)
    assert torch.allclose(foc / (1.0 + nu.sum()), lag, atol=1e-6)


def test_focops_zero_multipliers_leave_the_reward_advantage_alone():
    adv = torch.randn(30, 2)
    cadv = torch.randn(30, 2, 3)
    nu = torch.zeros(3)
    assert torch.allclose(STEMSAgent.focops_advantage(adv, cadv, nu), adv, atol=1e-6)


def test_the_focops_kl_estimate_is_non_negative_and_zero_at_the_behaviour_policy(mock_env):
    cfg = STEMSConfig()
    cfg.lagrangian.algorithm = "focops"
    agent = build_controller(ARMS["mech-both+exact"], mock_env, cfg)
    logp = torch.randn(64)
    _, kl = agent._focops_loss(logp, logp.clone(), torch.randn(64))
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)
    _, kl2 = agent._focops_loss(logp, logp - 0.5, torch.randn(64))
    assert torch.all(kl2 >= -1e-7)


def test_the_focops_trust_region_defaults_to_ppos_early_stopping_threshold(mock_env):
    """Same-sized trust region in both families, so the comparison is not confounded."""
    cfg = STEMSConfig()
    assert cfg.lagrangian.focops_kl_limit is None
    assert cfg.training.target_kl == 0.02


# ---------------------------------------------------------------------------------
# the learned safety filter is gone
# ---------------------------------------------------------------------------------

def test_the_untrained_neural_safety_filter_was_removed():
    import stems.cbf as cbf

    assert not hasattr(cbf, "NeuralSafetyFilter")


def test_nothing_in_the_tree_still_refers_to_it():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = [p for p in list(root.glob("stems/*.py")) + list(root.glob("experiments/*.py"))
                 if "NeuralSafetyFilter(" in p.read_text(encoding="utf-8")]
    assert offenders == []


# ---------------------------------------------------------------------------------
# end to end, on the real simulator
# ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def short_scenario():
    from experiments.scenario import Scenario

    sc = Scenario(season="winter", days=2, hvac_control="power",
                  degradation_price_per_kwh=150.0,
                  degradation_limit_kwh_per_episode=0.02,
                  degradation_dod_exponent=0.5)
    try:
        sc.schema_path()
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"real CityLearn schema unavailable: {exc!r}")
    return sc


def _real_env(sc, phase):
    from stems.environment import STEMSEnvironment

    return STEMSEnvironment(schema=sc.schema_path(), seed=0, heat_pump=True,
                            env_kwargs=sc.env_kwargs(phase), hvac_control="power")


def test_the_degradation_arm_runs_end_to_end_and_reports_fade(short_scenario):
    from experiments.runner import evaluate, make_config, train

    env = _real_env(short_scenario, "train")
    cfg = make_config(short_scenario)
    agent = build_controller(ARMS["rl+calibrated+degr-dod"], env, cfg)
    assert cfg.lagrangian.num_constraints == 4, "the budget must add a cost channel"
    rows = train(agent, env, cfg, episodes=1, log=lambda m: None, max_steps=24)
    assert rows[0]["capacity_loss_kwh"] > 0.0
    k = evaluate(agent, _real_env(short_scenario, "eval"), cfg, 24)["kpis"]
    assert k["battery_capacity_loss_kwh"] > 0.0
    assert k["battery_degradation_cost"] == pytest.approx(
        150.0 * k["battery_capacity_loss_kwh"])
    assert "battery_equivalent_full_cycles" in k


def test_focops_runs_end_to_end_on_the_real_simulator(short_scenario):
    from experiments.runner import make_config, train

    env = _real_env(short_scenario, "train")
    cfg = make_config(short_scenario)
    cfg.lagrangian.algorithm = "focops"
    agent = build_controller(ARMS["mech-both+exact"], env, cfg)
    rows = train(agent, env, cfg, episodes=2, log=lambda m: None, max_steps=24)
    assert rows[-1]["gradient_steps"] > 0
    assert np.isfinite(rows[-1]["reward"])


def test_the_comfort_arm_runs_end_to_end_against_an_identified_envelope(short_scenario):
    from stems.comfort import identify_from_rollout
    from experiments.runner import evaluate, make_config

    env = _real_env(short_scenario, "train")
    rc, fit = identify_from_rollout(env, steps=48, seed=0)
    cfg = make_config(short_scenario)
    controller = build_controller(ARMS["rbc+calibrated+comfort"], env, cfg,
                                  rc_model=rc)
    assert [b.name for b in controller.comfort_barriers] == ["comfort_heating",
                                                             "comfort_cooling"]
    k = evaluate(controller, _real_env(short_scenario, "eval"), cfg, 24)["kpis"]
    # Both reported, always: a barrier that binds and a band that breaks anyway are
    # different facts, and a hard constraint whose breach rate is not zero is not hard.
    assert "comfort_barrier_binding_rate" in k
    assert "comfort_band_breach_rate" in k
    assert 0.0 <= k["comfort_band_breach_rate"] <= 1.0


def test_the_envelope_identified_on_citylearn_is_reported_not_trusted(short_scenario):
    """The fit must come back with its goodness attached, because on this simulator it
    is poor: the one-step residual is several K against a 2 K comfort tolerance."""
    from stems.comfort import identify_from_rollout

    _, fit = identify_from_rollout(_real_env(short_scenario, "train"), steps=48, seed=0)
    s = fit.summary()
    assert set(s) >= {"r2_min", "r2_median", "rmse_k_max", "tau_h_min", "tau_h_max"}
    assert np.isfinite(s["rmse_k_max"])


def test_the_dataset_route_refuses_rather_than_returning_a_wrong_envelope(short_scenario):
    """CityLearn's dataset gives the *ideal* thermal load, not delivered power against
    a free-running temperature, so the pair is not an input-output pair for an
    envelope. The identification must say so rather than hand back a fitted model."""
    from stems.comfort import identify_from_citylearn

    with pytest.raises(ValueError, match="not physical"):
        identify_from_citylearn(_real_env(short_scenario, "train"))


# ---------------------------------------------------------------------------------

def _batch(agent, env, violating: bool = False, n: int = 24):
    """A small synthetic rollout batch of the shape `STEMSAgent.update` requires."""
    rng = np.random.default_rng(0)
    B, A, D = agent.B, agent.action_dim, agent.obs_dim
    W = agent.cfg.transformer.window_size
    K = agent.cfg.lagrangian.num_constraints
    f32 = lambda *s: rng.normal(0.0, 1.0, size=s).astype(np.float32)
    acts = np.clip(f32(n, B, A), -0.9, 0.9)
    costs = np.ones((n, B, K), np.float32) if violating else np.zeros((n, B, K), np.float32)
    return {"obs": f32(n, B, D), "next_obs": f32(n, B, D),
            "history": f32(n, B, W, D), "next_history": f32(n, B, W, D),
            # `dones` is one flag per step, not per building: `EpisodeBuffer` stores a
            # scalar, and `_compute_gae` relies on it broadcasting against both the
            # (N, B) reward advantages and the (N, B, K) cost advantages.
            "rewards": f32(n, B), "dones": np.zeros(n, np.float32),
            "safe_actions": acts, "raw_actions": acts, "pre_tanh": np.arctanh(acts),
            "behaviour_log_probs": f32(n, B), "constraint_costs": costs}
