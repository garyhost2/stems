"""Known-answer tests for the nine comparison controllers.

Same shape as ``tests/test_learning.py::test_policy_moves_to_the_known_optimum``: put a
controller in the mock environment, pay it a reward whose maximiser is a known constant
action, train it for a few episodes, and require the deterministic policy to have moved
toward that constant. A controller that cannot find the optimum of a stationary quadratic
in two action dimensions is not a controller, whatever its loss curves say.

The reward is :math:`r_i = -4(a_{i,h} - 0.6)^2 - 4(a_{i,e} + 0.3)^2`, with :math:`h` the
heat-pump column and :math:`e` the battery column, so the optimum is
:math:`a_h = +0.6, a_e = -0.3` for every building. It does not depend on the observation,
which is deliberate: this test asks whether the optimiser works, not whether the
representation does.

Budgets here are small on purpose -- the point is to detect a controller that does not
learn at all, not to measure how well it learns. Training grids are run centrally.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.observations import obs_index
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

from experiments.controllers import (ARMS, COMPARISON_POLICIES, ComparisonController,
                                     build_controller)

COMPARISON_POLICIES = set(COMPARISON_POLICIES)

STEPS, EPISODES = 80, 14
TARGET_HVAC, TARGET_BATT = 0.6, -0.3

#: Smaller networks, smaller batches and a shorter warm-up than the defaults, so the
#: whole file runs in the time a test suite is allowed. Every other hyper-parameter is
#: the controller's own default.
TEST_OVERRIDES = dict(hidden_dim=64, batch_size=64, start_steps=64,
                      updates_per_step=1.0)


def _env():
    set_seed(0)
    return STEMSEnvironment(force_mock=True, heat_pump=True)


def _reward(actions, h, e):
    return [-4.0 * (a[h] - TARGET_HVAC) ** 2 - 4.0 * (a[e] - TARGET_BATT) ** 2
            for a in actions]


def _episode(controller, env, cfg, explore, buf=None):
    h, e = env.hvac_action_index, env.electrical_storage_action_index
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    for k in range(STEPS):
        window = hist.get()
        actions = controller.select_action(obs, window, explore=explore)
        nxt = env.step(actions)[0]
        if hasattr(controller, "observe"):
            controller.observe(nxt)
        rewards = _reward(controller._last_raw_actions, h, e)
        hist.update(nxt)
        if buf is not None:
            buf.add(obs=obs, actions=actions, rewards=rewards, next_obs=nxt,
                    done=(k == STEPS - 1), history=window, next_history=hist.get(),
                    raw_actions=controller._last_raw_actions,
                    safe_actions=controller._last_safe_actions,
                    pre_tanh=controller._last_pre_tanh,
                    behaviour_log_probs=controller._last_log_probs,
                    constraint_costs=np.zeros((env.num_buildings, 3), np.float32))
        obs = nxt
    raw = controller._last_raw_actions
    return float(raw[:, h].mean()), float(raw[:, e].mean())


def _train(controller, env, cfg, episodes=EPISODES):
    for _ in range(episodes):
        buf = EpisodeBuffer()
        _episode(controller, env, cfg, explore=True, buf=buf)
        controller.update(buf.get_batch())
    return _episode(controller, env, cfg, explore=False)


def _build(name, env, **overrides):
    """Build the arm, then re-make its learner with the test-sized hyper-parameters."""
    cfg = STEMSConfig()
    controller = build_controller(ARMS[name], env, cfg)
    if overrides:
        base = type(controller.base)
        kwargs = dict(obs_dim=env.obs_dim, action_dim=env.action_dim,
                      num_buildings=env.num_buildings, **overrides)
        controller.base = base(**kwargs)
    return controller, cfg


# ---------------------------------------------------------------------------
# Every comparison arm is reachable, builds, and acts in the legal range
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES))
def test_the_arm_builds_and_acts_in_range(name):
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    actions = controller.select_action(obs, hist.get(), explore=False)
    assert actions.shape == (env.num_buildings, env.action_dim)
    assert np.isfinite(actions).all()
    assert (np.abs(actions) <= 1.0 + 1e-6).all(), f"{name} left the action box"


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES))
def test_the_arm_presents_the_attributes_the_runner_reads(name):
    """``experiments/runner.py::train`` reads these off the agent it is handed."""
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    for attr in ("elec_idx", "control_indices", "_last_raw_actions", "_last_safe_actions",
                 "_last_pre_tanh", "_last_log_probs", "_last_nominal_actions",
                 "fleet_shield", "update", "save", "load", "observe"):
        assert hasattr(controller, attr), f"{name} is missing {attr!r}"
    assert isinstance(controller, ComparisonController)


# ---------------------------------------------------------------------------
# Known-answer: the policy moves to the optimum of a stationary quadratic
# ---------------------------------------------------------------------------
def test_single_agent_sac_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("sac", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_dmappo_moves_to_the_known_optimum():
    # Twice the episodes of the off-policy arms: on-policy PPO discards each episode
    # after one update and its approximate-KL early stop caps how far a single update
    # can move the policy, so it needs more of them to cover the same ground.
    env = _env()
    controller, cfg = _build("dmappo", env)
    hvac, batt = _train(controller, env, cfg, episodes=2 * EPISODES)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_maddpg_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("maddpg", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_marlisa_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("marlisa", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_mappo_with_a_centralised_critic_moves_to_the_known_optimum():
    env = _env()
    controller = build_controller(ARMS["mappo-cc"], env, STEMSConfig())
    hvac, batt = _train(controller, env, STEMSConfig())
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_metaems_moves_to_the_known_optimum():
    """Reptile with buildings as tasks still has to solve a task all buildings share.

    Stated as a paired before-and-after reduction in the distance to the optimum rather
    than as an absolute threshold. Reptile moves the initialisation only a fraction of
    the way toward each task's adapted parameters, so where it lands after a fixed
    budget is noisy; that it moved most of the way there is the claim worth testing, and
    it does not depend on picking a threshold that happens to sit beside the answer.
    """
    env = _env()
    controller, cfg = _build("metaems", env, hidden_dim=64, batch_size=64,
                             start_steps=64, inner_steps=8, lr_outer=0.7)
    optimum = np.array([TARGET_HVAC, TARGET_BATT])
    before = np.linalg.norm(np.array(_episode(controller, env, cfg, explore=False)) - optimum)
    after = np.linalg.norm(np.array(_train(controller, env, cfg, episodes=2 * EPISODES))
                           - optimum)
    assert after < 0.6 * before, (
        f"distance to the optimum went from {before:.3f} to {after:.3f}; Reptile did "
        "not meta-learn the shared task")


def test_madcq_moves_to_the_known_optimum_on_its_unconstrained_axis():
    """The discrete arm is asked for the heat-pump optimum only.

    Its battery column is restricted to the plant-feasible set at every step, so the
    unconstrained optimum -0.3 is not always available to it. That restriction is the
    method, not a defect, and it is checked separately in
    ``test_madcq_never_proposes_an_action_outside_the_safe_set``.
    """
    env = _env()
    controller, cfg = _build("madcq", env, hidden_dim=64, batch_size=64, start_steps=64,
                             updates_per_step=0.25, battery_model=env.battery_model(),
                             elec_idx=env.electrical_storage_action_index,
                             eps_decay_steps=400)
    hvac, _ = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"


def test_madcq_never_proposes_an_action_outside_the_safe_set():
    """Constrained Q-learning: the greedy action is restricted to the feasible bins.

    The safe set comes from ``BatteryModel.safe_interval`` on the exact plant, so the
    check is against the model, not against a rate constant.
    """
    env = _env()
    controller = build_controller(ARMS["madcq"], env, STEMSConfig())
    agent, battery = controller.base, env.battery_model()
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    soc_idx = obs_index("electrical_storage_soc")
    e = env.electrical_storage_action_index
    worst = 0.0
    for _ in range(40):
        soc = np.array([float(o[soc_idx]) for o in obs], dtype=np.float64)
        lo, hi = battery.safe_interval(soc, np.full(env.num_buildings, cfg.cbf.SOC_min),
                                       np.full(env.num_buildings, cfg.cbf.SOC_max))
        raw = agent.select_action(obs, hist.get(), explore=True)
        worst = max(worst, float(np.maximum(lo - raw[:, e], raw[:, e] - hi).max()))
        obs = env.step(controller.select_action(obs, hist.get(), explore=True))[0]
        hist.update(obs)
    # One bin width is 0.2; the tolerance is a tenth of that.
    assert worst <= 0.02, (
        f"MADCQ proposed a battery action {worst:.4f} outside the exact plant's safe "
        "interval; the constrained maximisation is not being applied")


# ---------------------------------------------------------------------------
# The specific defects audit A3 named, pinned so they cannot come back
# ---------------------------------------------------------------------------
def test_dmappo_uses_the_behaviour_log_probability_from_the_trajectory():
    """Audit A3: the importance ratio must be new-policy over *behaviour* policy.

    On the first minibatch of the first epoch the two policies are the same network, so
    the ratio is exactly one and the approximate KL is zero. If the update recomputed
    ``old_log_prob`` from an already-stepped actor, it would not be.
    """
    env = _env()
    controller, cfg = _build("dmappo", env)
    controller.base.epochs = 1
    controller.base.minibatch_size = 10_000
    buf = EpisodeBuffer()
    _episode(controller, env, cfg, explore=True, buf=buf)
    stats = controller.update(buf.get_batch())
    assert stats["recomputed_behaviour_logp"] is False
    # The tolerance is 1e-4, not 0: the behaviour log-probability was computed on a
    # one-row tensor inside ``select_action`` and is recomputed here on an 80-row one,
    # and float32 matrix multiplication is not associative, so the two differ in the
    # last couple of digits. A ratio taken against a *recomputed* old log-probability --
    # the defect this pins -- gives an approximate KL two or more orders of magnitude
    # larger than this, so the test still separates the two cases.
    assert abs(stats["approx_kl"]) < 1e-4, stats
    assert stats["clip_frac"] == 0.0


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES - {"mpc", "mpc-oracle"}))
def test_the_stored_behaviour_samples_are_not_all_the_same_object(name):
    """Every step must store its *own* pre-squash sample and log-probability.

    The trajectory collector keeps the reference it is handed for the whole episode. A
    learner that fills one reusable buffer in place therefore stores N references to
    one array, and every transition ends up carrying the last step's draw. It is silent
    -- shapes are right, training runs, losses move -- and it removes all per-step
    signal from the importance ratio. ``DMAPPOAgent`` did exactly this.
    """
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    seen_pre, seen_logp = [], []
    for _ in range(6):
        actions = controller.select_action(obs, hist.get(), explore=True)
        seen_pre.append(controller._last_pre_tanh)
        seen_logp.append(controller._last_log_probs)
        obs = env.step(actions)[0]
        hist.update(obs)
    assert len({id(a) for a in seen_pre}) == len(seen_pre), "pre-tanh buffer is reused"
    assert len({id(a) for a in seen_logp}) == len(seen_logp), "log-prob buffer is reused"
    stacked = np.stack(seen_pre)
    assert stacked.std(axis=0).max() > 1e-6, (
        f"{name}'s stored pre-squash samples are identical across steps")


def test_dmappo_has_no_hard_coded_battery_model():
    """Audit A3: the "CBF penalty" built from ``soc + 0.1 * action`` is gone."""
    import inspect

    import stems.baselines as baselines

    source = inspect.getsource(baselines.DMAPPOAgent)
    assert "_cbf_penalty" not in source
    assert "0.1" not in source.split('"""')[2] if source.count('"""') > 2 else True


def test_marlisa_builds_the_same_augmentation_when_acting_and_when_updating():
    """Audit A3: the critic must see byte-for-byte what the actor saw.

    ``augment`` is the single function both paths call; this checks that the vector the
    actor conditioned on at selection time is reproduced exactly from the stored raw
    actions, for every building, including the sequential block.
    """
    env = _env()
    controller = build_controller(ARMS["marlisa"], env, STEMSConfig())
    agent = controller.base
    obs, _ = env.reset()
    actions = agent.select_action(obs, None, explore=True)
    for i in range(env.num_buildings):
        seen = agent.augment(obs, actions, i)
        assert seen.shape[0] == agent.aug_dim[i]
        if i > 0:
            block = seen[env.obs_dim:env.obs_dim + i * env.action_dim]
            assert np.allclose(block, actions[:i].reshape(-1)), (
                f"building {i}'s sequential block is not buildings 0..{i - 1}'s actions")
        rebuilt = agent.augment(obs, actions, i)
        assert np.array_equal(seen, rebuilt)


def test_madcq_does_not_enumerate_the_joint_action_space():
    """Audit A3: 11**3 = 1331 outputs per building became 3 * 11 = 33."""
    env = _env()
    controller = build_controller(ARMS["madcq"], env, STEMSConfig())
    head = controller.base.q[0].heads
    assert head.out_features == env.action_dim * controller.base.n_bins
    assert head.out_features < 11 ** 3


def test_metaems_is_reptile_over_more_than_one_task():
    """Audit A3: a 50/50 average of one task's parameters is not meta-learning."""
    env = _env()
    controller = build_controller(ARMS["metaems"], env, STEMSConfig())
    agent = controller.base
    assert len(agent.buffers) == env.num_buildings, "one replay per task"
    assert agent.inner_steps >= 1
    assert hasattr(agent, "adapt"), "a meta-learner must expose test-time adaptation"


def test_single_agent_sac_is_actually_single_agent():
    """The policy takes the concatenated district observation, not a per-building one."""
    env = _env()
    controller = build_controller(ARMS["sac"], env, STEMSConfig())
    agent = controller.base
    first = agent.policy.trunk[0]
    assert first.in_features == env.num_buildings * env.obs_dim
    assert agent.policy.mean_head.out_features == env.num_buildings * env.action_dim


def test_the_mpc_agent_name_refuses_rather_than_resurrecting_the_old_one():
    from stems.baselines import MPCAgent

    with pytest.raises(NotImplementedError) as excinfo:
        MPCAgent()
    assert "stems.mpc.StorageMPC" in str(excinfo.value)
