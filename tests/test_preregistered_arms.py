"""Step 8: arm names reserved for the baseline and constraint tracks.

Each is registered in ``ARMS`` with the fields it needs and no builder. The contract
these tests pin is: the name resolves, ``Arm.implemented`` is False, and
``build_controller`` raises ``NotImplementedError`` with a message that says what to
build — never a ``KeyError`` that reads like a typo, and never a silent fall-through to
some other branch that would produce a run record looking like a result.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.controllers import (ARMS, COMPARISON_POLICIES,
                                     LEARNING_COMPARISON_POLICIES, MECHANISMS,
                                     PLANT_MODELS, PRE_REGISTERED_ARMS,
                                     PRE_REGISTERED_POLICIES, build_controller)

#: The comparison controllers now have builders (CHANGELOG.md steps 1-3), so they are
#: no longer pre-registered names. `mpc-oracle` joined them: the perfect-foresight arm
#: is a second controller, not a mode of the first.
COMPARISON_CONTROLLERS = ("sac", "dmappo", "mpc", "mpc-oracle", "maddpg", "marlisa",
                          "madcq", "metaems", "mappo-cc")
MECHANISM_2X2 = tuple(f"mech-{m}+{p}"
                      for m in ("none", "lagrangian", "projection", "both")
                      for p in ("uniform", "linear", "exact"))
COMFORT_ARMS = ("rl+calibrated+comfort", "rbc+calibrated+comfort")
DEGRADATION_ARMS = ("rl+calibrated+degr-throughput", "rl+calibrated+degr-dod")

ALL_PRE_REGISTERED = MECHANISM_2X2 + COMFORT_ARMS + DEGRADATION_ARMS


@pytest.mark.parametrize("name", ALL_PRE_REGISTERED)
def test_the_name_is_registered(name):
    assert name in ARMS, (
        f"{name!r} is not in ARMS; a downstream track would hit a KeyError that reads "
        "like a typo")
    assert ARMS[name].name == name


@pytest.mark.parametrize("name", ALL_PRE_REGISTERED)
def test_the_arm_reports_itself_unimplemented(name):
    assert ARMS[name].implemented is False


@pytest.mark.parametrize("name", ALL_PRE_REGISTERED)
def test_building_it_raises_not_implemented_with_a_useful_message(name):
    with pytest.raises(NotImplementedError) as excinfo:
        build_controller(ARMS[name], env=None, config=None)
    message = str(excinfo.value)
    assert name in message
    assert "pre-registered, not implemented" in message
    assert len(message) > 120, "the message must say what to build, not just that it is missing"


def test_the_pre_registered_set_is_exactly_what_we_declared():
    assert set(PRE_REGISTERED_ARMS) == set(ALL_PRE_REGISTERED)
    assert len(PRE_REGISTERED_ARMS) == 12 + 2 + 2 == 16


def test_the_comparison_controllers_cover_table_one_plus_the_two_arms_it_lacked():
    """The seven STEMS Table I methods, plus a centralised critic, plus an MPC oracle.

    These eight names were reserved by step 8 and are now built (CHANGELOG.md steps
    1-3), so the assertion flipped from "every one raises" to "every one resolves to a
    controller". Recording the change under the test-editing rule: the previous version
    of this test asserted `set(PRE_REGISTERED_POLICIES) == set(COMPARISON_CONTROLLERS)`
    and that each arm reported `implemented is False`, which is the exact statement the
    baselines track was asked to falsify. What it pinned that still matters -- that the
    names exist, that MAPPO with a centralised critic is distinct from the
    decentralised D-MAPPO, and that each carries a description -- is pinned here
    against `COMPARISON_POLICIES` instead.
    """
    assert len(COMPARISON_CONTROLLERS) == 9
    assert set(COMPARISON_POLICIES) == set(COMPARISON_CONTROLLERS)
    assert not set(PRE_REGISTERED_POLICIES) & set(COMPARISON_CONTROLLERS)
    # MAPPO with a centralised critic is the one that answers audit D1, so it must be
    # distinct from the D-MAPPO of the STEMS table.
    assert COMPARISON_POLICIES["mappo-cc"] != COMPARISON_POLICIES["dmappo"]
    for policy, note in COMPARISON_POLICIES.items():
        assert len(note) > 30, f"{policy} needs a note saying what it is"
    for name in COMPARISON_CONTROLLERS:
        assert ARMS[name].implemented is True, f"{name} still reports no builder"
        assert ARMS[name].is_comparison is True
        assert name not in PRE_REGISTERED_ARMS


def test_the_two_model_predictive_arms_are_a_pair():
    """One causal, one with perfect foresight, and neither of them learns."""
    assert ARMS["mpc"].policy == "mpc" and ARMS["mpc-oracle"].policy == "mpc-oracle"
    assert ARMS["mpc"].barrier == ARMS["mpc-oracle"].barrier == "calibrated"
    assert ARMS["mpc"].learns is False and ARMS["mpc-oracle"].learns is False
    assert {"mpc", "mpc-oracle"}.isdisjoint(LEARNING_COMPARISON_POLICIES)


def test_the_mechanism_cross_is_complete():
    """Four mechanisms x three battery plant models, all twelve present."""
    assert len(MECHANISM_2X2) == 12
    seen = {(ARMS[n].mechanism, ARMS[n].plant_model) for n in MECHANISM_2X2}
    assert seen == {(m, p) for m in ("none", "lagrangian", "projection", "both")
                    for p in ("uniform", "linear", "exact")}
    for name in MECHANISM_2X2:
        arm = ARMS[name]
        assert arm.policy == "rl"
        assert arm.mechanism in MECHANISMS and arm.mechanism != "auto"


def test_plant_model_is_read_from_barrier_not_stored_twice():
    assert ARMS["rl+calibrated"].plant_model == "exact"
    assert ARMS["rl+linear"].plant_model == "linear"
    assert ARMS["rl+basic"].plant_model == "uniform"
    assert ARMS["rl"].plant_model is None
    for plant, barrier in PLANT_MODELS.items():
        assert ARMS[f"mech-both+{plant}"].barrier == barrier
        assert ARMS[f"mech-both+{plant}"].plant_model == plant


def test_the_comfort_and_degradation_flags_are_set():
    for name in COMFORT_ARMS:
        assert ARMS[name].comfort_barrier is True
    assert ARMS["rl+calibrated+degr-throughput"].degradation == "throughput"
    assert ARMS["rl+calibrated+degr-dod"].degradation == "throughput+dod"
    # the implemented arms must not have acquired either flag
    assert ARMS["rl+calibrated"].comfort_barrier is False
    assert ARMS["rl+calibrated"].degradation == "none"


def test_every_pre_existing_arm_still_builds_as_before():
    """Adding the fields must not make an existing arm look unimplemented."""
    existing = ("idle", "idle+calibrated", "rbc", "rbc+calibrated", "rl", "rl+basic",
                "rl+linear", "rl+calibrated", "rl-res+calibrated", "rl+calibrated+pen",
                "rbc-offpeak+calibrated", "rbc-never+calibrated", "rl+calibrated+own",
                "rl+calibrated+floor", "hp-shift", "rl-hp", "rl+calibrated+meanpool")
    for name in existing:
        arm = ARMS[name]
        assert arm.implemented is True, f"{name} became unimplemented"
        assert arm.mechanism == "auto"
        assert name not in PRE_REGISTERED_ARMS


def test_arm_names_are_unique_and_match_their_keys():
    assert len({a.name for a in ARMS.values()}) == len(ARMS)
    for key, arm in ARMS.items():
        assert key == arm.name


def test_the_learning_comparison_arms_now_ask_for_a_training_budget():
    """`learns` drives episode counts and checkpointing.

    Recording this under the test-editing rule: step 8 asserted `learns is False` for
    all eight comparison names, with the reason "until it has a builder". They have
    builders now, and seven of them take gradient steps, so an untrained one would be a
    randomly-initialised network in a comparison table -- the strawman the port exists
    to remove. The model-predictive pair still reports False, because it solves an
    optimisation rather than fitting parameters.
    """
    for name in COMPARISON_CONTROLLERS:
        expected = name not in ("mpc", "mpc-oracle")
        assert ARMS[name].learns is expected, (
            f"{name}.learns should be {expected}: a learner with no training budget "
            "enters the table untrained, and a budget given to the MPC is wasted")
