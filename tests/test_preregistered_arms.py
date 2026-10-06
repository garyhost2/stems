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

from experiments.controllers import (ARMS, MECHANISMS, PLANT_MODELS,
                                     PRE_REGISTERED_ARMS,
                                     PRE_REGISTERED_POLICIES, build_controller)

COMPARISON_CONTROLLERS = ("sac", "dmappo", "mpc", "maddpg", "marlisa", "madcq",
                          "metaems", "mappo-cc")
MECHANISM_2X2 = tuple(f"mech-{m}+{p}"
                      for m in ("none", "lagrangian", "projection", "both")
                      for p in ("uniform", "linear", "exact"))
COMFORT_ARMS = ("rl+calibrated+comfort", "rbc+calibrated+comfort")
DEGRADATION_ARMS = ("rl+calibrated+degr-throughput", "rl+calibrated+degr-dod")

ALL_PRE_REGISTERED = (COMPARISON_CONTROLLERS + MECHANISM_2X2 + COMFORT_ARMS
                      + DEGRADATION_ARMS)


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
    assert len(PRE_REGISTERED_ARMS) == 8 + 12 + 2 + 2 == 24


def test_the_eight_comparison_controllers_cover_table_one_plus_a_centralised_critic():
    assert len(COMPARISON_CONTROLLERS) == 8
    assert set(PRE_REGISTERED_POLICIES) == set(COMPARISON_CONTROLLERS)
    # MAPPO with a centralised critic is the one that answers audit D1, so it must be
    # distinct from the DMAPPO of the STEMS table.
    assert "mappo-cc" in PRE_REGISTERED_POLICIES
    assert "dmappo" in PRE_REGISTERED_POLICIES
    assert PRE_REGISTERED_POLICIES["mappo-cc"] != PRE_REGISTERED_POLICIES["dmappo"]
    for policy, note in PRE_REGISTERED_POLICIES.items():
        assert len(note) > 30, f"{policy} needs a note saying what to build"


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


def test_learning_arms_are_unchanged_by_the_new_policies():
    """`learns` drives episode counts and checkpointing; it must stay conservative."""
    for name in COMPARISON_CONTROLLERS:
        assert ARMS[name].learns is False, (
            f"{name} must not report learns=True until it has a builder, or the grid "
            "driver will allocate it a training budget it cannot use")
