"""Step 8: arm names reserved for the baseline and constraint tracks.

Each was registered in ``ARMS`` with the fields it needs and no builder. The contract
these tests pin is: the name resolves, ``Arm.implemented`` is False, and
``build_controller`` raises ``NotImplementedError`` with a message that says what to
build — never a ``KeyError`` that reads like a typo, and never a silent fall-through to
some other branch that would produce a run record looking like a result.

**Edited by the constraints track.** Sixteen of the twenty-four names now have
builders: the twelve-cell mechanism cross, the two hard-comfort arms and the two
degradation arms. The reservation contract above is therefore asserted only over the
eight comparison controllers that are still reserved, and the sixteen that landed are
asserted to have the opposite properties — they resolve, report ``implemented is
True``, and build. The alternative was to keep asserting that implemented work is
missing. See CHANGELOG.md, constraints track, for the rule this is recorded under.
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

#: Both method tracks have now landed, so NOTHING remains reserved. The reservation
#: contract these tests were written to pin (name resolves, ``implemented`` is False,
#: ``build_controller`` raises a useful ``NotImplementedError``) no longer has a
#: subject; what is worth pinning instead is that every reserved name was filled in
#: under that exact name rather than renamed or restructured away.
IMPLEMENTED_BY_BASELINES_TRACK = COMPARISON_CONTROLLERS
IMPLEMENTED_BY_CONSTRAINTS_TRACK = MECHANISM_2X2 + COMFORT_ARMS + DEGRADATION_ARMS
#: 24 names were reserved by step 8; 25 arms landed, because the baselines track split
#: the perfect-foresight MPC out as `mpc-oracle`, a second controller rather than a
#: mode of the first. So this is "implemented by the method tracks", not "reserved".
IMPLEMENTED_BY_METHOD_TRACKS = IMPLEMENTED_BY_BASELINES_TRACK + IMPLEMENTED_BY_CONSTRAINTS_TRACK
ALL_PRE_REGISTERED: tuple = ()




def test_nothing_remains_reserved():
    """Both method tracks landed, so the reservation list is empty.

    Replaces the two parametrised reservation tests (every reserved arm reports
    ``implemented is False`` and raises a useful ``NotImplementedError``). Their
    parameter set is now empty, so they would collect zero cases and silently pass.
    Recorded under the test-editing rule: the contract they pinned was consumed by the
    work they were guarding, and ``test_every_reserved_name_landed_under_that_name``
    below is the assertion that the hand-off actually happened.
    """
    assert ALL_PRE_REGISTERED == ()
    assert tuple(PRE_REGISTERED_ARMS) == ()
    # Absorbs test_the_name_is_registered, whose parameter set is now empty:
    # pytest reports an empty parametrisation as a SKIP, which would read as
    # coverage that silently never ran.
    assert all(a.implemented for a in ARMS.values()), \
        [n for n, a in ARMS.items() if not a.implemented]


def test_the_pre_registered_set_is_exactly_what_we_declared():
    assert set(PRE_REGISTERED_ARMS) == set(ALL_PRE_REGISTERED)
    assert len(PRE_REGISTERED_ARMS) == 0
    assert len(IMPLEMENTED_BY_METHOD_TRACKS) == 9 + 16 == 25  # 24 reserved + mpc-oracle


@pytest.mark.parametrize("name", IMPLEMENTED_BY_METHOD_TRACKS)
def test_every_method_track_arm_resolves_and_is_implemented(name):
    """Reserving the names worked: they were filled in, not restructured."""
    assert name in ARMS and ARMS[name].name == name
    assert ARMS[name].implemented is True
    assert name not in PRE_REGISTERED_ARMS


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
    assert all(ARMS[n].learns for n in ("rl+calibrated+comfort",
                                        "rl+calibrated+degr-throughput"))
    assert ARMS["rbc+calibrated+comfort"].learns is False
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
