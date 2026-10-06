"""Audit A3's "exported but unreachable" finding, pinned so it cannot recur.

The finding was that ``stems/__init__.py`` exported seven controller classes that no
experiment instantiated: they were importable, they looked like part of the system, and
nothing ran them, so nothing noticed that they would not have survived review. The fix
is not only to build them but to make the correspondence checkable, in both directions:

* every controller class exported from ``stems`` is reachable through at least one arm
  in ``experiments/controllers.py::ARMS``, and
* every comparison arm builds the class it claims to.

``stems.CONTROLLERS_REACHABLE_FROM_ARMS`` is the declared mapping; these tests check it
against what ``build_controller`` actually returns, so a stale declaration fails too.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stems
from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment

from experiments.controllers import ARMS, COMPARISON_POLICIES, build_controller

#: Exported names that are not controllers: configuration, buffers, components.
NOT_CONTROLLERS = {
    "STEMSConfig", "STEMSEnvironment", "BuildingGraph", "STEncoder", "CBFShield",
    "STEMSReward", "MetricsCalculator", "ReplayBuffer", "EpisodeBuffer", "HistoryBuffer",
    "CausalForecaster", "OracleForecaster", "CONTROLLERS_REACHABLE_FROM_ARMS",
}


def _exported_controllers():
    # `stems.FRAMEWORK_EXPORTS` is the flexibility framework's public interface -- the
    # deadline load, the portfolio, the environment protocols. None of them is a
    # controller, and unlike NOT_CONTROLLERS the package declares the set itself, so
    # adding a framework name does not require editing this file. See CHANGELOG.md,
    # abstraction track step 2.
    return sorted(set(stems.__all__) - NOT_CONTROLLERS - stems.FRAMEWORK_EXPORTS
                  - {"FRAMEWORK_EXPORTS"})


def test_every_exported_controller_declares_an_arm():
    declared = stems.CONTROLLERS_REACHABLE_FROM_ARMS
    exported = _exported_controllers()
    assert set(declared) == set(exported), (
        "stems.__all__ and CONTROLLERS_REACHABLE_FROM_ARMS disagree; a controller was "
        "exported or removed without updating the mapping.\n"
        f"  exported, not declared: {sorted(set(exported) - set(declared))}\n"
        f"  declared, not exported: {sorted(set(declared) - set(exported))}")


@pytest.mark.parametrize("name", sorted(stems.CONTROLLERS_REACHABLE_FROM_ARMS))
def test_the_declared_arms_exist(name):
    for arm in stems.CONTROLLERS_REACHABLE_FROM_ARMS[name]:
        assert arm in ARMS, f"{name} claims arm {arm!r}, which is not in ARMS"


def test_no_controller_is_exported_without_being_built_by_some_arm():
    """The direction audit A3 actually caught: importable but never instantiated."""
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    built = set()
    for arm_name in sorted(COMPARISON_POLICIES):
        controller = build_controller(ARMS[arm_name], env, STEMSConfig())
        built.add(type(controller.base).__name__)
    for arm_name in ("rbc", "rl"):
        controller = build_controller(ARMS[arm_name], env, STEMSConfig())
        built.add(type(getattr(controller, "base", controller)).__name__)
    unreachable = [n for n in _exported_controllers() if n not in built]
    assert not unreachable, (
        f"exported from stems but built by no arm: {unreachable}. This is audit A3 "
        "recurring: a controller nothing runs is a controller nothing checks.")


def test_the_retired_mpc_agent_name_is_not_exported_as_a_controller():
    assert "MPCAgent" not in stems.__all__
    assert not hasattr(stems, "MPCAgent")


def test_every_comparison_arm_builds_a_distinct_controller_class():
    """Nine arms, nine implementations -- not one class behind several names."""
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    classes = {}
    for arm_name in sorted(COMPARISON_POLICIES):
        classes[arm_name] = type(build_controller(ARMS[arm_name], env, STEMSConfig()).base)
    # The MPC pair deliberately shares one class and differs in its forecaster; every
    # other arm is its own implementation.
    assert classes["mpc"] is classes["mpc-oracle"] is stems.StorageMPC
    learners = {k: v for k, v in classes.items() if k not in ("mpc", "mpc-oracle")}
    assert len(set(learners.values())) == len(learners), (
        f"two comparison arms share an implementation: {learners}")


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES))
def test_every_comparison_controller_documents_itself(name):
    """A baseline whose deviations are not written down is not a baseline.

    Each class carries a docstring naming the reference implementation it follows and,
    where this action space made the method ambiguous, the variant that was implemented
    instead. This checks the docstring exists and is substantive, not that it is right
    -- that is what ``CHANGELOG.md`` and the known-answer tests are for.
    """
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    cls = type(build_controller(ARMS[name], env, STEMSConfig()).base)
    doc = inspect.getdoc(cls) or ""
    assert len(doc) > 200, f"{cls.__name__} needs a docstring saying what it implements"
