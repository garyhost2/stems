"""The interfaces the framework is written against, so nothing above them imports a simulator.

Four protocols, split along the line of what a deployment can actually supply.

``PlantModel``
    A device whose one-step dynamics are known *and invertible*: ``next_soc`` maps a
    command to the next state, ``action_for_soc`` maps a wanted next state back to the
    command that realises it. The second is what makes a projection a projection rather
    than a guess. ``stems.battery.BatteryModel`` and ``stems.battery.TankModel``
    implement it; ``stems.fleet.EVFleetModel`` implements the same idea in power
    (``draw_kw`` / ``action_for_draw``) because a charger is commanded in kW.

``Environment``
    What a controller needs: a step function, an observation layout, and the actions
    the plant actually executed. **Every real-building adapter must implement this.**

``PlantProvider``
    What a *projection* needs on top: the plant models for this site and the action
    columns they own. An adapter implements it when the plant has been characterised.
    Without it the shield cannot be exact, and must not pretend to be.

``EVProvider``
    What the cap shield needs when there are chargers. Optional by construction: a
    building without vehicles is not a degenerate case to be special-cased, it simply
    does not implement this protocol.

Why split rather than one big interface: an interface a real adapter cannot satisfy is
an interface nobody implements. A metered house with no battery model can serve
``Environment`` today; it cannot serve ``PlantProvider``, and the framework should say
which capabilities are missing rather than fail at the first attribute error.

**What ``isinstance`` checks here do and do not prove.** These protocols are
``runtime_checkable``, so ``isinstance(x, Environment)`` verifies that the *names*
exist. It does not verify signatures, shapes, units or semantics. Treat a passing
check as "the adapter is wired", never as "the adapter is correct";
``stems.replay.CSVReplayEnvironment`` carries the conformance assertions that actually
bite (observation layout against ``selected_obs_names``, action width, dtype).

Units follow ``stems/observations.py``: kW, kWh, degC, hours, state of charge in [0, 1].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "PlantModel",
    "Environment",
    "PlantProvider",
    "EVProvider",
    "missing_capabilities",
]


@runtime_checkable
class PlantModel(Protocol):
    """A store whose one-step map is known and can be inverted exactly.

    ``soc`` is dimensionless in [0, 1]; ``action`` is the environment's command for
    this device's column. Implementations take extra keyword arguments for exogenous
    inputs they need (``TankModel`` needs ``demand``, the hot-water draw in kWh for the
    step), so callers that are generic over plants must pass them through.
    """

    def next_soc(self, soc: np.ndarray, action: np.ndarray, *args: Any,
                 **kwargs: Any) -> np.ndarray:
        """State of charge after one step under ``action``."""

    def action_for_soc(self, soc: np.ndarray, target: np.ndarray, *args: Any,
                       **kwargs: Any) -> np.ndarray:
        """Smallest-magnitude command whose successor state is ``target``.

        Smallest magnitude is load-bearing, not a tie-break: ``next_soc`` is flat in the
        action wherever the device cannot move energy, and a plain bisection over the
        whole interval returns a full-power command that changes nothing in the model
        and a great deal in the plant's accounting. See ``stems.battery``.
        """


@runtime_checkable
class Environment(Protocol):
    """The seam between a controller and whatever it is controlling.

    ``step`` returns ``(obs_list, rewards, terminated, truncated, info)`` with
    ``obs_list`` a list of ``num_buildings`` float32 vectors of length ``obs_dim``, laid
    out as ``stems.observations.selected_obs_names`` describes. ``reset`` returns
    ``(obs_list, info)``.

    ``executed_actions`` is what the plant *did*, shape ``(num_buildings, action_dim)``,
    which is not the commanded action whenever a device saturates or a lower-level
    controller intervenes. Every metric in ``stems.metrics`` that attributes energy to a
    command reads this, not the command.
    """

    @property
    def num_buildings(self) -> int: ...

    @property
    def obs_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def obs_names(self) -> List[str]: ...

    @property
    def action_names(self) -> List[str]: ...

    @property
    def executed_actions(self) -> np.ndarray: ...

    def index_of(self, name: str) -> int:
        """Position of ``name`` in this environment's observation vector."""

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]: ...

    def step(self, actions: np.ndarray
             ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]: ...


@runtime_checkable
class PlantProvider(Protocol):
    """Sites that can describe their own devices well enough to be projected onto.

    The index properties return ``-1`` when the device is absent, matching
    ``STEMSEnvironment``; callers must check rather than assume a column exists.
    """

    @property
    def electrical_storage_action_index(self) -> int: ...

    @property
    def dhw_action_index(self) -> int: ...

    @property
    def hvac_action_index(self) -> int: ...

    def battery_model(self) -> PlantModel: ...

    def battery_info(self) -> Dict[str, np.ndarray]:
        """``capacity`` (kWh), ``nominal_power`` (kW), ``soc_rate`` (per step)."""

    def dhw_tank_model(self) -> Optional[PlantModel]: ...

    def dhw_info(self) -> Dict[str, np.ndarray]:
        """``capacity`` (kWh), ``nominal_power`` (kW), ``efficiency``,
        ``loss_coefficient``, ``action_bound``."""


@runtime_checkable
class EVProvider(Protocol):
    """Sites with chargers. Optional: a building without vehicles does not implement it."""

    def ev_action_indices(self) -> List[int]: ...

    def ev_obs_layout(self) -> List[Dict[str, int]]: ...

    def ev_fleet_model(self, slot: int = 0) -> Any: ...

    @property
    def ev_draw_kwh(self) -> np.ndarray: ...

    @property
    def ev_departures(self) -> List[Dict[str, float]]: ...


_PROTOCOLS = {"Environment": Environment, "PlantProvider": PlantProvider,
              "EVProvider": EVProvider}


def missing_capabilities(adapter: object) -> Dict[str, Sequence[str]]:
    """Which protocols ``adapter`` fails, and which attribute is missing from each.

    Written for adapter authors: ``isinstance`` answers yes or no, and the useful
    question when the answer is no is *which name*. Protocols the adapter satisfies are
    absent from the result, so an empty dict means it implements all three.
    """
    out: Dict[str, Sequence[str]] = {}
    cls = type(adapter)
    for name, proto in _PROTOCOLS.items():
        # The capability is declared by the class. Probing the *instance* alone is
        # wrong here: ``STEMSEnvironment.executed_actions`` is a property that raises
        # until ``reset()`` has run, and a pre-reset adapter would be reported as
        # missing three capabilities it has.
        missing = [m for m in getattr(proto, "__protocol_attrs__", set())
                   if not (hasattr(cls, m) or hasattr(adapter, m))]
        if missing:
            out[name] = sorted(missing)
    return out
