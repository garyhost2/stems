"""One object for every flexible load, and one cap over a set of them.

``docs/FRAMEWORK.md`` states the abstraction; this module *is* it. A household flexible
load is a store with a rate limit that must reach a required level by a deadline:

    FlexibleLoad(name, kind, action_index, plant, soc_fn, band, barrier)

and a neighbourhood is a set of them under one shared import cap:

    FlexibilityPortfolio(loads, cap_kw, dt_hours, coordination)

Before this module the four loads were four construction sites. ``EVReadinessBarrier``
was built in ``experiments/controllers.py``, ``DHWReadinessBarrier`` in
``stems/thermal.py::build_thermal_stack``, ``ThermalComfortBarrier`` in
``stems/comfort.py::build_comfort_barriers``, and the house battery was not a load at
all -- it was a pair of state-of-charge bounds inside ``CBFShield.project`` and a
shedding branch inside ``FleetShield``. They are now one constructor with four argument
sets, assembled in one place (``FlexibilityPortfolio.from_environment``), and the
shield delegates to the portfolio rather than looping over barriers itself.

Two faces, and why they are separate
------------------------------------
A store carries two constraints with different shapes, and conflating them is how the
battery came to be a special case.

*The band*, ``SOC_min <= soc <= SOC_max``, holds at every step and is enforced by
inverting the plant exactly (``plant.safe_interval``). Nothing is owed; the store is
simply not allowed to leave the interval.

*The deadline*, ``soc >= required_soc`` at ``steps_to_deadline == 0``, holds at one
step, is enforced by ``DeadlineStorageBarrier.project``, and is inert while
``slack > 0``.

A house battery has a band and no deadline (``barrier=None``). A hot-water tank and the
Legionella cycle have deadlines on one store. An electric vehicle has both. Writing the
faces apart makes the battery the *degenerate* member of one family rather than a fifth
code path.

Symbols are those of ``docs/FRAMEWORK.md`` §2 and of ``stems/deadline.py``; units are
kW, kWh, hours, and dimensionless state of charge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple)

import numpy as np

from stems.deadline import (DeadlineRequirement, DeadlineStorageBarrier,
                            coupled_feasibility, prioritise)
from stems.protocols import EVProvider, PlantProvider

__all__ = [
    "FlexibleLoad",
    "FlexibilityPortfolio",
    "LOAD_KINDS",
]

#: The kinds of flexible load the framework instantiates. ``"envelope"`` is the
#: building's thermal mass (``stems.comfort``), in the family and gated off on
#: CityLearn; see ``docs/FRAMEWORK.md`` §5.
LOAD_KINDS: Tuple[str, ...] = ("battery", "dhw", "legionella", "ev", "envelope")

SocFn = Callable[[Sequence[np.ndarray]], np.ndarray]


@dataclass
class FlexibleLoad:
    """One store, its rate limit, its required level, its deadline.

    ``plant``
        The device's one-step model, satisfying ``stems.protocols.PlantModel``. Used
        for the band (via ``safe_interval``) and, when the barrier was built with
        ``plant=...``, for the deadline projection as well.
    ``band``
        ``(soc_lo, soc_hi)``, the interval the store may never leave, or ``None`` for a
        load with no band constraint. Enforced only when ``plant`` exposes
        ``safe_interval``; a tank's state of charge is bounded by its own dynamics
        rather than by a configured band, so it passes ``None``.
    ``barrier``
        The deadline face, or ``None`` for a store with nothing owed (a house battery).
    ``power_kw``
        Rated charging power per building, kW, when the load has one. The cap shield
        needs it to convert an action into an import; ``None`` means this load cannot
        participate in power-based coordination and is left alone by it.
    """

    name: str
    kind: str
    action_index: int
    plant: Optional[Any] = None
    soc_fn: Optional[SocFn] = None
    band: Optional[Tuple[Any, Any]] = None
    barrier: Optional[DeadlineStorageBarrier] = None
    action_bound: float = 1.0
    capacity_kwh: Optional[np.ndarray] = None
    power_kw: Optional[np.ndarray] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in LOAD_KINDS:
            raise ValueError(f"unknown load kind {self.kind!r}; choose from {LOAD_KINDS}")
        if self.barrier is None and self.band is None:
            raise ValueError(
                f"flexible load {self.name!r} has neither a band nor a deadline, so it "
                "constrains nothing. A load with no constraint is not a load; leave it "
                "out of the portfolio.")
        if self.capacity_kwh is not None:
            self.capacity_kwh = np.asarray(self.capacity_kwh, dtype=np.float64).reshape(-1)
        if self.power_kw is not None:
            self.power_kw = np.asarray(self.power_kw, dtype=np.float64).reshape(-1)
        if self.barrier is not None and self.barrier.action_index != self.action_index:
            raise ValueError(
                f"load {self.name!r} owns action column {self.action_index} but its "
                f"barrier projects onto column {self.barrier.action_index}")

    # -- the two faces ----------------------------------------------------------

    @property
    def has_band(self) -> bool:
        return self.band is not None and hasattr(self.plant, "safe_interval")

    @property
    def has_deadline(self) -> bool:
        return self.barrier is not None

    @property
    def exact_projection(self) -> bool:
        """True when *both* faces this load uses invert the plant exactly."""
        band_ok = (not self.has_band) or self.plant is not None
        deadline_ok = (not self.has_deadline) or self.barrier.exact_projection
        return bool(band_ok and deadline_ok)

    def project_band(self, actions: np.ndarray,
                     obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """Clip this load's action into the set whose successor state stays in ``band``."""
        if not self.has_band or self.soc_fn is None:
            return actions
        soc = np.asarray(self.soc_fn(obs_list), dtype=np.float32)
        lo, hi = self.band
        a_lo, a_hi = self.plant.safe_interval(soc, lo, hi, a_max=self.action_bound)
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[:, self.action_index] = np.clip(actions[:, self.action_index], a_lo, a_hi)
        return actions

    def project_deadline(self, actions: np.ndarray,
                         obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """Raise this load's action to the minimum the deadline still allows."""
        if self.barrier is None:
            return actions
        return self.barrier.project(actions, list(obs_list))

    def project(self, actions: np.ndarray,
                obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """Band first, then deadline: a deadline may legitimately push to the band edge."""
        return self.project_deadline(self.project_band(actions, obs_list), obs_list)

    # -- reporting --------------------------------------------------------------

    def urgency(self, obs_list: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        if self.barrier is None:
            return {}
        return self.barrier.urgency(list(obs_list))

    def energy_still_required_kwh(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        if self.barrier is None:
            return np.zeros(len(obs_list), dtype=np.float32)
        return self.barrier.energy_still_required_kwh(list(obs_list))

    def describe(self) -> Dict[str, Any]:
        """One row of the instantiation table in ``docs/FRAMEWORK.md`` §5."""
        return {"name": self.name, "kind": self.kind, "action_index": self.action_index,
                "plant": type(self.plant).__name__ if self.plant is not None else None,
                "has_band": self.has_band, "has_deadline": self.has_deadline,
                "exact_projection": self.exact_projection,
                "capacity_kwh": (None if self.capacity_kwh is None
                                 else [round(float(c), 4) for c in self.capacity_kwh]),
                "power_kw": (None if self.power_kw is None
                             else [round(float(p), 4) for p in self.power_kw]),
                "notes": self.notes}


@dataclass
class FlexibilityPortfolio:
    """A set of flexible loads contending for one shared import cap.

    ``project`` reproduces, exactly, what ``CBFShield`` used to do inline: every
    deadline barrier projects in turn, then -- when ``coordination`` is not
    ``"independent"`` -- the loads that draw power are cut back to the headroom under
    the cap. The behaviour is unchanged by construction; the point of moving it here is
    that the cap is now a property of a *set of loads* rather than a private method of
    the battery shield, which is what lets the Legionella cycle join the set without
    touching ``stems/cbf.py``.
    """

    loads: List[FlexibleLoad] = field(default_factory=list)
    cap_kw: float = float("inf")
    dt_hours: float = 1.0
    coordination: str = "independent"
    net_index: int = 0

    #: ClassVar, not a field: it is the vocabulary of ``coordination``, not per-instance
    #: state, and a dataclass field here would put it in the constructor signature.
    COORDINATIONS: ClassVar[Tuple[str, ...]] = ("independent", "proportional", "edf")

    def __post_init__(self) -> None:
        if self.coordination not in self.COORDINATIONS:
            raise ValueError(f"coordination must be one of {self.COORDINATIONS}, "
                             f"got {self.coordination!r}")
        names = [l.name for l in self.loads]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate load names in the portfolio: {names}")

    # -- construction -----------------------------------------------------------

    @classmethod
    def from_barriers(cls, barriers: Sequence[DeadlineStorageBarrier],
                      cap_kw: float = float("inf"), dt_hours: float = 1.0,
                      coordination: str = "independent",
                      net_index: int = 0) -> "FlexibilityPortfolio":
        """Wrap already-built barriers as deadline-only loads.

        The compatibility path: ``CBFShield`` is handed barriers, not loads, and this
        keeps its behaviour identical while putting them behind the framework object.
        """
        loads = [cls._load_from_barrier(b) for b in barriers]
        return cls(loads=loads, cap_kw=cap_kw, dt_hours=dt_hours,
                   coordination=coordination, net_index=net_index)

    @staticmethod
    def _load_from_barrier(barrier: DeadlineStorageBarrier) -> FlexibleLoad:
        kind = {"ev": "ev", "dhw": "dhw", "legionella": "legionella"}.get(
            barrier.name, "envelope" if barrier.name.startswith("comfort") else None)
        if kind is None:
            kind = "envelope"
        # ``_p_charge`` is ``EVReadinessBarrier``'s rated charger power. It is read
        # through ``getattr`` because that is exactly what ``CBFShield`` did, and the
        # shared-power allocation must keep skipping the barriers that do not have it.
        power = getattr(barrier, "_p_charge", None)
        return FlexibleLoad(name=barrier.name, kind=kind,
                            action_index=barrier.action_index, barrier=barrier,
                            action_bound=float(np.max(barrier.action_bound)),
                            capacity_kwh=barrier.capacity,
                            power_kw=None if power is None else np.asarray(power),
                            notes="wrapped from an existing barrier")

    @classmethod
    def from_environment(cls, env: PlantProvider, cbf_cfg: Any,
                         soc_band: Optional[Tuple[Any, Any]] = None,
                         dhw_barrier: Optional[DeadlineStorageBarrier] = None,
                         ev_barrier: Optional[DeadlineStorageBarrier] = None,
                         legionella_barrier: Optional[DeadlineStorageBarrier] = None,
                         extra: Sequence[DeadlineStorageBarrier] = (),
                         coordination: str = "independent"
                         ) -> "FlexibilityPortfolio":
        """Assemble the loads this site actually has, in one place.

        The house battery is always present when the environment has a battery column;
        the others join when their barrier is supplied. Nothing here invents a device
        the environment does not report.
        """
        from stems.observations import obs_index

        loads: List[FlexibleLoad] = []
        e = env.electrical_storage_action_index
        if e >= 0:
            info = env.battery_info()
            lo, hi = soc_band if soc_band is not None else (cbf_cfg.SOC_min, cbf_cfg.SOC_max)
            idx = obs_index("electrical_storage_soc")
            loads.append(FlexibleLoad(
                name="battery", kind="battery", action_index=e,
                plant=env.battery_model(),
                soc_fn=lambda obs, i=idx: np.array([float(o[i]) for o in obs],
                                                   dtype=np.float32),
                band=(lo, hi), barrier=None,
                capacity_kwh=info["capacity"], power_kw=info["nominal_power"],
                notes="band only: a house battery owes nothing by a deadline"))
        for barrier in (dhw_barrier, legionella_barrier, ev_barrier, *extra):
            if barrier is not None:
                loads.append(cls._load_from_barrier(barrier))
        return cls(loads=loads, cap_kw=float(cbf_cfg.P_grid_max),
                   coordination=coordination,
                   net_index=obs_index("net_electricity_consumption"))

    # -- the set ----------------------------------------------------------------

    @property
    def barriers(self) -> List[DeadlineStorageBarrier]:
        return [l.barrier for l in self.loads if l.barrier is not None]

    def __len__(self) -> int:
        return len(self.loads)

    def __iter__(self):
        return iter(self.loads)

    def of_kind(self, kind: str) -> List[FlexibleLoad]:
        return [l for l in self.loads if l.kind == kind]

    def describe(self) -> List[Dict[str, Any]]:
        return [l.describe() for l in self.loads]

    # -- projection -------------------------------------------------------------

    def project_deadlines(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                          cap_kw: Optional[float] = None) -> np.ndarray:
        """Every deadline, then the shared cap. The body ``CBFShield`` used to hold."""
        for barrier in self.barriers:
            actions = barrier.project(actions, list(obs_list))
        if self.coordination != "independent":
            actions = self.allocate_under_cap(actions, obs_list, cap_kw)
        return actions

    def project(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                cap_kw: Optional[float] = None) -> np.ndarray:
        """Bands, then deadlines, then the cap."""
        for load in self.loads:
            actions = load.project_band(actions, obs_list)
        return self.project_deadlines(actions, obs_list, cap_kw)

    def allocate_under_cap(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                           cap_kw: Optional[float] = None) -> np.ndarray:
        """Cut the power-drawing loads back to the headroom under the cap.

        ``"proportional"`` scales every request by the same factor; ``"edf"`` serves
        the earliest deadline first and cuts the rest. Loads with no ``power_kw`` are
        skipped, which is how a load whose action is not a power fraction (the
        envelope's set-point offset) stays out of a power allocation.
        """
        cap = self.cap_kw if cap_kw is None else float(cap_kw)
        net = np.array([float(o[self.net_index]) for o in obs_list], dtype=np.float32)
        headroom = max(cap - float(np.maximum(net, 0.0).sum()), 0.0)

        entries, requested_total = [], 0.0
        for load in self.loads:
            if load.power_kw is None or load.barrier is None:
                continue
            u = load.barrier.urgency(list(obs_list))
            kw = np.maximum(actions[:, load.action_index], 0.0) * load.power_kw
            for i in range(len(kw)):
                if kw[i] > 1e-9:
                    entries.append((float(u["steps_to_deadline"][i]), load, i, float(kw[i])))
                    requested_total += float(kw[i])

        if requested_total <= headroom + 1e-9 or requested_total <= 1e-9:
            return actions
        if self.coordination == "proportional":
            scale = headroom / requested_total
            for _, load, i, kw in entries:
                actions[i, load.action_index] = actions[i, load.action_index] * scale
            return actions
        entries.sort(key=lambda e: (e[0], -e[3]))
        budget = headroom
        for _, load, i, kw in entries:
            grant = min(kw, budget)
            budget -= grant
            j = load.action_index
            actions[i, j] = actions[i, j] * (grant / kw) if kw > 1e-9 else 0.0
        return actions

    # -- joint feasibility ------------------------------------------------------

    def feasibility(self, obs_list: Sequence[np.ndarray],
                    cap_kw: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Is the whole set still serviceable under the cap, and if not, who misses.

        Conservative by construction: the horizon is the *earliest* deadline anyone
        owes, so the budget is the smallest one that any member of the set has to live
        within. Two loads can each be feasible alone and jointly infeasible; this is
        the question that cannot be asked one device at a time.
        """
        barriers = self.barriers
        if not barriers:
            return None
        cap = self.cap_kw if cap_kw is None else float(cap_kw)
        report = coupled_feasibility(barriers, list(obs_list), power_cap_kw=cap,
                                     dt_hours=self.dt_hours)
        if not report["feasible"]:
            report["priority"] = prioritise(barriers, list(obs_list), power_cap_kw=cap,
                                            dt_hours=self.dt_hours)
        return report
