"""The weekly Legionella disinfection cycle, as the framework's fourth flexible load.

Implements ``docs/dhw_legionella_design.md``. The rule it encodes: once per window, the
hot-water store must be heated to a disinfection temperature; *when* in the window is
free. That is a store, a rate limit, a required level and a deadline -- so it is
``stems.deadline.DeadlineStorageBarrier`` with a different requirement function, not a
new mechanism. The slack is the interesting part: a week-long window on a load that
needs a few hours is an enormous block of discretion, which a price-aware policy can
place in the cheapest hours while the barrier guarantees the cycle happens.

Why the power-limited heat source has to come with it
-----------------------------------------------------
CityLearn's hot-water device serves every draw directly, in the hour it occurs, and is
sized so that it always can (``Building.update_energy_from_dhw_device``). An empty tank
therefore costs nothing: no draw is ever unmet, and the "readiness" state of charge is
a proxy with no physical consequence. A real heat-pump water heater is the opposite --
its source is small and cannot meet a shower in real time, which is the entire reason
the tank exists. So ``ShadowTank`` below re-books the same energy against a
*power-limited* source, which makes **unmet hot water** a real service failure and the
quantity a barrier must keep at zero.

Equivalence to CityLearn when the limit does not bind, from the design note: with a
draw ``d`` and a tank charge ``c`` in the same hour, CityLearn's heater supplies
``d + c`` and the tank gains ``c``; a tank-served system supplies ``d`` from the tank
and puts ``d + c`` in. Same electricity, same tank change. They differ only when
``d + c > P_hp``, which is exactly the case this model exists to capture.

**Where it runs.** ``ShadowTank`` is driven *alongside* the simulator, not inside
``STEMSEnvironment.step``. Putting it inside would change the energy accounting of
every existing run in this repository, which is an experiment-phase decision with a
before/after burden, not something to slip into a module. ``LegionellaStack.observe``
is the hook; ``experiments/legionella_demo.py`` shows the wiring.

Symbols, SI/device units, named as in ``docs/FRAMEWORK.md`` §2
---------------------------------------------------------------

===================  ================================================  ==========
symbol               meaning                                           unit
===================  ================================================  ==========
``C``                tank rated energy capacity                        kWh
``E``                stored energy                                     kWh
``soc``              ``E / C``                                         --
``P_hp``             heat-source thermal power limit                   kW
``loss``             standing-loss fraction per step                   per step
``draw``             hot-water draw demanded in the step               kWh
``unmet``            ``draw`` the store and source could not serve     kWh
``spill``            heat the store could not hold                     kWh
``T_cold``           cold-water inlet temperature                      degC
``T_rated``          temperature at which ``C`` is defined             degC
``T_normal``         normal operating set point                        degC
``T_legionella``     disinfection set point                            degC
``soc_legionella``   ``(T_legionella - T_cold) / (T_rated - T_cold)``  --
``period_hours``     length of the disinfection window                 h
``cop_legionella``   coefficient of performance at the disinfection sink  --
===================  ================================================  ==========

What is sourced here, and what is not
-------------------------------------
**Nothing in ``LegionellaSpec`` has a default.** The design note's three open
questions -- realistic ``P_hp``, the tank temperatures, and the coefficient of
performance at a 60 degC sink -- have no sourced answers anywhere in this repository,
and a number invented here would end up in a paper. Every one is a required argument,
``LegionellaSpec.provenance`` records where each came from, and
``LegionellaSpec.unsourced`` lists the ones still marked unsourced so a run record can
carry the list. The only quantity with a defensible default is ``period_hours``: the
design note states the rule as "once a week ... in new units the interval is a user
setting, 7-10 days", so 168 h is the shortest of the stated range and therefore the
conservative choice, and it is still overridable.

**The novelty claim, worded as the evidence supports.** The literature track searched
for prior work treating a weekly Legionella cycle as a deadline constraint in
reinforcement learning and found none (``docs/LITERATURE.md``). It also struck one of
the two citations the earlier report leaned on: [reyespremer2025model] is model
predictive control for a 120 V heat-pump water heater and **does not mention
Legionella at all**. [engelbrecht2021optimal] is confirmed verbatim -- a field study of
77 water heaters reporting median savings of "6.3% for temperature-matching, 21.9% for
energy-matching and 16.2% for energy-matching with Legionella prevention" -- but its
method is A* search optimal control, not reinforcement learning, and the abstract
supports neither "daily" nor "as a temperature constraint". So: *no reinforcement-
learning treatment of a weekly Legionella cycle as a deadline constraint was found.
That is an absence of evidence from one search, not a proven absence.* The claim is
**plausible and not established**, and must be written that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.flexibility import FlexibleLoad

__all__ = [
    "LegionellaSpec",
    "ShadowTank",
    "LegionellaCycleBarrier",
    "LegionellaStack",
    "HOURS_PER_WEEK",
]

#: The shortest interval in the range the design note reports for new units
#: ("7-10 days"), and therefore the conservative window. Overridable.
HOURS_PER_WEEK = 168.0

_UNSOURCED = "UNSOURCED"


@dataclass
class LegionellaSpec:
    """Every parameter of the cycle, each with its provenance.

    No field that the design note lists as an open question has a default. Construct
    with explicit values and record where each came from in ``provenance``; anything
    left as ``"UNSOURCED"`` is reported by ``unsourced`` so a run record carries it.

    Parameters
    ----------
    capacity_kwh, heat_source_kw, loss_coefficient
        ``C``, ``P_hp`` and the standing loss, per building.
    t_cold_c, t_rated_c, t_normal_c, t_legionella_c
        The four temperatures, degC. ``t_rated_c`` is the temperature at which the
        tank's rated capacity ``C`` is defined -- it is what makes ``soc`` a
        temperature, and getting it wrong rescales every level in the model.
    cop_legionella, element_efficiency, t_heat_pump_max_c
        The two unit variants of the design note. A heat pump lifts the store up to
        ``t_heat_pump_max_c`` at ``cop_legionella``; above that an electric element
        finishes the job at ``element_efficiency``. Set ``t_heat_pump_max_c`` at or
        above ``t_legionella_c`` for the propane variant that reaches the set point on
        the heat pump alone.
    """

    capacity_kwh: np.ndarray
    heat_source_kw: np.ndarray
    t_cold_c: float
    t_rated_c: float
    t_normal_c: float
    t_legionella_c: float
    cop_legionella: float
    element_efficiency: float
    t_heat_pump_max_c: float
    loss_coefficient: np.ndarray = 0.0
    #: Hot-water draw, kWh per step, reserved against while the cycle is running.
    #: ``0.0`` -- the default -- means the deadline is guaranteed only against a *zero*
    #: draw, which is the honest default because nothing in this repository forecasts
    #: the draw during the disinfection hours. See `LegionellaCycleBarrier.current_rate`
    #: for why a reserve is needed at all and what setting it buys.
    draw_margin_kwh: np.ndarray = 0.0
    period_hours: float = HOURS_PER_WEEK
    dt_hours: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)

    #: Fields the design note lists as open questions. A value for any of these that
    #: is not recorded in ``provenance`` is reported by ``unsourced``. ClassVar, so it
    #: does not become a constructor argument.
    OPEN_QUESTIONS: ClassVar[Tuple[str, ...]] = (
        "heat_source_kw", "t_cold_c", "t_rated_c", "t_normal_c", "t_legionella_c",
        "cop_legionella", "element_efficiency", "t_heat_pump_max_c")

    def __post_init__(self) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity_kwh = f(self.capacity_kwh)
        self.B = self.capacity_kwh.size
        self.heat_source_kw = np.broadcast_to(f(self.heat_source_kw), (self.B,)).copy()
        self.loss_coefficient = np.broadcast_to(f(self.loss_coefficient),
                                                (self.B,)).copy()
        self.draw_margin_kwh = np.broadcast_to(f(self.draw_margin_kwh),
                                               (self.B,)).copy()
        if self.t_rated_c <= self.t_cold_c:
            raise ValueError(
                f"t_rated_c ({self.t_rated_c}) must exceed t_cold_c ({self.t_cold_c}): "
                "the rated capacity is the energy between the inlet and the rated "
                "temperature, and a non-positive span makes every state of charge "
                "meaningless.")
        if self.t_legionella_c <= self.t_normal_c:
            raise ValueError(
                f"t_legionella_c ({self.t_legionella_c}) must exceed t_normal_c "
                f"({self.t_normal_c}); a disinfection set point at or below the normal "
                "one is satisfied by doing nothing, and the constraint would be vacuous.")
        if not np.all(self.heat_source_kw > 0.0):
            raise ValueError("heat_source_kw must be positive: a store with no source "
                             "can never reach its required level.")
        if np.any(self.draw_margin_kwh < 0.0):
            raise ValueError("draw_margin_kwh must be non-negative")
        if np.any(self.draw_margin_kwh >= self.heat_source_kw * self.dt_hours):
            raise ValueError(
                "draw_margin_kwh is at or above the source's output per step, so the "
                "reserved draw alone consumes everything the source can deliver and "
                "the cycle can never progress. Either the source is undersized for "
                "this draw or the reserve is wrong.")
        if self.period_hours < self.dt_hours:
            raise ValueError(f"period_hours ({self.period_hours}) is shorter than one "
                             f"step ({self.dt_hours})")
        self.provenance = {k: str(v) for k, v in dict(self.provenance).items()}

    # -- temperatures and levels ------------------------------------------------

    def soc_of_temperature(self, t_c: float) -> float:
        """State of charge corresponding to a uniform store temperature.

        Linear in temperature above the cold inlet, which assumes a **fully mixed**
        tank. A real cylinder stratifies, so the energy needed to bring the whole
        volume to the disinfection temperature is understated by this model whenever
        the draw profile has left a cold bottom layer. Stated rather than modelled:
        a stratified model needs node temperatures the data here does not have.
        """
        return float(np.clip((t_c - self.t_cold_c) / (self.t_rated_c - self.t_cold_c),
                             0.0, 1.0))

    @property
    def soc_legionella(self) -> float:
        return self.soc_of_temperature(self.t_legionella_c)

    @property
    def soc_normal(self) -> float:
        return self.soc_of_temperature(self.t_normal_c)

    @property
    def soc_heat_pump_max(self) -> float:
        return self.soc_of_temperature(self.t_heat_pump_max_c)

    @property
    def rate(self) -> np.ndarray:
        """Gross ``soc`` gained per step at the source's power limit."""
        return (self.heat_source_kw * self.dt_hours
                / np.maximum(self.capacity_kwh, 1e-9))

    @property
    def net_rate(self) -> np.ndarray:
        """``soc`` gained per step after the standing loss and the reserved draw.

        The standing loss is deducted at ``soc_legionella``, the level the deadline
        owes, because the loss grows with the stored energy and the worst case over
        the run-up is the level at the end of it.
        """
        gross = (self.heat_source_kw * self.dt_hours - self.draw_margin_kwh)
        return np.maximum(gross / np.maximum(self.capacity_kwh, 1e-9)
                          - self.loss_coefficient * self.soc_legionella, 1e-6)

    @property
    def period_steps(self) -> float:
        return float(self.period_hours / self.dt_hours)

    @property
    def unsourced(self) -> List[str]:
        """Parameters carrying a value that nothing in this repository sources.

        The design note's open questions, plus ``draw_margin_kwh`` whenever it is
        non-zero. A zero reserve is not an unsourced number -- it is the declaration
        that no reserve is being held, which is honest and is the default. A non-zero
        one is a claim about the hot-water draw during the disinfection hours, and
        nothing here forecasts that.
        """
        out = [k for k in self.OPEN_QUESTIONS
               if self.provenance.get(k, _UNSOURCED) == _UNSOURCED]
        if (np.any(self.draw_margin_kwh > 0.0)
                and self.provenance.get("draw_margin_kwh", _UNSOURCED) == _UNSOURCED):
            out.append("draw_margin_kwh")
        return out

    def summary(self) -> Dict[str, Any]:
        return {"capacity_kwh": [round(float(c), 4) for c in self.capacity_kwh],
                "heat_source_kw": [round(float(p), 4) for p in self.heat_source_kw],
                "t_cold_c": self.t_cold_c, "t_rated_c": self.t_rated_c,
                "t_normal_c": self.t_normal_c, "t_legionella_c": self.t_legionella_c,
                "soc_normal": round(self.soc_normal, 4),
                "soc_legionella": round(self.soc_legionella, 4),
                "soc_heat_pump_max": round(self.soc_heat_pump_max, 4),
                "cop_legionella": self.cop_legionella,
                "element_efficiency": self.element_efficiency,
                "period_hours": self.period_hours,
                "rate_soc_per_step": [round(float(r), 5) for r in self.rate],
                "net_rate_soc_per_step": [round(float(r), 5) for r in self.net_rate],
                "draw_margin_kwh": [round(float(d), 4) for d in self.draw_margin_kwh],
                "hours_to_disinfect_from_normal": [
                    round(float((self.soc_legionella - self.soc_normal) / r), 3)
                    for r in self.net_rate],
                "unsourced_parameters": self.unsourced,
                "provenance": dict(self.provenance)}


class ShadowTank:
    """A hot-water store whose heat source is power-limited, so draws can go unmet.

    One step, in the order the design note writes it::

        E_avail = max(E * (1 - loss), 0)
        heat_in = clip(requested, 0, P_hp * dt)       # the power limit, the whole point
        served  = min(draw, E_avail + heat_in)
        unmet   = draw - served                       # the service failure
        E_next  = min(E_avail + heat_in - served, C)
        spill   = (E_avail + heat_in - served) - E_next

    ``heat_in`` is capped by power only, not by the headroom before the draw, because
    the source and the draw act over the same step; ``spill`` reports heat the store
    could not hold so the energy balance stays closed and visible.
    """

    def __init__(self, spec: LegionellaSpec, soc0: Optional[np.ndarray] = None) -> None:
        self.spec = spec
        self.B = spec.B
        self.reset(soc0)

    def reset(self, soc0: Optional[np.ndarray] = None) -> None:
        start = (np.full(self.B, self.spec.soc_normal) if soc0 is None
                 else np.clip(np.asarray(soc0, dtype=np.float64).reshape(-1), 0.0, 1.0))
        self.E = start * self.spec.capacity_kwh
        self.steps = 0
        self.unmet_kwh = np.zeros(self.B)
        self.demand_kwh = np.zeros(self.B)
        self.spill_kwh = np.zeros(self.B)
        self.heat_kwh = np.zeros(self.B)
        self.electricity_kwh = np.zeros(self.B)
        self.unmet_events = np.zeros(self.B, dtype=np.int64)
        self.draw_events = np.zeros(self.B, dtype=np.int64)

    @property
    def soc(self) -> np.ndarray:
        return self.E / np.maximum(self.spec.capacity_kwh, 1e-9)

    def heat_for_action(self, action: np.ndarray) -> np.ndarray:
        """Thermal energy a non-negative action commands, kWh for the step."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), 0.0, 1.0)
        return a * self.spec.heat_source_kw * self.spec.dt_hours

    def electricity_for_heat(self, heat_kwh: np.ndarray,
                             soc_before: np.ndarray) -> np.ndarray:
        """Electricity the lift costs, splitting at the heat pump's temperature ceiling.

        Below ``soc_heat_pump_max`` the heat pump supplies the heat at
        ``cop_legionella``; above it an electric element finishes at
        ``element_efficiency``. Setting ``t_heat_pump_max_c >= t_legionella_c`` gives
        the propane variant of the design note, where the heat pump reaches the set
        point alone.

        **``cop_legionella`` is an unsourced parameter and this split is a modelling
        choice, not a measurement.** A real unit's coefficient of performance falls
        continuously with sink temperature rather than stepping at a ceiling; the step
        is the coarsest model that distinguishes the two unit generations the design
        note names.
        """
        heat = np.maximum(np.asarray(heat_kwh, dtype=np.float64).reshape(-1), 0.0)
        cap = self.spec.capacity_kwh
        ceiling = self.spec.soc_heat_pump_max * cap
        e0 = np.clip(np.asarray(soc_before, dtype=np.float64).reshape(-1), 0.0, 1.0) * cap
        by_pump = np.clip(ceiling - e0, 0.0, heat)
        by_element = heat - by_pump
        return (by_pump / max(self.spec.cop_legionella, 1e-6)
                + by_element / max(self.spec.element_efficiency, 1e-6))

    def step(self, heat_requested_kwh: np.ndarray,
             draw_kwh: np.ndarray) -> Dict[str, np.ndarray]:
        s = self.spec
        draw = np.maximum(np.asarray(draw_kwh, dtype=np.float64).reshape(-1), 0.0)
        soc_before = self.soc.copy()
        e_avail = np.maximum(self.E * (1.0 - s.loss_coefficient), 0.0)
        heat_in = np.clip(np.asarray(heat_requested_kwh, dtype=np.float64).reshape(-1),
                          0.0, s.heat_source_kw * s.dt_hours)
        served = np.minimum(draw, e_avail + heat_in)
        unmet = draw - served
        raw = e_avail + heat_in - served
        e_next = np.minimum(raw, s.capacity_kwh)
        spill = raw - e_next
        electricity = self.electricity_for_heat(heat_in, soc_before)

        self.E = e_next
        self.steps += 1
        self.unmet_kwh += unmet
        self.demand_kwh += draw
        self.spill_kwh += spill
        self.heat_kwh += heat_in
        self.electricity_kwh += electricity
        self.unmet_events += (unmet > 1e-9).astype(np.int64)
        self.draw_events += (draw > 1e-9).astype(np.int64)
        return {"soc": self.soc.copy(), "unmet_kwh": unmet, "served_kwh": served,
                "heat_in_kwh": heat_in, "spill_kwh": spill,
                "electricity_kwh": electricity}

    def report(self) -> Dict[str, float]:
        """The hot-water service KPIs. ``unmet`` is the headline, not the state of charge."""
        demand = float(self.demand_kwh.sum())
        draws = int(self.draw_events.sum())
        return {"unmet_kwh": float(self.unmet_kwh.sum()),
                "demand_kwh": demand,
                "unmet_share_of_demand": (float(self.unmet_kwh.sum()) / demand
                                          if demand > 0 else 0.0),
                "unmet_draws": int(self.unmet_events.sum()),
                "draws": draws,
                "unmet_share_of_draws": (float(self.unmet_events.sum()) / draws
                                         if draws else 0.0),
                "heat_kwh": float(self.heat_kwh.sum()),
                "electricity_kwh": float(self.electricity_kwh.sum()),
                "spill_kwh": float(self.spill_kwh.sum()),
                "steps": int(self.steps)}


class LegionellaCycleBarrier(DeadlineStorageBarrier):
    """The weekly disinfection cycle, as a deadline on the hot-water store.

    Identical machinery to the electric-vehicle departure deadline, with three
    differences and no new mechanism: the required level is a temperature rather than
    a contract, the deadline recurs instead of arriving once, and the window resets
    when the level is *reached* rather than when a vehicle leaves.

    The barrier is stateful -- it has to be, because "once per window" is a property of
    the history, not of the current observation. ``observe()`` advances the window
    after each environment step and resets it when the store has reached
    ``soc_legionella``; ``cycles_completed`` and ``missed_windows`` are the audit trail.
    """

    def __init__(self, spec: LegionellaSpec, tank: ShadowTank, action_index: int,
                 action_bound: Optional[np.ndarray] = None, margin: float = 0.0,
                 name: str = "legionella") -> None:
        self.spec = spec
        self.tank = tank
        self.B = spec.B
        self.steps_since_cycle = np.zeros(self.B, dtype=np.float64)
        self.cycles_completed = np.zeros(self.B, dtype=np.int64)
        self.missed_windows = np.zeros(self.B, dtype=np.int64)
        bound = np.ones(self.B) if action_bound is None else action_bound
        super().__init__(
            rate=spec.net_rate,
            action_bound=bound,
            action_index=int(action_index),
            capacity=spec.capacity_kwh,
            # The store's own round-trip efficiency is 1 here: the coefficient of
            # performance is applied in `ShadowTank.electricity_for_heat`, where the
            # split between the heat pump and the element lives, and applying it twice
            # would understate the energy a deadline owes.
            efficiency=np.ones(self.B),
            requirement_fn=self._requirement,
            soc_fn=lambda _obs: self.tank.soc.astype(np.float32),
            margin=margin,
            soc_cap=1.0,
            name=name,
        )

    def current_rate(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        """``soc`` the source adds in one step, net of loss and the reserved draw.

        The base class computes ``steps_needed = ceil(gap / rate)`` and starts forcing
        when ``steps_to_deadline`` falls to that. With ``rate`` set to the *gross*
        source output, that estimate is optimistic in two ways a hot-water store feels
        acutely: the tank loses heat while it is being charged, and -- unlike an
        electric vehicle, which is not driven while plugged in -- it is being *drawn
        from* at the same time. Both consume part of the source's output, so the
        barrier starts forcing later than it should and can arrive at the deadline
        short.

        The standing loss is known from the spec and is deducted exactly. The draw is
        not: nothing in this repository forecasts hot-water demand during the
        disinfection hours, so it is a declared reserve, ``draw_margin_kwh``, zero by
        default. **With a zero reserve the deadline is guaranteed only against a zero
        draw**, and `tests/test_legionella.py` pins both sides of that: a cycle that
        completes with the reserve set, and one that misses without it.
        """
        return self.rate

    def _requirement(self, obs_list: Sequence[np.ndarray]) -> DeadlineRequirement:
        left = np.maximum(self.spec.period_steps - self.steps_since_cycle, 0.0)
        return DeadlineRequirement(
            soc=np.full(self.B, self.spec.soc_legionella),
            steps_to_deadline=left,
            # Always active: the window always exists. Unlike a vehicle, a tank is
            # never unplugged.
            active=np.ones(self.B, dtype=bool))

    def observe(self, soc: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Advance the window by one step, and reset it where the cycle completed.

        Call once per environment step, *after* the step. ``soc`` defaults to the
        shadow tank's own state.
        """
        level = self.tank.soc if soc is None else np.asarray(soc, dtype=np.float64)
        done = level + 1e-9 >= self.spec.soc_legionella
        self.steps_since_cycle += 1.0
        expired = (~done) & (self.steps_since_cycle >= self.spec.period_steps)
        self.missed_windows += expired.astype(np.int64)
        self.cycles_completed += done.astype(np.int64)
        self.steps_since_cycle = np.where(done | expired, 0.0, self.steps_since_cycle)
        return {"completed": done, "expired": expired,
                "steps_since_cycle": self.steps_since_cycle.copy()}

    def report(self) -> Dict[str, Any]:
        return {"cycles_completed": [int(c) for c in self.cycles_completed],
                "missed_windows": [int(m) for m in self.missed_windows],
                "steps_since_cycle": [float(s) for s in self.steps_since_cycle],
                "soc_legionella": self.spec.soc_legionella,
                "period_steps": self.spec.period_steps}


@dataclass
class LegionellaStack:
    """The shadow tank, the cycle barrier, and the flexible load that carries them.

    The whole point of the framework is that this is three lines of assembly and no new
    enforcement path: the resulting ``FlexibleLoad`` goes into the same
    ``FlexibilityPortfolio`` as the battery, the tank and the vehicle, and contends for
    the same cap.
    """

    spec: LegionellaSpec
    tank: ShadowTank
    barrier: LegionellaCycleBarrier
    load: FlexibleLoad

    @classmethod
    def build(cls, spec: LegionellaSpec, action_index: int,
              action_bound: Optional[np.ndarray] = None,
              soc0: Optional[np.ndarray] = None) -> "LegionellaStack":
        tank = ShadowTank(spec, soc0=soc0)
        barrier = LegionellaCycleBarrier(spec, tank, action_index,
                                         action_bound=action_bound)
        load = FlexibleLoad(
            name="legionella", kind="legionella", action_index=action_index,
            barrier=barrier, capacity_kwh=spec.capacity_kwh,
            power_kw=spec.heat_source_kw,
            action_bound=float(np.max(barrier.action_bound)),
            notes=("weekly disinfection deadline on the hot-water store; unsourced "
                   f"parameters: {spec.unsourced or 'none'}"))
        return cls(spec=spec, tank=tank, barrier=barrier, load=load)

    def observe(self, action: np.ndarray, draw_kwh: np.ndarray) -> Dict[str, np.ndarray]:
        """Advance the shadow tank and the window by one step.

        ``action`` is this load's action column *after* projection, i.e. what the
        controller actually commanded.
        """
        out = self.tank.step(self.tank.heat_for_action(action), draw_kwh)
        out.update(self.barrier.observe())
        return out

    def report(self) -> Dict[str, Any]:
        return {"service": self.tank.report(), "cycle": self.barrier.report(),
                "spec": self.spec.summary()}
