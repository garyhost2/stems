"""Battery and electric-vehicle capacity fade, as a cost and as a constraint.

The repository already counts ``battery_equivalent_full_cycles`` and never prices it.
This module turns that count into a quantity with units -- kWh of *lost storage
capacity* -- so it can enter the reward in the same currency as the electricity bill and
enter the constraint set in the same form as the state-of-charge band.

Honest placement in the field, from ``docs/LITERATURE.md``: in every source the
literature track retrieved, degradation is a **scored KPI, never an enforced
constraint**; Khouja et al. (2026) propose "battery storage lifetime" explicitly as a
*novel* KPI, which is evidence that it was not standard and that it is scored rather
than enforced. Treating it as a constraint is therefore ahead of the field, not a
reproduction of it, and must be presented that way. The audit's one citation for a
vehicle-to-grid degradation term (Khezri et al. 2024) did not resolve against Crossref
and was struck (``docs/LITERATURE.md`` section 8.1); it is not cited here.

Symbols, SI units:

=========================  =====================================================  ========
symbol                     meaning                                                unit
=========================  =====================================================  ========
``Q_rated``                rated (nameplate) storage capacity                     kWh
``Q_t``                    current capacity after fade, at step t                 kWh
``E_t``                    signed energy through the cell over the step           kWh
``soc_t``                  state of charge, ``Q_stored / Q_t``                    1
``delta``                  depth of a closed half cycle, ``|soc_max - soc_min|``  1
``kappa``                  capacity-loss coefficient per equivalent full cycle    1
``p``                      depth-of-discharge stress exponent                     1
``c_cal``                  calendar fade, fraction of ``Q_rated`` per second      1/s
``dQ_t``                   capacity lost over the step                            kWh
``pi_deg``                 price of one kWh of lost capacity                      currency
=========================  =====================================================  ========

Model, term by term, with the source of each parameter.

**Throughput.** Identical to the degradation CityLearn itself applies, so the reward
prices exactly what the plant does rather than a second, disagreeing model::

    dQ_thr,t = kappa * Q_rated * |E_t| / (2 * Q_t)                              (1)

Source: CityLearn 2.6.0b1, ``citylearn/energy_model.py::Battery.degrade``, which returns
``capacity_loss_coefficient * capacity * abs(energy_balance[t]) / (2 * degraded_capacity)``.
``kappa`` is read per building off the simulator's own ``Battery`` objects; CityLearn's
documented default is sampled from the range ``(1e-5, 1e-4)`` (same file, ``Battery``
docstring and ``capacity_loss_coefficient`` setter). The factor two makes ``kappa`` the
fade per *equivalent full cycle*, since one such cycle is ``2 * Q_rated`` of throughput.

**Depth of discharge.** Empirically, cycle life ``N`` falls with the depth of the cycle,
so fade per unit of throughput rises with depth. Writing ``N(delta) = N_ref *
delta^-(p+1)`` gives a fade-per-throughput multiplier ``delta^p``::

    dQ_dod over a closed half cycle = delta^p * (throughput fade accumulated in it)   (2)

``p = DegradationConfig.dod_exponent`` and its **default is 0.0**, which reproduces (1)
exactly. That default is deliberate: no value of ``p`` is asserted here, because no
value was verified. The standard empirical reference for the shape is Xu, Oudalov,
Ulbig, Andersson and Kirschen, "Modeling of Lithium-Ion Battery Degradation for Cell
Life Assessment", IEEE Transactions on Smart Grid, DOI ``10.1109/TSG.2016.2578950``
(title and DOI resolved through Crossref; the paper is closed access and its fitted
stress-function coefficients were **not** read, so none are reproduced). Setting ``p``
to a non-zero value is a modelling choice the user must source.

Half cycles are closed by a reversal detector, not by rainflow counting: the running
state-of-charge extremum is tracked, and when the direction of travel reverses the
excursion since the previous extremum is closed with depth ``delta``. This is the
online, causal approximation to rainflow; it charges each unit of throughput exactly
once, so (2) can never double-count against (1).

**Calendar.** ``dQ_cal,t = c_cal * Q_rated * dt``, with ``c_cal`` in 1/s. There is no
default: calendar fade depends on chemistry, temperature and mean state of charge, none
of which the simulator reports, so it must be supplied or the term stays off.

**Pricing.** ``dCost_t = pi_deg * dQ_t``, in the same currency per kWh as the tariff, so
the degradation cost and the bill are additive. ``pi_deg`` defaults to 0.0 -- fade is
measured and reported but does not enter the reward -- because a replacement cost is a
market number that belongs to the scenario, not to the code.

**Constraint.** The episode is given a capacity-loss budget ``L`` (kWh per building).
The per-step cost signal is the indicator ``1{dQ_t > L / T}`` for an episode of ``T``
steps, i.e. the step spent capacity faster than the budget's uniform rate. That form
matches the three existing cost channels exactly, which is what lets the same cost
critics carry it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

__all__ = ["DegradationModel", "DegradationAccountant", "CITYLEARN_KAPPA_RANGE"]

#: CityLearn 2.6.0b1's documented default range for ``capacity_loss_coefficient``
#: (``citylearn/energy_model.py``, ``Battery.capacity_loss_coefficient`` setter).
CITYLEARN_KAPPA_RANGE = (1e-5, 1e-4)

MODES = ("none", "throughput", "throughput+dod")


@dataclass(frozen=True)
class DegradationModel:
    """Parameters of the fade model, per building. All SI, all sourced in the module docstring.

    ``kappa`` is dimensionless fade per equivalent full cycle; ``capacity_kwh`` is
    ``Q_rated``; ``dod_exponent`` is ``p``; ``calendar_loss_per_second`` is ``c_cal``
    (1/s) or ``None`` for off; ``price_per_kwh`` is ``pi_deg``.
    """

    kappa: np.ndarray
    capacity_kwh: np.ndarray
    mode: str = "throughput"
    dod_exponent: float = 0.0
    calendar_loss_per_second: Optional[float] = None
    price_per_kwh: float = 0.0
    dt_s: float = 3600.0
    #: Range gate on the half-cycle detector, in state-of-charge units. A retracement
    #: smaller than this does not close a half cycle. 0.05 is the repository's choice,
    #: not a measured value; it is reported in every run record so a result that
    #: depends on it can be seen to.
    reversal_threshold: float = 0.05
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "kappa", f(self.kappa))
        object.__setattr__(self, "capacity_kwh", f(self.capacity_kwh))
        if self.kappa.shape != self.capacity_kwh.shape:
            raise ValueError("kappa and capacity_kwh need one entry per building")
        if np.any(self.kappa < 0.0):
            raise ValueError("kappa must be non-negative")
        if self.dod_exponent < 0.0:
            raise ValueError("dod_exponent p must be non-negative: a deeper cycle "
                             "cannot cost less per kWh than a shallow one")
        if self.mode == "throughput+dod" and self.dod_exponent == 0.0:
            # Not an error: p = 0 is the honest default and reduces (2) to (1). Flagged
            # so a run record cannot claim a depth-of-discharge effect it did not apply.
            pass

    @property
    def B(self) -> int:
        return int(self.kappa.size)

    @classmethod
    def from_environment(cls, env, cfg) -> "DegradationModel":
        """Read ``kappa`` and ``Q_rated`` off the simulator's own battery objects."""
        info = env.battery_degradation_info()
        return cls(kappa=info["capacity_loss_coefficient"],
                   capacity_kwh=info["capacity"],
                   mode=cfg.mode,
                   dod_exponent=float(cfg.dod_exponent),
                   reversal_threshold=float(cfg.reversal_threshold),
                   calendar_loss_per_second=(cfg.calendar_loss_per_second
                                             if cfg.calendar else None),
                   price_per_kwh=float(cfg.price_eur_per_kwh),
                   dt_s=float(info["seconds_per_time_step"]),
                   source=str(info["source"]))

    def describe(self) -> Dict[str, object]:
        return {"mode": self.mode, "kappa": self.kappa.tolist(),
                "capacity_kwh": self.capacity_kwh.tolist(),
                "dod_exponent": self.dod_exponent,
                "reversal_threshold": self.reversal_threshold,
                "calendar_loss_per_second": self.calendar_loss_per_second,
                "price_per_kwh": self.price_per_kwh, "dt_s": self.dt_s,
                "source": self.source}


class DegradationAccountant:
    """Online capacity-fade accounting from state-of-charge transitions.

    One instance per consumer of the signal (the training loop's reward and cost, the
    KPI calculator) -- they must not share state, because each sees a different number
    of steps.

    ``step(soc_prev, soc_next)`` returns the per-building capacity lost over the step,
    in kWh, and updates the half-cycle detector. Energy through the cell is taken as
    ``|soc_next - soc_prev| * Q_t``, which is the cell-side throughput the simulator's
    own ``energy_balance`` records for a lossless accounting of the stored energy; the
    efficiency losses sit outside the cell and are already priced as electricity.
    """

    def __init__(self, model: DegradationModel, limit_kwh_per_episode: Optional[float] = None,
                 episode_steps: Optional[int] = None) -> None:
        self.model = model
        B = model.B
        self.limit_kwh_per_episode = (None if limit_kwh_per_episode is None
                                      else float(limit_kwh_per_episode))
        self.episode_steps = None if episode_steps is None else max(int(episode_steps), 1)
        self._B = B
        self.reset()

    def reset(self) -> None:
        m = self.model
        self.capacity_kwh = m.capacity_kwh.astype(np.float64).copy()
        self.cumulative_loss_kwh = np.zeros(self._B, dtype=np.float64)
        self.throughput_kwh = np.zeros(self._B, dtype=np.float64)
        self._ref_soc: Optional[np.ndarray] = None   # start of the open excursion
        self._ext_soc: Optional[np.ndarray] = None   # extreme reached since _ref_soc
        self._direction = np.zeros(self._B, dtype=np.int8)
        self._pending_fade = np.zeros(self._B, dtype=np.float64)
        self.half_cycles = np.zeros(self._B, dtype=np.int64)
        self.mean_half_cycle_depth = np.zeros(self._B, dtype=np.float64)
        self._depth_sum = np.zeros(self._B, dtype=np.float64)
        self.steps = 0

    @property
    def budget_per_step_kwh(self) -> Optional[float]:
        if self.limit_kwh_per_episode is None or self.episode_steps is None:
            return None
        return self.limit_kwh_per_episode / self.episode_steps

    def step(self, soc_prev: np.ndarray, soc_next: np.ndarray) -> np.ndarray:
        m = self.model
        if m.mode == "none":
            self.steps += 1
            return np.zeros(self._B, dtype=np.float64)
        soc_prev = np.asarray(soc_prev, dtype=np.float64).reshape(-1)
        soc_next = np.asarray(soc_next, dtype=np.float64).reshape(-1)
        d_soc = soc_next - soc_prev
        energy_kwh = np.abs(d_soc) * self.capacity_kwh
        self.throughput_kwh += energy_kwh

        # (1) throughput fade, CityLearn's own expression.
        fade = (m.kappa * m.capacity_kwh * energy_kwh
                / (2.0 * np.maximum(self.capacity_kwh, 1e-9)))

        if m.mode == "throughput+dod":
            fade = self._accrue_half_cycle(soc_prev, soc_next, fade)

        if m.calendar_loss_per_second is not None:
            fade = fade + m.calendar_loss_per_second * m.capacity_kwh * m.dt_s

        self.capacity_kwh = np.maximum(self.capacity_kwh - fade, 0.0)
        self.cumulative_loss_kwh += fade
        self.steps += 1
        return fade

    def _accrue_half_cycle(self, soc_prev: np.ndarray, soc_next: np.ndarray,
                           fade: np.ndarray) -> np.ndarray:
        """Hold this step's fade until the half cycle closes, then scale it by delta^p.

        Half cycles are closed by a **hysteresis-filtered reversal detector**, the
        standard range gate applied ahead of rainflow counting. Each building carries
        the state of charge at which the open excursion started (``ref``) and the
        extreme reached since (``ext``). A retracement from ``ext`` smaller than
        ``reversal_threshold`` does not close anything -- it is noise on the way to a
        larger swing -- while a larger one closes a half cycle of depth
        ``|ext - ref|``.

        Without the gate this detector closes a half cycle on every sign change of the
        state of charge, so a policy that jitters produces a long run of micro-cycles of
        depth ~1e-4 and, with ``p > 0``, is charged almost nothing: the depth term then
        *rewards* jitter, which is the opposite of what it is for. The gate is what
        makes the depth term behave; it is an approximation to rainflow, not rainflow,
        and a run that leans on it should say so.

        Returns the fade committed at this step. Every unit of throughput is committed
        exactly once, so (2) cannot double-count against (1).
        """
        m = self.model
        if self._ref_soc is None:
            self._ref_soc = soc_prev.copy()
            self._ext_soc = soc_prev.copy()
        thr = float(m.reversal_threshold)
        ext = self._ext_soc
        d = self._direction

        step_dir = np.sign(soc_next - soc_prev).astype(np.int8)
        starting = (d == 0) & (np.abs(soc_next - self._ref_soc) > thr)
        d = np.where(starting, step_dir, d).astype(np.int8)
        ext = np.where(starting, soc_next, ext)

        extending = (d != 0) & ((soc_next - ext) * d > 0.0)
        ext = np.where(extending, soc_next, ext)

        retracement = (ext - soc_next) * d
        closing = (d != 0) & ~extending & (retracement > thr)

        committed = np.zeros(self._B, dtype=np.float64)
        if np.any(closing):
            depth = np.abs(ext - self._ref_soc)
            scale = np.power(np.clip(depth, 1e-9, 1.0), m.dod_exponent)
            committed = np.where(closing, self._pending_fade * scale, 0.0)
            self._pending_fade = np.where(closing, 0.0, self._pending_fade)
            self._ref_soc = np.where(closing, ext, self._ref_soc)
            self.half_cycles += closing.astype(np.int64)
            self._depth_sum += np.where(closing, depth, 0.0)
            self.mean_half_cycle_depth = np.where(
                self.half_cycles > 0,
                self._depth_sum / np.maximum(self.half_cycles, 1), 0.0)
            ext = np.where(closing, soc_next, ext)
            d = np.where(closing, -d, d).astype(np.int8)

        self._ext_soc = ext
        self._direction = d
        self._pending_fade += fade
        return committed

    def flush(self) -> np.ndarray:
        """Close the open half cycle at the end of an episode and commit its fade.

        The open excursion is charged at its depth so far. Leaving it uncommitted would
        silently drop the fade of whatever the policy was doing when the episode ended,
        which on a short window is a large fraction of the total.
        """
        if self.model.mode != "throughput+dod" or self._ref_soc is None:
            return np.zeros(self._B, dtype=np.float64)
        depth = np.abs(self._ext_soc - self._ref_soc)
        scale = np.power(np.clip(depth, 1e-9, 1.0), self.model.dod_exponent)
        pending = self._pending_fade * scale
        self._pending_fade = np.zeros(self._B, dtype=np.float64)
        self.capacity_kwh = np.maximum(self.capacity_kwh - pending, 0.0)
        self.cumulative_loss_kwh += pending
        return pending

    def constraint_cost(self, fade_kwh: np.ndarray) -> np.ndarray:
        """Per-building 0/1 indicator that this step overspent the fade budget."""
        budget = self.budget_per_step_kwh
        if budget is None:
            return np.zeros(self._B, dtype=np.float32)
        return (np.asarray(fade_kwh, dtype=np.float64) > budget).astype(np.float32)

    def reward_penalty(self, fade_kwh: np.ndarray) -> np.ndarray:
        """Monetary cost of this step's fade, same currency per kWh as the tariff."""
        return (self.model.price_per_kwh
                * np.asarray(fade_kwh, dtype=np.float64)).astype(np.float64)

    def kpis(self, prefix: str = "battery") -> Dict[str, float]:
        m = self.model
        rated = float(m.capacity_kwh.sum())
        lost = float(self.cumulative_loss_kwh.sum())
        out = {
            f"{prefix}_capacity_loss_kwh": lost,
            f"{prefix}_capacity_loss_fraction": (lost / rated if rated > 1e-9
                                                 else float("nan")),
            f"{prefix}_throughput_kwh": float(self.throughput_kwh.sum()),
            f"{prefix}_degradation_cost": float(m.price_per_kwh * lost),
            f"{prefix}_capacity_loss_kwh_worst_building": float(
                self.cumulative_loss_kwh.max()) if self._B else float("nan"),
        }
        if m.mode == "throughput+dod":
            out[f"{prefix}_half_cycles"] = float(self.half_cycles.mean())
            out[f"{prefix}_mean_half_cycle_depth"] = float(self.mean_half_cycle_depth.mean())
        return out
