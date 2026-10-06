from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np


def _bisect_toward_zero(f, target: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                        iters: int = 40, tol: float = 1e-9) -> np.ndarray:
    """Smallest-magnitude ``a`` in ``[lo, hi]`` with ``f(a) == target``, ``f`` non-decreasing.

    Bisects on the side of zero the target lies on, so a target that idling already
    achieves returns exactly zero rather than an arbitrary point of a flat region. An
    unreachable target clamps to the nearer endpoint.
    """
    zero = np.zeros_like(lo)
    at_zero = f(zero)
    need_up = target > at_zero + tol
    need_down = target < at_zero - tol

    left_up, right_up = zero.copy(), hi.copy()
    for _ in range(iters):
        mid = 0.5 * (left_up + right_up)
        below = f(mid) < target
        left_up = np.where(below, mid, left_up)
        right_up = np.where(below, right_up, mid)

    left_dn, right_dn = lo.copy(), zero.copy()
    for _ in range(iters):
        mid = 0.5 * (left_dn + right_dn)
        above = f(mid) > target
        right_dn = np.where(above, mid, right_dn)
        left_dn = np.where(above, left_dn, mid)

    out = np.where(need_up, right_up, np.where(need_down, left_dn, zero))
    return np.clip(out, lo, hi).astype(np.float64)


class BatteryModel:
    def __init__(self, capacity: np.ndarray, nominal_power: np.ndarray, loss: np.ndarray,
                 eta_curves: List[np.ndarray], power_curves: List[np.ndarray],
                 hours_per_step: float = 1.0) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity, self.nominal_power, self.loss = f(capacity), f(nominal_power), f(loss)
        self.B = self.capacity.size
        self.dt = float(hours_per_step)
        self._eta_x, self._eta_y = self._stack(eta_curves)
        self._pow_x, self._pow_y = self._stack(power_curves)

    @staticmethod
    def _stack(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        curves = [np.asarray(c, dtype=np.float64) for c in curves]
        if len({c.shape for c in curves}) != 1:
            raise ValueError("battery curves must have the same number of breakpoints")
        return np.stack([c[0] for c in curves]), np.stack([c[1] for c in curves])

    @classmethod
    def from_citylearn(cls, buildings: List[Any], seconds_per_time_step: float = 3600.0
                       ) -> "BatteryModel":
        es = [b.electrical_storage for b in buildings]
        return cls(capacity=[e.capacity for e in es],
                   nominal_power=[e.nominal_power for e in es],
                   loss=[e.loss_coefficient for e in es],
                   eta_curves=[np.asarray(e.power_efficiency_curve) for e in es],
                   power_curves=[np.asarray(e.capacity_power_curve) for e in es],
                   hours_per_step=seconds_per_time_step / 3600.0)

    @classmethod
    def linear(cls, soc_rate: np.ndarray) -> "BatteryModel":
        rate = np.asarray(soc_rate, dtype=np.float64).reshape(-1)
        flat = np.array([[0.0, 1.0], [1.0, 1.0]])
        return cls(capacity=np.ones_like(rate), nominal_power=rate, loss=np.zeros_like(rate),
                   eta_curves=[flat] * rate.size, power_curves=[flat] * rate.size)

    @staticmethod
    def _lookup(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
        le = x[:, None] <= xs
        idx = np.maximum(np.argmax(le, axis=1) - 1, 0)
        r = np.arange(xs.shape[0])
        x0, x1, y0, y1 = xs[r, idx], xs[r, idx + 1], ys[r, idx], ys[r, idx + 1]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def _step(self, soc: np.ndarray, action: np.ndarray):
        soc = np.asarray(soc, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        cap, p_nom = self.capacity, self.nominal_power
        e_init = np.maximum(soc * cap * (1.0 - self.loss), 0.0)
        max_power = p_nom * self._lookup(self._pow_x, self._pow_y, e_init / cap)
        requested = action * p_nom * self.dt
        eta = self._lookup(self._eta_x, self._eta_y,
                           np.minimum(np.abs(requested), max_power) / p_nom)
        root = np.sqrt(eta)
        charge = np.minimum(np.minimum(max_power, p_nom),
                            np.minimum(cap - e_init, np.maximum(requested, 0.0)))
        discharge = np.maximum(-max_power, np.minimum(requested, 0.0))
        e_final = np.where(requested >= 0.0,
                           np.minimum(e_init + charge * root, cap),
                           np.maximum(e_init + discharge / root, 0.0))
        terminal = np.where(requested >= 0.0, (e_final - e_init) / root,
                            (e_final - e_init) * root)
        return e_final / cap, terminal

    def next_soc(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._step(soc, action)[0]

    def accepted_kwh(self, soc: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._step(soc, action)[1]

    def action_for_soc(self, soc: np.ndarray, target: np.ndarray,
                       a_max: float = 1.0, iters: int = 40) -> np.ndarray:
        """Smallest-magnitude action that drives ``soc`` to ``target`` in one step.

        ``next_soc`` is non-decreasing in the action, so each side is a bisection on the
        exact model; an unreachable target clamps to the nearest endpoint. The
        model-predictive controller plans in state of charge and converts here, which
        makes its state trajectory exact instead of a linearisation of one -- the
        battery saturates as it fills, so a secant between the idle and full-charge
        corners understates the state of charge in between by up to 0.11 on this
        schema's devices, and the controller would then plan trajectories that leave
        the band it is supposed to respect.

        *Smallest magnitude* is load-bearing, not a tie-break nicety. ``next_soc`` is
        flat in the action wherever the device cannot move energy -- a battery already
        at its floor cannot discharge further, a tank with no draw cannot discharge at
        all -- and a plain bisection over the whole interval converges to whichever end
        of the flat region it started from, which is a full-power command that happens
        to change nothing in the model and a great deal in the plant's accounting.
        Bracketing around zero returns the idle command instead, which is the one that
        means what it says.
        """
        soc = np.asarray(soc, dtype=np.float64)
        target = np.broadcast_to(np.asarray(target, dtype=np.float64), soc.shape)
        return _bisect_toward_zero(lambda a: self.next_soc(soc, a), target,
                                   np.full(soc.shape, -a_max, dtype=np.float64),
                                   np.full(soc.shape, a_max, dtype=np.float64), iters)

    def safe_interval(self, soc: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      a_max: float = 1.0, iters: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        soc = np.asarray(soc, dtype=np.float64)
        lo, hi = np.broadcast_to(lo, soc.shape), np.broadcast_to(hi, soc.shape)
        top = self.next_soc(soc, np.full(self.B, a_max))
        bottom = self.next_soc(soc, np.full(self.B, -a_max))

        left, right = np.full(self.B, -a_max), np.full(self.B, a_max)
        for _ in range(iters):
            mid = 0.5 * (left + right)
            ok = self.next_soc(soc, mid) >= lo
            left, right = np.where(ok, left, mid), np.where(ok, mid, right)
        a_lo = np.where(bottom >= lo, -a_max, right)

        left, right = np.full(self.B, -a_max), np.full(self.B, a_max)
        for _ in range(iters):
            mid = 0.5 * (left + right)
            ok = self.next_soc(soc, mid) <= hi
            left, right = np.where(ok, mid, left), np.where(ok, right, mid)
        a_hi = np.where(top <= hi, a_max, left)

        a_lo = np.where(top < lo, a_max, a_lo)
        a_hi = np.where(top < lo, a_max, a_hi)
        a_lo = np.where(bottom > hi, -a_max, a_lo)
        a_hi = np.where(bottom > hi, -a_max, a_hi)
        crossed = a_lo > a_hi
        mid = 0.5 * (a_lo + a_hi)
        return (np.where(crossed, mid, a_lo).astype(np.float32),
                np.where(crossed, mid, a_hi).astype(np.float32))


class TankModel:
    def __init__(self, capacity, heater_power, heater_efficiency, storage_efficiency, loss,
                 hours_per_step: float = 1.0) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity, self.heater_power = f(capacity), f(heater_power)
        self.heater_efficiency, self.storage_efficiency = f(heater_efficiency), f(storage_efficiency)
        self.loss = f(loss)
        self.B = self.capacity.size
        self.dt = float(hours_per_step)

    @classmethod
    def from_citylearn(cls, buildings: List[Any], seconds_per_time_step: float = 3600.0
                       ) -> "TankModel":
        from citylearn.energy_model import ElectricHeater

        for b in buildings:
            if not isinstance(b.dhw_device, ElectricHeater):
                raise TypeError(f"{b.name}: TankModel covers an ElectricHeater, "
                                f"not {type(b.dhw_device).__name__}")
        return cls(capacity=[b.dhw_storage.capacity for b in buildings],
                   heater_power=[b.dhw_device.nominal_power for b in buildings],
                   heater_efficiency=[b.dhw_device.efficiency for b in buildings],
                   storage_efficiency=[b.dhw_storage.round_trip_efficiency for b in buildings],
                   loss=[b.dhw_storage.loss_coefficient for b in buildings],
                   hours_per_step=seconds_per_time_step / 3600.0)

    def _step(self, soc: np.ndarray, action: np.ndarray, demand: np.ndarray):
        """Shared core of ``drawn_kwh`` and ``next_soc``.

        Returns ``(electricity_kwh, stored_energy_kwh)``: the heater electricity the
        storage exchange costs (positive when charging, negative when discharging saves
        the heater work) and the tank energy content after the step. Factored out so the
        two public methods cannot drift apart -- the model-predictive controller needs
        both and they must describe the same tank.
        """
        soc = np.asarray(soc, dtype=np.float64)
        e = np.asarray(action, dtype=np.float64) * self.capacity * self.dt
        demand = np.maximum(np.asarray(demand, dtype=np.float64), 0.0)
        C, r = self.capacity, self.storage_efficiency
        e_init = np.maximum(soc * C * (1.0 - self.loss), 0.0)
        spare = np.maximum(self.heater_power * self.heater_efficiency * self.dt - demand, 0.0)
        heat_in = np.minimum(np.maximum(e, 0.0), spare)
        stored = np.minimum(e_init + heat_in * r, C) - e_init
        charge = np.maximum(stored, 0.0) / r / self.heater_efficiency
        heat_out = np.minimum(np.maximum(-e, 0.0), demand)
        e_discharged = np.maximum(e_init - heat_out / np.maximum(r, 1e-9), 0.0)
        discharge = (e_init - e_discharged) * r / self.heater_efficiency
        electricity = np.where(e >= 0.0, charge, -discharge)
        content = np.where(e >= 0.0, e_init + np.maximum(stored, 0.0), e_discharged)
        return electricity, content

    def drawn_kwh(self, soc: np.ndarray, action: np.ndarray, demand: np.ndarray) -> np.ndarray:
        return self._step(soc, action, demand)[0]

    def next_soc(self, soc: np.ndarray, action: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """State of charge after one step, same convention as ``BatteryModel.next_soc``.

        ``drawn_kwh`` already simulated the tank internally and then threw the state
        away; a receding-horizon controller has to propagate it, so it is returned here
        from the same arithmetic rather than from a second implementation.
        """
        return self._step(soc, action, demand)[1] / np.maximum(self.capacity, 1e-12)

    def action_for_soc(self, soc: np.ndarray, target: np.ndarray, demand: np.ndarray,
                       a_max: Optional[np.ndarray] = None, iters: int = 40) -> np.ndarray:
        """Smallest-magnitude tank action that drives ``soc`` to ``target``.

        Same role, and the same zero-bracketing, as ``BatteryModel.action_for_soc``.
        The tank makes the flat-region problem acute: ``next_soc`` discharges only as
        far as the hot-water draw allows, so with no draw it is *constant* for every
        non-positive action, and a bisection over the whole interval returns -1 -- a
        full discharge command, issued at every step, for a tank that cannot discharge.
        ``a_max`` is the schema's own upper bound on the hot-water action, which differs
        per building.
        """
        soc = np.asarray(soc, dtype=np.float64)
        target = np.broadcast_to(np.asarray(target, dtype=np.float64), soc.shape)
        hi_bound = (np.ones_like(soc) if a_max is None
                    else np.broadcast_to(np.asarray(a_max, dtype=np.float64), soc.shape))
        return _bisect_toward_zero(lambda a: self.next_soc(soc, a, demand), target,
                                   np.full(soc.shape, -1.0, dtype=np.float64),
                                   np.ascontiguousarray(hi_bound, dtype=np.float64),
                                   iters)
