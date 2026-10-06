"""Household-level benefit accounting: what a resident gets, in money, CO2, comfort
and grid exposure.

Every number this module produces is *per household per year*, because that is the unit
a resident and a utility programme designer both understand. The district-level KPIs in
:mod:`stems.metrics` answer a different question (is the controller comparable with
published CityLearn results) and are not interchangeable with these.

Symbols
-------
``B``        number of buildings (households) in the district, dimensionless.
``T``        number of simulation steps, dimensionless.
``h``        hours per simulation step, h. One hour throughout this repository.
``e_bt``     *signed* net electricity consumption of building ``b`` at step ``t``, kW.
             Negative means the building exports to the grid.
``i_bt``     household **import**, kW: ``i_bt = max(e_bt, 0)``. Exports are excluded,
             because a tariff charges for imported energy and credits exports under a
             separate and usually different rule.
``E_bt``     household import energy over the step, kWh: ``E_bt = i_bt * h``.
``p_t``      energy price, USD/kWh.
``c_t``      grid carbon intensity, kg CO2 per kWh of *consumed* electricity.
``P_cap``    district import cap, kW. A hard constraint elsewhere in this repository.
``theta_bt`` indoor dry-bulb air temperature of building ``b`` at step ``t``, degC.

Why there are no default tariffs or emission factors
----------------------------------------------------
A household-benefit claim is a claim about a specific utility and a specific grid. Both
the tariff and the emission factor are therefore **required arguments with no
defaults**. The named presets below each carry their source in the docstring; choosing
one is an explicit, citable modelling decision, and nothing in this module will invent
one for you.

The geography defect this module exists to make visible
-------------------------------------------------------
The building stock in ``citylearn_schemas/tx_travis_8b`` is ResStock Travis County,
Texas, driven by 2018 Travis County AMY weather. The Travis County CityLearn dataset
ships **no** pricing and **no** carbon-intensity file (verified: the dataset directory
contains only building CSVs, ``weather.csv``, ``weather.epw``, ``schema.json`` and
``dynamics_error_summary.csv``). ``setup_citylearn_8b.py`` therefore borrows both from
``citylearn_challenge_2022_phase_all``, whose neighbourhood is in Fontana, California
(Nweye et al., NeurIPS 2022 Competition Track: the 17 single-family buildings "were
based on data from a real-world zero net energy neighborhood in Fontana, California").

So every cost and emission number previously produced in this repository prices Texas
houses with a California time-of-use tariff and a California grid. That is fine for
comparability with the CityLearn literature and wrong as a statement about a Travis
County household. Both readings are supported here, side by side, and which one is in
use is recorded in :attr:`HouseholdCase.provenance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

#: Pounds to kilograms, exact by definition (NIST SP 811, 1 lb = 0.45359237 kg).
LB_TO_KG = 0.45359237

#: US average transmission-and-distribution loss factor used by the EPA Greenhouse Gas
#: Equivalencies Calculator to convert a *generated* emission rate into a *delivered*
#: one: the calculator divides by ``1 - 0.051``. Source: EPA, "Greenhouse Gas
#: Equivalencies Calculator - Calculations and References", electricity-used factor,
#: citing EIA 2022 State Electricity Profiles Table 10.
EPA_TD_LOSS_FRACTION = 0.051


class Tariff(Protocol):
    """Maps household import energy to a bill.

    Implementations receive ``import_kwh`` of shape ``(T, B)`` in kWh and the
    CityLearn ``month`` and ``hour`` columns of shape ``(T,)`` (both 1-indexed, as
    CityLearn stores them), and return per-household bill components in USD, each of
    shape ``(B,)``.
    """

    name: str
    source: str

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        ...


@dataclass(frozen=True)
class HourlyEnergyTariff:
    """A pure energy-only tariff given as an hourly price vector.

    ``price_usd_per_kwh`` has shape ``(T,)``. There is no fixed charge and no tier, so
    the total equals ``sum_t E_bt * p_t`` exactly -- which is also what
    :meth:`stems.metrics.MetricsCalculator.compute_all` reports as ``cost``, summed over
    buildings. A test pins that identity, so this class is the bridge between the
    district KPI and the household bill rather than a second opinion about it.
    """

    price_usd_per_kwh: np.ndarray
    name: str
    source: str

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(self.price_usd_per_kwh, dtype=np.float64)
        E = np.asarray(import_kwh, dtype=np.float64)
        if p.shape != (E.shape[0],):
            raise ValueError(
                f"price vector has shape {p.shape}, expected ({E.shape[0]},) to match "
                "the number of simulation steps")
        energy = (E * p[:, None]).sum(axis=0)
        zero = np.zeros_like(energy)
        return {"energy_usd": energy, "fixed_usd": zero.copy(),
                "adders_usd": zero.copy(), "total_usd": energy.copy()}


@dataclass(frozen=True)
class TieredMonthlyTariff:
    """An increasing-block tariff billed on calendar-month cumulative consumption.

    This is the structure of a US municipal residential rate: a fixed monthly customer
    charge, an energy charge whose rate steps up as the month's cumulative kWh crosses
    each tier boundary, and a set of per-kWh riders that apply to every kWh regardless
    of tier.

    ``tier_upper_kwh`` are the cumulative monthly upper bounds of all but the last tier,
    strictly increasing; the last tier is unbounded. ``tier_rate_usd_per_kwh`` has one
    more entry than ``tier_upper_kwh``.

    ``rider_usd_per_kwh`` is either a scalar, applied to every kWh, or a ``(T,)`` vector,
    which is how a time-of-use power-supply charge is represented: the tier structure is
    unchanged and only the rider varies with the clock.

    The property that matters for control, when the rider is a scalar: **the tariff then
    has no intra-day price variation at all.** Within a month the marginal price depends
    only on cumulative consumption, so a controller that shifts a kWh from 18:00 to
    03:00 saves exactly nothing on the bill. Any bill saving must come from consuming
    fewer kWh, possibly dropping a tier, not from when they are consumed. That is a
    statement about the tariff, not about the controller, and it is the reason this class
    exists alongside :class:`HourlyEnergyTariff`.
    """

    customer_charge_usd_per_month: float
    tier_upper_kwh: Tuple[float, ...]
    tier_rate_usd_per_kwh: Tuple[float, ...]
    rider_usd_per_kwh: object
    name: str
    source: str

    @property
    def time_varying(self) -> bool:
        return np.ndim(self.rider_usd_per_kwh) > 0

    def __post_init__(self) -> None:
        if len(self.tier_rate_usd_per_kwh) != len(self.tier_upper_kwh) + 1:
            raise ValueError(
                f"{len(self.tier_upper_kwh)} tier boundaries need "
                f"{len(self.tier_upper_kwh) + 1} rates, got "
                f"{len(self.tier_rate_usd_per_kwh)}")
        if list(self.tier_upper_kwh) != sorted(set(self.tier_upper_kwh)):
            raise ValueError("tier_upper_kwh must be strictly increasing")

    def _month_energy_charge(self, monthly_kwh: np.ndarray) -> np.ndarray:
        """Energy charge for one month's consumption, vectorised over buildings."""
        edges = np.concatenate([[0.0], np.asarray(self.tier_upper_kwh, dtype=np.float64),
                                [np.inf]])
        rates = np.asarray(self.tier_rate_usd_per_kwh, dtype=np.float64)
        charge = np.zeros_like(monthly_kwh)
        for k, rate in enumerate(rates):
            lo, hi = edges[k], edges[k + 1]
            in_tier = np.clip(monthly_kwh, lo, hi) - lo
            charge += rate * in_tier
        return charge

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        E = np.asarray(import_kwh, dtype=np.float64)
        m = np.asarray(month, dtype=np.int64)
        if m.shape != (E.shape[0],):
            raise ValueError(
                f"month column has shape {m.shape}, expected ({E.shape[0]},)")
        months = np.unique(m)
        energy = np.zeros(E.shape[1], dtype=np.float64)
        for mm in months:
            energy += self._month_energy_charge(E[m == mm].sum(axis=0))
        fixed = np.full(E.shape[1],
                        self.customer_charge_usd_per_month * float(months.size))
        if self.time_varying:
            r = np.asarray(self.rider_usd_per_kwh, dtype=np.float64)
            if r.shape != (E.shape[0],):
                raise ValueError(
                    f"rider vector has shape {r.shape}, expected ({E.shape[0]},)")
            adders = (E * r[:, None]).sum(axis=0)
        else:
            adders = float(self.rider_usd_per_kwh) * E.sum(axis=0)
        return {"energy_usd": energy, "fixed_usd": fixed, "adders_usd": adders,
                "total_usd": energy + fixed + adders}


class EmissionFactor(Protocol):
    """Maps household import energy to CO2 mass."""

    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HourlyEmissionFactor:
    """A time-varying grid carbon intensity, ``(T,)`` in kg CO2 per kWh consumed."""

    kg_per_kwh: np.ndarray
    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        c = np.asarray(self.kg_per_kwh, dtype=np.float64)
        E = np.asarray(import_kwh, dtype=np.float64)
        if c.shape != (E.shape[0],):
            raise ValueError(
                f"carbon series has shape {c.shape}, expected ({E.shape[0]},)")
        return (E * c[:, None]).sum(axis=0)


@dataclass(frozen=True)
class ConstantEmissionFactor:
    """A time-invariant grid carbon intensity, kg CO2 per kWh consumed.

    Use this when the only sourced figure available is an annual average. Note the
    consequence and state it in any paper that uses it: with a constant factor, CO2
    savings are **exactly proportional to energy savings**, so no CO2 benefit from load
    shifting can be measured or claimed. Measuring a shifting benefit requires an hourly
    series for the right balancing authority.
    """

    kg_per_kwh: float
    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        return float(self.kg_per_kwh) * np.asarray(
            import_kwh, dtype=np.float64).sum(axis=0)


def egrid_constant_factor(
    lb_co2_per_mwh: float,
    name: str,
    source: str,
    delivered: bool = True,
) -> ConstantEmissionFactor:
    """Convert an eGRID-style output emission rate in lb CO2/MWh to kg CO2/kWh.

    ``delivered=True`` grosses the generated rate up by the EPA's transmission-and-
    distribution loss factor, which is the correct basis for a *consumption* footprint
    and is what the EPA Greenhouse Gas Equivalencies Calculator does. Set it to
    ``False`` to keep the generation-basis rate.
    """
    kg_per_kwh = lb_co2_per_mwh * LB_TO_KG / 1000.0
    if delivered:
        kg_per_kwh /= (1.0 - EPA_TD_LOSS_FRACTION)
    basis = "delivered" if delivered else "generated"
    return ConstantEmissionFactor(
        kg_per_kwh=kg_per_kwh, name=name,
        source=f"{source} [{lb_co2_per_mwh} lb CO2/MWh, {basis} basis]")


@dataclass(frozen=True)
class ComfortBand:
    """The temperature band a household is entitled to, degC.

    CityLearn supplies a cooling set point and a heating set point per building per
    step. A household is "in band" at step ``t`` when
    ``theta_set_heat - tolerance <= theta <= theta_set_cool + tolerance``.

    ``tolerance_c`` is a required argument because it is a comfort-standard choice, not
    a physical constant, and the fraction of hours judged comfortable is highly
    sensitive to it.
    """

    tolerance_c: float
    source: str
    occupied_only: bool = True


@dataclass(frozen=True)
class HouseholdCase:
    """The fixed household-benefit case: one tariff, one emission factor, one band.

    Pinning these once, before any grid is run, is what stops the benefit headline from
    being chosen after the fact from whichever combination looked best.
    """

    tariff: Tariff
    emissions: EmissionFactor
    comfort: ComfortBand
    district_cap_kw: Optional[float]
    ev_penetration: Optional[float]
    hours_per_step: float = 1.0

    @property
    def provenance(self) -> Dict[str, str]:
        return {
            "tariff": f"{self.tariff.name} -- {self.tariff.source}",
            "emissions": f"{self.emissions.name} -- {self.emissions.source}",
            "comfort": f"tolerance {self.comfort.tolerance_c} degC, "
                       f"occupied_only={self.comfort.occupied_only} -- "
                       f"{self.comfort.source}",
            "district_cap_kw": repr(self.district_cap_kw),
            "ev_penetration": repr(self.ev_penetration),
        }


@dataclass
class HouseholdBenefit:
    """Per-household outcomes over one rollout. Arrays have shape ``(B,)``."""

    bill_usd: np.ndarray
    bill_components: Dict[str, np.ndarray]
    co2_kg: np.ndarray
    import_kwh: np.ndarray
    export_kwh: np.ndarray
    #: Gross site electricity consumption, kWh: ``(e_bt + pv_bt) * h`` summed over t.
    #: This is what the household's devices actually consumed, and the only one of the
    #: three energy quantities here that measures efficiency. ``import_kwh`` can fall
    #: while this rises, by self-consuming photovoltaic output that would otherwise have
    #: been exported, and the two must never be reported under one label.
    gross_load_kwh: Optional[np.ndarray]
    #: Metered photovoltaic output, kWh. ``None`` when no PV series was supplied.
    pv_kwh: Optional[np.ndarray]
    comfort_hours_in_band: np.ndarray
    comfort_hours_assessed: np.ndarray
    own_peak_kw: np.ndarray
    coincident_peak_kw: np.ndarray
    district_peak_kw: float
    district_peak_step: int
    cap_exceedance_kwh: Optional[float]
    steps: int
    hours_per_step: float
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def comfort_fraction_in_band(self) -> np.ndarray:
        assessed = np.maximum(self.comfort_hours_assessed, 1e-12)
        return self.comfort_hours_in_band / assessed

    def annualised(self) -> Dict[str, np.ndarray]:
        """Scale extensive quantities to a 365-day year.

        A rollout shorter than a year is scaled by ``8760 h / (steps * h)``. This is a
        linear extrapolation and is wrong for any seasonal quantity, which is why
        :func:`household_benefit` records ``steps`` and why the only window that should
        be reported without a caveat is a full year.
        """
        hours = self.steps * self.hours_per_step
        k = 8760.0 / hours
        return {"bill_usd": self.bill_usd * k, "co2_kg": self.co2_kg * k,
                "import_kwh": self.import_kwh * k, "scale_factor": np.float64(k)}


def household_benefit(
    net_kw: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
    indoor_c: np.ndarray,
    cooling_set_c: np.ndarray,
    heating_set_c: np.ndarray,
    occupant_count: np.ndarray,
    case: HouseholdCase,
    pv_kw: Optional[np.ndarray] = None,
) -> HouseholdBenefit:
    """Compute per-household benefit from a rollout.

    Parameters
    ----------
    net_kw
        ``(T, B)`` signed net electricity consumption ``e_bt``, kW.
    month, hour
        ``(T,)`` CityLearn calendar columns, 1-indexed.
    indoor_c, cooling_set_c, heating_set_c, occupant_count
        ``(T, B)`` per-building indoor temperature, the two set points, and occupancy.
    case
        The pinned :class:`HouseholdCase`.
    """
    net = np.asarray(net_kw, dtype=np.float64)
    if net.ndim != 2:
        raise ValueError(f"net_kw must be (T, B), got shape {net.shape}")
    T, B = net.shape
    h = float(case.hours_per_step)

    imp_kw = np.maximum(net, 0.0)
    exp_kw = np.maximum(-net, 0.0)
    import_kwh = imp_kw * h

    components = case.tariff.bill_usd(import_kwh, np.asarray(month), np.asarray(hour))
    co2 = case.emissions.co2_kg(import_kwh)

    theta = np.asarray(indoor_c, dtype=np.float64)
    lo = np.asarray(heating_set_c, dtype=np.float64) - case.comfort.tolerance_c
    hi = np.asarray(cooling_set_c, dtype=np.float64) + case.comfort.tolerance_c
    for nm, arr in (("indoor_c", theta), ("cooling_set_c", hi), ("heating_set_c", lo)):
        if arr.shape != (T, B):
            raise ValueError(f"{nm} must be ({T}, {B}), got {arr.shape}")
    in_band = (theta >= lo) & (theta <= hi)
    if case.comfort.occupied_only:
        assessed = np.asarray(occupant_count, dtype=np.float64) > 0.0
    else:
        assessed = np.ones((T, B), dtype=bool)
    comfort_in = (in_band & assessed).sum(axis=0) * h
    comfort_assessed = assessed.sum(axis=0) * h

    district = imp_kw.sum(axis=1)
    peak_step = int(np.argmax(district))
    own_peak = imp_kw.max(axis=0)
    coincident = imp_kw[peak_step, :].copy()

    if case.district_cap_kw is None:
        exceed = None
    else:
        exceed = float(
            np.maximum(district - float(case.district_cap_kw), 0.0).sum() * h)

    if pv_kw is None:
        gross_kwh = pv_total = None
    else:
        pv = np.abs(np.asarray(pv_kw, dtype=np.float64))
        if pv.shape != (T, B):
            raise ValueError(f"pv_kw must be ({T}, {B}), got {pv.shape}")
        gross_kwh = ((net + pv) * h).sum(axis=0)
        pv_total = (pv * h).sum(axis=0)

    return HouseholdBenefit(
        bill_usd=components["total_usd"], bill_components=components, co2_kg=co2,
        import_kwh=import_kwh.sum(axis=0), export_kwh=(exp_kw * h).sum(axis=0),
        gross_load_kwh=gross_kwh, pv_kwh=pv_total,
        comfort_hours_in_band=comfort_in, comfort_hours_assessed=comfort_assessed,
        own_peak_kw=own_peak, coincident_peak_kw=coincident,
        district_peak_kw=float(district[peak_step]), district_peak_step=peak_step,
        cap_exceedance_kwh=exceed, steps=T, hours_per_step=h,
        provenance=case.provenance)


def benefit_delta(
    treatment: HouseholdBenefit, reference: HouseholdBenefit
) -> Dict[str, np.ndarray]:
    """Per-household savings of ``treatment`` against ``reference``.

    Positive means the household is better off: money saved, CO2 avoided, comfort hours
    gained, peak contribution reduced. Sign conventions are fixed here so that no figure
    has to explain its own.
    """
    if treatment.bill_usd.shape != reference.bill_usd.shape:
        raise ValueError(
            f"treatment covers {treatment.bill_usd.shape[0]} households, reference "
            f"{reference.bill_usd.shape[0]}")
    if treatment.steps != reference.steps:
        raise ValueError(
            f"treatment ran {treatment.steps} steps, reference {reference.steps}; "
            "a benefit delta between different windows is not meaningful")
    out = {
        "bill_saved_usd": reference.bill_usd - treatment.bill_usd,
        "bill_saved_frac": np.where(
            reference.bill_usd > 0.0,
            (reference.bill_usd - treatment.bill_usd)
            / np.maximum(reference.bill_usd, 1e-12), np.nan),
        "co2_avoided_kg": reference.co2_kg - treatment.co2_kg,
        # Deliberately NOT called "energy saved". Import can fall while the household's
        # devices consume strictly more, by self-consuming photovoltaic output that
        # would otherwise have been exported. The three quantities below decompose that,
        # and they satisfy
        #     import_avoided - export_forgone + gross_load_increase = 0
        # exactly, which is the household energy balance and is asserted in the tests.
        "import_avoided_kwh": reference.import_kwh - treatment.import_kwh,
        "export_forgone_kwh": reference.export_kwh - treatment.export_kwh,
        "comfort_hours_gained": (treatment.comfort_hours_in_band
                                 - reference.comfort_hours_in_band),
        "own_peak_cut_kw": reference.own_peak_kw - treatment.own_peak_kw,
        "coincident_peak_cut_kw": (reference.coincident_peak_kw
                                   - treatment.coincident_peak_kw),
    }
    if treatment.gross_load_kwh is not None and reference.gross_load_kwh is not None:
        out["gross_load_saved_kwh"] = (reference.gross_load_kwh
                                       - treatment.gross_load_kwh)
    return out


# --------------------------------------------------------------------------------------
# Named, sourced presets. None of these is a default: a HouseholdCase must be given one
# explicitly. Each docstring carries the source that licenses the numbers.
# --------------------------------------------------------------------------------------

CITYLEARN_2022_TARIFF_NAME = "citylearn_challenge_2022_phase_all"
CITYLEARN_2022_TARIFF_SOURCE = (
    "pricing.csv shipped with the CityLearn citylearn_challenge_2022_phase_all dataset "
    "(Nweye, Sankaranarayanan & Nagy 2023, doi:10.18738/T8/0YLJ6Q). Time-of-use, five "
    "distinct prices spanning 0.21-0.54 USD/kWh, peak 16:00-20:00 local. The "
    "neighbourhood is in Fontana, California (Nweye et al., NeurIPS 2022 Competition "
    "Track, PMLR 220:85-103), NOT Travis County, Texas."
)

CITYLEARN_2022_CARBON_SOURCE = (
    "carbon_intensity.csv shipped with citylearn_challenge_2022_phase_all "
    "(doi:10.18738/T8/0YLJ6Q); hourly, mean 0.1565 kg CO2/kWh over 8760 h. California "
    "grid, NOT ERCOT."
)

# --------------------------------------------------------------------------------------
# Austin Energy, the actual utility for Travis County, Texas.
#
# All figures below are transcribed from the primary source: City of Austin, "Electric
# Tariff", effective November 1, 2025 (Fiscal Year 2026), retrieved 2026-10-06 from
# https://austinenergy.com/-/media/project/websites/shared/pdfs/rates/tariff.pdf
# (49 pages, 2,541,581 bytes). Page numbers below are PDF page numbers.
#
# Two mismatches to state wherever these are used: the schedule is the one in force at
# retrieval, not the 2018 schedule matching the AMY2018 weather year; and Austin Energy
# is a municipal utility, so these rates are not representative of the competitive
# retail market in the rest of ERCOT.
# --------------------------------------------------------------------------------------

_AE_TARIFF_SOURCE = (
    "City of Austin Electric Tariff effective 2025-11-01 (FY2026), "
    "https://austinenergy.com/-/media/project/websites/shared/pdfs/rates/tariff.pdf, "
    "retrieved 2026-10-06")

#: Residential Service, Inside City Limits, energy charge tiers (PDF p. 4).
#: Customer charge 16.50 USD/month; monthly cumulative blocks 0-300 / 301-900 /
#: 901-2000 / over-2000 kWh at 4.640 / 5.138 / 7.525 / 10.884 cents per kWh.
AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH = 16.50
AE_RES_TIER_UPPER_KWH = (300.0, 900.0, 2000.0)
AE_RES_TIER_RATE_USD_PER_KWH = (0.04640, 0.05138, 0.07525, 0.10884)

#: Standard Power Supply Adjustment, Residential Service, Inside City Limits (PDF p. 4).
AE_RES_PSA_USD_PER_KWH = 0.04118

#: Community Benefit Charges, Residential Service, Inside City Limits (PDF p. 4):
#: Customer Assistance Program 0.564, Service Area Lighting 0.254, Energy Efficiency
#: Services 0.457 cents per kWh.
AE_RES_CBC_USD_PER_KWH = 0.00564 + 0.00254 + 0.00457

#: Regulatory Charge, Secondary voltage, energy basis (PDF p. 27). Recovers ERCOT
#: transmission service charges, NERC/TRE fees and ERCOT nodal and administrative fees.
AE_REGULATORY_SECONDARY_USD_PER_KWH = 0.01338

#: Every per-kWh charge that is flat in time under the standard residential schedule.
AE_RES_FLAT_RIDERS_USD_PER_KWH = (
    AE_RES_PSA_USD_PER_KWH + AE_RES_CBC_USD_PER_KWH
    + AE_REGULATORY_SECONDARY_USD_PER_KWH)

#: Residential Time-of-Use pilot power-supply charges (PDF p. 37). These are applied
#: **in lieu of** the standard PSA, leaving every other component unchanged. Mid-peak
#: equals the standard PSA exactly, so the pilot is a spread around the standard rate.
#: The pilot is capped at 100 individual meters (PDF p. 36), which is a material
#: limitation on any claim that a household can actually obtain this rate.
AE_TOU_OFF_PEAK_USD_PER_KWH = 0.02677
AE_TOU_MID_PEAK_USD_PER_KWH = 0.04118
AE_TOU_ON_PEAK_USD_PER_KWH = 0.08442

#: CityLearn ``hour`` is 1-indexed and hour ``h`` covers the clock interval
#: ``[h-1, h)``. CityLearn ``day_type`` is 1=Monday .. 7=Sunday (verified against the
#: 2018 calendar on the Travis County dataset: step 0 is 2018-01-01, a Monday, and
#: day_type 6 and 7 coincide exactly with Saturday and Sunday).
_AE_TOU_ON_PEAK_HOURS = frozenset({16, 17, 18})          # 15:00-18:00
_AE_TOU_MID_PEAK_HOURS = frozenset(range(8, 16)) | frozenset({19, 20, 21, 22})
_AE_WEEKEND_DAY_TYPES = frozenset({6, 7})


def austin_tou_psa_vector(hour: np.ndarray, day_type: np.ndarray) -> np.ndarray:
    """Build the ``(T,)`` time-of-use power-supply charge vector, USD/kWh.

    ``hour`` and ``day_type`` are the CityLearn calendar columns. Weekends are off-peak
    for the entire day.
    """
    h = np.asarray(hour, dtype=np.int64)
    dt_ = np.asarray(day_type, dtype=np.int64)
    if h.shape != dt_.shape:
        raise ValueError(f"hour {h.shape} and day_type {dt_.shape} must match")
    bad = sorted(set(np.unique(h).tolist()) - set(range(1, 25)))
    if bad:
        raise ValueError(f"hour must be 1..24 (CityLearn convention), saw {bad}")
    bad = sorted(set(np.unique(dt_).tolist()) - set(range(1, 8)))
    if bad:
        raise ValueError(f"day_type must be 1..7 (1=Monday), saw {bad}")
    psa = np.full(h.shape, AE_TOU_OFF_PEAK_USD_PER_KWH, dtype=np.float64)
    weekday = ~np.isin(dt_, list(_AE_WEEKEND_DAY_TYPES))
    psa[weekday & np.isin(h, list(_AE_TOU_MID_PEAK_HOURS))] = \
        AE_TOU_MID_PEAK_USD_PER_KWH
    psa[weekday & np.isin(h, list(_AE_TOU_ON_PEAK_HOURS))] = \
        AE_TOU_ON_PEAK_USD_PER_KWH
    return psa


def austin_energy_residential() -> TieredMonthlyTariff:
    """Austin Energy Residential Service, Inside City Limits, standard schedule.

    Flat in time: see :class:`TieredMonthlyTariff` for the consequence.
    """
    return TieredMonthlyTariff(
        customer_charge_usd_per_month=AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH,
        tier_upper_kwh=AE_RES_TIER_UPPER_KWH,
        tier_rate_usd_per_kwh=AE_RES_TIER_RATE_USD_PER_KWH,
        rider_usd_per_kwh=AE_RES_FLAT_RIDERS_USD_PER_KWH,
        name="austin_energy_residential_inside_city",
        source=(f"{_AE_TARIFF_SOURCE}; Residential Service Inside City Limits (p. 4), "
                f"Regulatory Charge Secondary energy basis (p. 27). Customer charge "
                f"16.50 USD/mo; tiers 4.640/5.138/7.525/10.884 c/kWh at 300/900/2000 "
                f"kWh; flat riders PSA 4.118 + CBC 1.275 + Regulatory 1.338 c/kWh. "
                f"No time-of-use component."),
    )


def austin_energy_residential_tou_pilot(
    hour: np.ndarray, day_type: np.ndarray
) -> TieredMonthlyTariff:
    """Austin Energy Residential Service under the Time-of-Use pilot power-supply rate.

    Identical to :func:`austin_energy_residential` except that the flat PSA is replaced
    by the pilot's three-period charge. This is the only Austin Energy residential rate
    with any intra-day price signal, and it is restricted to 100 meters.
    """
    psa = austin_tou_psa_vector(hour, day_type)
    rider = psa + AE_RES_CBC_USD_PER_KWH + AE_REGULATORY_SECONDARY_USD_PER_KWH
    return TieredMonthlyTariff(
        customer_charge_usd_per_month=AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH,
        tier_upper_kwh=AE_RES_TIER_UPPER_KWH,
        tier_rate_usd_per_kwh=AE_RES_TIER_RATE_USD_PER_KWH,
        rider_usd_per_kwh=rider,
        name="austin_energy_residential_tou_pilot",
        source=(f"{_AE_TARIFF_SOURCE}; Residential Service Inside City Limits (p. 4) "
                f"with the Pilot Programs Residential Time-of-Use power supply charges "
                f"(p. 37) in lieu of the standard PSA: off-peak 2.677, mid-peak 4.118, "
                f"on-peak 8.442 c/kWh; weekday on-peak 15:00-18:00, mid-peak "
                f"07:00-15:00 and 18:00-22:00, off-peak 22:00-07:00, weekends off-peak "
                f"all day. Pilot limited to 100 individual meters (p. 36)."),
    )


#: Value-of-Solar Rider, Non-Demand rate-schedule type (PDF p. 32): avoided cost
#: 7.61 c/kWh plus societal benefit 2.3 c/kWh.
AE_VOS_NON_DEMAND_USD_PER_KWH = 0.0761 + 0.023


def value_of_solar_credit_usd(pv_output_kwh: np.ndarray) -> np.ndarray:
    """Austin Energy Value-of-Solar credit, USD, from gross PV output.

    The credit is on the **metered output of the photovoltaic system** -- all of it, not
    only the exported part (PDF p. 32). Correspondingly, a residential customer on this
    rider is billed on **gross** consumption: the tariff states that billable usage is
    "the sum of (1) the energy produced by the solar installation and consumed on-site,
    and (2) the energy delivered to the customer by Austin Energy" (PDF p. 5).

    Two consequences follow directly, and both bear on what a controller can achieve:

    1. Self-consumption carries no premium. A kWh of PV earns the same credit whether it
       is exported or consumed behind the meter, and is billed at the retail rate either
       way. So the usual "increase self-consumption" objective has no value here.
    2. Because billable usage is gross and the tariff is flat in time, a home battery on
       this rider cannot earn arbitrage and its round-trip losses add to billable
       consumption. Under the standard schedule it is strictly bill-increasing.

    ``pv_output_kwh`` has shape ``(T, B)``. The monthly credit cap ("credits applied as
    an offset to the electric bill cannot exceed the total electric charges"), with
    carry-forward of the excess, is **not** modelled here; this function returns the
    uncapped credit, which is an upper bound.
    """
    pv = np.asarray(pv_output_kwh, dtype=np.float64)
    if pv.ndim != 2:
        raise ValueError(f"pv_output_kwh must be (T, B), got {pv.shape}")
    return AE_VOS_NON_DEMAND_USD_PER_KWH * pv.sum(axis=0)


def value_of_solar_bill_usd(
    tariff: Tariff,
    gross_load_kwh: np.ndarray,
    pv_output_kwh: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
) -> Dict[str, np.ndarray]:
    """The bill actually faced by an Austin Energy residential customer with PV.

    Implements the tariff's own two rules, which differ from net metering in both
    directions:

    * **Billable usage is gross, not net** (PDF p. 5): "usage that is billable under the
      residential rate schedule is the sum of (1) the energy produced by the solar
      installation and consumed on-site, and (2) the energy delivered to the customer by
      Austin Energy". So the tiered energy charge is assessed on ``gross_load_kwh``.
    * **The credit is on gross photovoltaic output** (PDF p. 32), not on exports.

    The consequence for storage control is a sign reversal worth stating plainly: because
    the credit does not depend on when or whether a photovoltaic kWh is exported, and
    because the standard schedule is flat in time, a battery earns nothing here and its
    round-trip losses land in ``gross_load_kwh``. Raising self-consumption, the usual
    objective, moves no term in this expression at all.

    The monthly cap on credits and their carry-forward are not modelled, so
    ``net_usd`` is a lower bound on the bill when credits exceed charges in a month.
    """
    gross = np.asarray(gross_load_kwh, dtype=np.float64)
    pv = np.abs(np.asarray(pv_output_kwh, dtype=np.float64))
    if gross.shape != pv.shape:
        raise ValueError(
            f"gross_load_kwh {gross.shape} and pv_output_kwh {pv.shape} must match")
    charges = tariff.bill_usd(gross, month, hour)
    credit = value_of_solar_credit_usd(pv)
    out = {k: v.copy() for k, v in charges.items()}
    out["vos_credit_usd"] = credit
    out["net_usd"] = charges["total_usd"] - credit
    return out


def egrid_erct_delivered() -> ConstantEmissionFactor:
    """ERCOT All (eGRID subregion ERCT) total-output CO2 rate, delivered basis.

    Source: US EPA, "Greenhouse Gas Equivalencies Calculator - Calculations and
    References", eGRID Regions table, subregion ERCT "ERCOT All", total output
    (baseload) emission rate 771.1 lb CO2/MWh, citing EPA (2024) eGRID Table 1
    (eGRID2022), year 2022 data; grossed up for delivery by the same page's
    1/(1-0.051) factor.

    Two mismatches to state wherever this is used: the data year is 2022 while the
    building stock is driven by 2018 weather, and the rate is an annual average, so no
    load-shifting CO2 benefit can be measured with it.
    """
    return egrid_constant_factor(
        lb_co2_per_mwh=771.1, name="egrid2022_erct_delivered",
        source=("US EPA, Greenhouse Gas Equivalencies Calculator - Calculations and "
                "References, eGRID Regions table, ERCT 'ERCOT All' total output "
                "(baseload) rate, eGRID2022 (year 2022 data)"),
        delivered=True)


def citylearn_2022_tariff(price_usd_per_kwh: Sequence[float]) -> HourlyEnergyTariff:
    """Wrap the shipped 2022-challenge price series, carrying its provenance."""
    return HourlyEnergyTariff(
        price_usd_per_kwh=np.asarray(price_usd_per_kwh, dtype=np.float64),
        name=CITYLEARN_2022_TARIFF_NAME, source=CITYLEARN_2022_TARIFF_SOURCE)


def citylearn_2022_emissions(kg_per_kwh: Sequence[float]) -> HourlyEmissionFactor:
    """Wrap the shipped 2022-challenge carbon series, carrying its provenance."""
    return HourlyEmissionFactor(
        kg_per_kwh=np.asarray(kg_per_kwh, dtype=np.float64),
        name="citylearn_challenge_2022_phase_all",
        source=CITYLEARN_2022_CARBON_SOURCE)
