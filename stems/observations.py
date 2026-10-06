"""The canonical STEMS observation vector, defined once.

Every module that reads an observation by position — the reward, the control barrier
shield, the KPI calculator, the baselines, the fleet shield, the thermal stack, the
experiment runner — and every script that writes a CityLearn schema must take its
indices and its names from here, so the two cannot disagree (audit B8, B9).

Layout of the vector a controller sees, in order:

    [0 .. 27]   ``OBS_NAMES``            always present
    [28, 29]    ``HEATPUMP_OBS_NAMES``   appended only when ``heat_pump=True``
    [30 ..]     ``EV_SLOT_FIELDS`` per electric-vehicle charger slot, ``ev_slots`` of them

``obs_index(name)`` resolves a name against the first two blocks, which are the only
ones whose position is fixed at import time. Electric-vehicle slot positions depend on
how many chargers the schema exposes and on whether the heat-pump block is present, so
they are resolved per environment through ``STEMSEnvironment.index_of`` /
``STEMSEnvironment.ev_obs_layout`` instead.

Units are SI throughout, as CityLearn reports them: temperatures in degrees Celsius,
power and demand in kW (one-hour steps, so kW and kWh/step coincide), irradiance in
W/m^2, carbon intensity in kgCO2/kWh, pricing in currency/kWh, state of charge
dimensionless in [0, 1], ``hour`` in 1..24 and ``day_type`` in 1..8 (CityLearn's own
conventions; see ``stems/thermal.py`` for the hour-origin correction).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

OBS_NAMES: List[str] = [
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "solar_generation",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "cooling_demand",
    "dhw_demand",
    "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
]

HEATPUMP_OBS_NAMES: List[str] = [
    "indoor_dry_bulb_temperature_heating_set_point",
    "heating_electricity_consumption",
]

#: Names whose index is fixed at import time: the base block plus the heat-pump block.
CANONICAL_OBS_NAMES: List[str] = OBS_NAMES + HEATPUMP_OBS_NAMES

#: Forecast lead times, in hours, of ``outdoor_dry_bulb_temperature_predicted_{1,2,3}``.
T_OUT_PRED_LEAD_H: Tuple[float, float, float] = (6.0, 12.0, 24.0)

OBS_DIM = len(OBS_NAMES)
ACTION_DIM = 3

EV_SLOT_FIELDS: List[str] = [
    "connected_state",
    "departure_time",
    "required_soc_departure",
    "soc",
    "battery_capacity",
]

EV_ACTION_PREFIX = "electric_vehicle_storage"

_CANONICAL_INDEX: Dict[str, int] = {n: i for i, n in enumerate(CANONICAL_OBS_NAMES)}

if len(_CANONICAL_INDEX) != len(CANONICAL_OBS_NAMES):
    raise RuntimeError("CANONICAL_OBS_NAMES contains a duplicate observation name")


def obs_index(name: str) -> int:
    """Position of ``name`` in the canonical observation vector.

    Raises ``KeyError`` rather than returning a sentinel: a wrong observation index is
    silent corruption of the reward, the shield and the KPIs at once, which is the
    failure mode this registry exists to prevent.
    """
    try:
        return _CANONICAL_INDEX[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a canonical STEMS observation. Electric-vehicle slot "
            "fields have no fixed index -- ask the environment "
            "(STEMSEnvironment.index_of / .ev_obs_layout). Known names: "
            f"{CANONICAL_OBS_NAMES}"
        ) from None


def obs_indices(*names: str) -> Tuple[int, ...]:
    """``obs_index`` for several names at once."""
    return tuple(obs_index(n) for n in names)


def ev_slot_obs_names(slot: int) -> List[str]:
    return [f"ev{slot}_{f}" for f in EV_SLOT_FIELDS]


def ev_native_obs_names(charger_id: str) -> Dict[str, str]:
    base = f"connected_electric_vehicle_at_charger_{charger_id}"
    return {
        "connected_state": f"electric_vehicle_charger_{charger_id}_connected_state",
        "departure_time": f"{base}_departure_time",
        "required_soc_departure": f"{base}_required_soc_departure",
        "soc": f"{base}_soc",
        "battery_capacity": f"{base}_battery_capacity",
    }


def ev_slot_action_name(slot: int) -> str:
    return f"{EV_ACTION_PREFIX}_{slot}"


def schema_observation_names(heat_pump: bool = True) -> List[str]:
    """Observations a generated CityLearn schema must mark ``active``.

    ``heat_pump=True`` (the default) includes the heating set point and the heating
    electricity consumption. They cost nothing when the scenario does not use the heat
    pump, and leaving them out is what let ``setup_citylearn_8b.py`` and
    ``stems/environment.py`` disagree (audit B8): the schema builder omitted
    ``dhw_storage_soc`` and both heat-pump entries, and the generated schema only worked
    because the upstream Travis schema happens to have them active already.
    """
    return list(CANONICAL_OBS_NAMES) if heat_pump else list(OBS_NAMES)


def selected_obs_names(heat_pump: bool = False, ev_slots: int = 0) -> List[str]:
    """The observation vector an environment with this configuration presents."""
    names = list(OBS_NAMES)
    if heat_pump:
        names += list(HEATPUMP_OBS_NAMES)
    for slot in range(int(ev_slots)):
        names += ev_slot_obs_names(slot)
    return names


def index_in(names: Sequence[str], name: str) -> int:
    """Position of ``name`` in a concrete environment's observation-name list."""
    try:
        return list(names).index(name)
    except ValueError:
        raise KeyError(
            f"{name!r} is not in this environment's observation vector. Present: "
            f"{list(names)}"
        ) from None
