"""Every consumer's observation index must be derived from the canonical name list.

Audit B9: nine observation indices were written as integer literals in eight modules.
They were all correct, but nothing bound them to ``OBS_NAMES``, so one insertion into
that list would have silently corrupted the reward, the shield and the KPIs at once.
Audit B8: the schema builder kept a second copy of the observation list that disagreed
with the environment's.

The tests below pin three things:
  1. the (constant -> name) mapping each consumer intends, written out here, so a
     consumer cannot quietly start reading a different observation;
  2. that inserting a name into the canonical list shifts every consumer together;
  3. that no module reintroduces an integer-literal observation index.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from stems.observations import (CANONICAL_OBS_NAMES, OBS_NAMES, obs_index,
                                schema_observation_names)

REPO = Path(__file__).resolve().parents[1]

# (module, attribute, observation name it is supposed to point at)
CONSUMER_BINDINGS = [
    ("stems.cbf", "_IDX_T_OUT", "outdoor_dry_bulb_temperature"),
    ("stems.cbf", "_IDX_SOC_ELEC", "electrical_storage_soc"),
    ("stems.cbf", "_IDX_NET", "net_electricity_consumption"),
    ("stems.reward", "_IDX_T_IN", "indoor_dry_bulb_temperature"),
    ("stems.reward", "_IDX_LOAD", "non_shiftable_load"),
    ("stems.reward", "_IDX_SOLAR", "solar_generation"),
    ("stems.reward", "_IDX_NET", "net_electricity_consumption"),
    ("stems.reward", "_IDX_PRICE", "electricity_pricing"),
    ("stems.reward", "_IDX_OCCUPANT", "occupant_count"),
    ("stems.reward", "_IDX_T_SET", "indoor_dry_bulb_temperature_cooling_set_point"),
    ("stems.metrics", "_IDX_PRICE", "electricity_pricing"),
    ("stems.metrics", "_IDX_CARBON", "carbon_intensity"),
    ("stems.metrics", "_IDX_T_IN", "indoor_dry_bulb_temperature"),
    ("stems.metrics", "_IDX_T_SET", "indoor_dry_bulb_temperature_cooling_set_point"),
    ("stems.metrics", "_IDX_OCCUPANT", "occupant_count"),
    ("stems.metrics", "_IDX_NET", "net_electricity_consumption"),
    ("stems.metrics", "_IDX_SOC_ELEC", "electrical_storage_soc"),
    ("stems.metrics", "_IDX_SOC_DHW", "dhw_storage_soc"),
    ("stems.metrics", "_IDX_DHW_DEMAND", "dhw_demand"),
    ("stems.metrics", "_IDX_SOLAR", "solar_generation"),
    ("stems.baselines", "_IDX_HOUR", "hour"),
    ("stems.baselines", "_IDX_PRICE", "electricity_pricing"),
    ("stems.baselines", "_IDX_SOC_ELEC", "electrical_storage_soc"),
    ("stems.baselines", "_IDX_T_IN", "indoor_dry_bulb_temperature"),
    ("stems.baselines", "_IDX_LOAD", "non_shiftable_load"),
    ("stems.baselines", "_IDX_SOLAR", "solar_generation"),
    ("stems.baselines", "_IDX_T_COOL", "indoor_dry_bulb_temperature_cooling_set_point"),
    ("stems.baselines", "_IDX_T_HEAT", "indoor_dry_bulb_temperature_heating_set_point"),
    ("stems.baselines", "_IDX_NET", "net_electricity_consumption"),
    ("stems.thermal", "IDX_DAY_TYPE", "day_type"),
    ("stems.thermal", "IDX_HOUR", "hour"),
    ("stems.thermal", "IDX_T_OUT", "outdoor_dry_bulb_temperature"),
    ("stems.thermal", "IDX_T_OUT_PRED", "outdoor_dry_bulb_temperature_predicted_1"),
    ("stems.thermal", "IDX_SOC_DHW", "dhw_storage_soc"),
    ("stems.thermal", "IDX_DHW_DEMAND", "dhw_demand"),
    ("stems.fleet", "_IDX_LOAD", "non_shiftable_load"),
    ("stems.fleet", "_IDX_SOLAR", "solar_generation"),
    ("stems.fleet", "_IDX_SOC_DHW", "dhw_storage_soc"),
    ("stems.fleet", "_IDX_DHW_DEMAND", "dhw_demand"),
    ("stems.fleet", "_IDX_NET", "net_electricity_consumption"),
    ("experiments.runner", "_IDX_SOC_DHW", "dhw_storage_soc"),
    ("experiments.runner", "_IDX_SOC", "electrical_storage_soc"),
    ("experiments.runner", "_IDX_NET", "net_electricity_consumption"),
    ("experiments.ev_coupling", "_IDX_HOUR", "hour"),
    ("experiments.ev_coupling", "_IDX_NET", "net_electricity_consumption"),
    ("experiments.ev_coupling", "_IDX_PRICE", "electricity_pricing"),
]

CONSUMER_MODULES = sorted({m for m, _, _ in CONSUMER_BINDINGS})


@pytest.mark.parametrize("module_name,attr,obs_name", CONSUMER_BINDINGS)
def test_consumer_index_matches_its_observation_name(module_name, attr, obs_name):
    module = importlib.import_module(module_name)
    assert getattr(module, attr) == obs_index(obs_name), (
        f"{module_name}.{attr} should be the index of {obs_name!r}")


def test_indices_still_equal_the_values_the_audit_verified():
    """The values the 6 October audit checked by hand, pinned verbatim.

    If the canonical list is deliberately reordered this test is expected to fail and
    to be updated in the same commit; its job is to make that visible rather than
    silent, because every stored run record in results/ was produced under this layout.
    """
    pinned = {
        "day_type": 0, "hour": 1, "outdoor_dry_bulb_temperature": 2,
        "carbon_intensity": 14, "indoor_dry_bulb_temperature": 15,
        "non_shiftable_load": 16, "solar_generation": 17, "dhw_storage_soc": 18,
        "electrical_storage_soc": 19, "net_electricity_consumption": 20,
        "electricity_pricing": 21, "cooling_demand": 24, "dhw_demand": 25,
        "occupant_count": 26,
        "indoor_dry_bulb_temperature_cooling_set_point": 27,
        "indoor_dry_bulb_temperature_heating_set_point": 28,
        "heating_electricity_consumption": 29,
    }
    assert {n: obs_index(n) for n in pinned} == pinned


def test_inserting_an_observation_shifts_every_consumer_together(monkeypatch):
    """The failure mode B9 describes: an insertion into OBS_NAMES.

    Insert a name at position 2 of the canonical list, reload the registry and every
    consumer, and require that each consumer's index moved by exactly the amount the
    registry says. Before this refactor the literal constants would not have moved at
    all and the reward, shield and KPIs would have read the wrong columns in silence.
    """
    import stems.observations as obs_mod

    original = list(obs_mod.OBS_NAMES)
    before = {}
    for module_name, attr, name in CONSUMER_BINDINGS:
        before[(module_name, attr)] = getattr(
            importlib.import_module(module_name), attr)

    patched = original[:2] + ["synthetic_probe_observation"] + original[2:]
    monkeypatch.setattr(obs_mod, "OBS_NAMES", patched)
    monkeypatch.setattr(obs_mod, "CANONICAL_OBS_NAMES",
                        patched + list(obs_mod.HEATPUMP_OBS_NAMES))
    monkeypatch.setattr(obs_mod, "_CANONICAL_INDEX",
                        {n: i for i, n in enumerate(obs_mod.CANONICAL_OBS_NAMES)})
    try:
        for module_name in CONSUMER_MODULES:
            importlib.reload(importlib.import_module(module_name))
        for (module_name, attr, name) in CONSUMER_BINDINGS:
            now = getattr(importlib.import_module(module_name), attr)
            expected = obs_mod.obs_index(name)
            assert now == expected, f"{module_name}.{attr} did not follow the insertion"
            shift = 1 if before[(module_name, attr)] >= 2 else 0
            assert now == before[(module_name, attr)] + shift
    finally:
        monkeypatch.undo()
        for module_name in CONSUMER_MODULES:
            importlib.reload(importlib.import_module(module_name))
    for (module_name, attr, name) in CONSUMER_BINDINGS:
        assert getattr(importlib.import_module(module_name), attr) == obs_index(name)


_LITERAL_OBS_INDEX = re.compile(r"^_?IDX_[A-Z0-9_]*\s*(:\s*int\s*)?=\s*-?\d")


def test_no_module_reintroduces_a_literal_observation_index():
    """A new ``_IDX_SOMETHING = 19`` anywhere in stems/ or experiments/ fails here."""
    offenders = []
    for path in sorted(list((REPO / "stems").rglob("*.py"))
                       + list((REPO / "experiments").rglob("*.py"))):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _LITERAL_OBS_INDEX.match(line.strip()):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "observation indices must come from stems.observations.obs_index, not a "
        "literal:\n" + "\n".join(offenders))


def test_obs_index_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        obs_index("not_an_observation")
    with pytest.raises(KeyError):
        obs_index("ev0_soc")  # slot fields have no fixed index; ask the environment


def test_schema_builder_and_environment_share_one_observation_list():
    """Audit B8: the two copies used to disagree on three observations."""
    import setup_citylearn_8b

    assert setup_citylearn_8b.STEMS_OBSERVATIONS == schema_observation_names()
    required = set(OBS_NAMES)
    assert required <= set(setup_citylearn_8b.STEMS_OBSERVATIONS)
    for name in ("dhw_storage_soc", "indoor_dry_bulb_temperature_heating_set_point",
                 "heating_electricity_consumption"):
        assert name in setup_citylearn_8b.STEMS_OBSERVATIONS
    assert set(schema_observation_names()) == set(CANONICAL_OBS_NAMES)


def test_the_schema_builder_holds_no_second_observation_list():
    """The literal list must be gone, not merely shadowed."""
    source = (REPO / "setup_citylearn_8b.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if "STEMS_OBSERVATIONS" in names:
                assert not isinstance(node.value, (ast.List, ast.Tuple)), (
                    "setup_citylearn_8b.STEMS_OBSERVATIONS is a literal list again; "
                    "it must come from stems.observations.schema_observation_names()")
