"""The second implementation of the environment protocol, and what it refuses.

A protocol with one implementation has not abstracted anything, so the claim under test
is that a controller and a shield built against ``stems.protocols`` run on logged CSV
data with no CityLearn import. The other half of the claim is that the adapter *fails
loudly* on data that does not meet the contract, because an adapter that silently
zero-fills a missing column is worse than no adapter: the shield reads observations by
position and zero is a legal value for every one of them.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.battery import BatteryModel, TankModel
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.flexibility import FlexibilityPortfolio
from stems.observations import selected_obs_names
from stems.protocols import Environment, EVProvider, PlantProvider, missing_capabilities
from stems.replay import (CSVReplayEnvironment, ReplayManifestError,
                          write_manifest_template)

NAMES = selected_obs_names(heat_pump=True, ev_slots=0)
BATTERY = {"capacity_kwh": 10.0, "nominal_power_kw": 5.0, "loss_coefficient": 0.0}
DHW = {"capacity_kwh": 4.0, "heater_power_kw": 4.0, "heater_efficiency": 0.9,
       "storage_efficiency": 1.0, "loss_coefficient": 0.0}


def write_site(root, rows=24, buildings=("b1", "b2"), drop=None, plant=True,
               bad_soc=False, bad_clock=False, dt_hours=1.0):
    t0 = datetime(2026, 1, 1, 0, 0)
    cols = [n for n in NAMES if n != drop]
    for b, name in enumerate(buildings):
        with (root / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["timestamp"] + cols)
            w.writeheader()
            for i in range(rows):
                gap = timedelta(hours=dt_hours * i)
                if bad_clock and i == 3:
                    gap = timedelta(hours=dt_hours * i + 5)
                row = {"timestamp": (t0 + gap).isoformat()}
                for n in cols:
                    row[n] = 0.0
                row["hour"] = (i % 24) + 1
                row["day_type"] = 1
                row["outdoor_dry_bulb_temperature"] = 5.0
                row["indoor_dry_bulb_temperature"] = 21.0
                row["electrical_storage_soc"] = 85.0 if bad_soc else 0.5
                row["dhw_storage_soc"] = 0.4
                row["net_electricity_consumption"] = 2.0 + b
                row["non_shiftable_load"] = 2.0 + b
                row["electricity_pricing"] = 0.3
                w.writerow(row)
    doc = {"dt_hours": dt_hours, "heat_pump": True,
           "action_names": ["dhw_storage", "electrical_storage",
                            "cooling_or_heating_device"],
           "buildings": [{"name": n, "csv": f"{n}.csv",
                          **({"battery": dict(BATTERY), "dhw": dict(DHW)} if plant else {})}
                         for n in buildings]}
    path = root / "manifest.json"
    path.write_text(json.dumps(doc))
    return path


# ------------------------------------------------------- the protocol holds --

def test_the_adapter_implements_the_environment_and_plant_protocols(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path))
    assert isinstance(env, Environment)
    assert isinstance(env, PlantProvider)
    assert missing_capabilities(env) == {"EVProvider": sorted(
        getattr(EVProvider, "__protocol_attrs__"))}, (
        "a site with no chargers must fail EVProvider and nothing else")


def test_the_observation_vector_is_the_canonical_layout(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path))
    assert env.obs_names == NAMES
    assert env.obs_dim == len(NAMES)
    assert env.num_buildings == 2
    obs, info = env.reset()
    assert info["open_loop"] is True
    assert len(obs) == 2 and obs[0].shape == (len(NAMES),)
    assert obs[0][env.index_of("net_electricity_consumption")] == pytest.approx(2.0)
    assert obs[1][env.index_of("net_electricity_consumption")] == pytest.approx(3.0)


def test_the_plant_models_are_the_same_classes_the_simulator_adapter_returns(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path))
    assert isinstance(env.battery_model(), BatteryModel)
    assert isinstance(env.dhw_tank_model(), TankModel)
    np.testing.assert_allclose(env.battery_info()["soc_rate"], 0.5)


def test_a_shield_built_against_the_protocol_runs_on_logged_data(tmp_path):
    """The point of the seam: no CityLearn in this test, same shield object."""
    env = CSVReplayEnvironment(write_site(tmp_path))
    cfg = CBFConfig(SOC_min=0.1, SOC_max=0.9, P_grid_max=50.0, P_building_max=20.0)
    shield = CBFShield(cfg, num_buildings=env.num_buildings,
                       battery_model=env.battery_model(),
                       nominal_power=env.battery_info()["nominal_power"],
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=SafetyConfig(anticipatory=False, robust_margins=False))
    portfolio = FlexibilityPortfolio.from_environment(env, cfg)
    assert [l.name for l in portfolio.loads] == ["battery"]

    obs, _ = env.reset()
    for _ in range(8):
        actions = np.ones((env.num_buildings, env.action_dim), dtype=np.float32)
        safe = shield.project(actions, obs)
        # soc 0.5, rate 0.5 per step, ceiling 0.9 less the 1e-3 `soc_tolerance` the
        # shield always holds back: (0.899 - 0.5) / 0.5 = 0.798.
        np.testing.assert_allclose(safe[:, env.electrical_storage_action_index], 0.798,
                                   atol=1e-5)
        obs, rewards, term, trunc, info = env.step(safe)
        assert rewards == [0.0] * env.num_buildings
        assert info["open_loop"] is True
    assert not term


def test_the_run_terminates_at_the_end_of_the_log(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path, rows=6))
    env.reset()
    done = False
    n = 0
    while not done and n < 20:
        _o, _r, done, _t, _i = env.step(np.zeros((2, 3), dtype=np.float32))
        n += 1
    assert done and n == 5


# ------------------------------------------------------ what it refuses to do --

def test_a_missing_observation_column_is_fatal_and_not_zero_filled(tmp_path):
    with pytest.raises(ReplayManifestError, match="missing observation column"):
        CSVReplayEnvironment(write_site(tmp_path, drop="occupant_count"))


def test_a_state_of_charge_reported_in_percent_is_caught(tmp_path):
    with pytest.raises(ReplayManifestError, match="electrical_storage_soc"):
        CSVReplayEnvironment(write_site(tmp_path, bad_soc=True))


def test_an_irregular_clock_is_caught(tmp_path):
    with pytest.raises(ReplayManifestError, match="step is"):
        CSVReplayEnvironment(write_site(tmp_path, bad_clock=True))


def test_buildings_of_different_lengths_are_refused(tmp_path):
    write_site(tmp_path, rows=24, buildings=("b1", "b2"))
    with (tmp_path / "b2.csv").open() as fh:
        rows = list(csv.reader(fh))
    with (tmp_path / "b2.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows[:10])
    with pytest.raises(ReplayManifestError, match="one clock"):
        CSVReplayEnvironment(tmp_path / "manifest.json")


def test_a_site_with_no_plant_block_is_an_environment_but_not_a_plant_provider(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path, plant=False))
    assert isinstance(env, Environment)
    assert env.dhw_tank_model() is None
    with pytest.raises(ReplayManifestError, match="will not invent plant parameters"):
        env.battery_model()


def test_the_adapter_does_not_claim_to_know_what_the_plant_executed(tmp_path):
    """Returning the command as the executed action would be believed by every metric."""
    env = CSVReplayEnvironment(write_site(tmp_path))
    env.reset()
    env.step(np.full((2, 3), 0.7, dtype=np.float32))
    np.testing.assert_allclose(env.executed_actions, 0.0)
    assert "executed actions are unknown" in " ".join(env.plant_notes)


def test_the_open_loop_limitation_is_declared_not_buried(tmp_path):
    env = CSVReplayEnvironment(write_site(tmp_path))
    notes = " ".join(env.plant_notes)
    assert "open loop" in notes
    assert "reward is 0.0" in notes
    assert "not fitted to this site" in notes


def test_the_manifest_template_lists_every_column_the_adapter_requires(tmp_path):
    path = write_manifest_template(tmp_path / "m.json", ["h1"])
    doc = json.loads(path.read_text())
    assert doc["required_csv_columns"] == ["timestamp"] + NAMES
    assert doc["buildings"][0]["battery"]["capacity_kwh"] is None, (
        "the template must leave plant parameters blank rather than supply a default")
