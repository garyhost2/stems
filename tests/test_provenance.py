"""Audit E3 (the seed never reached CityLearn) and E1 (the fingerprint missed the data)."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import code_fingerprint, data_fingerprint
from experiments.scenario import Scenario

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"
EV_SCHEMA = REPO / "citylearn_schemas" / "tx_travis_8b_ev" / "schema.json"

needs_schema = pytest.mark.skipif(not SCHEMA.is_file(),
                                  reason=f"generated schema absent: {SCHEMA}")
needs_ev_schema = pytest.mark.skipif(not EV_SCHEMA.is_file(),
                                     reason=f"generated EV schema absent: {EV_SCHEMA}")


# ------------------------------------------------------------------ audit E3 --

@needs_schema
def test_seeding_citylearn_changes_the_buildings_themselves():
    """Why audit E3's prescription is not implemented. Measured, not argued.

    CityLearnEnv(random_seed=s) overrides the per-building random_seed the schema
    carries and re-randomises device autosizing, so the houses become a function of the
    seed. Seeds are the replication unit *within* a scenario, so forwarding the run seed
    would make five "seeds" five different neighbourhoods.
    """
    from stems.environment import STEMSEnvironment

    def capacities(**kw):
        env = STEMSEnvironment(schema=str(SCHEMA), seed=0, heat_pump=True, **kw)
        return [round(float(b.electrical_storage.capacity), 4)
                for b in env._env.buildings]

    unseeded = capacities()
    # Battery capacities in kWh over the eight Travis buildings. Unseeded, CityLearn
    # uses the per-building random_seed the schema carries, which is deterministic:
    # the same list comes back in a fresh process every time.
    assert unseeded == [10.8, 6.6, 5.0, 9.7, 13.5, 5.4, 16.2, 16.0]
    assert capacities(citylearn_seed=7) == [13.5, 10.0, 5.0, 13.5, 5.4, 13.5, 13.5, 17.5]
    assert capacities(citylearn_seed=4321) == [12.0, 13.5, 3.3, 8.0, 3.5, 10.8, 23.1, 10.5]
    assert capacities(citylearn_seed=7) != unseeded


@needs_schema
def test_the_run_seed_does_not_reach_citylearn_by_default():
    """The building stock must be a property of the schema, not of the replication."""
    from stems.environment import STEMSEnvironment

    caps = {}
    for seed in (0, 7, 4321):
        env = STEMSEnvironment(schema=str(SCHEMA), seed=seed, heat_pump=True)
        assert env.citylearn_seed is None, (
            "the run seed must not be forwarded to CityLearn; it would re-randomise "
            "device autosizing and change the houses between replications")
        caps[seed] = [float(b.electrical_storage.capacity) for b in env._env.buildings]
    assert caps[0] == caps[7] == caps[4321]


@needs_schema
def test_citylearn_can_still_be_seeded_deliberately_and_the_choice_is_reported():
    """Audit E3's real requirement: the record must state what seeded the simulator."""
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(schema=str(SCHEMA), seed=1, heat_pump=True,
                           citylearn_seed=4321)
    assert env.citylearn_seed == 4321
    assert int(getattr(env._env, "random_seed")) == 4321

    env = STEMSEnvironment(schema=str(SCHEMA), seed=1, heat_pump=True,
                           env_kwargs={"random_seed": 99})
    assert env.citylearn_seed == 99
    assert int(getattr(env._env, "random_seed")) == 99


# ------------------------------------------------------------------ audit E1 --

@needs_schema
def test_data_fingerprint_is_stable_for_the_same_experiment():
    a, b = data_fingerprint(str(SCHEMA)), data_fingerprint(str(SCHEMA))
    assert a["data_fingerprint"] == b["data_fingerprint"]
    assert a["data_files"] > 0


@needs_schema
def test_data_fingerprint_covers_the_files_the_code_fingerprint_misses():
    fp = data_fingerprint(str(SCHEMA))
    covered = fp["data_files"]
    assert covered >= 10, (
        "expected the schema plus the ResStock series, weather, price, carbon and "
        f"dynamics checkpoints; only {covered} files were hashed")
    assert fp["data_missing"] == [], f"unresolved data references: {fp['data_missing']}"
    assert fp["citylearn"] not in ("unknown", "unavailable")


@needs_schema
def test_two_different_schemas_get_different_data_fingerprints(tmp_path):
    """The failure E1 names: same code fingerprint, different experiment."""
    doc = json.loads(SCHEMA.read_text())
    included = [n for n, c in doc["buildings"].items() if c.get("include")]
    assert len(included) > 1
    doc["buildings"][included[-1]]["include"] = False   # a 7-building experiment
    other = tmp_path / "schema.json"
    other.write_text(json.dumps(doc))

    assert code_fingerprint()["fingerprint"] == code_fingerprint()["fingerprint"]
    assert (data_fingerprint(str(SCHEMA))["data_fingerprint"]
            != data_fingerprint(str(other))["data_fingerprint"])


@needs_schema
def test_the_fingerprint_does_not_depend_on_the_install_path(tmp_path):
    """root_directory is machine-specific, so it must not enter the digest."""
    doc = json.loads(SCHEMA.read_text())
    original_root = doc["root_directory"]
    doc["root_directory"] = "/somewhere/else/entirely"
    moved = tmp_path / "schema.json"
    moved.write_text(json.dumps(doc))
    # The referenced building files resolve through root_directory, so point the copy at
    # the real root via absolute paths to isolate the root_directory field itself.
    doc_abs = json.loads(SCHEMA.read_text())
    for cfg in doc_abs["buildings"].values():
        if not cfg.get("include"):
            continue
        for key in ("energy_simulation", "weather"):
            if isinstance(cfg.get(key), str) and not Path(cfg[key]).is_absolute():
                cfg[key] = str(Path(original_root) / cfg[key])
        dyn = cfg.get("dynamics")
        if isinstance(dyn, dict):
            for k, v in (dyn.get("attributes") or {}).items():
                if isinstance(v, str) and v.endswith(".pth") and not Path(v).is_absolute():
                    dyn["attributes"][k] = str(Path(original_root) / v)
    a = dict(doc_abs, root_directory=original_root)
    b = dict(doc_abs, root_directory="/a/completely/different/place")
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert (data_fingerprint(str(pa))["data_fingerprint"]
            == data_fingerprint(str(pb))["data_fingerprint"])


@needs_ev_schema
def test_charger_csvs_enter_the_fingerprint_but_line_endings_do_not(tmp_path):
    """Audit E2: CRLF drift is not a data change and must not move the digest."""
    staging = tmp_path / "tx_travis_8b_ev"
    staging.mkdir()
    shutil.copy(EV_SCHEMA, staging / "schema.json")
    chargers = sorted(EV_SCHEMA.parent.glob("charger_*.csv"))
    assert chargers, "the EV schema directory should carry charger CSVs"
    # Normalise to LF first: the working-tree copies may be in either form, depending on
    # whether the schema has been regenerated since the last checkout.
    for c in chargers:
        (staging / c.name).write_bytes(c.read_bytes().replace(b"\r\n", b"\n"))

    before = data_fingerprint(str(staging / "schema.json"))["data_fingerprint"]

    # identical content, CRLF line endings
    for c in chargers:
        target = staging / c.name
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert data_fingerprint(str(staging / "schema.json"))["data_fingerprint"] == before

    # a real one-character data change must move it
    target = staging / chargers[0].name
    lines = target.read_bytes().split(b"\r\n")
    assert len(lines) > 2
    lines[1] = lines[1] + b",0"
    target.write_bytes(b"\r\n".join(lines))
    assert data_fingerprint(str(staging / "schema.json"))["data_fingerprint"] != before


@needs_schema
def test_a_missing_data_file_is_reported_not_swallowed(tmp_path):
    doc = json.loads(SCHEMA.read_text())
    for cfg in doc["buildings"].values():
        if cfg.get("include"):
            cfg["energy_simulation"] = "this_file_does_not_exist.csv"
            break
    broken = tmp_path / "schema.json"
    broken.write_text(json.dumps(doc))
    fp = data_fingerprint(str(broken))
    assert "this_file_does_not_exist.csv" in fp["data_missing"]


def test_data_fingerprint_survives_a_missing_schema():
    fp = data_fingerprint("citylearn_schemas/does_not_exist/schema.json")
    assert fp["data_fingerprint"]
    assert fp["data_missing"] == ["citylearn_schemas/does_not_exist/schema.json"]


def test_the_original_code_fingerprint_field_is_untouched():
    """400+ stored records carry `fingerprint`; its meaning must not change."""
    fp = code_fingerprint()
    assert set(fp) == {"git", "fingerprint", "files", "citylearn"}
    assert len(fp["fingerprint"]) == 16


@needs_schema
def test_a_run_record_would_carry_both_digests():
    from experiments.runner import code_fingerprint as cf

    scenario = Scenario(schema=str(SCHEMA), season="summer", days=14)
    meta = {"code": cf(), "data": data_fingerprint(scenario.schema_path())}
    assert meta["code"]["fingerprint"] != meta["data"]["data_fingerprint"]
    assert meta["data"]["data_files"] > 0
