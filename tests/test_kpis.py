from __future__ import annotations

import math

import numpy as np
import pytest

from stems.config import CBFConfig
from stems.metrics import MetricsCalculator

D = 35
IDX = dict(carbon=14, t_in=15, load=16, solar=17, soc_dhw=18, soc=19, net=20,
           price=21, dhw=25, occ=26, t_set=27)
EV = dict(connected_state=30, departure_time=31, required_soc_departure=32,
          soc=33, battery_capacity=34)


def obs(**kw) -> np.ndarray:
    o = np.zeros(D, dtype=np.float32)
    o[IDX["t_set"]] = 24.0
    o[IDX["t_in"]] = 24.0
    o[IDX["occ"]] = 1.0
    o[IDX["price"]] = 0.2
    o[IDX["soc"]] = 0.5
    for key, value in kw.items():
        if key.startswith("ev_"):
            o[EV[key[3:]]] = value
        else:
            o[IDX[key]] = value
    return o


def calc(**kw) -> MetricsCalculator:
    kw.setdefault("cbf_config", CBFConfig())
    return MetricsCalculator(num_buildings=kw.pop("B", 1), **kw)


def a(*values) -> np.ndarray:
    return np.array([values], dtype=np.float32)


def test_pv_self_consumption_and_self_sufficiency():
    m = calc()
    m.add_step([obs(solar=4.0)], a(0, 0, 0), [obs(net=1.0)])
    m.add_step([obs(solar=4.0)], a(0, 0, 0), [obs(net=-3.0)])
    r = m.compute_all()
    assert r["pv_self_consumption"] == pytest.approx(5 / 8)
    assert r["self_sufficiency"] == pytest.approx(5 / 6)


def test_pv_kpis_are_nan_without_pv():
    m = calc()
    m.add_step([obs()], a(0, 0, 0), [obs(net=1.0, solar=0.0)])
    assert math.isnan(m.compute_all()["pv_self_consumption"])


def test_peak_load_factor_and_cap_exceedance():
    m = calc(cbf_config=CBFConfig(P_grid_max=2.0, P_building_max=100.0))
    m.add_step([obs()], a(0, 0, 0), [obs(net=3.0)])
    m.add_step([obs()], a(0, 0, 0), [obs(net=1.0)])
    r = m.compute_all()
    assert r["peak_import_kw"] == pytest.approx(3.0)
    assert r["load_factor"] == pytest.approx(2.0 / 3.0)
    assert r["cap_exceedance_kwh"] == pytest.approx(1.0)


def test_exports_do_not_offset_another_buildings_import_for_peak():
    m = calc(B=2)
    m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
               [obs(net=5.0), obs(net=-4.0)])
    assert m.compute_all()["peak_import_kw"] == pytest.approx(5.0)


def test_discomfort_degree_hours_counts_severity_beyond_the_band():
    m = calc()
    m.add_step([obs()], a(0, 0, 0), [obs(t_in=27.0)])
    m.add_step([obs()], a(0, 0, 0), [obs(t_in=27.0)])
    m.add_step([obs()], a(0, 0, 0), [obs(t_in=25.5)])
    assert m.compute_all()["discomfort_degree_hours"] == pytest.approx(2.0)


def test_unoccupied_hours_do_not_count_as_discomfort():
    m = calc()
    m.add_step([obs(occ=0.0)], a(0, 0, 0), [obs(t_in=30.0)])
    assert m.compute_all()["discomfort_degree_hours"] == pytest.approx(0.0)


def test_worst_building_discomfort():
    m = calc(B=2)
    m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
               [obs(t_in=30.0), obs(t_in=24.0)])
    r = m.compute_all()
    assert r["discomfort_rate_worst_building"] == pytest.approx(1.0)
    assert r["discomfort_rate"] == pytest.approx(0.5)


def test_battery_equivalent_full_cycles():
    m = calc()
    m.add_step([obs(soc=0.2)], a(0, 1, 0), [obs(soc=0.6)])
    m.add_step([obs(soc=0.6)], a(0, -1, 0), [obs(soc=0.2)])
    assert m.compute_all()["battery_equivalent_full_cycles"] == pytest.approx(0.4)


def test_hvac_on_transitions_per_building_day():
    m = calc(hvac_idx=2)
    for hvac in (0.0, 0.5, 0.5, 0.0, 0.6):
        m.add_step([obs()], a(0, 0, hvac), [obs()])
    assert m.compute_all()["hvac_on_transitions_per_building_day"] == pytest.approx(
        2 / (5 / 24))


def test_hvac_kpi_omitted_without_index():
    m = calc()
    m.add_step([obs()], a(0, 0, 1), [obs()])
    assert "hvac_on_transitions_per_building_day" not in m.compute_all()


def test_barrier_intervention_only_counts_controlled_actuators():
    m = calc(control_indices=[1])
    m.add_step([obs()], a(0.0, 0.2, 0.3), [obs()], raw_actions=a(0.9, 0.5, 0.3))
    m.add_step([obs()], a(0.0, 0.1, 0.0), [obs()], raw_actions=a(0.0, 0.1, 0.0))
    r = m.compute_all()
    assert r["barrier_intervention_rate"] == pytest.approx(0.5)
    assert r["barrier_intervention_magnitude"] == pytest.approx(0.15)


def test_intervention_kpis_omitted_without_raw_actions():
    m = calc()
    m.add_step([obs()], a(0, 0, 0), [obs()])
    r = m.compute_all()
    assert "barrier_intervention_rate" not in r


def test_ev_missed_departure_and_shortfall():
    m = calc(ev_layout=EV)
    m.add_step([obs(ev_connected_state=1, ev_soc=0.5, ev_required_soc_departure=0.8,
                    ev_battery_capacity=60)],
               a(0, 0, 0), [obs(ev_connected_state=0)])
    m.add_step([obs(ev_connected_state=1, ev_soc=0.9, ev_required_soc_departure=0.8,
                    ev_battery_capacity=40)],
               a(0, 0, 0), [obs(ev_connected_state=0)])
    m.add_step([obs(ev_connected_state=1, ev_soc=0.1, ev_required_soc_departure=0.8,
                    ev_battery_capacity=40)],
               a(0, 0, 0), [obs(ev_connected_state=1, ev_soc=0.2)])
    r = m.compute_all()
    assert r["ev_departures"] == 2
    assert r["ev_missed_departures"] == 1
    assert r["ev_missed_departure_rate"] == pytest.approx(0.5)
    assert r["ev_energy_shortfall_kwh"] == pytest.approx(18.0)


def test_ev_kpis_omitted_without_layout():
    m = calc()
    m.add_step([obs()], a(0, 0, 0), [obs()])
    assert "ev_departures" not in m.compute_all()


def test_existing_table_one_keys_are_unchanged():
    m = calc(cbf_config=CBFConfig(P_grid_max=2.0, P_building_max=100.0))
    m.add_step([obs(price=0.5, carbon=0.4)], a(0, 0, 0), [obs(net=3.0)])
    m.add_step([obs(price=0.1, carbon=0.2)], a(0, 0, 0), [obs(net=1.0)])
    r = m.compute_all()
    assert r["cost"] == pytest.approx(3.0 * 0.5 + 1.0 * 0.1)
    assert r["emission"] == pytest.approx(3.0 * 0.4 + 1.0 * 0.2)
    assert r["electricity_consumption"] == pytest.approx(4.0)
    assert r["ramping_rate"] == pytest.approx(2.0)
    assert r["grid_violation_rate"] == pytest.approx(0.5)
    for key in ("avg_daily_peak", "discomfort_rate", "safety_violation_rate",
                "soc_violation_rate", "avoidable_violation_rate"):
        assert key in r


def test_baseline_normalisation_leaves_extended_kpis_alone():
    m = calc()
    m.add_step([obs()], a(0, 0, 0), [obs(net=2.0, solar=1.0)])
    r = m.compute_all(baseline_metrics={"cost": 2.0, "peak_import_kw": 100.0})
    assert r["peak_import_kw"] == pytest.approx(2.0)


def test_price_of_the_hour_is_read_before_the_action():
    m = calc()
    m.add_step([obs(price=0.10)], a(0, 0, 0), [obs(net=2.0, price=0.90)])
    assert m.compute_all()["cost"] == pytest.approx(2.0 * 0.10)


def test_avoidable_split_is_nan_without_a_measured_rate():
    m = calc()
    m.add_step([obs(soc=0.05)], a(0, 0, 0), [obs(soc=0.05)])
    r = m.compute_all()
    assert r["soc_violation_rate"] == pytest.approx(1.0)
    assert math.isnan(r["avoidable_violation_rate"])


# --- audit B5: one definition of district import for every grid-side KPI -------
#
# District import at step t, kW:  G_t = sum_b max(e_bt, 0)
# where e_bt is building b's net electricity consumption. The signed alternative
# sum_b e_bt lets one building's export cancel another's import, which no physical
# path in the model permits and which none of the CBF, the fleet shield or the reward
# does. These tests pin G_t for the two KPIs that used the signed sum.

def test_exports_do_not_offset_another_buildings_import_for_avg_daily_peak():
    """Two buildings, +5 kW and -4 kW. Import is 5 kW; the signed sum would say 1 kW."""
    m = calc(B=2)
    m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
               [obs(net=5.0), obs(net=-4.0)])
    r = m.compute_all()
    assert r["avg_daily_peak"] == pytest.approx(5.0)
    assert r["avg_daily_peak"] == pytest.approx(r["peak_import_kw"]), (
        "with a single day the average daily peak is the peak import, by definition")


def test_exports_do_not_offset_another_buildings_import_for_ramping_rate():
    """Ramping is |G_t - G_{t-1}| averaged over t, on the import series.

    Step 1: imports 5 and 0 -> G = 5. Step 2: imports 0 and 0 (one exports 4) -> G = 0.
    So the ramp is 5 kW. Under the signed sum it would have been |1 - (-4)| = 5 by
    coincidence here, so the second pair below breaks the tie: G goes 5 -> 2, ramp 3,
    while the signed sum goes 1 -> -6, ramp 7.
    """
    m = calc(B=2)
    m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
               [obs(net=5.0), obs(net=-4.0)])
    m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
               [obs(net=2.0), obs(net=-8.0)])
    assert m.compute_all()["ramping_rate"] == pytest.approx(3.0)


def test_every_grid_side_kpi_uses_the_same_import_series():
    """peak, average daily peak, ramping, cap exceedance and the violation rate agree.

    Three steps, two buildings, with an exporter present throughout so the signed and
    positive-part definitions differ at every step:
        t=0: (10, -6) -> G=10
        t=1: ( 4, -6) -> G=4
        t=2: ( 7, -1) -> G=7
    G = (10, 4, 7). peak 10; one day so avg_daily_peak 10; ramps |4-10|, |7-4| -> mean
    4.5; with P_grid_max = 5 the exceedance is (10-5) + 0 + (7-5) = 7 kWh at dt = 1 h and
    the violation rate is 2/3.
    """
    m = calc(B=2, cbf_config=CBFConfig(P_grid_max=5.0, P_building_max=100.0))
    for e0, e1 in ((10.0, -6.0), (4.0, -6.0), (7.0, -1.0)):
        m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
                   [obs(net=e0), obs(net=e1)])
    r = m.compute_all()
    assert r["peak_import_kw"] == pytest.approx(10.0)
    assert r["avg_daily_peak"] == pytest.approx(10.0)
    assert r["ramping_rate"] == pytest.approx(4.5)
    assert r["cap_exceedance_kwh"] == pytest.approx(7.0)
    assert r["grid_violation_rate"] == pytest.approx(2 / 3)


def test_the_two_definitions_still_agree_when_nobody_exports():
    """Without exports the change is a no-op, so single-building runs are unaffected."""
    m = calc(B=2)
    for e0, e1 in ((3.0, 1.0), (5.0, 2.0), (1.0, 1.0)):
        m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
                   [obs(net=e0), obs(net=e1)])
    r = m.compute_all()
    signed = np.array([4.0, 7.0, 2.0])
    assert r["avg_daily_peak"] == pytest.approx(float(signed.max()))
    assert r["ramping_rate"] == pytest.approx(float(np.abs(np.diff(signed)).mean()))
