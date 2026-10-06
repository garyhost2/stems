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


# --- audit B5, as corrected: two district series, named and kept apart -------
#
# e_bt   building b's net electricity consumption at step t, kW, positive = import
# S_t    signed district series,  S_t = sum_b e_bt            (CityLearn's convention)
# G_t    district import series,  G_t = sum_b max(e_bt, 0)    (what a cap constrains)
#
# The audit called the coexistence of the two a bug and asked for G everywhere. That is
# wrong: CityLearn's own CostFunction.peak consumes the signed series, so moving
# avg_daily_peak onto G would make it disagree with the simulator and with the Table I
# normalisation paper_table.py compares against. Both series are legitimate; the defect
# was that the split was implicit. These tests pin which KPI is in which family.

from stems.metrics import (CITYLEARN_COMPARABLE_KPIS, CONSTRAINT_FACING_KPIS,
                           district_import_kw, district_signed_kw)


def _exporting_run():
    """Three steps, two buildings, an exporter present at every step, so S_t != G_t.

        t=0: (10, -6) -> S=4,  G=10
        t=1: ( 4, -6) -> S=-2, G=4
        t=2: ( 7, -1) -> S=6,  G=7
    """
    m = calc(B=2, cbf_config=CBFConfig(P_grid_max=5.0, P_building_max=100.0))
    for e0, e1 in ((10.0, -6.0), (4.0, -6.0), (7.0, -1.0)):
        m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
                   [obs(net=e0), obs(net=e1)])
    return m, np.array([[10.0, -6.0], [4.0, -6.0], [7.0, -1.0]])


def test_the_two_district_series_differ_whenever_a_building_exports():
    _, net = _exporting_run()
    s, g = district_signed_kw(net), district_import_kw(net)
    assert s == pytest.approx([4.0, -2.0, 6.0])
    assert g == pytest.approx([10.0, 4.0, 7.0])
    assert not np.allclose(s, g), (
        "if these ever coincide on exporting data the two families have been silently "
        "unified and the regression guard below is testing nothing")


def test_citylearn_comparable_kpis_use_the_signed_series():
    m, net = _exporting_run()
    r = m.compute_all()
    s = district_signed_kw(net)
    assert r["avg_daily_peak"] == pytest.approx(float(np.maximum(s, 0.0).max()))
    assert r["ramping_rate"] == pytest.approx(float(np.abs(np.diff(s)).mean()))
    # and demonstrably not the import series
    g = district_import_kw(net)
    assert r["ramping_rate"] != pytest.approx(float(np.abs(np.diff(g)).mean()))


def test_constraint_facing_kpis_use_the_import_series():
    m, net = _exporting_run()
    r = m.compute_all()
    g = district_import_kw(net)
    assert r["peak_import_kw"] == pytest.approx(float(g.max()))          # 10
    assert r["cap_exceedance_kwh"] == pytest.approx(
        float(np.maximum(g - 5.0, 0.0).sum()))                            # 5 + 0 + 2
    assert r["grid_violation_rate"] == pytest.approx(2 / 3)
    assert r["load_factor"] == pytest.approx(float(g.mean() / g.max()))
    # and demonstrably not the signed series
    s = district_signed_kw(net)
    assert r["peak_import_kw"] != pytest.approx(float(s.max()))


def test_avg_daily_peak_equals_citylearns_cost_function():
    """The scalar must match CityLearn's published KPI, or it is not comparable.

    ``CostFunction.peak`` groups the signed series into 24-step windows, takes each
    window's max, then a running mean; its final element is the plain mean of the daily
    maxima, which is what this repository reports. Checked on three days of synthetic
    data that exports heavily, so the signed and import series differ at most hours.
    """
    from citylearn.cost_function import CostFunction

    rng = np.random.default_rng(0)
    net = rng.normal(3.0, 12.0, size=(72, 2)).astype(np.float32)
    m = calc(B=2)
    for row in net:
        m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
                   [obs(net=float(row[0])), obs(net=float(row[1]))])
    s = district_signed_kw(net)
    assert (np.asarray([s[i:i + 24].max() for i in range(0, 72, 24)]) > 0).all(), (
        "this fixture needs every day to have at least one importing hour, because the "
        "repository floors a day's peak at zero and CityLearn does not")
    assert m.compute_all()["avg_daily_peak"] == pytest.approx(
        CostFunction.peak(list(map(float, s)))[-1], rel=1e-5)


def test_the_two_kpi_families_are_declared_and_disjoint():
    assert set(CITYLEARN_COMPARABLE_KPIS).isdisjoint(CONSTRAINT_FACING_KPIS)
    m, _ = _exporting_run()
    r = m.compute_all()
    for key in CITYLEARN_COMPARABLE_KPIS + CONSTRAINT_FACING_KPIS:
        assert key in r, f"{key} is declared in a family but not produced"


def test_both_families_agree_when_nobody_exports():
    """Without exports S == G, so the distinction is invisible on import-only data."""
    m = calc(B=2)
    net = np.array([[3.0, 1.0], [5.0, 2.0], [1.0, 1.0]])
    for e0, e1 in net:
        m.add_step([obs(), obs()], np.zeros((2, 3), np.float32),
                   [obs(net=e0), obs(net=e1)])
    assert district_signed_kw(net) == pytest.approx(district_import_kw(net))
    r = m.compute_all()
    assert r["avg_daily_peak"] == pytest.approx(7.0)
    assert r["peak_import_kw"] == pytest.approx(7.0)
