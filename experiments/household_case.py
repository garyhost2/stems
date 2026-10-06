"""E0 -- pin the household-benefit case, then quantify how much the case choice matters.

What this script answers
------------------------
Every "X% cost saving" in the CityLearn literature is a saving *under a tariff*. The
Travis County building stock used here ships no tariff and no carbon-intensity series,
so both are borrowed from ``citylearn_challenge_2022_phase_all`` -- a neighbourhood in
Fontana, California. This script prices the same physical rollouts under the borrowed
Californian tariff and under the two rates that a Travis County household can actually
be on, and reports the four household outcomes (bill, CO2, comfort, peak contribution)
for each.

The point is not that one tariff is right. It is that the household-benefit headline is
a property of the (tariff, emission factor) pair, which must therefore be pinned once,
before any grid is run, and reported alongside every number it produces.

Usage
-----
    python -m experiments.household_case --season year --arms idle rbc

Outputs ``experiments/diagnostics/household_case/case.json`` and ``case.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "experiments" / "diagnostics" / "household_case"

# --------------------------------------------------------------------------------------
# The pinned case. Changing anything here changes every household number downstream, so
# it is written once, in one place, with its source attached.
# --------------------------------------------------------------------------------------

#: Comfort band tolerance, degC, either side of the CityLearn set points. ASHRAE 55
#: operative-temperature comfort zones are about 2.5 degC wide either side of neutral
#: for typical residential clothing and air speed; 1.0 degC is deliberately tighter than
#: that so the reported in-band fraction is a conservative reading rather than a
#: flattering one. This is a modelling choice, not a measured constant.
COMFORT_TOLERANCE_C = 1.0
COMFORT_SOURCE = (
    "Chosen for this study: +/-1.0 degC around the CityLearn cooling and heating set "
    "points, tighter than the ASHRAE 55 comfort zone width, counted over occupied "
    "hours only. Not a measured constant; reported with every comfort number.")

#: District import cap, kW, for the 8-building neighbourhood. This is the scenario
#: default already used throughout experiments/scenario.py, restated here so the
#: household case does not silently depend on a value defined elsewhere.
DISTRICT_CAP_KW = 300.0

#: EV penetration for the household case, as a fraction of households with a charger.
#: ``None`` means the EV-free case; the EV grids (E5) set it explicitly.
EV_PENETRATION = None


def case_variants(
    price: np.ndarray, carbon: np.ndarray, hour: np.ndarray, day_type: np.ndarray
) -> Dict[str, Any]:
    """Build the pinned (tariff, emission factor) variants.

    Three tariffs:

    ``citylearn_tou``
        The shipped Californian time-of-use series. The only variant comparable with
        published CityLearn numbers, and not a statement about Travis County.
    ``austin_standard``
        Austin Energy Residential Service, the default rate for the actual buildings.
        Flat in time.
    ``austin_tou_pilot``
        Austin Energy's residential time-of-use pilot, the only local rate with an
        intra-day signal, capped at 100 meters.

    Two emission factors: the shipped Californian hourly series, and the ERCOT annual
    average. The second cannot express a shifting benefit; see
    :class:`stems.household.ConstantEmissionFactor`.
    """
    from stems.household import (
        ComfortBand, HouseholdCase, austin_energy_residential,
        austin_energy_residential_tou_pilot, citylearn_2022_emissions,
        citylearn_2022_tariff, egrid_erct_delivered)

    tariffs = {
        "citylearn_tou": citylearn_2022_tariff(price),
        "austin_standard": austin_energy_residential(),
        "austin_tou_pilot": austin_energy_residential_tou_pilot(hour, day_type),
    }
    factors = {
        "citylearn_hourly": citylearn_2022_emissions(carbon),
        "erct_annual": egrid_erct_delivered(),
    }
    band = ComfortBand(tolerance_c=COMFORT_TOLERANCE_C, source=COMFORT_SOURCE)
    return {
        f"{tn}|{fn}": HouseholdCase(
            tariff=t, emissions=f, comfort=band,
            district_cap_kw=DISTRICT_CAP_KW, ev_penetration=EV_PENETRATION)
        for tn, t in tariffs.items() for fn, f in factors.items()
    }, tariffs, factors


# --------------------------------------------------------------------------------------
# Rollout recording
# --------------------------------------------------------------------------------------


def calendar(schema_path: Path, start: int, end: int) -> Dict[str, np.ndarray]:
    """Read ``month``, ``hour`` and ``day_type`` for the window from the dataset itself.

    These come from the building CSV rather than from the observation vector because the
    observation vector carries no ``month`` at all, and because the CSV is the
    authoritative calendar: ``hour`` is 1..24 and ``day_type`` is 1=Monday..7=Sunday,
    both verified against the 2018 calendar for this dataset.
    """
    import pandas as pd

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    root = Path(schema["root_directory"])
    included = [v for v in schema["buildings"].values() if v.get("include", True)]
    if not included:
        raise RuntimeError(f"{schema_path} includes no buildings")
    csv = root / included[0]["energy_simulation"]
    df = pd.read_csv(csv, usecols=["month", "hour", "day_type"])
    out = {k: df[k].to_numpy()[start:end + 1].astype(np.int64)
           for k in ("month", "hour", "day_type")}
    n = end - start + 1
    for k, v in out.items():
        if v.shape != (n,):
            raise RuntimeError(
                f"{csv} gave {v.shape[0]} rows of {k} for a {n}-step window")
    return out


def record_rollout(controller, env, config, max_steps: int) -> Dict[str, np.ndarray]:
    """Step the controller and keep the per-building series the household case needs.

    Deliberately a separate function from ``experiments.runner.evaluate``: that one
    returns aggregated KPIs, and aggregating before billing would make the monthly tier
    structure and the coincident-peak attribution impossible to compute.
    """
    from stems.observations import obs_indices
    from stems.utils import HistoryBuffer

    i_net, i_tin, i_cool, i_occ, i_solar = obs_indices(
        "net_electricity_consumption", "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_cooling_set_point", "occupant_count",
        "solar_generation")
    i_heat = env.heating_setpoint_idx

    cols: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("net", "t_in", "t_cool", "t_heat", "occ", "pv")}
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    n, done = 0, False
    while not done:
        actions = controller.select_action(obs, hist.get(), explore=False)
        nxt, _, term, trunc, _ = env.step(actions)
        if hasattr(controller, "observe"):
            controller.observe(nxt, env.ev_draw_kwh)

        def grab(src, idx):
            return np.array([o[idx] for o in src], dtype=np.float64)

        # Flows and states are read AFTER the step, set points and occupancy BEFORE it,
        # matching stems.metrics.MetricsCalculator.add_step so the two agree.
        cols["net"].append(grab(nxt, i_net))
        cols["t_in"].append(grab(nxt, i_tin))
        cols["pv"].append(grab(nxt, i_solar))
        cols["t_cool"].append(grab(obs, i_cool))
        cols["occ"].append(grab(obs, i_occ))
        cols["t_heat"].append(grab(obs, i_heat) if i_heat is not None
                              else grab(obs, i_cool) - 4.0)
        hist.update(nxt)
        obs = nxt
        n += 1
        done = bool(term or trunc) or n >= max_steps
    out = {k: np.stack(v, axis=0) for k, v in cols.items()}
    out["steps"] = np.int64(n)
    return out


def run_arm(arm_name: str, scenario, seed: int, log) -> Dict[str, np.ndarray]:
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import make_config
    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed

    arm = ARMS[arm_name]
    if arm.learns:
        raise ValueError(
            f"arm {arm_name!r} learns; E0 pins the case on non-learning references only "
            "so that it costs no training compute and cannot be tuned to a tariff")
    set_seed(seed)
    kwargs = scenario.env_kwargs("eval")
    (start, end), = kwargs["episode_time_steps"]
    env = STEMSEnvironment(
        schema=scenario.schema_path(), seed=seed, heat_pump=scenario.heat_pump,
        env_kwargs=kwargs, hvac_control=scenario.hvac_control,
        allow_missing_obs=scenario.allow_missing_obs)
    config = make_config(scenario, {})
    if env.hvac_action_index < 0:
        config.reward.lambda_indoor = 0.0
    controller = build_controller(arm, env, config)
    t0 = time.time()
    rec = record_rollout(controller, env, config, end - start + 1)
    log(f"  {arm_name}: {int(rec['steps'])} steps in {time.time() - t0:.1f}s")
    rec["window"] = np.array([start, end], dtype=np.int64)
    return rec


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def summarise(benefit, pv_credit_usd: Optional[np.ndarray]) -> Dict[str, Any]:
    b = benefit
    out: Dict[str, Any] = {
        "households": int(b.bill_usd.size),
        "bill_usd_mean": float(b.bill_usd.mean()),
        "bill_usd_total": float(b.bill_usd.sum()),
        "bill_components_mean": {k: float(v.mean())
                                 for k, v in b.bill_components.items()},
        "co2_kg_mean": float(b.co2_kg.mean()),
        "import_kwh_mean": float(b.import_kwh.mean()),
        "export_kwh_mean": float(b.export_kwh.mean()),
        "comfort_fraction_in_band_mean": float(b.comfort_fraction_in_band.mean()),
        "gross_load_kwh_mean": (None if b.gross_load_kwh is None
                                else float(b.gross_load_kwh.mean())),
        "pv_kwh_mean": None if b.pv_kwh is None else float(b.pv_kwh.mean()),
        "comfort_hours_assessed_mean": float(b.comfort_hours_assessed.mean()),
        "own_peak_kw_mean": float(b.own_peak_kw.mean()),
        "coincident_peak_kw_mean": float(b.coincident_peak_kw.mean()),
        "district_peak_kw": float(b.district_peak_kw),
        "district_peak_step": int(b.district_peak_step),
        "cap_exceedance_kwh": b.cap_exceedance_kwh,
        "steps": int(b.steps),
    }
    if pv_credit_usd is not None:
        out["value_of_solar_credit_usd_mean"] = float(pv_credit_usd.mean())
        out["bill_net_of_solar_credit_usd_mean"] = float(
            (b.bill_usd - pv_credit_usd).mean())
    return out


def markdown(report: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# E0 -- the household-benefit case\n")
    L.append(f"Generated {report['generated']}. "
             f"Scenario `{report['scenario']['key']}`, window "
             f"{report['window'][0]}-{report['window'][1]} "
             f"({report['steps']} transitions), {report['households']} households, "
             f"seed {report['seed']}.\n")
    L.append("## The case, pinned\n")
    L.append(f"- Comfort band: +/-{COMFORT_TOLERANCE_C} degC around the CityLearn set "
             f"points, occupied hours only.")
    L.append(f"- District import cap: {DISTRICT_CAP_KW} kW.")
    L.append(f"- EV penetration: {EV_PENETRATION}.\n")
    L.append("### Tariffs\n")
    for name, src in report["tariff_sources"].items():
        L.append(f"- **`{name}`** -- {src}\n")
    L.append("### Emission factors\n")
    for name, src in report["factor_sources"].items():
        L.append(f"- **`{name}`** -- {src}\n")

    L.append("## Price signal, by tariff\n")
    L.append("| tariff | mean marginal USD/kWh | min | max | max/min | "
             "time-varying |")
    L.append("|---|---:|---:|---:|---:|:--:|")
    for name, s in report["price_signal"].items():
        L.append(f"| `{name}` | {s['mean']:.5f} | {s['min']:.5f} | {s['max']:.5f} | "
                 f"{s['ratio']:.3f} | {'yes' if s['time_varying'] else 'no'} |")
    L.append("")

    L.append("## Annual household outcome, by arm and case\n")
    L.append("| arm | tariff | emissions | bill USD | CO2 kg | comfort in band | "
             "own peak kW | coincident peak kW |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|")
    for arm, per_case in report["arms"].items():
        for key, s in per_case["cases"].items():
            tn, fn = key.split("|")
            L.append(f"| `{arm}` | `{tn}` | `{fn}` | {s['bill_usd_mean']:.2f} | "
                     f"{s['co2_kg_mean']:.1f} | "
                     f"{s['comfort_fraction_in_band_mean'] * 100:.1f}% | "
                     f"{s['own_peak_kw_mean']:.2f} | "
                     f"{s['coincident_peak_kw_mean']:.2f} |")
    L.append("")

    L.append("## Austin Energy with the Value-of-Solar rider\n")
    L.append("Gross-billed, production-credited: billable usage is on-site "
             "photovoltaic self-consumption plus delivered energy, and the credit is "
             "on gross photovoltaic output. Net metering is not available.\n")
    L.append("| arm | tariff | gross-billed USD | VoS credit USD | net bill USD |")
    L.append("|---|---|---:|---:|---:|")
    for arm, per_case in report["arms"].items():
        for key, s_ in per_case["cases"].items():
            if "vos_net_bill_usd_mean" not in s_:
                continue
            tn, fn = key.split("|")
            if fn != "erct_annual":
                continue
            L.append(f"| `{arm}` | `{tn}` | {s_['vos_gross_billed_usd_mean']:.2f} | "
                     f"{s_['vos_credit_usd_mean']:.2f} | "
                     f"{s_['vos_net_bill_usd_mean']:.2f} |")
    L.append("")

    cap = report.get("cap_diagnostic")
    if cap:
        L.append("## Is the district cap binding?\n")
        L.append(f"Scenario default cap: **{cap['scenario_cap_kw']} kW**. Measured "
                 f"district import peak, reference arm `{cap['arm']}`: "
                 f"**{cap['peak_import_kw']:.2f} kW** "
                 f"({cap['peak_over_cap_pct']:.1f}% of the cap), mean "
                 f"{cap['mean_import_kw']:.2f} kW, hours above the cap "
                 f"**{cap['hours_above_cap']}** of {cap['steps']}.\n")
        L.append(f"A cap of {cap['scenario_cap_kw']} kW cannot bind on this "
                 f"neighbourhood, so no grid-constraint arm can be distinguished from "
                 f"an unconstrained one under it. A cap that binds for this stock is "
                 f"about **{cap['suggested_cap_kw']:.0f} kW** "
                 f"({cap['suggested_fraction'] * 100:.0f}% of the uncontrolled peak).\n")

    if report.get("deltas"):
        L.append("## Saving of each arm against the reference, per household per year\n")
        L.append(f"Reference arm: `{report['reference_arm']}`.\n")
        L.append("| arm | tariff | emissions | bill saved USD | bill saved % | "
                 "CO2 avoided kg | import avoided kWh | export forgone kWh | "
                 "gross load saved kWh | coincident peak cut kW |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm, per_case in report["deltas"].items():
            for key, d in per_case.items():
                tn, fn = key.split("|")
                pct = d["bill_saved_frac_mean"]
                pct_s = "n/a" if pct is None else f"{pct * 100:.2f}%"
                L.append(f"| `{arm}` | `{tn}` | `{fn}` | "
                         f"{d['bill_saved_usd_mean']:.2f} | {pct_s} | "
                         f"{d['co2_avoided_kg_mean']:.1f} | "
                         f"{d['import_avoided_kwh_mean']:.1f} | "
                         f"{d['export_forgone_kwh_mean']:.1f} | "
                         f"{d['gross_load_saved_kwh_mean']:.1f} | "
                         f"{d['coincident_peak_cut_kw_mean']:.3f} |")
        L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="year")
    ap.add_argument("--arms", nargs="+", default=["idle", "rbc"])
    ap.add_argument("--reference-arm", default="idle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schema",
                    default="citylearn_schemas/tx_travis_8b/schema.json")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    import pandas as pd

    from experiments.scenario import Scenario
    from stems.household import benefit_delta, household_benefit, \
        value_of_solar_credit_usd

    def log(msg: str) -> None:
        print(msg, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = Scenario(schema=args.schema, season=args.season)
    schema_path = Path(scenario.schema_path())
    kwargs = scenario.env_kwargs("eval")
    (start, end), = kwargs["episode_time_steps"]

    rollouts: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in args.arms:
        log(f"rollout {arm}")
        rollouts[arm] = run_arm(arm, scenario, args.seed, log)

    # A window of N time steps yields N-1 transitions, so the recorded series is one
    # step shorter than the window. Align every exogenous series to the recorded steps
    # using the SAME convention as stems.metrics.MetricsCalculator.add_step: the price
    # and carbon intensity are read at step t (from ``obs``), the net consumption at
    # step t+1 (from ``next_obs``). Matching it is what keeps the household bill equal
    # to the district ``cost`` KPI.
    steps = {arm: int(rec["steps"]) for arm, rec in rollouts.items()}
    if len(set(steps.values())) != 1:
        raise RuntimeError(f"arms produced different step counts: {steps}")
    n = next(iter(steps.values()))
    cal = calendar(schema_path, start, start + n - 1)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    included = [v for v in schema["buildings"].values() if v.get("include", True)]
    price = pd.read_csv(included[0]["pricing"])["electricity_pricing"].to_numpy()
    carbon = pd.read_csv(included[0]["carbon_intensity"]).iloc[:, 0].to_numpy()
    price, carbon = price[start:start + n], carbon[start:start + n]

    cases, tariffs, factors = case_variants(
        price, carbon, cal["hour"], cal["day_type"])

    log(f"window {start}-{end} ({end - start + 1} time steps, {n} transitions); "
        f"{len(cases)} (tariff, emission factor) variants")

    # Price-signal statistics. For a tiered tariff the "marginal price" reported is the
    # first-block rate plus riders, which is the price a median household actually faces
    # for an extra kWh for most of the month.
    from stems.household import (
        AE_REGULATORY_SECONDARY_USD_PER_KWH, AE_RES_CBC_USD_PER_KWH,
        AE_RES_TIER_RATE_USD_PER_KWH, TieredMonthlyTariff)
    signal: Dict[str, Any] = {}
    for name, t in tariffs.items():
        if isinstance(t, TieredMonthlyTariff):
            base = AE_RES_TIER_RATE_USD_PER_KWH[0]
            if t.time_varying:
                marg = base + np.asarray(t.rider_usd_per_kwh, dtype=np.float64)
            else:
                marg = np.full(1, base + float(t.rider_usd_per_kwh))
        else:
            marg = np.asarray(t.price_usd_per_kwh, dtype=np.float64)
        signal[name] = {"mean": float(marg.mean()), "min": float(marg.min()),
                        "max": float(marg.max()),
                        "ratio": float(marg.max() / marg.min()),
                        "time_varying": bool(np.ptp(marg) > 0)}

    benefits: Dict[str, Dict[str, Any]] = {}
    raw: Dict[str, Dict[str, Any]] = {}
    from stems.household import value_of_solar_bill_usd

    for arm, rec in rollouts.items():
        pv_kwh = np.abs(rec["pv"])
        credit = value_of_solar_credit_usd(pv_kwh)
        per_case, objs = {}, {}
        for key, case in cases.items():
            b = household_benefit(
                net_kw=rec["net"], month=cal["month"], hour=cal["hour"],
                indoor_c=rec["t_in"], cooling_set_c=rec["t_cool"],
                heating_set_c=rec["t_heat"], occupant_count=rec["occ"], case=case,
                pv_kw=rec["pv"])
            objs[key] = b
            s = summarise(b, credit if key.startswith("austin") else None)
            # The bill an Austin Energy customer with PV actually faces: gross-billed,
            # production-credited. This is a different billing rule, not a different
            # tariff, so it is reported alongside rather than as a seventh variant.
            if key.startswith("austin"):
                vos = value_of_solar_bill_usd(
                    case.tariff, (rec["net"] + pv_kwh), pv_kwh,
                    cal["month"], cal["hour"])
                s["vos_gross_billed_usd_mean"] = float(vos["total_usd"].mean())
                s["vos_credit_usd_mean"] = float(vos["vos_credit_usd"].mean())
                s["vos_net_bill_usd_mean"] = float(vos["net_usd"].mean())
            per_case[key] = s
        benefits[arm] = {"cases": per_case}
        raw[arm] = objs

    deltas: Dict[str, Dict[str, Any]] = {}
    ref = args.reference_arm
    if ref in raw:
        for arm in raw:
            if arm == ref:
                continue
            deltas[arm] = {}
            for key in raw[arm]:
                d = benefit_delta(raw[arm][key], raw[ref][key])
                frac = d["bill_saved_frac"]
                deltas[arm][key] = {
                    "bill_saved_usd_mean": float(d["bill_saved_usd"].mean()),
                    "bill_saved_frac_mean": (None if np.all(np.isnan(frac))
                                             else float(np.nanmean(frac))),
                    "co2_avoided_kg_mean": float(d["co2_avoided_kg"].mean()),
                    "import_avoided_kwh_mean": float(d["import_avoided_kwh"].mean()),
                    "export_forgone_kwh_mean": float(d["export_forgone_kwh"].mean()),
                    "gross_load_saved_kwh_mean": float(
                        d["gross_load_saved_kwh"].mean()),
                    "comfort_hours_gained_mean": float(
                        d["comfort_hours_gained"].mean()),
                    "own_peak_cut_kw_mean": float(d["own_peak_cut_kw"].mean()),
                    "coincident_peak_cut_kw_mean": float(
                        d["coincident_peak_cut_kw"].mean()),
                }

    # Is the cap the scenarios already use capable of binding at all? If the
    # uncontrolled district import never approaches it, every constrained arm is
    # indistinguishable from an unconstrained one and the violation-rate column is
    # structurally zero -- which is exactly the shape of the stored results.
    ref_rec = rollouts.get(ref, rollouts[args.arms[0]])
    district = np.maximum(ref_rec["net"], 0.0).sum(axis=1)
    cap_diag = {
        "arm": ref if ref in rollouts else args.arms[0],
        "scenario_cap_kw": float(scenario.grid_cap_kw),
        "peak_import_kw": float(district.max()),
        "mean_import_kw": float(district.mean()),
        "peak_over_cap_pct": float(100.0 * district.max() / scenario.grid_cap_kw),
        "hours_above_cap": int((district > scenario.grid_cap_kw).sum()),
        "steps": int(district.size),
        "suggested_fraction": 0.8,
        "suggested_cap_kw": float(0.8 * district.max()),
    }

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cap_diagnostic": cap_diag,
        "scenario": scenario.describe(),
        "seed": args.seed,
        "window": [int(start), int(end)],
        "steps": int(n),
        "households": int(rollouts[args.arms[0]]["net"].shape[1]),
        "comfort_tolerance_c": COMFORT_TOLERANCE_C,
        "comfort_source": COMFORT_SOURCE,
        "district_cap_kw": DISTRICT_CAP_KW,
        "ev_penetration": EV_PENETRATION,
        "tariff_sources": {k: v.source for k, v in tariffs.items()},
        "factor_sources": {k: v.source for k, v in factors.items()},
        "price_signal": signal,
        "arms": benefits,
        "reference_arm": ref,
        "deltas": deltas,
    }
    (out_dir / "case.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "case.md").write_text(markdown(report), encoding="utf-8")
    log(f"wrote {out_dir / 'case.json'} and {out_dir / 'case.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
