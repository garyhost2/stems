# Changelog

Every entry records **what** changed, **why**, and the **evidence** that the change is right
(a test name, a measured number, or a quoted source line). Entries are appended in the order the
changes landed. Audit finding ids refer to `docs/AUDIT_2026-10-06.md`.

Branch: `audit/2026-10`.

---

## Step 1 — Snapshot and changelog

### 1.1 Branch `audit/2026-10` created and the whole working tree committed

**What.** Branched from `main` at `7d33f43` and committed everything that was uncommitted,
including the two previously untracked items: `results/ws_paper/` (23 run records, 23 logs and
20 saved-policy directories, 8.6 MB) and `train.out` (72 kB).

**Why.** `results/ws_paper/` is the only grid in the repository with five seeds (audit A1) and was
outside version control; `results/**/*_model/` is in `.gitignore`, so none of its saved policies
could be replayed by anyone cloning the repository (audit E4). Nothing is to be lost before the
correctness edits start.

**Evidence.** `git status --short` on `main` listed `?? results/ws_paper/` and `?? train.out`;
`find results/ws_paper -type f | wc -l` = 186 files (46 outside `*_model`, 140 inside). The model
directories were added with `git add -f` against `.gitignore` deliberately, so the grid is
replayable.

### 1.2 `*.csv text eol=lf` added to `.gitattributes`

**What.** Added `*.csv text eol=lf` and rewrote the six working-tree copies of
`citylearn_schemas/tx_travis_8b_ev/charger_*.csv` from CRLF to LF in place.

**Why.** All six tracked charger CSVs showed as modified in `git status` with 52,566 insertions and
52,566 deletions, which is every line of every file. This is line-ending drift, not a data change
(audit E2), and it was being carried into the provenance discussion in `docs/REPORT_2026-10.md` §9.

**Evidence.** For `charger_1_1.csv`: working-tree md5 `e6141343b8d7c8885b8fb369ba571f6b`, but
md5 after `tr -d '\r'` = `1c6a1eb6467d7ca44568fb9a644db9e7`, which equals both `git show
HEAD:…| md5sum` and `git show HEAD:… | tr -d '\r' | md5sum`. So the committed content and the
working-tree content are the same bytes modulo `\r`. After normalisation `git status --short` no
longer lists any `charger_*.csv`.

### 1.3 Duplicate audit file at the repository root removed

**What.** `AUDIT_2026-10-06.md` existed at the repository root and at `docs/AUDIT_2026-10-06.md`.
Both were committed in 1.1, then the root copy was removed with `git rm` so it stays recoverable in
history.

**Why.** Two byte-identical copies of a 40 kB document invite them to diverge.

**Evidence.** `md5sum` of both files = `cd83c27c3e5115b0972525107d7ae961`.

---

## Step 2 — Bind observation indices to names (audit B9, B8)

### 2.1 New module `stems/observations.py` owns the canonical observation vector

**What.** The observation names, their order, the heat-pump block, the electric-vehicle
slot fields and the `T_OUT_PRED_LEAD_H` forecast leads now live in one module. It exposes
`obs_index(name) -> int`, `obs_indices(*names)`, `selected_obs_names(heat_pump, ev_slots)`,
`schema_observation_names()` and `index_in(names, name)`. `stems/environment.py` imports
and re-exports them, so `from stems.environment import OBS_NAMES` still works.

**Why.** Nine observation indices were written as integer literals in eight modules
(`cbf.py`, `reward.py`, `metrics.py`, `baselines.py`, `fleet.py`, `thermal.py`,
`experiments/runner.py`, `experiments/ev_coupling.py`, plus two diagnostics). They were all
correct, but nothing bound them to `OBS_NAMES`: one insertion into that list would have
silently made the reward, the shield and the KPIs read different columns, with no test
failure pointing at the cause (audit B9).

**Evidence.** `tests/test_observation_indices.py` (52 cases) passes.
`test_inserting_an_observation_shifts_every_consumer_together` inserts a name at position 2
of the canonical list, reloads every consumer, and asserts each index moved by exactly the
amount the registry says — the literal constants would not have moved at all.

### 2.2 `STEMSEnvironment.index_of(name)`

**What.** One accessor returning the position of `name` in the vector *this* environment
emits, covering the electric-vehicle slot fields whose position depends on `heat_pump` and
on how many chargers the schema exposes. Raises `KeyError` with the available names rather
than returning a sentinel.

**Why.** `obs_index` can only resolve the blocks whose position is fixed at import time
(indices 0–29). Everything beyond that has to be asked of the environment.

**Evidence.** `test_obs_index_rejects_an_unknown_name` pins that `obs_index("ev0_soc")`
raises and directs the caller to the environment.

### 2.3 The refactor does not change behaviour

**What.** Every literal index was replaced by a lookup; no arithmetic changed.

**Evidence, part 1 (exact).** All 23 module constants resolve to the same integers the audit
verified by hand: `cbf` (2, 19, 20); `reward` (15, 16, 17, 20, 21, 26, 27);
`metrics` (21, 14, 15, 27, 26, 20, 19, 18, 25, 17); `baselines` (1, 21, 19, 15, 16, 17, 27,
28, 20); `thermal` (0, 1, 2, 3, 18, 25); `fleet` (16, 17, 18, 25, 20);
`experiments/runner` (18, 19, 20). `test_indices_still_equal_the_values_the_audit_verified`
pins them.

**Evidence, part 2 (end-to-end).** A 336-step rollout with a fixed action sequence
(`numpy` default_rng(0), 8 buildings, heat pump on) run against the pre-refactor tree
(`git archive 2bc29f6`) and against the working tree gives bit-identical values for all 23
KPIs and for the summed reward: cost 5752.5498046875, `avg_daily_peak` 247.47560010637557,
`peak_import_kw` 261.6355285644531, `ramping_rate` 17.339920043945312,
`discomfort_rate` 0.7862700228832952, `safety_violation_rate` 0.24404761904761904,
reward sum −33664.449781223826. **Caveat: this rollout is on the synthetic mock**, because
the CityLearn dataset is not reachable from this sandbox (see "Deferred", below). The
index-equality evidence above does not depend on the environment and is the stronger of the
two.

### 2.4 The schema builder and the environment now share one observation list

**What.** `setup_citylearn_8b.py::STEMS_OBSERVATIONS` is now
`stems.observations.schema_observation_names()` instead of a second hand-maintained list.

**Why.** The two copies disagreed on three names (audit B8): the builder omitted
`dhw_storage_soc` (canonical index 18), `indoor_dry_bulb_temperature_heating_set_point` (28)
and `heating_electricity_consumption` (29), all three of which `stems/environment.py`
requires. The generated schema worked only because those are already active in the upstream
Travis schema; had upstream changed, the environment would either have raised or, with
`--allow-missing-obs`, silently zero-filled the DHW state of charge that the tank model, the
shield and the KPIs all read.

**Evidence.** `test_schema_builder_and_environment_share_one_observation_list` and
`test_the_schema_builder_holds_no_second_observation_list` (an AST check that
`STEMS_OBSERVATIONS` is not a literal list again). No schema was regenerated and no file
under `citylearn_schemas/` was touched.

### 2.5 Guard against reintroduction

**What.** `test_no_module_reintroduces_a_literal_observation_index` scans every `.py` under
`stems/` and `experiments/` and fails on any line matching `_IDX_NAME = <integer>`.

**Evidence.** Passes with zero offenders after the rewrite; it failed on 23 lines before it.

---

## Environment and baseline (context for everything below)

The CityLearn dataset cache now lives inside the repository at `.citylearn_cache/`
(142 MB, gitignored — added to `.gitignore` in the step 2 commit). Runs need
`XDG_CACHE_HOME=/home/yassine/Documents/Yassin-code/stems/.citylearn_cache`.

**True test baseline before the correctness fixes: 284 passed, 0 failed, 0 skipped,
109 s.** That is the floor; no change below may reduce it. Two earlier counts quoted
during this work (171 passed / 35 skipped / 6 errors, and 186 passed / 19 failed) were
artefacts of a half-installed environment and say nothing about the repository.

The README states the suite takes about ten minutes; it takes 109 s on this machine.
Not corrected here — `docs/REPORT_2026-10.md` and `README.md` are out of scope for this
track — but it should be corrected when someone next edits the README.

Regenerating the EV schema from seed 17 rewrote the six tracked charger CSVs, and all six
are byte-identical to the committed copies after stripping carriage returns: md5
`1c6a1eb6467d7ca44568fb9a644db9e7`, `c85806ffbab7c557f16221367baaaffe`,
`fdb89985e2b219a1a30fb1214a4fe4aa`, `b80ffcd8741401d00e776e1d54a926ee`,
`a93576a771f383ce1dba3e8e1976d02b`, `1128d46876d1c7081e40deecc0de0b2a`. The synthetic EV
schedules are therefore genuinely deterministic, which is worth stating in the paper's
reproducibility section, and the apparent git modification was only the CRLF drift that
entry 1.2 fixed.

---

## Step 3 — Fix the shield and Lagrangian math (audit B1, B3)

### 3.1 `CBFShield._apply_hvac_power_guard` now bisects on the achieved import

**What.** The district stage of the heat-pump guard was

```python
total = sum(max(net + |a_hvac| * p_nom, 0))
if total > g_cap: safe[:, hvac] *= g_cap / total
```

It is now the same bisection the battery grid guard uses: find the largest common shed
factor `s` in [0, 1] such that `I(s) = sum_b max(e_b + s*|a_b|*p_b, 0)` meets the cap.

**Why.** Scaling the action by `cap/total` does not bring the import to the cap, because
`total` contains the uncontrollable `net` term and the rescale cannot touch it
(audit B1). This is the same arithmetic error that `docs/REPORT_2026-10.md` §6.7 records
as found and corrected in the battery grid guard; the heat-pump guard still had the
original form.

**Evidence.** `tests/test_constraint_math.py::test_hvac_guard_brings_the_district_import_to_the_cap`.
Four buildings at `e_b = 20 kW` (80 kW uncontrollable) each requesting full heat-pump
power `p_b = 10 kW` (40 kW controllable), against `P_grid_max = 100 kW`:

| | shed factor | achieved import | over cap |
|---|---|---|---|
| before (`cap/total`) | 0.8333 | **113.33 kW** | **+13.33 kW (+13.3%)** |
| after (bisection) | 0.5000 | 100.00 kW | 0 |

Three further cases are pinned: the fixed load alone above the cap (the guard sheds the
whole controllable draw rather than leaving a positive heat-pump request on top of an
already-infeasible import); a feasible request passing through untouched; and one
building exporting 30 kW while three import 30 kW, where the signed sum is 60 kW but the
import is 90 kW — the guard enforces the positive-part definition, the same one step 4
gives the KPIs.

**Numerical note.** The bisection is evaluated on the float32 action it writes back, so
the shield meets the cap *exactly* in its own arithmetic
(`_achieved_import_f32(safe, states) <= cap` is asserted). Recomputing the same import in
float64 with a float64 nominal power disagrees by about 4e-6 kW — 4 microwatts on a
100 kW cap, 4e-8 relative — so the float64 assertions use a 1e-3 kW tolerance. That is
representation error in the comparison, not slack in the method.

### 3.2 Decision: the heat-pump guard stays unwired

**Decision.** The guard is fixed but **not** wired up: `experiments/controllers.py`
still does not pass `cop_model`, so `CBFShield._apply_hvac_power_guard` returns at its
first line in every experiment, exactly as before.

**Why not delete it.** The per-building stage — derating the cap by the
coefficient-of-performance shortfall at the current outdoor temperature — is a real
constraint that the heat-pump papers will need, and the module is the natural home for
it. Leaving arithmetically wrong code in place one keyword argument away from being used
was the hazard; that hazard is now gone.

**Why not wire it.** Two reasons, the second decisive.
1. Wiring it would change the dynamics of every stored run, so no existing result could
   be compared against a new one — and the whole point of this track is to leave the
   measured numbers intact unless a bug forces a change.
2. **It is dimensionally wrong under the control mode the house scenarios use.** The
   guard multiplies `safe[:, hvac_idx]` by a nominal electrical power to get kW. That is
   only valid when the environment runs `hvac_control="power"`. The house scenarios run
   `hvac_control="setpoint"`, where that column is a ±1.5 °C set-point offset and the
   integral thermostat — not the policy — issues the power command. Multiplying a
   temperature offset by kW is meaningless. Before this guard is switched on, either the
   scenario must use `hvac_control="power"`, or the guard must be re-expressed on the
   thermostat's output rather than on the policy action.

Recorded in the method's docstring as well, so the next reader meets it before the code.

### 3.3 Cost advantages are standardised like reward advantages

**What.** The combination of reward and cost advantages moved out of `STEMSAgent.update`
into a testable static method `STEMSAgent.effective_advantage(adv, cadv, lam_k)`, and the
cost advantage is now standardised over the time axis:

```
cadv = (cadv - cadv.mean(0)) / (cadv.std(0) + 1e-8)      # was: cadv - cadv.mean(0)
A_eff = (A - sum_k lambda_k * A^c_k) / (1 + sum_k lambda_k)
```

**Why.** `adv` was standardised to unit variance and `cadv` was mean-centred only, so the
two terms were on incompatible scales (audit B3). With 0/1 indicator costs and γ = 0.99
the cost advantage has a spread of order 10 while `adv` has unit variance, so λ's
effective weight depended on the raw spread of the costs rather than on the constraint. A
λ tuned on one scenario did not transfer to another, and the per-episode λ traces logged
by different runs were not comparable.

**Evidence.** `test_effective_advantage_is_invariant_to_the_cost_scale` — rescaling the
costs by 0.01x, 10x and 1000x leaves `A_eff` unchanged to 1e-4.
`test_effective_advantage_was_not_invariant_before_the_fix` reproduces the old
mean-centring and shows it fails that same invariance, so the first test is testing
something. `test_cost_advantages_reach_unit_variance` recovers the standardised cost
advantage from `A_eff` at λ = (1, 0, 0) and checks its mean (<1e-5) and standard
deviation (1 ± 1e-3). `test_effective_advantage_reduces_to_the_reward_advantage_at_zero_lambda`
pins the λ = 0 limit.

### 3.4 Partial dispute: the λ *trajectory* was already reward-scale invariant

The step asked for "a test that the same violation rate produces the same λ trajectory
under two different reward scales". That test now exists
(`test_the_same_violation_rate_gives_the_same_lambda_trajectory_at_two_reward_scales`)
and it passes — **but it would also have passed before the fix**, and it is worth being
precise about why, because the audit's wording invites the wrong conclusion.

`STEMSAgent._update_lambdas` is a PID controller on the measured episode-mean cost rate
against `cost_limit`. It never reads the reward, the reward scale or the advantages. So λ
itself was never reward-scale dependent, before or after. The scale problem B3 identifies
is real but it is in the **objective** `A_eff`, not in the dual update: a given λ bought a
different amount of constraint pressure depending on the spread of the cost advantages.
That is what 3.3 fixes and what the invariance tests measure. The λ-trajectory test is
kept as a regression guard so a future change to `_update_lambdas` cannot introduce a
reward-scale dependence unnoticed, and its docstring says exactly this.

---

## Reproducibility hazard noted, not fixed (out of scope for this track)

**Fifteen tests silently convert a network failure into a skip.** `test_env_widening.py`
(10) and `test_ev_real.py` (5) wrap environment construction in `except ... pytest.skip`.
`STEMSEnvironment._resolve_schema` falls through to
`citylearn.data.DataSet().get_dataset_names()`, which calls `api.github.com`. On a shared
or rate-limited egress address that returns

```
403 API rate limit exceeded for <egress ip>
```

and `_resolve_schema` catches it and reports "schema not found" — **even when the dataset
is present on disk**, which it is: all four datasets are in `.citylearn_cache/`. Observed
directly in this session: the same tree gave 0 skips when the quota was intact and 15
skips once it was exhausted, with no code change in between.

The consequence is that an offline or rate-limited CI run reports all-green while 5% of
the suite never executed, and that 5% is exactly the EV-schema and
heterogeneous-observation coverage — the padding and charger-discovery logic the EV
results depend on.

Two defensible fixes, **deliberately not implemented here** because they change test
semantics and this track must not touch the test contract: resolve a cached dataset from
disk before ever consulting the GitHub API, and make the skip loud (or a failure) when the
data is in fact present locally.

---

## Step 4 — Unify the district-import definition (audit B5)

### 4.1 `avg_daily_peak` and `ramping_rate` now use the positive-part sum

**What.** `MetricsCalculator.compute_all` computed a single series

```python
total_net = net.sum(axis=1)          # signed: one building's export offsets another's import
```

and used it for `avg_daily_peak` and `ramping_rate`, while `peak_import_kw`,
`cap_exceedance_kwh` and `grid_violation_rate` used

```python
grid_series = np.maximum(net, 0.0).sum(axis=1)   # positive part
```

There is now one series, the positive-part one, used by all five.

**Why.** Writing `e_bt` for building b's net electricity consumption at step t, the
district import is `G_t = sum_b max(e_bt, 0)`. The signed alternative `sum_b e_bt` lets
one house's PV export cancel another's import, which no physical path in the model
permits. More to the point, the CBF (`_apply_power_guard`, `_apply_hvac_power_guard`),
the fleet shield and the reward's grid term all enforce the positive-part definition, so
the headline "average daily peak" was not the quantity any constraint in the repository
controls (audit B5).

**Evidence (tests).** Four new cases in `tests/test_kpis.py`, alongside the
`test_exports_do_not_offset_another_buildings_import_for_peak` that already pinned the
convention for `peak_import_kw`:
`..._for_avg_daily_peak`, `..._for_ramping_rate`,
`test_every_grid_side_kpi_uses_the_same_import_series` (one worked example where peak 10,
avg daily peak 10, ramping 4.5, cap exceedance 7 kWh and violation rate 2/3 all follow
from the same `G = (10, 4, 7)`), and `test_the_two_definitions_still_agree_when_nobody_exports`.
22 of 22 pass.

**Evidence (magnitude).** Idle rollouts on the real Travis 8-building schema, so the net
series is the simulator's own:

| window | hours with ≥1 exporter | `avg_daily_peak` signed → positive | `ramping_rate` signed → positive | `peak_import_kw` |
|---|---|---|---|---|
| summer, 14 d | 50.7% | 12.183 → 12.382 kW (+1.63%) | 4.991 → 1.853 kW (**−62.9%**) | 18.989 kW, unchanged |
| winter, 14 d | 37.3% | 19.281 → 19.281 kW (0.00%) | 4.990 → 2.750 kW (**−44.9%**) | 39.634 kW, unchanged |
| full year | 43.7% | 14.919 → 14.947 kW (+0.18%) | 4.810 → 2.257 kW (**−53.1%**) | 39.634 kW, unchanged |

`avg_daily_peak` barely moves because the hour that sets a day's peak is usually an hour
in which nobody is exporting. `ramping_rate` roughly halves, because the signed sum
swings through the PV midday while the import series does not: most of what the old
ramping metric measured was solar generation, not load change.

### 4.2 Report numbers invalidated: none

I checked rather than assumed.

- **`ramping_rate` is never quoted in `docs/REPORT_2026-10.md`**, in `README.md` or in
  `docs/SUMMARY.md`. `grep -in ramping` over all three returns nothing.
- **`avg_daily_peak` is never quoted either.** Every "Peak kW" / "Peak [kW]" column in the
  report comes from `peak_import_kw`, which this change does not touch
  (`ev_rl_report.py:18`, `hp_report.py:20`, `ev_report.py:142`). Confirmed numerically
  against the stored records: recomputing the §5 table from `results/ablation_v1/`
  reproduces the report's column exactly from `peak_import_kw` —

  | arm | report §5 "Peak kW" | mean `peak_import_kw` | mean `avg_daily_peak` |
  |---|---|---|---|
  | idle | 35.1 | **35.1** | 23.5 |
  | idle + calibrated | 35.2 | **35.2** | 23.6 |
  | rule | 45.1 | **45.1** | 26.9 |
  | rule + calibrated | 45.1 | **45.1** | 26.9 |
  | RL (Lagrangian only) | 41.4 | **41.4** | 21.1 |

  `hp_report.py:21` does emit an `avg_daily_peak` column labelled "Daily peak [kW]", and
  `paper_table.py` normalises both `avg_daily_peak` and `ramping_rate` — but neither table
  appears in the report as it stands.

**What *is* invalidated: 213 stored run records.** Every record under `results/` that
carries `eval.avg_daily_peak` and `eval.ramping_rate` holds them under the old signed
definition — `ablation_v1` 80, `mixed_v1` 32, `ws_paper` 23, `ev_rl_v2` 18, `heatpump_v1`
16, `ablation_v2` 16, `ev_rl_v1` 12, `pilot_v2` 10, `heatpump_power_v1` 6. Those two
fields must not be compared with, or pooled with, values produced by current code. The
other KPIs in those records are unaffected. `experiments/aggregate.py` already refuses to
aggregate across code fingerprints, and step 5 widens that fingerprint, so a mixed
comparison will be refused rather than silently averaged — but anyone reading a stored
record by hand should know. Nothing under `results/` was deleted or regenerated.
