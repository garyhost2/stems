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
(`_achieved_import_f32(safe, states) <= cap` is asserted, and returns 100.0 on the nose).
Recomputing the same import in float64 with a float64 nominal power disagrees in the last
few bits: the measured residual is **3.81e-08 relative** (3.81e-06 kW, i.e. 3.8 mW on a
100 kW cap), so the float64 assertions use a 1e-3 kW tolerance. That is representation
error in the comparison, not slack in the method.

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
2. **It is dimensionally wrong under the control mode the experiments actually use.**
   The guard multiplies `safe[:, hvac_idx]` by a nominal electrical power to get kW,
   which is only valid when the environment runs `hvac_control="power"`. Two different
   defaults are involved and they disagree: `STEMSEnvironment`'s own parameter defaults
   to `"power"`, but `experiments/scenario.py::Scenario` defaults to `"setpoint"`, so
   every run driven through a `Scenario` — which is every run in `results/` — uses
   `"setpoint"`. There that column is a ±1.5 °C set-point offset and the integral
   thermostat, not the policy, issues the power command; multiplying a temperature
   offset by kW is meaningless. Before this guard is switched on, either the scenario
   must set `hvac_control="power"`, or the guard must be re-expressed on the
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

## Step 4 — Name the two district series and keep them apart (audit B5, **corrected**)

### 4.1 Correction: audit B5's prescription was wrong, and was not implemented

The audit found that `MetricsCalculator` uses the **signed** district sum for
`avg_daily_peak` and `ramping_rate` and the **positive-part** sum for `peak_import_kw`,
`cap_exceedance_kwh` and `grid_violation_rate`, called the mixture a bug, and asked for
the positive-part definition everywhere. The observation is right; the prescription is
wrong, and it was reverted before landing.

**Evidence, from the installed `citylearn` 2.6.0b1 source.**

- `CityLearnEnv.net_electricity_consumption` is documented as the "Summed
  `Building.net_electricity_consumption` time series, in [kWh]" and applies no clipping.
  It is signed.
- `CostFunction.peak(net_electricity_consumption, window=24)` consumes that signed
  series directly: it groups by 24-step window, takes each window's `max()`, then a
  `rolling(window=n, min_periods=1).mean()`.

So the signed sum is **CityLearn's own convention**, and `avg_daily_peak` already
matches it. Moving it to the positive-part sum would have made the repository's headline
peak disagree with the simulator, with CityLearn's published cost function, and with the
STEMS paper's Table I normalisation that `experiments/paper_table.py` compares against —
the opposite of what the audit intended.

**The real defect** is that one object carried two district definitions with nothing in
the code or the output saying which was which. That is what is fixed.

### 4.2 The split is now explicit

**What.** Two named helpers in `stems/metrics.py`, each with the docstring that says what
it is for, and two declared KPI families:

```python
district_signed_kw(net) = net.sum(axis=1)                 # S_t = sum_b e_bt
district_import_kw(net) = np.maximum(net, 0).sum(axis=1)  # G_t = sum_b max(e_bt, 0)

CITYLEARN_COMPARABLE_KPIS = ("avg_daily_peak", "ramping_rate")
CONSTRAINT_FACING_KPIS    = ("peak_import_kw", "load_factor",
                             "cap_exceedance_kwh", "grid_violation_rate")
```

`compute_all` binds both series once at the top and every KPI takes one of them.

**Consequence that must reach the paper.** A neighbourhood peak reported under
CityLearn's convention is **not** the quantity the cap constrains. `S_t` can sit well
below `G_t` whenever any house exports — on the full-year idle rollout at least one
building exports in 43.7% of hours — so a paper sentence of the form "the controller
holds the neighbourhood peak to X kW" has to say which of the two X is. Only `G_t` can
be compared against a cap; only `S_t` can be compared against a CityLearn baseline.

**Evidence.** Six cases in `tests/test_kpis.py` (24 of 24 pass):
`test_the_two_district_series_differ_whenever_a_building_exports` (the guard against a
future silent unification), `test_citylearn_comparable_kpis_use_the_signed_series` and
`test_constraint_facing_kpis_use_the_import_series` (each asserts its family's values
*and* that they differ from the other family's), `test_the_two_kpi_families_are_declared_and_disjoint`,
`test_both_families_agree_when_nobody_exports`, and — the strongest one —
`test_avg_daily_peak_equals_citylearns_cost_function`, which asserts
`avg_daily_peak == CostFunction.peak(S)[-1]` on three days of heavily exporting
synthetic data.

### 4.3 Plain mean versus running mean: they agree on the reported scalar

CityLearn's `CostFunction.peak` returns a *running* mean of the daily maxima; this
repository returns a plain mean. They coincide at the final element, and the final
element is the scalar CityLearn reports, so the two agree exactly. Verified: on 72 steps
of synthetic data, CityLearn's series is (32.329269, 38.267218, 39.189044) and the plain
mean of the daily maxima is 39.189044. **Nothing changed here**; the equality is now
pinned by `test_avg_daily_peak_equals_citylearns_cost_function`.

One deliberate, pre-existing difference is kept and documented in the code: this
repository floors a day's peak at zero, so an all-exporting day contributes 0 rather than
a negative peak. CityLearn does not floor. The two therefore differ only on a day in
which the district exports at every single hour, which occurs in none of the windows used
here.

`ramping_rate` is a different matter and the code now says so. It is the mean absolute
step-to-step change of the signed series. CityLearn's `CostFunction.ramping` clips to
*positive* ramps by default (`down_ramp=False`) and returns a running **sum**, not a
mean: on the same 72-step series CityLearn gives 689.2994 and this repository gives
19.2597. They share the underlying series and nothing else, so `ramping_rate` must not be
presented as a CityLearn-comparable number even though it is in the signed family.

### 4.4 Behaviour unchanged, and no report number invalidated

**Behaviour.** `avg_daily_peak` and `ramping_rate` keep the values they had. Real idle
rollouts on the Travis 8-building schema, before the step-4 edit and after it:

| window | `avg_daily_peak` | `ramping_rate` | `peak_import_kw` |
|---|---|---|---|
| summer, 14 d | 12.183 → 12.183 kW | 4.991 → 4.991 kW | 18.989 → 18.989 kW |
| winter, 14 d | 19.281 → 19.281 kW | 4.990 → 4.990 kW | 39.634 → 39.634 kW |

**Report numbers invalidated: none.** This is now true by construction, since no KPI
changed value. For the record, had the audit's prescription been implemented it would
also have invalidated nothing in `docs/REPORT_2026-10.md`: `ramping_rate` is never quoted
there, in `README.md` or in `docs/SUMMARY.md`, and every "Peak kW" / "Peak [kW]" column
comes from `peak_import_kw`, which the prescription did not touch. Confirmed numerically
— recomputing the §5 table from `results/ablation_v1/` reproduces the report's column
exactly from `peak_import_kw` (idle 35.1, idle+calibrated 35.2, rule 45.1,
rule+calibrated 45.1, RL 41.4), while the corresponding `avg_daily_peak` means are
23.5, 23.6, 26.9, 26.9 and 21.1. The 213 stored run records that carry both KPIs remain
valid and comparable with current code.

---

## Step 5 — Repair provenance and seeding (audit E3, E1)

### 5.1 The seed now reaches CityLearn

**What.** `STEMSEnvironment._build_real_env` passes `random_seed=self._seed` to
`CityLearnEnv`. An explicit `random_seed` in `env_kwargs` still wins.

**Why.** `__init__` stored `self._seed` and used it only for the mock; the real
`CityLearnEnv` was constructed without it (audit E3), so "seed" seeded torch, numpy and
python and not the simulator. Harmless while the simulation is deterministic given the
schema, but a stochastic element — a randomised EV schedule, a stochastic occupancy
model, CityLearn's own `random_episode_split` — would have been unreproducible, and the
run record would not have said so.

**Evidence.** `tests/test_provenance.py::test_the_seed_reaches_citylearn` asserts
`env._env.random_seed == 4321` for `STEMSEnvironment(seed=4321)`;
`test_an_explicit_random_seed_in_env_kwargs_wins` pins the override.
`CityLearnEnv.__init__` does accept `random_seed: int = None`, checked by
`inspect.signature` on the installed 2.6.0b1.

### 5.2 A second, data-side fingerprint

**What.** New `experiments/runner.py::data_fingerprint(schema_path)`, recorded as
`record["meta"]["data"]` alongside the existing `record["meta"]["code"]`. It digests:

- the CityLearn version string;
- `requirements.lock.txt`;
- the resolved schema JSON, **with `root_directory` removed and keys sorted**, so the
  digest tracks the experiment rather than the absolute install path;
- every data file the *included* buildings actually read — the ResStock energy-simulation
  CSVs, `weather.csv`, the price and carbon series, and the learned `.pth` dynamics
  checkpoints;
- the electric-vehicle `charger_*.csv` files beside the schema, hashed with CRLF
  normalised to LF so the line-ending drift of audit E2 cannot move the digest.

For `citylearn_schemas/tx_travis_8b/schema.json` this covers the resolved schema plus
every data file its included buildings reference — 21 files and 7.5 MB as the schema
stands, about 15 ms — and reports `data_missing: []`. The count is derived, not fixed:
it moves when a building or a charger is added, so `data_files` is recorded in the run
record rather than asserted against a literal.

*Correction:* the commit message of `37f211b` says 20 files. The measured value is 21
(`data_fingerprint(...)["data_files"]`). The commit message is wrong and is left as it
stands rather than rewriting published history; this entry is the correct record.

**Why a second field rather than a wider one.** More than 400 stored records already
carry `meta.code.fingerprint`. Changing what that field hashes would make every old
record incomparable with every new one *and* with each other, since nothing in a record
says which definition produced it. Adding `meta.data.data_fingerprint` leaves the old
field's meaning intact: an old record simply has no data digest, which is honest and
detectable. `code_fingerprint()`'s return keys are pinned by a test for the same reason.

**Why it was needed.** `code_fingerprint` hashes `stems/*.py` and `experiments/*.py` only.
It did not cover the schema JSON (gitignored, regenerated per machine), the charger CSVs,
the building time series, the dependency lock or anything but the CityLearn version
string, so two records with the same `fingerprint` could be different experiments
(audit E1).

**Evidence.** `tests/test_provenance.py`, 12 cases, all passing:
`test_two_different_schemas_get_different_data_fingerprints` builds a 7-building variant
of the 8-building schema and shows the code fingerprint is identical while the data
fingerprint differs — the exact failure E1 names;
`test_the_fingerprint_does_not_depend_on_the_install_path`;
`test_charger_csvs_enter_the_fingerprint_but_line_endings_do_not` (CRLF conversion leaves
the digest alone, a one-cell edit moves it);
`test_a_missing_data_file_is_reported_not_swallowed`;
`test_data_fingerprint_survives_a_missing_schema`;
`test_the_original_code_fingerprint_field_is_untouched`.

**Known gap, stated not hidden.** `requirements.lock.txt` is hashed, but the
actually-installed package versions are not resolved and compared against it, so an
environment that drifts from the lock file is not detected. Recorded in the function's
docstring.

---

## Step 5, revisited — the run seed must **not** reach CityLearn (audit E3, **corrected**)

Entry 5.1 above is superseded. The forwarding it describes was implemented, found to be
harmful by the test suite, and reverted in the same session.

**What went wrong.** `CityLearnEnv(random_seed=s)` overrides the per-building
`random_seed` the schema carries and **re-randomises device autosizing**, so the houses
become a function of the seed. Battery capacity, kWh, over the eight Travis buildings:

| | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 |
|---|---|---|---|---|---|---|---|---|
| unseeded (schema's own per-building seeds) | 10.8 | 6.6 | 5.0 | 9.7 | 13.5 | 5.4 | 16.2 | 16.0 |
| `random_seed=7` | 13.5 | 10.0 | 5.0 | 13.5 | 5.4 | 13.5 | 13.5 | 17.5 |
| `random_seed=4321` | 12.0 | 13.5 | 3.3 | 8.0 | 3.5 | 10.8 | 23.1 | 10.5 |

Four tests caught it immediately — `test_battery_info_is_per_building_and_plausible`
(`soc_rate` containing values at or below 0.1), `test_barrier_keeps_the_band_and_uses_all_of_it`,
`test_cbf_feasibility_and_safety_on_real_data`, and
`test_battery_info_is_per_building_and_plausible`. They were right and the change was
wrong.

**Why it is wrong in principle, not just inconvenient.** The seed is the replication unit
*within* a scenario, and `experiments/aggregate.py` averages seeds inside a scenario
before treating the scenario as the unit of analysis. If the seed changes the building
stock, then five "seeds" of one scenario are five different neighbourhoods, the
within-scenario average is across building stocks rather than across policy
initialisations, and every confidence interval in the study silently widens for the wrong
reason. The schema defines the houses; the seed must not.

**Audit E3 is therefore partly wrong.** Its observation — the seed reaches torch, numpy
and python but not the simulator — is correct. Its premise, that forwarding it is
"harmless today because the simulation is deterministic given the schema", is not: the
*construction* of the environment is seed-sensitive even though its dynamics are not.

**What was implemented instead.** An explicit, opt-in `citylearn_seed` argument on
`STEMSEnvironment`, default `None`, plus a `citylearn_seed` property reporting the value
actually handed to the simulator (or `None`). That meets E3's real requirement — a run
record can state what seeded the simulator instead of leaving it to be inferred —
without making the houses a function of the replication. The reasoning and the measured
capacities are in the code comment at the call site, so the next person to reach for
`random_seed=` meets the evidence first.

**Evidence.** `tests/test_provenance.py`:
`test_seeding_citylearn_changes_the_buildings_themselves` pins all three capacity vectors
above; `test_the_run_seed_does_not_reach_citylearn_by_default` asserts that seeds 0, 7 and
4321 give identical capacities and `citylearn_seed is None`;
`test_citylearn_can_still_be_seeded_deliberately_and_the_choice_is_reported` pins the
opt-in path and the `env_kwargs` override. Unseeded autosizing is deterministic —
the same capacity vector comes back in three fresh processes and twice in one process.

Entry 5.2 (the data fingerprint) is unaffected and stands.

---

## Step 6 — Give the building graph real inputs (audit B6, B7)

### 6.1 There are no coordinates and no floor areas. The graph is now feature-based.

**What was checked.** CityLearn 2.6.0b1 exposes no building geometry anywhere:

- the live `Building` objects have no `latitude`, `longitude`, `floor_area`, `area`,
  `coordinates` or `location` attribute (asserted in a test, so a future CityLearn that
  adds one will fail it rather than go unnoticed);
- the schema's per-building keys are devices, storage, dynamics, `include`, `pricing`,
  `carbon_intensity`, `type`, `weather` — no geometry;
- the dataset directory holds per-building time series, `.pth` dynamics models and
  `dynamics_error_summary.csv`;
- the ResStock time-series header is month, hour, day type, daylight savings, indoor
  temperature and humidity, non-shiftable load, DHW/cooling/heating demand, solar
  generation, occupant count, the two set points, HVAC mode;
- this repository's own `citylearn_schemas/_sizing/tx_travis_county_neighborhood__fullyear_raw.json`
  gives device sizes for 100 buildings and no geometry.

So `get_building_info()` always fell through to coordinates generated from the building's
index, `(30.26 + 0.01*i, -97.74 + 0.01*i)`, and a constant 150 m² floor area (audit B6).

**Stated plainly, because the paper has to say it: the published architecture's spatial
term is not recoverable from this dataset. The graph is feature-based, and the
"spatio-temporal" claim must be worded accordingly.** No coordinates were invented.

**What the graph is built from instead.** Eight per-building device characteristics read
off the live CityLearn objects, all nameplate ratings or efficiencies fixed before the
first time step, so none of them can leak information from an evaluation window:
battery capacity (kWh), battery nominal power (kW), PV nominal power (kW), heating and
cooling heat-pump nominal power (kW), DHW heater nominal power (kW), DHW tank capacity
(kWh), heating efficiency (dimensionless). They are z-scored per column so no
large-magnitude column dominates the distance. A test asserts none of them is constant
across buildings — the old feature vector was `[battery capacity, 150.0]`, and the second
entry contributed nothing.

**Bandwidth.** `GraphConfig.sigma_f = None` now selects the median heuristic: sigma_f is
set so the median off-diagonal squared feature distance maps to exp(−1) ≈ 0.368. A fixed
bandwidth has no meaning independent of how the features are scaled, which is how
sigma_d = 1.0 came to produce a 2.4e-3 spread. On the Travis eight the heuristic picks
sigma_f = 2.8604.

**Edge-weight spread, 56 off-diagonal edges, 8 buildings.**

| | min | max | **spread** | mean | std |
|---|---|---|---|---|---|
| before, positional half-weight alone | 0.497556 | 0.499950 | **2.394e-03** | — | — |
| before, full adjacency (positional + old features) | 0.802466 | 0.999870 | **0.197404** | 0.929133 | 0.064515 |
| **after, `mode="feature"`** | 0.040744 | 0.838355 | **0.797610** | 0.392925 | 0.204998 |
| after, `mode="mean_pool"` (the ablation) | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 |

**Read the table like for like.** The audit's 2.394e-03 is the spread of the *old
positional term alone*, and row 1 reproduces it exactly. It must not be compared against
row 3, which is a *full adjacency*. The comparison that matters is **full adjacency
against full adjacency: 0.197404 before, 0.797610 after, a factor of 4.0** — not the
factor of ~330 that comparing row 1 against row 3 would suggest.

The old full adjacency was therefore less degenerate than the positional half alone,
because the min-max-scaled battery capacity did vary. It was still nearly uniform: mean
0.929133 with standard deviation 0.064515, so a typical edge sat within 7% of the mean,
and the weights spanned 0.802466–0.999870. The feature graph spans 0.040744–0.838355
with mean 0.392925 and standard deviation 0.204998. The qualitative change is that the
adjacency now distinguishes buildings instead of being a mean pool with jitter; the
quantitative change is 4.0x on the like-for-like spread.

### 6.2 `mode="mean_pool"` ablation arm registered

`ARMS["rl+calibrated+meanpool"]` is identical to `rl+calibrated` in policy, barrier,
residual, penalty, EV request, forced penalty, EV floor and control mask, and differs
only in `graph_mode`. A test asserts that field-by-field, so the contrast isolates the
GCN. With the fabricated positional term gone, this is the honest test of whether the
graph earns its place: if `rl+calibrated` does not beat `rl+calibrated+meanpool`, the GCN
is a mean pool with extra parameters and the paper should say so.

`Arm` gained a `graph_mode` field (default `"feature"`) and `build_controller` sets
`config.graph.mode` from it.

### 6.3 History window is primed, not zero-filled (B7)

`HistoryBuffer.prime(obs_list)` fills the whole window with the episode's first
observation; `experiments/runner.py` (both the training and the evaluation loop) and
`experiments/diagnostics/ev_rl_requests.py` call it immediately after `env.reset()`
instead of `update`.

**Why.** The buffer was zero-filled at construction and pushed one observation per step,
so for the first `window_size - 1` steps of every episode the transformer attended over a
window that was mostly the *normalised value of zero* — not a missing-data token, but a
spurious observation that normalises to whatever `-mean/std` happens to be. On a 14-day
episode with a 24-step window that is 23 of 336 steps, 7% of the episode, and it is the
stretch where the running normaliser is least calibrated (audit B7). Repeating the first
observation is constant extrapolation: the window says "nothing has changed yet", which
is true at t = 0, rather than "every signal was zero", which is false for a temperature
in °C or a state of charge.

**Evidence.** `test_history_prime_fills_the_window_with_the_first_observation`,
`test_a_primed_window_contains_no_spurious_zeros` (counts the zeros: `B*(W-1)*D` under
the old call, 0 under the new one), `test_priming_then_stepping_still_slides_the_window`.

**This changes learned behaviour.** Every learning arm sees a different first-23-step
observation window from now on, so new runs are not bit-comparable with the stored
records. No stored record was altered.

### 6.4 Test inventory updated deliberately

`tests/test_experiments.py::test_the_ablation_arms` is an explicit inventory of `ARMS`
and fails whenever an arm is added. It was updated to include
`rl+calibrated+meanpool` and to assert the `graph_mode` of both it and `rl+calibrated`.
Recording it here because the standing rule is that a test is not edited to accommodate a
change without saying so: in this case the test's whole purpose is to force exactly this
acknowledgement, and it will be updated once more in step 8.

---

## Step 7 — Batch the GCN forward pass (audit F)

**What.** `STEncoder.batch_forward` ran `for n in range(N): h_nb[n] = self.spatial_gcn(x_nb[n], adj)`
— N separate three-layer GCN calls per update, each of which also rebuilt the normalised
adjacency from scratch inside every layer. `GCNConv.forward` now relies on
`adj_norm @ x` broadcasting over a leading batch axis, so the same code path serves `(B, C)`
and `(N, B, C)` and the loop is gone. A new `normalised_adjacency(adj)` helper computes
D^(-1/2)(A+I)D^(-1/2) once and caches it on the tensor; the adjacency is fixed for a run,
and it was being recomputed 3·N times per update.

**Numerical equivalence.** Not an approximation: the GCN is linear in the node axis and
the adjacency is shared, so the batched form is the same arithmetic.
`tests/test_encoder_batching.py` keeps the pre-fix loop verbatim as `_loop_reference` and
asserts agreement to `atol=1e-5, rtol=0`. Measured maximum absolute difference over all
shapes tested: **2.38e-07**. Fourteen cases cover N×B of (1,2), (7,3), (33,8) at
three seeds (nine parametrised cases),
the N=1 agreement between `forward` and `batch_forward`, gradient flow through the
batched path, and symmetry plus caching of the normalised adjacency.

**Speed-up, 8 buildings, 30 observations, 24-step window, 4 threads.**

| what is measured | N = 336 (14 d) | N = 672 (28 d) | N = 8760 (1 y) |
|---|---|---|---|
| GCN stage alone, loop → batched | 11.5 → 0.2 ms (**57x**) | — | 301.1 → 9.0 ms (**33x**) |
| whole `batch_forward` | 27.8 → 11.1 ms (2.5x) | 51.4 → 34.2 ms (1.5x) | 695.6 → 501.7 ms (1.4x) |
| one PPO-Lagrangian update | 1.26 → 0.97 s (1.30x) | 2.81 → 2.10 s (1.34x) | — |

**Dispute: the audit overstates this one.** It says "training cost is dominated by
`STEncoder.batch_forward`'s Python loop" and that batching it "should make 5 seeds × 8
scenarios cost roughly what 2 seeds cost now" — which would need a 2.5x end-to-end
saving. The GCN loop was 43% of `batch_forward` at N = 8760 (301.1 ms of 695.6 ms); the
temporal transformer is the rest, and `batch_forward` is itself only part of an update.
At the level that matters — one PPO update — the measured saving is **1.30–1.34x**, not
2.5x. The change is still worth having, it is free of numerical risk, and it removes an
O(N) Python loop from the inner training path. But the budget must be planned on 1.32x,
not on parity:

    cost(5 seeds, new code) / cost(2 seeds, old code) = (5 / 1.32) / 2 = 1.9

So a five-seed grid costs about **1.9 times** what the current two-seed grid costs — an
improvement on the 2.5x it would have cost with no speed-up at all, but not the parity
the audit projected. If parity is actually needed, the temporal transformer is the next
place to look: it is the remaining 57% of `batch_forward` at N = 8760.

*Correction:* an earlier draft of this paragraph, and the commit message of `4f4b53a`,
said "roughly 3.8 times a two-seed grid". That is 5/1.32, which compares a five-seed
grid against a **one**-seed grid while the sentence claimed a two-seed baseline. The
figure is 1.9. The commit message is left as it stands rather than rewriting published
history; this entry is the correct record.

---

## Step 8 — Pre-register the new ARMS names

**What.** Twenty-four arm names are registered in `experiments/controllers.py::ARMS`
with the fields they need and no builder. `build_controller` raises
`NotImplementedError` for every one, with a message naming what has to be built and
where the relevant code or audit finding is.

| group | count | names |
|---|---|---|
| comparison controllers | 8 | `sac`, `dmappo`, `mpc`, `maddpg`, `marlisa`, `madcq`, `metaems`, `mappo-cc` |
| constraint-mechanism cross | 12 | `mech-{none,lagrangian,projection,both}+{uniform,linear,exact}` |
| hard comfort | 2 | `rl+calibrated+comfort`, `rbc+calibrated+comfort` |
| degradation | 2 | `rl+calibrated+degr-throughput`, `rl+calibrated+degr-dod` |

**Why a reserved name rather than nothing.** Both downstream tracks would otherwise have
to decide the naming and the field layout themselves, and a disagreement between them
would mean restructuring `ARMS` and re-running whatever had already been produced under
the other convention. Reserving the names fixes the interface now. It also turns
`--arms mappo-cc` from a `KeyError` that reads like a typo into a `NotImplementedError`
that says what is missing.

**Why they must not silently fall through.** An arm with no builder that reached some
other branch would produce a run record indistinguishable from a result. `build_controller`
refuses before touching the environment.

**New `Arm` fields.** `mechanism` (`"auto"` — the historical coupling — plus `"none"`,
`"lagrangian"`, `"projection"`, `"both"`), `comfort_barrier` (bool) and `degradation`
(`"none"`, `"throughput"`, `"throughput+dod"`). `graph_mode` arrived in step 6.
`plant_model` is a **property**, not a field: it reads `barrier` (`basic` → uniform-rate,
`linear` → linear, `calibrated` → exact inverse), so the battery-model axis has one
source of truth rather than two fields that can disagree. `implemented` is likewise
derived. `learns` now consults `LEARNING_POLICIES` and stays `False` for every
pre-registered policy, so the grid driver cannot allocate a training budget to an arm
that cannot use one.

**What the mechanism cross will require of whoever implements it.** `build_controller`
currently *couples* the two mechanisms: the Lagrangian runs whenever the policy learns,
and the projection runs whenever `barrier != "none"`. Neither can be switched off
independently today, which is exactly audit C2's complaint, and decoupling them is the
work those twelve arms name. The `NotImplementedError` says so.

**A note on the hard-comfort arms, for whoever picks them up.** Hard thermal comfort is
not unusual in this field — it is standard practice where the thermal model is
trustworthy; Panagi et al. (2026) embed a calibrated 3R2C grey-box model in a
network-constrained optimal power flow while explicitly enforcing thermal comfort, DER
limits and full power-flow physics. The reservation here is specific to this testbed:
`docs/REPORT_2026-10.md` §2 shows CityLearn's learned temperature model is not physical
(−0.25 cooling drops a house 4–13 °C in an hour; one house does not respond to cooling at
all), so a hard band enforced against *that* model is satisfied in simulation and
meaningless in reality. The barrier belongs behind a flag on CityLearn and the claim
belongs on an RC model fitted to real data. The `NotImplementedError` carries this.

**Evidence.** `tests/test_preregistered_arms.py`, 80 cases: every name resolves, every
one reports `implemented is False`, every one raises `NotImplementedError` whose message
contains the arm name and more than 120 characters of instruction; the cross is complete
(all twelve `(mechanism, plant_model)` pairs present and distinct); `plant_model` agrees
with `barrier` on both new and existing arms; `learns` is `False` for all eight
comparison controllers; and every one of the seventeen pre-existing arms still reports
`implemented is True` with `mechanism == "auto"`.

`tests/test_experiments.py::test_the_ablation_arms` was scoped to
`{n: a for n, a in ARMS.items() if a.implemented}`. Recording it under the
test-editing rule: the alternative was to paste twenty-four names that deliberately do
not run into an inventory of arms that do, which would make that test a list of things
that are not there. The pre-registered names are inventoried by the new file instead,
and the scoped test still fails the moment an *implemented* arm is added or changed.

---

# Constraints track — branch `methods/constraints`

Audit finding ids refer to `docs/AUDIT_2026-10-06.md` §4C. The starting point was that
only two constraints are genuinely hard in the running code — the battery
state-of-charge band and the district import cap — that comfort is a soft reward term,
that the per-building power cap is Lagrangian-only, that degradation and transformer
limits do not exist, and that the Lagrangian is vestigial in the arms actually run
because the exact barrier drives its cost to zero.

**Test floor on entry.** `python -m pytest tests/ -q` in this worktree, before the first
edit: **404 passed, 15 skipped, 1 warning, 0 failed** (419 collected, 123.76 s). The
brief states the floor as "419 passing"; 419 is the collected count, and the 15 skips
are all one cause — the CityLearn EV dataset
`citylearn_challenge_2022_phase_all_plus_evs` is not present on this machine, so every
test in `tests/test_env_widening.py` and `tests/test_ev_real.py` that needs it skips
with `EV dataset unavailable: RuntimeError("Schema ... not found")`. No test fails, and
nothing in this track touches that code path. The floor carried forward is therefore
**404 passed / 15 skipped / 0 failed**, and every step below is measured against it.

---

## Step 1 — Thermal comfort as a deadline-storage constraint, behind a flag

### 1.1 `stems/comfort.py`: the building envelope as a `(store, rate, required level, deadline)` device

**What.** A new module with three objects.

`RCThermalModel` is a first-order lumped-capacitance envelope,
`C dT_in/dt = UA (T_out − T_in) + Φ + Φ_g`, discretised at the control step. Every
parameter is SI and per building: `C` [J/K], `UA` [W/K], `Φ_g` [W], `dt` [s]. It ships
**no default parameter values** and refuses `C < 1e5 J/K` or `UA < 1 W/K` — a hard
thermal constraint built on an invented capacitance is a guarantee about nothing.
`RCThermalModel.identify` fits `(C, UA, Φ_g)` per building by ordinary least squares on
logged `(T_in, T_out, Φ)` and returns an `RCThermalFit` (R², one-step RMSE in K, sample
count, time constant τ = C/UA in hours) alongside the model, so the fit quality is
reported rather than assumed.

`ThermalComfortBarrier` subclasses `stems.deadline.DeadlineStorageBarrier` — the same
class the hot-water tank, the house battery and the electric vehicle instantiate, so
the framework claim is structural and a test asserts the `isinstance`. The store is the
zone's sensible heat, the rate limit is the heat pump's per-step temperature gain at
full action, the required level is the set-point band less a tolerance θ, and the
deadline is the current step at every occupied step. One instance per side of the band:
`direction=+1` raises the heat-pump action towards the heating floor, `direction=−1`
lowers it towards the cooling ceiling.

The barrier's state is the temperature **predicted one step ahead with the heat pump
off**. That matters: the free-running drift is then outside the controllable part, the
delivered temperature change is exactly linear in the action, and the base class's
`action_for_soc_gain` is exact rather than a linearisation.

`build_comfort_barriers` is the gate. It is **off by default** (`ComfortConfig.enabled
= False`) and refuses, with a specific reason, four configurations in which it could
not actually bound a temperature: no heat-pump action; `hvac_control="setpoint"`; no
heating set point in the observation vector; and no `RCThermalModel` supplied.

**Why `hvac_control="setpoint"` is refused.** Under the set-point mode — which is
`experiments/scenario.py::Scenario`'s default — the heat-pump action column is a
±1.5 °C set-point offset and `stems/environment.py::thermostat_step` issues the power
command through an integral loop the controller does not command. Projecting that
column cannot bound a temperature. This is the same dimensional error already
documented for `CBFShield._apply_hvac_power_guard`; it is refused here rather than
repeated.

**Why the barrier is off on CityLearn, measured.** Two findings, both from this track:

1. Identifying the envelope from the **dataset's own series** — `energy_simulation.
   indoor_dry_bulb_temperature`, the weather file's outdoor temperature, and the
   recorded heating and cooling demands — returns a **non-physical fit**. On
   `citylearn_schemas/tx_travis_8b`, winter training window, building 0:
   `a = −2.158e−3 K/K per step` and `b = −5.880e−5 K/W per step`, i.e. negative `UA`
   and negative `C`. The cause is that CityLearn's dataset reports the *ideal thermal
   load* needed to hold the set point, not delivered power against a free-running
   temperature, so the pair is not an input-output pair for an envelope at all.
   `identify_from_citylearn` raises instead of clipping the fit, and a test pins the
   raise.
2. Identifying it from an **open-loop excitation rollout** — the heat-pump action
   driven with an independent ±1/0 random signal per building, the *simulated*
   temperature recorded, 672 steps, seed 0, same scenario — returns a signed-physical
   but quantitatively useless envelope:

   | quantity | across the 8 buildings |
   |---|---|
   | `C` | 13.55 – 103.94 MJ/K |
   | `UA` | 779.8 – 7937.6 W/K |
   | `τ = C/UA` | 2.27 – 5.68 h |
   | R² on ΔT_in | 0.303 – 0.613 |
   | one-step RMSE | **4.231 – 8.894 K** |

   A one-step residual of 4–9 K against a 2 K comfort tolerance means the model's own
   error is two to four times the bound it would enforce. A `UA` of 7.9 kW/K is also
   not a dwelling. This is the quantitative version of `docs/REPORT_2026-10.md` §2
   ("−0.25 cooling in winter lowers it 4–13 °C, which is not physical"; "one house's
   model does not respond to cooling in autumn at all"), and it is the reason comfort
   stays a soft reward term here and **no hard-comfort claim is made on CityLearn**.

**Why the design is nonetheless standard.** `docs/LITERATURE.md` (Thermal comfort row)
records that in the optimisation literature hard comfort is normal practice:
[panagi2026thermal] embeds a calibrated 3R2C grey-box thermal model in a
network-constrained optimal power flow "while explicitly enforcing thermal comfort,
Distributed Energy Resource (DER) limits, and full power flow physics". The condition
is a physical model one trusts. `identify_from_rollout` is the function that supplies
one the day real measurements exist — the same call, the same barrier, the same arms,
with a fit summary that can be inspected before the claim is made.

**Evidence.** `tests/test_comfort_barrier.py`, 22 cases. The identification recovers
`C`, `UA` and `Φ_g` from data generated by a known model to `rtol=1e-6`, `1e-6` and
`1e-4` with R² > 0.999; it refuses a non-physical fit; the barrier is a
`DeadlineStorageBarrier`; `gap × capacity` is the zone's missing sensible heat
`C ΔT / 3.6e6` in kWh and `energy_still_required_kwh` divides it by the coefficient of
performance to give electrical kWh; the projected heating action lands the predicted
temperature exactly on the floor (`atol=1e-4 K`) and the cooling action exactly on the
ceiling; the barrier only ever raises a heating action and never lowers it; over 200
randomised band and temperature draws the two sides never bind together; and each of
the five refusals raises with its own message. End-to-end on the real simulator:
`tests/test_constraint_mechanisms.py::test_the_comfort_arm_runs_end_to_end_against_an_identified_envelope`
and `::test_the_dataset_route_refuses_rather_than_returning_a_wrong_envelope`.

### 1.2 Two comfort KPIs, reported separately

**What.** `MetricsCalculator` gained `comfort_barrier_binding_rate` (how often the
barrier had to intervene) and `comfort_band_breach_rate` (how often the band broke
anyway, despite it).

**Why.** They are different facts and collapsing them would hide the one that matters:
a hard constraint whose breach rate is not zero is not hard. Reporting both is how the
simulation-only status of the guarantee stays visible in the run record.

**Evidence.** `tests/test_constraint_mechanisms.py::test_the_comfort_arm_runs_end_to_end_against_an_identified_envelope`
asserts both keys are present and that the breach rate is a rate.

---
