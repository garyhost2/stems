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

# Baselines track — branch `methods/baselines`

Nine comparison controllers: the seven methods of the STEMS paper's Table I, ported to
reference implementations; a model-predictive pair, causal and perfect-foresight; and
MAPPO with a centralised critic. Audit finding ids refer to `docs/AUDIT_2026-10-06.md`.

**Test floor.** The brief gave the floor as "419 passing, 0 failed". In this worktree
the suite collects 419 and reports **404 passed, 15 skipped, 0 failed** before any edit
of mine (`python -m pytest -q`, 123.75 s). The fifteen skips are all dataset
availability, not failures: ten in `tests/test_env_widening.py` skip because the
generated schema `citylearn_challenge_2022_phase_all_plus_evs` is not present, and five
in `tests/test_ev_real.py` skip because fetching the CityLearn electric-vehicle dataset
returns HTTP 403 from the GitHub API (rate limit) from this machine. So the floor I was
asked to hold is 419 *collected*; the floor I can actually hold is 404 passed, 0 failed,
and that is the number every figure below is measured against.

**Nothing here was trained.** The learning arms are built, validated and left at their
initialisation. Every number below comes either from the synthetic mock environment
(known-answer tests) or from short rollouts on `citylearn_schemas/tx_travis_8b`.

---

## Step 1 — Port the seven STEMS Table I controllers

**What.** `stems/baselines.py` rewritten below `RuleBasedAgent`, which is untouched.
`SingleAgentSAC`, `DMAPPOAgent`, `MADDPGAgent`, `MARLISAAgent`, `MADCQAgent` and
`MetaEMSAgent` are now reference implementations; `MPCAgent` is a name that raises and
points at `stems/mpc.py` (step 2).

| arm | what it now implements | named deviation |
|---|---|---|
| `sac` | Soft actor-critic (Haarnoja et al. 2018) with automatic temperature tuning, twin target critics, replay. One agent over the **concatenated** district observation and joint action; reward is the district sum. | — |
| `dmappo` | Independent PPO per building on local observations (DTDE): GAE, clipped surrogate, clipped value loss, entropy bonus, advantage standardisation, approximate-KL early stop. | — |
| `maddpg` | MADDPG (Lowe et al. 2017): deterministic local actors, per-agent critic on the joint observation and action, target networks, replay. | Gaussian exploration noise with geometric decay instead of the paper's Ornstein–Uhlenbeck process. |
| `marlisa` | MARLISA (Vázquez-Canteli, Henze, Nagy 2020): soft actor-critic with iterative sequential action selection, plus information sharing. | **"MARLISA with measured sharing".** The paper shares each building's *predicted* consumption from an online regressor. No source in `docs/LITERATURE.md` specifies that regressor's form or fitting protocol, so the shared signal here is the *measured current* `net_electricity_consumption`, which the observation already carries. |
| `madcq` | Constrained Q-learning (Kalweit et al. 2020) — the maximisation runs over a safe action set in **both** the greedy policy and the bootstrap target — over a branching architecture (Tavakoli et al. 2018), with Double DQN (van Hasselt et al. 2016), replay, a target network and linearly decayed ε. | **The name could not be resolved.** No entry in `docs/LITERATURE.md` maps "MADCQ" to a paper, so this is reported as the constrained-Q family described, not as a reproduction of an unidentified source. |
| `metaems` | Reptile (Nichol et al. 2018) on a shared soft actor-critic, with an `adapt` method for test-time adaptation. | **"Reptile-SAC, buildings as tasks".** Meta-learning needs a task distribution and the STEMS row's could not be identified. A task here is a building: the eight differ in battery capacity (5.0–16.2 kWh), nominal power (1.61–5.0 kW), tank capacity and photovoltaic rating. |

**Why.** Audit A3. Each of the seven was exported from `stems/__init__.py`, instantiated
by no experiment, and as written would have been a strawman in a comparison table.

**The specific defects, and where each is now pinned.**

1. *MPC held price constant over its horizon, was battery-only, used a hard-coded 0.1
   state-of-charge rate, and had no load, photovoltaic, comfort or cap model.* Replaced
   wholesale (step 2). `tests/test_baseline_controllers.py::test_the_mpc_agent_name_refuses_rather_than_resurrecting_the_old_one`.
2. *D-MAPPO used a one-step temporal-difference residual as its advantage.* Now GAE;
   `stems/baselines.py::_gae`.
3. *D-MAPPO stepped the optimiser inside the epoch loop against an `old_log_prob`
   recomputed from the already-updated actor.* The behaviour log-probability now comes
   from the trajectory and is fixed for the whole update.
   `test_dmappo_uses_the_behaviour_log_probability_from_the_trajectory` asserts the
   approximate KL of the first minibatch is below 1e-4 (not zero: the behaviour
   log-probability was computed on a one-row tensor and is recomputed on an 80-row one,
   and float32 matrix multiplication is not associative; a *recomputed* old
   log-probability gives a KL two or more orders of magnitude larger).
4. *D-MAPPO subtracted a "CBF penalty" computed from `soc + 0.1 * action`.* Deleted.
   `test_dmappo_has_no_hard_coded_battery_model`.
5. *MADCQ discretised to 11³ = 1331 joint actions with fixed ε and no replay buffer.*
   Now 3 × 11 = 33 branched outputs, decayed ε, replay, Double DQN, and the safe set
   from `BatteryModel.safe_interval`. `test_madcq_does_not_enumerate_the_joint_action_space`
   (33 < 1331) and `test_madcq_never_proposes_an_action_outside_the_safe_set` (worst
   excursion beyond the exact plant's feasible interval ≤ 0.02, a tenth of a bin width).
6. *MARLISA augmented with same-timestep actions when acting and stored actions when
   updating.* One `augment` function serves both paths, and both augmented vectors are
   built at collection time, where step *t+1* is available, and stored in the replay
   buffer. `test_marlisa_builds_the_same_augmentation_when_acting_and_when_updating`.
7. *MetaEMS was SAC with a 50/50 parameter average and was not MAML or Reptile.* Now
   Reptile over buildings-as-tasks with a fresh inner optimiser per task.
   `test_metaems_is_reptile_over_more_than_one_task`.

### 1.1 Two bugs found by the known-answer tests, and what actually caused what

The known-answer test is the one from `tests/test_learning.py`: pay
`r = -4(a_h - 0.6)² - 4(a_e + 0.3)²`, train briefly, require the deterministic policy to
move toward `(+0.6, -0.3)`. Three changes were made while chasing two failures, and they
are recorded separately because two of them were *not* the fix.

**Observation normalisation (`_ObsNorm`), real effect measured.** The observation vector
mixes an hour index in 1..24, temperatures near 20 °C, irradiance in hundreds of W/m²
and a state of charge in [0, 1]; none of the ported learners normalised it, while
`stems/agent.py::STEMSAgent` always has. Effect on the single-agent soft-actor-critic
arm: from `a_e = -0.089` after 14 episodes (short of the test threshold) to
`(a_h, a_e) = (+0.93, -0.83)` after 12. Effect on the proximal-policy arm: **none** —
still `a_h = -0.002`. Kept because it is right on its own merits and makes the arms
comparable on input scaling, not because it fixed anything.

**Return scaling in `DMAPPOAgent`, no measured effect on the failure.** Added for the
same reason `STEMSAgent` has it (the critic starts wrong by the magnitude of the
discounted return, order 300 on an 80-step episode). Measured with the real bug still
present: `a_h = +0.072` at episode 40 without it, `+0.024` at episode 20 with it. Kept
on its own merits; it was not the fix.

**The actual bug: an aliased behaviour buffer.** `DMAPPOAgent.select_action` filled one
reusable `_last_pre_tanh` array in place. The trajectory collector stores the reference
it is handed, so every transition in the episode held *the same array object* and
therefore the last step's pre-squash sample. The importance ratio was computed against
one draw repeated N times and the policy gradient carried no per-step signal. Shapes
were right, losses moved, nothing else showed it. After allocating fresh arrays per
call: `a_h` went from `-0.002` to `+0.169` at the same budget, and the arm passes at
twice the episodes. `tests/test_baseline_controllers.py::test_the_stored_behaviour_samples_are_not_all_the_same_object`
is parameterised over all seven learning arms and checks both that the object identities
differ across steps and that the stored samples actually vary. `ComparisonController`
now copies defensively as well.

**Evidence.** `tests/test_baseline_controllers.py`, 40 cases, all passing: every arm
builds and acts inside `[-1, 1]`; every arm presents the attributes
`experiments/runner.py::train` reads; six arms reach the known optimum and the seventh
(`metaems`) reduces its distance to it by more than 40% — stated as a paired reduction
rather than an absolute threshold because Reptile's landing point after a fixed budget
is noisy.

---

## Step 2 — The model-predictive pair

**What.** `stems/mpc.py::StorageMPC` and `stems/forecast.py`. One controller class; the
two arms differ **only** in which forecaster is plugged in, so a difference between them
is a forecast effect and nothing else (`tests/test_mpc.py::test_the_two_mpc_arms_differ_only_in_the_forecaster`).

* `mpc` — receding horizon, `CausalForecaster`: the current observation and a rolling
  24-step window, which is exactly the reinforcement-learning agent's information set.
* `mpc-oracle` — `OracleForecaster`: replays a recording of the identical environment.

`mpc-oracle` is a **new arm name**, added to `ARMS`. The dict was not restructured; the
perfect-foresight controller is a second controller, not a mode of the first, and
registering it as its own arm keeps the run records separable.

**Formulation.** Decision variables per building per step: battery charge and discharge
fractions, hot-water tank charge and discharge fractions, and grid import. Objective is
the import bill over the horizon at the real price series, plus heavily weighted slacks.
Constraints: the import definition, the per-building cap, the district cap, the
state-of-charge band, and a terminal condition requiring the battery to be handed back
at least as charged as it was found. Solved with HiGHS through cvxpy, with the problem
compiled once and only its parameters updated per step.

**Comfort and the heat pump.** The heat pump is not a decision variable, and this is a
limitation of the testbed rather than of the formulation. In `hvac_control="setpoint"`
— the scenario default — `STEMSEnvironment.step` overwrites the controller's heat-pump
column with a thermostat loop regardless, so the only lever is a set-point shift, and
predicting its effect requires an indoor-temperature model, which here means CityLearn's
learned dynamics. `docs/REPORT_2026-10.md` §2 shows those are not physical. An MPC
optimising against them would report a comfort result that is an artefact of the
surrogate. The comfort band is therefore held by the same proportional thermostat every
other arm uses, the heat pump's electricity enters the optimisation inside the forecast
base load, and in `hvac_control="power"` the MPC runs that identical thermostat itself
so the heat pump behaves the same under every arm.

### 2.1 Price forecast leads: measured, not assumed

`electricity_pricing_predicted_1` and `_predicted_2` reproduce the realised price at
**t + 6 h** and **t + 12 h** with a mean absolute error of **0.00000** currency/kWh over
300 steps on `tx_travis_8b`; every other lead in {1, 2, 3, 6, 12, 24} h is off by ≥ 0.07.
The tariff has exactly three levels (0.22, 0.40, 0.54 currency/kWh) and a 24 h period.
`CausalForecaster` is therefore exact at leads 0, 6 and 12, and holds the last known
level between anchors rather than interpolating — a linear interpolation of a step
tariff invents prices the tariff never charges.
`tests/test_mpc.py::test_the_causal_forecaster_uses_the_price_leads_it_is_given`.

### 2.2 The base load is measured by subtraction, not composed from parts

The quantity the MPC cannot change is defined as
`base = net_electricity_consumption − battery terminal energy − tank electricity`, both
storage terms from the exact models at the pre-step states and the applied action.
Measured against the alternative of composing it from
`non_shiftable_load − solar_generation + dhw_demand/efficiency + heating_electricity`:

| decomposition | residual mean [kW] | residual s.d. [kW] | worst \|residual\| [kW] |
|---|---|---|---|
| by subtraction (used) | 0.055 | 0.623 | 5.49 |
| composed from parts | 0.010 | 1.198 | 7.32 |

(mean \|net\| over the same window = 2.993 kW, eight buildings, 120 steps). The
subtraction form predicts the realised net consumption of a random-action rollout with
*R*² = 0.964, so it is the one used, and its 0.62 kW residual is reported rather than
hidden: it is unmodelled hot-water coupling, because discharging the tank changes how
much the heater must make up on later steps.

### 2.3 The oracle is a perfect-foresight reference for the exogenous signals only

Its tape is recorded by rolling the same environment once with the zero action
(`record_idle_episode`). Price and hot-water draw are exogenous, so the recording gives
them exactly. The base load is the *idle counterfactual*, which is not exactly what the
oracle's own trajectory will produce — the gap is the 0.62 kW above. Stated in the class
docstring and here rather than presented as an exact oracle. Per
`docs/LITERATURE.md` §10, a perfect-foresight oracle could not be established as field
convention, so it is presented as **our** choice of reference point.

The tape costs one extra episode and no correctness:
`tests/test_mpc.py::test_resetting_after_the_oracle_tape_restores_the_environment` checks
that every building's observation after the post-recording reset is bit-identical to the
one before the recording.

### 2.4 Planning in state of charge, not in action

The exact plant is nonlinear, so the linear programme needs an affine model. The first
version used a secant through the idle, full-charge and full-discharge corners and
treated the solution as a normalised action. That is wrong at the state-of-charge band,
and the test said so by how much: the battery saturates as it fills, the true next state
of charge is concave in the command, and the secant sits below it, so a trajectory
constrained to `[0.2, 0.8]` left the band by **0.025** and the secant itself was off by
up to **0.110** across the eight real batteries.

The controller now plans a *state-of-charge trajectory*, and
`BatteryModel.action_for_soc` / `TankModel.action_for_soc` invert the exact model by
bisection to find the command that realises each step. The band constraint is then exact
(`test_the_planned_state_of_charge_is_reachable_exactly`: worst miss < 1e-6), and only
the *cost* model stays approximate. Sequential linear programming refits the energy
coefficients at the operating point the previous pass chose; after convergence the
energy the optimiser charges itself for the first step agrees with what the exact plant
draws to within 1% (`test_the_energy_coefficients_converge_on_the_operating_point`).
`test_reading_the_plan_as_a_raw_action_would_leave_the_band` and
`test_the_secant_state_of_charge_model_would_not_have_been_good_enough` pin both of the
measured failures above, so the design choice is justified by numbers and will fail
loudly if the plant ever changes enough to make it unnecessary.

`TankModel` gained `next_soc` and `action_for_soc`, both factored out of the arithmetic
`drawn_kwh` already performed and discarded, so the tank's state and its electricity
cannot be described by two different implementations.

---

## Step 3 — MAPPO with a centralised critic

**What.** `stems/mappo.py::MAPPOCentralisedCritic`, arm `mappo-cc`. Yu et al. (2022):
parameter-shared actors on local information, one critic on the joint state, GAE,
clipped surrogate, entropy bonus, approximate-KL early stop, observation and return
normalisation.

**What the critic conditions on.** The shared spatio-temporal encoder
(`stems/encoder.py::STEncoder`, the same class `STEMSAgent` uses) maps the *B* current
observations and their 24-step histories to a representation matrix *R* ∈ ℝ^(B×d). The
actor reads row *R_i* — local information after graph mixing, exactly what `STEMSAgent`'s
actors read, so the two are comparable. The critic reads the **concatenation of every
row**, *s* = vec(*R*) ∈ ℝ^(Bd), and returns one value per building from a single head:
the agent-specific global state of Yu et al. Concatenation rather than a
permutation-invariant pool is deliberate — the buildings have different device sizes and
the critic is allowed to know which is which.

**How it differs from D-MAPPO.** D-MAPPO's critic for building *i* is
*V_i*(*o_i*) — a separate network per building reading only that building's own raw
observation. Nothing in it can represent the shared import limit. `mappo-cc`'s critic is
*V*(*s*) over the joint encoded observation. That is the only structural difference
between the two arms: both are on-policy PPO with the same clipping, the same GAE, the
same entropy bonus and the same safety layer.

**Why it is the arm that settles the question for us.** `docs/LITERATURE.md` has Khouja
et al. (2026) reporting decentralised beating centralised on CityLearn *without* a
district cap, and Shojaeighadikolaei et al. (2024) reporting a centralised critic
helping when electric vehicles share a transformer. The clean 2×2 — one algorithm, one
environment, cap on and off, DTDE and CTDE in each cell — appears in no retrieved source.

**A caveat for whoever runs that 2×2.** On `tx_travis_8b` with eight buildings the
uncontrolled district peak import is **39.6 kW** against the default
`grid_cap_kw = 300`. The cap is not binding, so the "cap on" column would be identical
to the "cap off" column and the experiment would measure nothing. The cap has to be set
near the uncontrolled peak for the coupling to exist. Recorded in the module docstring
as well, because it is the kind of thing that is discovered after the grid has run.

---

## Step 4 — Validation, and what it caught

**What.** `experiments/validate_controllers.py` is the eligibility gate. For every arm
it runs a rollout on the real environment, applies
`experiments/runner.py::ActuatorEvidence` — which correlates commands against what the
devices actually did — and times `select_action`. Learning arms are driven with
exploration on, because an untrained deterministic policy can emit a near-constant
command and the check then cannot separate "the actuator is dead" from "the command
never varied". Output: `experiments/diagnostics/controller_validation/validation.json`.

### 4.1 The gate caught two bugs that no known-answer test could have

Both were in the model-predictive controller, both were invisible to every test that
looks only at states of charge or at plan optimality, and both were found by looking at
what the controller actually commanded over a real rollout.

**Bug 1: the hot-water command was pinned at −1.000 for all 960 building-steps.**
`action_for_soc` bisected over the whole action interval for the action achieving a
target state of charge. But `next_soc` is *flat* in the action wherever the device
cannot move energy, and the tank is the acute case: it discharges only into an actual
hot-water draw, so with no draw its next state of charge is identical for every
non-positive action. A plain bisection converges to an end of that flat region — a
full-discharge command, issued every step, for a tank that cannot discharge. It changed
nothing in the model, which is why it was silent. Fixed by bracketing around zero
(`stems/battery.py::_bisect_toward_zero`): the returned action is the
smallest-magnitude one achieving the target, so a target that idling already reaches
maps to exactly 0. Pinned by `tests/test_mpc.py::test_the_plant_inverse_returns_the_idle_command_on_a_flat_region`
and `::test_the_mpc_does_not_hold_the_hot_water_command_at_an_extreme`.

**Bug 2: the tank was then never used at all — a self-confirming fixed point.** With
the inverse fixed, the hot-water command became exactly 0.000 at every step. The cause:
`_tank_corners` read the discharge coefficient at the *current* state of charge. The
tank model's one-step drop is min(draw/η, stored)/C — a *rate* limit from the draw and
an *energy* limit from what is stored — and reading both at the current state conflates
them. An empty tank reports that discharging is worth nothing, so the optimiser never
charges it, so it stays empty. Measured: `gd_t` was identically zero at every step of a
200-step rollout even with 1.4 kWh of forecast draw inside the horizon. Fixed by
evaluating the discharge *rate* at a reference state holding enough energy and leaving
the energy limit to the linear programme, where it already is as soc ≥ 0. After the
fix, over the same 200 steps: hot-water commands span [−1.000, +0.751], tank state of
charge reaches 0.933 with a mean of 0.232, and the cost saving against doing nothing is
7.71%.

Neither bug would have been caught by a known-answer test on the plan, and the first
would not have been caught by any test of the state-of-charge band. The gate exists
because of this class of defect.

### 4.2 Eligibility

Window: 336 steps (two weeks) on `citylearn_schemas/tx_travis_8b`, eight buildings,
winter, seed 0. Every arm below reaches `verified = True`: battery, hot-water and
heat-pump channels all respond to their commands.

| arm | verified | battery | hot water | heat pump | median ms/step | p95 ms/step |
|---|---|---|---|---|---|---|
| `idle+calibrated` | **None** | None | None | True | 1.671 | 1.769 |
| `rbc+calibrated` | True | True | True | True | 1.714 | 1.749 |
| `sac` | True | True | True | True | 1.937 | 2.086 |
| `maddpg` | True | True | True | True | 2.033 | 2.200 |
| `mappo-cc` | True | True | True | True | 2.311 | 2.454 |
| `dmappo` | True | True | True | True | 2.588 | 2.794 |
| `metaems` | True | True | True | True | 2.692 | 2.897 |
| `marlisa` | True | True | True | True | 2.785 | 2.972 |
| `rl+calibrated` | True | True | True | True | 3.090 | 3.332 |
| `madcq` | True | True | True | True | 4.109 | 4.313 |
| `mpc` | True | True | True | True | 75.291 | 77.518 |
| `mpc-oracle` | True | True | True | True | 75.311 | 77.919 |

**Nine comparison controllers are eligible. None failed.** The only `None` is
`idle+calibrated`, which is an incumbent arm, not one of mine: it commands nothing by
definition, so there is no contrast to correlate and the check correctly answers
"cannot tell" rather than "broken". That is worth stating plainly, because it shows
`None` is not a synonym for failure — the eligibility rule has to be *`False` fails*,
with `None` reported together with the reason, or the repository's own do-nothing
reference would be disqualified.

**A knife-edge worth recording.** On a 168-step window the causal `mpc` returned
`verified = None`, because its hot-water channel produced 19 charging samples against
the check's minimum of 20 (`mpc-oracle` had 23 and passed). Nothing was wrong with it;
the window was too short for a controller that uses the tank sparingly. The window was
doubled to 336 steps and both pass. Recorded because "the gate said None" and "the gate
said None because it was one sample short" are different facts, and only the second is
true here.

### 4.3 Computation time per control step

Reported because this literature reports it: [maier2023approximating] matches a
rule-based controller at 15% of an MPC's compute. The measurement is around
`select_action` only — environment stepping and metric bookkeeping are excluded, and
are identical across arms. Single-threaded on CPU, `OMP_NUM_THREADS=4`.

The learning arms cost **1.9–4.1 ms** per control step, within a factor of 2.4 of the
rule-based controller's 1.71 ms, and the repository's own agent sits mid-range at
3.09 ms. The model-predictive pair costs **75.3 ms**, about **44 times** the rule-based
controller and **24 times** the STEMS agent — a 12-hour horizon, eight buildings, two
storage devices each, and two sequential-linear-programming passes per step. Put the
other way round, and in the form that literature uses: the rule-based controller runs
at **2.3%** of the MPC's computation time. All of it is solver time; the problem is
compiled once by cvxpy and only its parameters are updated per step, so this is the
cost of 2 HiGHS solves, not of rebuilding the model.

### 4.4 Every exported controller is reachable from `ARMS`

Audit A3's "exported but unreachable" finding is now checkable in both directions.
`stems/__init__.py` declares `CONTROLLERS_REACHABLE_FROM_ARMS`, and
`tests/test_controller_reachability.py` (24 cases) asserts that the declaration matches
`stems.__all__`, that every arm it names exists, that every exported controller is
actually constructed by some arm's builder, that the nine comparison arms build nine
distinct implementations (the MPC pair sharing one class by design), and that each
carries a substantive docstring. `MPCAgent` is no longer exported.

---

## Tests changed, and why

Three test files were edited to accommodate changes. Recording each under the
test-editing rule.

**`tests/test_preregistered_arms.py`.** Step 8 registered 24 arm names and asserted
that every one reports `implemented is False` and raises `NotImplementedError`. Eight
of them are what this track was asked to build, so that assertion is the exact
statement the work falsifies. `ALL_PRE_REGISTERED` is now the 16 arms still reserved
for the constraint track (12 mechanism cells, 2 comfort, 2 degradation), and
`test_the_comparison_controllers_cover_table_one_plus_the_two_arms_it_lacked` asserts
the opposite of what it used to: that all nine now resolve to controllers, are absent
from `PRE_REGISTERED_ARMS`, and carry a description. What the old test pinned that
still matters — the names exist, MAPPO with a centralised critic is distinct from
D-MAPPO, each carries a note — is pinned against `COMPARISON_POLICIES` instead.

`test_learning_arms_are_unchanged_by_the_new_policies` asserted `learns is False` for
all eight, with the stated reason "until it has a builder". They have builders now, and
seven of them take gradient steps; leaving `learns` False would put a
randomly-initialised network into a comparison table, which is the strawman this track
exists to remove. It is now
`test_the_learning_comparison_arms_now_ask_for_a_training_budget`, which requires
`learns` True for the seven learners and False for the model-predictive pair.

**`tests/test_experiments.py::test_the_ablation_arms`.** This is an inventory of the
arms that have a builder, so implementing nine of them necessarily changes it. The nine
were added to the expected mapping and the seven learners to the expected `learns`
list, with an explicit assertion that `mpc` and `mpc-oracle` remain non-learning. No
assertion was weakened: the test still fails if an implemented arm is added or changed
without updating it.

**`ARMS` gained one entry, `mpc-oracle`.** The dict was not restructured. The brief
named two model-predictive controllers and the registry had one name; a perfect-
foresight controller is a second controller, not a mode of the first, and separate arm
names keep its run records separable from the causal arm's.

---

## What this track disputes

1. **The test floor is 404 passed, 15 skipped, not 419 passed.** Measured before any
   edit; the 15 skips are dataset availability on this machine (a missing generated
   electric-vehicle schema, and an HTTP 403 rate limit from the GitHub API). 419 is the
   collected count.
2. **The MPC horizon is 12 h, not the 24 h the tariff-period argument gives.** The
   argument was mine and the sweep overruled it: both forecasters saturate at 12 h, the
   causal arm is slightly *worse* at 24 h, and 24 h costs 1.9 times the solve time.
3. **`verified = None` must not be read as failure.** The incumbent `idle+calibrated`
   arm returns `None` by construction. The eligibility rule is `False` fails; `None` is
   reported with its reason.

## What is not done, and should not be assumed

* **Nothing was trained.** Every learning arm is validated at its initialisation. The
  known-answer tests show each one can optimise a stationary objective on the mock
  environment; they say nothing about performance on CityLearn.
* **No comparison table exists yet**, and none should be made from these numbers. The
  cost figures here come from single-seed rollouts of 168–336 steps, used to validate
  controllers, not to rank them.
* **The district cap does not bind in the default scenario** (39.6 kW uncontrolled peak
  against a 300 kW cap), so the centralised-versus-decentralised question `mappo-cc`
  exists to answer cannot be asked at that setting.
* **A cost-only MPC raises peak demand** when the cap is slack — by up to 58% over
  doing nothing in the sweep. Any table reporting peak demand must set the cap near the
  uncontrolled peak, or this arm will look bad for a reason unrelated to model-
  predictive control.
* **`MARLISA`, `MADCQ` and `MetaEMS` are named variants, not reproductions.** Their
  defining sources could not be verified against `docs/LITERATURE.md`; what was
  implemented is described above and must be reported under those descriptions.
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

## Step 2 — Battery and electric-vehicle degradation, priced and constrainable

### 2.1 `stems/degradation.py`: capacity fade in kWh, with every parameter sourced

**What.** `battery_equivalent_full_cycles` was a count the repository produced and never
priced. It is now accompanied by a quantity with units — kWh of lost storage capacity —
computed by `DegradationModel` and `DegradationAccountant` and reported next to it.

Three terms, each with its source.

**Throughput.** `ΔQ_thr,t = κ · Q_rated · |E_t| / (2 · Q_t)` [kWh]. This is
*character-for-character the expression CityLearn itself applies*:
`citylearn/energy_model.py::Battery.degrade` in 2.6.0b1 returns
`capacity_loss_coefficient * capacity * abs(energy_balance[time_step]) / (2 *
max(degraded_capacity, ZERO_DIVISION_PLACEHOLDER))`. Using it means the reward prices
exactly what the plant does rather than a second, disagreeing model. `κ` is read per
building off the simulator's own `Battery` objects through the new
`STEMSEnvironment.battery_degradation_info()`. On `citylearn_schemas/tx_travis_8b` the
eight values are `6.09e−5, 3.18e−5, 7.32e−5, 4.46e−5, 6.78e−5, 3.49e−5, 3.66e−5,
9.08e−5` — inside CityLearn's own documented default range `(1e−5, 1e−4)` (same file,
`Battery` docstring and the `capacity_loss_coefficient` setter). The factor two makes
`κ` the fade per *equivalent full cycle*, one such cycle being `2 Q_rated` of
throughput; a test checks that arithmetic directly. The same call also exposes
`depth_of_discharge`, which on that schema is `0.9, 1.0, 0.85, 1.0, 1.0, 0.9, 0.9, 1.0`.

**Depth of discharge.** Cycle life falls with cycle depth, so fade per unit of
throughput rises with it. Writing `N(δ) = N_ref δ^−(p+1)` gives a multiplier `δ^p` on
the throughput fade accumulated inside a closed half cycle. **`p` defaults to 0.0,
which reproduces the throughput model exactly.** That default is deliberate: no value
of `p` is asserted anywhere in this repository, because none was verified. The standard
empirical reference for the shape is Xu, Oudalov, Ulbig, Andersson and Kirschen,
"Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment", IEEE
Transactions on Smart Grid, DOI `10.1109/TSG.2016.2578950`. Its title and DOI resolved
through Crossref; the paper is **closed access** (Unpaywall, Semantic Scholar and PMC
all returned no open-access location) and **its fitted stress-function coefficients
were not read**, so none are reproduced here. Setting `p` to a non-zero value is a
modelling choice whoever sets it must source.

**Calendar.** `ΔQ_cal,t = c_cal · Q_rated · dt`, `c_cal` in 1/s. No default: calendar
fade depends on chemistry, temperature and mean state of charge, none of which the
simulator reports. Off unless supplied.

**Pricing.** `π_deg` [currency per kWh of lost capacity], default 0.0 — fade is
measured and reported but does not enter the reward. A replacement cost is a market
number belonging to the scenario, so it is a `Scenario` field
(`degradation_price_per_kwh`), not a constant in the code. When set, the penalty is
`π_deg · ΔQ_t`, in the same currency per kWh as the tariff, so the degradation cost and
the electricity bill are additive.

**Constraint.** The episode carries a per-building capacity-loss budget `L` [kWh]
(`Scenario.degradation_limit_kwh_per_episode`). The per-step cost signal is
`1{ΔQ_t > L / T}` for a `T`-step episode. That is the same per-step 0/1 indicator shape
as the three existing channels, which is what lets the same cost critics carry it with
no change to the critic, the generalised-advantage estimator or the multiplier update.

### 2.2 Half cycles are closed by a hysteresis-filtered reversal detector

**What.** The depth term needs a cycle depth. Rather than rainflow counting, which is
not causal, the accountant tracks the state of charge at which the open excursion
started and the extreme reached since; a retracement smaller than
`reversal_threshold` (0.05 state-of-charge units, a repository choice, reported in every
run record) does not close anything, and a larger one closes a half cycle of that depth.

**Why, and this was found by measuring.** The first implementation closed a half cycle
on every sign change. On a two-day winter window the policy's state of charge jitters
every step, so it produced half cycles of mean depth `4.07e−4` — and with `p = 0.5`
the `δ^p` multiplier then discounted the fade by about 98%. Without the range gate the
depth term *rewards* jitter, which is the exact opposite of its purpose. The gate is
the standard range filter applied ahead of rainflow counting. It is an approximation to
rainflow, not rainflow, and a result that leans on `p > 0` should say so.

### 2.3 Honest placement: this is ahead of the field, not a reproduction of it

**What.** Stated here and in the module docstring rather than left for a reviewer to
notice.

**Why.** `docs/LITERATURE.md` records that in **every source the literature track
retrieved**, degradation is a scored KPI and never an enforced constraint;
[khouja2026characterizing] proposes "battery storage lifetime" explicitly as a *novel*
KPI, which is itself evidence that it was neither standard nor enforced. Wiring it as a
constraint is therefore a contribution to be defended, not a convention to be cited.
`docs/LITERATURE.md` §8 also notes that battery degradation modelling was not searched
as its own area, so the literature position above is provisional on that search.

Separately: the audit's one citation for a vehicle-to-grid degradation term, **Khezri
et al. 2024, did not resolve** against Crossref (`docs/LITERATURE.md` §8.1) and was
struck. It is **not** cited in `stems/degradation.py`, in this changelog, or anywhere
else in this track.

**Evidence.** `tests/test_degradation.py`, 18 cases. The throughput term is checked
against a local re-implementation of CityLearn's literal expression over a four-step
state-of-charge path to `rtol=1e-12`; two full state-of-charge sweeps cost exactly `κ`
of rated capacity to `rtol=2e-4`; `κ` read from an environment is asserted to lie in
`(1e−5, 1e−4)` and to carry its CityLearn provenance string; `p = 0` reproduces the
throughput total to `rtol=1e-9` over a 400-step random walk, which is also the proof
that every unit of throughput is committed exactly once and the depth term cannot
double-count; one 0.8-deep round trip costs strictly more than eight 0.1-deep round
trips of the same total throughput at `p = 1`; the range gate holds a jittery 0.1→0.9
climb to at most two closed half cycles instead of one per wiggle; calendar ageing
charges `c_cal · Q_rated · dt` with no throughput at all; the price defaults to zero;
the constraint channel is a `(B,)` 0/1 array; and the fade KPIs appear next to
`battery_equivalent_full_cycles` in `MetricsCalculator.compute_all`. End-to-end:
`tests/test_constraint_mechanisms.py::test_the_degradation_arm_runs_end_to_end_and_reports_fade`.

---

## Step 3 — A second constrained-RL family, and the fate of the learned filter

### 3.1 FOCOPS against the same cost critics

**What.** `stems/agent.py` gained `STEMSAgent.focops_advantage` and
`STEMSAgent._focops_loss`, selected by `LagrangianConfig.algorithm` ∈
`{"ppo-lagrangian", "focops"}` (`CONSTRAINED_ALGORITHMS`). FOCOPS — First Order
Constrained Optimization in Policy Space, Zhang, Vuong and Ross, NeurIPS 2020 — moves
the policy towards the non-parametric optimum `π* ∝ π_k exp((A − ν A_C)/τ)` by
first-order descent on `KL(π_θ ‖ π*)`, giving the surrogate

```
L = E_{a~π_k} [ ( KL(π_θ‖π_k)[s] − (1/τ) r(θ) (A − ν A_C) ) · 1{KL ≤ δ} ]
```

**Why FOCOPS and not CPO.** CPO needs a conjugate-gradient solve of the Fisher system
and a backtracking line search per update, i.e. a second-order machinery that shares
nothing with the existing first-order Adam path. FOCOPS is first-order, so it reuses
the identical optimiser, encoder and critics, and the comparison is between surrogates
rather than between optimisers.

**What is shared, deliberately.** The same `CostCritic` ensemble, the same width, the
same generalised-advantage estimates, the same encoder, the same standardisation of the
reward and cost advantages, the same multiplier state tensors, the same minibatching
and the same value and entropy losses. `focops_advantage` differs from
`effective_advantage` by exactly the missing `1/(1 + Σν_k)` denominator — a test asserts
`focops / (1 + Σν) == lagrangian` to `atol=1e-6` — because FOCOPS divides by its
temperature `τ` instead. The trust region `δ` defaults to `TrainingConfig.target_kl`
(0.02), the same threshold PPO already uses for early stopping, so the two families
operate in trust regions of the same size and the comparison is not confounded by it.

**Two deviations, stated rather than hidden.** (a) `π_k`'s distribution parameters are
not in the buffer — only its log-probabilities are — so `KL(π_θ‖π_k)` is estimated with
the standard non-negative, differentiable estimator `E[r log r − (r − 1)]` rather than
in closed form. (b) The published FOCOPS updates `ν` by plain projected gradient; here
`ν` is updated by whatever `LagrangianConfig` says, which defaults to this repository's
PID controller. That keeps the *only* difference between the two arms the actor
surrogate, which is what the brief asks for; setting `use_pid=False` recovers the
published update, and a run that wants the published algorithm should.

### 3.2 What our cost critics actually measure, named

**What.** `stems/config.py` now carries `BASE_CONSTRAINT_CHANNELS` with the definition
written next to it, and `constraint_channel_names(config)` as the one source of truth
for the channel list, its order and the critic width.

Every channel is a **per-step 0/1 indicator, evaluated per building**:

| channel | indicator |
|---|---|
| `soc_band` | `1{state of charge after the step outside [SOC_min, SOC_max]}` |
| `building_power_cap` | `1{|net electricity consumption| > P_building_max}` |
| `district_import_cap` | `1{Σ_b max(net_b, 0) > P_grid_max}`, identical for every b |
| `battery_degradation` | `1{ΔQ_t > L/T}`, present only when a budget is set |

So the quantity the cost critics estimate is a **discounted expected fraction of
violating steps** (GAE with γ = 0.99, λ = 0.95 on those indicators), and the quantity
the PID controller compares against `cost_limit` is the **undiscounted mean of the
indicators over the batch and over buildings**. In the three incompatible definitions
the literature track found — average episodic cost against a limit, fraction of
violating steps, and fraction of episodes with any violation — ours is the **fraction
of violating building-steps**. The literature track established that no common protocol
exists across the constrained-RL papers and that the three are not interconvertible, so
this definition has to travel with every number we report. `MetricsCalculator`'s
`safety_violation_rate`, `soc_violation_rate`, `power_violation_rate` and
`grid_violation_rate` are the same fraction measured on the evaluation rollout, which
is why they are comparable with the training-time cost at all.

Note the one asymmetry a reader should know about: `district_import_cap` is a district
quantity broadcast to every building, so a violating step contributes `B` violating
building-steps to that channel while a single building's `soc_band` violation
contributes one. The channels are therefore comparable across *arms* but not across
*channels*.

### 3.3 `NeuralSafetyFilter` removed

**What.** Deleted from `stems/cbf.py`, with the argument left in a comment at the site.
`import torch` and `import torch.nn as nn` went with it; they had no other user in that
module.

**Why remove rather than train it against the LP shield.** Three reasons.

1. **It can only lose on the axis we report.** A learned regression onto the shield's
   output carries no certificate that its output lies in the safe set. The repository's
   constraint claim is that the projection is *exact* against the simulator's own plant
   model; replacing it with an approximation can only raise the violation rate, and
   there is nothing it can win back.
2. **There is no cost to amortise.** A learned safety filter earns its place when it
   replaces an expensive online optimisation. Here the exact battery inverse is a
   24-iteration bisection on a scalar (`BatteryModel.safe_interval`) and the only
   optimisation in the loop is the fleet LP, which the filter was not written to
   replace. Trained, it would be a slower, less accurate version of a microsecond
   closed form.
3. **It did not implement its own interface.** `num_ensemble=5` and
   `uncertainty_threshold=0.05` are parameters of an ensemble-with-uncertainty-gate
   mechanism; the class held one trunk and one head, built no ensemble, and never read
   the threshold. Keeping it would have meant first fixing dead code to match its own
   signature.

**What this costs.** The brief's four constraint families become three: PID-Lagrangian,
FOCOPS, and projection (the exact barrier plus the LP fleet shield, which covers both
"safety layer / shielding" and "action projection"). That is still a comparison across
mechanism families, and it is an honest one.

**Evidence.** `tests/test_constraint_mechanisms.py`: both families registered and
`ppo-lagrangian` the default; an unknown family raises; FOCOPS and PPO-Lagrangian share
the cost-critic class, width and multiplier shapes; FOCOPS takes gradient steps on the
mock and on the real simulator and reports `mechanism_algorithm == "focops"`; the KL
estimate is exactly zero at the behaviour policy and non-negative away from it; the two
advantage combinations differ by exactly the Lagrangian denominator; and
`stems.cbf` no longer has the attribute, with a tree scan asserting nothing constructs
it.

---

## Step 4 — The constraint-mechanism 2×2, wired

### 4.1 The two mechanisms are decoupled

**What.** `experiments/controllers.py::mechanism_switches(arm)` returns
`(projection_on, lagrangian_on)`, and `build_controller` now reads both instead of
inferring them. `projection_on` drives `STEMSAgent(use_cbf=...)`, the `ShieldedController`
versus `PlainController` choice, **and the fleet shield** — the LP is a projection too,
so a cell with the projection off must have no projection of any kind or the cross
isolates nothing. `lagrangian_on` drives the new `LagrangianConfig.enabled`.

| `mechanism` | projection | constrained policy |
|---|---|---|
| `auto` | `barrier != "none"` | the policy learns |
| `none` | off | off |
| `lagrangian` | off | on |
| `projection` | on | off |
| `both` | on | on |

`"auto"` reproduces the historical coupling exactly, so all seventeen pre-existing arms
are unchanged; a test asserts that for every one of them.

**Why this was the missing ablation.** Today `rl` has the Lagrangian and no barrier
while `rl+calibrated` has both, so neither isolates a mechanism (audit C2). The twelve
cells `mech-{none,lagrangian,projection,both}+{uniform,linear,exact}` do, and a test
asserts that the twelve `Arm` records are identical in every field except `barrier` and
`mechanism` — so once the grid driver gives them a common `--episodes`, the constraint
mechanism is the only thing that differs.

**The one thing `enabled=False` must not switch off.** The cost critics keep training in
every cell, including `mechanism="none"`. They are the instrument that measures the
violation rate, and the cells have to measure it the same way to be comparable; what
`enabled` removes is only their effect on the actor. Two tests pin this: the cost-value
loss is non-zero in the `none` cell, and the multipliers do not move there while they do
move in the `lagrangian` cell.

### 4.2 Three of the twelve cells are the same controller — flagged, not hidden

**What.** `Arm.mechanism_is_plant_sensitive` is `False` for `mech-none+*` and
`mech-lagrangian+*`.

**Why.** The battery plant model is consulted only by the projection. With the
projection off, `mech-none+uniform`, `mech-none+linear` and `mech-none+exact` are the
same controller under three names, and likewise for `mech-lagrangian+*`. The cross keeps
all twelve so the factorial is complete and the grid driver needs no special case, but
**six of the twelve cells are duplicates of two**, and running them at five seeds each
would spend 30 runs to produce 10 runs' worth of information. The aggregation should
collapse them. (A smaller correction while here: the step-8 comment in
`experiments/controllers.py` said "the four with mechanism='none'"; the cross has three
plant models, so it is three. Corrected in the comment.)

### 4.3 Provenance of each run

**What.** `run_one` now records `meta.mechanism` (the two switches),
`meta.constraint_channels` (the channel list the collector actually produced),
`meta.rc_model` and `meta.rc_fit` for a comfort arm, and `meta.degradation_model` for a
degradation arm, and `meta.config` gained the `comfort` and `degradation` sections.
`agent.update` returns `mechanism_algorithm`.

**Why.** A 2×2 whose run records do not say which cell they came from is not a 2×2. The
fit summary in particular has to be in the record: a comfort result is only readable
next to the R² and RMSE of the envelope it was enforced against.

### 4.4 The envelope is identified on the training window only

**What.** When `arm.comfort_barrier`, `run_one` runs the excitation rollout on a
**training-window** environment and passes the resulting `RCThermalModel` to both the
training and the evaluation builds, through the new `build_controller(..., rc_model=)`.

**Why.** Identifying it on the evaluation window would fit the constraint to the period
it is scored on. This is the same time-based split rule the project applies to
everything else.

### 4.5 Cost-critic width is derived, never configured

**What.** `build_controller` sets `config.lagrangian.num_constraints` from
`constraint_channel_names(config)` *after* the degradation mode is resolved and *before*
the agent is constructed; `experiments/runner.py::finalise_constraint_width` does the
same for the collector and the run record.

**Why, and this was a real bug caught in a smoke test.** The critic's output width is
read at construction time. Deriving the channel list later built a three-output critic
for a four-column cost array.

### 4.6 Tests edited to accommodate these changes — recorded under the test-editing rule

Three existing tests asserted that this work was *missing*. They now assert that it
landed. Recording each, with why editing was the right move rather than a way around a
failure:

* `tests/test_preregistered_arms.py` asserted `implemented is False` and a
  `NotImplementedError` for all 24 reserved names. Sixteen of them now have builders.
  The reservation contract — the name resolves, reports unimplemented, and raises a
  message longer than 120 characters saying what to build — is still asserted in full
  over the eight comparison controllers that remain reserved for the baseline track,
  and the sixteen that landed are asserted to have the opposite properties under a new
  parametrised case. The alternative was to keep asserting that implemented work is
  absent.
* `tests/test_experiments.py::test_the_ablation_arms` is an inventory of arms that
  build. It grew from 17 entries to 33, and its learning-arm list from 10 to 25. An
  inventory that does not list new arms is not an inventory; it still fails the moment
  an arm is added or changed without being declared.
* `Arm.implemented` itself was narrowed from "no reserved policy **and** `mechanism ==
  "auto"` **and** no comfort barrier **and** no degradation" to "no reserved policy".
  The three extra conditions existed only to mark this track's work as absent.

**Evidence.** `tests/test_constraint_mechanisms.py`, 40 cases, covering §§3 and 4; plus
`tests/test_preregistered_arms.py` (48) and `tests/test_experiments.py` (42) as edited.

---

## Test floor after the constraints track

`python -m pytest tests/ -q` on the final commit: **452 passed, 15 skipped, 1 warning,
0 failed** (141.37 s), against the entry floor of 404 passed / 15 skipped / 0 failed.
The 15 skips are the same missing-EV-dataset skips as on entry; the warning is the
pre-existing `requires_grad` one in `tests/test_experiments.py`. Net **+48 passing, 0
failing, nothing newly skipped**. The new cases are
`tests/test_comfort_barrier.py` (22), `tests/test_degradation.py` (18) and
`tests/test_constraint_mechanisms.py` (40, of which five drive the real simulator
end to end); `tests/test_preregistered_arms.py` and `tests/test_experiments.py` were
edited rather than added to, as recorded in step 4.6.

## Not done, and why

* **Nothing was trained.** Per the brief, this track makes the controllers exist, be
  correct and be validated; the mechanism grid is a later phase.
* **Transformer thermal limits** (the other half of audit C1's "do not exist") were not
  implemented. `docs/LITERATURE.md` records transformer loss of life as a reported
  *outcome* in [panagi2026thermal] and a genuine constraint only in
  [botkinlevy2020distributed], where charging is coordinated "under nonlinear
  transformer temperature ratings". A loading-history-dependent thermal limit needs a
  transformer model and a network the CityLearn schema does not contain; the district
  import cap is the only grid-side limit this testbed can support. Stated rather than
  stubbed.
* **No `p > 0` default for the depth-of-discharge term**, and no calendar-ageing rate.
  Both would be unsourced numbers inside a constraint.

---

# Abstraction track — branch `framework/abstraction`

Entry state, confirmed before the first edit: `PYTHONPATH=. XDG_CACHE_HOME=<repo>/.citylearn_cache
python -m pytest tests/ -q -p no:randomly` → **538 passed, 0 failed, 0 skipped**, 294.34 s,
one pre-existing `requires_grad` warning in `stems/mappo.py:258`.

**The invocation matters and is recorded here because it cost a confused half hour.**
Without `PYTHONPATH=.` the same command reports **524 passed, 14 skipped**: the fourteen
are the electric-vehicle and schema-dependent cases in `tests/test_ev_real.py`,
`tests/test_env_widening.py`, `tests/test_ev_schema.py`, `tests/test_fleet.py` and
`tests/test_mpc.py`, which catch an `ImportError` on `stems` and convert it into
`pytest.skip("... dataset unavailable")`. The dataset is present; `stems` was not
importable. A skip whose reason names the data when the cause is the path is a trap,
but fixing it is a test edit outside this track's brief, so it is recorded rather than
changed. **Every test count in this track is taken with `PYTHONPATH=.`.**

## Step 1 — The framework stated in one page

**What.** `docs/FRAMEWORK.md`, new. It states the abstraction the repository already
implements implicitly — a household flexible load is a *store* with a *rate limit* that
must reach a *required level* by a *deadline*, and a neighbourhood is a set of such
loads under one shared *import cap* — defines every symbol once against the name the
code uses, and separates what is measured from what is assumed and what is unsourced.

**Why.** The object was real but distributed: `stems/deadline.py` holds the device,
`stems/fleet.py` holds the cap, and `stems/comfort.py`'s module docstring was the only
place the tuple was written down, for one of the four loads. A framework that has to be
reconstructed from four modules is not a contribution anyone else can use.

**Evidence.** The exactness claim in §3 is measured, not asserted.
`experiments/plant_projection_error.py` (new) drives the real CityLearn simulator and
reports, per plant model, the forward error against the simulator and the inverse
round-trip error of the bisection the projection uses. Output written to
`experiments/diagnostics/plant_projection/projection_error.json`:

| plant model | forward max | inverse max | samples |
|---|---|---|---|
| `BatteryModel` | 2.976e-08 soc | 5.341e-13 soc | 938 steps (70.2% unsaturated) |
| `TankModel` | 3.273e-07 kWh/step | 9.490e-13 soc | 1912 |
| `EVFleetModel` | 9.776e-05 soc (7.362e-03 kW on the draw) | 6.545e-07 kW | 981 |

Three findings came out of measuring rather than quoting.

1. The battery number is over **unsaturated steps only** — 70.2% of the rollout. On the
   other 29.8% the device clips at its depth-of-discharge floor or at `soc = 0.95` and
   the model does not agree with the plant; it is conservative in the protected
   direction instead, which is the property the projection needs and which
   `tests/test_battery_model.py::test_model_errs_toward_the_bound_being_protected`
   already pins. Reporting one error figure without that split would overstate the
   model.
2. The EV charger's inverse initially measured a **1.038 kW** maximum miss. That is not
   solver error: `p_min = 1.4 kW` on this schema makes the attainable draw set
   `{0} u [p_min, p_max]`, a non-convex set, so any target inside the dead band is
   unattainable by construction. Split apart, the inverse is exact to 6.545e-07 kW on
   attainable targets and the dead-band miss is reported separately
   (`dead_band_miss_kw`, max 1.038 kW over 1996 unattainable targets).
   `stems.fleet.apply_dead_band` is what resolves those in the shield.
3. **`DeadlineStorageBarrier` does not use any of these exact inverses.** Its forced
   action comes from `action_for_soc_gain`, the linear map `gap / rate * action_bound`,
   and `EVReadinessBarrier` compensates with `rate_derate = 0.85`. The exact inverses
   are used by `stems/fleet.py` (the cap LP) and `stems/mpc.py`. This is recorded as a
   dispute with the brief under step 2, not silently fixed.

**Not claimed.** `docs/FRAMEWORK.md` §6 forward-references `stems/protocols.py`,
`stems/replay.py` and `stems/legionella.py`, which land in steps 3 and 4 of this track.

## Step 2 — The abstraction made explicit in code

**What.** Three new modules and one deleted code path.

* `stems/protocols.py` — `PlantModel`, `Environment`, `PlantProvider`, `EVProvider`,
  plus `missing_capabilities(adapter)`, which names the attribute an adapter is missing
  rather than only answering yes or no.
* `stems/flexibility.py` — `FlexibleLoad` (one store, its rate limit, its required
  level, its deadline) and `FlexibilityPortfolio` (a set of them under one cap).
* `stems/deadline.py` — `DeadlineStorageBarrier` gained an optional `plant=` and the
  properties `exact_projection` / `plant_inputs`. **Default `None`, i.e. the historical
  linear path, unchanged.**
* `stems/cbf.py` — `_apply_shared_power_allocation` (42 lines) deleted;
  `_apply_deadline_barriers` and `feasibility_report` now delegate to
  `CBFShield.portfolio`, a `FlexibilityPortfolio` built per call from the public
  `deadline_barriers` list. Per call, not cached: that list is a public attribute and
  tests append to it after construction (`tests/test_flexibility.py::
  test_appending_a_barrier_after_construction_still_reaches_the_shield`), so a cached
  portfolio would enforce a stale set — the failure mode a safety layer least wants.
  *(Correction, same session: an earlier draft of this entry said
  `experiments/controllers.py` also appends after construction. It does not — it passes
  `deadline_barriers=comfort or None` as a constructor argument at lines 740 and 774
  and never mutates the list. The design decision stands on the attribute being public
  and the tests exercising it; the production-code claim was wrong and is withdrawn.)*
* `stems/__init__.py` — exports the framework and declares `FRAMEWORK_EXPORTS`.

**Why.** The four loads were four construction sites: `EVReadinessBarrier` built in
`experiments/controllers.py`, `DHWReadinessBarrier` in `stems/thermal.py`,
`ThermalComfortBarrier` in `stems/comfort.py`, and the house battery not a load at all
— a pair of state-of-charge bounds inside `CBFShield.project` and a shedding branch
inside `FleetShield`. The knowledge that *a set of deadline loads contends for one cap*
lived in a private method of the battery shield, so a fifth load could not join the set
without editing that shield. It now lives in the portfolio, which is why the Legionella
cycle (step 4) joins without `stems/cbf.py` changing again.

The house battery is the framework's *degenerate* member, not an exception to it: a
band and no deadline (`barrier=None`, `active` nowhere). `FlexibleLoad` refuses a load
with neither face, because a load that constrains nothing is not a load.

### 2.1 Behaviour did not change, and here is the number

`experiments/refactor_kpi_check.py` (new) runs three deterministic rollouts on the real
electric-vehicle schema — `idle+calibrated` (the shield alone), `rbc+calibrated` (the
rule plus the LP cap shield) and `rl+calibrated` (the untrained agent, every projection
live) — for 167 steps at seed 0, and writes every key-performance indicator the
repository computes plus the per-column sums of the executed action stream.

Determinism was established first, because a zero difference across a refactor means
nothing if the rollout is not reproducible: two runs at the same commit,
`before.json` against `before_repeat.json`, gave **0 differing fields, largest absolute
difference 0.000e+00** over 167 steps × 3 arms.

Across the refactor, `before.json` against `after.json`:

> **0 differing field(s), largest absolute difference 0.000e+00.**

Bit for bit, including `barrier_intervention_rate` (0.660180 / 0.205838 / 0.974551 for
the three arms), `avg_daily_peak` (55.143626 / 55.999106 / 55.375294 kW) and the
executed-action column sums. The comparison is exact (`--tol 0`), not approximate.

### 2.2 Disputed: "BatteryModel, TankModel and EVFleetModel already provide three
exact projections" is true of the plant models and **not** of the barrier

The brief says each instantiation needs a plant model with an exact projection and that
three already exist. The three exact inverses exist and are measured in step 1. They
are **not what the deadline barrier uses.** `DeadlineStorageBarrier.project` computes
its forced action from `action_for_soc_gain`, the linear map `gap / rate *
action_bound`; `EVReadinessBarrier` derates `rate` by `rate_derate = 0.85` precisely
because that map is not the plant. The exact inverses are called by `stems/fleet.py`
(the cap LP, via `BatteryModel._pow_x/_pow_y` as piecewise-linear constraints and
`action_for_draw` to realise a granted kW) and by `stems/mpc.py`.

So the instruction describes a property the framework *should* have and does not yet.
What landed is the seam, not the switch: `DeadlineStorageBarrier(plant=...)` makes the
forced action `plant.action_for_soc(soc, soc + gap)`, and `exact_projection` reports
which path a barrier is on. It is **off in every shipped barrier**, asserted by
`tests/test_flexibility.py::test_switching_the_plant_in_is_off_in_every_shipped_barrier`.

Turning it on would change trajectories — the linear map and the exact inverse disagree
whenever `rate` over- or under-states the plant, which is always — and this step's
contract was that behaviour must not change. Switching it on is a measured comparison
(does exactness reduce the deadline-miss rate, and at what cost in forced energy?), and
that belongs in the experiment phase, not in a refactor.

### 2.3 A test was edited, recorded under the test-editing rule

`tests/test_controller_reachability.py::_exported_controllers` subtracted a hand-kept
`NOT_CONTROLLERS` set from `stems.__all__` and asserted the remainder are controllers
with an arm. Eight framework names are now exported and none is a controller. Rather
than grow the hard-coded set, the helper now also subtracts `stems.FRAMEWORK_EXPORTS`,
which the package declares itself — so a future framework export does not require
editing a test. The assertion it protects (every exported *controller* is reachable
from `ARMS`) is unchanged and still fails if a controller is exported without an arm.

**Evidence.** `tests/test_flexibility.py`, 20 new cases: the two faces, the refusals
(`a load with neither face`, `a barrier projecting a different column`, `duplicate
names`), the exact-inverse seam and its default-off state, joint infeasibility that no
device sees alone, and the shield-delegation equivalence across all three coordination
rules including the append-after-construction case.

## Step 3 — The swappable environment adapter

**What.** `stems/protocols.py` (landed with step 2's commit) defines the seam, and
`stems/replay.py::CSVReplayEnvironment` is the second implementation.
`experiments/replay_roundtrip.py` demonstrates it.

The interface is deliberately **split into three protocols rather than one**:

* `Environment` — step, reset, the observation layout, the executed actions. Every
  real-building adapter must implement this.
* `PlantProvider` — the plant models for this site and the action columns they own.
  An adapter implements it when the plant has been characterised; without it the
  shield cannot be exact and must not pretend to be.
* `EVProvider` — the charger fleet. Optional by construction: a building with no
  vehicles is not a degenerate case to special-case, it simply does not implement it.

An interface a real adapter cannot satisfy is an interface nobody implements. A metered
house with no battery model can serve `Environment` today; `missing_capabilities()`
then names the attributes it lacks rather than failing at the first attribute error
four thousand steps into a run.

**Why the adapter refuses more than it accepts.** `CSVReplayEnvironment` validates at
construction and raises `ReplayManifestError` on: a missing observation column (it does
**not** zero-fill — the shield reads observations by position and zero is a legal value
for every one of them, so a fill is silent corruption, not a missing value); a
non-ISO-8601 or non-increasing timestamp; a step length disagreeing with `dt_hours`; a
state of charge outside [0, 1], an hour outside 1..24 or a temperature outside
[-60, 60] degC (the unit errors that are actually detectable); buildings of different
lengths; and a request for a plant model with no plant block in the manifest.

**What it does not do, stated in the module docstring and asserted in tests.** It is
*open loop*: recorded observations do not respond to the action, so every closed-loop
quantity is counterfactual and the adapter cannot produce it. `step` returns a reward
of exactly 0.0 rather than a plausible-looking number, `info["open_loop"]` is True at
every step, and `executed_actions` returns **zeros** rather than echoing the command —
echoing it would be a lie that every metric in `stems/metrics.py` would believe.

**Evidence.** `experiments/replay_roundtrip.py` exports a 72-step CityLearn rollout to
the logged-data format, reads it back through the adapter, and projects the same
`CBFShield` and `FlexibilityPortfolio` over both. Over 71 steps × 8 buildings:

| quantity | max absolute difference |
|---|---|
| observation vector, simulator vs replay | 0.000e+00 |
| projected action, adapter's assumed flat plant curves | 3.155e-02 |
| projected action, simulator's own plant curves | 0.000e+00 |

The third row is the one worth having. The interface costs nothing; the 3.155e-02 is
**entirely** the adapter's assumption that a logged export's battery efficiency and
capacity-power curves are flat, because a manifest does not carry measured ones. That
isolates how much of the shield's guarantee rests on characterising the site's devices
rather than on plumbing its data — and it says that an unfitted plant model is the
binding sim-to-real risk here, not the interface.

Plus `tests/test_replay_env.py`, 13 cases: protocol conformance for both the adapter
and a plant-less site, the canonical observation layout, a shield running on logged
data with no CityLearn import, and seven refusals.

## Step 4 — The Legionella cycle as the fourth instantiation

**What.** `stems/legionella.py`: `LegionellaSpec` (every parameter, with provenance),
`ShadowTank` (the power-limited store), `LegionellaCycleBarrier` (a
`DeadlineStorageBarrier` with a recurring window) and `LegionellaStack` (the three
assembled into a `FlexibleLoad`). Plus `experiments/legionella_demo.py` and
`tests/test_legionella.py` (26 cases).

No new enforcement path was added. The cycle is the same object as the vehicle
deadline with a different requirement function: the required level is a temperature,
the deadline recurs, and the window resets when the level is *reached* rather than
when a vehicle leaves. `tests/test_legionella.py::
test_the_cycle_joins_the_same_portfolio_as_the_other_loads` runs it through the same
`FlexibilityPortfolio` as a house battery and asserts it is enforced there.

**Why the power-limited source came with it.** CityLearn's hot-water device serves
every draw in the hour it occurs and is sized so it always can, so an empty tank costs
nothing and the readiness state of charge is a proxy with no physical consequence.
`ShadowTank` re-books the same energy against a source limited to `P_hp`, which makes
**unmet hot water** a real service failure. It is driven *alongside* the simulator, not
inside `STEMSEnvironment.step`: putting it inside would change the energy accounting of
every existing run, which is an experiment-phase decision with a before/after burden.

### 4.1 Not one unsourced number was invented

The design note's three open questions — realistic `P_hp`, the tank temperatures, and
the coefficient of performance at a 60 degC sink — have no sourced answers in this
repository. **No field of `LegionellaSpec` covering them has a default**, which
`tests/test_legionella.py::test_the_spec_has_no_default_for_any_open_question` enforces
by reflection over the dataclass, so a default cannot be added later without the test
failing. `LegionellaSpec.unsourced` lists at runtime which parameters still carry no
provenance, and `FlexibleLoad.describe()` carries that list into the load's own
description so a run record cannot lose it.

The single defaulted interval is `period_hours = 168`. The design note states the rule
as "once a week ... in new units the interval is a user setting, 7–10 days", so 168 h
is the shortest of the stated range and therefore the conservative choice.

Two assumptions are stated rather than hidden: `soc` is linear in store temperature
(a **fully mixed** tank — a real cylinder stratifies, so the energy needed to
disinfect the whole volume is understated when a draw has left a cold bottom layer),
and the coefficient of performance *steps* at the heat pump's temperature ceiling
rather than falling continuously with sink temperature. The step is the coarsest model
that distinguishes the design note's two unit generations.

### 4.2 Instantiating the load found a limitation of the abstraction

`DeadlineStorageBarrier` forces when `steps_to_deadline` falls to
`steps_needed = ceil(gap / rate)` — a **just-in-time schedule with no recourse**. Three
of the four loads tolerate that. The hot-water store does not, because unlike a vehicle
(which is not driven while plugged in) it is drawn from *while it is being charged*.

`experiments/legionella_demo.py`, 336 steps × 8 buildings (16 building-windows), the
schema's own hot-water series, a set-point thermostat and no policy:

| source size | draw reserve | cycles completed | windows missed | unmet kWh |
|---|---|---|---|---|
| 0.25 × sizing rule | median draw | 0 / 16 | 16 | 4.173 |
| 0.5 × | median draw | 0 / 16 | 16 | 0.000 |
| 1.0 × | median draw | 1 / 16 | 15 | 0.000 |
| 2.0 × | median draw | **0 / 16** | 16 | 0.000 |
| 1.0 × | 90th percentile of own draw | **10 / 16** | 6 | 0.000 |
| 1.0 × | maximum draw | refused by `LegionellaSpec` | — | — |

A **larger source completing fewer cycles** looks like a bug, so the mechanism was
traced before being reported. It is not a bug. A larger source raises `rate`, so
`steps_needed` falls — to **1** for six of the eight buildings at twice the sizing
rule — and the entire cycle must then succeed inside one hour. Measured on this
schema, the **median hourly hot-water draw is 0.0 kWh** (draws are bursty) while the
maximum is 3.122 to 6.302 kWh per building, so a median-quantile reserve is no reserve
at all and one burst in the single forced hour defeats the deadline. The lever is the
reserve, not the source: 1 → 10 of 16 at unchanged source size.

`draw_margin_kwh` is the declared reserve, **zero by default**, and
`LegionellaCycleBarrier.current_rate` deducts it and the standing loss from the rate.
Zero is the honest default: nothing in this repository forecasts hot-water demand
during the disinfection hours, and zero says plainly that the deadline is guaranteed
only against a zero draw. Both sides are pinned —
`test_the_barrier_drives_the_cycle_to_completion_over_a_week` with a reserve, and
`test_without_a_reserve_the_concurrent_draw_can_make_the_cycle_miss` without one, on
the identical draw sequence. Reserving the *maximum* draw is refused at construction,
because for at least one building it exceeds everything the source delivers in a step
and the constraint would be unsatisfiable by construction.

### 4.3 Two negative results about this testbed

* **Unmet hot water is zero at the design note's sizing rule**, and at every multiple
  above it. It appears only at a quarter of the rule (4.173 kWh over 336 steps × 8
  buildings). With the schema's own tank capacities the power limit does not bind, so
  the service-failure KPI the design note wants exists, is implemented, and measures
  zero here. A result that needs it to bind will need a sourced `P_hp` that is smaller
  than this rule, or a different building stock.
* **A just-in-time deadline does not guarantee a deadline under disturbance.** That is
  a property of the framework, not of a controller, and no controller was run.

### 4.4 The novelty claim, worded as the evidence supports

`docs/LITERATURE.md` was read, not edited. Its verdict is adopted verbatim in
`stems/legionella.py`:

* [engelbrecht2021optimal] is **confirmed** — a field study of 77 water heaters,
  median savings "6.3% for temperature-matching, 21.9% for energy-matching and 16.2%
  for energy-matching with Legionella prevention". But its method is **A\* search
  optimal control, not reinforcement learning**, and the abstract supports neither
  "daily" nor "as a temperature constraint". Both qualifiers are dropped.
* "Reyes Premer et al. (2025) use a soft penalty" is **struck**. That paper is model
  predictive control for a 120 V heat-pump water heater and does not mention
  Legionella at all.
* The claim is therefore written as: *no reinforcement-learning treatment of a weekly
  Legionella cycle as a deadline constraint was found. That is an absence of evidence
  from one search, not a proven absence.* **Plausible and not established.** The module
  docstring says exactly this, so the wording travels with the code.
