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

For `citylearn_schemas/tx_travis_8b/schema.json` this covers 20 files (7.5 MB, about
15 ms) and reports `data_missing: []`.

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

**Evidence.** `tests/test_provenance.py`, 11 cases, all passing:
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
