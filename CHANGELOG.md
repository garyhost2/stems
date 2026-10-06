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
