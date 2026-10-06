# Resume here

State of branch `audit/2026-10` as of commit `505de3a`. Written at a workstation
change, so it assumes nothing about the machine you are reading it on.

## Start

```bash
bash scripts/resume.sh        # rebuilds the environment, cache and schemas, then runs the tests
source scripts/env.sh         # in every new shell, before any CityLearn call
```

`scripts/resume.sh` is idempotent and never deletes results. It does **not** run
`setup_citylearn.sh`, whose fifth line is an `rm -rf` on `third_party/CityLearn`;
the PyPI package `citylearn==2.6.0b1` is byte-identical to that checkout across
all 19 top-level modules, so the vendored copy is not needed.

## The one thing to check first

The test suite was last confirmed green at **619 passed, 0 failed, 0 skipped** at
commit `f3bd5df`. Commit `505de3a` then changed the district and per-building cap
defaults, and only 113 tests were re-run (`test_preregistered_arms` 77,
`test_household` 36). **The remaining ~500 are unverified against the new caps.**
`scripts/resume.sh` runs the whole suite as its last step; treat a failure there
as the cap change before suspecting the new machine.

If some tests skip rather than fail, that is expected: about 14 convert a GitHub
API 403 into a skip, because `_resolve_schema` calls `DataSet().get_dataset_names()`
and anonymous GitHub API calls are rate-limited per IP. They pass once the limit
clears. This is recorded and deliberately not fixed.

## Done

| phase | result |
|---|---|
| Audit | `docs/AUDIT_2026-10-06.md`, 591 lines. Three of its own findings were later overturned by measurement and the corrections are in `CHANGELOG.md`. |
| Literature | `docs/LITERATURE.md`, `docs/references.bib` (65 entries, 67 sources, tiered 52 abstract / 15 metadata / 1 full text). |
| Correctness | observation indices bound to names, shield and Lagrangian arithmetic, provenance and seeding, batched GCN forward. |
| Baselines | the seven STEMS Table I controllers, an oracle and a receding-horizon MPC, and MAPPO with a centralised critic. All nine comparison arms `ActuatorEvidence verified=True`. |
| Constraints | comfort barrier behind a flag, capacity-fade model, FOCOPS, the mechanism 2x2 wired across 16 pre-registered arms. |
| Framework | `docs/FRAMEWORK.md`, `docs/FRAMEWORK_GUARANTEES.md`, `experiments/diagnostics/guarantee_residuals.py` reproducing every quoted number, `tests/test_guarantees.py` (22 adversarial cases). |
| E0 | `stems/household.py`, `experiments/household_case.py`, `tests/test_household.py` (36 cases). |

Everything is in git, including the 592 run records and 213 training logs under
`results/`, the 20 trained policy checkpoint directories, and the Austin Energy
tariff PDF at `docs/sources/austin_energy_electric_tariff_fy2026.pdf`.

## Three E0 findings that change what the grids should measure

Reproduce all three with:

```bash
source scripts/env.sh
python -m experiments.household_case --season year --arms idle rbc
```

**1. The tariff and the carbon series are Californian; the buildings are Texan.**
The Travis County dataset ships neither, so `setup_citylearn_8b.py` borrows both
from `citylearn_challenge_2022_phase_all`, a neighbourhood in Fontana, California.
Priced under Austin Energy's actual FY2026 residential tariff, the rule-based
arm's annual household saving against no control falls from **$526.02 to $49.10**
(17.90% to 3.50%). The borrowed tariff is 2.4x the real price level and carries
1.6x the peak/off-peak spread. Under the Value-of-Solar rider, which bills *gross*
consumption and credits *gross* photovoltaic output, the same arm **costs the
household $141.02/year**. Austin Energy has no residential net metering and no
residential time-of-use rate outside a pilot capped at 100 meters.

**2. "Energy saved" was import avoided.** The rule-based arm's 395.4
kWh/household/year of avoided import decomposes into 1376.8 kWh of *forgone
export* and 981.4 kWh of *additional gross consumption* — battery round-trip
losses. The identity `import_avoided - export_forgone + gross_increase = 0`
closes to 0.000. The field is now `import_avoided_kwh`, the decomposition is
reported beside it, and a test asserts nothing is called `energy_saved_kwh` again.

**3. The district cap could not bind.** A full-year no-control rollout peaks at
41.8822 kW against the old 300 kW default — 0 of 8759 hours above it. Every
violation-rate column was therefore structurally zero, including the 0.0000
reported for `rl+calibrated` on all five seeds in `results/ws_paper/`. Defaults
are re-pinned in `experiments/scenario.py` to 80% of the measured uncontrolled
peak: `grid_cap_kw` 300 -> **33.5**, `building_cap_kw` 80 -> **11.5**.

## Open decision

The 80% cap fraction is a design choice, not a sourced engineering limit. The
peaks it multiplies are measurements. If the cap should instead come from a real
service-transformer rating for eight US single-family homes, that changes E1's
operating point and is cheaper to settle before the grid runs than after.

## Next, in order

1. **E1, the gate.** Replace the in-sample grid. 23 runs, 4-6 h wall on 20 cores.
   Look at this table before spending compute on anything below — it decides whether
   the stored numbers survive.

   ```bash
   source scripts/env.sh
   python -m experiments.ablation \
       --seasons year-split --subsets ref \
       --arms idle rbc rbc+calibrated rl rl+basic rl+linear rl+calibrated \
       --seeds 0 1 2 3 4 --episodes 15 \
       --workers 10 --out results/e1_split
   ```

   Add `--dry-run` first to list the 23 records without running them. The arms and
   `--episodes 15` match `results/ws_paper/` exactly, so the only things that change
   are the window and the cap, which is what makes the comparison interpretable.
   `--seasons year-split` gives train `[0,4379]` / test `[4380,8759]`; the caps come
   from the `Scenario` defaults (33.5 / 11.5 kW) and need no flag. The scenario key
   written into every record is
   `tx_travis_8b__year-split__refn8__cap33.5-11.5`, which cannot be confused with
   the in-sample `...__year-insample__refn8__cap300-80` of the stored grid.

   `--workers 10` not 20: each run pins itself to one torch thread, but CityLearn
   holds the full year of eight buildings in memory, so 10 keeps headroom. Raise it
   if memory allows.

   Then read the table:

   ```bash
   python -m experiments.aggregate results/e1_split
   ```

   Two things to look at before anything else. Does `rl+calibrated` still report a
   zero violation rate now the cap binds? And how large is the drop from the
   in-sample numbers in `results/ws_paper/` — that gap is the generalisation cost
   that was previously invisible.
2. **E2** held-out buildings, 5 subsets x 5 seeds, disjoint train/test building
   sets. Each subset must **re-measure its own uncontrolled peak** rather than
   inherit 33.5/11.5 kW.
3. **E3** the constraint-mechanism 2x2 ablation, 5 seeds.
4. **E4** the baseline table. Two footnotes are owed: MADCQ's source paper is
   unidentified, and MARLISA's learned consumption regressor was substituted. The
   12 h MPC horizon was validated only under a *non-binding* cap and should be
   re-swept now that the cap binds.
5. **E5** EV and MARL under a binding cap.
6. **E6** robustness and the sim-to-real gap. Phase 3 already showed the shape:
   forecast failure degrades the cap by error *type*, not magnitude — a persistent
   +12 kW bias lifts the breach rate only 0.48% to 1.53%, while scaling the
   innovation by 2/4/8 gives worst residuals of 1.836 / 5.660 / 22.381 kW. Vary
   error structure, not just error size.
7. **E7** framework demonstration runs from a single documented command.
8. **Synthesis**: aggregate, write the WSED framework paper, rewrite
   `docs/REPORT_2026-10.md`, run the failure-mode checks. Then the explainability
   lane (shield attribution, policy-vs-rule surrogate, figures).

## Carry these into anything published

- **Six constraint breaches are unsurfaced**, documented and pinned by nine
  `test_documented_gap_*` tests rather than fixed, per the brief. The sharpest:
  `infeasible_hour_rate` is structurally **0.000 for six of the seven** EV fleet
  rules, because `FleetShield.project` writes `feasible`/`shortfall_kwh` into
  `self.last` only on rule `lp`, and `experiments/ev_coupling.py:135` counts
  `int(shield.last.get('feasible') is False)`, which is 0 when the key is absent.
  No published number is wrong yet — `ev_report.py` tabulates it for `lp` only —
  but `experiments/diagnostics/replay_stored_runs.py:19` lists it for any arm, so
  the first table that puts this metric beside a myopic rule awards a perfect
  score to a rule that cannot fail by construction.
- The myopic rules and `lp` resolve the cap-versus-deadline conflict in **opposite
  directions and neither records the choice**: `apply_dead_band` rounds an urgent
  charger up past the cap, `lp` holds the cap and misses the deadline.
- **All learning-arm numbers in `docs/REPORT_2026-10.md` are superseded** and must
  be re-run before quoting.
- The battery SOC band is genuinely hard: 0 breaches in 52,488 cases. But with the
  uniform 0.1 loss-rate surrogate there are 4,829 breaches (9.20%) — true per-step
  rates are 0.178 to 0.533, so 0.1 understates by 1.8x to 5.3x.
- The district cap is **soft by construction**: its margin is the 95th percentile
  of one-step forecast error, so about 5% exceedance is the design, not a defect.
- Phase 3's district-cap kilowatt figures come from a **synthetic** base-load
  process driven through `FleetShield` in memory, not a CityLearn rollout. Only
  the SOC-band section uses the real eight-building plant.
- **Every Legionella number is provisional**: the four cycle temperatures and the
  COP at the disinfection sink have no sourced value. They are required arguments
  with no defaults, so nothing invents them silently, but a standard (W 551,
  ASHRAE 188) must be cited before any of it is quotable.
- The degradation DoD exponent `p` and the calendar-ageing rate have **no sourced
  value** (Xu et al. 2018 is closed access), so `rl+calibrated+degr-dod` currently
  claims an effect it does not apply.
- Report claims still to strike or reword: the Reyes Premer Legionella sentence
  (unsupported); "daily" and "as a temperature constraint" on the Engelbrecht
  figures (the method is A* search); Gardlo use proportional fairness, not Jain's
  index; the Panagi transformer-aging KPI is a mitigated outcome, not a named KPI;
  EVLearn is 2024, not 2025; Horn 1974 is sole-author, so no "et al."; and the
  Subramanian 2013 citation is ambiguous against their 2012 ACC paper.

## Machine notes

- Git identity must be passed by environment variable — `.git/config` is not
  writable by the agent:
  `GIT_AUTHOR_NAME="stems audit" GIT_AUTHOR_EMAIL="audit@localhost"` plus the
  matching `GIT_COMMITTER_*`.
- The remote is `https://github.com/garyhost2/stems.git`. `audit/2026-10` contains
  every commit from `framework/abstraction`, `framework/guarantees`,
  `methods/baselines` and `methods/constraints` (verified 0 missing from each), so
  pushing that one branch preserves all of it.
- `.citylearn_cache/` is about 155 MB and gitignored. `citylearn_schemas/*/schema.json`
  are gitignored because they embed absolute paths into that cache.
- Disk on the previous machine was at 98% with 22 GB free, which the E1-E7 grids
  will not fit. Give the new one room first.
