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
