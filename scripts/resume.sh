#!/usr/bin/env bash
# Rebuild this repository's working state on a fresh machine, then verify it.
#
#   bash scripts/resume.sh
#
# Idempotent: safe to re-run. It never touches tracked source, never deletes
# results, and never runs setup_citylearn.sh (whose fifth line is an rm -rf on
# third_party/CityLearn).
#
# What is in git and needs nothing from you:
#   all source, tests and docs; the 592 run records and 213 training logs under
#   results/; the trained policy checkpoints; docs/sources/ with the tariff PDF.
#
# What is NOT in git and this script rebuilds:
#   the Python environment; the CityLearn dataset cache (~155 MB, downloaded);
#   citylearn_schemas/*/schema.json, which are gitignored because they embed
#   absolute paths to that cache and are therefore machine-specific.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${STEMS_VENV:-$ROOT/.venv/stems}"
CACHE="$ROOT/.citylearn_cache"
PY_WANT="3.12"
TORCH_WHEEL_INDEX="https://download.pytorch.org/whl/cpu"

say() { printf '\n\033[1m[resume]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[resume] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. interpreter
say "1/6  Python interpreter"
if [[ -x "$VENV/bin/python" ]]; then
  echo "      reusing $VENV"
else
  BASE=""
  for cand in "python$PY_WANT" python3.12 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      v="$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
      if [[ "$v" == "$PY_WANT" ]]; then BASE="$cand"; break; fi
      [[ -z "$BASE" ]] && BASE="$cand"
    fi
  done
  [[ -n "$BASE" ]] || die "no python3 on PATH; install Python $PY_WANT"
  got="$("$BASE" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
  if [[ "$got" != "$PY_WANT" ]]; then
    echo "      WARNING: results were produced on Python $PY_WANT, found $got."
    echo "      Continuing, but pin $PY_WANT if anything below fails."
  fi
  echo "      creating $VENV with $BASE ($got)"
  "$BASE" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip

# --------------------------------------------------------------- 2. dependencies
say "2/6  Dependencies"
if "$PY" -c 'import citylearn, torch, cvxpy' >/dev/null 2>&1; then
  echo "      already satisfied"
else
  # torch first: CPU wheel from a separate index.
  echo "      torch (CPU wheel)"
  "$PY" -m pip install --quiet "torch==2.12.1+cpu" --index-url "$TORCH_WHEEL_INDEX"
  echo "      pinned stack"
  "$PY" -m pip install --quiet -r requirements-lock.txt
  # citylearn last and with --no-deps: its own pins would downgrade numpy/pandas
  # and break everything installed above.
  echo "      citylearn 2.6.0b1 (--no-deps, deliberately)"
  "$PY" -m pip install --quiet --no-deps "citylearn==2.6.0b1"
fi
"$PY" - <<'PYCHK'
import importlib, sys
need = {"citylearn": "2.6.0b1", "numpy": "1.26.4", "pandas": "3.0.3",
        "cvxpy": "1.7.5", "gymnasium": "0.28.1"}
bad = []
for mod, want in need.items():
    try:
        got = getattr(importlib.import_module(mod), "__version__", "?")
    except Exception as exc:
        bad.append(f"{mod}: import failed ({exc})"); continue
    if got != want:
        bad.append(f"{mod}: have {got}, results were produced on {want}")
import torch
print(f"      torch {torch.__version__}, threads {torch.get_num_threads()}")
if bad:
    print("      VERSION DRIFT:"); [print("        " + b) for b in bad]
else:
    print("      versions match the lock file")
PYCHK

# -------------------------------------------------------------------- 3. the cache
say "3/6  CityLearn dataset cache"
mkdir -p "$CACHE"
cat > scripts/env.sh <<EOF
# Source this before any CityLearn call in this repo:  source scripts/env.sh
# CityLearn resolves its dataset directory through platformdirs, which reads
# XDG_CACHE_HOME. Without this the cache lands in \$HOME and the generated
# schemas, which hold absolute paths, stop resolving.
export XDG_CACHE_HOME="$CACHE"
export PYTHONPATH="$ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export PATH="$VENV/bin:\$PATH"
export OMP_NUM_THREADS=1
EOF
echo "      wrote scripts/env.sh (XDG_CACHE_HOME -> $CACHE)"
export XDG_CACHE_HOME="$CACHE"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

"$PY" - <<'PYDL'
import os, sys
from pathlib import Path
from citylearn.data import DataSet
root = Path(os.environ["XDG_CACHE_HOME"]) / "citylearn"
need = ["tx_travis_county_neighborhood", "citylearn_challenge_2022_phase_all",
        "citylearn_challenge_2022_phase_all_plus_evs",
        "citylearn_challenge_2020_climate_zone_1"]
ds = DataSet()
for name in need:
    hits = list(root.glob(f"*/datasets/{name}"))
    if hits:
        print(f"      present  {name}")
        continue
    print(f"      download {name} ...", flush=True)
    try:
        ds.get_dataset(name)
        print(f"      ok       {name}")
    except Exception as exc:
        print(f"      FAILED   {name}: {exc!r}")
        print("               (GitHub rate-limits anonymous API calls; retry later.)")
PYDL

# ------------------------------------------------------------------- 4. schemas
say "4/6  Regenerate the machine-specific schemas"
run_setup() {
  local label="$1"; shift
  if "$PY" "$@" >/tmp/stems_setup_$$.log 2>&1; then
    echo "      ok     $label"
  else
    echo "      FAILED $label — last lines:"
    tail -6 /tmp/stems_setup_$$.log | sed 's/^/             /'
  fi
  rm -f /tmp/stems_setup_$$.log
}
run_setup "tx_travis_8b"    setup_citylearn_8b.py
run_setup "tx_travis_8b_ev" setup_citylearn_ev.py
run_setup "cl2020_zone1..4" setup_citylearn_mixed.py
echo "      NOTE setup_citylearn.sh is NOT run here: it begins with rm -rf on"
echo "           third_party/CityLearn. The pip package is byte-identical to"
echo "           that checkout across all 19 top-level modules, so it is not needed."

# ------------------------------------------- 5. the EV charger CSVs are reproducible
say "5/6  Verify the generated EV charger CSVs still match the committed ones"
if git diff --quiet -- 'citylearn_schemas/tx_travis_8b_ev/charger_*.csv' 2>/dev/null; then
  echo "      ok, byte-identical (seed 17 reproduces them)"
else
  echo "      DIFFER from the committed copies. Check whether it is only line"
  echo "      endings (CRLF drift is not a data change):"
  echo "        git diff --stat -- 'citylearn_schemas/tx_travis_8b_ev/charger_*.csv'"
fi

# --------------------------------------------------------------------- 6. verify
say "6/6  Test suite"
echo "      Last confirmed green: 619 passed, 0 failed, 0 skipped at commit f3bd5df."
echo "      Commit 505de3a changed the cap defaults (300/80 -> 33.5/11.5 kW) and"
echo "      only 113 tests were re-run afterwards, so THIS RUN IS THE REAL CHECK."
set +e
"$PY" -m pytest tests -q -p no:cacheprovider --tb=line -rf
RC=$?
set -e

cat <<EOF

$(printf '=%.0s' {1..74})
Resume complete (pytest exit $RC).

Before every CityLearn call in a new shell:
    source scripts/env.sh

Where the work stopped, and what is next, is in RESUME.md. The immediate step:

    python -m experiments.ablation --help     # see the grid runner's options

E1 is the gate: train [0,4379] / test [4380,8759], 5 seeds, same arms as
results/ws_paper/, now under the binding 33.5 kW cap. Roughly 4-6 h on 20 cores.

If any test failed above, fix that before running E1 — the cap change is the
most likely cause and it is a one-line default in experiments/scenario.py.
$(printf '=%.0s' {1..74})
EOF
exit $RC
