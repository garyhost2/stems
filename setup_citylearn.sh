#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/third_party/CityLearn"
rm -rf "$SRC"
mkdir -p "$ROOT/third_party"
git clone --depth 1 --branch v2.6.0b1 https://github.com/intelligent-environments-lab/CityLearn.git "$SRC"
python -m pip install --no-deps -e "$SRC"
python -c "import citylearn; print('CityLearn', citylearn.__version__, 'installed from', citylearn.__file__)"
