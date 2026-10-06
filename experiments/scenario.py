from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TX_SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
SUBSET_DIR = REPO / "citylearn_schemas" / "_subsets"

SEASON_FIRST_DAY = {"winter": 0, "spring": 90, "summer": 181, "autumn": 273}


def day_window(first_day: int, days: int) -> Tuple[int, int]:
    start = first_day * 24
    return start, start + days * 24 - 1


YEAR = "year"

#: Train on the first half of the year, evaluate on the second. This is the honest
#: counterpart to ``YEAR``, which returns the SAME window for both phases and is
#: therefore in-sample: audit finding A1 records that all 23 records in
#: results/ws_paper/ have ``train_window == eval_window == [0, 8759]``, so every
#: learning number there is a training-set score. Any claim about generalisation must
#: use this split, or the held-out building subsets, or a different weather year.
#:
#: The split is by time, never at random, because the state is autocorrelated and a
#: random split would leak neighbouring hours across the boundary. The two halves are
#: also seasonally different for Travis County -- the first is winter and spring, the
#: second summer and autumn -- so this is a deliberately hard split and the gap it
#: shows is an upper bound on what a same-season split would give.
YEAR_SPLIT = "year-split"
YEAR_STEPS = 8760
SEASONS = sorted(SEASON_FIRST_DAY) + [YEAR, YEAR_SPLIT]


def season_windows(season: str, days: int = 28) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if season == YEAR:
        return (0, YEAR_STEPS - 1), (0, YEAR_STEPS - 1)
    if season == YEAR_SPLIT:
        half = YEAR_STEPS // 2
        return (0, half - 1), (half, YEAR_STEPS - 1)
    if season not in SEASON_FIRST_DAY:
        raise ValueError(f"unknown season {season!r}; choose from {sorted(SEASON_FIRST_DAY)}")
    first = SEASON_FIRST_DAY[season]
    return day_window(first, days), day_window(first + days, days)


def _resolve(schema: str) -> Path:
    p = Path(schema)
    return p if p.is_absolute() else REPO / p


def schema_buildings(schema: str = TX_SCHEMA) -> Tuple[List[str], List[str]]:
    buildings = json.loads(_resolve(schema).read_text(encoding="utf-8"))["buildings"]
    return list(buildings), [k for k, v in buildings.items() if v.get("include", True)]


REFERENCE_SHARED_FIELDS = ("pricing", "carbon_intensity")


def device_signature(building: Dict[str, Any]) -> Tuple:
    present = tuple(sorted(k for k, v in building.items()
                           if isinstance(v, dict) and "type" in v and v["type"] is not None))
    return (present, tuple(sorted(building.get("inactive_actions") or [])),
            tuple(sorted(building.get("inactive_observations") or [])))


def _reference(data: Dict[str, Any], schema: str) -> Tuple[Tuple, Dict[str, Any]]:
    included = [v for v in data["buildings"].values() if v.get("include", True)]
    signatures = {device_signature(v) for v in included}
    if len(signatures) != 1:
        raise RuntimeError(f"{schema}: included buildings have {len(signatures)} different "
                           "device sets, so there is no single reference to sample against")
    shared = {}
    for field in REFERENCE_SHARED_FIELDS:
        values = {json.dumps(v.get(field)) for v in included}
        if len(values) != 1:
            raise RuntimeError(f"{schema}: included buildings disagree on {field!r}")
        shared[field] = included[0].get(field)
    return signatures.pop(), shared


def comparable_candidates(schema: str = TX_SCHEMA) -> List[str]:
    data = json.loads(_resolve(schema).read_text(encoding="utf-8"))
    signature, _ = _reference(data, schema)
    return [k for k, v in data["buildings"].items() if device_signature(v) == signature]


def sample_buildings(n: int, subset_seed: int, schema: str = TX_SCHEMA) -> List[str]:
    candidates = comparable_candidates(schema)
    if n > len(candidates):
        raise ValueError(f"asked for {n} buildings but {schema} has only "
                         f"{len(candidates)} with the reference device set")
    rng = np.random.default_rng(subset_seed)
    chosen = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    return [candidates[i] for i in chosen]


def materialize_subset_schema(schema: str, buildings: List[str], tag: str) -> str:
    src = _resolve(schema)
    data = json.loads(src.read_text(encoding="utf-8"))
    unknown = sorted(set(buildings) - set(data["buildings"]))
    if unknown:
        raise ValueError(f"buildings not in schema {schema}: {unknown}")
    root = data.get("root_directory")
    if not root or not Path(root).is_absolute():
        raise RuntimeError(
            f"schema {schema} has root_directory={root!r}; a copy stored elsewhere "
            "would not find its data files")
    signature, shared = _reference(data, schema)
    mismatched = sorted(k for k in buildings
                        if device_signature(data["buildings"][k]) != signature)
    if mismatched:
        raise ValueError(f"buildings without the reference device set: {mismatched}")
    for key, value in data["buildings"].items():
        value["include"] = key in buildings
        if value["include"]:
            value.update(shared)
    out = SUBSET_DIR / f"{src.parent.name}__{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2)
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.write_text(text, encoding="utf-8")
    return out.relative_to(REPO).as_posix()


@dataclass
class Scenario:
    schema: str = TX_SCHEMA
    season: str = "winter"
    subset_seed: Optional[int] = None
    n_buildings: int = 8
    days: int = 28
    # Caps sized to BIND. The previous defaults (300 kW district, 80 kW per building)
    # could not: a full-year no-control rollout of the eight-building reference stock
    # peaks at 41.8822 kW district and 14.3167 kW for the heaviest building, so the
    # district cap sat at 7.2x the peak it was meant to limit and was exceeded in 0 of
    # 8759 hours. Under a cap that cannot be reached, a constrained arm is
    # indistinguishable from an unconstrained one and every violation-rate column is
    # structurally zero -- which is what results/ws_paper/ reported.
    #
    # Both defaults are now 80% of the measured uncontrolled peak, rounded down to
    # 0.5 kW: 0.8 * 41.8822 = 33.5058 -> 33.5, and 0.8 * 14.3167 = 11.4534 -> 11.5.
    # The 80% fraction is a design choice, not a sourced engineering limit; the peaks it
    # multiplies are measurements, reproduced by
    #     python -m experiments.household_case --season year --arms idle
    # which writes them to experiments/diagnostics/household_case/case.json.
    #
    # These are the reference-stock values. A different building subset has a different
    # peak, so E2's held-out subsets must re-measure rather than inherit these.
    grid_cap_kw: float = 33.5
    building_cap_kw: float = 11.5
    hvac_control: str = "setpoint"
    heat_pump: bool = True
    allow_missing_obs: bool = False
    #: Price of one kWh of lost *storage capacity*, same currency per kWh as the
    #: tariff, so the degradation cost and the bill are additive. 0.0 means fade is
    #: measured and reported but does not enter the reward: a replacement cost is a
    #: market number and does not belong in the code. See `stems/degradation.py`.
    degradation_price_per_kwh: float = 0.0
    #: Per-building capacity-loss budget for one episode, kWh. ``None`` leaves
    #: degradation reported and unconstrained, which is where the literature stands.
    degradation_limit_kwh_per_episode: Optional[float] = None
    #: Depth-of-discharge stress exponent ``p``. 0.0 reduces the depth term to the
    #: throughput term exactly; no non-zero value is asserted here because none was
    #: verified against a source.
    degradation_dod_exponent: float = 0.0

    @property
    def buildings(self) -> Optional[List[str]]:
        if self.subset_seed is None:
            return None
        return sample_buildings(self.n_buildings, self.subset_seed, self.schema)

    def schema_path(self) -> str:
        if self.subset_seed is None:
            return self.schema
        return materialize_subset_schema(
            self.schema, self.buildings, f"subset{self.subset_seed}_n{self.n_buildings}")

    def env_kwargs(self, phase: str) -> Dict[str, Any]:
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        train, evaluation = season_windows(self.season, self.days)
        start, end = train if phase == "train" else evaluation
        return {"episode_time_steps": [(start, end)]}

    @property
    def key(self) -> str:
        subset = "ref" if self.subset_seed is None else f"subset{self.subset_seed}"
        if self.season == YEAR:
            span = "year-insample"
        elif self.season == YEAR_SPLIT:
            span = "year-split"
        else:
            span = f"{self.season}{self.days}d"
        key = (f"{_resolve(self.schema).parent.name}__{span}"
               f"__{subset}n{self.n_buildings}"
               f"__cap{self.grid_cap_kw:g}-{self.building_cap_kw:g}")
        return key if self.hvac_control == "setpoint" else f"{key}__{self.hvac_control}"

    def describe(self) -> Dict[str, Any]:
        train, evaluation = season_windows(self.season, self.days)
        d = asdict(self)
        d.update(key=self.key,
                 buildings=self.buildings or schema_buildings(self.schema)[1],
                 train_window=list(train), eval_window=list(evaluation))
        return d
