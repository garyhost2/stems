# STEMS: safe multi-agent energy management for buildings, heat pumps and electric vehicles

Control of a neighbourhood of buildings — battery, solar panels, heat pump, hot-water tank, EV chargers — that
lowers the electricity bill **without breaking limits**: the battery's state-of-charge band, thermal comfort,
each vehicle's departure deadline, and one shared limit on the power the neighbourhood may import.

The learning controller follows the STEMS architecture (a graph convolution and a transformer shared by all
buildings, constrained multi-agent reinforcement learning, a safety shield): Zhang, Wu, Zinflou and Boulet,
*STEMS: Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management*, 2025,
[arXiv:2510.14112](https://arxiv.org/abs/2510.14112). Everything runs on
[CityLearn](https://github.com/intelligent-environments-lab/CityLearn) 2.6.0b1.

The full account of the methods and every table is in [docs/REPORT_2026-10.md](docs/REPORT_2026-10.md);
a two-page plain-language version is in [docs/SUMMARY.md](docs/SUMMARY.md).

## What this repository contributes

1. **An audited simulator.** Three defects of CityLearn ≥ 2.4.0 were found, reproduced on plain CityLearn and
   corrected. The most serious: the indoor-temperature observation is the dataset's uncontrolled value, so no
   controller, reward or comfort metric can see what the heat pump does. Each correction announces itself,
   is recorded in every run, can be switched off and is covered by tests
   ([docs/citylearn_v2.4_defects.md](docs/citylearn_v2.4_defects.md)).
2. **An exact battery barrier.** The barrier inverts the simulator's own battery equations (efficiency that
   depends on power, power that depends on state of charge, standby loss) instead of assuming a linear
   battery.
3. **A shield for coupled constraints.** Several vehicles behind one import cap can each be chargeable in time
   and not be chargeable together. A linear programme plans all vehicles over their remaining parking hours,
   returns the least change to what the controller asked for, and says so when no feasible plan exists. The
   same shield holds the house batteries and hot-water tanks under the cap.
4. **A measurement that matches the simulator.** Cost, emissions, consumption, peak and discomfort equal the
   values recomputed from CityLearn's own series; statistics use the scenario, not the seed, as the unit of
   replication.

## Main results

Eight houses in Travis County, Texas (NREL ResStock, 2018 weather), four seasons, two building sets.

| Controller | Battery outside its band | Cost | Discomfort |
|---|---|---|---|
| no control | 100% | 1951 | 4.9% |
| time-of-use rule | 30% | 1677 | 4.9% |
| rule + exact barrier | 0% | 1665 | 4.9% |
| learner, constraint penalty only | 9.1% | 1666 | 2.2% |
| learner + barrier with a uniform battery rate | 3.9% | 1564 | 2.2% |
| learner + exact barrier | 0% | 1549 | 2.5% |

- **Safety.** The exact barrier gives zero violations in all eight scenarios for 0.3–0.5% of cost. On offices,
  restaurants, shops and multi-family buildings in a hot and a cold climate it is again zero (rule-based
  controllers; the learning runs on those buildings are still to be done).
- **Learning against a good rule.** The learner is 12–41% cheaper than the rule in spring and summer and
  10–25% dearer in winter. Pooled over the eight scenarios the difference is not statistically significant.
- **Electric vehicles under a shared cap.** When vehicles charge on arrival, earliest-deadline and
  least-laxity rules deliver as much as the joint programme. When charging is deferred, a per-vehicle rule
  leaves 248 kWh undelivered at a 40 kW cap and the joint programme 31 kWh.
- **Whole controllers under the cap.** The shields reduce the energy imported above the cap from 403–503 kWh
  to 3–16 kWh in two weeks, and battery violations from 29–39% to zero.
- **A learner behind a deadline shield stops charging the vehicles** and leaves it to the shield. Holding the
  later hours of the shield's plan to the day-ahead forecast's own error makes that safe; a rule that simply
  avoids the tariff peak then costs as little as the learner in winter.
- **Heat pump alone.** A learned set-point policy is 5–15% cheaper than the thermostat in every season, with
  lower discomfort.

The STEMS paper reports a safety-violation rate of 5.6% on its own buildings and protocol. The 0% above is on
a different building set and a shorter evaluation period, so the two numbers are not a like-for-like
comparison. A comparison in the paper's protocol (full year, five seeds) is being run.

## Limits

- Vehicle schedules are synthetic (a seeded commuter model); buildings, weather and devices are CityLearn's.
- Baselines are no control and one family of rules. No model-predictive controller and no other learning
  method has been run on these scenarios yet.
- Learned controllers under a binding cap were run at one cap, in two seasons, with two seeds.
- Comfort results rest on CityLearn's learned temperature model.

## Layout

| Path | Content |
|---|---|
| `stems/` | Environment wrapper and simulator corrections, battery and tank models, battery barrier, vehicle fleet model and cap shield, learner, reward, metrics |
| `experiments/` | Scenarios, controller arms, runner, experiment grids, aggregation and report scripts |
| `experiments/diagnostics/` | Scripts behind individual numbers in the report |
| `tests/` | 232 tests |
| `docs/` | Report, summary, the simulator defects, a design note on hot water |
| `results/` | One JSON record per run of every experiment in the report |
| `citylearn_schemas/` | Sizing data and vehicle schedules; the schemas themselves are generated locally |

## Getting started

The instructions are for a fresh Ubuntu 24.04 machine. Python 3.12 is required; the pinned packages are the
CPU build of PyTorch, so no GPU is needed.

```bash
sudo apt update
sudo apt install -y git python3.12 python3.12-venv
```

```bash
git clone https://github.com/garyhost2/stems.git
cd stems
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The pinned dependencies: `requirements.lock.txt` holds the exact versions behind the stored results
(`requirements.txt` lists the direct dependencies without versions). The extra index is where the CPU
builds of PyTorch are published:

```bash
python -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
```

CityLearn 2.6.0b1 from source. The script clones that release into `third_party/CityLearn` (about 1 GB, it
contains CityLearn's example data) and installs it without its own dependency list, which would otherwise
downgrade scikit-learn to a version that has no build for Python 3.12:

```bash
bash setup_citylearn.sh
```

The schemas. They point at CityLearn's data folder on the local machine, so they are generated, not stored.
The first call downloads the building data (about 90 MB) into CityLearn's cache:

```bash
python setup_citylearn_8b.py --validate
python setup_citylearn_ev.py --validate
python setup_citylearn_mixed.py
```

The tests (about 10 minutes; 232 should pass):

```bash
python -m pytest tests -q
```

On Ubuntu 22.04, Python 3.12 comes from the deadsnakes archive (`sudo add-apt-repository ppa:deadsnakes/ppa`)
before the `apt install` line. On Windows the virtual environment is activated with
`.venv\Scripts\activate` and the install script is run from Git Bash; everything else is the same.

Every command below assumes the virtual environment is active. A full experiment grid takes hours on eight
cores; a run over a whole year needs about 2 GB of memory per parallel process (`--workers`).

## Reproducing the experiments

The three simulator defects, on plain CityLearn:

```bash
python -m experiments.citylearn_defects
```

Safety layer and controller, four seasons and two building sets:

```bash
python -m experiments.ablation --seasons winter summer spring autumn --subsets ref 1 --seeds 0 1
```

```bash
python -m experiments.aggregate results/ablation_v1
```

Vehicles under a shared cap (stages `rules`, `policy`, `reserve`, `flexibility`):

```bash
python -m experiments.ev_coupling --stage rules
```

```bash
python -m experiments.ev_report
```

Whole controllers under a 40 kW cap:

```bash
python -m experiments.ablation --schema citylearn_schemas/tx_travis_8b_ev/schema.json --grid-cap 40 --seasons winter summer --subsets ref --seeds 0 1 --episodes 60 --days 14 --arms rbc+calibrated rbc-offpeak+calibrated rbc-never+calibrated rl+calibrated rl+calibrated+own rl+calibrated+floor --out results/ev_rl_v2
```

```bash
python -m experiments.ev_rl_report results/ev_rl_v2
```

Heat pump alone:

```bash
python -m experiments.ablation --seasons winter spring summer autumn --subsets ref --seeds 0 1 --episodes 30 --arms idle hp-shift rl-hp --out results/heatpump_v1
```

```bash
python -m experiments.hp_report results/heatpump_v1
```

Every run record stores a hash of the code that produced it; a results folder that mixes code versions, or
contains a failed run, is refused by the report scripts.
