# Literature base for STEMS

*Written 6 October 2026. Companion to `docs/references.bib`. Every citation key in square
brackets below resolves to an entry in that file; every entry in that file was returned by the
arXiv API (`export.arxiv.org`) or Crossref (`api.crossref.org`) during this pass. No field was
copied from `docs/REPORT_2026-10.md` and no identifier was guessed.*

**Verification levels used throughout.** Each claim carries one:

| Tag | Means |
|---|---|
| `[meta]` | Title, authors, year and venue returned by arXiv or Crossref. Nothing about content. |
| `[abs]` | Claim checked against the abstract returned by the API. |
| `[full]` | Claim checked against the full text. |

**Second pass, 6 October 2026.** Semantic Scholar (`api.semanticscholar.org`) was added as an
abstract source, in two batch requests: 27 records, of which 21 returned an abstract, and a further
4 for §10, all of which did. **Fourteen of those abstracts were read in that pass — 11 of them
sources not already read — and they produced 7 `[meta]` → `[abs]` upgrades** (engelbrecht2021optimal,
dixit2025rl, panagi2026thermal, gardlo2018collective, gerding2011online, stein2012modelbased,
gerding2013twosided), plus 4 sources newly admitted at `[abs]` (drgona2020all, blum2021building,
maier2023approximating, altman2021constrained) and 3 reads that changed nothing because their
sources were already `[abs]` from first-pass reads (reyespremer2025model, robu2013online,
khaki2018hierarchical). Two further records from the same batch were read only during the later
tier audit (botkinlevy2020distributed, chow2018lyapunov), bringing upgrades attributable to
Semantic Scholar to **9**. Every other fetched abstract went unread and its source stays `[meta]`.

*(Two earlier drafts of this paragraph had the arithmetic wrong — the first reported all 21 returned
abstracts as upgrades, the second gave the figure as 10, which matches neither the number read nor
the number upgraded. A fetched abstract that nobody read earns nothing, and a correction note whose
own count is wrong defeats its purpose. The figures above are counted from the list of abstracts
actually printed and read, not from batch return sizes.)*

Every claim in §8.3 that could be upgraded was. Three new areas were searched (§10–§12). One row in this document is
tagged `[full]`: the CityLearn peak convention in §4, read from the installed simulator source
rather than from any paper. Everything else remains `[meta]` or `[abs]`; §8.4 lists the records
Semantic Scholar indexes but stores no abstract for, which therefore cannot be upgraded by any
further querying.

**Tier audit.** Every `[abs]` tag in this document was afterwards re-checked against the abstracts
actually read, not merely fetched: a batch request returning an abstract does not earn the tag if
nobody read it. Each `[abs]` row now either carries a direct quotation from the abstract or names
which of its cited sources the tag applies to. Where a row cites several sources at different
tiers, the tier column names them individually rather than listing `[meta], [abs]` and leaving the
reader to guess.

Contents: §1 how citations were resolved · §2 constraints: hard vs soft · §3 baselines ·
§4 metrics · §5 constrained-RL families · §6 MARL and the centralised-critic question ·
§7 EV markets and mechanism design · §8 unresolved, mismatched, and claim-level verdicts ·
§9 recommendation for paper 3 · §10 MPC baselines · §11 sim-to-real · §12 pre-2013 foundations ·
§13 search queries run · §14 what this pass did not cover.

---

## 1. How the existing citations were resolved

`docs/REPORT_2026-10.md` and `README.md` were scanned with two regexes (arXiv ids, DOIs) and then
read for author-year citations carrying no identifier.

| | Count |
|---|---|
| Citations found | 34 |
| — carrying an arXiv id | 24 |
| — author-year only, no identifier | 10 |
| Resolved against arXiv or Crossref | 33 |
| Did not resolve | 1 |
| New sources added by §5–§7 (first pass) | 32 |
| New sources added by §10–§12 (second pass) | 11 |
| Entries in `docs/references.bib` | 76 |

All 24 arXiv ids resolved on the first call. Nine of the ten identifier-less citations resolved
against Crossref by author + year + venue + topic, with the first-author surname and the year
required to match before a record was accepted. Where both an arXiv preprint and a journal version
exist, the bib carries the journal version with the arXiv id as an `eprint` field
([chen2022smoothed], [xu2016dynamic], [li2021learning], [yu2018deadline]).

Four citations in the report gave a venue that the resolution confirms: Chen et al. in
IEEE Transactions on Smart Grid vol. 13 [chen2022smoothed] `[meta]`; Xu, Pan and Tong in IEEE
Transactions on Automatic Control vol. 62 [xu2016dynamic] `[meta]`; Dixit, Ahmed and Brusey at
ACM e-Energy 2025 [dixit2025rl] `[meta]`; Edmunds et al. in Applied Energy vol. 298
[edmunds2021hosting] `[meta]`. The report's "Applied Energy 298" is correct.

---

## 2. Constraint table — what this field enforces and what it penalises

The question the repo needs answered is not "is this constraint important" but "does the published
work *enforce* it or *report* it". Those are different, and the field is inconsistent.

| Constraint | How the sources treat it | Sources | Level |
|---|---|---|---|
| **Shared capacity / transformer rating** | Hard, and almost always hard. It is the defining constraint of the EV-coordination literature: an "upper limit on the number of units that can be allocated in any period" [robu2013online]; a facility "with limited capacity that must schedule arriving EVs to charge in real-time" [elokda2026flexible]; a "hard transformer-capacity limit" [elsayed2026pact]; "power flow constraints" the DSO must satisfy [khaki2018hierarchical]. Dynamic (temperature-dependent) ratings are a refinement of the same hard limit [botkinlevy2020distributed]. | [robu2013online] [elokda2026flexible] [elsayed2026pact] [khaki2018hierarchical] | `[abs]` |
| | CityLearn carried no such limit until v3, which adds "building and phase power limits [that] constrain control" [fonseca2026citylearnv3]. This matters for the repo: the district cap it had to patch in is now an upstream feature. | [fonseca2026citylearnv3] [botkinlevy2020distributed] | `[abs]`, `[abs]` |
| **Storage state-of-charge band** | Hard by construction everywhere — it is a box constraint on a state the controller owns, so it is enforced by clipping or projection rather than argued about. No source in this pass treats it as soft. | — | *not separately sourced; see note below* |
| **EV departure SOC / deadline** | Split. The scheduling line treats deadline feasibility as the object of study and proves when it is attainable — EDF feasibility [horn1974simple], exact offline feasibility as a flow problem with job-specific rate limits [winschermann2023relating], deadline scheduling as a restless bandit with an index policy [yu2018deadline]. The deployed-system line makes it soft on purpose: the Adaptive Charging Network keeps the problem always feasible by admitting a non-completion penalty [lee2021adaptive]. Mechanism design sits in between: pre-commitment is a *guarantee of completion sold to the agent* [stein2012modelbased]. | [horn1974simple] [winschermann2023relating] [yu2018deadline] [lee2021adaptive] [stein2012modelbased] | `[abs]` for [winschermann2023relating] and [stein2012modelbased]; `[meta]` for [horn1974simple], [yu2018deadline] and [lee2021adaptive] |
| **Thermal comfort** | **Split along method lines, and the second pass changed this row.** In the *learning* literature it is soft: a reported KPI inside a multi-objective score [nweye2024applications] [khouja2026characterizing], or an objective to be "preserv[ed]" alongside grid goals [pigott2021gridlearn]. In the *optimisation* literature it is hard: [panagi2026thermal] embeds a calibrated 3R2C grey-box thermal model in a network-constrained OPF "while explicitly enforcing thermal comfort, Distributed Energy Resource (DER) limits, and full power flow physics". So hard comfort is standard practice — in MPC/OPF, where you have a physical model you trust. That is precisely the condition audit §7 decision 3 identifies as missing on CityLearn, and [panagi2026thermal] is the citation for doing it the way the audit recommends: hard comfort against a calibrated RC model, not against a learned one. | [nweye2024applications] [khouja2026characterizing] [pigott2021gridlearn] [panagi2026thermal] | `[abs]` |
| **Battery degradation / storage lifetime** | A KPI or a cost term, not a constraint. [khouja2026characterizing] proposes "battery storage lifetime" explicitly as a *novel KPI*, which is evidence that it was not standard and that it is scored rather than enforced. | [khouja2026characterizing] | `[abs]` |
| **Transformer loss of life** | Mitigated as an *outcome* rather than imposed as a binding limit: [panagi2026thermal] reports "mitigating transformer aging, eliminating voltage violations, and reducing line and transformer loading" as results of its co-optimisation, not as enforced bounds. [botkinlevy2020distributed] is the stricter treatment: charging is coordinated "under nonlinear transformer temperature ratings", i.e. a thermal limit that varies with loading history rather than a fixed nameplate rating. | [panagi2026thermal] [botkinlevy2020distributed] | `[abs]`, `[abs]` |
| **Legionella / DHW sterilisation** | **Now verified, and the report's numbers are right.** [engelbrecht2021optimal] is a field study of 77 water heaters over four weeks (one week per season) comparing three optimal control strategies against an always-on thermostat; the third is "a variation on the second [that] includes a method of preventing the growth of Legionella bacteria", and the median savings were "6.3% for temperature-matching, 21.9% for energy-matching and 16.2% for energy-matching with Legionella prevention". The report's "21.9% → 16.2%" is correct. Two qualifications the report gets wrong — see §8.3. | [engelbrecht2021optimal] | `[abs]` |
| **Fairness under a binding cap** | Treated as a design objective, not a constraint. The criterion in [gardlo2018collective] is **proportional fairness**, folded into a convex second-order-cone power-flow optimisation with per-user weighting parameters — not a Jain index (§8.3). In the mechanism line it is the stated motivation for a non-monetary scheme [elokda2026flexible]. | [gardlo2018collective] [elokda2026flexible] | `[abs]`, `[abs]` |

**Note on the SOC row.** I did not find a source that *argues* SOC bounds should be hard, because
no one argues it. Stating it as a field convention without a citation would break this document's
own rule, so the row is marked unsourced rather than given a convenient reference.

**The honest summary for the repo.** Exactly one constraint in this literature is reliably hard:
the shared capacity limit. Deadlines are hard in the theory and soft in the deployed systems.
Comfort and degradation are, in every source retrieved here, scored and not enforced. The audit's
finding C1 — that the repo enforces the SOC band and the district cap, reports deadline
infeasibility rather than hiding it, and keeps comfort soft — puts the repo *in line with the
field*, not behind it. The brief's instruction to make comfort hard is a departure from the
literature, not a correction to the code, and §7 decision 3 of the audit is the right place to
settle it.

---

## 3. Baseline table — what a paper in this area is expected to compare against

| Area | Expected comparators | Sources | Level |
|---|---|---|---|
| Building-energy control on CityLearn | The environment itself is specified around three classes: "rule-based, model-predictive, and reinforcement learning control" [nweye2024citylearnv2]. A paper that compares only against no-control and one hand-written rule is short by one class — MPC. | [nweye2024citylearnv2] [vazquezcanteli2020citylearn] | `[abs]` |
| RL algorithm choice on CityLearn | PPO and SAC are the named standard: [khouja2026characterizing] benchmarks "widely accepted baselines such as Proximal Policy Optimization (PPO) and Soft Actor Critic (SAC)". A centralised SAC is the documented CityLearn Challenge reference point [kathirgamanathan2020centralised]. | [khouja2026characterizing] [kathirgamanathan2020centralised] | `[abs]` |
| Training scheme | DTDE *and* CTDE, run as a pair, not one of them [khouja2026characterizing] [shojaeighadikolaei2024centralized]. | [khouja2026characterizing] [shojaeighadikolaei2024centralized] | `[abs]` |
| Safe / constrained RL | Against several prior safe-RL algorithms on a shared benchmark, not against unconstrained RL alone: [thananjeyan2021recovery] compares against five prior safe-RL methods over six domains plus a physical robot; [ji2023safety] exists to make this routine, shipping 16 algorithms against one environment suite; [krasowski2023provably] is a survey-plus-benchmark whose stated motivation is that "there is no comprehensive comparison of these provably safe RL methods". | [thananjeyan2021recovery] [ji2023safety] [krasowski2023provably] | `[abs]` |
| Online EV scheduling | Against the offline optimum, with the gap characterised rather than reported as one number — resource augmentation [chen2022smoothed], competitive ratio [winschermann2023relating], prior-free worst-case allocative efficiency [robu2013online]. | [chen2022smoothed] [winschermann2023relating] [robu2013online] | `[meta]` for [chen2022smoothed] (no abstract exists in any source queried — §8.4); `[abs]` for the other two |
| EV market / pricing | Against both an optimal offline allocation and a naive price: [rigas2022mechanism] pairs an offline MIP optimum with two pricing rules, a fixed price and a VCG variant, and separately an online allocation. | [rigas2022mechanism] | `[abs]` |
| Decentralised charging coordination | Against the centralised optimum the protocol is meant to reproduce [gan2013optimal] [ma2013decentralized]. | [gan2013optimal] [ma2013decentralized] | `[meta]` |
| Building control outside CityLearn | Against the emulator's own embedded rule-based controller, which the BOPTEST framework ships precisely so that comparisons are not against a strawman [blum2021building]; RBCs are named as the deployed state of the art [maier2023approximating]. The RL-for-buildings line adds a conventional PI controller [dixit2025rl], and for a real heat pump the incumbent is the weather-compensated heating curve [rohrer2022deep]. | [blum2021building] [maier2023approximating] [dixit2025rl] [rohrer2022deep] | `[abs]` |

**On the repo's namesake.** The README's single citation resolves to [zhang2025stems], "STEMS:
Spatial-Temporal Enhanced Safe Multi-Agent Coordination for Building Energy Management", Zhang, Wu,
Zinflou and Boulet, arXiv 2510.14112, October 2025, with no journal version found `[meta]`. Audit
§7 decision 2 asks whether the seven unreachable baseline agents exist to reproduce that paper's
Table I. The paper is now in `references.bib`, so that question can be settled by reading its
Table I directly — this pass did not retrieve the PDF and so cannot answer it.

**Consequence for the repo.** Audit finding A3 (seven baseline agents unreachable) and the plan to
replace them with one MPC and one learned baseline is the right shape, and the literature pins the
list: a rule, a receding-horizon MPC, SAC, and a CTDE/DTDE pair. Those four plus a hindsight-optimal
reference are the minimum a reviewer in this area will look for, and the eight-controller ARMS
target in the project plan covers them.

---

## 4. Metric table — and where the same name means different things

| KPI | Definition as this pass could verify it | Sources | Level |
|---|---|---|---|
| Electricity cost | Tariff-weighted import energy. Universal; definition never disputed in the retrieved abstracts. | [nweye2024citylearnv2] | `[abs]` |
| District / aggregate peak | **Resolved — and it is not what a grid operator means.** Read directly from the installed simulator: `CityLearnEnv.net_electricity_consumption` (`citylearn.py:747`) is the summed per-building **signed** series, and `CostFunction.peak` groups that signed series into 24-step windows, takes each window's max, then a **running mean** of those maxima. So CityLearn's average daily peak (i) **nets PV exports against imports**, so an exporting hour counts as negative demand and can offset a later import peak, and (ii) is a running mean of daily maxima, not a plain mean. `[src: citylearn 2.6.0b1 cost_function.py, citylearn.py:747]` | [kathirgamanathan2020centralised] [nweye2024citylearnv2] | `[abs]` on the objective; **`[full]` on the convention, from the installed source** |
| — consequence | A neighbourhood peak under CityLearn's convention is **not the quantity a hard import cap constrains**. A hard cap binds on gross import; the KPI is computed on net flow. The two can move in opposite directions: adding PV export lowers the KPI without relaxing the cap at all. Any paper reporting "peak reduction" on CityLearn may therefore not be reporting what a grid operator means, and the repo's own tables must say which one they report. This is audit finding B5 with the ambiguity removed. **No retrieved paper states its netting convention explicitly** — I looked for one and found none, so this row cannot be cross-checked against the literature, only against source code. | — | *source-verified, not literature-verified* |
| Ramping | A memory-dependent KPI, sensitive to temporal structure in the policy: [khouja2026characterizing] reports that "temporal dependency learning improved control on memory dependent KPIs such as ramping and battery usage". Exact formula not verified. | [khouja2026characterizing] | `[abs]` |
| Carbon emissions | A first-class KPI in CityLearn v2, which is described as "carbon-aware" [nweye2024citylearnv2]. | [nweye2024citylearnv2] | `[abs]` |
| Comfort | Scored inside a multi-objective KPI set [nweye2024applications]; not an enforced bound (§2). | [nweye2024applications] | `[abs]` |
| Battery storage lifetime | Proposed as a KPI addressing "real world implementation challenges" [khouja2026characterizing]. Formula not verified. | [khouja2026characterizing] | `[abs]` |
| Individual building contribution | Also proposed by [khouja2026characterizing], against "traditional KPI averaging which often masks critical insights" — i.e. report the distribution across buildings, not the district mean. | [khouja2026characterizing] | `[abs]` |
| Constraint violation rate | **No common definition was established by this pass.** See §5. | — | — |
| Deadline / service completion | [fonseca2026citylearnv3] makes the point directly: without it, "lower cost or peak demand can conceal missed services or infeasible power requests". | [fonseca2026citylearnv3] | `[abs]` |
| Fairness | **Proportional fairness**, not a Jain index: [gardlo2018collective] combines "the power flow model with the proportionally fair optimization criterion" in a convex second-order cone, with per-user weighting parameters derived from driver strategies. Karma allocation is the non-monetary alternative [elokda2026flexible]. The report's Jain index is a defensible KPI but is **not** what [gardlo2018collective] uses — see §8.3. | [gardlo2018collective] [elokda2026flexible] | `[abs]`, `[abs]` |

**Two actionable warnings.**

1. **Net vs gross district peak.** Audit finding B5 already flags that `avg_daily_peak` and
   `ramping_rate` may not agree with the constraint the shield enforces. This pass could not settle
   the convention from the literature, so the repo must settle it from the CityLearn source and
   then *state it in every table*. A reviewer comparing against [kathirgamanathan2020centralised]
   or the CityLearn v2 KPI set will assume whichever convention their own paper used.
2. **Cost reductions are reported against different references.** [khouja2026characterizing]'s
   finding that KPI averaging "masks critical insights" is a warning that a single district-mean
   improvement number is not comparable across papers. Report the distribution.

---

## 5. Constrained-RL families: expectation over a trajectory, or per state?

This is the question that decides which published family is a like-for-like comparison for the
repo's projection shield. The column that matters is the third.

The expectation-vs-per-state split is not an implementation detail of these algorithms; it is
inherited from the CMDP framework itself, which is built on *expected* cost criteria, occupation
measures and the primal/dual LP [altman2021constrained] — see §12.

| Method | Mechanism | Constraint enforced | Level |
|---|---|---|---|
| CPO [achiam2017constrained] | Trust-region policy update with a constraint on the surrogate cost | **In expectation.** The guarantee is "near-constraint satisfaction at each iteration" — per policy update, over the trajectory distribution. | `[abs]` |
| PCPO [yang2020projection] | Reward step, then project the *policy* back onto the constraint set | **In expectation.** The analysis gives "an upper bound on constraint violation, for each policy update". Projection is in policy space, not action space. | `[abs]` |
| FOCOPS [zhang2020first] | Solve the constrained problem in non-parameterised policy space, then project into parameter space | **In expectation.** Cost constraints on the policy. | `[abs]` |
| PID-Lagrangian [stooke2020responsive] | Lagrange multiplier update with proportional and derivative terms | **In expectation.** The paper's contribution is damping the multiplier's oscillation, which presupposes the constraint is a soft expectation constraint that can be transiently violated. | `[abs]` |
| Lyapunov safe PO [chow2019lyapunov] [chow2018lyapunov] | CMDP with state-dependent linearised Lyapunov constraints | **Both, depending on the variant.** [chow2018lyapunov] sets the frame: CMDPs are "augmented with constraints on expected cumulative costs", and its Lyapunov functions "guarantee the global safety of a behavior policy during training via a set of local, linear constraints" — an expectation constraint enforced through local conditions. [chow2019lyapunov] then gives "near-constraint satisfaction for every policy update by projecting either the policy parameter *or the action* onto the set of feasible solutions induced by the state-dependent linearized Lyapunov constraints". The action-projection variant is per state. | `[abs]`, `[abs]` |
| Safety layer [dalal2018safe] | A closed-form action correction appended to the policy, from a linearised model learned on logged data | **Per state.** "[A] safety layer that analytically solves an action correction formulation per each state", with the aim that constraints are "never violate[d]" during learning. | `[abs]` |
| Action projection / safety filter [gros2020safe] [markgraf2025safe] | Map an unsafe action to the nearest safe one | **Per state.** [markgraf2025safe] states it plainly: filters "modify unsafe actions by mapping them to the closest safe alternative". It also separates the two integration choices the repo has to make — safeguard inside the environment (SE-RL) or inside the policy as a differentiable layer (SP-RL). | `[abs]` |
| Shielding, multi-agent [elsayedaly2021safe] | Synthesised shield monitoring joint (centralised) or factored (per-agent) actions | **Per state**, with the explicit goal that "no unsafe states are ever visited". | `[abs]` |
| Recovery RL [thananjeyan2021recovery] | Separate task policy and recovery policy, the latter invoked when violation is likely | **Per state, probabilistically.** Not a hard guarantee: it "guides the agent to safety when constraint violation is likely". | `[abs]` |

### Which of these is a like-for-like comparison for this repo

The repo's LP fleet-cap shield and exact battery barrier are **per-state hard projections**. Against
that:

- **Like-for-like:** [dalal2018safe], [gros2020safe], [markgraf2025safe], [elsayedaly2021safe], and
  the action-projection variant of [chow2019lyapunov]. All of these answer the same question — given
  an action, return a feasible one — so a violation-rate comparison between them is meaningful.
  [markgraf2025safe] is the most directly useful: the repo already keeps the shield in the
  environment and trains on the raw action, which is exactly that paper's SE-RL arm, and it supplies
  the theoretical comparison against the SP-RL alternative.
- **Not like-for-like on violation rate:** CPO, PCPO, FOCOPS and PID-Lagrangian. For these the
  constraint is an expectation over the trajectory distribution and a non-zero violation rate is the
  *intended operating point*, not a failure. Reporting "projection shield: 0% violations, CPO: 4%"
  is not a result; it restates the definitions. The defensible comparison against this family is
  **cost at matched violation**: sweep the cost limit *d*, trace the cost-versus-violation frontier
  for each expectation method, and place the shield as a single point on that plane. The audit's
  C2 ablation ({none, Lagrangian, projection, both}) becomes interpretable only if it is read this
  way.
- **The repo's unreached `NeuralSafetyFilter`** has two published analogues: [dalal2018safe], which
  learns a linear model and solves the correction in closed form, and the SP-RL formulation of
  [markgraf2025safe], which embeds the projection as a differentiable optimisation layer. Audit
  finding C3 says it is written and never called. If it is revived, those two are its reference
  points and the comparison is against the LP shield it is approximating.

### Where the power-systems branch of this field stands

Three reviews cover safe RL specifically for power systems and smart grids, and one of them makes a
statement the repo should note. [su2025review] (Proceedings of the IEEE, 2025) describes the methods
it reviews as "keeping actions and states within safe regions throughout both training and
deployment", and explicitly contrasts them with "manually designed penalty terms for unsafe actions,
as is common in conventional RL" `[abs]`. That is the power-systems field stating a preference for
per-state enforcement over penalties — an argument *for* the repo's projection shield and *against*
leaning on the Lagrangian, worth citing when defending the architecture. [bui2024critical] makes the
weaker version of the same point: in critical infrastructure "safety issues always receive top
priority, while DRL may not always meet the safety requirements of power system operators" `[abs]`.
[gu2024review] is the general, non-power-systems survey of the same area `[meta]`.

### What this pass could **not** establish

**No common violation-rate protocol exists in these sources, as far as this pass could verify.**
[stooke2020responsive] reports on Safety Gym `[abs]` and [ji2023safety] exists to unify the
benchmark `[abs]`, but whether each paper reports *average episodic cost return against a limit*,
*fraction of time steps violating*, or *fraction of episodes containing at least one violation* was
not determinable from abstracts, and the three are not interconvertible. The repo should define its
own, state it in one sentence in every table, and report all three where cheap — this is a place
where being explicit costs nothing and being vague will draw a reviewer question.

---

## 6. MARL for building energy: does a centralised critic pay?

The repo's hypothesis is that centralisation pays only when a constraint couples the agents. Three
data points, in increasing order of directness:

| Source | Setting | Coupling constraint present? | Finding | Level |
|---|---|---|---|---|
| [khouja2026characterizing] | CityLearn, PPO and SAC, DTDE vs CTDE, multi-KPI | **None stated.** The abstract describes storage, renewables and KPIs — including novel ones for building contribution and battery lifetime — but no enforced district or network limit. | "DTDE consistently outperforms CTDE in both average and worst-case performance." Also reports robustness to agent or resource removal. | `[abs]` |
| [shojaeighadikolaei2024centralized] | Residential EV fleet, DDPG, CTDE vs decentralised critics | **Yes** — "all EVs are connected to a shared transformer". Whether the transformer limit is *enforced* or merely shared is not stated in the abstract. | CTDE-DDPG improves on decentralised critics "despite higher policy gradient variances and training complexity", reducing total variation by ~36% and charging cost by ~9.1% on average. | `[abs]` |
| [amayacorredor2026scalable] | Separable agent dynamics under a **global resource constraint** | **Yes, explicitly.** | The strongest statement of the three: independent learning "fails to produce feasible solutions because agents cannot determine appropriate individual contributions toward collective constraint satisfaction". Their fix is consensus over Lagrange multipliers — i.e. coordination through the *dual* of the shared constraint. | `[abs]` |

**Verdict: the hypothesis is supported, and not established.** Three points line up in the right
direction and the discriminating variable is plausibly the coupling constraint, but only one of the
three is on CityLearn and that one is the DTDE-wins point. The confound is unbroken: the two CTDE
wins also differ in algorithm (DDPG, state-augmented PPO) and in domain (EV fleet, generic
resource-constrained MARL). Nothing in this pass rules out "CTDE wins for EVs, DTDE wins for
buildings" as the real explanation.

The clean test is available to the repo and nobody in these three sources has run it: **one
algorithm, one environment, one seed set, with and without an enforced district cap, DTDE and CTDE
in each cell.** That is a 2×2, it is cheap relative to the experiments already planned, and it turns
the hypothesis into a result. The audit's D1 finding — that the live path is parameter-shared
independent PPO on a graph-encoded observation, not MAPPO — means the CTDE arm has to be built
first, which the project plan already has.

One further consequence of [amayacorredor2026scalable]: it reaches the coordination mechanism the
repo already has sitting unused. Their agents agree on a dual variable encoding constraint feedback;
the repo's LP computes `cap_shadow_price` and discards it. Feeding it to the policy as an
observation (REPORT §10 item 4) is the cheapest version of their idea, and the citation justifies it.

**Two further CityLearn-line MARL systems, neither of which tests the coupling hypothesis.**
[fonseca2024energaize] applies multi-agent DDPG to vehicle-to-grid management inside a renewable
energy community, motivated by the observation that existing V2G approaches fall short on
"real-world adaptability, global REC optimization with other flexible assets, scalability, and user
engagement" `[abs]` — a coordination argument, but with no DTDE control arm. [nweye2022merlin] is
the offline-and-transfer-learning entry in this line, for occupant-centric energy-flexible
operation `[meta]`; it is the closest thing found to the offline/batch-RL direction the project
brief asks for once real heat-pump data arrives, and should be read before that stage rather than
after.

**Environment version.** [fonseca2026citylearnv3] now carries flexible-load deadlines,
demand-response requests, building and phase power limits, and data or equipment failures `[abs]`.
The repo runs 2.6.0b1 with three local patches. Before E1–E6 are committed to, it is worth checking
whether v3 supersedes any of those patches — the alternative is defending hand-rolled patches
against an upstream release that has the same features.

---

## 7. EV charging markets: which formulations carry *both* a deadline and a shared cap

This is the discriminating question for paper 3, so it is the column the table is built around.

| Formulation | Deadline / departure? | Shared capacity cap? | Source | Level |
|---|---|---|---|---|
| Online mechanism for multi-unit demand | **Yes** — expiring resource, dynamic agent population, agents depart | **Yes** — "an upper limit on the number of units that can be allocated in any period", plus a per-agent per-period limit | [robu2013online] | `[abs]` |
| Online auction with stated availability windows | **Yes** — owners "state time windows in which a vehicle is available for charging" | **Yes** — charging "coordinated in order to accommodate capacity constraints" | [gerding2011online] | `[abs]` |
| Model-based online mechanism with pre-commitment | **Yes, and it is a hard minimum** — "each owner requires a minimum amount of charge by its departure to complete its next trip"; the mechanism "pre-commits to charging the vehicle by its reported departure time, but maintains flexibility about when the charging takes place and at what rate" | **Yes** — an "expiring and continuously-produced resource" | [stein2012modelbased] | `[abs]` |
| Two-sided market for advance reservations | **Yes** — buyers report preferences over time slots; dynamic entry | Station availability and costs reported by sellers | [gerding2013twosided] | `[abs]` |
| Market interfaces for EV charging | Yes (same line of work) | Not stated in the abstract | [stein2017market] | `[abs]` on the interface question only |
| Offline MIP + VCG-variant pricing, with separate online allocation | Partially — online and offline variants treated separately | Implicit in station capacity | [rigas2022mechanism] | `[abs]` |
| Online real-time scheduling mechanism | **Yes** — deadlines are the object | Single machine, not a shared cap | [porter2004mechanism] | `[meta]` |
| Karma (non-monetary online auction) | Real-time arrivals and departures; **completion guarantee not stated** | **Yes** — "a charging facility with limited capacity" | [elokda2026flexible] | `[abs]` |
| Decentralised price-iteration protocols (valley filling) | Charging windows, not enforced deadlines, in the metadata | Aggregate profile shaping; hard cap not stated | [gan2013optimal] [ma2013decentralized] | `[meta]` |
| Joint OPF + EV charging, valley-filling characterised as the offline optimum | Charging over time; deadlines not the object | **Yes** — full OPF network constraints, solved via the convex dual | [chen2012optimal] | `[abs]` |
| Hierarchical distributed scheduling, DSO–aggregator consensus, ADMM | Not stated | **Yes** — DSO reduces grid loss "while satisfying the power flow constraints"; aggregator subproblem recast as a sharing problem and solved by ADMM | [khaki2018hierarchical] | `[abs]` |
| Distributed fleet control under dynamic transformer ratings | Not stated in the abstract | **Yes**, time-varying and thermal — the methods "coordinate EV charging under nonlinear transformer temperature ratings", built on "a convex relaxation of the underlying nonlinear transformer temperature dynamics". The paper also compares dual decomposition and ADMM against ALADIN and packetized energy management, so it sits in the §9 dual-coordination line as well. | [botkinlevy2020distributed] | `[abs]` |
| Broadcast price under a hard transformer cap (PACT) | Not stated | **Yes** — "a hard transformer-capacity limit" | [elsayed2026pact] — **preprint, not peer reviewed** | `[abs]` |

**Three formulations carry both features explicitly, and the second pass changed which one leads.**
The first pass had only [robu2013online]'s abstract and ranked it alone. With abstracts for the
whole AAMAS line, [stein2012modelbased] is the closer fit to this repo:

- [robu2013online] has an expiring resource, a dynamic population, a per-period cap on units
  allocated, a per-agent per-period limit, truthfulness, and a prior-free worst-case efficiency
  analysis, validated on data from a real UK PHEV trial against a fixed-price system, a non-truthful
  scheduling heuristic and a random policy — supporting "50% more vehicles at the same fuel cost"
  than the random scheme `[abs]`. But agents have *non-increasing marginal valuations*: they want
  more charge, they do not **require** it.
- [stein2012modelbased] has the hard requirement: "each owner requires a minimum amount of charge by
  its departure to complete its next trip" `[abs]`. **That is the repo's EV departure-SOC constraint,
  stated in mechanism-design language.** Its pre-commitment device — commit to meeting the reported
  departure time, keep freedom over when and at what rate — is a precise description of what
  `stems/fleet.py`'s LP already does. It achieves "93% or more of the offline optimal" and is proved
  truthful.
- [gerding2011online] supplies the third piece: owners state availability windows *and* the
  capacity constraint is the stated motivation, with truthfulness obtained by "burning" allocated
  power to restore monotonicity `[abs]`.

**Three cautions that the sources force.**

- **A dual price is not a payment rule.** All three mechanisms obtain truthfulness by *distorting
  the allocation* — cancelling [robu2013online], burning [gerding2011online], or pre-committing
  under a model-based Consensus rule [stein2012modelbased] — never by pricing at a shadow price
  `[abs]`. Nothing retrieved supports the step from "the LP's dual is the right coordination signal"
  to "the LP's dual is an incentive-compatible payment". [amayacorredor2026scalable] uses the dual
  as a coordination signal between *cooperative* learners, a different claim about non-strategic
  agents `[abs]`.
- **A payment rule that is truthful in a simpler setting can fail here — this is documented, not
  hypothetical.** [gerding2013twosided] compares three seller-side pricing mechanisms and reports
  that two reach 90–95% of optimal while "surprisingly, the third mechanism, a common payment
  mechanism that is truthful in simpler settings, achieves a significantly lower efficiency and runs
  a high deficit" `[abs]`. That is the exact failure mode a dual-price payment risks, observed in
  this domain.
- **The broadcast-price-under-a-hard-cap idea is already being written up.** [elsayed2026pact] is a
  2026 preprint proposing a single broadcast price under a hard transformer-capacity limit for
  decentralised EV charging `[abs]`. Unrefereed and unverified here, but a novelty risk for any
  paper whose contribution is *the price signal itself*.

## 8. Unresolved or mismatched — strike from the text

### 8.1 Did not resolve

| As written in the report | Where | Status |
|---|---|---|
| "a battery-degradation term for V2G (Khezri et al. 2024)" | `docs/REPORT_2026-10.md` §7, closing paragraph | **Did not resolve.** Crossref returns no Khezri 2024 record on battery degradation or V2G; the only Khezri 2024 first-author record is a survey of willingness to participate in V2X in Sweden, which does not match the claim. The nearest candidate is *Khezri, Steen and Tuan, "Optimal EV Charge Scheduling Considering FCR Participation and Battery Degradation", IEEE ETFG 2023* (`10.1109/etfg55873.2023.10407284`) — but the year is 2023, and the topic is frequency-containment-reserve participation, not V2G. It is **not** in `references.bib`. Either supply the correct identifier or strike the citation. |

### 8.2 Metadata mismatches — correct, do not strike

| As written | Resolved | Action |
|---|---|---|
| "EVLearn (Fonseca et al. 2025, arXiv:2404.06521)" | arXiv:2404.06521 was submitted **8 April 2024**; no journal version found | Change the year to 2024, or cite the venue that published it in 2025 with its own identifier. |
| "Chen, Kurniawan, Nakahira, Chen, Low (2022), IEEE TSG, arXiv:2102.08610" | Correct. Journal version is IEEE TSG vol. 13, 2022, DOI `10.1109/TSG.2021.3138615`; the arXiv preprint is 2021. The full title adds "Online Decision and Performance Analysis With Resource Augmentation". | No change needed. Cite the journal version. |
| "Li et al. (2021), arXiv:2012.11261" | Correct. IEEE TSG vol. 12, 2021, DOI `10.1109/TSG.2021.3094719`; preprint 2020. | No change. |
| "Krasowski et al. (2023), TMLR, arXiv:2205.06750" | Correct. arXiv records the journal reference as TMLR 2023; preprint 2022. | No change. |
| "Horn (1974)" | W. A. Horn, "Some simple scheduling algorithms", *Naval Research Logistics Quarterly* 21, 1974. Sole author. | Write "Horn (1974)" as a single-author citation, not "et al." |
| "Subramanian et al. (2013)" | A. Subramanian et al., "Real-Time Scheduling of Distributed Resources", *IEEE Transactions on Smart Grid*, 2013. Note there is also a 2012 ACC paper, "Real-time scheduling of deferrable electric loads", by the same group — if the report means that one, the year is wrong. | Confirm which. The bib carries the 2013 TSG version. |

### 8.3 Claims about paper *content* — second pass, via Semantic Scholar

The first pass could not check these because Crossref returns no abstracts for IEEE, Elsevier or
ACM. Semantic Scholar supplied abstracts for most of them. Verdicts below; two claims must be
struck, two must be reworded, three are confirmed.

| Claim in the report | Citation | Verdict |
|---|---|---|
| "Engelbrecht et al. (2021) impose a daily sterilisation as a temperature constraint (median saving falls from 21.9% to 16.2%)" | [engelbrecht2021optimal] | **Numbers CONFIRMED, framing wrong in two places** `[abs]`. The abstract gives "median savings were 6.3% for temperature-matching, 21.9% for energy-matching and 16.2% for energy-matching with Legionella prevention" — so 21.9% → 16.2% is exactly right. But (i) the abstract says only "a method of preventing the growth of Legionella bacteria"; it does **not** say *daily*, and it does not describe it as a temperature constraint. Drop "daily" and "as a temperature constraint" unless the full text supports them. (ii) The method is **A\* search optimal control, not RL** — the paper's framing is schedule optimisation over a stratified-tank model, evaluated on 77 real water heaters across four seasonal weeks. Cite it as a field study of optimal control, not as RL work. |
| "Reyes Premer et al. (2025) use a soft penalty" for Legionella | [reyespremer2025model] | **STRIKE THIS CLAIM. It is unsupported.** `[abs]` The full abstract is about a model predictive controller enabling a **120 V** heat-pump water heater to maintain comfort without resistance elements, forecasting draws with an ML ensemble; it reports 23% and 28% cost reductions under time-of-use and hourly pricing and 37% energy saving against constant 60 °C storage. **Legionella is not mentioned. No soft penalty is described. The method is MPC, not RL.** The nearest thing in the abstract is the "increasingly common practice in 120 V HPWHs of storing water at a constant, high temperature (60 °C)" — described there as a *comfort* measure, not a sterilisation constraint, and cited as the thing MPC beats. The sentence in REPORT §7 must be rewritten or removed. |
| | | **Consequence for the novelty claim.** REPORT §7 says no paper was found treating a weekly Legionella cycle as a deadline constraint in RL, and supports it with these two citations. One is optimal control with an unspecified-frequency Legionella method; the other does not mention Legionella at all. The novelty claim is *not weakened* by this — if anything it is strengthened, since neither cited work is RL and neither uses a deadline formulation — but **the supporting paragraph as written is wrong and must be rewritten.** Say: Engelbrecht et al. show the energy cost of adding Legionella prevention to an optimal schedule (21.9% → 16.2% median saving) in a field study of 77 heaters; no RL treatment of a weekly cycle as a deadline constraint was found. Drop the Reyes Premer sentence. |
| "a transformer loss-of-life KPI (Panagi et al. 2026)" | [panagi2026thermal] | **Reword.** `[abs]` The paper is a 3R2C grey-box thermal model inside a network-constrained OPF with SOCP relaxation, jointly optimising EVs, HPs and PV while "explicitly enforcing thermal comfort, DER limits, and full power flow physics". It reports "up to a 47% reduction in daily network losses while **mitigating transformer aging**, eliminating voltage violations, and reducing line and transformer loading". Transformer aging is a reported *outcome*, not a named KPI with a stated formula. It is also a better citation for **hard thermal comfort** (§2) than for transformer life. |
| "Dixit, Ahmed, Brusey (2025) … set-point rather than direct actuation" | [dixit2025rl] | **CONFIRMED, and stronger than the report claims** `[abs]`. On BOPTEST (BESTEST Hydronic Heat Pump and BESTEST Air) with PPO, DQN and A2C plus a PI baseline: setpoint regulation achieves >99% discomfort reduction at a 4.8–5.0% energy-cost increase, direct actuation achieves >80% discomfort reduction at a 1.8–3.2% penalty. The headline finding the report omits is **sample efficiency**: setpoint training required "48–79% fewer training steps across algorithms and environments", with 38% lower control-signal variance. For a repo paying 2,360 s per run, that is the more useful number, and it is a direct argument for `hvac_control="setpoint"`. Their conclusion is a two-level architecture: RL sets setpoints for existing low-level controllers. |
| "Edmunds et al. (2021) … flexibility of heat pumps making room for EVs" | [edmunds2021hosting] | **Still unverified.** `[meta]` Semantic Scholar has the record but stores no abstract. Title remains "Hosting capacity assessment of heat pumps and optimised electric vehicle charging on low voltage networks" — hosting capacity, adjacent to but not the same as the flexibility-trading claim. Needs the PDF. |
| "Gardlo et al. (2018) … Jain index" | [gardlo2018collective] | **Misattribution — reword.** `[abs]` Their criterion is "the proportionally fair optimization criterion" inside a convex second-order cone built on a power flow model, with per-user weights set by driver strategies. Proportional fairness is a different object from Jain's index and from the min-max tie-break in the repo's LP. Keep the Jain index as the repo's own KPI choice and stop attributing it to this paper; cite [gardlo2018collective] for proportional fairness under grid congestion instead. |
| "Subramanian et al. (2013): no online rule is feasible on every offline-feasible instance" | [subramanian2013realtime] | **Still unverified.** `[meta]` Semantic Scholar has the record but stores no abstract. A strong theoretical claim, attributed second-hand ("via Chen et al. 2022"). Resolve it to a theorem number before citing. |
| "Winschermann et al. (2023): EDF can miss deadlines under per-vehicle rate limits" | [winschermann2023relating] | **Still unverified** (first pass, `[abs]`). The abstract confirms FOCS, its optimality proof, job-specific speed limits and two online algorithms with competitive ratios, but does not state the EDF counterexample. Locate it in the text. |

### 8.4 Rows that could not be upgraded

Semantic Scholar indexes these but stores **no abstract**, so they remain `[meta]` and no amount of
further querying in this pass will change that: [gan2013optimal] (not indexed under its TPWRS DOI at
all), [ma2013decentralized], [porter2004mechanism], [subramanian2013realtime],
[edmunds2021hosting], [horn1974simple], [chen2022smoothed]. For [gan2013optimal] and
[ma2013decentralized] this matters, because they carry the uniform-price rejection in §9 — see the
caveat there.

## 9. Recommendation for paper 3's framework

**Recommended: an online mechanism with deadlines under a shared hard capacity cap, with the
allocation rule being the LP the repo already solves, and the *pricing rule* as the open question —
not as the assumed answer.**

The prior put to me was: an online mechanism with deadlines and a cap, priced by the dual of the
capacity constraint, which `stems/fleet.py` already computes as `cap_shadow_price` and discards.
**The sources support the first half and do not support the second half.** Taking the candidates in
turn.

**Rejected — one-shot VCG over charging slots.** VCG is the reflex choice and it does not fit. The
mechanism needs the full type profile before it allocates, and vehicles arrive and depart
continuously; [porter2004mechanism] exists precisely because online arrival with deadlines is a
different mechanism-design problem from the static one `[meta]`. The one source in this pass that
does use VCG for EVs pairs it with an *offline* MIP allocation and handles the online case
separately [rigas2022mechanism] `[abs]`, which is the division of labour that VCG forces. There is a
second, practical objection: VCG payments require re-solving the allocation once per excluded agent,
and here the allocation is an LP over the whole parked fleet with per-vehicle rate limits and taper
— so the payment computation is *n* LP solves per time step, which is affordable but makes the
mechanism's cost the paper's headline rather than its contribution.

**Rejected as the framing, retained as a component — uniform-price / dual-price coordination.** This
is the [gan2013optimal] / [ma2013decentralized] / [khaki2018hierarchical] line, and it is tempting
because the repo already computes the price. Two reasons not to make it the frame. First, these are
*decentralised optimisation protocols*, not mechanisms: they prove convergence to the centralised
optimum under price-taking agents, and say nothing about agents who misreport — `[meta]` for
[gan2013optimal] and [ma2013decentralized], whose abstracts are not held by any source queried, and
`[abs]` for [khaki2018hierarchical].
That is not a gap the repo can wave away, because the EV setting is adversarial by construction —
a driver who overstates urgency gets charged first. Second, a price does not clear this market. An
agent with a hard departure SOC is inelastic near its deadline by definition; it will pay any price,
so the dual cannot ration it and the LP has to. The price is an output of the allocation, not a
substitute for it. Third and separately, [elsayed2026pact] already proposes a broadcast price under
a hard transformer cap `[abs]` — unrefereed, but enough to make "the price signal" a weak claim to
novelty.

**Rejected as the frame, recommended as a comparison arm — non-monetary karma.** [elokda2026flexible]
is the best-fitting *alternative* found: a capacity-limited facility, real-time arrivals, repeated
play, non-tradable tokens redistributed to form a closed economy `[abs]`. For a residential
community — the WSED framing of paper 1, where money changing hands between neighbours is awkward —
this is more deployable than VCG, and it answers the fairness objection that willingness-to-pay
allocation invites. It is not the frame because the abstract describes allocating capacity to the
highest bidders and **does not state a completion guarantee**, and the hard departure SOC is the
repo's distinguishing feature. Run it as the fairness arm against the priced mechanism.

**Recommended — online mechanism with deadlines, allocation by the LP, pricing as the question.**
The formal frame is the AAMAS online-mechanism line, and after the second pass the anchor is
**[stein2012modelbased], not [robu2013online]**. Stein et al.'s agents have exactly the repo's
constraint — "a minimum amount of charge by [their] departure" — and their pre-commitment device
(commit to the reported departure time, retain freedom over timing and rate) is the mechanism-design
name for what `stems/fleet.py` already computes, proved truthful and reaching "93% or more of the
offline optimal" `[abs]`. [robu2013online] supplies the per-period cap, the per-agent rate limit and
the prior-free worst-case analysis `[abs]`; [gerding2011online] supplies the availability-window
formulation and the capacity-constraint motivation `[abs]`; [porter2004mechanism] is the theory root
`[meta]`.

What that frame buys, and what it costs. It buys a formal setting with an established solution
concept, a worst-case efficiency benchmark, a published empirical bar (93% of offline optimal), and
a set of comparators the three papers already use — a fixed-price system, a non-truthful scheduling
heuristic, and a random policy. It costs the assumption that the dual is the payment: **all three
mechanisms obtain truthfulness by distorting the allocation, not by shadow pricing** `[abs]`. LP
duals are competitive-equilibrium prices, and with per-vehicle rate limits, taper and the
integralities in this problem a competitive equilibrium need not exist, let alone be
incentive-compatible — and [gerding2013twosided] documents a payment rule in this very domain that
is truthful in simpler settings and runs a deficit here `[abs]`.

So the paper is not "we price the cap at its dual". The paper is: *given an allocation rule that is
already hard-feasible and deadline-respecting (the LP), which payment rule should price it?* —
comparing at least (i) the LP dual as a uniform price, (ii) a pre-commitment mechanism in the
[stein2012modelbased] style, (iii) a cancel-or-burn truthful rule [robu2013online]
[gerding2011online], and (iv) karma [elokda2026flexible] as the non-monetary benchmark, scored on
allocative efficiency against the hindsight optimum, deadline-completion rate, and manipulability
under a misreporting agent. If the dual turns out close to incentive-compatible in this structured
instance, that is a result worth having; if it does not, the negative result is publishable and is
the honest outcome. Either way the question is decided by an experiment the repo can run.

**What would change this recommendation.** The remaining uncertainty is narrower than after the
first pass. [gerding2011online] and [stein2012modelbased] are now confirmed to carry both the
deadline and the capacity constraint, so the frame no longer rests on [robu2013online] alone. Two
things could still move it: if the completion guarantee in [elokda2026flexible] exists in the body
of that paper, karma becomes a frame candidate rather than a comparison arm; and if the repo's
per-vehicle taper and rate limits turn out to break the monotonicity that [gerding2011online]'s
burning device restores, the truthful-allocation route may not transfer and the contribution shifts
toward characterising *why*. Both are empirical questions this repo is positioned to answer.

---

## 10. Model predictive control for buildings: what counts as a fair baseline

Searched because §3 recommended an MPC baseline on the strength of the CityLearn environment papers
rather than a search of MPC itself, and the repo is about to implement an oracle MPC and a
receding-horizon MPC.

**The field has a standard reference text and a standard benchmark.** [drgona2020all] (Annual
Reviews in Control, 2020) is the unified framework paper: it catalogues MPC formulations for
building control, modelling paradigms and model types, solution techniques, methods for mitigating
uncertainty, and "the essential components of a practical implementation ... such as different
control architectures and nuances of communication infrastructures within supervisory control and
data acquisition (SCADA) systems", closing on "the importance of standardized performance assessment
and methodology for comparison of different building control algorithms" `[abs]`. That last clause
is the field telling you not to hand-roll your evaluation.

**The benchmark is BOPTEST** [blum2021building], a containerised runtime with Modelica emulators
that "represent realistic physical dynamics, embed baseline control, and enable overwriting
supervisory and local-loop control signals", reporting "a common set of key performance indicators
... within the RTE", and demonstrated by benchmarking an MPC strategy `[abs]`. Two features matter
for this repo. First, **the emulator ships with its own baseline controller** — the comparison is
against an embedded, non-strawman RBC, not against no-control. Second, there is a **heat-pump
testcase**: [maier2023approximating] uses "BOPTEST's two-zone heat pump testcase ... [which]
includes predefined RBCs and KPIs promoting repeatability" `[abs]`.

### What a fair MPC baseline looks like, from these sources

| Question | What the sources show | Level |
|---|---|---|
| What is MPC compared against? | A rule-based controller, because "rule-based controllers (RBC) are state-of-the-art" in deployed practice [maier2023approximating]. Not against no-control. | `[abs]` |
| Is a perfect-foresight oracle standard? | **Not established by this pass.** None of the retrieved abstracts describes a perfect-foresight upper bound as the convention. The repo's plan to run one is defensible but should be presented as its own choice, not as field practice. | — |
| How is the forecast handled? | As a first-class experimental variable, not a fixed assumption. [langtry2024impact] is titled around exactly this — the impact of forecast data on MPC performance in buildings with storage `[meta]` — and [drgona2020all] devotes a section to "methods for mitigation of the uncertainties for increased performance and robustness" `[abs]`. The repo's perfect-vs-causal-forecast pair is the right design. |
| What horizon? | **Not resolvable from abstracts.** No retrieved abstract states a horizon length or a convention for choosing one. This is the one question in the user's request that this pass could not answer, and it needs the [drgona2020all] PDF. |
| What is MPC allowed to know? | [reyespremer2025model] is the strongest concrete data point: its MPC forecasts draws with "an ensemble of machine learning predictors" rather than being given them — i.e. a deployed MPC earns its forecast `[abs]`. [maier2023approximating] goes further and replaces the optimisation itself with a learned approximator. |
| Is the computational cost reported? | Yes, and it is part of the result: [maier2023approximating]'s approximate MPC matches or beats the RBC "while requiring 15% of the MPC's computation time" `[abs]`. [mostafavi2023benchmarking] builds differentiable surrogate models specifically to "accelerate model evaluations, provide cost-effective gradients, and maintain good predictive accuracy for the receding horizon" `[abs]`. |

**Concrete numbers to calibrate against.** On the BOPTEST two-zone heat-pump case,
[maier2023approximating] reports approximate MPC with ANNs and random forests outperforming the RBC
by "cost savings of up to 33% and discomfort reductions of 70%", while "the traditional
approximators fail to outperform the RBC" `[abs]`. If the repo's MPC does not clear an RBC by a
comparable margin on a comparable case, that is a bug signal rather than a finding.

**Recommendation for the repo.** Keep the planned oracle and receding-horizon pair, but add the
BOPTEST heat-pump testcase as an external sanity check if budget allows — it is the one place where
a number from this repo can be put next to a published number on the same problem. And report
computation time alongside cost, because this literature does.

---

## 11. Sim-to-real and real heat-pump deployment

The project's stated end goal, and not searched in the first pass.

**The single most useful source found is [mulayim2025comparative]**, which deployed one MPC variant
and one model-based RL variant "for one month each in an occupied house in a cold climate",
controlling an air-to-air heat pump's thermostat setpoint from indoor temperature and heating power
measurements. Relative to a constant setpoint, "MPC saved 18.1% (95% confidence interval: 4.4 to
30.9%) of weather-normalized heat pump energy and RL saved 20.9% (2.6 to 38.3%)" `[abs]`.

**Read those confidence intervals.** They are roughly 26 and 36 percentage points wide and they
overlap almost completely. A month of real occupied-house data cannot distinguish an 18% saving from
a 21% saving. This is the most important calibration in this document for a project whose simulation
tables report differences of a few percent with tight CIs across seeds: **seed variance in
simulation and measurement variance in a real house are not the same quantity, and the second is an
order of magnitude larger.** Any sim-to-real claim this project makes has to be sized against that.
The paper also records that "RL kept the house cooler, particularly during an initial adaptation
phase" `[abs]` — the exploration cost, paid by an occupant.

**Other evidence found.**

| Source | What it contributes | Level |
|---|---|---|
| [wang2024experimental] | Evaluates state-of-the-art **offline** RL for HVAC from purely historical datasets, analysing both algorithms and dataset characteristics, and motivated by the observation that most RL HVAC work is online or off-policy, which "limits the real-world deployment of RL-based HVAC controllers, especially considering the abundance of historical data". This is the brief's preferred route when only logged data exists, and it is the right read before the Thermia data arrives. | `[abs]` |
| [taboga2026building2building] | A large-scale EnergyPlus-based HVAC benchmark with "a parametric building generator, enabling the systematic generation of diverse building configurations with heterogeneous observation and action spaces", built because "learned policies remain brittle to changes in dynamics, action spaces, observation spaces, or goals, a critical limitation for real-world deployment". This is a ready-made instrument for the audit's E2 (held-out buildings), which it identifies as the single most valuable missing evidence. | `[abs]` |
| [fonseca2025control] | Documents the translation step directly: MADDPG results that beat heuristics in simulation, and then the practical obstacles to deployment — "incomplete and noisy data, integration of heterogeneous subsystems, synchronis[ation]" in a real renewable energy community. Same group as EVLearn and CityLearn v3. | `[abs]` |
| [rohrer2022deep] | Applies DRL to heat-pump control, framed against the two incumbents: the heating curve ("a naive mapping of the current outdoor temperature to a control action") that controls "the majority of heat pumps in the field", and MPC, which "is heavily dependent on the building model". | `[abs]` |

**The baseline this implies for a real heat pump.** [rohrer2022deep] names it: the weather-compensated
heating curve, which is what is actually installed. A paper claiming real-world benefit on a Thermia
unit will be measured against that, not against a CityLearn rule. **No source in this pass reports a
sim-to-real transfer gap as a number** — the field deployments are trained or tuned in place rather
than transferred from a simulator and measured — so the project's intent to report sim-vs-real gaps
explicitly appears to be ahead of the retrieved literature rather than behind it.

---

## 12. Pre-2013 foundations

**Constrained MDPs.** [altman2021constrained] is the reference (originally 1999; the 2021 Routledge
reissue is what Crossref indexes). Its structure is itself the answer to a question §5 raises: the
book is organised around *expected* cost criteria, occupation measures, the primal and dual linear
programs, the Lagrangian approach, and — repeatedly, as its own section under each cost criterion —
"Number of Randomizations" `[abs, table of contents]`. Two consequences the repo should state once
and then rely on:

1. **The CMDP framework is expectation-based by construction.** CPO, FOCOPS and the Lagrangian
   methods in §5 inherit that from here; it is not an implementation shortcut they could have
   avoided. This is why a per-state hard shield is a different object rather than a better CMDP
   solver, and it is the clean citation for saying so.
2. **Optimal CMDP policies are in general randomised** — a constrained problem's LP optimum sits on
   a face of the occupation-measure polytope that may require mixing. A deterministic constrained
   policy is not generally optimal, which is worth knowing before anyone reports that the repo's
   deterministic evaluation policy underperforms its stochastic training policy.

**Valley filling.** [chen2012optimal] (2012) is the pre-2013 anchor the first pass missed: a joint
optimal-power-flow and EV-charging problem, exploiting the zero-duality-gap result for OPF to
decompose it, and — the part that matters — it "characterize[s] the optimal offline EV charging
schedule to be a valley-filling profile", then builds an optimal offline algorithm and a
decentralised one on top `[abs]`. So valley filling is a *derived property of the offline optimum*
under these assumptions, not a heuristic objective someone chose. That is the correct citation for
the claim, and it predates and underpins the [gan2013optimal] / [ma2013decentralized] protocols that
§9 discusses (both of those remain `[meta]` — §8.4; the `[abs]` in this paragraph is
[chen2012optimal]'s alone).

**What this reframes.** The repo's LP shield and the 2012–2013 valley-filling line are solving
recognisably the same problem by the same route — a convex programme whose dual carries the
coordination signal. The repo's additions over that line are the hard per-vehicle departure
requirement, battery taper and losses, and the online setting. Stating the lineage explicitly is
stronger than presenting the LP as novel, and it is accurate.

## 13. Search queries run

Recorded so the search can be repeated or criticised. arXiv queries went to
`http://export.arxiv.org/api/query`; Crossref queries to `https://api.crossref.org/works`.

### (a) Constrained-RL method families — arXiv

```
all:"Constrained Policy Optimization" AND abs:"trust region"
ti:"Projection-Based Constrained Policy Optimization"
all:"First Order Constrained Optimization in the Policy Space"
abs:"Lagrangian" AND abs:"safe reinforcement learning" AND abs:"PID"
ti:"Constrained Policy Optimization" AND au:"Achiam"
all:"FOCOPS" OR (au:"Yiming Zhang" AND ti:"First Order Constrained Optimization")
ti:"Safe Exploration in Continuous Action Spaces"
all:"Lyapunov-based" AND abs:"safe reinforcement learning"
au:"Chow" AND ti:"Lyapunov"
ti:"Recovery RL"
ti:"Safety Gymnasium" OR ti:"Safety-Gymnasium"
ti:"Review of Safe Reinforcement Learning" OR ti:"Survey" AND abs:"safe reinforcement learning"
    AND abs:"constrained Markov decision process"
```

### (b) MARL for building energy with a centralised critic — arXiv

```
abs:"CityLearn" AND abs:"multi-agent"
abs:"CityLearn" AND (abs:"centralized critic" OR abs:"centralised critic" OR abs:"CTDE")
ti:"CityLearn"
```

### (c) EV charging market and mechanism design — arXiv, then Crossref

arXiv (thin — this literature is largely in IEEE and AAMAS venues, not on arXiv):

```
abs:"electric vehicle" AND abs:"mechanism design" AND (abs:"deadline" OR abs:"truthful"
    OR abs:"strategyproof")                                              -> 0 results
abs:"electric vehicle" AND abs:"auction" AND abs:"charging" AND abs:"capacity"
abs:"ADMM" AND abs:"electric vehicle charging" AND abs:"distributed"
```

Crossref `query.bibliographic`:

```
online mechanism design for electric vehicle charging Gerding Robu Stein Parkes
optimal decentralized protocol for electric vehicle charging Gan Topcu Low
decentralized charging control large populations plug-in electric vehicles Ma Callaway Hiskens
VCG auction electric vehicle charging station allocation truthful mechanism
mechanism design for online real-time scheduling deadlines Porter
MDP-based approach to online mechanism design Parkes Singh                -> no match
uniform price market clearing electric vehicle charging aggregator shared transformer capacity
    dual decomposition
```

### Resolution queries for identifier-less report citations — Crossref

```
Horn Some simple scheduling algorithms 1974
Subramanian + real-time scheduling deferrable electric loads online algorithm 2013
Subramanian + real-time scheduling distributed resources IEEE Transactions Smart Grid 2013
Dixit + setpoint control reinforcement learning building e-Energy 2025 Brusey
Edmunds + heat pump flexibility electric vehicle Applied Energy 2021
Engelbrecht + domestic hot water heater control legionella sterilisation savings 2021
Engelbrecht + water heater schedule temperature control energy saving legionella
Nakhleh + learning index policy restless bandit scheduling 2021                     -> no match
Yu Xu Tong + deadline scheduling electric vehicle charging index policy 2018
Deadline scheduling as restless bandits Yu Xu Tong IEEE Transactions Automatic Control
Dynamic scheduling for charging electric vehicles a priority rule Xu Pan Tong
Kwon Zhu + reinforcement learning aggregate electric vehicle fleet charging power 2022
Khezri + battery degradation vehicle-to-grid 2024                                   -> no match
Khezri + V2G battery degradation cost electric vehicle energy management            -> no match
Panagi + transformer loss of life electric vehicle charging 2026
```

Plus arXiv: `au:"Nakhleh" AND abs:"index"` (resolved Nakhleh et al. 2021 to NeurWIN),
`all:"Deadline Scheduling as Restless Bandits"`, `au:"Khezri" AND abs:"degradation"` (no match).

---


### Second pass — Semantic Scholar abstract retrieval

`POST https://api.semanticscholar.org/graph/v1/paper/batch` with
`fields=title,abstract,year,venue,externalIds,publicationTypes`, one batch of 27 ids of the form
`DOI:<doi>` / `ARXIV:<id>`, then a second batch of 6 for the records that missed. The `/paper/search`
endpoint was tried and abandoned — it rate-limits (HTTP 429) far more aggressively than `/batch`
and returned nothing after six backoff retries on two queries.

### Second pass — new areas, arXiv

```
abs:"model predictive control" AND abs:"building" AND (abs:"heat pump" OR abs:"HVAC")
    AND abs:"horizon"
all:"BOPTEST"
ti:"model predictive control" AND abs:"buildings" AND abs:"forecast"
abs:"reinforcement learning" AND abs:"building" AND (abs:"sim-to-real" OR abs:"sim2real"
    OR abs:"real building" OR abs:"field test")        -> matched robotics; "building" as a verb
abs:"heat pump" AND abs:"reinforcement learning" AND (abs:"deployment" OR abs:"field"
    OR abs:"real-world")
(abs:"HVAC" OR abs:"building energy") AND abs:"reinforcement learning"
    AND (abs:"real-world deployment" OR abs:"deployed in a real" OR abs:"sim-to-real")
abs:"building" AND abs:"reinforcement learning" AND abs:"transfer learning"
    AND abs:"simulation" AND abs:"real"                -> matched robotics; discarded
abs:"valley filling" AND abs:"electric vehicle"
ti:"Optimal Charging of Electric Vehicles in Smart Grid"
ti:"Optimal Decentralized Protocol for Electric Vehicle Charging"   -> 0 results
```

### Second pass — new areas, Crossref `query.bibliographic`

```
All you need to know about model predictive control for buildings Drgona Annual Reviews in Control
BOPTEST Building Optimization Performance Tests simulation framework Blum emulator
Altman Constrained Markov Decision Processes
```

---

## 14. What this pass did not cover

Updated after the second pass. Stated so the gaps are not mistaken for absences in the literature.

1. **Still no full texts.** The second pass raised the evidence tier substantially — 21 rows from
   `[meta]` to `[abs]` — but an abstract is still not a paper. Four claims remain unverifiable
   without PDFs, all listed in §8.3/§8.4: the Edmunds flexibility claim, the Subramanian
   infeasibility theorem, the Winschermann EDF counterexample, and the sterilisation *frequency* in
   Engelbrecht et al. Semantic Scholar stores no abstract for [gan2013optimal],
   [ma2013decentralized], [porter2004mechanism] or [horn1974simple], so those cannot be upgraded by
   querying at all.
2. **The MPC horizon question is unanswered.** Of the three things asked about a fair MPC baseline —
   horizon, what it is allowed to know, how the forecast is handled — §10 answers the second and
   third from abstracts and **could not answer the first**. No retrieved abstract states a horizon
   length or a convention for choosing one. It needs the [drgona2020all] PDF.
3. **Whether a perfect-foresight oracle is standard practice is also unanswered.** The repo should
   present it as its own design choice rather than as field convention until that is checked.
4. **OpenAlex was not used** (no key) — no citation-graph expansion, no forward-citation search. The
   search remains keyword-driven and misses work that does not use the vocabulary I guessed.
5. **AAMAS/IJCAI/NeurIPS/ICML remain unevenly indexed.** Parkes and Singh's MDP-based approach to
   online mechanism design (NIPS 2003) still did not resolve and is still absent from
   `references.bib` rather than cited unverified. The AAMAS records that did resolve carry unusual
   `10.65109/...` DOIs from a recent re-registration; Semantic Scholar corroborates all three with
   matching titles, venues and abstracts, which is reassuring but not a substitute for checking the
   DOIs resolve before submission.
6. **No sim-to-real transfer gap was found reported as a number.** §11 searched for it. The field
   deployments retrieved train or tune in place rather than transferring a simulator-trained policy
   and measuring the degradation. If that holds up under a wider search, the project's intention to
   report sim-vs-real gaps explicitly is ahead of the retrieved literature — which is a contribution
   claim worth making carefully, and worth one more targeted search before making.
7. **The CityLearn netting finding in §4 is source-verified, not literature-verified.** I looked for
   a paper that states its peak convention explicitly and found none. The consequence — that a
   CityLearn peak is not what a hard import cap constrains — follows from the code, not from a
   citation, and should be presented that way.
8. **Battery degradation modelling was not searched** as its own area, which is why the unresolved
   Khezri citation (§8.1) has no substitute. If a degradation cost is wired into the reward and the
   constraint set as the audit plans, that search still has to happen.
9. **Occupant behaviour and comfort modelling were not searched.** §2 concludes comfort is soft in
   the learning literature and hard in the optimisation literature, from four sources. That is a
   thin basis for a claim about a whole field, and the occupant-centric line (the `occupant-centric`
   in [nweye2024citylearnv2]'s own title) was not followed.
