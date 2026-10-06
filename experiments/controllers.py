from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from stems.baselines import RuleBasedAgent

import numpy as np

UNCALIBRATED_SOC_RATE = 0.1

RESERVE_HOURS = 1
LEAD_MARGIN = True

RBC_ACTIONS = ["dhw_storage", "electrical_storage", "cooling_or_heating_device"]

#: Comparison controllers: the seven methods of the STEMS paper's Table I plus MAPPO
#: with a genuinely centralised critic. Each value says what the arm *is* now -- the
#: reference implementation it follows -- not what remains to be built. Built by
#: `build_controller`; see CHANGELOG.md steps 1-3 and `stems/baselines.py`.
COMPARISON_POLICIES: Dict[str, str] = {
    "sac": "single-agent soft actor-critic over the concatenated district observation "
           "and the concatenated joint action, with twin target critics, a replay "
           "buffer and automatic entropy tuning (stems.baselines.SingleAgentSAC)",
    "dmappo": "independent PPO per building on local observations -- decentralised "
              "training, decentralised execution -- with GAE, a clipped value loss, an "
              "entropy bonus and a fixed behaviour log-probability "
              "(stems.baselines.DMAPPOAgent)",
    "mpc": "receding-horizon economic model-predictive control over the exact battery "
           "and tank inverses, with the forecast the RL agent sees "
           "(stems.mpc.StorageMPC + stems.forecast.CausalForecaster)",
    "mpc-oracle": "the same MPC with perfect foresight of price, hot-water draw and the "
                  "idle-counterfactual base load -- the upper reference "
                  "(stems.mpc.StorageMPC + stems.forecast.OracleForecaster)",
    "maddpg": "MADDPG: deterministic actors on local observations, a per-agent critic "
              "on the joint observation and joint action, replay and target networks "
              "(stems.baselines.MADDPGAgent)",
    "marlisa": "MARLISA with measured sharing: soft actor-critic with iterative "
               "sequential action selection, the augmentation stored at collection "
               "time so the critic sees what the actor saw (stems.baselines.MARLISAAgent)",
    "madcq": "constrained Q-learning over a branched discrete action set, the safe set "
             "taken from BatteryModel.safe_interval and applied inside the bootstrap "
             "target as well as the greedy policy (stems.baselines.MADCQAgent)",
    "metaems": "Reptile meta-learning on a shared soft actor-critic with buildings as "
               "tasks (stems.baselines.MetaEMSAgent)",
    "mappo-cc": "MAPPO with a critic conditioned on the joint encoded observation -- "
                "the CTDE comparator that D-MAPPO does not provide, and the arm that "
                "answers audit D1 (stems.mappo.MAPPOCentralisedCritic)",
}

#: Comparison controllers that take gradient steps. The MPC pair solves an optimisation
#: at every control step and has no parameters to fit, so it gets no training budget.
LEARNING_COMPARISON_POLICIES = frozenset(
    set(COMPARISON_POLICIES) - {"mpc", "mpc-oracle"})

#: Policies whose builder exists today and that take gradient steps.
LEARNING_POLICIES = frozenset({"rl"}) | LEARNING_COMPARISON_POLICIES

#: Policies registered as names only. `build_controller` raises NotImplementedError for
#: each of them, with the reference the implementer needs. Registered so a downstream
#: track adds a builder rather than restructuring ARMS, and so that
#: `experiments/ablation.py --arms <name>` fails loudly at build time instead of with a
#: KeyError that reads like a typo. See CHANGELOG.md step 8. The eight comparison
#: controllers that used to be listed here now have builders and have moved to
#: COMPARISON_POLICIES; what remains is reserved for the constraint track, which is why
#: this dict is empty rather than deleted.
PRE_REGISTERED_POLICIES: Dict[str, str] = {}

#: Constraint mechanisms. "auto" is the historical behaviour; the rest are the clean 2x2
#: of audit C2 and have no builder yet.
MECHANISMS = ("auto", "none", "lagrangian", "projection", "both")

#: Battery plant models the projection can invert against, as named by `Arm.barrier`.
PLANT_MODELS = {"uniform": "basic", "linear": "linear", "exact": "calibrated"}


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str
    barrier: str
    residual: bool = False
    penalty: float = 0.0
    ev_request: str = "asap"
    forced_penalty: float = 0.0
    ev_floor: float = 0.0
    control: Optional[Tuple[str, ...]] = None
    #: Adjacency the GCN mixes over: "feature" (the default, built from real per-building
    #: device characteristics) or "mean_pool" (uniform, the ablation the GCN must beat).
    graph_mode: str = "feature"
    #: Which constraint mechanism is active, for the clean 2x2 of audit C2.
    #: "auto" reproduces the historical behaviour -- the Lagrangian runs whenever the
    #: policy learns, and the projection runs whenever `barrier != "none"` -- and is what
    #: every pre-existing arm uses. The explicit values isolate one mechanism at a time:
    #: "none", "lagrangian", "projection", "both".
    mechanism: str = "auto"
    #: Comfort as a hard constraint via a thermal deadline barrier, rather than the
    #: quadratic reward penalty (audit C1). Not implemented; see CHANGELOG.md step 8.
    comfort_barrier: bool = False
    #: Battery and EV degradation as a cost and a constraint (audit C1):
    #: "none", "throughput", "throughput+dod".
    degradation: str = "none"

    @property
    def learns(self) -> bool:
        return self.policy in LEARNING_POLICIES

    @property
    def plant_model(self) -> Optional[str]:
        """Which battery model the projection inverts against, derived from `barrier`.

        One source of truth: `barrier` already encodes it, so this is a reading of that
        field rather than a second field that could disagree with it.
        """
        return {"none": None, "basic": "uniform", "linear": "linear",
                "calibrated": "exact"}[self.barrier]

    @property
    def implemented(self) -> bool:
        """False for an arm that is registered as a name but has no builder yet."""
        return (self.policy not in PRE_REGISTERED_POLICIES
                and self.mechanism == "auto"
                and not self.comfort_barrier
                and self.degradation == "none")

    @property
    def is_comparison(self) -> bool:
        """True for the nine comparison controllers of CHANGELOG.md steps 1-3."""
        return self.policy in COMPARISON_POLICIES


ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("idle", "idle", "none"),
    Arm("idle+calibrated", "idle", "calibrated"),
    Arm("rbc", "rbc", "none"),
    Arm("rbc+calibrated", "rbc", "calibrated"),
    Arm("rl", "rl", "none"),
    Arm("rl+basic", "rl", "basic"),
    Arm("rl+linear", "rl", "linear"),
    Arm("rl+calibrated", "rl", "calibrated"),
    Arm("rl-res+calibrated", "rl", "calibrated", residual=True),
    Arm("rl+calibrated+pen", "rl", "calibrated", penalty=1.0),
    Arm("rbc-offpeak+calibrated", "rbc", "calibrated", ev_request="offpeak"),
    Arm("rbc-never+calibrated", "rbc", "calibrated", ev_request="never"),
    Arm("rl+calibrated+own", "rl", "calibrated", forced_penalty=0.22),
    Arm("rl+calibrated+floor", "rl", "calibrated", ev_floor=0.5),
    Arm("hp-shift", "hp-shift", "none"),
    Arm("rl-hp", "rl", "none", control=("cooling_or_heating_device",)),
    # Audit B6: with the fabricated positional term gone, the GCN has to earn its
    # place on features alone. This arm is identical to rl+calibrated except that
    # every edge weight is 1, i.e. the encoder mean-pools over buildings.
    Arm("rl+calibrated+meanpool", "rl", "calibrated", graph_mode="mean_pool"),

    # ---------------------------------------------------------------------------
    # Pre-registered below: names reserved so the baseline and constraint tracks add
    # a builder rather than restructuring this dict. `build_controller` raises
    # NotImplementedError for every one of them. See CHANGELOG.md step 8.
    # ---------------------------------------------------------------------------

    # (a) The comparison controllers: the seven STEMS Table I methods, plus MAPPO with
    #     a genuine centralised critic (audit D1), plus the perfect-foresight MPC that
    #     pairs with the causal one. Every one carries barrier="calibrated", so each
    #     sees the identical exact-inverse safety layer and a difference between two of
    #     them is a difference between algorithms. CHANGELOG.md steps 1-3.
    *(Arm(name, policy, "calibrated") for name, policy in (
        ("sac", "sac"),
        ("dmappo", "dmappo"),
        ("mpc", "mpc"),
        ("mpc-oracle", "mpc-oracle"),
        ("maddpg", "maddpg"),
        ("marlisa", "marlisa"),
        ("madcq", "madcq"),
        ("metaems", "metaems"),
        ("mappo-cc", "mappo-cc"),
    )),

    # (b) The constraint-mechanism 2x2 of audit C2, crossed with the battery plant
    #     model. `barrier` carries the plant model (basic = uniform rate, linear =
    #     linear from the exact parameters, calibrated = the exact inverse), and
    #     `mechanism` says which mechanism is switched on. Twelve arms; the four with
    #     mechanism="none" differ only in a projection that is switched off, and are
    #     kept so the cross is complete and the grid driver does not need a special case.
    *(Arm(f"mech-{mech}+{plant}", "rl", PLANT_MODELS[plant], mechanism=mech)
      for mech in ("none", "lagrangian", "projection", "both")
      for plant in ("uniform", "linear", "exact")),

    # (c) Comfort as a hard constraint (audit C1), against the soft reward penalty that
    #     is the current behaviour. Both a learning and a rule arm, so the barrier can
    #     be attributed separately from the policy.
    Arm("rl+calibrated+comfort", "rl", "calibrated", comfort_barrier=True),
    Arm("rbc+calibrated+comfort", "rbc", "calibrated", comfort_barrier=True),

    # (d) Degradation as a cost and a constraint (audit C1), which the EV and V2G story
    #     needs and which is currently a KPI only.
    Arm("rl+calibrated+degr-throughput", "rl", "calibrated", degradation="throughput"),
    Arm("rl+calibrated+degr-dod", "rl", "calibrated", degradation="throughput+dod"),
)}

#: Arms registered as names with no builder yet. Computed, not hand-maintained.
PRE_REGISTERED_ARMS = tuple(n for n, a in ARMS.items() if not a.implemented)


class RuleColumns:
    def __init__(self, rule, names) -> None:
        self.rule = rule
        self.columns = [RBC_ACTIONS.index(n) for n in names]

    def reset(self) -> None:
        self.rule.reset()

    def notify_executed(self, executed: np.ndarray) -> None:
        return None

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        return self.rule.select_action(obs_list, history, explore)[:, self.columns]


class EVRule:
    def __init__(self, house, action_dim: int, ev_index: int, layout: Dict[str, int],
                 ev_request: str = "asap") -> None:
        if ev_request not in ("asap", "offpeak", "never"):
            raise ValueError(f"unknown ev_request {ev_request!r}")
        self.house, self.action_dim, self.ev_index, self.layout = house, action_dim, ev_index, layout
        self.ev_request = ev_request

    def reset(self) -> None:
        if hasattr(self.house, "reset"):
            self.house.reset()

    def notify_executed(self, executed: np.ndarray) -> None:
        if hasattr(self.house, "notify_executed"):
            self.house.notify_executed(executed)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        a = np.zeros((len(obs_list), self.action_dim), dtype=np.float32)
        if self.house is not None:
            a[:, :3] = self.house.select_action(obs_list, history, explore)
        col = lambda key: np.array([float(o[self.layout[key]]) for o in obs_list])
        short = (col("connected_state") > 0.5) & (col("soc") < col("required_soc_departure"))
        hour = int(round(float(obs_list[0][1])))
        if self.ev_request == "never" or (self.ev_request == "offpeak"
                                          and hour in RuleBasedAgent.PEAK_HOURS):
            short = np.zeros_like(short)
        a[:, self.ev_index] = np.where(short, 1.0, 0.0)
        return a


class ChargerFloor:
    def __init__(self, action_dim: int, ev_index: int, layout: Dict[str, int],
                 fraction: float) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.rule = EVRule(None, action_dim, ev_index, layout, ev_request="offpeak")
        self.ev_index, self.fraction = ev_index, float(fraction)

    def __call__(self, obs_list) -> np.ndarray:
        floor = np.full((len(obs_list), self.rule.action_dim), -1.0, dtype=np.float32)
        floor[:, self.ev_index] = self.fraction * self.rule.select_action(obs_list)[:, self.ev_index]
        return floor


class SetpointShiftPolicy:
    PREP_HOURS = range(13, 17)

    def __init__(self, env) -> None:
        if env.hvac_control != "setpoint":
            raise RuntimeError("SetpointShiftPolicy needs hvac_control='setpoint'")
        self.env = env

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        env = self.env
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        hour = int(round(float(obs_list[0][1])))
        heating = env.executed_actions[:, env.hvac_action_index] >= 0.0
        if hour in self.PREP_HOURS:
            a[:, env.hvac_action_index] = np.where(heating, 1.0, -1.0)
        elif hour in RuleBasedAgent.PEAK_HOURS:
            a[:, env.hvac_action_index] = np.where(heating, -1.0, 1.0)
        return a


class IdlePolicy:
    def __init__(self, num_buildings: int, action_dim: int) -> None:
        self.shape = (num_buildings, action_dim)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        return np.zeros(self.shape, dtype=np.float32)


class PlainController:
    def __init__(self, base) -> None:
        self.base = base
        self._last_raw_actions: Optional[np.ndarray] = None
        self._last_safe_actions: Optional[np.ndarray] = None

    def _base_action(self, obs_list, history, explore) -> np.ndarray:
        a = np.asarray(self.base.select_action(obs_list, history, explore), dtype=np.float32)
        return np.clip(a, -1.0, 1.0)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        a = self._base_action(obs_list, history, explore)
        self._last_raw_actions, self._last_safe_actions = a.copy(), a.copy()
        return a

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        return None

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class ShieldedController(PlainController):
    def __init__(self, base, shield, dhw_barrier=None, fleet_shield=None) -> None:
        super().__init__(base)
        self.shield = shield
        self.dhw_barrier = dhw_barrier
        self.fleet_shield = fleet_shield

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        raw = self._base_action(obs_list, history, explore)
        safe = np.clip(self.shield.project(raw, obs_list), -1.0, 1.0).astype(np.float32)
        if self.fleet_shield is not None:
            safe = self.fleet_shield.project(safe, obs_list)
        self._last_raw_actions, self._last_safe_actions = raw.copy(), safe.copy()
        if hasattr(self.base, "notify_executed"):
            self.base.notify_executed(safe)
        return safe

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        forecaster = getattr(self.dhw_barrier, "forecaster", None)
        if forecaster is not None:
            forecaster.update(next_obs_list)
        if self.fleet_shield is not None and ev_draw_kwh is not None:
            self.fleet_shield.observe(next_obs_list, ev_draw_kwh)


class ComparisonController(PlainController):
    """Adapter: a comparison controller behind the interface the runner trains against.

    ``experiments/runner.py::train`` was written against ``stems.agent.STEMSAgent`` and
    reads ``elec_idx``, ``control_indices``, ``_last_raw_actions``, ``_last_safe_actions``,
    ``_last_pre_tanh``, ``_last_log_probs`` and ``_last_nominal_actions`` off the agent
    it is given. Rather than special-case nine controllers in the loop, this presents
    those attributes for all of them and forwards ``update``/``save``/``load``.

    It also applies the arm's safety layer, so every comparison arm goes through the
    *same* projection the reinforcement-learning arms do and the arms stay comparable.
    For the model-predictive pair the projection is expected to be nearly inactive --
    they enforce the same bounds inside their own optimisation -- and how often it does
    intervene is a reported quantity, not an assumption.

    ``_last_pre_tanh`` and ``_last_log_probs`` come from the wrapped learner when it
    exposes them (the policy-gradient arms do, because an importance ratio needs the
    behaviour density). For the off-policy and discrete arms they are the ``atanh`` of
    the action and zero: those learners never read them, and the fields exist only
    because the collector writes them unconditionally.
    """

    def __init__(self, base, shield=None, fleet_shield=None, elec_idx: int = 1,
                 action_dim: int = 3, num_buildings: int = 1,
                 control_indices: Optional[list] = None) -> None:
        super().__init__(base)
        self.shield = shield
        self.fleet_shield = fleet_shield
        self.elec_idx = int(elec_idx)
        self.action_dim = int(action_dim)
        self.B = int(num_buildings)
        self.control_indices = (list(range(self.action_dim)) if control_indices is None
                                else list(control_indices))
        self.base_policy = None
        self.request_floor = None
        z = np.zeros((self.B, self.action_dim), dtype=np.float32)
        self._last_raw_actions = z.copy()
        self._last_nominal_actions = z.copy()
        self._last_safe_actions = z.copy()
        self._last_pre_tanh = z.copy()
        self._last_log_probs = np.zeros(self.B, dtype=np.float32)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        raw = self._base_action(obs_list, history, explore)
        self._last_raw_actions = raw.copy()
        self._last_nominal_actions = raw.copy()
        safe = raw if self.shield is None else np.clip(
            self.shield.project(raw, obs_list), -1.0, 1.0).astype(np.float32)
        if self.fleet_shield is not None:
            safe = self.fleet_shield.project(safe, obs_list)
        self._last_safe_actions = safe.copy()
        # Copies, not references. The collector stores whatever object it is handed and
        # keeps it for the whole episode, so passing a buffer a learner reuses would
        # make every stored row alias the last step.
        self._last_pre_tanh = np.array(
            getattr(self.base, "_last_pre_tanh",
                    np.arctanh(np.clip(raw, -1.0 + 1e-6, 1.0 - 1e-6))),
            dtype=np.float32, copy=True)
        self._last_log_probs = np.array(
            getattr(self.base, "_last_log_probs", np.zeros(self.B)),
            dtype=np.float32, copy=True)
        if hasattr(self.base, "notify_executed"):
            self.base.notify_executed(safe)
        return safe

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        if hasattr(self.base, "observe"):
            self.base.observe(next_obs_list, ev_draw_kwh)
        if self.fleet_shield is not None and ev_draw_kwh is not None:
            self.fleet_shield.observe(next_obs_list, ev_draw_kwh)

    def update(self, batch):
        return self.base.update(batch) if hasattr(self.base, "update") else {}

    def save(self, path: str) -> None:
        if hasattr(self.base, "save"):
            self.base.save(path)

    def load(self, path: str) -> None:
        if hasattr(self.base, "load"):
            self.base.load(path)


def safety_layer(barrier: str, env):
    from stems.battery import BatteryModel
    from stems.config import SafetyConfig

    plain = SafetyConfig(anticipatory=False, robust_margins=False)
    if barrier == "none":
        return plain, None
    if barrier == "basic":
        return plain, BatteryModel.linear(np.full(env.num_buildings, UNCALIBRATED_SOC_RATE))
    if barrier == "linear":
        exact = env.battery_model()
        return plain, BatteryModel.linear(exact.nominal_power * exact.dt / exact.capacity)
    if barrier == "calibrated":
        return plain, env.battery_model()
    raise ValueError(f"unknown barrier {barrier!r}")


def _refuse_unimplemented(arm: Arm) -> None:
    """Fail loudly and usefully for an arm that is a reserved name, not a controller.

    A pre-registered arm must not quietly fall through to some other branch and produce
    a run record that looks like a result. Each message names what has to be built and
    where the relevant code or audit finding is.
    """
    if arm.policy in PRE_REGISTERED_POLICIES:
        raise NotImplementedError(
            f"arm {arm.name!r} is pre-registered, not implemented. Its policy "
            f"{arm.policy!r} needs a builder: {PRE_REGISTERED_POLICIES[arm.policy]}. "
            "Add the branch to experiments/controllers.py::build_controller; the arm "
            "name and its fields are already fixed so nothing downstream has to change.")
    if arm.mechanism != "auto":
        raise NotImplementedError(
            f"arm {arm.name!r} is pre-registered, not implemented. It asks for "
            f"mechanism={arm.mechanism!r} with the {arm.plant_model!r} battery model, "
            "which is the constraint-mechanism 2x2 of audit C2. build_controller "
            "currently couples the two: the Lagrangian runs whenever the policy learns "
            "and the projection runs whenever barrier != 'none', so neither can be "
            "switched off independently. Decoupling them is the work this arm names.")
    if arm.comfort_barrier:
        raise NotImplementedError(
            f"arm {arm.name!r} is pre-registered, not implemented. Hard comfort (audit "
            "C1) needs a thermal deadline barrier built on "
            "stems.deadline.DeadlineStorageBarrier, with the soft reward penalty kept "
            "as the comparison arm. Note that enforcing a hard band against CityLearn's "
            "learned temperature model gives a constraint satisfied in simulation and "
            "meaningless in reality (REPORT section 2); the claim belongs on an RC model "
            "fitted to real data.")
    if arm.degradation != "none":
        raise NotImplementedError(
            f"arm {arm.name!r} is pre-registered, not implemented. It asks for "
            f"degradation={arm.degradation!r} (audit C1): throughput and depth-of-"
            "discharge ageing wired as both a reward term and a constraint. Today "
            "degradation is counted as the KPI battery_equivalent_full_cycles and "
            "nothing acts on it.")


#: Episode length used to record the perfect-foresight tape when the caller does not
#: say. 8760 covers a full year at one hour per step, which is the longest window
#: `experiments/scenario.py` can ask for.
ORACLE_TAPE_STEPS = 8760


def _build_comparison(arm: Arm, env, config, battery_model, battery, ev_indices, fleet):
    """One of the nine comparison controllers, wrapped in the arm's safety layer."""
    from stems.cbf import CBFShield

    B, A = env.num_buildings, env.action_dim
    obs_dim = env.obs_dim
    shield = None
    if arm.barrier != "none":
        shield = CBFShield(config.cbf, B, battery_model=battery_model,
                           nominal_power=battery["nominal_power"],
                           action_scale=config.training.action_scale,
                           elec_idx=env.electrical_storage_action_index,
                           safety_cfg=config.safety, enforce_soc=True,
                           hvac_idx=env.hvac_action_index)

    if arm.policy in ("mpc", "mpc-oracle"):
        base = _build_mpc(arm, env, config, shield)
    elif arm.policy == "sac":
        from stems.baselines import SingleAgentSAC
        base = SingleAgentSAC(obs_dim, A, num_buildings=B)
    elif arm.policy == "dmappo":
        from stems.baselines import DMAPPOAgent
        base = DMAPPOAgent(obs_dim, A, num_buildings=B,
                           gae_lambda=config.training.gae_lambda,
                           clip_eps=config.training.ppo_clip,
                           ppo_epochs=config.training.update_epochs,
                           minibatch_size=config.training.minibatch_size,
                           entropy_coef=config.training.entropy_coef,
                           target_kl=config.training.target_kl,
                           gamma=config.actor_critic.gamma)
    elif arm.policy == "maddpg":
        from stems.baselines import MADDPGAgent
        base = MADDPGAgent(obs_dim, A, num_buildings=B, gamma=config.actor_critic.gamma)
    elif arm.policy == "marlisa":
        from stems.baselines import MARLISAAgent
        base = MARLISAAgent(obs_dim, A, num_buildings=B, gamma=config.actor_critic.gamma)
    elif arm.policy == "madcq":
        from stems.baselines import MADCQAgent
        base = MADCQAgent(obs_dim, A, num_buildings=B, gamma=config.actor_critic.gamma,
                          soc_min=config.cbf.SOC_min, soc_max=config.cbf.SOC_max,
                          battery_model=env.battery_model(),
                          elec_idx=env.electrical_storage_action_index)
    elif arm.policy == "metaems":
        from stems.baselines import MetaEMSAgent
        base = MetaEMSAgent(obs_dim, A, num_buildings=B, gamma=config.actor_critic.gamma)
    elif arm.policy == "mappo-cc":
        from stems.graph import BuildingGraph
        from stems.mappo import MAPPOCentralisedCritic
        info = env.get_building_info()
        config.graph.mode = arm.graph_mode
        graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
        base = MAPPOCentralisedCritic(
            obs_dim, A, B, graph, config=config,
            electrical_storage_action_index=env.electrical_storage_action_index)
    else:
        raise ValueError(f"unknown comparison policy {arm.policy!r}")

    controller = ComparisonController(
        base, shield=shield, fleet_shield=fleet(shield) if shield is not None else None,
        elec_idx=env.electrical_storage_action_index, action_dim=A, num_buildings=B)
    return controller


def _build_mpc(arm: Arm, env, config, shield):
    """The receding-horizon or the perfect-foresight model-predictive controller.

    The two differ only in the forecaster. The oracle's tape is recorded by rolling the
    *same* environment once with the zero action and then resetting it, which is sound
    because ``STEMSEnvironment.reset`` restarts the CityLearn episode from the window's
    first step; ``tests/test_mpc.py::test_resetting_after_the_oracle_tape_restores_the_environment``
    pins that the observations after the reset are identical to the ones before the
    recording, so the tape costs nothing but time.
    """
    from stems.forecast import CausalForecaster, record_idle_episode
    from stems.mpc import DEFAULT_HORIZON_H, StorageMPC

    B = env.num_buildings
    tank = None
    if env.dhw_action_index >= 0 and not env.using_mock:
        tank = env.dhw_tank_model()
    battery = env.battery_model()
    if arm.policy == "mpc-oracle":
        (start, end), = env.env_kwargs.get("episode_time_steps", [(0, ORACLE_TAPE_STEPS - 1)])
        forecaster = record_idle_episode(env, int(end - start + 1))
        env.reset()
    else:
        forecaster = CausalForecaster(B, battery, tank,
                                      env.electrical_storage_action_index,
                                      env.dhw_action_index)
    soc_lo, soc_hi = ((config.cbf.SOC_min, config.cbf.SOC_max) if shield is None
                      else (float(np.max(shield.enforced_soc_bounds()[0])),
                            float(np.min(shield.enforced_soc_bounds()[1]))))
    return StorageMPC(
        B, env.action_dim, battery, tank, forecaster,
        elec_idx=env.electrical_storage_action_index,
        dhw_idx=env.dhw_action_index, hvac_idx=env.hvac_action_index,
        hvac_control=env.hvac_control, horizon=DEFAULT_HORIZON_H,
        soc_min=soc_lo, soc_max=soc_hi,
        dhw_soc_cap=config.thermal.dhw_soc_cap,
        dhw_action_bound=env.dhw_info()["action_bound"],
        p_grid_max=config.cbf.P_grid_max, p_building_max=config.cbf.P_building_max)


def build_controller(arm: Arm, env, config):
    from stems.agent import STEMSAgent
    from stems.cbf import CBFShield
    from stems.graph import BuildingGraph

    _refuse_unimplemented(arm)

    B = env.num_buildings
    safety, battery_model = safety_layer(arm.barrier, env)
    config.safety = safety
    battery = env.battery_info()

    ev_indices = [] if env.using_mock else env.ev_action_indices()

    def rule():
        names = [n for n in env.action_names if n in RBC_ACTIONS]
        if "electrical_storage" not in names or names != [n for n in RBC_ACTIONS if n in names]:
            raise RuntimeError(f"RuleBasedAgent needs a battery and the action order of "
                               f"{RBC_ACTIONS}; this environment has {env.action_names}")
        house = RuleBasedAgent(num_buildings=B, hvac_control=env.hvac_control,
                               battery_nominal_power=battery["nominal_power"],
                               has_hvac=env.hvac_action_index >= 0)
        if names != RBC_ACTIONS:
            if ev_indices:
                raise RuntimeError("a schema with chargers needs all three house devices")
            return RuleColumns(house, names)
        if not ev_indices:
            return house
        return EVRule(house, env.action_dim, ev_indices[0], env.ev_obs_layout()[0],
                      ev_request=arm.ev_request)

    def fleet(barrier):
        if not ev_indices or arm.barrier == "none":
            return None
        from stems.fleet import BaseLoadForecaster, FleetShield, HouseStorage
        from stems.observations import obs_index

        lo, hi = barrier.enforced_soc_bounds()
        names = list(env.action_names)
        house = HouseStorage(env.battery_model(), env.electrical_storage_action_index,
                             obs_index("electrical_storage_soc"), lo, hi,
                             tank=env.dhw_tank_model() if "dhw_storage" in names else None,
                             tank_action=names.index("dhw_storage") if "dhw_storage" in names else 0)
        barrier.grid_guard = False
        return FleetShield(env.ev_fleet_model(), env.ev_obs_layout()[0], ev_indices[0],
                           config.cbf.P_grid_max, "lp",
                           BaseLoadForecaster(B, daily_pattern_days=7),
                           reserve_hours=RESERVE_HOURS, house=house, lead_margin=LEAD_MARGIN)

    if arm.policy == "rl":
        info = env.get_building_info()
        config.graph.mode = arm.graph_mode
        graph = BuildingGraph(B, info["positions"], info["features"], config.graph)
        config.training.intervention_penalty = float(arm.penalty)
        config.training.forced_charge_penalty = float(arm.forced_penalty)
        names = list(env.action_names)
        control = None if arm.control is None else [names.index(n) for n in arm.control]
        agent = STEMSAgent(env.obs_dim, env.action_dim, B, graph, config=config,
                           battery_info=battery, use_cbf=arm.barrier != "none",
                           electrical_storage_action_index=env.electrical_storage_action_index,
                           control_indices=control, hvac_action_index=env.hvac_action_index,
                           battery_model=battery_model or env.battery_model(),
                           base_policy=rule() if arm.residual else None)
        agent.fleet_shield = fleet(agent.cbf)
        if arm.ev_floor and ev_indices:
            agent.request_floor = ChargerFloor(env.action_dim, ev_indices[0],
                                               env.ev_obs_layout()[0], arm.ev_floor)
        return agent

    if arm.is_comparison:
        return _build_comparison(arm, env, config, battery_model, battery, ev_indices,
                                 fleet)

    if arm.policy == "idle":
        base = IdlePolicy(B, env.action_dim)
    elif arm.policy == "rbc":
        base = rule()
    elif arm.policy == "hp-shift":
        base = SetpointShiftPolicy(env)
    else:
        raise ValueError(f"unknown policy {arm.policy!r}")
    if arm.barrier == "none":
        controller = PlainController(base)
        if arm.policy == "hp-shift":
            controller.control_indices = [env.hvac_action_index]
        return controller
    shield = CBFShield(config.cbf, B, battery_model=battery_model,
                       nominal_power=battery["nominal_power"],
                       action_scale=config.training.action_scale,
                       elec_idx=env.electrical_storage_action_index,
                       safety_cfg=safety, enforce_soc=True, hvac_idx=env.hvac_action_index)
    return ShieldedController(base, shield, fleet_shield=fleet(shield))
