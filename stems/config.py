from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: The three constraint channels the cost critics have always carried, in order. Every
#: one is a **per-step 0/1 indicator**, so the quantity the critics estimate is the
#: discounted expected *fraction of steps that violate*, and the quantity the PID
#: controller compares against `cost_limit` is the undiscounted mean of those
#: indicators over the batch. See CHANGELOG.md step 3 for why that definition has to be
#: stated: the constrained-RL literature uses at least three non-interconvertible
#: violation rates and does not agree on one.
BASE_CONSTRAINT_CHANNELS: Tuple[str, ...] = (
    "soc_band",             # 1{next state of charge outside [SOC_min, SOC_max]}
    "building_power_cap",   # 1{|net electricity consumption| > P_building_max}
    "district_import_cap",  # 1{sum_b max(net_b, 0) > P_grid_max}, same for every b
)

#: Appended when degradation is enforced rather than only reported.
DEGRADATION_CONSTRAINT_CHANNEL = "battery_degradation"


def constraint_channel_names(config: "STEMSConfig") -> Tuple[str, ...]:
    """The cost channels this configuration actually produces, in buffer order.

    One source of truth for the trajectory collector, the cost critics' output width
    and the reporting, so they cannot disagree about which column means what.
    """
    names = list(BASE_CONSTRAINT_CHANNELS)
    if (config.degradation.mode != "none"
            and config.degradation.limit_kwh_per_episode is not None):
        names.append(DEGRADATION_CONSTRAINT_CHANNEL)
    return tuple(names)


@dataclass
class GraphConfig:
    """Adjacency of the building graph the GCN mixes over.

    ``mode``:
      ``"feature"``            w_ij = exp(-||f_i - f_j||^2 / (2 sigma_f^2)), the default.
      ``"feature+position"``   the original alpha/beta mix of a positional and a feature
                               kernel. Needs real coordinates, which CityLearn does not
                               provide; see CHANGELOG.md step 6.
      ``"mean_pool"``          w_ij = 1 for i != j. Uniform pooling, the ablation the GCN
                               has to beat.

    ``sigma_f = None`` selects the median heuristic: sigma_f is set so that the median
    off-diagonal squared feature distance maps to exp(-1) ~ 0.368. A fixed bandwidth has
    no meaning independent of how the features are scaled, which is how the original
    sigma_d = 1.0 came to produce edge weights spanning 2.4e-3.
    """

    mode: str = "feature"
    alpha: float = 0.5
    beta: float = 0.5
    sigma_d: float = 1.0
    sigma_f: Optional[float] = None


@dataclass
class GCNConfig:
    num_layers: int = 3
    hidden_dim: int = 64


@dataclass
class TransformerConfig:
    num_heads: int = 4
    embed_dim: int = 32
    window_size: int = 24


@dataclass
class FusionConfig:
    output_dim: int = 64


@dataclass
class ActorCriticConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    share_parameters: bool = False


@dataclass
class RewardConfig:
    mu: float = 1.0
    alpha_grid: float = 0.5
    alpha_build: float = 0.3
    beta_ramp: float = 0.2
    lambda_indoor: float = 0.4
    xi: float = 0.0
    T_ref: float = 22.0
    T_comfort_threshold: float = 2.0

    ev_service: float = 25.0
    ev_shaping: float = 0.5


@dataclass
class CBFConfig:
    SOC_min: float = 0.1
    SOC_max: float = 0.9
    P_grid_max: float = 300.0
    P_building_max: float = 80.0
    gamma_cbf: float = 1.0


@dataclass
class SafetyConfig:
    feasibility_qp: bool = True
    anticipatory: bool = True
    invariance_horizon: int = 1
    robust_margins: bool = True
    soc_margin: float = 0.03
    soc_tolerance: float = 1e-3
    power_derate: float = 0.05


@dataclass
class TrainingConfig:
    episodes: int = 50
    batch_size: int = 512
    buffer_capacity: int = 100_000
    exploration_noise: float = 0.1
    action_scale: float = 1.0

    update_epochs: int = 10
    minibatch_size: int = 64
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    entropy_coef: float = 0.01
    scale_rewards: bool = True

    actor_target: str = "raw"

    residual_scale: float = 0.5
    residual_log_std: float = -1.2

    intervention_penalty: float = 0.0

    forced_charge_penalty: float = 0.0


@dataclass
class LagrangianConfig:
    """The constrained-policy-optimisation mechanism.

    ``enabled=False`` leaves the cost critics training -- so every arm measures the
    violation rate with the same instrument -- but feeds nothing from them into the
    actor. That is the "no mechanism" cell of the constraint-mechanism cross
    (audit C2): it is not the same as removing the cost critics, and must not be.

    ``algorithm`` selects the family:

    ``"ppo-lagrangian"``
        The pre-existing mechanism. Clipped PPO surrogate on the combined advantage
        ``A_eff = (A - sum_k lambda_k A^c_k) / (1 + sum_k lambda_k)``, with lambda
        driven by a PID controller on the constraint error.

    ``"focops"``
        First Order Constrained Optimization in Policy Space (Zhang, Vuong and Ross,
        NeurIPS 2020). Same cost critics, same generalised-advantage estimates, same
        encoder and the same multiplier state -- only the actor surrogate and the
        multiplier update change, which is what makes the two comparable. The surrogate
        is a KL-regularised first-order projection towards the non-parametric optimum
        rather than a clipped ratio; see ``stems/agent.py::STEMSAgent._focops_loss``.

    ``focops_temperature`` is the paper's ``lambda`` (the inverse weight on the KL term;
    not the constraint multiplier, which is called ``nu`` there and ``_lambdas`` here).
    ``focops_kl_limit`` is the trust region ``delta`` outside which a sample is dropped
    from the surrogate; it defaults to ``TrainingConfig.target_kl`` so the two families
    use trust regions of the same size and the comparison is not confounded by it.
    """

    num_constraints: int = 3
    cost_limit: float = 0.05
    lambda_init: float = 0.1
    lambda_max: float = 10.0

    lambda_lr: float = 0.005

    use_pid: bool = True
    pid_kp: float = 1.0
    pid_ki: float = 0.3
    pid_kd: float = 0.0

    enabled: bool = True
    algorithm: str = "ppo-lagrangian"
    focops_temperature: float = 1.5
    focops_kl_limit: Optional[float] = None


@dataclass
class HeatPumpConfig:
    enabled: bool = False
    cop_heating_mild: float = 3.5
    cop_heating_cold: float = 2.0
    cop_cooling: float = 4.0
    cold_snap_temp: float = 0.0
    rated_power_kw: float = 5.0
    heating_setpoint: float = 20.0
    cooling_setpoint: float = 22.0


@dataclass
class ThermalConfig:
    dhw_readiness: bool = True
    preheat_horizon: int = 2
    dhw_margin: float = 0.05
    dhw_soc_cap: float = 0.95

    forecast_alpha: float = 0.2
    forecast_warmup: int = 24
    forecast_temp_gain: float = 0.02

    weather_anticipation: bool = True
    weather_gain: float = 0.15
    weather_horizon: int = 3

    cop_aware_power: bool = True


@dataclass
class ComfortConfig:
    """Thermal comfort as a hard constraint. OFF by default; see `stems/comfort.py`.

    ``enabled`` is ``False`` on purpose and must stay that way on CityLearn: its learned
    temperature model drops a house 4-13 degC in an hour under a -0.25 cooling action
    and one house does not respond to cooling at all (`docs/REPORT_2026-10.md` section
    2), so a band enforced against it is satisfied in simulation and meaningless in
    reality. The barrier is built, tested and ready for the day an RC model fitted to
    real measurements exists.

    ``tolerance_k`` (theta, K) is the slack either side of the simulator's set-point
    band. 2.0 K is the threshold `stems/metrics.py` already uses for
    ``discomfort_rate``, so the constraint and the KPI measure the same band.

    ``design_t_out_heating_c`` / ``design_t_out_cooling_c`` (degC) are the outdoor
    temperatures at which the available heat-pump thermal power is evaluated. The
    barrier needs a bound that holds at every step, so it uses the coefficient of
    performance at the design condition rather than the instantaneous one.
    """

    enabled: bool = False
    tolerance_k: float = 2.0
    occupied_only: bool = True
    enforce_cooling: bool = True
    design_t_out_heating_c: float = -10.0
    design_t_out_cooling_c: float = 35.0


@dataclass
class DegradationConfig:
    """Battery and electric-vehicle degradation as a cost and as a constraint.

    See `stems/degradation.py` for the model and the provenance of every parameter.
    ``mode`` is ``"none"``, ``"throughput"`` or ``"throughput+dod"``; ``calendar`` adds
    the time term. ``price_eur_per_kwh`` prices one kWh of *lost storage capacity* and
    has no default that could be mistaken for a measurement -- it must be set by the
    caller from a quoted replacement cost, and is 0.0 (degradation reported, not
    priced) until then.

    ``limit_kwh_per_episode`` is the per-building capacity-loss budget an episode may
    spend; ``None`` means the degradation channel is reported but not constrained.
    """

    mode: str = "none"
    calendar: bool = False
    dod_exponent: float = 0.0
    #: Range gate on the half-cycle detector, state-of-charge units; see
    #: `stems/degradation.py::DegradationAccountant._accrue_half_cycle`.
    reversal_threshold: float = 0.05
    price_eur_per_kwh: float = 0.0
    limit_kwh_per_episode: Optional[float] = None
    #: Fallback capacity-loss coefficient, dimensionless, used only when the simulator
    #: does not expose one. CityLearn's own default range is (1e-5, 1e-4).
    fallback_capacity_loss_coefficient: float = 1e-5
    #: Calendar ageing, fraction of rated capacity lost per second of elapsed time.
    #: No default: must be supplied from a measurement or a datasheet.
    calendar_loss_per_second: Optional[float] = None


@dataclass
class STEMSConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    gcn: GCNConfig = field(default_factory=GCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lagrangian: LagrangianConfig = field(default_factory=LagrangianConfig)
    heat_pump: HeatPumpConfig = field(default_factory=HeatPumpConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    def tricks_active(self) -> List[str]:
        active = ["feasibility_qp", "real_battery_dynamics"]
        if self.safety.anticipatory:
            active.append("anticipatory")
        if self.safety.robust_margins:
            active.append("robust_margins")
        if self.lagrangian.use_pid:
            active.append("pid_lagrangian")
        if self.thermal.dhw_readiness:
            active.append(f"dhw_readiness(L={self.thermal.preheat_horizon})")
        if self.thermal.weather_anticipation:
            active.append("weather_anticipation")
        if self.thermal.cop_aware_power:
            active.append("cop_aware_power")
        if self.comfort.enabled:
            active.append(f"hard_comfort(theta={self.comfort.tolerance_k}K)")
        if self.degradation.mode != "none":
            active.append(f"degradation({self.degradation.mode}"
                          + (", constrained" if self.degradation.limit_kwh_per_episode
                             is not None else ", reported") + ")")
        return active
