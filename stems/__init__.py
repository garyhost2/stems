from stems.config import STEMSConfig
from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.environment import STEMSEnvironment
from stems.flexibility import FlexibilityPortfolio, FlexibleLoad
from stems.protocols import (Environment, EVProvider, PlantModel, PlantProvider,
                             missing_capabilities)
from stems.replay import CSVReplayEnvironment
from stems.graph import BuildingGraph
from stems.legionella import (LegionellaCycleBarrier, LegionellaSpec,
                              LegionellaStack, ShadowTank)
from stems.encoder import STEncoder
from stems.cbf import CBFShield
from stems.reward import STEMSReward
from stems.agent import STEMSAgent
from stems.metrics import MetricsCalculator
from stems.utils import ReplayBuffer, EpisodeBuffer, HistoryBuffer
from stems.forecast import CausalForecaster, OracleForecaster
from stems.mpc import StorageMPC
from stems.mappo import MAPPOCentralisedCritic
from stems.baselines import (
    RuleBasedAgent,
    SingleAgentSAC,
    DMAPPOAgent,
    MADDPGAgent,
    MARLISAAgent,
    MADCQAgent,
    MetaEMSAgent,
)

#: Every controller exported here must also be reachable as an entry in
#: ``experiments/controllers.py::ARMS``, so that "exported but instantiated by no
#: experiment" (audit A3) cannot recur. ``tests/test_provenance.py`` checks the
#: correspondence in both directions. ``MPCAgent`` is deliberately absent: the name now
#: raises, and the controller is ``StorageMPC``.
CONTROLLERS_REACHABLE_FROM_ARMS = {
    "RuleBasedAgent": ("rbc", "rbc+calibrated"),
    "STEMSAgent": ("rl", "rl+calibrated"),
    "SingleAgentSAC": ("sac",),
    "DMAPPOAgent": ("dmappo",),
    "StorageMPC": ("mpc", "mpc-oracle"),
    "MADDPGAgent": ("maddpg",),
    "MARLISAAgent": ("marlisa",),
    "MADCQAgent": ("madcq",),
    "MetaEMSAgent": ("metaems",),
    "MAPPOCentralisedCritic": ("mappo-cc",),
}

#: The framework's public interface: the object ``docs/FRAMEWORK.md`` defines, the
#: protocols a non-CityLearn adapter implements, and nothing else. Declared as a set so
#: that ``tests/test_controller_reachability.py`` can exclude it from the
#: "every exported name is a controller with an arm" rule without a second hand-kept
#: list going stale every time the framework gains a name.
FRAMEWORK_EXPORTS = {
    "DeadlineRequirement",
    "DeadlineStorageBarrier",
    "FlexibleLoad",
    "FlexibilityPortfolio",
    "PlantModel",
    "Environment",
    "PlantProvider",
    "EVProvider",
    "CSVReplayEnvironment",
    "LegionellaCycleBarrier",
    "LegionellaSpec",
    "LegionellaStack",
    "ShadowTank",
    "missing_capabilities",
}

__all__ = [
    "STEMSConfig",
    "STEMSEnvironment",
    *sorted(FRAMEWORK_EXPORTS),
    "FRAMEWORK_EXPORTS",
    "BuildingGraph",
    "STEncoder",
    "CBFShield",
    "STEMSReward",
    "STEMSAgent",
    "MetricsCalculator",
    "ReplayBuffer",
    "EpisodeBuffer",
    "HistoryBuffer",
    "CausalForecaster",
    "OracleForecaster",
    "StorageMPC",
    "MAPPOCentralisedCritic",
    "RuleBasedAgent",
    "SingleAgentSAC",
    "DMAPPOAgent",
    "MADDPGAgent",
    "MARLISAAgent",
    "MADCQAgent",
    "MetaEMSAgent",
    "CONTROLLERS_REACHABLE_FROM_ARMS",
]
