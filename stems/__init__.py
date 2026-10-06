from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
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

__all__ = [
    "STEMSConfig",
    "STEMSEnvironment",
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
