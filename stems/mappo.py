r"""MAPPO with a genuinely centralised critic -- the CTDE arm.

Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent
Games*: parameter-shared actors acting on local information, one critic conditioned on
the joint state, GAE, a clipped surrogate, advantage standardisation, an entropy bonus
and a KL early stop.

**Why this arm exists.** It is the only controller in the repository that answers the
question the district cap poses. ``stems.baselines.DMAPPOAgent`` is decentralised
training, decentralised execution: every building's critic sees only its own
observation, so nothing in the value function knows that the buildings share an import
limit. ``stems.agent.STEMSAgent`` is often described as multi-agent but is not: its
actors and critics all read the *same* graph-encoded representation and each critic
still takes only its own node's row, so it is parameter-shared independent PPO on a
graph-mixed observation (audit D1). This class is the first where a critic takes the
joint observation as its argument.

**What the critic conditions on.** The shared spatio-temporal encoder
(``stems.encoder.STEncoder``) maps the :math:`B` current observations and their 24-step
histories to a representation matrix :math:`R \in \mathbb{R}^{B \times d}`. The actor
for building :math:`i` reads row :math:`R_i` -- local information after graph mixing,
exactly what ``STEMSAgent``'s actors read, so the two are comparable. The critic reads
the **concatenation of every row**, :math:`s = \mathrm{vec}(R) \in \mathbb{R}^{Bd}`, and
returns one value per building from one head, :math:`V(s) \in \mathbb{R}^{B}`. Building
:math:`i`'s advantage uses :math:`V_i(s)`: the *agent-specific global state* of Yu et
al., where every agent's value sees the whole system but is allowed to differ by agent.
Concatenation (rather than a permutation-invariant pool) is deliberate -- the buildings
are not interchangeable, they have different device sizes, and the critic is allowed to
know which is which.

**What it does not do.** No Lagrangian, no cost critic. Those are
``stems.agent.STEMSAgent``'s contribution and folding them in here would confound the
centralisation question with the constraint-handling question. Constraint handling for
this arm is whatever safety layer the arm is configured with, the same as for every
other comparison controller.

**The experiment this makes possible.** One algorithm, one environment, the district cap
on and off, decentralised and centralised critic in each cell. ``docs/LITERATURE.md``
section on multi-agent reinforcement learning for CityLearn records Khouja et al. (2026)
reporting decentralised training beating centralised *without* a district cap and
Shojaeighadikolaei et al. (2024) reporting a centralised critic helping when electric
vehicles share a transformer; the clean two-by-two that separates the two results
appears in no retrieved source and is available here. A caveat for whoever runs it: on
``citylearn_schemas/tx_travis_8b`` with eight buildings the uncontrolled district peak
is near 40 kW, so the default ``grid_cap_kw = 300`` is **not binding** and the two cells
of the "cap on" column would be identical to the "cap off" column. The cap has to be set
near the uncontrolled peak for the coupling to exist at all.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.agent import Actor
from stems.config import STEMSConfig
from stems.encoder import STEncoder
from stems.graph import BuildingGraph
from stems.utils import RunningNormalizer

__all__ = ["CentralisedCritic", "MAPPOCentralisedCritic"]


class CentralisedCritic(nn.Module):
    r""":math:`V(s) \in \mathbb{R}^{B}` from the joint representation :math:`s`.

    Input is ``(N, B, d)``, flattened to ``(N, Bd)``; output is ``(N, B)``. One network,
    one forward pass, every agent's value informed by every agent's observation.
    """

    def __init__(self, repr_dim: int, num_buildings: int, hidden_dim: int) -> None:
        super().__init__()
        self.B = int(num_buildings)
        self.net = nn.Sequential(
            nn.Linear(repr_dim * self.B, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, self.B),
        )

    def forward(self, repr_nb: torch.Tensor) -> torch.Tensor:
        if repr_nb.dim() == 2:
            repr_nb = repr_nb.unsqueeze(0)
        return self.net(repr_nb.reshape(repr_nb.shape[0], -1))


class MAPPOCentralisedCritic:
    """MAPPO over the shared spatio-temporal encoder.

    Presents the interface ``experiments/runner.py`` trains against -- ``select_action``,
    ``observe``, ``update``, ``save``, ``load``, and the ``_last_*`` attributes the
    collector reads -- so it drops into the existing loop with no special case.
    """

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int,
                 building_graph: BuildingGraph, config: Optional[STEMSConfig] = None,
                 control_indices: Optional[List[int]] = None,
                 electrical_storage_action_index: int = 1,
                 device: str = "cpu") -> None:
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.B = int(num_buildings)
        self.cfg = config or STEMSConfig()
        self.device = torch.device(device)
        self.graph = building_graph
        self.adj = building_graph.compute_edge_weights().to(self.device)
        self.elec_idx = int(electrical_storage_action_index)
        self.control_indices = (list(range(self.action_dim)) if control_indices is None
                                else list(control_indices))
        self._ctrl = torch.tensor(self.control_indices, dtype=torch.long, device=self.device)

        self.encoder = STEncoder(
            obs_dim=obs_dim, spatial_dim=self.cfg.gcn.hidden_dim,
            temporal_dim=self.cfg.transformer.embed_dim,
            output_dim=self.cfg.fusion.output_dim,
            gcn_num_layers=self.cfg.gcn.num_layers,
            num_heads=self.cfg.transformer.num_heads,
            window_size=self.cfg.transformer.window_size).to(self.device)
        repr_dim, hidden = self.cfg.fusion.output_dim, self.cfg.actor_critic.hidden_dim
        # One actor, shared by every building: parameter sharing is MAPPO's default and
        # is what makes the centralised critic the only structural difference from the
        # decentralised arm.
        self.actor = Actor(repr_dim, hidden, action_dim).to(self.device)
        self.critic = CentralisedCritic(repr_dim, self.B, hidden).to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters())
            + list(self.critic.parameters()), lr=self.cfg.actor_critic.lr)
        self.obs_normalizer = RunningNormalizer(obs_dim).to(self.device)
        self.return_normalizer = RunningNormalizer(1).to(self.device)

        self._last_raw_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_safe_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_nominal_actions = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_pre_tanh = np.zeros((self.B, action_dim), dtype=np.float32)
        self._last_log_probs = np.zeros(self.B, dtype=np.float32)
        self.fleet_shield = None

    # -- acting -----------------------------------------------------------------
    def _encode(self, obs_nb: torch.Tensor, hist_nb: torch.Tensor) -> torch.Tensor:
        o = self.obs_normalizer(obs_nb)
        h = self.obs_normalizer(hist_nb.reshape(-1, self.obs_dim)).view(hist_nb.shape)
        return self.encoder.batch_forward(o, self.adj, h)

    def select_action(self, obs_list, history, explore: bool = True) -> np.ndarray:
        self.encoder.eval(); self.actor.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.stack(obs_list, 0), dtype=torch.float32, device=self.device)
            h = torch.as_tensor(np.asarray(history, dtype=np.float32), device=self.device)
            repr_mat = self.encoder(self.obs_normalizer(x), self.adj,
                                    self.obs_normalizer(h.view(-1, self.obs_dim)).view(h.shape))
            dist = self.actor.distribution(repr_mat)
            z = dist.sample() if explore else dist.mean
            self._last_log_probs = dist.log_prob(z)[..., self._ctrl].sum(-1).cpu().numpy()
            self._last_pre_tanh = z.cpu().numpy().astype(np.float32)
            raw = torch.tanh(z).cpu().numpy().astype(np.float32)
        self._last_raw_actions = raw.copy()
        self._last_nominal_actions = raw.copy()
        self._last_safe_actions = raw.copy()
        return raw

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        return None

    # -- learning ---------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        cfgt = self.cfg.training
        gamma, lam = self.cfg.actor_critic.gamma, float(cfgt.gae_lambda)
        N = len(batch["obs"])
        if N == 0:
            return {}
        for key in ("history", "next_history", "pre_tanh", "behaviour_log_probs"):
            if batch.get(key) is None:
                raise KeyError(
                    f"batch is missing required {key!r}. MAPPO needs the behaviour "
                    "policy's own samples and log-probabilities; see experiments/runner.py.")
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        obs, next_obs = t(batch["obs"]), t(batch["next_obs"])
        hist, next_hist = t(batch["history"]), t(batch["next_history"])
        rewards, episode_end = t(batch["rewards"]), t(batch["dones"])
        z_all, old_logp = t(batch["pre_tanh"]), t(batch["behaviour_log_probs"])

        if int(self.obs_normalizer.count) == 0:
            # First episode fits the observation normaliser only, exactly as
            # STEMSAgent does, so the two arms see the same warm-up protocol.
            with torch.no_grad():
                self.obs_normalizer.update(obs.view(-1, self.obs_dim))
            return {"policy": 0.0, "value": 0.0, "entropy": 0.0, "gradient_steps": 0,
                    "warmup": True, "approx_kl": 0.0, "clip_frac": 0.0}

        for m in (self.encoder, self.actor, self.critic):
            m.eval()
        with torch.no_grad():
            if cfgt.scale_rewards:
                ret = torch.zeros_like(rewards)
                running = torch.zeros(self.B, device=self.device)
                for k in reversed(range(N)):
                    running = rewards[k] + gamma * (1.0 - episode_end[k]) * running
                    ret[k] = running
                self.return_normalizer.update(ret.reshape(-1, 1))
                scale = float(self.return_normalizer.var.sqrt().clamp_min(1e-6))
            else:
                scale = 1.0
            repr_all = self._encode(obs, hist)
            repr_next = self._encode(next_obs, next_hist)
            values = self.critic(repr_all)
            next_values = self.critic(repr_next)
            adv = torch.zeros_like(rewards)
            last = torch.zeros(self.B, device=self.device)
            for k in reversed(range(N)):
                delta = (rewards[k] / scale
                         + gamma * (1.0 - episode_end[k]) * next_values[k] - values[k])
                last = delta + gamma * lam * (1.0 - episode_end[k]) * last
                adv[k] = last
            returns = adv + values
            adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)

        mb = max(2, min(int(cfgt.minibatch_size), N))
        clip = float(cfgt.ppo_clip)
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        sums = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "clip_frac": 0.0}
        n_updates, stopped = 0, False
        for m in (self.encoder, self.actor, self.critic):
            m.train()
        for _ in range(max(1, int(cfgt.update_epochs))):
            if stopped:
                break
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                if idx.numel() < 2:
                    continue
                repr_mb = self._encode(obs[idx], hist[idx])
                dist = self.actor.distribution(repr_mb)
                logp = dist.log_prob(z_all[idx])[..., self._ctrl].sum(-1)
                log_ratio = logp - old_logp[idx]
                ratio = log_ratio.clamp(-20.0, 20.0).exp()
                a_b = adv[idx]
                policy_loss = -torch.min(
                    ratio * a_b, ratio.clamp(1 - clip, 1 + clip) * a_b).mean()
                v_pred = self.critic(repr_mb)
                v_clipped = values[idx] + (v_pred - values[idx]).clamp(-clip, clip)
                value_loss = torch.max((v_pred - returns[idx]) ** 2,
                                       (v_clipped - returns[idx]) ** 2).mean()
                entropy = dist.entropy()[..., self._ctrl].sum(-1).mean()
                loss = (policy_loss + float(cfgt.value_coef) * value_loss
                        - float(cfgt.entropy_coef) * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, float(cfgt.max_grad_norm))
                self.optimizer.step()
                with torch.no_grad():
                    kl = float(((ratio - 1.0) - log_ratio).mean())
                sums["policy"] += float(policy_loss)
                sums["value"] += float(value_loss)
                sums["entropy"] += float(entropy)
                sums["approx_kl"] += kl
                sums["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
                n_updates += 1
                if cfgt.target_kl and kl > 1.5 * cfgt.target_kl:
                    stopped = True
                    break
        with torch.no_grad():
            self.obs_normalizer.update(obs.view(-1, self.obs_dim))
        k = max(n_updates, 1)
        return {**{key: v / k for key, v in sums.items()},
                "gradient_steps": n_updates, "stopped_early": stopped,
                "reward_scale": scale}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(), "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "obs_normalizer": self.obs_normalizer.state_dict(),
                    "return_normalizer": self.return_normalizer.state_dict()},
                   os.path.join(path, "mappo_cc.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "mappo_cc.pt"), map_location=self.device)
        self.encoder.load_state_dict(d["encoder"])
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.obs_normalizer.load_state_dict(d["obs_normalizer"])
        self.return_normalizer.load_state_dict(d["return_normalizer"])
