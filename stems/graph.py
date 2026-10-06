from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from stems.config import GraphConfig


class BuildingGraph:
    """Weighted adjacency over buildings, consumed by ``SpatialGCN``.

    Audit B6: the original adjacency mixed a positional kernel with a feature kernel,
    but CityLearn's ``Building`` exposes no ``latitude``, ``longitude`` or ``floor_area``
    (verified on 2.6.0b1 against the live objects, the schema, the dataset directory, the
    ResStock time-series header and the repository's own sizing file). The positional
    term therefore ran on coordinates generated from the building's index in the schema,
    `30.26 + 0.01*i`, and a floor area of 150 m^2 constant across buildings. With
    sigma_d = 1.0 every positional half-weight lay in [0.497556, 0.500000] — a spread of
    2.4e-3 across all 56 off-diagonal edges — so the "spatial" half of the
    spatio-temporal encoder was a near-uniform mean pool over a function of schema
    ordering.

    No real coordinates are recoverable from this dataset, so the graph is now
    **feature-based** and the paper must say so. ``positions`` may be ``None``.
    """

    def __init__(
        self,
        num_buildings: int,
        positions: Optional[np.ndarray],
        features: np.ndarray,
        config: Optional[GraphConfig] = None,
    ) -> None:
        self.num_buildings = num_buildings
        self.positions = (None if positions is None
                          else np.asarray(positions, dtype=np.float32))
        self.features = np.asarray(features, dtype=np.float32)
        self.config = config or GraphConfig()
        self._adj: Optional[torch.Tensor] = None

    @staticmethod
    def _sq_distances(x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]
        return (diff ** 2).sum(axis=-1)

    @staticmethod
    def _median_bandwidth(sq: np.ndarray) -> float:
        """sigma such that the median off-diagonal squared distance maps to exp(-1).

        The median heuristic for an RBF kernel. It makes the bandwidth a property of the
        data rather than a constant that silently depends on how the features happen to
        be scaled.
        """
        B = sq.shape[0]
        if B < 2:
            return 1.0
        off = sq[~np.eye(B, dtype=bool)]
        med = float(np.median(off))
        if not np.isfinite(med) or med <= 1e-12:
            return 1.0
        return float(np.sqrt(med / 2.0))

    def compute_edge_weights(self) -> torch.Tensor:
        B = self.num_buildings
        cfg = self.config
        mode = getattr(cfg, "mode", "feature")

        if mode == "mean_pool":
            w = np.ones((B, B), dtype=np.float64)
        else:
            f_sq = self._sq_distances(self.features)
            sigma_f = cfg.sigma_f
            if sigma_f is None:
                sigma_f = self._median_bandwidth(f_sq)
            self.sigma_f_used = float(sigma_f)
            w_feat = np.exp(-f_sq / (2.0 * float(sigma_f) ** 2))

            if mode == "feature":
                w = w_feat
            elif mode == "feature+position":
                if self.positions is None:
                    raise ValueError(
                        "GraphConfig.mode='feature+position' needs real coordinates, "
                        "and this environment supplies none. CityLearn's Building has "
                        "no latitude/longitude; see CHANGELOG.md step 6. Use "
                        "mode='feature' or mode='mean_pool'.")
                d_sq = self._sq_distances(self.positions)
                sigma_d = cfg.sigma_d
                if sigma_d is None:
                    sigma_d = self._median_bandwidth(d_sq)
                self.sigma_d_used = float(sigma_d)
                w = (cfg.alpha * np.exp(-d_sq / (2.0 * float(sigma_d) ** 2))
                     + cfg.beta * w_feat)
            else:
                raise ValueError(
                    f"unknown GraphConfig.mode {mode!r}; choices: 'feature', "
                    "'feature+position', 'mean_pool'")

        np.fill_diagonal(w, 0.0)

        self._adj = torch.tensor(w, dtype=torch.float32)
        return self._adj

    def edge_weight_spread(self) -> Dict[str, float]:
        """min, max and spread of the off-diagonal edge weights. For CHANGELOG/reports."""
        adj = self.adj.numpy()
        B = adj.shape[0]
        off = adj[~np.eye(B, dtype=bool)]
        return {"min": float(off.min()), "max": float(off.max()),
                "spread": float(off.max() - off.min()),
                "mean": float(off.mean()), "std": float(off.std()),
                "edges": int(off.size)}


    def get_node_features(self, observations: List[np.ndarray]) -> torch.Tensor:
        x = np.stack(observations, axis=0).astype(np.float32)
        return torch.tensor(x, dtype=torch.float32)


    @property
    def adj(self) -> torch.Tensor:
        if self._adj is None:
            self.compute_edge_weights()
        return self._adj
