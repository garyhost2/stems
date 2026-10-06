from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalised_adjacency(adj: torch.Tensor) -> torch.Tensor:
    r"""Symmetrically normalised adjacency with self-loops, :math:`\hat{D}^{-1/2}
    (A + I) \hat{D}^{-1/2}`.

    Cached on the tensor object: the adjacency is fixed for a run (it is built once from
    the building graph), but this was recomputed inside every one of the three GCN
    layers, for every sample, inside the Python loop over the batch.
    """
    cached = getattr(adj, "_stems_norm_cache", None)
    if cached is not None and cached.shape == adj.shape and cached.device == adj.device:
        return cached
    B = adj.size(0)
    adj_hat = adj + torch.eye(B, device=adj.device, dtype=adj.dtype)
    d_inv_sqrt = adj_hat.sum(dim=1).pow(-0.5)
    norm = d_inv_sqrt.unsqueeze(1) * adj_hat * d_inv_sqrt.unsqueeze(0)
    try:
        adj._stems_norm_cache = norm
    except AttributeError:
        pass
    return norm


class GCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, C) for one time step or (N, B, C) for a batch of N of them.

        ``adj_norm @ x`` broadcasts over the leading batch axis in both cases, so the
        same code path serves the single-step and the batched call; there is no Python
        loop over the batch (audit F).
        """
        out = normalised_adjacency(adj) @ x @ self.weight
        if self.bias_param is not None:
            out = out + self.bias_param
        return out


class SpatialGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        dims = [in_channels] + [hidden_dim] * num_layers

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, adj)
            h = F.relu(h)
        return h


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        window_size: int = 24,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(obs_dim, embed_dim)

        self.register_buffer("pos_enc", self._build_pos_enc(window_size, embed_dim))

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    @staticmethod
    def _build_pos_enc(window: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(window, dim)
        pos = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: dim // 2])
        return pe.unsqueeze(0)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        B, T, _ = history.shape
        x = self.input_proj(history) + self.pos_enc[:, :T, :]

        attn_out, _ = self.attn(x, x, x)
        z = self.norm(x + attn_out)

        return z[:, -1, :]


class STEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        spatial_dim: int = 64,
        temporal_dim: int = 32,
        output_dim: int = 64,
        gcn_num_layers: int = 3,
        num_heads: int = 4,
        window_size: int = 24,
    ) -> None:
        super().__init__()

        self.spatial_gcn = SpatialGCN(
            in_channels=obs_dim,
            hidden_dim=spatial_dim,
            num_layers=gcn_num_layers,
        )

        self.temporal_transformer = TemporalTransformer(
            obs_dim=obs_dim,
            embed_dim=temporal_dim,
            num_heads=num_heads,
            window_size=window_size,
        )

        self.W_s = nn.Linear(spatial_dim, output_dim, bias=False)
        self.W_t = nn.Linear(temporal_dim, output_dim, bias=True)
        self.out_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        h = self.spatial_gcn(x, adj)
        z = self.temporal_transformer(history)
        r = self.W_s(h) + self.W_t(z)
        return r

    def batch_forward(
        self,
        x_nb: torch.Tensor,
        adj: torch.Tensor,
        history_nb: torch.Tensor,
    ) -> torch.Tensor:
        N, B, obs_dim = x_nb.shape
        T = history_nb.shape[2]

        hist_flat = history_nb.view(N * B, T, obs_dim)
        z_flat = self.temporal_transformer(hist_flat)
        z_nb = z_flat.view(N, B, -1)

        # Audit F: this was a Python loop `for n in range(N): h_nb[n] =
        # self.spatial_gcn(x_nb[n], adj)`, N separate three-layer GCN calls per update,
        # each of which also rebuilt the normalised adjacency. The GCN is linear in the
        # node axis and the adjacency is shared, so the whole batch is one set of matmuls
        # on an (N, B, C) tensor. Numerically identical, not an approximation.
        h_nb = self.spatial_gcn(x_nb, adj)

        r_nb = self.W_s(h_nb) + self.W_t(z_nb)
        return r_nb
