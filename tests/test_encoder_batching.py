"""Audit F: STEncoder.batch_forward looped over the batch axis in Python.

The GCN is linear in the node axis and the adjacency is shared across the batch, so
``adj_norm @ x`` broadcasts and the whole batch is one set of matmuls. The replacement
must be numerically identical, not merely close: these tests pin that to 1e-5 against
the loop it replaces, and measure the speed-up.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.encoder import STEncoder, normalised_adjacency

TOL = 1e-5


def _loop_reference(encoder: STEncoder, x_nb, adj, history_nb):
    """The pre-fix batch_forward, verbatim, as the thing to agree with."""
    N, B, obs_dim = x_nb.shape
    T = history_nb.shape[2]
    hist_flat = history_nb.view(N * B, T, obs_dim)
    z_flat = encoder.temporal_transformer(hist_flat)
    z_nb = z_flat.view(N, B, -1)
    h_nb = torch.zeros(N, B, encoder.spatial_gcn.out_dim, device=x_nb.device)
    for n in range(N):
        h_nb[n] = encoder.spatial_gcn(x_nb[n], adj)
    return encoder.W_s(h_nb) + encoder.W_t(z_nb)


def _fixture(N=64, B=8, obs_dim=30, T=24, seed=0):
    torch.manual_seed(seed)
    encoder = STEncoder(obs_dim=obs_dim, window_size=T).eval()
    x_nb = torch.randn(N, B, obs_dim)
    history_nb = torch.randn(N, B, T, obs_dim)
    raw = torch.rand(B, B).abs()
    adj = ((raw + raw.T) / 2.0)
    adj.fill_diagonal_(0.0)
    return encoder, x_nb, adj, history_nb


def test_batched_gcn_matches_the_python_loop():
    encoder, x_nb, adj, history_nb = _fixture()
    with torch.no_grad():
        fast = encoder.batch_forward(x_nb, adj, history_nb)
        slow = _loop_reference(encoder, x_nb, adj.clone(), history_nb)
    assert fast.shape == slow.shape
    assert torch.allclose(fast, slow, atol=TOL, rtol=0.0), (
        f"max abs difference {(fast - slow).abs().max().item():.3e} exceeds {TOL}")


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("shape", [(1, 2), (7, 3), (33, 8)])
def test_batched_gcn_matches_the_loop_on_random_shapes(seed, shape):
    N, B = shape
    encoder, x_nb, adj, history_nb = _fixture(N=N, B=B, obs_dim=11, T=6, seed=seed)
    with torch.no_grad():
        fast = encoder.batch_forward(x_nb, adj, history_nb)
        slow = _loop_reference(encoder, x_nb, adj.clone(), history_nb)
    assert torch.allclose(fast, slow, atol=TOL, rtol=0.0)


def test_single_step_forward_is_unchanged_by_the_batching():
    """`forward` (one time step) must still agree with `batch_forward` on N = 1."""
    encoder, x_nb, adj, history_nb = _fixture(N=1, B=8, obs_dim=30, T=24)
    with torch.no_grad():
        one = encoder(x_nb[0], adj, history_nb[0])
        many = encoder.batch_forward(x_nb, adj, history_nb)
    assert torch.allclose(one, many[0], atol=TOL, rtol=0.0)


def test_gradients_flow_through_the_batched_path():
    encoder, x_nb, adj, history_nb = _fixture(N=16, B=8, obs_dim=12, T=6)
    encoder.train()
    out = encoder.batch_forward(x_nb, adj, history_nb)
    out.square().mean().backward()
    grads = [p.grad for p in encoder.spatial_gcn.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_normalised_adjacency_is_symmetric_and_cached():
    raw = torch.rand(6, 6)
    adj = (raw + raw.T) / 2.0
    adj.fill_diagonal_(0.0)
    norm = normalised_adjacency(adj)
    assert torch.allclose(norm, norm.T, atol=1e-6)
    assert normalised_adjacency(adj) is norm, "the normalisation should be cached"


def test_the_batched_path_is_faster_than_the_loop():
    """Speed is the point of the change; the figure is recorded in CHANGELOG.md.

    A loose threshold, because wall-clock on a shared machine is noisy. The measured
    factor at N = 336 (a 14-day episode) is reported in the changelog, not asserted.
    """
    encoder, x_nb, adj, history_nb = _fixture(N=336, B=8, obs_dim=30, T=24)
    with torch.no_grad():
        encoder.batch_forward(x_nb, adj, history_nb)      # warm up
        t0 = time.perf_counter()
        for _ in range(3):
            encoder.batch_forward(x_nb, adj, history_nb)
        fast = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            _loop_reference(encoder, x_nb, adj.clone(), history_nb)
        slow = (time.perf_counter() - t0) / 3
    assert fast < slow, f"batched {fast*1e3:.1f} ms vs loop {slow*1e3:.1f} ms"
