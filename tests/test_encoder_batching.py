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


def test_the_batched_path_is_not_materially_slower_than_the_loop():
    """End-to-end encoder timing is near parity, and the old assertion overstated it.

    This test used to assert ``fast < slow`` on the mean of three repetitions, and it
    failed on a loaded machine with "batched 39.8 ms vs loop 35.0 ms". That is not
    flakiness to be papered over: measured over nine repetitions at N = 336 with both
    paths warmed, the batched path is 1.21x faster on the median but 0.93x on the
    minimum, i.e. it can be the slower of the two on a best-case run.

    That is consistent with the other measurement in CHANGELOG.md: the graph
    convolution stage alone is 33-57x faster batched, but it is only about 43% of
    ``batch_forward``, and one full PPO-Lagrangian update gains just 1.30-1.34x. So the
    claim this test can defend across machines is the weaker one -- batching did not
    cost anything -- and the correctness claim lives in the tests above, which pin that
    the two paths produce the same tensors.

    The median is used rather than the mean because a single scheduler preemption
    inflates a mean, and compared with a tolerance rather than strictly because the true
    end-to-end factor is close enough to 1 that the sign is machine-dependent.
    """
    encoder, x_nb, adj, history_nb = _fixture(N=336, B=8, obs_dim=30, T=24)
    reps = 9
    with torch.no_grad():
        for _ in range(3):   # warm BOTH paths; warming only one biases the comparison
            encoder.batch_forward(x_nb, adj, history_nb)
            _loop_reference(encoder, x_nb, adj.clone(), history_nb)

        def timings(fn):
            out = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                out.append(time.perf_counter() - t0)
            return sorted(out)[reps // 2]

        fast = timings(lambda: encoder.batch_forward(x_nb, adj, history_nb))
        slow = timings(lambda: _loop_reference(encoder, x_nb, adj.clone(), history_nb))
    assert fast < slow * 1.25, (
        f"batched {fast * 1e3:.1f} ms vs loop {slow * 1e3:.1f} ms (median of {reps}); "
        "batching is expected to be near parity end to end, but not materially slower")
