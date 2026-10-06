"""Audit B6 (the graph's spatial term was fabricated) and B7 (zero-filled history).

Notation:
  f_i      node feature vector of building i, z-scored per column, dimensionless
  w_ij     edge weight between buildings i and j, dimensionless in [0, 1]
  sigma_f  RBF bandwidth on the feature distance, in the same units as ||f_i - f_j||
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import GraphConfig
from stems.graph import BuildingGraph
from stems.utils import HistoryBuffer

REPO = Path(__file__).resolve().parents[1]
SCHEMA = REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"
needs_schema = pytest.mark.skipif(not SCHEMA.is_file(),
                                  reason=f"generated schema absent: {SCHEMA}")

#: The spread the audit measured for the old positional half-weight, over 56 edges.
FABRICATED_POSITIONAL_SPREAD = 2.394e-03


# ------------------------------------------------------------------ audit B6 --

def test_the_fabricated_positional_term_carried_no_information():
    """Reproduce the defect, so the improvement below has a measured baseline.

    Coordinates generated from the building's index, (30.26 + 0.01*i, -97.74 + 0.01*i),
    with sigma_d = 1.0: the positional half-weight 0.5*exp(-d^2/2) spans 2.4e-3 across
    all 56 off-diagonal edges of an 8-building graph.
    """
    B = 8
    pos = np.array([[30.26 + 0.01 * i, -97.74 + 0.01 * i] for i in range(B)],
                   dtype=np.float32)
    d_sq = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    half = 0.5 * np.exp(-d_sq / 2.0)
    off = half[~np.eye(B, dtype=bool)]
    assert off.size == 56
    assert off.min() == pytest.approx(0.497556, abs=1e-6)
    assert off.max() == pytest.approx(0.499950, abs=1e-6)
    assert (off.max() - off.min()) == pytest.approx(FABRICATED_POSITIONAL_SPREAD,
                                                    rel=1e-3)


@needs_schema
def test_the_environment_supplies_no_coordinates_and_says_so():
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(schema=str(SCHEMA), seed=0, heat_pump=True)
    info = env.get_building_info()
    assert info["positions"] is None, (
        "if CityLearn ever starts exposing coordinates, revisit the graph rather than "
        "leaving it feature-only")
    for building in env._env.buildings:
        for attr in ("latitude", "longitude", "floor_area"):
            assert not hasattr(building, attr), (
                f"CityLearn Building now has {attr}; audit B6's premise has changed")


@needs_schema
def test_the_feature_graph_edge_weights_actually_vary():
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(schema=str(SCHEMA), seed=0, heat_pump=True)
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"],
                          GraphConfig(mode="feature"))
    spread = graph.edge_weight_spread()
    assert spread["edges"] == 56
    assert spread["spread"] > 0.5, (
        f"edge weights span only {spread['spread']:.3e}; the graph is a mean pool "
        "again and the GCN has no spatial signal to use")
    assert spread["spread"] > 100 * FABRICATED_POSITIONAL_SPREAD
    assert spread["std"] > 0.1


@needs_schema
def test_no_graph_feature_is_constant_across_buildings():
    """The old feature vector was [battery capacity, 150.0]; the second did nothing."""
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(schema=str(SCHEMA), seed=0, heat_pump=True)
    info = env.get_building_info()
    assert info["constant_features"] == [], (
        f"these features are identical for every building and contribute nothing: "
        f"{info['constant_features']}")
    assert len(info["feature_names"]) == info["features"].shape[1] >= 6


@needs_schema
def test_graph_features_are_standardised():
    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(schema=str(SCHEMA), seed=0, heat_pump=True)
    f = env.get_building_info()["features"]
    assert np.abs(f.mean(axis=0)).max() < 1e-5
    assert np.allclose(f.std(axis=0), 1.0, atol=1e-5)


def test_mean_pool_mode_is_uniform():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(8, 5)).astype(np.float32)
    graph = BuildingGraph(8, None, features, GraphConfig(mode="mean_pool"))
    adj = graph.adj.numpy()
    assert np.allclose(np.diag(adj), 0.0)
    off = adj[~np.eye(8, dtype=bool)]
    assert np.allclose(off, 1.0)
    assert graph.edge_weight_spread()["spread"] == pytest.approx(0.0)


def test_the_mean_pool_ablation_arm_is_registered():
    from experiments.controllers import ARMS

    arm = ARMS["rl+calibrated+meanpool"]
    assert arm.graph_mode == "mean_pool"
    assert arm.learns and arm.barrier == "calibrated"
    reference = ARMS["rl+calibrated"]
    assert reference.graph_mode == "feature"
    # identical in every respect except the graph, so the contrast isolates the GCN
    for field in ("policy", "barrier", "residual", "penalty", "ev_request",
                  "forced_penalty", "ev_floor", "control"):
        assert getattr(arm, field) == getattr(reference, field), field


def test_the_median_bandwidth_puts_the_typical_edge_at_exp_minus_one():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(12, 6)).astype(np.float32)
    graph = BuildingGraph(12, None, features, GraphConfig(mode="feature", sigma_f=None))
    adj = graph.adj.numpy()
    off = adj[~np.eye(12, dtype=bool)]
    assert np.median(off) == pytest.approx(np.exp(-1.0), rel=0.02)
    assert 0.0 < graph.sigma_f_used < np.inf


def test_position_mode_refuses_to_invent_coordinates():
    features = np.eye(4, dtype=np.float32)
    graph = BuildingGraph(4, None, features, GraphConfig(mode="feature+position"))
    with pytest.raises(ValueError, match="no latitude/longitude|needs real coordinates"):
        graph.compute_edge_weights()


def test_an_unknown_graph_mode_is_rejected():
    graph = BuildingGraph(3, None, np.eye(3, dtype=np.float32),
                          GraphConfig(mode="telepathy"))
    with pytest.raises(ValueError, match="unknown GraphConfig.mode"):
        graph.compute_edge_weights()


# ------------------------------------------------------------------ audit B7 --

def test_history_prime_fills_the_window_with_the_first_observation():
    B, D, W = 3, 5, 24
    hist = HistoryBuffer(B, D, W)
    obs = [np.full(D, float(i + 1), dtype=np.float32) for i in range(B)]
    hist.prime(obs)
    window = hist.get()
    assert window.shape == (B, W, D)
    for i in range(B):
        assert np.allclose(window[i], float(i + 1)), (
            "every slot of the window must hold the first observation, not zero")


def test_a_primed_window_contains_no_spurious_zeros():
    """The defect B7 names: 23 of the first 24 steps attended over normalised zeros."""
    B, D, W = 2, 4, 24
    hist = HistoryBuffer(B, D, W)
    obs = [np.array([21.5, 0.5, 3.2, 1.0], dtype=np.float32) for _ in range(B)]

    hist.update(obs)                      # the old behaviour, for contrast
    assert (hist.get() == 0.0).sum() == B * (W - 1) * D

    hist.reset()
    hist.prime(obs)
    assert (hist.get() == 0.0).sum() == 0


def test_priming_then_stepping_still_slides_the_window():
    B, D, W = 1, 2, 4
    hist = HistoryBuffer(B, D, W)
    hist.prime([np.array([1.0, 1.0], dtype=np.float32)])
    hist.update([np.array([2.0, 2.0], dtype=np.float32)])
    hist.update([np.array([3.0, 3.0], dtype=np.float32)])
    assert hist.get()[0, :, 0] == pytest.approx([1.0, 1.0, 2.0, 3.0])
