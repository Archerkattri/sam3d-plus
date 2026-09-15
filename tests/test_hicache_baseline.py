"""CPU contract tests for the stock SAM 3D Hermite baseline."""

import pytest
import torch

from sam3d_objects.model.backbone.generator.flow_matching import accel
from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler, Midpoint


def _tree(value):
    return {"x": torch.tensor([value, value + 1.0]), "nested": {"y": torch.tensor([value])}}


def test_facade_reuses_central_hermite_arithmetic_and_rejects_dmd():
    import hicache_pp.tree as central

    assert accel.hermite_coeff is central.hermite_coeff
    assert accel.tree_axpy is central.tree_axpy
    state = accel.hicache_init(num_steps=8, interval=3, first_enhance=0)
    assert state["backend"] == "hermite"
    with pytest.raises(ValueError, match="only backend='hermite'"):
        accel.hicache_init(num_steps=8, backend="dmd")


def test_euler_cache_trace_guidance_and_reset():
    calls = 0

    def dynamics(x, _t):
        nonlocal calls
        calls += 1
        return {"x": torch.ones_like(x["x"]), "nested": {"y": torch.ones_like(x["nested"]["y"])}}

    solver = Euler().enable_hicache(interval=3, first_enhance=0)
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 7)))
    telemetry = solver.get_hicache_telemetry()
    assert calls == 2
    assert telemetry["decisions"] == {"full": 2, "forecast": 4}
    assert telemetry["method_counts"]["hermite"] == 4
    assert solver.get_hicache_status()["active"] is True

    # A second trajectory starts from a clean central state and repeats the cadence.
    calls = 0
    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 7)))
    assert calls == 2
    assert solver.get_hicache_telemetry()["decisions"] == telemetry["decisions"]


def test_guidance_reconstruction_and_unsupported_solver_status():
    cond = _tree(2.0)
    uncond = _tree(1.0)
    guidance = accel.guidance_term_tree(cond, uncond, 3.0)
    reconstructed = accel.reconstruct_cfg_tree(cond, guidance)
    expected = accel.tree_axpy(4.0, uncond, accel.tree_sub_div(cond, uncond, 1.0))
    for actual, target in zip(torch.utils._pytree.tree_leaves(reconstructed),
                              torch.utils._pytree.tree_leaves(expected)):
        assert torch.equal(actual, target)

    solver = Midpoint().enable_hicache(interval=2, first_enhance=0)
    calls = 0

    def dynamics(x, _t):
        nonlocal calls
        calls += 1
        return {"x": torch.ones_like(x["x"]), "nested": {"y": torch.ones_like(x["nested"]["y"])}}

    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 4)))
    assert calls == 6
    assert solver.get_hicache_status()["reason"] == "unsupported_solver"
    assert solver.get_hicache_telemetry()["fallbacks"] == {"unsupported_solver": 1}

    solver.disable_hicache()
    assert solver.get_hicache_status()["reason"] == "disabled"
    assert solver.get_hicache_telemetry() is None
