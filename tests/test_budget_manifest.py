"""Budget-manifest integration for the native SAM3D Euler seam."""

import torch

from sam3d_objects.model.backbone.generator.flow_matching.solver import Euler


def _tree(value):
    return {"x": torch.tensor([float(value)]), "nested": {"y": torch.tensor([1.0])}}


def test_euler_manifest_records_actual_decisions_and_costs():
    solver = Euler().enable_hicache(interval=3, first_enhance=0, max_horizon=1)

    def dynamics(x, t):
        return _tree(float(t))

    list(solver.solve_iter(dynamics, _tree(0.0), torch.linspace(0, 1, 9)))
    report = solver.get_hicache_manifest()
    assert report["schema"] == "hicache-pp.run-manifest.v1"
    assert report["identity"]["stage"] == "flow"
    assert report["counts"]["fallback"] > 0
    assert len(report["measurements"]) == 8
    assert all("path" not in row for row in report["events"])

