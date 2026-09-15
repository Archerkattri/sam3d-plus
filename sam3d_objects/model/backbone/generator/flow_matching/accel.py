"""Hermite HiCache adapter for SAM 3D's structured flow-matching velocities.

The shared PyTree arithmetic and cache state live in ``hicache-pp``. This fork
intentionally exposes only the Hermite backend: the sibling ``sam3d-plus-plus``
owns the DMD variant, so a stock-baseline run cannot change forecast methods by
passing an accidental backend option.
"""

try:
    from hicache_pp.tree import (
        adaptive_cfg_decide,
        adaptive_cfg_init,
        forecast_guidance_tree,
        guidance_term_tree,
        hicache_decide,
        hicache_forecast_tree,
        hicache_init as _central_hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_tree,
        hermite_coeff,
        physicists_hermite,
        reconstruct_cfg_tree,
        tree_axpy,
        tree_cosine,
        tree_detach,
        tree_sub_div,
    )
except ImportError as exc:  # pragma: no cover - exercised by installation failures
    raise ImportError(
        "sam3d-plus acceleration requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


def hicache_init(num_steps, interval=4, max_order=1, first_enhance=2,
                 end_enhance=None, sigma=0.5, backend="hermite"):
    """Create the SAM3D Hermite cache while rejecting non-baseline backends."""
    if backend != "hermite":
        raise ValueError(
            "sam3d-plus supports only backend='hermite'; use sam3d-plus-plus for DMD"
        )
    return _central_hicache_init(
        num_steps=num_steps,
        interval=interval,
        max_order=max_order,
        first_enhance=first_enhance,
        end_enhance=end_enhance,
        sigma=sigma,
        backend="hermite",
    )


__all__ = [
    "adaptive_cfg_decide", "adaptive_cfg_init", "forecast_guidance_tree",
    "guidance_term_tree", "hicache_decide", "hicache_forecast_tree", "hicache_init",
    "hicache_reset", "hicache_telemetry", "hicache_update_tree", "hermite_coeff",
    "physicists_hermite", "reconstruct_cfg_tree", "tree_axpy", "tree_cosine",
    "tree_detach", "tree_sub_div",
]


if __name__ == "__main__":
    import torch

    state = hicache_init(num_steps=8, interval=3, first_enhance=0)
    sample = {"x": torch.tensor([1.0, 2.0]), "nested": {"y": torch.tensor([3.0])}}
    for step in range(3):
        state["step"] = step
        decision = hicache_decide(state)
        if decision == "full":
            hicache_update_tree(state, sample)
        else:
            hicache_forecast_tree(state)
    assert state["backend"] == "hermite"
    assert hicache_telemetry(state)["decisions"]["full"] > 0
    print("sam3d-plus Hermite PyTree smoke passed")
