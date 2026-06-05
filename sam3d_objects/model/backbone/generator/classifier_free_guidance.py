# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from numbers import Number
import torch
import random
from torch.utils import _pytree
from torch.utils._pytree import tree_map_only
from loguru import logger

from .flow_matching.accel import (
    adaptive_cfg_init,
    adaptive_cfg_decide,
    forecast_guidance_tree,
    reconstruct_cfg_tree,
    tree_cosine,
)

def _zeros_like(struct):
    def make_zeros(x):
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return x

    return _pytree.tree_map(make_zeros, struct)


def zero_out(args, kwargs):
    args = _zeros_like(args)
    kwargs = _zeros_like(kwargs)
    return args, kwargs


def discard(args, kwargs):
    return (), {}


def _drop_tensors(struct):
    """
    Drop any conditioning that are tensors
    Not using _pytree since we actually want to throw them instead of keeping them.
    """
    if isinstance(struct, dict):
        return {
            k: _drop_tensors(v)
            for k, v in struct.items()
            if not isinstance(v, torch.Tensor)
        }
    elif isinstance(struct, (list, tuple)):
        filtered = [_drop_tensors(x) for x in struct if not isinstance(x, torch.Tensor)]
        return tuple(filtered) if isinstance(struct, tuple) else filtered
    else:
        return struct


def drop_tensors(args, kwargs):
    args = _drop_tensors(args)
    kwargs = _drop_tensors(kwargs)
    return args, kwargs


def add_flag(args, kwargs):
    kwargs["cfg"] = True
    return args, kwargs


class ClassifierFreeGuidance(torch.nn.Module):
    UNCONDITIONAL_HANDLING_TYPES = {
        "zeros": zero_out,
        "discard": discard,
        "drop_tensors": drop_tensors,
        "add_flag": add_flag,
    }

    def __init__(
        self,
        backbone,  # backbone should be a backbone/generator (e.g. DDPM/DDIM/FlowMatching)
        p_unconditional=0.1,
        strength=3.0,
        # "zeros" = set cond tensors to 0,
        # "discard" = remove cond arguments and let underlying model handle it
        # "drop_tensors" = drop all tensors but leave non-tensors
        # "add_flag" = add an argument in kwargs as "cfg" and defer the handling to generator backbone
        unconditional_handling="zeros",
        interval=None,  # only perform cfg if t within interval
    ):
        super().__init__()

        if not (
            unconditional_handling
            in ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES
        ):
            raise RuntimeError(
                f"'{unconditional_handling}' is not valid for `unconditional_handling`, should be in {ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES}"
            )

        self.backbone = backbone
        self.p_unconditional = p_unconditional
        self.strength = strength
        self.unconditional_handling = unconditional_handling
        self.interval = interval
        self._make_unconditional_args = (
            ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES[
                self.unconditional_handling
            ]
        )

    def enable_adaptive_guidance(self, gamma_bar: float = 0.94, warmup: int = 2, max_order: int = 1):
        """Enable Adaptive-CFG (Adaptive Guidance, arXiv:2312.12487): once the conditional
        and unconditional velocities align (cosine >= gamma_bar) drop the unconditional
        backbone pass and reconstruct the guidance term from cached anchors (~half the
        per-step compute on aligned steps). Native — no monkey-patching. Per-trajectory
        state is reset by FlowMatching.generate_iter via reset_adaptive_guidance()."""
        self._adacfg_cfg = dict(gamma_bar=gamma_bar, warmup=warmup, max_order=max_order)
        self._adacfg = None
        return self

    def disable_adaptive_guidance(self):
        self._adacfg_cfg = None
        self._adacfg = None
        return self

    def reset_adaptive_guidance(self, num_steps: int):
        """(Re)initialise per-trajectory Adaptive-CFG state; no-op unless enabled."""
        if getattr(self, "_adacfg_cfg", None) is not None:
            self._adacfg = adaptive_cfg_init(num_steps=num_steps, **self._adacfg_cfg)

    def _cfg_step_tensor(self, y_cond, y_uncond, strength):
        return (1 + strength) * y_cond - strength * y_uncond

    def _cfg_step(self, y_cond, y_uncond, strength):
        if isinstance(strength, dict):
            return _pytree.tree_map(self._cfg_step_tensor, y_cond, y_uncond, strength)
        else:
            return _pytree.tree_map(partial(self._cfg_step_tensor, strength=strength), y_cond, y_uncond)

    def inner_forward(self, x, t, is_cond, strength, *args_cond, **kwargs_cond):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)
        if is_cond:
            return y_cond

        # Adaptive-CFG: once cond/uncond align, skip the unconditional pass and
        # reconstruct the guidance term g = v_cfg - y_cond from cached anchors.
        adacfg = getattr(self, "_adacfg", None)
        if adacfg is not None and not adaptive_cfg_decide(adacfg, adacfg["last_gamma"]):
            g = forecast_guidance_tree(adacfg["anchors"], adacfg["step"], adacfg["max_order"])
            adacfg["step"] += 1
            adacfg["n_skip"] += 1
            return reconstruct_cfg_tree(y_cond, g)

        args_cond, kwargs_cond = self._make_unconditional_args(
            args_cond,
            kwargs_cond,
        )
        y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
        result = self._cfg_step(y_cond, y_uncond, strength)
        if adacfg is not None:
            # cache the cosine (for the next decision) + the guidance term
            # g = v_cfg - y_cond (convention-agnostic: scalar or per-modality strength).
            adacfg["last_gamma"] = tree_cosine(y_cond, y_uncond)
            g = _pytree.tree_map(lambda r, c: r - c, result, y_cond)
            adacfg["anchors"].append((adacfg["step"], g))
            keep = adacfg["max_order"] + 2
            if len(adacfg["anchors"]) > keep:
                adacfg["anchors"] = adacfg["anchors"][-keep:]
            adacfg["step"] += 1
            adacfg["n_full"] += 1
        return result

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )

def get_strength(strength, interval, t):
    if interval is None:
        return _pytree.tree_map(lambda x: 0.0, strength)
    
    # If interval is not a dict (single tuple), broadcast it
    if not isinstance(interval, dict):
        return _pytree.tree_map(
            lambda x: x if interval[0] <= t <= interval[1] else 0.0,
            strength
        )

    return _pytree.tree_map(
        lambda x, iv: x if iv[0] <= t <= iv[1] else 0.0,
        strength,
        interval
    )

class PointmapCFG(ClassifierFreeGuidance):

    def __init__(self, *args, strength_pm=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength_pm = strength_pm

    def _cfg_step_tensor(self, y_cond, y_uncond, y_unpm, strength, strength_pm):
        # https://arxiv.org/abs/2411.18613
        return y_cond \
            + strength_pm * (y_cond - y_unpm) \
            + strength * (y_unpm - y_uncond)

    def _cfg_step(self, y_cond, y_uncond, y_pm, strength, strength_pm):
        if isinstance(strength, dict):
            return _pytree.tree_map(self._cfg_step_tensor, y_cond, y_uncond, y_pm, strength, strength_pm)
        else:
            return _pytree.tree_map(partial(self._cfg_step_tensor, strength=strength, strength_pm=strength_pm), y_cond, y_uncond, y_pm)

    def inner_forward(self, x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)

        if is_cond:
            return y_cond

        # Adaptive-CFG: once cond/uncond align, skip BOTH extra backbone passes
        # (the pointmap-dropped pass AND the unconditional pass) and reconstruct the
        # full guidance term g = v_cfg - y_cond from cached anchors (~2/3 less compute
        # on aligned steps). g captures both the pointmap and uncond contributions, so
        # this is convention-agnostic.
        adacfg = getattr(self, "_adacfg", None)
        if adacfg is not None and not adaptive_cfg_decide(adacfg, adacfg["last_gamma"]):
            g = forecast_guidance_tree(adacfg["anchors"], adacfg["step"], adacfg["max_order"])
            adacfg["step"] += 1
            adacfg["n_skip"] += 1
            return reconstruct_cfg_tree(y_cond, g)

        force_drop_modalities = self.backbone.condition_embedder.force_drop_modalities
        self.backbone.condition_embedder.force_drop_modalities = ['pointmap', 'rgb_pointmap']
        y_pm = self.backbone(x, t, *args_cond, **kwargs_cond)
        self.backbone.condition_embedder.force_drop_modalities = force_drop_modalities

        args_cond, kwargs_cond = self._make_unconditional_args(
            args_cond,
            kwargs_cond,
        )
        y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
        result = self._cfg_step(y_cond, y_uncond, y_pm, strength, strength_pm)
        if adacfg is not None:
            adacfg["last_gamma"] = tree_cosine(y_cond, y_uncond)
            g = _pytree.tree_map(lambda r, c: r - c, result, y_cond)
            adacfg["anchors"].append((adacfg["step"], g))
            keep = adacfg["max_order"] + 2
            if len(adacfg["anchors"]) > keep:
                adacfg["anchors"] = adacfg["anchors"][-keep:]
            adacfg["step"] += 1
            adacfg["n_full"] += 1
        return result

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                strength_pm = get_strength(self.strength_pm, self.interval, t)
                return self.inner_forward(
                    x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond
                )

class ClassifierFreeGuidanceWithExternalUnconditionalProbability(ClassifierFreeGuidance):

    def __init__(self, *args, use_unconditional_from_flow_matching=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_unconditional_from_flow_matching = use_unconditional_from_flow_matching

    def forward(self, x, t, *args_cond, p_unconditional=None, **kwargs_cond):
        # p_unconditional should be a value in [0, 1], indicating the probability of unconditional

        if p_unconditional is None:
            coin_flip = random.random() < self.p_unconditional
        else:
            coin_flip = random.random() < p_unconditional

        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )
