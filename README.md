<div align="center">

# SAM 3D Objects + HiCache

**Image-to-3D with a training-free, tree-aware HiCache velocity cache on the slat-stage flow matching.**

*A fork of [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (Meta) that skips most
network evaluations in the slat-stage flow-matching sampler by **forecasting** the cached velocity with a
scaled-Hermite polynomial ([HiCache](https://arxiv.org/abs/2508.16984)) — generalised to SAM3D's
**PyTree** (structured) velocities.*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![no&#8209;monkeypatch](https://img.shields.io/badge/native%20hook-no%20monkeypatch-2e6db0)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![license SAM](https://img.shields.io/badge/license-SAM-2e6db0)

</div>

---

## What this is

[SAM 3D Objects](https://ai.meta.com/sam3d/) reconstructs full 3D shape, texture, and layout from a
single masked image. The expensive part of inference is the **slat-stage flow-matching** sampler: an
Euler ODE solve that calls the backbone once per step over many steps.

This fork adds **HiCache** to that sampler — *training-free, geometry-preserving* feature caching. On
most solver steps it **skips the backbone** and instead **forecasts** the (CFG-combined) velocity from
cached anchors, calling the network only every `interval` steps. The single twist over upstream HiCache
is that SAM3D's solver state and backbone velocity are `torch.utils._pytree` **structures**, not single
tensors, so the cache is **tree-aware**: the Hermite/finite-difference coefficients are scalars shared by
all leaves, and a forecast is one `tree_map` per order.

> This is the **HiCache (Hermite polynomial)** fork. For the **exponential (DMD/Prony)** forecaster that
> stays lossless out to a *larger* skip interval, see the sibling fork **`sam3d-plus-plus`** and the
> standalone library [`hicache-plus-plus`](../hicache-plus-plus).

## Method

A flow-matching sampler integrates `dx/dt = v_θ(x, t)`. HiCache treats the per-step velocity as a smooth
trajectory in the step index and, on skipped steps, **extrapolates it from cached finite-difference
derivatives using a dual-scaled physicist's-Hermite polynomial** instead of evaluating `v_θ` — skipping
`(interval-1)/interval` of the backbone passes. Here the velocity is a PyTree, so each derivative order is
stored as a tree and the scalar Hermite coefficient `H̃_n(-k)/n!` is applied leafwise via `tree_map`
(`accel.py`). A complementary **Adaptive-CFG** path drops the unconditional pass once conditional and
unconditional velocities align and rebuilds the guidance term `g = w·(y_cond − y_uncond)` from cached
anchors. The hook is **native** — the Euler solver and CFG module call the cache helpers directly; there
is no runtime monkey-patching.

## Enable (real API)

HiCache lives on the Euler solver of the slat-stage `FlowMatching` module and is exposed as a convenience
method on the flow-matching object (`flow_matching/model.py`, `flow_matching/solver.py`):

```python
# fm is the slat-stage FlowMatching module inside the SAM 3D Objects pipeline.
# HiCache requires the Euler solver (one dynamics_fn eval per step).

fm.enable_hicache(
    interval=3,        # call the backbone every 3rd step; forecast the other 2
    max_order=1,       # Hermite/finite-difference order
    first_enhance=2,   # always run full for the first 2 (warm-up) steps
    end_enhance=None,  # always run full for the final step(s); None = last step
    sigma=0.5,         # Hermite scale (0,1)
)

# optional, composable: drop the unconditional CFG pass once it aligns
fm.enable_adaptive_guidance(gamma_bar=0.94, warmup=2)

# ... run the normal SAM 3D Objects inference (generate / demo.py) ...

fm.disable_hicache()            # restore the exact uncached sampler
fm.disable_adaptive_guidance()
```

Internally `enable_hicache` is also available directly on the solver (`ODESolver.enable_hicache(...)`); the
solver resets the per-trajectory cache at the start of every run and only activates it for `Euler`. A CPU
unit test that needs no GPU or model weights ships in the accel module:

```bash
python -m sam3d_objects.model.backbone.generator.flow_matching.accel
```

## Results

On the slat-stage `FlowMatching` (real SAM 3D Objects weights, F1 vs the uncached baseline), **HiCache
(Hermite) is geometry-lossless (F1 = 1.000) out to interval-3 at ~1.44×**. The polynomial basis drifts
beyond that, which is exactly the skip ceiling the exponential basis is built to push.

| config | speedup | F1 vs baseline |
|---|---:|---:|
| vanilla (uncached) | 1.00× | **1.000** |
| **HiCache i3** | **1.44×** | **1.000** |

For the **exponential (DMD/Prony)** forecaster that holds **F1 = 1.000 out to interval-6 at 1.56×** on the
same stage — where Hermite is lossless only to interval-3 — see the sibling fork **`sam3d-plus-plus`** and
the standalone library **[`hicache-plus-plus`](../hicache-plus-plus)** (with the controlled forecast
microbenchmark and the full Hunyuan3D / SAM3D tables).

## Attribution

- **SAM 3D Objects** © Meta Platforms, Inc. — model, weights, and code under the [SAM License](./LICENSE).
  The full upstream README (install, demos, benchmark, citation) is preserved below.
- **HiCache** — scaled-Hermite velocity forecasting, [arXiv:2508.16984](https://arxiv.org/abs/2508.16984).
  The Hermite basis here is a clean reimplementation, generalised to PyTree velocities.
- **Adaptive Guidance** — [arXiv:2312.12487](https://arxiv.org/abs/2312.12487).

The acceleration code added by this fork lives in
`sam3d_objects/model/backbone/generator/flow_matching/{accel,solver,model}.py`.

## Citation

If you use this fork, please cite the base model and the acceleration methods it builds on.

**SAM 3D Objects** (base model):

```bibtex
@article{sam3dteam2025sam3d3dfyimages,
      title={SAM 3D: 3Dfy Anything in Images}, 
      author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
      year={2025},
      eprint={2511.16624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16624}, 
}
```

**HiCache** (scaled-Hermite velocity forecasting):

```bibtex
@misc{hicache2025,
      title={HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting},
      eprint={2508.16984},
      archivePrefix={arXiv},
      year={2025}
}
```

**Adaptive Guidance**:

```bibtex
@misc{adaptiveguidance2023,
      title={Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models},
      eprint={2312.12487},
      archivePrefix={arXiv},
      year={2023}
}
```

---
---

# SAM 3D

SAM 3D Objects is one part of SAM 3D, a pair of models for object and human mesh reconstruction.  If you’re looking for SAM 3D Body, [click here](https://github.com/facebookresearch/sam-3d-body).

# SAM 3D Objects

**SAM 3D Team**, [Xingyu Chen](https://scholar.google.com/citations?user=gjSHr6YAAAAJ&hl=en&oi=sra)\*, [Fu-Jen Chu](https://fujenchu.github.io/)\*, [Pierre Gleize](https://scholar.google.com/citations?user=4imOcw4AAAAJ&hl=en&oi=ao)\*, [Kevin J Liang](https://kevinjliang.github.io/)\*, [Alexander Sax](https://alexsax.github.io/)\*, [Hao Tang](https://scholar.google.com/citations?user=XY6Nh9YAAAAJ&hl=en&oi=sra)\*, [Weiyao Wang](https://sites.google.com/view/weiyaowang/home)\*, [Michelle Guo](https://scholar.google.com/citations?user=lyjjpNMAAAAJ&hl=en&oi=ao), [Thibaut Hardin](https://github.com/Thibaut-H), [Xiang Li](https://ryanxli.github.io/)⚬, [Aohan Lin](https://github.com/linaohan), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ziqi Ma](https://ziqi-ma.github.io/)⚬, [Anushka Sagar](https://www.linkedin.com/in/anushkasagar/), [Bowen Song](https://scholar.google.com/citations?user=QQKVkfcAAAAJ&hl=en&oi=sra)⚬, [Xiaodong Wang](https://scholar.google.com/citations?authuser=2&user=rMpcFYgAAAAJ), [Jianing Yang](https://jedyang.com/)⚬, [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen/)⚬, [Piotr Dollár](https://pdollar.github.io/)†, [Georgia Gkioxari](https://georgiagkioxari.com/)†, [Matt Feiszli](https://scholar.google.com/citations?user=A-wA73gAAAAJ&hl=en&oi=ao)†§, [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)†§

***Meta Superintelligence Labs***

*Core contributor (Alphabetical, Equal Contribution), ⚬Intern, †Project leads, §Equal Contribution

[[`Paper`](https://ai.meta.com/research/publications/sam-3d-3dfy-anything-in-images/)] [[`Code`](https://github.com/facebookresearch/sam-3d-objects)] [[`Website`](https://ai.meta.com/sam3d/)] [[`Demo`](https://www.aidemos.meta.com/segment-anything/editor/convert-image-to-3d)] [[`Blog`](https://ai.meta.com/blog/sam-3d/)] [[`BibTeX`](#citing-sam-3d-objects)] [[`Roboflow`](https://blog.roboflow.com/sam-3d/)]

**SAM 3D Objects** is a foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image, excelling in real-world scenarios with occlusion and clutter by using progressive training and a data engine with human feedback. It outperforms prior 3D generation models in human preference tests on real-world objects and scenes. We released code, weights, online demo, and a new challenging benchmark.


<p align="center"><img src="doc/intro.png"/></p>

-----

<p align="center"><img src="doc/arch.png"/></p>

## Latest updates

* **06/02/2026** - [3D Artist Object Set](https://ai.meta.com/datasets/sa-3dao-sam-3d-artist-objects/) and [HF Leaderboard](https://huggingface.co/spaces/facebook/sa3dao-leaderboard) are out.
* **06/01/2026** - Encoder weights are out.
* **11/19/2025** - Checkpoints Launched, Web Demo and Paper are out.

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

## Single or Multi-Object 3D Generation

SAM 3D Objects can convert masked objects in an image, into 3D models with pose, shape, texture, and layout. SAM 3D is designed to be robust in challenging natural images, handling small objects and occlusions, unusual poses, and difficult situations encountered in uncurated natural scenes like this kidsroom:

<p align="center">
  <img src="notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png" width="55%"/>
  <img src="doc/kidsroom_transparent.gif" width="40%"/>
</p>

For a quick start, run `python demo.py` or use the the following lines of code:

```python
import sys

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image and mask
image = load_image("notebook/images/shutterstock_stylish_kidsroom_1640806567/image.png")
mask = load_single_mask("notebook/images/shutterstock_stylish_kidsroom_1640806567", index=14)

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
```

For  more details and multi-object reconstruction, please take a look at out two jupyter notebooks:
* [single object](notebook/demo_single_object.ipynb)
* [multi object](notebook/demo_multi_object.ipynb)


## SAM 3D Body

[SAM 3D Body (3DB)](https://github.com/facebookresearch/sam-3d-body) is a robust promptable foundation model for single-image 3D human mesh recovery (HMR).

As a way to combine the strengths of both **SAM 3D Objects** and **SAM 3D Body**, we provide an example notebook that demonstrates how to combine the results of both models such that they are aligned in the same frame of reference. Check it out [here](notebook/demo_3db_mesh_alignment.ipynb).

## License

The SAM 3D Objects model checkpoints and code are licensed under [SAM License](./LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributors

The SAM 3D Objects project was made possible with the help of many contributors.

Robbie Adkins,
Paris Baptiste,
Karen Bergan,
Kai Brown,
Michelle Chan,
Ida Cheng,
Khadijat Durojaiye,
Patrick Edwards,
Daniella Factor,
Facundo Figueroa,
Rene  de la Fuente,
Eva Galper,
Cem Gokmen,
Alex He,
Enmanuel Hernandez,
Dex Honsa,
Leonna Jones,
Arpit Kalla,
Kris Kitani,
Helen Klein,
Kei Koyama,
Robert Kuo,
Vivian Lee,
Alex Lende,
Jonny Li,
Kehan Lyu,
Faye Ma,
Mallika Malhotra,
Sasha Mitts,
William Ngan,
George Orlin,
Peter Park,
Don Pinkus,
Roman Radle,
Nikhila Ravi,
Azita Shokrpour,
Jasmine Shone,
Zayida Suber,
Phillip Thomas,
Tatum Turner,
Joseph Walker,
Meng Wang,
Claudette Ward,
Andrew Westbury,
Lea Wilken,
Nan Yang,
Yael Yungster


## Citing SAM 3D Objects

If you use SAM 3D Objects in your research, please use the following BibTeX entry.

```
@article{sam3dteam2025sam3d3dfyimages,
      title={SAM 3D: 3Dfy Anything in Images}, 
      author={SAM 3D Team and Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
      year={2025},
      eprint={2511.16624},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16624}, 
}
```

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.
