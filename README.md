<div align="center">

# Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R

<p>
  <a href="https://trust3r-z.github.io/"><img alt="Project page" src="https://img.shields.io/badge/Project_Page-Trust3R-0e7c86?style=flat-square"></a>
  <a href="https://arxiv.org/abs/2605.19539"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.19539-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="https://huggingface.co/SingleBicycle/Trust3R"><img alt="Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97_Checkpoints-Hugging_Face-ffcc4d?style=flat-square"></a>
  <a href="#citation"><img alt="BibTeX" src="https://img.shields.io/badge/Cite-BibTeX-1c2230?style=flat-square"></a>
  <img alt="License: CC BY-NC-SA 4.0" src="https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-6b7280?style=flat-square">
</p>

<p>
  <a href="https://scholar.google.com/citations?user=YfuA8zoAAAAJ&hl=en">Zihao Zhu</a><sup>*</sup>,
  <a href="https://wyzhao23.github.io/">Wenyuan Zhao</a><sup>*</sup>,
  <a href="https://nuochen1203.github.io/">Nuo Chen</a>,
  <a href="https://tiangroup.engr.tamu.edu/">Chao Tian</a><sup>&dagger;</sup>,
  <a href="https://phai-lab.github.io/index.html">Zhiwen Fan</a><sup>&dagger;</sup>
</p>

<p>Department of Electrical and Computer Engineering, Texas A&amp;M University</p>

<p><i>ICML 2026 — Seoul, South Korea</i></p>

<sub><sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>&dagger;</sup>Equal advising</sub>

<br>

<img src="assets/pipeline.png" alt="Trust3R pipeline overview" width="92%">

</div>

---

## Overview

**Trust3R** turns a feed-forward 3D reconstructor into a model that not only predicts
geometry, but also tells you *where* that geometry can be trusted. Starting from a frozen
MASt3R backbone, we add two lightweight heads:

- **Evidential uncertainty head** — predicts the parameters of a Normal-Inverse-Wishart
  (NIW) prior `(κ, ν, Ψ = L Lᵀ)` over each 3D point. Marginalizing yields a closed-form
  multivariate Student-t predictive distribution, giving a calibrated per-pixel
  uncertainty map in a **single forward pass — no ensembles, no Monte Carlo sampling**.
- **Gated residual head** — produces small, gated corrections to the pretrained pointmap,
  perturbing it only where the model is uncertain enough to warrant it.

Trust3R consistently improves risk–coverage and sparsification on ScanNet++, TUM RGB-D,
KITTI, and ETH3D, with moderate inference overhead and no loss of geometric accuracy.

This repository provides the Trust3R models, pretrained checkpoints, training and
inference code, dataset preprocessing, and the evaluation scripts that reproduce the main
results. This release is our MASt3R-based implementation.

---

## Quick start

The code is tested on Linux + CUDA 12.1 + PyTorch 2.x + Python 3.11.

```bash
git clone git@github.com:phai-lab/Trust3R.git
cd Trust3R

conda create -n trust3r python=3.11 cmake=3.14.0 -y
conda activate trust3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y   # match your CUDA
pip install -r requirements.txt

# Trust3R checkpoint
mkdir -p checkpoints
pip install -U "huggingface_hub[cli]"
hf download SingleBicycle/Trust3R \
    trust3r_niw_mast3r_224.pth trust3r_niw_mast3r_224.pth.sha256 \
    --local-dir checkpoints/

# Two images in, pointmaps + per-pixel uncertainty out
python infer.py \
    --checkpoint checkpoints/trust3r_niw_mast3r_224.pth \
    --img1 examples/a.jpg --img2 examples/b.jpg \
    --output-dir infer_out/
```

<details>
<summary>Optional: build the RoPE CUDA kernels</summary>

These accelerate the CroCo attention with rotary positional embeddings. The code falls
back to a pure-PyTorch implementation if you skip this step.

```bash
cd dust3r/croco/models/curope && python setup.py build_ext --inplace && cd ../../../..
```
</details>

Sanity check: `python -c "from mast3r.model import AsymmetricMASt3R; print('OK')"`

---

## Checkpoints

Hosted on the Hugging Face Hub: **[SingleBicycle/Trust3R](https://huggingface.co/SingleBicycle/Trust3R)**

| Checkpoint | Head | Backbone | Train / eval res. |
|---|---|---|---|
| `trust3r_niw_mast3r_224.pth` | **NIW** evidential (full 3×3 covariance) + gated residual — *main model* | frozen MASt3R ViT-L | 224 |
| `trust3r_nig_mast3r_224.pth` | **NIG** evidential (diagonal variance) + gated residual — *ablation* | frozen MASt3R ViT-L | 224 |

```bash
hf download SingleBicycle/Trust3R \
    trust3r_niw_mast3r_224.pth trust3r_nig_mast3r_224.pth \
    trust3r_niw_mast3r_224.pth.sha256 trust3r_nig_mast3r_224.pth.sha256 \
    --local-dir checkpoints/
(cd checkpoints && sha256sum -c *.sha256)
```

Both were trained and evaluated at **224px**; other resolutions are outside the trained
regime and will not match the reported numbers. On `huggingface_hub < 0.34` the CLI is
named `huggingface-cli` instead of `hf`.

Checkpoints inherit the MASt3R / DUSt3R license terms — CC BY-NC-SA 4.0, non-commercial
use only. See [`CHECKPOINTS_NOTICE`](CHECKPOINTS_NOTICE).

---

## Inference

`infer.py` is a minimal pair-forward template: it loads a checkpoint, runs the network on
two images, and writes the raw output tensors.

```bash
python infer.py \
    --checkpoint checkpoints/trust3r_niw_mast3r_224.pth \
    --img1 examples/a.jpg --img2 examples/b.jpg \
    --image-size 224 \
    --output-dir infer_out/
```

`infer_out/preds.pt` is a dict `{'pred1': ..., 'pred2': ...}`. Each entry holds the
standard MASt3R outputs (`pts3d`, `pts3d_in_other_view`, `conf`, descriptors) plus the
Trust3R evidential parameters — for NIW, `xyz_niw_kappa`, `xyz_niw_nu` and `xyz_niw_Psi`.
The script prints the full key list on every run. See
[`eval/uq_eval_utils.py`](eval/uq_eval_utils.py) for how those parameters are turned into
the aleatoric / epistemic / total uncertainty maps used in the paper.

---

## Evaluation

Reproduce the paper's Table 1 (uncertainty ranking) and Table 2 (reconstruction accuracy)
from the released checkpoints:

```bash
CKPT_NIW=checkpoints/trust3r_niw_mast3r_224.pth \
CKPT_NIG=checkpoints/trust3r_nig_mast3r_224.pth \
SCANNETPP_ROOT=/path/to/scannetpp_test_set_processed \
ETH3D_ROOT=/path/to/eth3d_processed_dust3r \
KITTI_ROOT=/path/to/kitti_val_selection_processed_dust3r \
TUM_ROOT=/path/to/tum_processed_v1 \
OUT_DIR=eval_out/trust3r \
bash eval/reproduce_table1_table2.sh
```

Writes `table1_uq.csv`, `table2_recon.csv`, `table3_nll.csv` and the Figure 3
risk–coverage / sparsification curves into `OUT_DIR`; read the `ours_niw_epi` rows. One
GPU, roughly 2–3 hours.

`eval/evaluate_uq.py` is a general UQ evaluator — MASt3R confidence, heteroscedastic
Gaussian, MC Dropout and Deep Ensembles are all scored through the same protocol.
**[`eval/README.md`](eval/README.md)** documents the protocol, the expected numbers, the
settings that must not be changed, and how to benchmark your own uncertainty head.

---

## Datasets

Download each dataset from its official source, then run the matching script in
[`dust3r/datasets_preprocess/`](dust3r/datasets_preprocess) to produce the
DUSt3R-style layout the dataset classes expect.

| Dataset | Used for | Official source |
|---|---|---|
| ScanNet++ | train + test | https://kaldir.vc.in.tum.de/scannetpp/ |
| ARKitScenes | train | https://github.com/apple/ARKitScenes |
| Waymo Open Dataset | train | https://waymo.com/open/ |
| MegaDepth | train | https://www.cs.cornell.edu/projects/megadepth/ |
| TUM RGB-D | test | https://cvg.cit.tum.de/data/datasets/rgbd-dataset |
| KITTI (depth prediction val selection) | test | https://www.cvlibs.net/datasets/kitti/eval_depth.php |
| ETH3D | test (ablation) | https://www.eth3d.net/datasets |

Each dataset carries its own license and access terms; ScanNet++, Waymo and ETH3D require
registration.

---

## Training

The launchers are environment-variable driven — no need to edit them to change paths, GPU
id, or hyperparameters. First fetch the MASt3R backbone the heads are trained on top of:

```bash
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
```

Then, for the released NIW model — frozen backbone, 224px, 10 epochs over 150k pairs per
epoch from the four-dataset mix:

```bash
GPU=0 \
SCANNET_ROOT=/path/to/scannetpp_processed \
ARKIT_ROOT=/path/to/arkitscenes_processed \
WAYMO_ROOT=/path/to/waymo_processed \
MEGA_ROOT=/path/to/megadepth_processed \
bash scripts/run_gated_niw_train.sh
```

`scripts/run_gated_nig_train.sh` trains the NIG ablation with the same protocol.
Checkpoints and TensorBoard logs land under `output*/`, which is git-ignored.

The remaining launchers are baselines and later variants, not the recipe behind the
released checkpoints: `run_hetero_train.sh` (heteroscedastic Gaussian),
`run_ens_parallel.sh` (deep ensembles), and `run_gated_{niw,nig}_train_grpost.sh` (a
two-stage 448px variant with bilinear residual upsampling). See the scripts for their
hyperparameters.

---

## Code structure

| Component | Location |
|---|---|
| Trust3R model, evidential heads, gated residual | [`mast3r/`](mast3r) |
| Evidential losses | [`mast3r/losses_evidential.py`](mast3r/losses_evidential.py) |
| Training | [`train.py`](train.py), [`scripts/`](scripts) |
| Inference | [`infer.py`](infer.py) |
| Evaluation | [`eval/`](eval) |
| Dataset preprocessing | [`dust3r/datasets_preprocess/`](dust3r/datasets_preprocess) |
| Vendored DUSt3R + CroCo | [`dust3r/`](dust3r) |

---

## Citation

```bibtex
@misc{zhu2026trustnotevidentialuncertainty,
      title         = {Trust It or Not: Evidential Uncertainty for Feed-Forward 3D Reconstruction with Trust3R},
      author        = {Zihao Zhu and Wenyuan Zhao and Nuo Chen and Chao Tian and Zhiwen Fan},
      year          = {2026},
      eprint        = {2605.19539},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CV},
      url           = {https://arxiv.org/abs/2605.19539},
}
```

## Acknowledgements

This codebase builds directly on the excellent
[MASt3R](https://github.com/naver/mast3r),
[DUSt3R](https://github.com/naver/dust3r),
and [CroCo](https://github.com/naver/croco) releases from Naver Labs Europe.
We thank the authors for open-sourcing their work.

## License

Trust3R is released under **CC BY-NC-SA 4.0** (non-commercial use only), inherited from
MASt3R / DUSt3R / CroCo. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE),
[`dust3r/LICENSE`](dust3r/LICENSE), and [`dust3r/croco/LICENSE`](dust3r/croco/LICENSE)
for the full terms.
