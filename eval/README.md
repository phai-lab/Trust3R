# Evaluation

This directory holds the uncertainty-quantification benchmark used in the paper.
It does two things:

1. **Reproduce** the paper's tables from the released checkpoints, exactly.
2. **Benchmark any other method** — a baseline, or your own model — under the
   identical protocol, so the comparison is apples to apples.

| File | Purpose |
|---|---|
| `evaluate_uq.py` | Evaluator. Computes AURC, AUSE, Spearman ρ, Sim(3)-aligned MAE/RMSE, NLL, and the risk–coverage / sparsification curves. |
| `uq_eval_utils.py` | Metrics, Sim(3) alignment, valid-pixel masking, per-method uncertainty readouts. |
| `reproduce_table1_table2.sh` | One command → the paper's Table 1, 2, 5, 6 and Figure 3. |
| `evaluate_baselines.sh` | Same protocol for the baselines (MASt3R confidence, heteroscedastic, MC Dropout, Deep Ensembles). |

---

## 1. Reproducing the paper

### Outputs

| Paper table | Metric | Output file | Row to read |
| --- | --- | --- | --- |
| Table 1 | AURC ↓, AUSE ↓, Spearman ρ ↑ | `table1_uq.csv` | `ours_niw_epi` |
| Table 2 | MAE ↓, RMSE ↓ (Sim(3)-aligned) | `table2_recon.csv` | `ours_niw_epi` |
| Table 5 | uncertainty-readout ablation on ETH3D | `table1_uq.csv` | `ours_niw_{alea,total,epi}` |
| Table 6 | NIG vs NIW evidential family | `table1_uq.csv` | `ours_nig_epi` vs `ours_niw_epi` |
| Figure 3 | risk–coverage + sparsification curves | `curves_<BENCH>_<method>.npz` / `.png` | — |

`table3_nll.csv` (appendix NLL) is written by the same run.

### Checkpoints

Both released checkpoints are required — see the note on exactness below.

```bash
mkdir -p checkpoints
hf download SingleBicycle/Trust3R \
    trust3r_niw_mast3r_224.pth trust3r_nig_mast3r_224.pth \
    trust3r_niw_mast3r_224.pth.sha256 trust3r_nig_mast3r_224.pth.sha256 \
    --local-dir checkpoints/
(cd checkpoints && sha256sum -c *.sha256)
```

### Test sets

All four benchmarks are read from DUSt3R-style preprocessed directories at
`resolution=224`. Download each dataset from its official source (linked in the
root README) and run the matching script in `dust3r/datasets_preprocess/`:

| Benchmark | Dataset class | Preprocessing script |
| --- | --- | --- |
| ScanNet++ | `ScanNetpp(split='test')` | `preprocess_scannetpp.py` |
| ETH3D | `ETH3DProcessedDust3R(split='test')` | `preprocess_eth3d.py` |
| KITTI | `KITTIDust3RProcessed(split='test')` | `preprocess_kitti_depth.py` |
| TUM RGB-D | `TUMRGBD(split='test')` | `preprocess_tum_rgbd_final.py` |

### Run

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

Single GPU, roughly 10 minutes per (benchmark, method) pair at 5000 pairs — about
4 hours for the full 4-benchmark × 6-method reproduction. `matplotlib` is only
needed for the curve `.png` files.

### Expected numbers

Table 1 / Table 2, `ours_niw_epi`:

| Benchmark | AURC ↓ | AUSE ↓ | ρ ↑ | MAE ↓ | RMSE ↓ |
| --- | --- | --- | --- | --- | --- |
| ScanNet++ | 0.123280 | 0.044445 | 0.493043 | 0.195861 | 0.284884 |
| TUM RGB-D | 0.048135 | 0.017793 | 0.516943 | 0.087329 | 0.149637 |
| KITTI | 0.986890 | 0.443080 | 0.459576 | 1.664771 | 3.077223 |
| **Avg** | **0.386102** | **0.168439** | **0.489854** | — | — |

Table 5 / Table 6 on ETH3D:

| Row | AURC ↓ | AUSE ↓ | ρ ↑ |
| --- | --- | --- | --- |
| `ours_niw_alea` | 0.317490 | 0.145209 | 0.309347 |
| `ours_niw_total` | 0.306399 | 0.134117 | 0.345515 |
| `ours_niw_epi` | 0.304036 | 0.131755 | 0.348302 |
| `ours_nig_epi` | 0.321256 | 0.149341 | 0.322906 |

Pair subsets are selected deterministically from `--subset_seed 0` and written to
`<OUT_DIR>/<BENCH>/subset_pairs_idx.txt`, so a correct setup reproduces these
values exactly rather than approximately.

---

## 2. Benchmarking another method

`evaluate_uq.py` is a general evaluator, not a Trust3R-only script. Point it at a
checkpoint and pick a method:

| `--methods` value | Uncertainty signal | Checkpoint flag |
| --- | --- | --- |
| `conf` | MASt3R's predicted confidence | `--ckpt_conf` |
| `hetero` | heteroscedastic Gaussian per-pixel variance | `--ckpt_hetero` |
| `mc_dropout` | MC Dropout over `--mc_samples` passes | `--mc_ckpt` |
| `ensemble` | Deep Ensembles variance | `--ensemble_ckpts a.pth,b.pth,...` |
| `ours_nig_{total,alea,epi}` | NIG evidential | `--ckpt_nig` |
| `ours_niw_{total,alea,epi}` | NIW evidential | `--ckpt_niw` |

The baselines wrapper covers the common cases:

```bash
METHODS=conf \
CKPT_CONF=checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
BENCHMARKS=ScanNetpp,TUM \
SCANNETPP_ROOT=... TUM_ROOT=... \
OUT_DIR=eval_out/baseline_conf \
bash eval/evaluate_baselines.sh
```

### Adding a new uncertainty head

`extract_method_outputs` in `uq_eval_utils.py` is the single seam between a model's
raw forward output and the scalar per-pixel uncertainty map the metrics consume.
To score a new method, add a branch there that returns a `(H, W)` uncertainty map
(higher = less trustworthy) plus the predicted pointmap, then register the method
name in `METHOD_MODEL_ARG` and `ckpt_map` in `evaluate_uq.py`. Everything
downstream — masking, alignment, curves, tables — is shared, which is what keeps
the comparison fair.

---

## Protocol notes

Details that change the numbers if you deviate from them.

- **Readout.** The paper's Trust3R row is the **epistemic** uncertainty
  `u_epi = Ψ / (κ(ν−4))` — the `ours_niw_epi` row, not total or aleatoric. All
  three readouts come from the same forward pass, so MAE/RMSE are identical across
  them up to float noise.

- **Alignment.** MAE/RMSE and the ranking metrics use a **per-image Sim(3)**
  alignment (`--sim3_align`), so they measure local geometric reliability rather
  than global pose/scale drift. NLL is computed *without* alignment
  (`--nll_no_sim3 1`).

- **Resolution.** These checkpoints are trained and evaluated at 224px. Other
  resolutions will not match the tables.

- **Spearman ρ depends on the method list.** AURC, AUSE, MAE and RMSE are
  deterministic. ρ is computed on a 200k-point subsample (`--rho_max_points`)
  drawn from a *single* RNG stream created once per run, so removing a method or a
  benchmark shifts the stream and moves ρ by roughly ±0.001. This is why
  `reproduce_table1_table2.sh` runs all six methods over all four benchmarks in a
  fixed order even when only the NIW rows are of interest. AURC/AUSE/MAE/RMSE
  reproduce regardless of the method list; ρ only reproduces under the full run.

- **Gated-residual settings are read from the model expression**, not from the
  checkpoint's training args. `reproduce_table1_table2.sh` passes the expression
  stored in the checkpoint, which is what the published tables used — don't add or
  remove `head_type` keyword arguments when reproducing.
