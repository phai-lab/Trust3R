#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty-quantification evaluation for Trust3R.

Computes AURC / AUSE / Spearman rho (Table 1), Sim(3)-aligned MAE / RMSE (Table 2)
and optionally NLL, for one or more benchmarks and UQ methods.

See eval/README.md for the exact invocation that reproduces the paper tables.
"""

import os
import sys
import json
import argparse
import hashlib
import re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uq_eval_utils import (  # noqa: E402
    add_repo_root, build_eval_context, eval_from_str, load_state_dict,
    batchify_view, compute_gt_xyz_cam, get_valid_mask, resize_mask_to_hw, resize_to_hw,
    estimate_sim3, apply_sim3, per_image_error, mae_rmse,
    gaussian_nll_diag, calibrate_conf_scale_isotropic,
    nig_nll_3d, niw_nll_3d,
    base_method, extract_method_outputs,
)


def is_rank0() -> bool:
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    try:
        return torch.distributed.get_rank() == 0
    except Exception:
        return True


def parse_benchmarks(s: str):
    """
    Format:
      "ScanNetpp:ScanNetpp(...);TUM:TUMRGBD(...);KITTI:KITTIDepth(...);ETH3D:ETH3D(...)".
    """
    items = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        name, expr = part.split(":", 1)
        items.append((name.strip(), expr.strip()))
    return items


def save_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


SCENE_KEYS = ("scene_id", "scene", "scene_name", "scan", "sequence", "seq_name")


def get_scene_id(view1, view2, idx):
    for view in (view1, view2):
        if isinstance(view, dict):
            for k in SCENE_KEYS:
                if k in view and view[k] is not None:
                    return str(view[k])
    return None


def get_scene_id_from_dataset(dataset, idx):
    # Fast path for datasets exposing pairs/sceneids (e.g., ScanNetpp).
    try:
        if hasattr(dataset, "pairs") and hasattr(dataset, "sceneids"):
            pair = dataset.pairs[idx]
            if isinstance(pair, (list, tuple, np.ndarray)) and len(pair) > 0:
                sid = dataset.sceneids[int(pair[0])]
                if hasattr(dataset, "scenes"):
                    return str(dataset.scenes[int(sid)])
                return str(int(sid))
    except Exception:
        pass
    return None


def stable_hash01(s: str) -> float:
    d = hashlib.md5(s.encode("utf-8")).digest()
    v = int.from_bytes(d[:4], "little")
    return (v % 10000) / 10000.0


def extract_split(expr: str):
    m = re.search(r"split\s*=\s*[\"']([^\"']+)[\"']", expr)
    return m.group(1) if m else None


def spearmanr_from_samples(u: np.ndarray, e: np.ndarray) -> float:
    if u.size == 0 or e.size == 0 or u.size != e.size:
        return float("nan")
    ur = u.argsort().argsort().astype(np.float64)
    er = e.argsort().argsort().astype(np.float64)
    ur = ur - ur.mean()
    er = er - er.mean()
    denom = (np.sqrt((ur ** 2).mean()) * np.sqrt((er ** 2).mean()) + 1e-12)
    return float((ur * er).mean() / denom)


class HistRC:
    def __init__(self, bins: int, u_log_lo: float, u_log_hi: float,
                 e_log_lo: float, e_log_hi: float, device: torch.device,
                 rho_max_points: int = 0, rho_max_per_update: int = 2048):
        self.bins = bins
        self.u_log_lo = float(u_log_lo)
        self.u_log_hi = float(u_log_hi)
        self.e_log_lo = float(e_log_lo)
        self.e_log_hi = float(e_log_hi)
        self.count_u = torch.zeros(bins, device=device, dtype=torch.float64)
        self.sumerr_u = torch.zeros(bins, device=device, dtype=torch.float64)
        self.count_e = torch.zeros(bins, device=device, dtype=torch.float64)
        self.sumerr_e = torch.zeros(bins, device=device, dtype=torch.float64)
        self.total = 0.0
        self.rho_max_points = int(rho_max_points) if rho_max_points else 0
        self.rho_max_per_update = int(rho_max_per_update)
        self.rho_u = []
        self.rho_e = []
        self.rho_count = 0
        self.rho_seen = 0

    def _bin_log(self, x: torch.Tensor, lo: float, hi: float, eps: float = 1e-12) -> torch.Tensor:
        x = torch.log(torch.clamp(x, min=eps))
        x = torch.clamp(x, lo, hi)
        span = max(hi - lo, eps)
        t = (x - lo) / span
        b = torch.clamp((t * (self.bins - 1)).long(), 0, self.bins - 1)
        return b

    def update(self, unc_flat: torch.Tensor, err_flat: torch.Tensor, rng: np.random.RandomState):
        if err_flat.numel() == 0:
            return
        bu = self._bin_log(unc_flat, self.u_log_lo, self.u_log_hi)
        be = self._bin_log(err_flat, self.e_log_lo, self.e_log_hi)

        cu = torch.bincount(bu, minlength=self.bins).double()
        su = torch.bincount(bu, weights=err_flat.double(), minlength=self.bins)
        ce = torch.bincount(be, minlength=self.bins).double()
        se = torch.bincount(be, weights=err_flat.double(), minlength=self.bins)

        self.count_u += cu
        self.sumerr_u += su
        self.count_e += ce
        self.sumerr_e += se
        self.total += float(err_flat.numel())

        if self.rho_max_points <= 0:
            return

        n = int(err_flat.numel())
        k = min(self.rho_max_per_update, n)
        if k <= 0:
            return

        replace = n < k
        idx = rng.choice(n, size=k, replace=replace)
        idx_t = torch.from_numpy(idx).to(err_flat.device)
        u_vals = unc_flat.index_select(0, idx_t).detach().cpu().numpy()
        e_vals = err_flat.index_select(0, idx_t).detach().cpu().numpy()

        for u, e in zip(u_vals, e_vals):
            self.rho_seen += 1
            if self.rho_count < self.rho_max_points:
                self.rho_u.append(float(u))
                self.rho_e.append(float(e))
                self.rho_count += 1
            else:
                j = rng.randint(0, self.rho_seen)
                if j < self.rho_max_points:
                    self.rho_u[j] = float(u)
                    self.rho_e[j] = float(e)

    def finalize_curves(self):
        if self.total <= 0:
            return float("nan"), float("nan"), (np.array([0.0, 1.0]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
        cum_cu = torch.cumsum(self.count_u, dim=0)
        cum_su = torch.cumsum(self.sumerr_u, dim=0)
        cov_u = (cum_cu / cum_cu[-1].clamp_min(1.0)).cpu().numpy()
        risk_u = (cum_su / cum_cu.clamp_min(1.0)).cpu().numpy()

        cum_ce = torch.cumsum(self.count_e, dim=0)
        cum_se = torch.cumsum(self.sumerr_e, dim=0)
        cov_e = (cum_ce / cum_ce[-1].clamp_min(1.0)).cpu().numpy()
        risk_e = (cum_se / cum_ce.clamp_min(1.0)).cpu().numpy()

        grid = np.linspace(0.0, 1.0, 1001)
        risk_u_g = np.interp(grid, cov_u, risk_u)
        risk_e_g = np.interp(grid, cov_e, risk_e)

        aurc = float(np.trapz(risk_u_g, grid))
        ause = float(np.trapz(risk_u_g - risk_e_g, grid))
        return aurc, ause, (grid, risk_u_g, risk_e_g)


def save_rc_curve(out_path_npz, out_path_png, grid, risk_model, risk_oracle, aurc, ause):
    np.savez_compressed(
        out_path_npz,
        grid=grid,
        risk_model=risk_model,
        risk_oracle=risk_oracle,
        aurc=aurc,
        ause=ause,
    )
    if not out_path_png:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(grid, risk_model, label="model")
    ax.plot(grid, risk_oracle, label="oracle")
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk")
    ax.set_title(f"AURC={aurc:.4g} AUSE={ause:.4g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)


def save_rc_multi(out_path_png, curves: dict, title: str):
    if not curves:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for name, (grid, risk_model) in curves.items():
        ax.plot(grid, risk_model, label=name)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    plt.close(fig)


def resize_bhw(x: torch.Tensor, hw):
    if x.shape[-2:] == hw:
        return x
    if x.ndim == 3:
        x = x.unsqueeze(1)
        x = resize_to_hw(x, hw, mode="bilinear")
        return x[:, 0]
    if x.ndim == 4 and x.shape[1] == 1:
        x = resize_to_hw(x, hw, mode="bilinear")
        return x[:, 0]
    raise ValueError(f"resize_bhw expects [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def maybe_align_for_error(pred_xyz: torch.Tensor, unc_map: torch.Tensor, sim3, xyz_reduce: str):
    if sim3 is None:
        return pred_xyz, unc_map
    s, r, t = sim3
    pred_aligned = apply_sim3(pred_xyz, s, r, t)
    if xyz_reduce == "trace":
        unc_aligned = unc_map * (s ** 2)
    else:
        unc_aligned = unc_map * abs(s)
    return pred_aligned, unc_aligned


def main():
    ap = argparse.ArgumentParser("Evaluate UQ tables (AURC/AUSE/rho, MAE/RMSE, NLL).")
    ap.add_argument("--benchmarks", type=str, required=True,
                    help="name:DatasetExpr;name2:Expr2;...")
    ap.add_argument("--model", type=str, required=True)

    # Optional per-method model expressions (useful when methods require different head_type).
    # If not provided, falls back to --model.
    ap.add_argument("--model_conf", type=str, default=None)
    ap.add_argument("--model_hetero", type=str, default=None)
    ap.add_argument("--model_nig", type=str, default=None)
    ap.add_argument("--model_niw", type=str, default=None)
    ap.add_argument("--model_ensemble", type=str, default=None)
    ap.add_argument("--model_mc", type=str, default=None,
                    help="Model expr for mc_dropout (if different from --model).")

    ap.add_argument("--methods", type=str, default="conf,mc_dropout,ensemble,hetero,ours_nig,ours_niw")
    ap.add_argument("--device", type=str, default="cuda")

    # checkpoints
    ap.add_argument("--ckpt_conf", type=str, default=None)
    ap.add_argument("--ckpt_hetero", type=str, default=None)
    ap.add_argument("--ckpt_nig", type=str, default=None)
    ap.add_argument("--ckpt_niw", type=str, default=None)
    ap.add_argument("--ensemble_ckpts", type=str, default="")
    ap.add_argument("--mc_ckpt", type=str, default=None)

    # settings
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--scene_fraction", type=float, default=1.0)
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--mc_samples", type=int, default=16)
    ap.add_argument("--conf_unc_mode", type=str, default="inv", choices=["inv", "neglog"])
    ap.add_argument("--xyz_reduce", type=str, default="trace", choices=["trace", "l2"])
    ap.add_argument("--ours_unc_variant", type=str, default="total", choices=["total", "alea", "epi"])
    ap.add_argument("--sim3_align", action="store_true")
    ap.add_argument("--no_sim3_align", dest="sim3_align", action="store_false")
    ap.set_defaults(sim3_align=True)
    ap.add_argument("--nll_no_sim3", type=int, default=1, choices=[0, 1],
                    help="When 1 (default), NLL ignores Sim3 even if --sim3_align is set.")

    # NLL
    ap.add_argument("--do_nll", action="store_true")
    ap.add_argument("--nll_sigma2_floor", type=float, default=0.0,
                    help="Isotropic observation-noise floor (in xyz units^2) added ONLY when computing NLL for mc_dropout/ensemble.")
    ap.add_argument("--nll_mode_for_dropout_ens", type=str, default="gauss_mm",
                    choices=["gauss_mm"])  # keep simple
    ap.add_argument("--do_conf_calib", type=int, default=0, choices=[0, 1],
                    help="Enable conf calibration (1 to enable); default 0 to avoid leakage.")
    ap.add_argument("--allow_calib_leak", action="store_true",
                    help="Allow conf calibration on the same eval split (not recommended).")
    ap.add_argument("--conf_calib_id_bench", type=str, default=None,
                    help="Which benchmark name is ID val for conf calibration scale.")
    ap.add_argument("--conf_calib_max_pairs", type=int, default=2000)

    # RC curves / Spearman
    ap.add_argument("--curve_bins", type=int, default=4096)
    ap.add_argument("--u_log_lo", type=float, default=-20.0)
    ap.add_argument("--u_log_hi", type=float, default=20.0)
    ap.add_argument("--e_log_lo", type=float, default=-20.0)
    ap.add_argument("--e_log_hi", type=float, default=20.0)
    ap.add_argument("--rho_max_points", type=int, default=200000)

    # Backward-compatible aliases
    ap.add_argument("--rc_bins", type=int, default=None)
    ap.add_argument("--rc_u_log_max", type=float, default=None)
    ap.add_argument("--rc_e_log_max", type=float, default=None)
    ap.add_argument("--rho_samples", type=int, default=None)

    # debug
    ap.add_argument("--debug_keys", action="store_true")
    ap.add_argument("--debug_shapes", action="store_true")
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    add_repo_root()
    ctx = build_eval_context()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # Cache model instances per expression (different head_type per method).
    model_cache = {}

    METHOD_MODEL_ARG = {
        "conf": "model_conf",
        "hetero": "model_hetero",
        "ours_nig": "model_nig",
        "ours_niw": "model_niw",
        "ensemble": "model_ensemble",
        "mc_dropout": "model_mc",
    }

    def model_expr_for_method(method: str) -> str:
        key = METHOD_MODEL_ARG.get(method, None)
        expr = getattr(args, key, None) if key else None
        return expr or args.model

    def get_model(expr: str):
        if expr not in model_cache:
            model_cache[expr] = eval_from_str(expr, ctx).to(device)
        return model_cache[expr]

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.do_nll and ("conf" in methods) and (not args.do_conf_calib):
        print("[NLL] conf NLL disabled unless --do_conf_calib and a separate calibration split are provided.")

    curve_bins = args.curve_bins
    if args.rc_bins is not None:
        curve_bins = args.rc_bins
    if args.rc_u_log_max is not None:
        args.u_log_hi = float(args.rc_u_log_max)
        args.u_log_lo = -float(args.rc_u_log_max)
    if args.rc_e_log_max is not None:
        args.e_log_hi = float(args.rc_e_log_max)
        args.e_log_lo = -float(args.rc_e_log_max)
    rho_max_points = args.rho_max_points
    if args.rho_samples is not None:
        rho_max_points = args.rho_samples

    def replace_head_type(expr: str, head_type: str) -> str:
        if "head_type=" in expr:
            s2 = re.sub(r"head_type='[^']*'", f"head_type='{head_type}'", expr)
            s2 = re.sub(r'head_type=\"[^\"]*\"', f'head_type=\"{head_type}\"', s2)
            return s2
        if expr.endswith(")"):
            return expr[:-1] + f", head_type='{head_type}')"
        return expr

    def derive_model(expr: str, head_type: str) -> str:
        out = replace_head_type(expr, head_type)
        if out == expr and "head_type=" not in expr:
            print(f"[MODEL] WARNING: cannot override head_type; using base expr for {head_type}.")
        return out

    if args.model_conf is None:
        args.model_conf = derive_model(args.model, "catmlp+dpt")
    if args.model_hetero is None:
        args.model_hetero = derive_model(args.model, "catmlp+dpt+xyz_hetero_dpt")
    if args.model_nig is None:
        args.model_nig = derive_model(args.model, "catmlp+dpt+xyz_evi_dpt+residual_gated")
    if args.model_niw is None:
        args.model_niw = derive_model(args.model, "catmlp+dpt+xyz_niw_dpt+residual_gated")
    if args.model_ensemble is None:
        args.model_ensemble = args.model_conf
    if args.model_mc is None:
        args.model_mc = args.model_conf

    ckpt_map = {
        "conf": args.ckpt_conf,
        "hetero": args.ckpt_hetero,
        "ours_nig": args.ckpt_nig,
        "ours_niw": args.ckpt_niw,
        "mc_dropout": args.mc_ckpt or args.ckpt_conf or args.ckpt_nig or args.ckpt_niw,
    }

    ensemble_states = []
    if args.ensemble_ckpts.strip():
        ens_paths = [p for p in args.ensemble_ckpts.split(",") if p.strip()]
        ensemble_states = [load_state_dict(p.strip(), device="cpu") for p in ens_paths]

    benchmarks = parse_benchmarks(args.benchmarks)

    # -----------------------
    # conf post-hoc calibration
    # -----------------------
    conf_scale = None
    calib_bench_name = args.conf_calib_id_bench
    calib_split = None
    if args.do_nll and ("conf" in methods):
        if not args.do_conf_calib:
            print("[CALIB] conf calibration disabled; conf NLL will be skipped.")
        else:
            if not args.ckpt_conf:
                raise ValueError("do_nll requires --ckpt_conf for conf calibration")
            if not calib_bench_name:
                raise ValueError("do_conf_calib requires --conf_calib_id_bench")
            # find ID bench expr
            id_expr = None
            for name, expr in benchmarks:
                if name == calib_bench_name:
                    id_expr = expr
                    break
            if id_expr is None:
                raise ValueError(f"conf_calib_id_bench={calib_bench_name} not found in benchmarks list")
            calib_split = extract_split(id_expr)

            leak = False
            for name, expr in benchmarks:
                eval_split = extract_split(expr)
                if name == calib_bench_name and (calib_split == eval_split):
                    leak = True
                    break
            if leak and (not args.allow_calib_leak):
                print("[CALIB] WARNING: calib split == eval split; skipping calibration to avoid leakage.")
            else:
                id_dataset = eval_from_str(id_expr, ctx)

                conf_expr = model_expr_for_method("conf")
                conf_model = get_model(conf_expr)

                conf_scale = calibrate_conf_scale_isotropic(
                    model=conf_model,
                    dataset=id_dataset,
                    model_expr=conf_expr,
                    ckpt_conf=args.ckpt_conf,
                    device=device,
                    sim3_align=False,
                    conf_unc_mode=args.conf_unc_mode,
                    max_pairs=args.conf_calib_max_pairs,
                )
                with open(os.path.join(args.out_dir, "conf_calibration.json"), "w") as f:
                    json.dump({"conf_scale_isotropic": conf_scale}, f, indent=2)
                print(f"[conf calib] fitted conf->sigma^2 scale s = {conf_scale:.6g}")

    # Output tables:
    # Table1: AURC/AUSE/rho
    t1_header = ["Method", "Benchmark", "AURC", "AUSE", "SpearmanRho"]
    t1_rows = []

    # Table2: MAE/RMSE
    t2_header = ["Method", "Benchmark", "MAE3D", "RMSE3D"]
    t2_rows = []

    # Table3: NLL
    t3_header = ["Method", "Benchmark", "NLL"]
    t3_rows = []

    rng = np.random.RandomState(0)
    nll_var_floor = 1e-6
    warned_sigma2_floor = set()

    subset_info_all = []

    # -----------------------
    # evaluate per benchmark
    # -----------------------
    for bench_name, bench_expr in benchmarks:
        print(f"\n===== Benchmark: {bench_name} =====")
        dataset = eval_from_str(bench_expr, ctx)
        L = len(dataset) if hasattr(dataset, "__len__") else None
        if L is None and not args.max_pairs:
            raise ValueError("dataset length unknown; set --max_pairs for subset selection")
        max_pairs = args.max_pairs if (args.max_pairs is not None and args.max_pairs > 0) else None
        total_pairs = L if L is not None else max_pairs

        print(f"[SUBSET] scene_fraction={args.scene_fraction} max_pairs={args.max_pairs}")
        kept_indices = []
        kept_scenes = set()
        all_scenes = set()
        missing_scene = False
        idx_iter = range(L) if L is not None else range(max_pairs)
        for idx in idx_iter:
            scene_id = get_scene_id_from_dataset(dataset, idx)
            if scene_id is None:
                v1, v2 = dataset[idx]
                scene_id = get_scene_id(v1, v2, idx)
            if scene_id is None:
                missing_scene = True
                scene_key = f"idx:{idx}:seed:{args.subset_seed}"
            else:
                scene_key = f"scene:{scene_id}"
                all_scenes.add(scene_id)
            keep_scene = stable_hash01(scene_key) < args.scene_fraction
            if keep_scene:
                if max_pairs is None or len(kept_indices) < max_pairs:
                    kept_indices.append(idx)
                if scene_id is not None:
                    kept_scenes.add(scene_id)

        n_iter = len(kept_indices)
        scenes_total = len(all_scenes) if all_scenes else 0
        scenes_used = len(kept_scenes)
        print(f"[{bench_name}] pairs_total={total_pairs} pairs_used={n_iter}")
        print(f"[{bench_name}] scenes_total={scenes_total} scenes_used={scenes_used}")
        print(f"[SUBSET] kept_pairs={len(kept_indices)}/{total_pairs} kept_unique_scenes={len(kept_scenes)}")
        if len(kept_indices) <= 32:
            print(f"[SUBSET] pairs_idx={kept_indices}")
        else:
            head = kept_indices[:8]
            tail = kept_indices[-8:]
            print(f"[SUBSET] pairs_idx_count={len(kept_indices)} head={head} tail={tail}")
        if kept_scenes:
            kept_scenes_sorted = sorted(kept_scenes)
            if len(kept_scenes_sorted) <= 20:
                print(f"[SUBSET] scenes={kept_scenes_sorted}")
            else:
                head = kept_scenes_sorted[:8]
                tail = kept_scenes_sorted[-8:]
                print(f"[SUBSET] scenes_count={len(kept_scenes_sorted)} head={head} tail={tail}")
        if missing_scene:
            print("[SUBSET] scene_id missing for some pairs; used idx+seed fallback.")

        out_bench_dir = os.path.join(args.out_dir, bench_name)
        os.makedirs(out_bench_dir, exist_ok=True)
        with open(os.path.join(out_bench_dir, "subset_pairs_idx.txt"), "w") as f:
            for idx in kept_indices:
                f.write(f"{idx}\n")
        with open(os.path.join(out_bench_dir, "subset_scenes.json"), "w") as f:
            json.dump(sorted(kept_scenes), f, indent=2)
        subset_info_all.append({
            "benchmark": bench_name,
            "dataset_expr": bench_expr,
            "pairs_total": total_pairs,
            "pairs_used": n_iter,
            "scenes_total": scenes_total,
            "scenes_used": scenes_used,
            "scene_fraction": args.scene_fraction,
            "max_pairs": args.max_pairs,
            "pair_indices_used": kept_indices,
            "scene_ids_used": sorted(kept_scenes),
        })
        print(f"[SUBSET] saved subset to {out_bench_dir}")

        if args.do_nll:
            if args.sim3_align and (not args.nll_no_sim3):
                print("[NLL] WARNING: NLL will use Sim3 alignment; covariance rotation is not handled.")
            else:
                print(f"[NLL] sim3_align is {args.sim3_align} but NLL uses NO alignment by design.")

        bench_split = extract_split(bench_expr)
        rc_plot_curves = {}

        for method in methods:
            base, _ = base_method(method, args.ours_unc_variant)
            if base == "ensemble" and not ensemble_states:
                print("[skip] ensemble: no ensemble_ckpts")
                continue
            ckpt = ckpt_map.get(base, None)
            if base not in ("ensemble",) and (ckpt is None) and base not in ("mc_dropout",):
                print(f"[skip] {method}: missing checkpoint")
                continue

            # pick model instance for this method (head_type might differ)
            mexpr = model_expr_for_method(base)
            model = get_model(mexpr)

            # load weights
            if base == "ensemble":
                # will load per-member inside extractor
                pass
            elif base == "mc_dropout":
                if ckpt is None:
                    print("[skip] mc_dropout: no ckpt")
                    continue
                st = load_state_dict(ckpt, device="cpu")
                model.load_state_dict(st, strict=False)
            else:
                st = load_state_dict(ckpt, device="cpu")
                model.load_state_dict(st, strict=False)

            model.eval()

            mae_list, rmse_list = [], []
            nll_list = []
            rc_hist = HistRC(
                bins=curve_bins,
                u_log_lo=args.u_log_lo,
                u_log_hi=args.u_log_hi,
                e_log_lo=args.e_log_lo,
                e_log_hi=args.e_log_hi,
                device=device,
                rho_max_points=rho_max_points,
            )

            use_conf_calib = conf_scale is not None
            if use_conf_calib and calib_bench_name and (bench_name == calib_bench_name):
                if (calib_split is None) or (bench_split is None) or (calib_split == bench_split):
                    if not args.allow_calib_leak:
                        print("[CALIB] WARNING: calib split == eval split, skipping calibration for this benchmark.")
                        use_conf_calib = False

            if args.do_nll and base in ("mc_dropout", "ensemble") and args.nll_sigma2_floor <= 0.0:
                if base not in warned_sigma2_floor:
                    if is_rank0():
                        print("[WARN] computing dropout/ensemble NLL without sigma2_floor may be ill-calibrated; consider setting --nll_sigma2_floor or calibrating it on ID val.")
                    warned_sigma2_floor.add(base)

            with torch.no_grad():
                for i, idx in enumerate(kept_indices):
                    v1, v2 = dataset[idx]
                    b1 = batchify_view(v1, device)
                    b2 = batchify_view(v2, device)
                    H, W = b1["img"].shape[-2:]

                    gt = compute_gt_xyz_cam(b1)
                    gt = resize_to_hw(gt, (H, W), "bilinear")
                    mask = get_valid_mask(b1)
                    if mask is not None:
                        mask = resize_mask_to_hw(mask, (H, W))

                    outs = extract_method_outputs(
                        model=model,
                        b1=b1, b2=b2,
                        method=method,
                        ours_unc_variant=args.ours_unc_variant,
                        mc_samples=args.mc_samples,
                        ensemble_states=ensemble_states if base == "ensemble" else None,
                        conf_unc_mode=args.conf_unc_mode,
                        xyz_reduce=args.xyz_reduce,
                        dump_keys=(args.debug_keys and i == 0),
                    )

                    pred = outs.pred_xyz
                    unc = outs.unc_map  # [B,H,W]
                    if pred.shape[-2:] != (H, W):
                        pred = resize_to_hw(pred, (H, W), "bilinear")
                    if unc.shape[-2:] != (H, W):
                        unc = resize_bhw(unc, (H, W))

                    if args.debug_shapes and i == 0:
                        print(f"[SHAPES][{bench_name}][{method}] img={tuple(b1['img'].shape)} "
                              f"pred_xyz={tuple(pred.shape)} gt_xyz={tuple(gt.shape)} unc={tuple(unc.shape)}")

                    sim3 = None
                    if args.sim3_align:
                        s, R, t = estimate_sim3(pred, gt, mask)
                        sim3 = (s, R, t)
                    pred_for_err, unc_for_err = maybe_align_for_error(pred, unc, sim3, args.xyz_reduce)

                    err_map = per_image_error(pred_for_err, gt)  # [B,H,W]
                    mae, rmse = mae_rmse(err_map, mask)
                    mae_list.append(mae); rmse_list.append(rmse)

                    if mask is not None:
                        err_flat = err_map[mask]
                        unc_flat = unc_for_err[mask]
                    else:
                        err_flat = err_map.reshape(-1)
                        unc_flat = unc_for_err.reshape(-1)
                    if err_flat.numel() >= 10:
                        rc_hist.update(unc_flat, err_flat, rng)

                    # NLL
                    if args.do_nll:
                        if not outs.extra.get("supports_nll", True):
                            continue
                        use_sim3_nll = bool(args.sim3_align and (not args.nll_no_sim3))
                        pred_for_nll = pred
                        if use_sim3_nll and sim3 is not None:
                            pred_for_nll = apply_sim3(pred, sim3[0], sim3[1], sim3[2])
                        if base == "conf":
                            if (conf_scale is None) or (not use_conf_calib):
                                continue
                            conf = outs.extra["conf"]  # [B,H,W]
                            if args.conf_unc_mode == "inv":
                                u0 = (1.0 / conf.clamp_min(1e-8)).float()
                            else:
                                u0 = (-torch.log(conf.clamp_min(1e-8))).float()
                            sigma2 = conf_scale * u0  # [B,H,W]
                            sigma2 = sigma2.unsqueeze(1).repeat(1, 3, 1, 1)  # [B,3,H,W]
                            if use_sim3_nll and sim3 is not None:
                                sigma2 = sigma2 * (sim3[0] ** 2)
                            diff = (pred_for_nll - gt)
                            nll = gaussian_nll_diag(diff, sigma2, mask)
                            nll_list.append(nll)

                        elif base == "hetero":
                            var = outs.extra.get("var_diag", None)
                            if var is None:
                                continue
                            var = resize_to_hw(var, (H, W), "bilinear")
                            var = torch.clamp(var, min=nll_var_floor)
                            if use_sim3_nll and sim3 is not None:
                                var = var * (sim3[0] ** 2)
                            diff = (pred_for_nll - gt)
                            nll = gaussian_nll_diag(diff, var, mask)
                            nll_list.append(nll)

                        elif base in ("mc_dropout", "ensemble"):
                            var = outs.extra.get("var_diag", None)
                            if var is None:
                                raise ValueError(f"NLL requested for {base} but var_diag is missing; ensure extract_method_outputs provides var_diag.")
                            sigma2 = resize_to_hw(var, (H, W), "bilinear")
                            if args.nll_sigma2_floor > 0.0:
                                sigma2 = sigma2 + args.nll_sigma2_floor
                            sigma2 = sigma2.clamp_min(1e-12)
                            if use_sim3_nll and sim3 is not None:
                                sigma2 = sigma2 * (sim3[0] ** 2)
                            diff = (pred_for_nll - gt)
                            nll = gaussian_nll_diag(diff, sigma2, mask)
                            nll_list.append(nll)

                        elif base == "ours_nig":
                            gamma = outs.extra.get("gamma", None)
                            nu_p  = outs.extra.get("nu", None)
                            alpha = outs.extra.get("alpha", None)
                            beta  = outs.extra.get("beta", None)
                            if all(x is not None for x in [gamma, nu_p, alpha, beta]):
                                if use_sim3_nll and sim3 is not None:
                                    gamma = apply_sim3(gamma, sim3[0], sim3[1], sim3[2])
                                nll = nig_nll_3d(gt, gamma, nu_p, alpha, beta, mask)
                                nll_list.append(nll)
                            else:
                                var = outs.extra.get("var_diag", None)
                                if var is None:
                                    continue
                                var = torch.clamp(var, min=nll_var_floor)
                                if use_sim3_nll and sim3 is not None:
                                    var = var * (sim3[0] ** 2)
                                diff = (pred_for_nll - gt)
                                nll = gaussian_nll_diag(diff, var, mask)
                                nll_list.append(nll)

                        elif base == "ours_niw":
                            m = outs.extra.get("m", None)
                            kappa = outs.extra.get("kappa", None)
                            nu_p = outs.extra.get("nu", None)
                            Lp = outs.extra.get("Lp", None)
                            if all(x is not None for x in [m, kappa, nu_p, Lp]):
                                if use_sim3_nll and sim3 is not None:
                                    m = apply_sim3(m, sim3[0], sim3[1], sim3[2])
                                    Lp = Lp * abs(sim3[0])
                                nll = niw_nll_3d(gt, m, kappa, nu_p, Lp, mask)
                                nll_list.append(nll)
                            else:
                                var = outs.extra.get("var_diag", None)
                                if var is None:
                                    continue
                                var = torch.clamp(var, min=nll_var_floor)
                                if use_sim3_nll and sim3 is not None:
                                    var = var * (sim3[0] ** 2)
                                diff = (pred_for_nll - gt)
                                nll = gaussian_nll_diag(diff, var, mask)
                                nll_list.append(nll)

            # aggregate per method x benchmark
            def safe_mean(x):
                x = [v for v in x if np.isfinite(v)]
                return float(np.mean(x)) if len(x) else float("nan")

            aurc_m, ause_m, (grid, risk_m, risk_o) = rc_hist.finalize_curves()
            if rc_hist.rho_u:
                u_np = np.asarray(rc_hist.rho_u, dtype=np.float64)
                e_np = np.asarray(rc_hist.rho_e, dtype=np.float64)
            else:
                u_np = np.array([])
                e_np = np.array([])
            rho_m = spearmanr_from_samples(u_np, e_np)

            mae_m  = safe_mean(mae_list)
            rmse_m = safe_mean(rmse_list)
            t1_rows.append([method, bench_name, f"{aurc_m:.6f}", f"{ause_m:.6f}", f"{rho_m:.6f}"])
            t2_rows.append([method, bench_name, f"{mae_m:.6f}", f"{rmse_m:.6f}"])
            if args.do_nll:
                nll_m = safe_mean(nll_list)
                t3_rows.append([method, bench_name, f"{nll_m:.6f}"])

            rc_npz = os.path.join(args.out_dir, f"curves_{bench_name}_{method}.npz")
            save_rc_curve(rc_npz, None, grid, risk_m, risk_o, aurc_m, ause_m)
            rc_plot_curves[method] = (grid, risk_m)
            print(f"[RC][{bench_name}][{method}] total_valid_pixels={int(rc_hist.total)} "
                  f"aurc={aurc_m:.4g} ause={ause_m:.4g}")

            print(f"[{bench_name}][{method}] AURC={aurc_m:.4g} AUSE={ause_m:.4g} rho={rho_m:.4g}  MAE={mae_m:.4g} RMSE={rmse_m:.4g}"
                  + (f"  NLL={safe_mean(nll_list):.4g}" if args.do_nll else ""))

        rc_png = os.path.join(args.out_dir, f"curves_{bench_name}.png")
        save_rc_multi(rc_png, rc_plot_curves, f"{bench_name} risk-coverage")

    # save
    with open(os.path.join(args.out_dir, "subset_info.json"), "w") as f:
        json.dump(subset_info_all, f, indent=2)
    save_csv(os.path.join(args.out_dir, "table1_uq.csv"), t1_header, t1_rows)
    save_csv(os.path.join(args.out_dir, "table2_recon.csv"), t2_header, t2_rows)
    if args.do_nll:
        save_csv(os.path.join(args.out_dir, "table3_nll.csv"), t3_header, t3_rows)

    print("\nSaved:")
    print(" - table1_uq.csv")
    print(" - table2_recon.csv")
    print(" - subset_info.json")
    if args.do_nll:
        print(" - table3_nll.csv")
        if conf_scale is not None:
            print(" - conf_calibration.json")


if __name__ == "__main__":
    main()
