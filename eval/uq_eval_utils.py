#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for UQ evaluation in MASt3R/DUSt3R-style repos.

Supports:
- Sim3 alignment (Umeyama)
- Resize-to-RGB to avoid blocky error maps
- Per-image AURC / AUSE / Spearman rho
- NLL:
    * conf-only post-hoc isotropic Gaussian calibration on ID val
    * diagonal Gaussian NLL (hetero, dropout/ensemble moment match)
    * NIG Student-t (independent per xyz channel)
    * NIW multivariate Student-t (Lp=[l00,l10,l11,l20,l21,l22])
"""
import os
import sys
import math
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Repo eval-from-string helpers
# ----------------------------
def add_repo_root():
    """Make `mast3r.*` and the vendored `dust3r.*` importable without a preset PYTHONPATH."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # `<repo>/dust3r` must precede `<repo>`: the vendored tree is `<repo>/dust3r/dust3r`,
    # and `<repo>/dust3r` is itself an importable package that would shadow it otherwise.
    for path in (repo_root, os.path.join(repo_root, "dust3r")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def build_eval_context():
    ctx = {"torch": torch, "np": np}
    try:
        import dust3r.datasets as datasets  # noqa
        ctx.update(datasets.__dict__)
    except Exception:
        pass
    try:
        import dust3r.model as dmodel  # noqa
        ctx.update(dmodel.__dict__)
    except Exception:
        pass
    try:
        import mast3r.model as mmodel  # noqa
        ctx.update(mmodel.__dict__)
    except Exception:
        pass
    return ctx


def eval_from_str(expr: str, ctx: Dict):
    return eval(expr, ctx)


def load_state_dict(ckpt_path: str, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


# ----------------------------
# Tensor shape helpers
# ----------------------------
def batchify_view(view: Dict, device: torch.device):
    out = {}
    for k, v in view.items():
        if isinstance(v, np.ndarray) or torch.is_tensor(v):
            t = torch.as_tensor(v)
            if t.dtype == torch.float64:
                t = t.float()
            # img: allow HWC or CHW -> normalize to CHW
            if k == "img" and t.ndim == 3 and t.shape[-1] == 3 and t.shape[0] != 3:
                t = t.permute(2, 0, 1).contiguous()
            if t.ndim <= 3:
                t = t.unsqueeze(0)
            out[k] = t.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def to_bchw(x: torch.Tensor) -> torch.Tensor:
    # Accept [B,3,H,W] or [B,H,W,3]
    if x.ndim == 4 and x.shape[1] == 3:
        return x
    if x.ndim == 4 and x.shape[-1] == 3:
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Unexpected xyz-like shape: {tuple(x.shape)}")


def to_b1hw(x: torch.Tensor) -> torch.Tensor:
    # [B,1,H,W] or [B,H,W] or [B,H,W,1]
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    if x.ndim == 4 and x.shape[-1] == 1:
        return x.permute(0, 3, 1, 2).contiguous()
    if x.ndim == 3:
        return x.unsqueeze(1)
    raise ValueError(f"Unexpected scalar-map shape: {tuple(x.shape)}")


def get_valid_mask(v1: Dict) -> Optional[torch.Tensor]:
    mask = v1.get("valid_depth", None)
    if mask is None:
        return None
    if torch.is_tensor(mask) is False:
        mask = torch.as_tensor(mask)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    # return [B,H,W] bool
    if mask.ndim == 3:
        return mask > 0
    if mask.ndim == 4:
        return mask[:, 0] > 0
    raise ValueError(f"Unexpected valid mask shape: {tuple(mask.shape)}")


def compute_gt_xyz_cam(v1: Dict) -> torch.Tensor:
    pts = to_bchw(v1["pts3d"])
    cam_pose = v1.get("camera_pose", None)
    if cam_pose is None:
        return pts
    if torch.is_tensor(cam_pose) is False:
        cam_pose = torch.as_tensor(cam_pose, device=pts.device, dtype=pts.dtype)
    if cam_pose.ndim == 2:
        cam_pose = cam_pose.unsqueeze(0)
    # pts3d seems in world; convert world->cam
    world_to_cam = torch.linalg.inv(cam_pose)
    r = world_to_cam[:, :3, :3]
    t = world_to_cam[:, :3, 3]
    pts_cam = torch.einsum("bij,bjhw->bihw", r, pts) + t[:, :, None, None]
    return pts_cam


def resize_to_hw(x: torch.Tensor, hw: Tuple[int, int], mode="bilinear") -> torch.Tensor:
    if x.shape[-2:] == hw:
        return x
    if x.ndim == 4:
        return F.interpolate(x, size=hw, mode=mode, align_corners=False if mode == "bilinear" else None)
    raise ValueError(f"resize_to_hw expects 4D tensor, got {tuple(x.shape)}")


def resize_mask_to_hw(mask_bhw: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    # mask: [B,H,W] -> [B,1,H,W] nearest -> [B,H,W]
    if mask_bhw.shape[-2:] == hw:
        return mask_bhw
    m = mask_bhw.unsqueeze(1).float()
    m = F.interpolate(m, size=hw, mode="nearest")
    return (m[:, 0] > 0.5)


# ----------------------------
# Sim3 alignment (Umeyama)
# ----------------------------
def umeyama_sim3(src: torch.Tensor, dst: torch.Tensor):
    """
    src,dst: [N,3]
    returns scale(float), R[3,3], t[3]
    """
    n = src.shape[0]
    if n < 3:
        return 1.0, torch.eye(3, device=src.device), torch.zeros(3, device=src.device)
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.t() @ src_c) / n
    u, s, vt = torch.linalg.svd(cov)
    r = u @ vt
    if torch.det(r) < 0:
        vt[-1, :] *= -1
        s[-1] *= -1
        r = u @ vt
    var_src = (src_c ** 2).sum() / n
    scale = 1.0 if var_src <= 0 else (s.sum() / var_src)
    t = dst_mean - scale * (r @ src_mean)
    return float(scale), r, t


def weighted_umeyama_sim3(src: torch.Tensor, dst: torch.Tensor, weights: Optional[torch.Tensor]):
    """
    Weighted Umeyama alignment.

    src,dst: [N,3]
    weights: [N] or None
    returns scale(float), R[3,3], t[3]
    """
    n = src.shape[0]
    if n < 3:
        return 1.0, torch.eye(3, device=src.device), torch.zeros(3, device=src.device)
    if weights is None:
        return umeyama_sim3(src, dst)

    w = torch.as_tensor(weights, device=src.device, dtype=src.dtype).reshape(-1)
    valid = torch.isfinite(w) & (w > 0)
    if valid.sum() < 3:
        return 1.0, torch.eye(3, device=src.device), torch.zeros(3, device=src.device)
    src = src[valid]
    dst = dst[valid]
    w = w[valid]

    w_sum = w.sum()
    if not torch.isfinite(w_sum) or float(w_sum) <= 0.0:
        return 1.0, torch.eye(3, device=src.device), torch.zeros(3, device=src.device)
    w = w / w_sum

    src_mean = (src * w[:, None]).sum(dim=0)
    dst_mean = (dst * w[:, None]).sum(dim=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c * w[:, None]).t() @ src_c
    u, s, vt = torch.linalg.svd(cov)
    r = u @ vt
    if torch.det(r) < 0:
        vt[-1, :] *= -1
        s[-1] *= -1
        r = u @ vt

    var_src = ((src_c.square().sum(dim=1)) * w).sum()
    if not torch.isfinite(var_src) or float(var_src) <= 0.0:
        return 1.0, torch.eye(3, device=src.device), torch.zeros(3, device=src.device)

    scale = s.sum() / var_src
    t = dst_mean - scale * (r @ src_mean)
    return float(scale), r, t


def estimate_sim3(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor, mask_bhw: Optional[torch.Tensor], max_points=50000):
    # pred/gt: [B,3,H,W]
    pred = pred_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    gt = gt_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    if mask_bhw is not None:
        m = mask_bhw.reshape(-1)
        pred = pred[m]
        gt = gt[m]
    if pred.numel() == 0:
        return 1.0, torch.eye(3, device=pred_xyz.device), torch.zeros(3, device=pred_xyz.device)
    n = pred.shape[0]
    if n > max_points:
        idx = torch.randperm(n, device=pred.device)[:max_points]
        pred = pred[idx]
        gt = gt[idx]
    return umeyama_sim3(pred, gt)


def estimate_weighted_sim3(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    mask_bhw: Optional[torch.Tensor],
    weights_bhw: Optional[torch.Tensor] = None,
    max_points: int = 50000,
    retain_fraction: float = 1.0,
):
    # pred/gt: [B,3,H,W], weights: [B,H,W] or [B,1,H,W]
    pred = pred_xyz.permute(0, 2, 3, 1).reshape(-1, 3)
    gt = gt_xyz.permute(0, 2, 3, 1).reshape(-1, 3)

    valid = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
    if mask_bhw is not None:
        valid = valid & mask_bhw.reshape(-1).bool()

    w = None
    if weights_bhw is not None:
        if weights_bhw.ndim == 4 and weights_bhw.shape[1] == 1:
            weights_bhw = weights_bhw[:, 0]
        w = torch.as_tensor(weights_bhw, device=pred.device, dtype=pred.dtype).reshape(-1)
        valid = valid & torch.isfinite(w) & (w > 0)

    pred = pred[valid]
    gt = gt[valid]
    if w is not None:
        w = w[valid]

    if pred.numel() == 0 or pred.shape[0] < 3:
        return 1.0, torch.eye(3, device=pred_xyz.device), torch.zeros(3, device=pred_xyz.device)

    if retain_fraction < 1.0:
        keep = max(3, int(round(pred.shape[0] * retain_fraction)))
        if w is not None:
            keep_idx = torch.topk(w, k=min(keep, w.shape[0]), largest=True, sorted=False).indices
        else:
            keep_idx = torch.arange(min(keep, pred.shape[0]), device=pred.device)
        pred = pred[keep_idx]
        gt = gt[keep_idx]
        if w is not None:
            w = w[keep_idx]

    if pred.shape[0] > max_points:
        if w is not None:
            keep_idx = torch.topk(w, k=max_points, largest=True, sorted=False).indices
        else:
            keep_idx = torch.randperm(pred.shape[0], device=pred.device)[:max_points]
        pred = pred[keep_idx]
        gt = gt[keep_idx]
        if w is not None:
            w = w[keep_idx]

    return weighted_umeyama_sim3(pred, gt, w)


def apply_sim3(pred_xyz: torch.Tensor, scale: float, r: torch.Tensor, t: torch.Tensor):
    # pred_xyz: [B,3,H,W]
    pred = pred_xyz.permute(0, 2, 3, 1)  # [B,H,W,3]
    aligned = scale * torch.einsum("ij,bhwj->bhwi", r, pred) + t
    return aligned.permute(0, 3, 1, 2).contiguous()


# ----------------------------
# Curves & metrics
# ----------------------------
def per_image_error(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    # [B,3,H,W] -> [B,H,W] L2
    return torch.linalg.norm(pred_xyz - gt_xyz, dim=1)


def mae_rmse(err_map: torch.Tensor, mask_bhw: Optional[torch.Tensor]) -> Tuple[float, float]:
    # err_map: [B,H,W]
    if mask_bhw is not None:
        e = err_map[mask_bhw]
    else:
        e = err_map.reshape(-1)
    e = e.float()
    mae = float(e.mean().detach().cpu())
    rmse = float(torch.sqrt((e ** 2).mean()).detach().cpu())
    return mae, rmse


def risk_coverage_curve(err: np.ndarray, unc: np.ndarray):
    """
    Keep low-unc first:
      coverage = k/N
      risk = mean(err of kept pixels)
    """
    order = np.argsort(unc)  # ascending
    e = err[order]
    csum = np.cumsum(e)
    k = np.arange(1, e.size + 1)
    risk = csum / k
    coverage = k / e.size
    return coverage, risk


def sparsification_curve(err: np.ndarray, score: np.ndarray):
    """
    Remove high-score first:
      removed = k/N
      remaining risk = mean(err of remaining pixels)
    """
    order = np.argsort(-score)  # descending (remove highest)
    e = err[order]
    csum = np.cumsum(e)
    total = csum[-1]
    k = np.arange(0, e.size)  # removed count
    # remaining after removing k:
    rem_sum = total - np.concatenate([[0.0], csum[:-1]])
    rem_n = (e.size - k).astype(np.float64)
    rem_risk = rem_sum / np.maximum(rem_n, 1.0)
    removed_frac = k / e.size
    return removed_frac, rem_risk


def aurc_from_curve(coverage: np.ndarray, risk: np.ndarray) -> float:
    # integrate risk over coverage in [0,1]
    return float(np.trapz(risk, coverage))


def ause_from_curves(removed: np.ndarray, risk_method: np.ndarray, risk_oracle: np.ndarray) -> float:
    # integrate (method - oracle) over removed fraction
    return float(np.trapz(risk_method - risk_oracle, removed))


def spearmanr_sampled(err: np.ndarray, unc: np.ndarray, max_points=200000, rng: Optional[np.random.RandomState] = None) -> float:
    """
    Spearman rank correlation (sampled for speed).
    """
    n = err.size
    if n == 0:
        return float("nan")
    if rng is None:
        rng = np.random.RandomState(0)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        err = err[idx]
        unc = unc[idx]
    # rank data
    er = err.argsort().argsort().astype(np.float64)
    ur = unc.argsort().argsort().astype(np.float64)
    er -= er.mean()
    ur -= ur.mean()
    denom = (np.sqrt((er ** 2).mean()) * np.sqrt((ur ** 2).mean()) + 1e-12)
    return float((er * ur).mean() / denom)


# ----------------------------
# NLLs
# ----------------------------
def gaussian_nll_diag(diff_xyz: torch.Tensor, var_xyz: torch.Tensor, mask_bhw: Optional[torch.Tensor], eps=1e-8) -> float:
    """
    diff_xyz,var_xyz: [B,3,H,W]
    NLL per pixel (sum over xyz), then mean over valid pixels.
    """
    var = torch.clamp(var_xyz.float(), min=eps)
    diff2 = (diff_xyz.float() ** 2)
    nll = 0.5 * (torch.log(2 * math.pi * var) + diff2 / var)  # [B,3,H,W]
    nll = nll.sum(dim=1)  # [B,H,W]
    if mask_bhw is not None:
        denom = mask_bhw.sum().clamp_min(1)
        return float((nll * mask_bhw).sum().detach().cpu() / denom.detach().cpu())
    return float(nll.mean().detach().cpu())


def calibrate_conf_scale_isotropic(
    model, dataset, model_expr: str,
    ckpt_conf: str,
    device: torch.device,
    sim3_align: bool = True,
    conf_unc_mode: str = "inv",
    max_pairs: Optional[int] = None,
    eps=1e-8,
) -> float:
    """
    Fit scalar s such that per-coordinate variance sigma^2 = s * u0,
    where u0 = 1/conf (or -log(conf) if you really want, but inv is typical).
    Minimizes Gaussian NLL on ID validation.

    Closed-form:
      s = (Σ ||diff||^2 / u0) / (3N)
    """
    state = load_state_dict(ckpt_conf, device="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    num = 0.0
    den = 0.0  # counts *3 implicitly
    pairs = len(dataset) if hasattr(dataset, "__len__") else None
    n_iter = pairs if max_pairs is None else min(pairs, max_pairs)

    with torch.no_grad():
        for i in range(n_iter):
            v1, v2 = dataset[i]
            b1 = batchify_view(v1, device)
            b2 = batchify_view(v2, device)

            H, W = b1["img"].shape[-2:]
            gt = compute_gt_xyz_cam(b1)
            gt = resize_to_hw(gt, (H, W), mode="bilinear")
            mask = get_valid_mask(b1)
            if mask is not None:
                mask = resize_mask_to_hw(mask, (H, W))

            pred1, _ = model(b1, b2)
            pred_xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
            pred_xyz = resize_to_hw(pred_xyz, (H, W), mode="bilinear")

            if sim3_align:
                s, R, t = estimate_sim3(pred_xyz, gt, mask)
                pred_xyz = apply_sim3(pred_xyz, s, R, t)
            diff = pred_xyz - gt

            conf = get_first(pred1, ["conf", "conf1", "desc_conf"])
            if conf is None:
                continue
            conf = to_b1hw(conf)[:, 0]
            conf = resize_to_hw(conf.unsqueeze(1), (H, W), mode="bilinear")[:, 0]
            conf = conf.clamp_min(eps)

            if conf_unc_mode == "inv":
                u0 = (1.0 / conf).float()
            else:
                u0 = (-torch.log(conf)).float()

            d2 = (diff.float() ** 2).sum(dim=1)  # [B,H,W]
            if mask is not None:
                d2 = d2[mask]
                u0 = u0[mask]
            num += float((d2 / (u0 + eps)).sum().detach().cpu())
            den += float(d2.numel())

    if den <= 0:
        return 1.0
    s_fit = (num / (3.0 * den))
    return float(max(s_fit, eps))


def nig_nll_3d(y_xyz, gamma, nu, alpha, beta, mask_bhw: Optional[torch.Tensor], eps=1e-8) -> float:
    """
    Independent per xyz channel Student-t NLL induced by NIG params.
    y_xyz,gamma,nu,alpha,beta: [B,3,H,W]
    """
    y = y_xyz.reshape(-1, 1, *y_xyz.shape[-2:])
    # flatten channel into batch for convenience
    B, C, H, W = y_xyz.shape
    y_flat = y_xyz.reshape(B * C, 1, H, W)
    g_flat = gamma.reshape(B * C, 1, H, W)
    nu_flat = torch.clamp(nu.reshape(B * C, 1, H, W), min=eps)
    a_flat = torch.clamp(alpha.reshape(B * C, 1, H, W), min=1.0 + eps)
    b_flat = torch.clamp(beta.reshape(B * C, 1, H, W), min=eps)

    two_beta_nu = 2.0 * b_flat * (1.0 + nu_flat)
    nll = (
        0.5 * torch.log(torch.tensor(math.pi, device=y_xyz.device) / (nu_flat + eps))
        - a_flat * torch.log(two_beta_nu + eps)
        + (a_flat + 0.5) * torch.log(nu_flat * (y_flat - g_flat) ** 2 + two_beta_nu + eps)
        + torch.lgamma(a_flat)
        - torch.lgamma(a_flat + 0.5)
    )  # [B*3,1,H,W]
    nll = nll[:, 0]  # [B*3,H,W]

    if mask_bhw is not None:
        # broadcast mask over channels
        if mask_bhw.ndim == 3:
            m = mask_bhw.unsqueeze(1).expand(B, C, H, W).reshape(B * C, H, W)
        else:
            raise ValueError("mask_bhw must be [B,H,W]")
        denom = m.sum().clamp_min(1)
        return float((nll * m).sum().detach().cpu() / denom.detach().cpu())
    return float(nll.mean().detach().cpu())


def niw_nll_3d(y_xyz, m_xyz, kappa, nu, Lp, mask_bhw: Optional[torch.Tensor], eps=1e-8) -> float:
    """
    Multivariate Student-t NLL induced by NIW.
    y_xyz,m_xyz: [B,3,H,W]
    kappa,nu: [B,1,H,W]
    Lp: [B,6,H,W] with order [l00,l10,l11,l20,l21,l22]
    """
    B, C, H, W = y_xyz.shape
    d = 3
    y = y_xyz.permute(0, 2, 3, 1).reshape(-1, d)
    m = m_xyz.permute(0, 2, 3, 1).reshape(-1, d)
    diff = (y - m).unsqueeze(-1)  # [N,3,1]

    Lp_flat = Lp.permute(0, 2, 3, 1).reshape(-1, 6)
    l00 = Lp_flat[:, 0].clamp_min(eps)
    l10 = Lp_flat[:, 1]
    l11 = Lp_flat[:, 2].clamp_min(eps)
    l20 = Lp_flat[:, 3]
    l21 = Lp_flat[:, 4]
    l22 = Lp_flat[:, 5].clamp_min(eps)

    Lmat = torch.zeros((Lp_flat.shape[0], d, d), device=Lp.device, dtype=Lp.dtype)
    Lmat[:, 0, 0] = l00
    Lmat[:, 1, 0] = l10
    Lmat[:, 1, 1] = l11
    Lmat[:, 2, 0] = l20
    Lmat[:, 2, 1] = l21
    Lmat[:, 2, 2] = l22

    sol = torch.linalg.solve_triangular(Lmat, diff, upper=False)
    delta_psi = (sol.squeeze(-1) ** 2).sum(dim=-1)  # [N]

    kappa_flat = kappa.permute(0, 2, 3, 1).reshape(-1).clamp_min(eps)
    nu_flat = nu.permute(0, 2, 3, 1).reshape(-1).clamp_min(d + 1 + eps)

    nu_pred = (nu_flat - (d - 1)).clamp_min(eps)  # df-like
    scale = (kappa_flat + 1.0) / (kappa_flat * nu_pred)
    delta = delta_psi / scale

    logdet_psi = 2.0 * (torch.log(l00) + torch.log(l11) + torch.log(l22))
    logdet_sigma = logdet_psi + d * torch.log(scale)

    log_norm = (
        torch.lgamma(0.5 * (nu_pred + d))
        - torch.lgamma(0.5 * nu_pred)
        - 0.5 * (d * torch.log(nu_pred * math.pi) + logdet_sigma)
    )
    log_prob = log_norm - 0.5 * (nu_pred + d) * torch.log1p(delta / nu_pred)
    nll = -log_prob  # [N]
    nll_map = nll.view(B, H, W)

    if mask_bhw is not None:
        denom = mask_bhw.sum().clamp_min(1)
        return float((nll_map * mask_bhw).sum().detach().cpu() / denom.detach().cpu())
    return float(nll_map.mean().detach().cpu())


# ----------------------------
# Method I/O keys (robust getters)
# ----------------------------
def get_first(pred: Dict, keys: List[str]):
    for k in keys:
        if k in pred and pred[k] is not None:
            return pred[k]
    return None


_OURS_VARIANTS = ("total", "alea", "epi")
_WARNED_OURS_VARIANTS = set()


def base_method(method: str, default_variant: Optional[str] = None) -> Tuple[str, Optional[str]]:
    base = method
    variant = None
    for prefix in ("ours_nig", "ours_niw"):
        if method.startswith(prefix + "_"):
            base = prefix
            variant = method[len(prefix) + 1:]
            break
    if base in ("ours_nig", "ours_niw"):
        if variant is None:
            variant = default_variant or "total"
        if variant not in _OURS_VARIANTS:
            key = f"{base}:{variant}"
            if key not in _WARNED_OURS_VARIANTS:
                print(f"[WARN] Unknown ours_unc_variant '{variant}' for {base}; using 'total'.")
                _WARNED_OURS_VARIANTS.add(key)
            variant = default_variant if default_variant in _OURS_VARIANTS else "total"
    return base, variant


@dataclass
class MethodOutputs:
    pred_xyz: torch.Tensor          # [B,3,H,W] (already resized to RGB)
    unc_map: torch.Tensor           # [B,H,W]   (already resized to RGB)
    extra: Dict[str, torch.Tensor]  # optional distribution params for NLL


def _iter_mc_dropout_modules(model):
    dropout_types = (
        nn.Dropout, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout,
    )
    for m in model.modules():
        has_hook = hasattr(m, "set_mc_dropout") or hasattr(m, "mc_dropout_enabled")
        is_std_dropout = isinstance(m, dropout_types)
        if has_hook or is_std_dropout:
            yield m, is_std_dropout, has_hook


def _capture_mc_dropout_state(model):
    states = []
    for m, is_std_dropout, _ in _iter_mc_dropout_modules(model):
        st = {
            "module": m,
            "is_std_dropout": is_std_dropout,
            "has_set_mc_dropout": hasattr(m, "set_mc_dropout"),
        }
        if is_std_dropout:
            st["training"] = bool(m.training)
            if hasattr(m, "p"):
                st["p"] = float(m.p)
        if hasattr(m, "mc_dropout_enabled"):
            st["mc_dropout_enabled"] = bool(m.mc_dropout_enabled)
        if hasattr(m, "mc_dropout_p"):
            st["mc_dropout_p"] = float(m.mc_dropout_p)
        states.append(st)
    return states


def _restore_mc_dropout_state(states):
    for st in states:
        m = st["module"]
        if st.get("has_set_mc_dropout", False):
            enabled = bool(st.get("mc_dropout_enabled", False))
            m.set_mc_dropout(enabled)
        if "mc_dropout_enabled" in st:
            m.mc_dropout_enabled = bool(st["mc_dropout_enabled"])
        if "mc_dropout_p" in st and hasattr(m, "mc_dropout_p"):
            m.mc_dropout_p = float(st["mc_dropout_p"])
        if st.get("is_std_dropout", False):
            if "p" in st and hasattr(m, "p"):
                m.p = float(st["p"])
            if st.get("training", False):
                m.train(True)
            else:
                m.eval()


def _set_mc_dropout_flag(model, enabled: bool, p: Optional[float] = None):
    for m, is_std_dropout, _ in _iter_mc_dropout_modules(model):
        if hasattr(m, "set_mc_dropout"):
            if p is None:
                m.set_mc_dropout(enabled)
            else:
                m.set_mc_dropout(enabled, p)
        if hasattr(m, "mc_dropout_enabled"):
            m.mc_dropout_enabled = bool(enabled)
            if p is not None and hasattr(m, "mc_dropout_p"):
                m.mc_dropout_p = float(p)
        if is_std_dropout:
            if p is not None and hasattr(m, "p"):
                m.p = float(p)
            if enabled:
                m.train(True)
            else:
                m.eval()


def _mc_dropout_predict_xyz(model, v1: Dict, v2: Dict, mc_samples: int):
    mc_samples = int(mc_samples)
    if mc_samples <= 0:
        raise ValueError("mc_dropout requires mc_samples > 0")
    H, W = v1["img"].shape[-2:]

    prev_training = model.training
    dropout_state = _capture_mc_dropout_state(model)
    model.eval()

    det_xyz = None
    mean = None
    m2 = None
    count = 0
    do_sanity = not getattr(_mc_dropout_predict_xyz, "_sanity_done", False)
    try:
        with torch.no_grad():
            _set_mc_dropout_flag(model, False)
            pred_det, _ = model(v1, v2)
            det_xyz = pred_det.get("xyz_mu", pred_det.get("pts3d"))
            if det_xyz is None:
                raise KeyError("mc_dropout: cannot find xyz_mu/pts3d in pred")
            det_xyz = to_bchw(det_xyz).float()
            det_xyz = resize_to_hw(det_xyz, (H, W), mode="bilinear")

            _set_mc_dropout_flag(model, True)
            for _ in range(mc_samples):
                pred1, _ = model(v1, v2)
                xyz = pred1.get("xyz_mu", None)
                if xyz is None:
                    xyz = pred1.get("pts3d", None)
                if xyz is None:
                    raise KeyError("mc_dropout: cannot find xyz_mu/pts3d in pred")
                xyz = to_bchw(xyz).float()
                xyz = resize_to_hw(xyz, (H, W), mode="bilinear")

                count += 1
                if mean is None:
                    mean = xyz.clone()
                    m2 = torch.zeros_like(xyz)
                else:
                    delta = xyz - mean
                    mean = mean + delta / float(count)
                    delta2 = xyz - mean
                    m2 = m2 + delta * delta2
    finally:
        model.train(prev_training)
        _restore_mc_dropout_state(dropout_state)

    if mean is None:
        raise RuntimeError("mc_dropout: failed to collect any stochastic predictions")
    if count <= 1:
        var = torch.zeros_like(mean)
    else:
        var = m2 / float(count - 1)
    if do_sanity and mean is not None and det_xyz is not None:
        diff = (mean - det_xyz).abs().mean().item()
        print(f"[MC_DROPOUT][sanity] mean|mc_mean-det|={diff:.6g}")
        if diff > 1.0:
            print("[MC_DROPOUT][warning] MC dropout mean drift too large; check dropout enabling.")
        _mc_dropout_predict_xyz._sanity_done = True
    return mean, var


def extract_method_outputs(
    model,
    b1: Dict,
    b2: Dict,
    method: str,
    ours_unc_variant: str = "total",
    mc_samples: int = 16,
    ensemble_states: Optional[List[Dict]] = None,
    conf_unc_mode: str = "inv",
    xyz_reduce: str = "trace",
    dump_keys: bool = False,
):
    """
    Returns pred_xyz + scalar unc_map + extra params for NLL.
    """
    device = next(model.parameters()).device
    H, W = b1["img"].shape[-2:]
    base, ours_variant = base_method(method, ours_unc_variant)

    def reduce_var_to_unc(var_b3hw: torch.Tensor) -> torch.Tensor:
        v = torch.clamp(var_b3hw, min=0.0)
        if xyz_reduce == "trace":
            return v.sum(dim=1)  # [B,H,W]
        if xyz_reduce == "l2":
            return torch.sqrt(torch.clamp(v, min=0).sum(dim=1))
        raise ValueError(f"bad xyz_reduce={xyz_reduce}")

    if base == "conf":
        pred1, _ = model(b1, b2)
        if dump_keys:
            print("[conf] pred keys:", sorted(list(pred1.keys())))
        pred_xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
        pred_xyz = resize_to_hw(pred_xyz, (H, W), mode="bilinear")

        conf = get_first(pred1, ["conf", "conf1", "desc_conf"])
        if conf is None:
            raise KeyError("conf method: cannot find conf/conf1/desc_conf in pred")
        conf = to_b1hw(conf)[:, 0]
        conf = resize_to_hw(conf.unsqueeze(1), (H, W), mode="bilinear")[:, 0]
        conf = conf.clamp_min(1e-8)
        if conf_unc_mode == "inv":
            unc = (1.0 / conf).float()
        else:
            unc = (-torch.log(conf)).float()
        return MethodOutputs(pred_xyz=pred_xyz, unc_map=unc, extra={"conf": conf})

    if base == "hetero":
        pred1, _ = model(b1, b2)
        if dump_keys:
            print("[hetero] pred keys:", sorted(list(pred1.keys())))
        pred_xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
        pred_xyz = resize_to_hw(pred_xyz, (H, W), mode="bilinear")

        var = get_first(pred1, ["xyz_var", "xyz_var_diag"])
        if var is None:
            raise KeyError("hetero: cannot find xyz_var/xyz_var_diag")
        var = to_bchw(var)
        var = resize_to_hw(var, (H, W), mode="bilinear")
        unc = reduce_var_to_unc(var)
        return MethodOutputs(pred_xyz=pred_xyz, unc_map=unc, extra={"var_diag": var})

    if base == "ours_nig":
        pred1, _ = model(b1, b2)
        if dump_keys:
            print("[ours_nig] pred keys:", sorted(list(pred1.keys())))
        pred_xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
        pred_xyz = resize_to_hw(pred_xyz, (H, W), mode="bilinear")

        # NLL params (robust key search)
        gamma = get_first(pred1, ["xyz_nig_gamma", "xyz_evi_gamma", "xyz_gamma", "xyz_mu"])
        nu    = get_first(pred1, ["xyz_nig_nu", "xyz_evi_nu", "xyz_nu"])
        alpha = get_first(pred1, ["xyz_nig_alpha", "xyz_evi_alpha", "xyz_alpha"])
        beta  = get_first(pred1, ["xyz_nig_beta", "xyz_evi_beta", "xyz_beta"])
        var_diag = None
        extra = {}
        if all(x is not None for x in [gamma, nu, alpha, beta]):
            gamma = to_bchw(gamma); nu = to_bchw(nu); alpha = to_bchw(alpha); beta = to_bchw(beta)
            gamma = resize_to_hw(gamma, (H, W), "bilinear")
            nu    = resize_to_hw(nu,    (H, W), "bilinear")
            alpha = resize_to_hw(alpha, (H, W), "bilinear")
            beta  = resize_to_hw(beta,  (H, W), "bilinear")
            nu_c = torch.clamp(nu, min=1e-8)
            alpha_c = torch.clamp(alpha, min=1.0 + 1e-8)
            beta_c = torch.clamp(beta, min=1e-8)
            alea = beta_c / (alpha_c - 1.0)
            epi = alea / nu_c
            if ours_variant == "alea":
                var_diag = alea
            elif ours_variant == "epi":
                var_diag = epi
            else:
                var_diag = alea + epi
            extra = {"gamma": gamma, "nu": nu, "alpha": alpha, "beta": beta}

        if var_diag is None:
            var = get_first(pred1, ["xyz_var", "xyz_nig_var", "xyz_var_diag"])
            if var is not None:
                var = to_bchw(var)
                var = resize_to_hw(var, (H, W), mode="bilinear")
                var_diag = var
        if var_diag is not None:
            unc = reduce_var_to_unc(var_diag)
            extra["var_diag"] = var_diag
        else:
            unc = torch.zeros((pred_xyz.shape[0], H, W), device=device)
        return MethodOutputs(pred_xyz=pred_xyz, unc_map=unc, extra=extra)

    if base == "ours_niw":
        pred1, _ = model(b1, b2)
        if dump_keys:
            print("[ours_niw] pred keys:", sorted(list(pred1.keys())))
        pred_xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
        pred_xyz = resize_to_hw(pred_xyz, (H, W), mode="bilinear")

        m = get_first(pred1, ["xyz_niw_m", "xyz_mu"])
        kappa = get_first(pred1, ["xyz_niw_kappa", "kappa"])
        nu = get_first(pred1, ["xyz_niw_nu", "nu"])
        Lp = get_first(pred1, ["xyz_niw_Lp", "xyz_niw_L", "Lp", "L"])
        var_diag = None
        extra = {}
        if all(x is not None for x in [m, kappa, nu, Lp]):
            m = to_bchw(m)
            m = resize_to_hw(m, (H, W), "bilinear")
            kappa = to_b1hw(kappa)
            kappa = resize_to_hw(kappa, (H, W), "bilinear")
            nu = to_b1hw(nu)
            nu = resize_to_hw(nu, (H, W), "bilinear")
            # Lp: [B,6,H,W]
            if Lp.ndim == 4 and Lp.shape[1] == 6:
                pass
            elif Lp.ndim == 4 and Lp.shape[-1] == 6:
                Lp = Lp.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError(f"NIW Lp shape unexpected: {tuple(Lp.shape)}")
            Lp = resize_to_hw(Lp, (H, W), "bilinear")
            extra = {"m": m, "kappa": kappa, "nu": nu, "Lp": Lp}
            kappa_c = torch.clamp(kappa, min=1e-8)
            nu_pred = torch.clamp(nu - 2.0, min=1e-8)
            scale_alea = 1.0 / nu_pred
            scale_epi = 1.0 / (kappa_c * nu_pred)
            if ours_variant == "alea":
                scale = scale_alea
            elif ours_variant == "epi":
                scale = scale_epi
            else:
                scale = scale_alea + scale_epi

            l00 = Lp[:, 0]
            l10 = Lp[:, 1]
            l11 = Lp[:, 2]
            l20 = Lp[:, 3]
            l21 = Lp[:, 4]
            l22 = Lp[:, 5]
            diag0 = l00 * l00
            diag1 = l10 * l10 + l11 * l11
            diag2 = l20 * l20 + l21 * l21 + l22 * l22
            psi_diag = torch.stack([diag0, diag1, diag2], dim=1)
            var_diag = psi_diag * scale

        if var_diag is None:
            var = get_first(pred1, ["xyz_var", "xyz_niw_var_diag", "xyz_var_diag"])
            if var is not None:
                var = to_bchw(var)
                var = resize_to_hw(var, (H, W), mode="bilinear")
                var_diag = var
        if var_diag is not None:
            unc = reduce_var_to_unc(var_diag)
            extra["var_diag"] = var_diag
        else:
            unc = torch.zeros((pred_xyz.shape[0], H, W), device=device)
        return MethodOutputs(pred_xyz=pred_xyz, unc_map=unc, extra=extra)

    if base == "mc_dropout":
        pred_xyz, var = _mc_dropout_predict_xyz(model, b1, b2, mc_samples)
        unc = reduce_var_to_unc(var)
        return MethodOutputs(pred_xyz=pred_xyz, unc_map=unc, extra={"var_diag": var, "supports_nll": True})

    if base == "ensemble":
        if not ensemble_states:
            raise ValueError("ensemble requested but ensemble_states is empty")
        mean = None
        m2 = None
        count = 0
        for st in ensemble_states:
            model.load_state_dict(st, strict=False)
            model.eval()
            pred1, _ = model(b1, b2)
            xyz = to_bchw(pred1.get("xyz_mu", pred1.get("pts3d")))
            xyz = resize_to_hw(xyz, (H, W), "bilinear")
            if count == 0:
                mean = xyz.clone()
                m2 = torch.zeros_like(xyz)
                count = 1
            else:
                count += 1
                delta = xyz - mean
                mean = mean + delta / count
                delta2 = xyz - mean
                m2 = m2 + delta * delta2
        var = m2 / max(count, 1)
        unc = reduce_var_to_unc(var)
        return MethodOutputs(pred_xyz=mean, unc_map=unc, extra={"var_diag": var})

    raise ValueError(f"Unknown method: {method}")
