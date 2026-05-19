# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import math
import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf, inv
from mast3r.losses_evidential import (
    evidential_nig_loss,
    predictive_mean_var,
    evidential_nig_loss_3d,
    evidential_niw_loss_3d,
)


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def _get_first(view, keys):
    for k in keys:
        if isinstance(view, dict) and k in view:
            return view[k]
    return None


def _get_gt_xyz(view1, view2, device):
    pts1 = _get_first(view1, ("pts3d",))
    pts2 = _get_first(view2, ("pts3d",))
    cam1 = _get_first(view1, ("camera_pose",))

    if pts1 is None or pts2 is None or cam1 is None:
        return None

    pts1 = pts1.to(device, non_blocking=True)
    pts2 = pts2.to(device, non_blocking=True)
    cam1 = cam1.to(device, non_blocking=True)

    world_to_cam1 = inv(cam1)
    gt_xyz1 = geotrf(world_to_cam1, pts1)
    gt_xyz2 = geotrf(world_to_cam1, pts2)

    if gt_xyz1.ndim == 4 and gt_xyz1.shape[-1] == 3:
        gt_xyz1 = gt_xyz1.permute(0, 3, 1, 2)
    if gt_xyz2.ndim == 4 and gt_xyz2.shape[-1] == 3:
        gt_xyz2 = gt_xyz2.permute(0, 3, 1, 2)

    return gt_xyz1, gt_xyz2


def _to_bchw(x):
    if x is None:
        return None
    if x.ndim == 4 and x.shape[1] == 3:
        return x
    if x.ndim == 4 and x.shape[-1] == 3:
        return x.permute(0, 3, 1, 2)
    return x


def _to_b1hw(x):
    if x is None:
        return None
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    if x.ndim == 4 and x.shape[-1] == 1:
        return x.permute(0, 3, 1, 2).contiguous()
    return None


def _finite_mask(mask, *tensors):
    if mask is None:
        return None
    out = mask.float()
    for t in tensors:
        if t is None:
            continue
        out = out * torch.isfinite(t).to(dtype=out.dtype)
    return out


def _masked_pearson_corr_loss(u_t, u_o, mask, eps=1e-6) -> torch.Tensor:
    if u_t is None or u_o is None or mask is None:
        _masked_pearson_corr_loss._last_reason = "missing"
        ref = u_t if torch.is_tensor(u_t) else u_o
        return torch.zeros((), dtype=torch.float32) if ref is None else ref.new_zeros(())

    mask = mask.to(u_t.device, non_blocking=True).float()
    valid = mask.sum()
    if valid.detach().item() < 16.0:
        _masked_pearson_corr_loss._last_reason = "low_valid"
        return u_t.new_zeros(())

    denom = valid.clamp_min(1.0)
    mt = (u_t * mask).sum() / denom
    mo = (u_o * mask).sum() / denom

    vt = (u_t - mt) * mask
    vo = (u_o - mo) * mask

    cov = (vt * vo).sum() / denom
    var_t = (vt * vt).sum() / denom
    var_o = (vo * vo).sum() / denom
    if var_t.detach().item() <= eps or var_o.detach().item() <= eps:
        _masked_pearson_corr_loss._last_reason = "low_std"
        return u_t.new_zeros(())

    st = torch.sqrt(var_t + eps)
    so = torch.sqrt(var_o + eps)
    corr = cov / (st * so + eps)
    if not torch.isfinite(corr).detach().item():
        _masked_pearson_corr_loss._last_reason = "nan"
        return u_t.new_zeros(())

    corr = corr.clamp(-1.0, 1.0)
    loss = 1.0 - corr
    # Sanity: random finite tensors should produce a finite scalar loss.
    if not torch.isfinite(loss).detach().item():
        _masked_pearson_corr_loss._last_reason = "nan"
        return u_t.new_zeros(())
    _masked_pearson_corr_loss._last_reason = "ok"
    return loss


def gaussian_nll_xyz(gt_xyz, pred_mu, pred_logvar, mask, eps=1e-6):
    var = torch.exp(pred_logvar) + eps
    nll = 0.5 * (math.log(2.0 * math.pi) + pred_logvar + (gt_xyz - pred_mu) ** 2 / var)
    nll = nll.sum(dim=1)
    if mask is not None:
        mask = mask.to(nll.device, non_blocking=True).float()
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        nll = nll * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(nll.numel(), dtype=nll.dtype, device=nll.device)
    return nll.sum() / denom


def _compute_uq_loss(view1, view2, pred1, pred2, args):
    if args is None:
        return None

    lam_geom = getattr(args, 'lambda_uq', 0.0)
    lam_feat_evi = getattr(args, 'lambda_evi_feat', 0.0)
    lam_mean_gt = getattr(args, 'lambda_mean_gt', 0.0)
    lam_mean_distill = getattr(args, 'lambda_mean_distill', 0.0)
    lam_var_distill = getattr(args, 'lambda_var_distill', 0.0)
    lam_uq_xyz = getattr(args, "lambda_uq_xyz", 0.0)
    lam_hetero = getattr(args, "lambda_hetero_xyz", 0.0)
    lam_uc_corr = getattr(args, "lambda_uc_corr", 0.0)

    if max(
        lam_geom,
        lam_feat_evi,
        lam_mean_gt,
        lam_mean_distill,
        lam_var_distill,
        lam_uq_xyz,
        lam_hetero,
        lam_uc_corr,
    ) <= 0:
        return None

    depth1 = _get_first(view1, ('depth', 'depth_gt', 'depthmap', 'depth1', 'metric_depth'))
    depth2 = _get_first(view2, ('depth', 'depth_gt', 'depthmap', 'depth2', 'metric_depth'))
    valid1 = _get_first(view1, ('valid_depth', 'valid1', 'valid_mask', 'mask'))
    valid2 = _get_first(view2, ('valid_depth', 'valid2', 'valid_mask', 'mask'))
    if depth1 is None or depth2 is None or valid1 is None or valid2 is None:
        return None

    def _to_mask(mask, target):
        if mask is None:
            return None
        mask = mask.to(target.device, non_blocking=True).float()
        if mask.ndim == target.ndim - 1:
            mask = mask.unsqueeze(1)
        return mask

    def _masked_mean(x, mask):
        if mask is not None:
            x = x * mask
            denom = mask.sum().clamp_min(1.0)
        else:
            denom = torch.tensor(x.numel(), dtype=x.dtype, device=x.device)
        return x.sum() / denom

    candidate_devices = [
        _get_first(pred1, ("depth_gamma", "gamma", "depth_gamma_feat")),
        _get_first(pred1, ("pts3d",)),
        _get_first(pred2, ("depth_gamma", "gamma", "depth_gamma_feat")),
        _get_first(pred2, ("pts3d",)),
        depth1,
    ]
    target_device = None
    for cand in candidate_devices:
        if torch.is_tensor(cand):
            target_device = cand.device
            break
    target_device = target_device or depth1.device

    depth1 = depth1.to(target_device, non_blocking=True)
    depth2 = depth2.to(target_device, non_blocking=True)
    valid1 = _to_mask(valid1, depth1)
    valid2 = _to_mask(valid2, depth2)

    losses = {}
    stats = {}

    m1_xyz = _get_first(pred1, ("xyz_niw_m",))
    kappa1_xyz = _get_first(pred1, ("xyz_niw_kappa",))
    nu1_xyz = _get_first(pred1, ("xyz_niw_nu",))
    L1_xyz = _get_first(pred1, ("xyz_niw_L",))

    m2_xyz = _get_first(pred2, ("xyz_niw_m",))
    kappa2_xyz = _get_first(pred2, ("xyz_niw_kappa",))
    nu2_xyz = _get_first(pred2, ("xyz_niw_nu",))
    L2_xyz = _get_first(pred2, ("xyz_niw_L",))

    have_niw = all(
        x is not None
        for x in (m1_xyz, kappa1_xyz, nu1_xyz, L1_xyz, m2_xyz, kappa2_xyz, nu2_xyz, L2_xyz)
    )

    # Geometry-level evidential loss (teacher)
    if getattr(args, "uq_mode", "geom") in ("geom", "both") and lam_geom > 0:
        gamma1 = _get_first(pred1, ('depth_gamma', 'gamma'))
        nu1 = _get_first(pred1, ('depth_nu', 'nu'))
        alpha1 = _get_first(pred1, ('depth_alpha', 'alpha'))
        beta1 = _get_first(pred1, ('depth_beta', 'beta'))

        gamma2 = _get_first(pred2, ('depth_gamma', 'gamma'))
        nu2 = _get_first(pred2, ('depth_nu', 'nu'))
        alpha2 = _get_first(pred2, ('depth_alpha', 'alpha'))
        beta2 = _get_first(pred2, ('depth_beta', 'beta'))

        if all(x is not None for x in (gamma1, nu1, alpha1, beta1, gamma2, nu2, alpha2, beta2)):
            loss_uq_1 = evidential_nig_loss(depth1, gamma1, nu1, alpha1, beta1,
                                            mask=valid1,
                                            lambda_evi=getattr(args, 'lambda_evi', 1e-3))
            loss_uq_2 = evidential_nig_loss(depth2, gamma2, nu2, alpha2, beta2,
                                            mask=valid2,
                                            lambda_evi=getattr(args, 'lambda_evi', 1e-3))
            losses['loss_uq_geom'] = 0.5 * (loss_uq_1 + loss_uq_2)
            stats.update({
                'loss_uq_geom_1': float(loss_uq_1.detach()),
                'loss_uq_geom_2': float(loss_uq_2.detach()),
                'loss_uq_geom': float(losses['loss_uq_geom'].detach()),
            })

            depth_mu1_t, depth_var1_t = predictive_mean_var(gamma1, nu1, alpha1, beta1)
            depth_mu2_t, depth_var2_t = predictive_mean_var(gamma2, nu2, alpha2, beta2)
            stats.update({
                'depth_mu_geom_1_mean': float(depth_mu1_t.detach().mean()),
                'depth_mu_geom_2_mean': float(depth_mu2_t.detach().mean()),
                'depth_var_geom_1_mean': float(depth_var1_t.detach().mean()),
                'depth_var_geom_2_mean': float(depth_var2_t.detach().mean()),
            })

    # Feature-level evidential loss (student)
    if getattr(args, "uq_mode", "geom") in ("feat", "both") and (
        lam_feat_evi > 0 or lam_mean_gt > 0 or lam_mean_distill > 0 or lam_var_distill > 0
    ):
        gamma1_f = _get_first(pred1, ("depth_gamma_feat",))
        nu1_f = _get_first(pred1, ("depth_nu_feat",))
        alpha1_f = _get_first(pred1, ("depth_alpha_feat",))
        beta1_f = _get_first(pred1, ("depth_beta_feat",))

        gamma2_f = _get_first(pred2, ("depth_gamma_feat",))
        nu2_f = _get_first(pred2, ("depth_nu_feat",))
        alpha2_f = _get_first(pred2, ("depth_alpha_feat",))
        beta2_f = _get_first(pred2, ("depth_beta_feat",))

        have_feat = all(x is not None for x in (gamma1_f, nu1_f, alpha1_f, beta1_f, gamma2_f, nu2_f, alpha2_f, beta2_f))
        if have_feat:
            depth_mean1_feat = _get_first(pred1, ("depth_mean_feat",))
            depth_mean2_feat = _get_first(pred2, ("depth_mean_feat",))

            if lam_feat_evi > 0:
                loss_uq_f1 = evidential_nig_loss(depth1, gamma1_f, nu1_f, alpha1_f, beta1_f,
                                                 mask=valid1,
                                                 lambda_evi=getattr(args, 'lambda_evi', 1e-3))
                loss_uq_f2 = evidential_nig_loss(depth2, gamma2_f, nu2_f, alpha2_f, beta2_f,
                                                 mask=valid2,
                                                 lambda_evi=getattr(args, 'lambda_evi', 1e-3))
                losses['loss_uq_feat'] = 0.5 * (loss_uq_f1 + loss_uq_f2)
                stats.update({
                    'loss_uq_feat_1': float(loss_uq_f1.detach()),
                    'loss_uq_feat_2': float(loss_uq_f2.detach()),
                    'loss_uq_feat': float(losses['loss_uq_feat'].detach()),
                })

            depth_mu1_feat, depth_var1_feat = predictive_mean_var(gamma1_f, nu1_f, alpha1_f, beta1_f)
            depth_mu2_feat, depth_var2_feat = predictive_mean_var(gamma2_f, nu2_f, alpha2_f, beta2_f)

            student_mu1 = depth_mean1_feat if depth_mean1_feat is not None else depth_mu1_feat
            student_mu2 = depth_mean2_feat if depth_mean2_feat is not None else depth_mu2_feat

            if lam_mean_gt > 0:
                l_gt1 = _masked_mean((student_mu1 - depth1) ** 2, valid1)
                l_gt2 = _masked_mean((student_mu2 - depth2) ** 2, valid2)
                losses['loss_mean_gt'] = 0.5 * (l_gt1 + l_gt2)
                stats.update({
                    'loss_mean_gt_1': float(l_gt1.detach()),
                    'loss_mean_gt_2': float(l_gt2.detach()),
                    'loss_mean_gt': float(losses['loss_mean_gt'].detach()),
                })

            teacher_mu1 = _get_first(pred1, ("depth_mu", "depth_mean", "depth_gamma", "gamma"))
            teacher_mu2 = _get_first(pred2, ("depth_mu", "depth_mean", "depth_gamma", "gamma"))
            teacher_var1 = _get_first(pred1, ("depth_var",))
            teacher_var2 = _get_first(pred2, ("depth_var",))

            if lam_mean_distill > 0 and teacher_mu1 is not None and teacher_mu2 is not None:
                l_md1 = _masked_mean((student_mu1 - teacher_mu1) ** 2, valid1)
                l_md2 = _masked_mean((student_mu2 - teacher_mu2) ** 2, valid2)
                losses['loss_mean_distill'] = 0.5 * (l_md1 + l_md2)
                stats.update({
                    'loss_mean_distill_1': float(l_md1.detach()),
                    'loss_mean_distill_2': float(l_md2.detach()),
                    'loss_mean_distill': float(losses['loss_mean_distill'].detach()),
                })

            if lam_var_distill > 0 and teacher_var1 is not None and teacher_var2 is not None:
                l_vd1 = _masked_mean(torch.abs(depth_var1_feat - teacher_var1), valid1)
                l_vd2 = _masked_mean(torch.abs(depth_var2_feat - teacher_var2), valid2)
                losses['loss_var_distill'] = 0.5 * (l_vd1 + l_vd2)
                stats.update({
                    'loss_var_distill_1': float(l_vd1.detach()),
                    'loss_var_distill_2': float(l_vd2.detach()),
                    'loss_var_distill': float(losses['loss_var_distill'].detach()),
                })

            stats.update({
                'depth_mu_feat_1_mean': float(depth_mu1_feat.detach().mean()),
                'depth_mu_feat_2_mean': float(depth_mu2_feat.detach().mean()),
                'depth_var_feat_1_mean': float(depth_var1_feat.detach().mean()),
                'depth_var_feat_2_mean': float(depth_var2_feat.detach().mean()),
            })
            if depth_mean1_feat is not None:
                stats['depth_mean_feat_1_mean'] = float(depth_mean1_feat.detach().mean())
            if depth_mean2_feat is not None:
                stats['depth_mean_feat_2_mean'] = float(depth_mean2_feat.detach().mean())

    if lam_uq_xyz > 0:
        if have_niw:
            gt_xyz = _get_gt_xyz(view1, view2, target_device)
            if gt_xyz is not None:
                y_xyz1, y_xyz2 = gt_xyz
                lam_evi_xyz = getattr(args, "lambda_evi_xyz", 1e-3)

                loss_xyz1, stats_xyz1 = evidential_niw_loss_3d(
                    y_xyz1, m1_xyz, kappa1_xyz, nu1_xyz, L1_xyz,
                    mask=valid1, lambda_evi=lam_evi_xyz,
                )
                loss_xyz2, stats_xyz2 = evidential_niw_loss_3d(
                    y_xyz2, m2_xyz, kappa2_xyz, nu2_xyz, L2_xyz,
                    mask=valid2, lambda_evi=lam_evi_xyz,
                )
                losses["loss_uq_xyz"] = 0.5 * (loss_xyz1 + loss_xyz2)
                stats.update({
                    "loss_xyz_niw_1": float(loss_xyz1.detach()),
                    "loss_xyz_niw_2": float(loss_xyz2.detach()),
                    "loss_xyz_niw": float(losses["loss_uq_xyz"].detach()),
                })
                if stats_xyz1:
                    stats.update({f"xyz_niw_{k}_1": v for k, v in stats_xyz1.items()})
                if stats_xyz2:
                    stats.update({f"xyz_niw_{k}_2": v for k, v in stats_xyz2.items()})
        else:
            gamma1_xyz = _get_first(pred1, ("xyz_gamma",))
            nu1_xyz = _get_first(pred1, ("xyz_nu",))
            alpha1_xyz = _get_first(pred1, ("xyz_alpha",))
            beta1_xyz = _get_first(pred1, ("xyz_beta",))

            gamma2_xyz = _get_first(pred2, ("xyz_gamma",))
            nu2_xyz = _get_first(pred2, ("xyz_nu",))
            alpha2_xyz = _get_first(pred2, ("xyz_alpha",))
            beta2_xyz = _get_first(pred2, ("xyz_beta",))

            have_xyz = all(
                x is not None
                for x in (gamma1_xyz, nu1_xyz, alpha1_xyz, beta1_xyz, gamma2_xyz, nu2_xyz, alpha2_xyz, beta2_xyz)
            )
            if have_xyz:
                gt_xyz = _get_gt_xyz(view1, view2, target_device)
                if gt_xyz is not None:
                    y_xyz1, y_xyz2 = gt_xyz
                    lam_evi_xyz = getattr(args, "lambda_evi_xyz", 1e-3)

                    loss_xyz1 = evidential_nig_loss_3d(
                        y_xyz1, gamma1_xyz, nu1_xyz, alpha1_xyz, beta1_xyz,
                        mask=valid1, lambda_evi=lam_evi_xyz,
                    )
                    loss_xyz2 = evidential_nig_loss_3d(
                        y_xyz2, gamma2_xyz, nu2_xyz, alpha2_xyz, beta2_xyz,
                        mask=valid2, lambda_evi=lam_evi_xyz,
                    )
                    losses["loss_uq_xyz"] = 0.5 * (loss_xyz1 + loss_xyz2)
                    stats.update({
                        "loss_uq_xyz_1": float(loss_xyz1.detach()),
                        "loss_uq_xyz_2": float(loss_xyz2.detach()),
                        "loss_uq_xyz": float(losses["loss_uq_xyz"].detach()),
                    })

    if lam_uc_corr > 0 and have_niw:
        uc_conf_key = getattr(args, "uc_conf_key", "conf")
        if isinstance(uc_conf_key, str) and uc_conf_key:
            conf_keys = (uc_conf_key, "conf", "confidence")
        else:
            conf_keys = ("conf", "confidence")

        conf1 = _to_b1hw(_get_first(pred1, conf_keys))
        conf2 = _to_b1hw(_get_first(pred2, conf_keys))
        if conf1 is None or conf2 is None:
            stats["uc_corr_skipped"] = 1.0
        else:
            conf1 = conf1.to(target_device, non_blocking=True).float()
            conf2 = conf2.to(target_device, non_blocking=True).float()
            uT1 = (-torch.log(conf1.clamp_min(1e-6))).detach()
            uT2 = (-torch.log(conf2.clamp_min(1e-6))).detach()

            def _build_niw_uo(pred):
                use_mode = getattr(args, "uc_corr_use", "total")
                nu_map = _to_b1hw(_get_first(pred, ("xyz_niw_nu",)))
                if nu_map is None:
                    return None
                nu_map = nu_map.to(target_device, non_blocking=True).float()

                if use_mode == "epistemic":
                    kappa_map = _to_b1hw(_get_first(pred, ("xyz_niw_kappa",)))
                    L_map = _get_first(pred, ("xyz_niw_L",))
                    if kappa_map is None or L_map is None:
                        return None
                    if L_map.ndim == 4 and L_map.shape[-1] == 6:
                        L_map = L_map.permute(0, 3, 1, 2).contiguous()
                    if not (L_map.ndim == 4 and L_map.shape[1] == 6):
                        return None
                    kappa_map = kappa_map.to(target_device, non_blocking=True).float()
                    L_map = L_map.to(target_device, non_blocking=True).float()
                    l00, l10, l11, l20, l21, l22 = torch.chunk(L_map, 6, dim=1)
                    psi00 = l00 ** 2
                    psi11 = l10 ** 2 + l11 ** 2
                    psi22 = l20 ** 2 + l21 ** 2 + l22 ** 2
                    psi_diag = torch.cat([psi00, psi11, psi22], dim=1)
                    denom = (nu_map - 4.0).clamp_min(1e-6)
                    epi_cov_diag = psi_diag / (kappa_map.clamp_min(1e-6) * denom)
                    u_o = epi_cov_diag.sum(dim=1, keepdim=True)
                else:
                    xyz_var = _to_bchw(_get_first(pred, ("xyz_var", "xyz_niw_var_diag")))
                    if xyz_var is None or xyz_var.ndim != 4 or xyz_var.shape[1] != 3:
                        return None
                    xyz_var = xyz_var.to(target_device, non_blocking=True).float()
                    nu_pred = (nu_map - 2.0).clamp_min(1e-6)
                    cov_factor = nu_pred / (nu_pred - 2.0).clamp_min(1e-6)
                    cov_diag = xyz_var * cov_factor
                    u_o = cov_diag.sum(dim=1, keepdim=True)

                if int(getattr(args, "uc_corr_log", 1)) == 1:
                    u_o = torch.log1p(u_o.clamp_min(0.0))
                return u_o

            uO1 = _build_niw_uo(pred1)
            uO2 = _build_niw_uo(pred2)
            valid1_uc = _to_b1hw(valid1)
            valid2_uc = _to_b1hw(valid2)

            if uO1 is None or uO2 is None or valid1_uc is None or valid2_uc is None:
                stats["uc_corr_skipped"] = 1.0
            else:
                valid1_uc = valid1_uc.to(target_device, non_blocking=True).float()
                valid2_uc = valid2_uc.to(target_device, non_blocking=True).float()
                mask1 = _finite_mask(valid1_uc, uT1, uO1)
                mask2 = _finite_mask(valid2_uc, uT2, uO2)
                if mask1 is None or mask2 is None:
                    stats["uc_corr_skipped"] = 1.0
                else:
                    if mask1.sum().detach().item() < 16.0 or mask2.sum().detach().item() < 16.0:
                        stats["uc_corr_low_valid"] = 1.0
                    l_uc1 = _masked_pearson_corr_loss(uT1, uO1, mask1)
                    reason1 = getattr(_masked_pearson_corr_loss, "_last_reason", "ok")
                    l_uc2 = _masked_pearson_corr_loss(uT2, uO2, mask2)
                    reason2 = getattr(_masked_pearson_corr_loss, "_last_reason", "ok")
                    if "nan" in (reason1, reason2):
                        stats["uc_corr_nan"] = 1.0
                    if "low_std" in (reason1, reason2):
                        stats["uc_corr_low_std"] = 1.0
                    loss_uc = 0.5 * (l_uc1 + l_uc2)
                    if not torch.isfinite(loss_uc).detach().item():
                        loss_uc = uT1.new_zeros(())
                        stats["uc_corr_nan"] = 1.0
                    losses["loss_uc_corr"] = loss_uc
                    stats["loss_uc_corr"] = float(loss_uc.detach())

    if lam_hetero > 0:
        xyz_mu1 = _get_first(pred1, ("xyz_mu", "pts3d_refined", "pts3d"))
        xyz_mu2 = _get_first(pred2, ("xyz_mu", "pts3d_refined", "pts3d"))
        xyz_logvar1 = _get_first(pred1, ("xyz_logvar",))
        xyz_logvar2 = _get_first(pred2, ("xyz_logvar",))

        have_hetero = all(x is not None for x in (xyz_mu1, xyz_mu2, xyz_logvar1, xyz_logvar2))
        if have_hetero:
            gt_xyz = _get_gt_xyz(view1, view2, target_device)
            if gt_xyz is not None:
                y_xyz1, y_xyz2 = gt_xyz
                xyz_mu1 = _to_bchw(xyz_mu1).to(target_device, non_blocking=True)
                xyz_mu2 = _to_bchw(xyz_mu2).to(target_device, non_blocking=True)
                xyz_logvar1 = _to_bchw(xyz_logvar1).to(target_device, non_blocking=True)
                xyz_logvar2 = _to_bchw(xyz_logvar2).to(target_device, non_blocking=True)

                loss_h1 = gaussian_nll_xyz(y_xyz1, xyz_mu1, xyz_logvar1, valid1)
                loss_h2 = gaussian_nll_xyz(y_xyz2, xyz_mu2, xyz_logvar2, valid2)
                losses["loss_hetero_xyz"] = 0.5 * (loss_h1 + loss_h2)
                stats.update({
                    "loss_hetero_xyz_1": float(loss_h1.detach()),
                    "loss_hetero_xyz_2": float(loss_h2.detach()),
                    "loss_hetero_xyz": float(losses["loss_hetero_xyz"].detach()),
                })

    if not losses:
        return None

    return losses, stats


def loss_of_one_batch(batch, model, criterion, device, symmetrize_batch=False, use_amp=False, ret=None, args=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        pred1, pred2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            base_loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    loss_details = {}
    if base_loss is None:
        loss = torch.tensor(0.0, device=device)
    elif isinstance(base_loss, tuple):
        loss, loss_details = base_loss
    else:
        loss = base_loss
        if loss.ndim == 0:
            loss_details = {'loss_base': float(loss)}

    uq_res = _compute_uq_loss(view1, view2, pred1, pred2, args)
    if uq_res is not None:
        uq_losses, uq_stats = uq_res
        if 'loss_uq_geom' in uq_losses:
            loss = loss + getattr(args, 'lambda_uq', 1.0) * uq_losses['loss_uq_geom']
        if 'loss_uq_feat' in uq_losses:
            loss = loss + getattr(args, 'lambda_evi_feat', 0.0) * uq_losses['loss_uq_feat']
        if 'loss_mean_gt' in uq_losses:
            loss = loss + getattr(args, 'lambda_mean_gt', 0.0) * uq_losses['loss_mean_gt']
        if 'loss_mean_distill' in uq_losses:
            loss = loss + getattr(args, 'lambda_mean_distill', 0.0) * uq_losses['loss_mean_distill']
        if 'loss_var_distill' in uq_losses:
            loss = loss + getattr(args, 'lambda_var_distill', 0.0) * uq_losses['loss_var_distill']
        lam_uq_xyz = getattr(args, "lambda_uq_xyz", 0.0)
        if "loss_uq_xyz" in uq_losses and lam_uq_xyz > 0:
            loss = loss + lam_uq_xyz * uq_losses["loss_uq_xyz"]
            loss_details["loss_uq_xyz"] = float(uq_losses["loss_uq_xyz"].detach())
        lam_uc = getattr(args, "lambda_uc_corr", 0.0)
        if "loss_uc_corr" in uq_losses and lam_uc > 0:
            loss = loss + lam_uc * uq_losses["loss_uc_corr"]
            loss_details["loss_uc_corr"] = float(uq_losses["loss_uc_corr"].detach())
        lam_hetero = getattr(args, "lambda_hetero_xyz", 0.0)
        if "loss_hetero_xyz" in uq_losses and lam_hetero > 0:
            loss = loss + lam_hetero * uq_losses["loss_hetero_xyz"]
            loss_details["loss_hetero_xyz"] = float(uq_losses["loss_hetero_xyz"].detach())
        loss_details.update(uq_stats)

    lam_gate_tv = getattr(args, "lambda_gate_tv", 0.0) if args is not None else 0.0
    if lam_gate_tv > 0:
        def _to_gate_bchw(x):
            if x is None or not torch.is_tensor(x):
                return None
            if x.ndim != 4:
                return None
            if x.shape[1] <= 8:  # likely BCHW
                return x
            if x.shape[-1] <= 8:  # likely BHWC
                return x.permute(0, 3, 1, 2).contiguous()
            return None

        def _get_gate_map(pred):
            gate = _to_gate_bchw(_get_first(pred, ("rg_gate", "gr_gate")))
            if gate is not None:
                return gate
            gate_logits = _to_gate_bchw(_get_first(pred, ("rg_gate_logits", "gr_gate_logits", "gate_logits")))
            if gate_logits is not None:
                return torch.sigmoid(gate_logits)
            return None

        g1 = _get_gate_map(pred1)
        g2 = _get_gate_map(pred2)
        if g1 is not None and g2 is not None:
            tv1 = torch.mean(torch.abs(g1[:, :, :, 1:] - g1[:, :, :, :-1])) + \
                torch.mean(torch.abs(g1[:, :, 1:, :] - g1[:, :, :-1, :]))
            tv2 = torch.mean(torch.abs(g2[:, :, :, 1:] - g2[:, :, :, :-1])) + \
                torch.mean(torch.abs(g2[:, :, 1:, :] - g2[:, :, :-1, :]))
            gate_tv = 0.5 * (tv1 + tv2)
            gate_warmup = int(getattr(args, "gate_tv_warmup_steps", 2000))
            global_step = float(getattr(args, "global_step", getattr(args, "_global_step", 0.0)))
            if gate_warmup > 0:
                gate_warm = min(1.0, max(0.0, global_step / float(gate_warmup)))
            else:
                gate_warm = 1.0
            loss = loss + lam_gate_tv * gate_warm * gate_tv
            loss_details["loss_gate_tv"] = float(gate_tv.detach())
            loss_details["gate_tv_warm"] = float(gate_warm)

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=(loss, loss_details))
    return result[ret] if ret else result


@torch.no_grad()
def inference(pairs, model, device, batch_size=8, verbose=True):
    if verbose:
        print(f'>> Inference with model on {len(pairs)} image pairs')
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(pairs))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    for i in tqdm.trange(0, len(pairs), batch_size, disable=not verbose):
        res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        result.append(to_cpu(res))

    result = collate_with_cat(result, lists=multiple_shapes)

    return result


def check_if_same_size(pairs):
    shapes1 = [img1['img'].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2['img'].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(shapes2[0] == s for s in shapes2)


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif use_pose and 'pts3d_refined_in_other_view' in pred:
        assert use_pose is True
        return pred['pts3d_refined_in_other_view']  # return!

    elif use_pose and 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    elif 'pts3d_refined' in pred:
        # refined pts3d from my camera
        pts3d = pred['pts3d_refined']

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(gt_pts1, gt_pts2, pr_pts1, pr_pts2=None, fit_mode='weiszfeld_stop_grad', valid1=None, valid2=None):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None

    all_gt = torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1) if gt_pts2 is not None else nan_gt_pts1
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith('avg'):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith('median'):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith('weiszfeld'):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f'bad {fit_mode=}')

    if fit_mode.endswith('stop_grad'):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
