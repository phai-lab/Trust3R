import math
import torch
import torch.nn.functional as F
from typing import Optional


def nig_nll(y, gamma, nu, alpha, beta, eps: float = 1e-8):
    """
    Negative log-likelihood of the Student-t predictive distribution
    induced by a Normal-Inverse-Gamma prior.

    Args:
        y:      ground-truth scalar target (metric depth), [B,1,H,W]
        gamma:  NIG location parameter, same shape as y
        nu:     NIG evidence parameter (> 0)
        alpha:  NIG shape parameter (> 1)
        beta:   NIG scale parameter (> 0)
    """
    two_beta_nu = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / (nu + eps))
        - alpha * torch.log(two_beta_nu + eps)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_beta_nu + eps)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll


def nig_evidence_regularizer(y, gamma, nu, alpha):
    """
    Evidence regularizer corresponding to |y - gamma| * (2*nu + alpha).
    Penalizes being confident while wrong.
    """
    return torch.abs(y - gamma) * (2.0 * nu + alpha)


def nig_reg(y, gamma, nu, alpha):
    # alias for convenience
    return nig_evidence_regularizer(y, gamma, nu, alpha)


def evidential_nig_loss(
    y,
    gamma,
    nu,
    alpha,
    beta,
    mask=None,
    lambda_evi: float = 1e-3,
    eps: float = 1e-8,
):
    """
    Full evidential loss:
        L = NLL + lambda_evi * |y - gamma| * (2*nu + alpha),
    averaged over valid pixels defined by the mask.
    """
    nll = nig_nll(y, gamma, nu, alpha, beta, eps=eps)
    # detach y in the regularizer so that it doesn't fight the NLL too hard
    reg = nig_evidence_regularizer(y.detach(), gamma, nu, alpha)

    loss = nll + lambda_evi * reg

    if mask is not None:
        loss = loss * mask

    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(loss.numel(), dtype=loss.dtype, device=loss.device)

    return loss.sum() / denom


def predictive_mean_var(gamma, nu, alpha, beta, eps: float = 1e-8):
    """
    Compute predictive mean and variance of the Student-t induced by NIG params.
    """
    mean = gamma
    var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + eps)
    return mean, var


def evidential_nig_loss_3d(
    y_xyz: torch.Tensor,
    gamma_xyz: torch.Tensor,
    nu_xyz: torch.Tensor,
    alpha_xyz: torch.Tensor,
    beta_xyz: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_evi: float = 1e-3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    3D evidential NIG loss over per-pixel xyz coordinates.

    y_xyz, gamma_xyz, nu_xyz, alpha_xyz, beta_xyz: [B, 3, H, W]
    mask: optional [B, H, W] or [B, 1, H, W] boolean/float mask.
    We treat the 3 channels as 3 independent scalar NIG problems and average.
    """
    assert (
        y_xyz.shape == gamma_xyz.shape == nu_xyz.shape == alpha_xyz.shape == beta_xyz.shape
    ), "All xyz tensors must have the same shape"
    B, C, H, W = y_xyz.shape
    assert C == 3, f"expected 3D xyz, got C={C}"

    # Flatten xyz channel into the batch dimension: [B*3, 1, H, W]
    y_flat = y_xyz.reshape(B * C, 1, H, W)
    gamma_flat = gamma_xyz.reshape(B * C, 1, H, W)
    nu_flat = nu_xyz.reshape(B * C, 1, H, W)
    alpha_flat = alpha_xyz.reshape(B * C, 1, H, W)
    beta_flat = beta_xyz.reshape(B * C, 1, H, W)

    # Element-wise NIG NLL and evidence regularizer
    nll = nig_nll(y_flat, gamma_flat, nu_flat, alpha_flat, beta_flat, eps=eps)
    reg = nig_evidence_regularizer(y_flat, gamma_flat, nu_flat, alpha_flat)
    loss = nll + lambda_evi * reg  # [B*3, 1, H, W]

    if mask is not None:
        # Support mask of shape [B, H, W] or [B, 1, H, W]
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, H, W]
        elif mask.dim() != 4:
            raise ValueError(f"mask must have shape [B,H,W] or [B,1,H,W], got {mask.shape}")

        # Broadcast mask over xyz channels: [B,1,H,W] -> [B,3,H,W] -> [B*3,1,H,W]
        mask_exp = mask.expand(B, C, H, W)
        mask_flat = mask_exp.reshape(B * C, 1, H, W)

        loss = loss * mask_flat
        denom = mask_flat.sum().clamp_min(1.0)
    else:
        # Average over all elements
        denom = torch.tensor(loss.numel(), dtype=loss.dtype, device=loss.device)

    return loss.sum() / denom


def evidential_niw_loss_3d(
    y_xyz: torch.Tensor,
    m_xyz: torch.Tensor,
    kappa: torch.Tensor,
    nu: torch.Tensor,
    L: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_evi: float = 1e-3,
    eps: float = 1e-8,
):
    """
    3D evidential NIW loss using the predictive multivariate Student-t NLL.

    y_xyz, m_xyz: [B, 3, H, W]
    kappa, nu: [B, 1, H, W]
    L: [B, 6, H, W] lower-triangular params for Psi.
    mask: optional [B, H, W] or [B, 1, H, W] boolean/float mask.
    Returns (loss, stats_dict).
    """
    assert y_xyz.shape == m_xyz.shape, "y_xyz and m_xyz must have the same shape"
    B, C, H, W = y_xyz.shape
    assert C == 3, f"expected 3D xyz, got C={C}"
    assert kappa.shape == (B, 1, H, W)
    assert nu.shape == (B, 1, H, W)
    assert L.shape == (B, 6, H, W)

    d = 3
    y = y_xyz.permute(0, 2, 3, 1).reshape(-1, d)
    m = m_xyz.permute(0, 2, 3, 1).reshape(-1, d)
    diff = (y - m).unsqueeze(-1)

    Lp = L.permute(0, 2, 3, 1).reshape(-1, 6)
    l00 = Lp[:, 0].clamp_min(eps)
    l10 = Lp[:, 1]
    l11 = Lp[:, 2].clamp_min(eps)
    l20 = Lp[:, 3]
    l21 = Lp[:, 4]
    l22 = Lp[:, 5].clamp_min(eps)

    Lmat = torch.zeros((Lp.shape[0], d, d), device=Lp.device, dtype=Lp.dtype)
    Lmat[:, 0, 0] = l00
    Lmat[:, 1, 0] = l10
    Lmat[:, 1, 1] = l11
    Lmat[:, 2, 0] = l20
    Lmat[:, 2, 1] = l21
    Lmat[:, 2, 2] = l22

    sol = torch.linalg.solve_triangular(Lmat, diff, upper=False)
    delta_psi = (sol.squeeze(-1) ** 2).sum(dim=-1)

    kappa_flat = kappa.permute(0, 2, 3, 1).reshape(-1).clamp_min(eps)
    nu_flat = nu.permute(0, 2, 3, 1).reshape(-1).clamp_min(d + 1 + eps)
    nu_pred = (nu_flat - (d - 1)).clamp_min(eps)
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
    nll = -log_prob
    nll_map = nll.view(B, H, W)

    if lambda_evi > 0:
        reg = torch.sqrt(delta_psi + eps) * (kappa_flat + nu_flat)
        reg_map = reg.view(B, H, W)
        loss_map = nll_map + lambda_evi * reg_map
    else:
        reg_map = None
        loss_map = nll_map

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError(f"mask must have shape [B,H,W] or [B,1,H,W], got {mask.shape}")
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        loss_map = loss_map * mask
        denom = mask.sum().clamp_min(1.0)
        nll_mean = (nll_map * mask).sum() / denom
        if reg_map is not None:
            reg_mean = (reg_map * mask).sum() / denom
        else:
            reg_mean = torch.zeros((), dtype=nll_map.dtype, device=nll_map.device)
    else:
        denom = torch.tensor(loss_map.numel(), dtype=loss_map.dtype, device=loss_map.device)
        nll_mean = nll_map.mean()
        if reg_map is not None:
            reg_mean = reg_map.mean()
        else:
            reg_mean = torch.zeros((), dtype=nll_map.dtype, device=nll_map.device)

    loss = loss_map.sum() / denom
    stats = {
        "nll": float(nll_mean.detach()),
        "reg": float(reg_mean.detach()),
        "kappa_mean": float(kappa_flat.mean().detach()),
        "nu_mean": float(nu_flat.mean().detach()),
    }
    return loss, stats
