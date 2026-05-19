# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa
from dust3r.heads.dpt_head import DPTOutputAdapter_fix
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
from models.dpt_block import Interpolate  # noqa
from dust3r.heads.dpt_head import (
    create_dpt_head,          # pure DPT geomoetry head
    create_dpt_head_uq,       # Option A: geometry-level UQ
    create_dpt_head_uq_feat,  # Option B: feature-level UQ 
)


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    if desc_dim is not None:
        start = 3 + int(conf_mode is not None)
        res['desc'] = reg_desc(fmap[..., start:start + desc_dim], mode=desc_mode)
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start + desc_dim], mode=desc_conf_mode)
        else:
            res['desc_conf'] = res['conf'].clone()
    return res


def _normalize_residual_gated_mode(mode):
    mode = "pixel_shuffle" if mode is None else str(mode).strip().lower()
    if mode not in {"pixel_shuffle", "bilinear"}:
        raise ValueError(f"unknown residual_gated_mode={mode}")
    return mode


def _normalize_gr_gate_mode(mode):
    mode = "learned" if mode is None else str(mode).strip().lower()
    if mode not in {"learned", "fixed_one"}:
        raise ValueError(f"unknown gr_gate_mode={mode}")
    return mode


def _normalize_gr_post_ks(kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _init_zero_conv2d(conv):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def _residual_gated_output_dims(module):
    mode = _normalize_residual_gated_mode(getattr(module, "residual_gated_mode", "pixel_shuffle"))
    gate_channels = int(getattr(module, "gr_gate_channels", 1))
    if mode == "pixel_shuffle":
        return gate_channels * module.patch_size**2, 3 * module.patch_size**2
    return gate_channels, 3


def _init_residual_gated_predictors(module):
    ref_param = None
    old_gate = getattr(module, "gate_mlp", None)
    if old_gate is not None:
        for p in old_gate.parameters():
            ref_param = p
            break

    idim = int(getattr(module, "_gr_idim"))
    hidden_dim_factor = float(getattr(module, "_gr_hidden_dim_factor"))
    hidden_dim = int(hidden_dim_factor * idim)
    gate_out, delta_out = _residual_gated_output_dims(module)

    module.gate_mlp = Mlp(
        in_features=idim,
        hidden_features=hidden_dim,
        out_features=gate_out,
    )
    module.delta_mlp = Mlp(
        in_features=idim,
        hidden_features=hidden_dim,
        out_features=delta_out,
    )
    # Keep stable start: residual branch starts from zero.
    nn.init.zeros_(module.gate_mlp.fc2.weight)
    nn.init.zeros_(module.gate_mlp.fc2.bias)
    nn.init.zeros_(module.delta_mlp.fc2.weight)
    nn.init.zeros_(module.delta_mlp.fc2.bias)
    if ref_param is not None:
        module.gate_mlp.to(device=ref_param.device, dtype=ref_param.dtype)
        module.delta_mlp.to(device=ref_param.device, dtype=ref_param.dtype)


def _init_residual_gated_post_modules(module):
    gate_channels = int(getattr(module, "gr_gate_channels", 1))
    kernel_size = int(getattr(module, "gr_post_ks", 3))
    kernel_size = _normalize_gr_post_ks(kernel_size)
    module.gr_post_ks = kernel_size
    ref_param = getattr(getattr(module, "gate_mlp", None), "fc1", None)
    ref_param = getattr(ref_param, "weight", None)

    if not hasattr(module, "residual_gated") or module.residual_gated is None:
        module.residual_gated = nn.Module()

    gate_post = getattr(module.residual_gated, "gate_post", None)
    need_new_gate = (
        not isinstance(gate_post, nn.Conv2d)
        or gate_post.in_channels != gate_channels
        or gate_post.out_channels != gate_channels
        or gate_post.groups != gate_channels
        or gate_post.kernel_size != (kernel_size, kernel_size)
    )
    if need_new_gate:
        module.residual_gated.gate_post = nn.Conv2d(
            gate_channels,
            gate_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=gate_channels,
            bias=True,
        )
        _init_zero_conv2d(module.residual_gated.gate_post)
        if ref_param is not None:
            module.residual_gated.gate_post.to(device=ref_param.device, dtype=ref_param.dtype)

    delta_post = getattr(module.residual_gated, "delta_post", None)
    need_new_delta = (
        not isinstance(delta_post, nn.Conv2d)
        or delta_post.in_channels != 3
        or delta_post.out_channels != 3
        or delta_post.groups != 3
        or delta_post.kernel_size != (kernel_size, kernel_size)
    )
    if need_new_delta:
        module.residual_gated.delta_post = nn.Conv2d(
            3,
            3,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=3,
            bias=True,
        )
        _init_zero_conv2d(module.residual_gated.delta_post)
        if ref_param is not None:
            module.residual_gated.delta_post.to(device=ref_param.device, dtype=ref_param.dtype)


def _configure_residual_gated_modules(
    module,
    residual_gated_mode=None,
    gr_post_smooth=None,
    gr_post_ks=None,
    gate_channels=None,
    gr_gate_mode=None,
):
    if not getattr(module, "use_residual_gated", False):
        return

    mode = getattr(module, "residual_gated_mode", "pixel_shuffle")
    if residual_gated_mode is not None:
        mode = residual_gated_mode
    module.residual_gated_mode = _normalize_residual_gated_mode(mode)

    if gate_channels is not None:
        module.gr_gate_channels = max(1, int(gate_channels))
    elif not hasattr(module, "gr_gate_channels"):
        module.gr_gate_channels = 1

    gate_mode = getattr(module, "gr_gate_mode", "learned")
    if gr_gate_mode is not None:
        gate_mode = gr_gate_mode
    module.gr_gate_mode = _normalize_gr_gate_mode(gate_mode)

    if gr_post_ks is not None:
        module.gr_post_ks = _normalize_gr_post_ks(gr_post_ks)
    elif not hasattr(module, "gr_post_ks"):
        module.gr_post_ks = 3

    if gr_post_smooth is not None:
        module.gr_post_smooth = bool(gr_post_smooth)
    elif not hasattr(module, "gr_post_smooth"):
        module.gr_post_smooth = False
    module.use_gr_post_smooth = bool(module.gr_post_smooth)

    if not hasattr(module, "gr_debug"):
        module.gr_debug = False

    if not hasattr(module, "_gr_idim"):
        module._gr_idim = int(module.gate_mlp.fc1.in_features)
    if not hasattr(module, "_gr_hidden_dim_factor"):
        module._gr_hidden_dim_factor = float(module.gate_mlp.fc1.out_features) / float(module._gr_idim)

    gate_out, delta_out = _residual_gated_output_dims(module)
    need_rebuild = (
        not hasattr(module, "gate_mlp")
        or not hasattr(module, "delta_mlp")
        or module.gate_mlp.fc2.out_features != gate_out
        or module.delta_mlp.fc2.out_features != delta_out
    )
    if need_rebuild:
        _init_residual_gated_predictors(module)
    _init_residual_gated_post_modules(module)


def _init_residual_gated_modules(
    module,
    idim,
    hidden_dim_factor,
    gate_channels=1,
    residual_gated_mode="pixel_shuffle",
    gr_gate_mode="learned",
    gr_post_smooth=False,
    gr_post_ks=3,
):
    module.use_residual_gated = True
    module._gr_idim = int(idim)
    module._gr_hidden_dim_factor = float(hidden_dim_factor)
    module.gr_gate_channels = max(1, int(gate_channels))
    module.residual_gated_mode = _normalize_residual_gated_mode(residual_gated_mode)
    module.gr_gate_mode = _normalize_gr_gate_mode(gr_gate_mode)
    module.gr_post_smooth = bool(gr_post_smooth)
    module.use_gr_post_smooth = bool(gr_post_smooth)
    module.gr_post_ks = _normalize_gr_post_ks(gr_post_ks)
    module.gr_debug = bool(getattr(module, "gr_debug", False))

    _init_residual_gated_predictors(module)
    _init_residual_gated_post_modules(module)


def _apply_residual_gated(module, cat_output, img_shape, out):
    if not getattr(module, "use_residual_gated", False):
        return out
    if not isinstance(out, dict):
        out = {"raw": out}
    pts3d = out.get("pts3d")
    if pts3d is None:
        return out
    pts3d_is_bhwc = False
    if pts3d.ndim == 4 and pts3d.shape[-1] == 3:
        pts3d_is_bhwc = True
        pts3d_ch = pts3d.permute(0, 3, 1, 2).contiguous()
    elif pts3d.ndim == 4 and pts3d.shape[1] == 3:
        pts3d_ch = pts3d
    else:
        raise ValueError(f"unexpected pts3d shape for residual gating: {pts3d.shape}")

    H, W = img_shape
    B, S, _ = cat_output.shape
    patch_size = module.patch_size
    h, w = H // patch_size, W // patch_size
    gate_channels = int(getattr(module, "gr_gate_channels", 1))
    residual_mode = _normalize_residual_gated_mode(getattr(module, "residual_gated_mode", "pixel_shuffle"))
    gate_mode = _normalize_gr_gate_mode(getattr(module, "gr_gate_mode", "learned"))

    def _tokens_to_grid(tokens, channels, name):
        if tokens.shape[1] != h * w:
            raise ValueError(f"unexpected token count for {name}: {tokens.shape[1]} vs {h*w}")
        return tokens.transpose(1, 2).reshape(B, channels, h, w).contiguous()

    if residual_mode == "pixel_shuffle":
        delta_tokens = module.delta_mlp(cat_output)
        delta_coarse = delta_tokens.transpose(-1, -2).view(B, -1, h, w)
        delta = F.pixel_shuffle(delta_coarse, patch_size)
        if gate_mode == "learned":
            gate_tokens = module.gate_mlp(cat_output)
            gate_coarse = gate_tokens.transpose(-1, -2).view(B, -1, h, w)
            gate_logits = F.pixel_shuffle(gate_coarse, patch_size)
        else:
            gate_logits = delta.new_full((B, gate_channels, H, W), 20.0)
    else:
        # Bilinear mode avoids sub-pixel rearrangement artifacts from pixel_shuffle.
        delta_tokens = module.delta_mlp(cat_output)
        delta = _tokens_to_grid(delta_tokens, 3, "delta")
        delta = F.interpolate(delta, size=(H, W), mode="bilinear", align_corners=False)
        if gate_mode == "learned":
            gate_tokens = module.gate_mlp(cat_output)
            gate_logits = _tokens_to_grid(gate_tokens, gate_channels, "gate")
            gate_logits = F.interpolate(gate_logits, size=(H, W), mode="bilinear", align_corners=False)
        else:
            gate_logits = delta.new_full((B, gate_channels, H, W), 20.0)

    if bool(getattr(module, "gr_post_smooth", getattr(module, "use_gr_post_smooth", False))):
        if gate_mode == "learned":
            gate_post = getattr(getattr(module, "residual_gated", None), "gate_post", None)
            if gate_post is not None:
                gate_logits = gate_logits + gate_post(gate_logits)
        delta_post = getattr(getattr(module, "residual_gated", None), "delta_post", None)
        if delta_post is not None:
            delta = delta + delta_post(delta)

    if gate_mode == "fixed_one":
        gate = torch.ones_like(gate_logits)
    else:
        gate = torch.sigmoid(gate_logits)
    gate_for_xyz = gate
    if gate_for_xyz.shape[1] not in (1, 3):
        gate_for_xyz = gate_for_xyz.mean(dim=1, keepdim=True)

    pts3d_refined_ch = pts3d_ch + gate_for_xyz * delta
    if pts3d_is_bhwc:
        pts3d_refined = pts3d_refined_ch.permute(0, 2, 3, 1).contiguous()
    else:
        pts3d_refined = pts3d_refined_ch
    out["pts3d_raw"] = pts3d
    out["pts3d_refined"] = pts3d_refined
    out["pts3d"] = pts3d_refined
    # Keep both new (rg_*) and old (gr_*) keys for backward compatibility.
    out["rg_gate"] = gate
    out["rg_gate_logits"] = gate_logits
    out["rg_delta"] = delta
    out["gr_gate"] = gate
    out["gr_gate_logits"] = gate_logits
    out["gr_delta"] = delta
    return out


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None,
                 head_type="regression", residual_gated=False, residual_gated_mode="pixel_shuffle",
                 gr_gate_mode="learned",
                 gr_post_smooth=False, gr_post_ks=3, gr_gate_channels=1, gr_debug=False, **kwargs):
        mc_dropout_p = kwargs.pop("mc_dropout_p", 0.1)
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.mc_dropout_enabled = False
        self.mc_dropout_p = float(mc_dropout_p)
        self._mc_dropout_hook_handle = self.dpt.register_forward_hook(self._mc_dropout_hook)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)
        self.use_residual_gated = False
        self.gr_debug = bool(gr_debug)
        if residual_gated:
            _init_residual_gated_modules(
                self,
                idim=idim,
                hidden_dim_factor=hidden_dim_factor,
                gate_channels=gr_gate_channels,
                residual_gated_mode=residual_gated_mode,
                gr_gate_mode=gr_gate_mode,
                gr_post_smooth=gr_post_smooth,
                gr_post_ks=gr_post_ks,
            )

    def configure_residual_gated(
        self,
        residual_gated_mode=None,
        gr_post_smooth=None,
        gr_post_ks=None,
        gr_gate_channels=None,
        gr_gate_mode=None,
    ):
        _configure_residual_gated_modules(
            self,
            residual_gated_mode=residual_gated_mode,
            gr_post_smooth=gr_post_smooth,
            gr_post_ks=gr_post_ks,
            gate_channels=gr_gate_channels,
            gr_gate_mode=gr_gate_mode,
        )

    def set_mc_dropout(self, enabled: bool, p=None):
        self.mc_dropout_enabled = bool(enabled)
        if p is not None:
            self.mc_dropout_p = float(p)

    def _apply_mc_dropout(self, x):
        if self.mc_dropout_enabled:
            return F.dropout(x, p=self.mc_dropout_p, training=True)
        return x

    def _mc_dropout_hook(self, module, inputs, output):
        return self._apply_mc_dropout(output)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        out = _apply_residual_gated(self, cat_output, img_shape, out)
        return out


class MLP_MiniConv_Head(nn.Module):
    """
    A special Convolutional head inspired by DPT architecture
    A MLP predicts pixelwise feats in lower resolution. Prediction is upsampled to target res and goes through a mini convolutional head

    Input : [B, S, D]  # S = (H//p) * (W//p)

    MLP: 
        D -> (mlp_hidden_dim) -> out_mlp_dim * (p/2)*2 
        reshape to [out_mlp_dim, H/2, W/2] (MLP predicts in half-res)

    MiniConv head from DPT: 
        Upsample x2 -> [out_mlp_dim,H,W]
        Conv 3x3 -> [conv_inner_dim,H,W]
        ReLU
        Conv 1x1 -> [odim,H,W]

    """

    def __init__(self, idim, mlp_hidden_dim, mlp_odim, conv_inner_dim, odim, patch_size, subpatch=2, **kw):
        super().__init__()
        self.patch_size = patch_size
        self.subpatch = subpatch
        self.sub_patch_size = patch_size // subpatch
        self.mlp = Mlp(idim, mlp_hidden_dim, mlp_odim * self.sub_patch_size**2, **kw)  # D -> mlp_odim*sub_patch_size**2

        # DPT conv head
        self.head = nn.Sequential(Interpolate(scale_factor=self.subpatch, mode="bilinear", align_corners=True) if self.subpatch != 1 else nn.Identity(),
                                  nn.Conv2d(mlp_odim, conv_inner_dim, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(conv_inner_dim, odim, kernel_size=1, stride=1, padding=0)
                                  )

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        # extract features
        feat = self.mlp(tokens)  # [B, S, mlp_odim*sub_patch_size**2]
        feat = feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.sub_patch_size)  # B,mlp_odim,H/sub,W/sub

        return self.head(feat)  # B, odim, H, W

class Cat_MLP_LocalFeatures_DPT_Pts3d_XYZEviMiniConv(Cat_MLP_LocalFeatures_DPT_Pts3d):
    """
    Hybrid head:
      - pts3d (+conf) comes from heavy DPT refine head (pretrained-compatible)
      - local desc head stays as the original MLP (pretrained-compatible)
      - add a parallel lightweight evidential (NIG-like) head via MLP+MiniConv to predict (nu, alpha, beta)
    """

    def __init__(
        self,
        net,
        has_conf=False,
        local_feat_dim=16,
        hidden_dim_factor=4.0,
        mlp_odim=24,
        conv_inner_dim=100,
        subpatch=2,
        residual_gated=False,
        **kwargs,
    ):
        super().__init__(
            net,
            has_conf=has_conf,
            local_feat_dim=local_feat_dim,
            hidden_dim_factor=hidden_dim_factor,
            residual_gated=residual_gated,
            **kwargs,
        )
        self.eps = 1e-6
        idim = net.enc_embed_dim + net.dec_embed_dim

        # New UQ evidential head: output 9 channels = (log_nu, log_alpha, log_beta) each 3-ch for XYZ
        self.head_xyz_evi = MLP_MiniConv_Head(
            idim=idim,
            mlp_hidden_dim=int(hidden_dim_factor * idim),
            mlp_odim=mlp_odim,
            conv_inner_dim=conv_inner_dim,
            odim=9,
            subpatch=subpatch,
            patch_size=self.patch_size,
        )

        # Whether to detach gamma(mean) from the geometry branch
        self.detach_xyz_gamma = True

        # Init to neutral / stable prior
        for m in self.head_xyz_evi.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(self.head_xyz_evi.head[-1], nn.Conv2d):
            nn.init.zeros_(self.head_xyz_evi.head[-1].weight)
            nn.init.zeros_(self.head_xyz_evi.head[-1].bias)

    def forward(self, decout, img_shape):
        # 1) heavy geometry prediction (pretrained)
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))  # B, (3+conf), H, W

        # 2) concat encoder+decoder tokens
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # B, S, D

        # 3) local features (pretrained)
        H, W = img_shape
        B, S, D = cat_output.shape
        local_features = self.head_local_features(cat_output)  # B,S,?
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B, desc_dim(+desc_conf), H, W

        # 4) postprocess for original outputs
        out_raw = torch.cat([pts3d, local_features], dim=1)
        out = out_raw
        if self.postprocess:
            out = self.postprocess(
                out,
                depth_mode=self.depth_mode,
                conf_mode=self.conf_mode,
                desc_dim=self.local_feat_dim,
                desc_mode=self.desc_mode,
                two_confs=self.two_confs,
                desc_conf_mode=self.desc_conf_mode,
            )
        out = _apply_residual_gated(self, cat_output, img_shape, out)

        # 5) evidential UQ branch (new)
        xyz_evi_raw = self.head_xyz_evi([cat_output], img_shape)  # B,9,H,W
        log_nu, log_alpha, log_beta = torch.chunk(xyz_evi_raw, 3, dim=1)
        nu = F.softplus(log_nu) + self.eps
        alpha = F.softplus(log_alpha) + 1.0 + self.eps
        beta = F.softplus(log_beta) + self.eps

        # gamma/mean from (possibly refined) pts3d
        pts3d_for_gamma = out["pts3d"] if isinstance(out, dict) and "pts3d" in out else pts3d
        if pts3d_for_gamma.ndim == 4 and pts3d_for_gamma.shape[-1] == 3:
            pts3d_for_gamma = pts3d_for_gamma.permute(0, 3, 1, 2).contiguous()
        elif pts3d_for_gamma.ndim == 4 and pts3d_for_gamma.shape[1] == 3:
            pts3d_for_gamma = pts3d_for_gamma
        else:
            raise ValueError(f"unexpected pts3d shape for gamma: {pts3d_for_gamma.shape}")
        gamma = pts3d_for_gamma.detach() if self.detach_xyz_gamma else pts3d_for_gamma
        xyz_mu = gamma
        xyz_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + self.eps)

        if not isinstance(out, dict):
            out = {"raw": out}
        out.update(
            {
                "xyz_gamma": gamma,
                "xyz_nu": nu,
                "xyz_alpha": alpha,
                "xyz_beta": beta,
                "xyz_mu": xyz_mu,
                "xyz_var": xyz_var,
            }
        )
        return out

class Cat_MLP_LocalFeatures_DPT_Pts3d_XYZEviDPT(Cat_MLP_LocalFeatures_DPT_Pts3d):
    """
    Hybrid head:
      - pts3d (+conf) comes from heavy DPT refine head (pretrained-compatible)
      - local desc head stays as the original MLP (pretrained-compatible)
      - add a parallel heavy evidential head with SAME DPT architecture as pts3d head
        output 9 channels = (log_nu, log_alpha, log_beta) each 3-ch for XYZ
    """

    def __init__(
        self,
        net,
        has_conf=False,
        local_feat_dim=16,
        hidden_dim_factor=4.0,
        residual_gated=False,
        **kwargs,
    ):
        super().__init__(
            net,
            has_conf=has_conf,
            local_feat_dim=local_feat_dim,
            hidden_dim_factor=hidden_dim_factor,
            residual_gated=residual_gated,
            **kwargs,
        )
        self.eps = 1e-6
        self.detach_xyz_gamma = True

        # Match create_dpt_head(net) hyperparams exactly
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        # SAME DPT architecture as geometry head, only num_channels differs
        self.dpt_xyz_evi = DPTOutputAdapter_fix(
            output_width_ratio=1,
            num_channels=9,          # 9 = 3*(log_nu,log_alpha,log_beta)
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            head_type="regression",
        )
        self.dpt_xyz_evi.init(dim_tokens_enc=[ed, dd, dd, dd])

        # Optional: stabilize init (start near zero outputs)
        if hasattr(self.dpt_xyz_evi, "head") and isinstance(self.dpt_xyz_evi.head, nn.Conv2d):
            nn.init.zeros_(self.dpt_xyz_evi.head.weight)
            nn.init.zeros_(self.dpt_xyz_evi.head.bias)

    def forward(self, decout, img_shape):
        H, W = img_shape

        # 1) heavy geometry prediction (pretrained)
        pts3d = self.dpt(decout, image_size=(H, W))  # B, (3+conf), H, W

        # 2) concat encoder+decoder tokens
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # B, S, D

        # 3) local features (pretrained)
        B, S, D = cat_output.shape
        local_features = self.head_local_features(cat_output)  # B,S,?
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B, desc_dim(+desc_conf), H, W

        # 4) postprocess for original outputs
        out_raw = torch.cat([pts3d, local_features], dim=1)
        out = out_raw
        if self.postprocess:
            out = self.postprocess(
                out,
                depth_mode=self.depth_mode,
                conf_mode=self.conf_mode,
                desc_dim=self.local_feat_dim,
                desc_mode=self.desc_mode,
                two_confs=self.two_confs,
                desc_conf_mode=self.desc_conf_mode,
            )
        out = _apply_residual_gated(self, cat_output, img_shape, out)

        # 5) evidential UQ branch (SAME DPT architecture)
        xyz_evi_raw = self.dpt_xyz_evi(decout, image_size=(H, W))  # B,9,H,W
        log_nu, log_alpha, log_beta = torch.chunk(xyz_evi_raw, 3, dim=1)

        nu = F.softplus(log_nu) + self.eps
        alpha = F.softplus(log_alpha) + 1.0 + self.eps
        beta = F.softplus(log_beta) + self.eps

        # gamma/mean from (possibly refined) pts3d
        pts3d_for_gamma = out["pts3d"] if isinstance(out, dict) and "pts3d" in out else pts3d
        if pts3d_for_gamma.ndim == 4 and pts3d_for_gamma.shape[-1] == 3:
            pts3d_for_gamma = pts3d_for_gamma.permute(0, 3, 1, 2).contiguous()
        elif pts3d_for_gamma.ndim == 4 and pts3d_for_gamma.shape[1] == 3:
            pts3d_for_gamma = pts3d_for_gamma
        elif pts3d_for_gamma.ndim == 4 and pts3d_for_gamma.shape[1] > 3:
            pts3d_for_gamma = pts3d_for_gamma[:, :3]
        else:
            raise ValueError(f"unexpected pts3d shape for gamma: {pts3d_for_gamma.shape}")
        gamma = pts3d_for_gamma.detach() if self.detach_xyz_gamma else pts3d_for_gamma
        xyz_mu = gamma
        xyz_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + self.eps)

        if not isinstance(out, dict):
            out = {"raw": out}
        out.update(
            {
                "xyz_gamma": gamma,
                "xyz_nu": nu,
                "xyz_alpha": alpha,
                "xyz_beta": beta,
                "xyz_mu": xyz_mu,
                "xyz_var": xyz_var,
            }
        )
        return out


class Cat_MLP_LocalFeatures_DPT_Pts3d_XYZNIWDPT(Cat_MLP_LocalFeatures_DPT_Pts3d):
    """
    Hybrid head:
      - pts3d (+conf) comes from heavy DPT refine head (pretrained-compatible)
      - local desc head stays as the original MLP (pretrained-compatible)
      - add a parallel heavy NIW head with SAME DPT architecture as pts3d head
        output 8 channels = (kappa, nu, L_params) where L_params are 6 lower-tri entries
    """

    def __init__(
        self,
        net,
        has_conf=False,
        local_feat_dim=16,
        hidden_dim_factor=4.0,
        residual_gated=False,
        **kwargs,
    ):
        super().__init__(
            net,
            has_conf=has_conf,
            local_feat_dim=local_feat_dim,
            hidden_dim_factor=hidden_dim_factor,
            residual_gated=residual_gated,
            **kwargs,
        )
        self.eps = 1e-6
        self.detach_xyz_niw_m = True

        # Match create_dpt_head(net) hyperparams exactly
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        # SAME DPT architecture as geometry head, only num_channels differs
        self.dpt_xyz_niw = DPTOutputAdapter_fix(
            output_width_ratio=1,
            num_channels=8,          # 1(kappa) + 1(nu) + 6(L params)
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            head_type="regression",
        )
        self.dpt_xyz_niw.init(dim_tokens_enc=[ed, dd, dd, dd])

        # Optional: stabilize init (start near zero outputs)
        if hasattr(self.dpt_xyz_niw, "head") and isinstance(self.dpt_xyz_niw.head, nn.Conv2d):
            nn.init.zeros_(self.dpt_xyz_niw.head.weight)
            nn.init.zeros_(self.dpt_xyz_niw.head.bias)

    def forward(self, decout, img_shape):
        H, W = img_shape

        # 1) heavy geometry prediction (pretrained)
        pts3d = self.dpt(decout, image_size=(H, W))  # B, (3+conf), H, W

        # 2) concat encoder+decoder tokens
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # B, S, D

        # 3) local features (pretrained)
        B, S, D = cat_output.shape
        local_features = self.head_local_features(cat_output)  # B,S,?
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B, desc_dim(+desc_conf), H, W

        # 4) postprocess for original outputs
        out_raw = torch.cat([pts3d, local_features], dim=1)
        out = out_raw
        if self.postprocess:
            out = self.postprocess(
                out,
                depth_mode=self.depth_mode,
                conf_mode=self.conf_mode,
                desc_dim=self.local_feat_dim,
                desc_mode=self.desc_mode,
                two_confs=self.two_confs,
                desc_conf_mode=self.desc_conf_mode,
            )
        out = _apply_residual_gated(self, cat_output, img_shape, out)

        # 5) NIW UQ branch (SAME DPT architecture)
        xyz_niw_raw = self.dpt_xyz_niw(decout, image_size=(H, W))  # B,8,H,W
        raw_kappa = xyz_niw_raw[:, 0:1]
        raw_nu = xyz_niw_raw[:, 1:2]
        raw_L = xyz_niw_raw[:, 2:8]

        kappa = F.softplus(raw_kappa) + self.eps
        nu = 4.0 + F.softplus(raw_nu) + self.eps

        l00_raw, l10_raw, l11_raw, l20_raw, l21_raw, l22_raw = torch.chunk(raw_L, 6, dim=1)
        l00 = F.softplus(l00_raw) + self.eps
        l11 = F.softplus(l11_raw) + self.eps
        l22 = F.softplus(l22_raw) + self.eps
        l10 = l10_raw
        l20 = l20_raw
        l21 = l21_raw
        xyz_niw_Lp = torch.cat([l00, l10, l11, l20, l21, l22], dim=1)
        B, _, H_L, W_L = l00.shape
        xyz_niw_Lmat = torch.zeros((B, 3, 3, H_L, W_L), device=l00.device, dtype=l00.dtype)
        xyz_niw_Lmat[:, 0, 0] = l00[:, 0]
        xyz_niw_Lmat[:, 1, 0] = l10[:, 0]
        xyz_niw_Lmat[:, 1, 1] = l11[:, 0]
        xyz_niw_Lmat[:, 2, 0] = l20[:, 0]
        xyz_niw_Lmat[:, 2, 1] = l21[:, 0]
        xyz_niw_Lmat[:, 2, 2] = l22[:, 0]

        pts3d_for_m = (out.get("pts3d_refined", None) if isinstance(out, dict) else None)
        if pts3d_for_m is None:
            pts3d_for_m = (out.get("pts3d", None) if isinstance(out, dict) else None)
        if pts3d_for_m is None:
            pts3d_for_m = pts3d
        if pts3d_for_m.ndim == 4 and pts3d_for_m.shape[-1] == 3:
            pts3d_for_m = pts3d_for_m.permute(0, 3, 1, 2).contiguous()
        elif pts3d_for_m.ndim == 4 and pts3d_for_m.shape[1] == 3:
            pts3d_for_m = pts3d_for_m
        else:
            raise ValueError(f"unexpected pts3d shape for NIW mean: {pts3d_for_m.shape}")
        m = pts3d_for_m.detach() if self.detach_xyz_niw_m else pts3d_for_m
        nu_pred = (nu - 2.0).clamp_min(self.eps)
        scale = (kappa + 1.0) / (kappa * nu_pred)
        psi_00 = l00 ** 2 + self.eps
        psi_11 = l10 ** 2 + l11 ** 2 + self.eps
        psi_22 = l20 ** 2 + l21 ** 2 + l22 ** 2 + self.eps
        xyz_niw_psi_diag = torch.cat([psi_00, psi_11, psi_22], dim=1)
        xyz_niw_var_diag = xyz_niw_psi_diag * scale

        if not isinstance(out, dict):
            out = {"raw": out}
        out.update(
            {
                "xyz_niw_m": m,
                "xyz_niw_kappa": kappa,
                "xyz_niw_nu": nu,
                "xyz_niw_L": xyz_niw_Lp,
                "xyz_niw_Lp": xyz_niw_Lp,
                "xyz_niw_Lmat": xyz_niw_Lmat,
                "xyz_niw_psi_diag": xyz_niw_psi_diag,
                "xyz_niw_scale": scale,
                "xyz_niw_var_diag": xyz_niw_var_diag,
                "xyz_mu": m,
                "xyz_var": xyz_niw_var_diag,
            }
        )
        return out


class Cat_MLP_LocalFeatures_DPT_Pts3d_XYZHeteroDPT(Cat_MLP_LocalFeatures_DPT_Pts3d):
    """
    Hybrid head:
      - pts3d (+conf) comes from heavy DPT refine head (pretrained-compatible)
      - local desc head stays as the original MLP (pretrained-compatible)
      - add a parallel DPT head to predict per-pixel log-variance for xyz
    """

    def __init__(
        self,
        net,
        has_conf=False,
        local_feat_dim=16,
        hidden_dim_factor=4.0,
        hooks_idx=None,
        dim_tokens=None,
        num_channels=1,
        postprocess=None,
        feature_dim=256,
        last_dim=32,
        depth_mode=None,
        conf_mode=None,
        head_type="regression",
        residual_gated=False,
        **kwargs,
    ):
        super().__init__(
            net,
            has_conf=has_conf,
            local_feat_dim=local_feat_dim,
            hidden_dim_factor=hidden_dim_factor,
            hooks_idx=hooks_idx,
            dim_tokens=dim_tokens,
            num_channels=num_channels,
            postprocess=postprocess,
            feature_dim=feature_dim,
            last_dim=last_dim,
            depth_mode=depth_mode,
            conf_mode=conf_mode,
            head_type=head_type,
            residual_gated=residual_gated,
            **kwargs,
        )
        self.eps = 1e-6
        self.dpt_xyz_logvar = PixelwiseTaskWithDPT(
            num_channels=3,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=hooks_idx,
            dim_tokens=dim_tokens,
            postprocess=None,
            head_type=head_type,
            **kwargs,
        )

    def forward(self, decout, img_shape):
        out = super().forward(decout, img_shape)
        if not isinstance(out, dict):
            out = {"raw": out}

        xyz_mu = out["pts3d"]
        if xyz_mu.ndim == 4 and xyz_mu.shape[-1] == 3:
            xyz_mu = xyz_mu.permute(0, 3, 1, 2)

        logvar_raw = self.dpt_xyz_logvar(decout, img_shape)
        xyz_logvar = torch.clamp(logvar_raw, min=-20, max=10)
        xyz_var = torch.exp(xyz_logvar) + self.eps

        out.update(
            {
                "xyz_mu": xyz_mu,
                "xyz_logvar": xyz_logvar,
                "xyz_var": xyz_var,
            }
        )
        return out


class Cat_MLP_LocalFeatures_MiniConv_Pts3d(nn.Module):
    """ Mixture between MLP and MLP-Convolutional head that outputs 3d points (with miniconv) and local features (with MLP).
    simply contains two MLP_MiniConv_Head: one for 3D points and one for features.
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., mlp_odim=24,
                 conv_inner_dim=100, subpatch=2, residual_gated=False, residual_gated_mode="pixel_shuffle",
                 gr_gate_mode="learned",
                 gr_post_smooth=False, gr_post_ks=3, gr_gate_channels=1, gr_debug=False, **kw):
        super().__init__()

        self.local_feat_dim = local_feat_dim
        self.eps = 1e-6
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.desc_mode = net.desc_mode
        self.desc_conf_mode = net.desc_conf_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        idim = net.enc_embed_dim + net.dec_embed_dim
        self.use_residual_gated = False
        self.gr_debug = bool(gr_debug)
        if residual_gated:
            _init_residual_gated_modules(
                self,
                idim=idim,
                hidden_dim_factor=hidden_dim_factor,
                gate_channels=gr_gate_channels,
                residual_gated_mode=residual_gated_mode,
                gr_gate_mode=gr_gate_mode,
                gr_post_smooth=gr_post_smooth,
                gr_post_ks=gr_post_ks,
            )
        self.head_pts3d = MLP_MiniConv_Head(idim=idim,
                                            mlp_hidden_dim=int(hidden_dim_factor * idim),
                                            mlp_odim=mlp_odim + self.has_conf,
                                            conv_inner_dim=conv_inner_dim,
                                            odim=3 + self.has_conf,
                                            subpatch=subpatch,
                                            patch_size=self.patch_size,
                                            **kw)
        self.head_xyz_evi = MLP_MiniConv_Head(
            idim=idim,
            mlp_hidden_dim=int(hidden_dim_factor * idim),
            mlp_odim=mlp_odim,
            conv_inner_dim=conv_inner_dim,
            odim=9,
            subpatch=subpatch,
            patch_size=self.patch_size,
            **kw,
        )
        self.detach_xyz_gamma = True
        # Initialize the evidential head to start near an isotropic, moderately uncertain prior (log_* ~ 0).
        for m in self.head_xyz_evi.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.bias)
        if isinstance(self.head_xyz_evi.head[-1], nn.Conv2d):
            nn.init.zeros_(self.head_xyz_evi.head[-1].weight)
            nn.init.zeros_(self.head_xyz_evi.head[-1].bias)

        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=(self.local_feat_dim + self.two_confs) * self.patch_size**2)

    def configure_residual_gated(
        self,
        residual_gated_mode=None,
        gr_post_smooth=None,
        gr_post_ks=None,
        gr_gate_channels=None,
        gr_gate_mode=None,
    ):
        _configure_residual_gated_modules(
            self,
            residual_gated_mode=residual_gated_mode,
            gr_post_smooth=gr_post_smooth,
            gr_post_ks=gr_post_ks,
            gate_channels=gr_gate_channels,
            gr_gate_mode=gr_gate_mode,
        )

    def forward(self, decout, img_shape):
        enc_output, dec_output = decout[0], decout[-1]  # recover encoder and decoder outputs
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        # pass through the heads
        pts3d = self.head_pts3d([cat_output], img_shape)
        pts3d_xyz = pts3d[:, :3]

        xyz_evi_raw = self.head_xyz_evi([cat_output], img_shape)
        log_nu, log_alpha, log_beta = torch.chunk(xyz_evi_raw, 3, dim=1)
        nu = F.softplus(log_nu) + self.eps
        alpha = F.softplus(log_alpha) + 1.0 + self.eps
        beta = F.softplus(log_beta) + self.eps

        gamma = pts3d_xyz.detach() if self.detach_xyz_gamma else pts3d_xyz
        xyz_mu = gamma
        xyz_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + self.eps)

        H, W = img_shape
        B, S, D = cat_output.shape

        # extract 3D points
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W

        # post process 3D pts, descriptors and confidences
        out = postprocess(torch.cat([pts3d, local_features], dim=1),
                          depth_mode=self.depth_mode,
                          conf_mode=self.conf_mode,
                          desc_dim=self.local_feat_dim,
                          desc_mode=self.desc_mode,
                          two_confs=self.two_confs, desc_conf_mode=self.desc_conf_mode)
        out = _apply_residual_gated(self, cat_output, img_shape, out)
        if not isinstance(out, dict):
            out = {"raw": out}
        out.update({
            "xyz_gamma": gamma,
            "xyz_nu": nu,
            "xyz_alpha": alpha,
            "xyz_beta": beta,
            "xyz_mu": xyz_mu,
            "xyz_var": xyz_var,
        })
        return out


def mast3r_head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    head_type_base = head_type
    kw = {}
    if ":" in head_type:
        head_type_base, kw_str = head_type.split(":", 1)
        kw = eval("dict(" + kw_str + ")")
    residual_gated = False
    if head_type_base.endswith("+residual_gated"):
        residual_gated = True
        head_type_base = head_type_base[: -len("+residual_gated")]

    if head_type_base == "dpt" and output_mode == "pts3d":
       
        return create_dpt_head(net, has_conf=has_conf)

    if head_type_base == "dpt_uq" and output_mode == "pts3d":
        return create_dpt_head_uq(net, has_conf=has_conf)
    
    if head_type_base == "dpt_uq_feat" and output_mode == "pts3d":
        return create_dpt_head_uq_feat(net, has_conf=has_conf)
    
    if head_type_base == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression',
                                               residual_gated=residual_gated,
                                               **kw)
    elif head_type_base == 'catmlp+dpt+xyz_evi':
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        return Cat_MLP_LocalFeatures_DPT_Pts3d_XYZEviMiniConv(
            net,
            local_feat_dim=local_feat_dim,
            has_conf=has_conf,
            num_channels=out_nchan + has_conf,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            dim_tokens=[ed, dd, dd, dd],
            postprocess=postprocess,
            depth_mode=net.depth_mode,
            conf_mode=net.conf_mode,
            head_type='regression',
            residual_gated=residual_gated,
            **kw,
        )
    
    elif head_type_base.startswith('catmlp+dpt+xyz_evi_dpt') and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        return Cat_MLP_LocalFeatures_DPT_Pts3d_XYZEviDPT(
            net,
            local_feat_dim=local_feat_dim,
            has_conf=has_conf,
            num_channels=out_nchan + has_conf,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            dim_tokens=[ed, dd, dd, dd],
            postprocess=postprocess,
            depth_mode=net.depth_mode,
            conf_mode=net.conf_mode,
            head_type='regression',
            residual_gated=residual_gated,
            **kw,
        )

    elif head_type_base.startswith('catmlp+dpt+xyz_niw_dpt') and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        return Cat_MLP_LocalFeatures_DPT_Pts3d_XYZNIWDPT(
            net,
            local_feat_dim=local_feat_dim,
            has_conf=has_conf,
            num_channels=out_nchan + has_conf,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            dim_tokens=[ed, dd, dd, dd],
            postprocess=postprocess,
            depth_mode=net.depth_mode,
            conf_mode=net.conf_mode,
            head_type='regression',
            residual_gated=residual_gated,
            **kw,
        )

    elif head_type_base.startswith('catmlp+dpt+xyz_hetero_dpt') and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim

        return Cat_MLP_LocalFeatures_DPT_Pts3d_XYZHeteroDPT(
            net,
            local_feat_dim=local_feat_dim,
            has_conf=has_conf,
            num_channels=out_nchan + has_conf,
            feature_dim=feature_dim,
            last_dim=last_dim,
            hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
            dim_tokens=[ed, dd, dd, dd],
            postprocess=postprocess,
            depth_mode=net.depth_mode,
            conf_mode=net.conf_mode,
            head_type='regression',
            residual_gated=residual_gated,
            **kw,
        )

    elif head_type_base == 'catconv' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        return Cat_MLP_LocalFeatures_MiniConv_Pts3d(
            net,
            local_feat_dim=local_feat_dim,
            has_conf=has_conf,
            residual_gated=residual_gated,
            **kw,
        )
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
