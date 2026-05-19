# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import postprocess, reg_dense_conf, reg_dense_depth
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter  # noqa


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None, return_feature: bool = False):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        if return_feature:
            return out, path_1
        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


class PixelwiseTaskWithDPTUQ(nn.Module):
    """DPT module that outputs:
       - pts3d (+ optional conf) via the original DPT head (frozen),
       - evidential depth uncertainty via a separate conv head (trainable).
    """

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, has_conf=False, postprocess=None,
                 depth_mode=None, conf_mode=None, eps=1e-6, **kwargs):
        super().__init__()

        assert n_cls_token == 0, "Not implemented"
        self.return_all_layers = True
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.has_conf = has_conf
        self.eps = eps

        # Geometry head (frozen)
        out_nchan = 3
        num_channels_geom = out_nchan + int(has_conf)
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels_geom,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)
        for p in self.dpt.parameters():
            p.requires_grad = False

        # UQ head
        uq_in_ch = 3 + int(has_conf)
        uq_hidden = 128
        self.depth_evi_head = nn.Sequential(
            nn.Conv2d(uq_in_ch, uq_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(uq_hidden, 4, kernel_size=1),  # gamma, log_nu, log_alpha, log_beta
        )

    def forward(self, x, img_info):
        geom_out = self.dpt(x, image_size=(img_info[0], img_info[1]))

        if self.postprocess is not None:
            geom_dict = self.postprocess(geom_out, self.depth_mode, self.conf_mode)
        else:
            geom_dict = {"pts3d": geom_out[:, :3, :, :]}
            if self.has_conf:
                geom_dict["conf"] = geom_out[:, 3:4, :, :]

        pts3d = geom_dict["pts3d"]
        # Evidential head expects channel-first tensors; handle both channel-last and channel-first pts/conf.
        pts3d_ch = pts3d if pts3d.shape[1] == 3 else pts3d.permute(0, 3, 1, 2)
        conf_ch = None
        if "conf" in geom_dict:
            conf = geom_dict["conf"]
            if conf.ndim == 3:
                conf_ch = conf.unsqueeze(1)
            elif conf.ndim == 4 and conf.shape[1] == 1:
                conf_ch = conf
            elif conf.ndim == 4 and conf.shape[-1] == 1:
                conf_ch = conf.permute(0, 3, 1, 2)
            else:
                conf_ch = conf.unsqueeze(1)
        uq_input = torch.cat([pts3d_ch, conf_ch], dim=1) if conf_ch is not None else pts3d_ch

        nig_raw = self.depth_evi_head(uq_input)
        gamma, log_nu, log_alpha, log_beta = torch.chunk(nig_raw, 4, dim=1)

        nu = F.softplus(log_nu) + self.eps
        alpha = F.softplus(log_alpha) + 1.0 + self.eps
        beta = F.softplus(log_beta) + self.eps

        depth_mu = gamma
        denom = (alpha - 1.0).clamp_min(self.eps)
        depth_var = beta * (1.0 + nu) / (nu * denom + self.eps)

        geom_dict.update({
            "depth_gamma": gamma,
            "depth_nu": nu,
            "depth_alpha": alpha,
            "depth_beta": beta,
            "depth_mu": depth_mu,
            "depth_var": depth_var,
        })

        return geom_dict

#old 
# class PixelwiseTaskWithDPTUQFeat(nn.Module):
#     """
#     DPT module for feature-level evidential UQ (Option B):
#       - Geometry branch: standard DPT head -> pts3d (+conf) via postprocess.
#       - Feature branch: runs dedicated heads on the last DPT feature map (path_1)
#         to predict a deterministic depth mean and evidential NIG parameters.
#     """

#     def __init__(
#         self,
#         *,
#         n_cls_token=0,
#         hooks_idx=None,
#         dim_tokens=None,
#         output_width_ratio=1,
#         num_channels_geom=None,
#         has_conf=False,
#         feature_dim=256,
#         last_dim=None,
#         postprocess=None,
#         depth_mode=None,
#         conf_mode=None,
#         **kwargs,
#     ):
#         super().__init__()
#         assert n_cls_token == 0, "Not implemented"

#         if last_dim is None:
#             last_dim = feature_dim // 2

#         self.return_all_layers = True
#         self.postprocess = postprocess
#         self.depth_mode = depth_mode
#         self.conf_mode = conf_mode
#         self.has_conf = has_conf
#         self.feature_dim = feature_dim
#         self.last_dim = last_dim
#         self.eps = 1e-6

#         # Geometry DPT adapter (3D + conf)
#         num_channels_geom = 3 + int(has_conf) if num_channels_geom is None else num_channels_geom
#         dpt_args = dict(
#             output_width_ratio=output_width_ratio,
#             num_channels=num_channels_geom,
#             feature_dim=feature_dim,
#             last_dim=last_dim,
#             **kwargs,
#         )
#         if hooks_idx is not None:
#             dpt_args.update(hooks=hooks_idx)
#         self.dpt = DPTOutputAdapter_fix(**dpt_args)
#         dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
#         self.dpt.init(**dpt_init_args)

#         # Deterministic depth-from-feature head
#         self.depth_mean_head = nn.Sequential(
#             nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(feature_dim, 1, kernel_size=1),
#         )

#         # Feature-level UQ head on path_1
#         uq_hidden = feature_dim * 2
#         self.depth_evi_head_feat = nn.Sequential(
#             nn.Conv2d(feature_dim, uq_hidden, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(uq_hidden, 4, kernel_size=1),  # 4: gamma, log_nu, log_alpha, log_beta
#         )

#     def forward(self, x, img_info):
#         H, W = img_info
#         geom_out, feat = self.dpt(x, image_size=(H, W), return_feature=True)

#         if self.postprocess is not None:
#             geom_dict = self.postprocess(geom_out, self.depth_mode, self.conf_mode)
#         else:
#             geom_dict = {"pts3d": geom_out[:, :3, :, :]}
#             if self.has_conf:
#                 geom_dict["conf"] = geom_out[:, 3:4, :, :]

#         depth_mean_feat = self.depth_mean_head(feat)

#         nig_raw = self.depth_evi_head_feat(feat)
#         gamma, log_nu, log_alpha, log_beta = torch.chunk(nig_raw, 4, dim=1)

#         nu = F.softplus(log_nu) + self.eps
#         alpha = F.softplus(log_alpha) + 1.0 + self.eps
#         beta = F.softplus(log_beta) + self.eps

#         # upsample to full resolution to match GT depth maps
#         def _upsample(x):
#             return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

#         depth_mean_feat = _upsample(depth_mean_feat)
#         gamma = _upsample(gamma)
#         nu = _upsample(nu)
#         alpha = _upsample(alpha)
#         beta = _upsample(beta)

#         depth_mu = gamma
#         depth_var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + self.eps)

#         geom_dict.update(
#             {
#                 "depth_mean_feat": depth_mean_feat,
#                 "depth_gamma_feat": gamma,
#                 "depth_nu_feat": nu,
#                 "depth_alpha_feat": alpha,
#                 "depth_beta_feat": beta,
#                 "depth_mu_feat": depth_mu,
#                 "depth_var_feat": depth_var,
#             }
#         )

#         return geom_dict

#new
class PixelwiseTaskWithDPTUQFeat(nn.Module):
    """
    Option B: feature-level evidential UQ.

    - Geometry branch: DPT produces pts3d (+ optional confidence), and a geometric
      evidential UQ head acts as a fixed teacher (depth_mu, depth_var).
    - Feature branch: operates on the final DPT feature map (path_1) to predict
      depth_mean_feat and feature-level NIG parameters.
    """

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels_geom=None,
        has_conf=False,
        feature_dim=256,
        last_dim=None,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()

        assert n_cls_token == 0, "Not implemented"

        if last_dim is None:
            last_dim = feature_dim // 2

        self.return_all_layers = True
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.has_conf = has_conf
        self.feature_dim = feature_dim
        self.last_dim = last_dim
        self.eps = eps

        # DPT adapter for geometry (produces pts3d [+ optional confidence])
        num_channels_geom = 3 + int(has_conf) if num_channels_geom is None else num_channels_geom
        dpt_args = dict(
            output_width_ratio=output_width_ratio,
            num_channels=num_channels_geom,
            feature_dim=feature_dim,
            last_dim=last_dim,
            **kwargs,
        )
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

        # Geometric evidential UQ head (teacher), operates on pts3d(+conf).
        # The name `depth_evi_head` matches the geometry-UQ head so that
        # weights can be reused from a `head_type='dpt_uq'` checkpoint.
        uq_in_ch = 3 + int(has_conf)
        uq_hidden = 128
        self.depth_evi_head = nn.Sequential(
            nn.Conv2d(uq_in_ch, uq_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(uq_hidden, 4, kernel_size=1),  # gamma, log_nu, log_alpha, log_beta
        )

        # Feature branch: deterministic depth mean from path_1 features
        self.depth_mean_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
        )

        # Feature branch: evidential NIG head on path_1 features
        uq_hidden_feat = feature_dim * 2
        self.depth_evi_head_feat = nn.Sequential(
            nn.Conv2d(feature_dim, uq_hidden_feat, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(uq_hidden_feat, 4, kernel_size=1),  # gamma, log_nu, log_alpha, log_beta
        )

    def forward(self, x, img_info):
        H, W = img_info
        # geom_out: DPT output for geometry; feat: final DPT feature map (path_1)
        geom_out, feat = self.dpt(x, image_size=(H, W), return_feature=True)

        # 1) Build base geom_dict (pts3d + optional confidence)
        if self.postprocess is not None:
            geom_dict = self.postprocess(geom_out, self.depth_mode, self.conf_mode)
        else:
            geom_dict = {"pts3d": geom_out[:, :3, :, :]}
            if self.has_conf:
                geom_dict["conf"] = geom_out[:, 3:4, :, :]

        # 2) Geometric UQ teacher: run NIG head on pts3d(+conf)
        pts3d = geom_dict["pts3d"]
        # Ensure channel-first layout [B,3,H,W]
        pts3d_ch = pts3d if pts3d.shape[1] == 3 else pts3d.permute(0, 3, 1, 2)

        conf_ch = None
        if "conf" in geom_dict:
            conf = geom_dict["conf"]
            if conf.ndim == 3:
                conf_ch = conf.unsqueeze(1)
            elif conf.ndim == 4 and conf.shape[1] == 1:
                conf_ch = conf
            elif conf.ndim == 4 and conf.shape[-1] == 1:
                conf_ch = conf.permute(0, 3, 1, 2)
            else:
                conf_ch = conf.unsqueeze(1)

        uq_input = torch.cat([pts3d_ch, conf_ch], dim=1) if conf_ch is not None else pts3d_ch

        nig_raw_geom = self.depth_evi_head(uq_input)
        gamma_geom, log_nu_geom, log_alpha_geom, log_beta_geom = torch.chunk(nig_raw_geom, 4, dim=1)

        nu_geom = F.softplus(log_nu_geom) + self.eps
        alpha_geom = F.softplus(log_alpha_geom) + 1.0 + self.eps
        beta_geom = F.softplus(log_beta_geom) + self.eps

        denom_geom = (alpha_geom - 1.0).clamp_min(self.eps)
        depth_mu_geom = gamma_geom
        depth_var_geom = beta_geom * (1.0 + nu_geom) / (nu_geom * denom_geom + self.eps)

        # Expose geometry teacher outputs (depth_mu / depth_var) for distillation
        geom_dict.update(
            {
                "depth_gamma": gamma_geom,
                "depth_nu": nu_geom,
                "depth_alpha": alpha_geom,
                "depth_beta": beta_geom,
                "depth_mu": depth_mu_geom,
                "depth_var": depth_var_geom,
            }
        )

        # 3) Feature branch: depth mean + NIG on path_1 features
        depth_mean_feat = self.depth_mean_head(feat)

        nig_raw_feat = self.depth_evi_head_feat(feat)
        gamma_feat, log_nu_feat, log_alpha_feat, log_beta_feat = torch.chunk(nig_raw_feat, 4, dim=1)

        nu_feat = F.softplus(log_nu_feat) + self.eps
        alpha_feat = F.softplus(log_alpha_feat) + 1.0 + self.eps
        beta_feat = F.softplus(log_beta_feat) + self.eps

        # Upsample to the target spatial resolution (same as GT depth)
        def _upsample(x):
            return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        depth_mean_feat = _upsample(depth_mean_feat)
        gamma_feat = _upsample(gamma_feat)
        nu_feat = _upsample(nu_feat)
        alpha_feat = _upsample(alpha_feat)
        beta_feat = _upsample(beta_feat)

        depth_mu_feat = gamma_feat
        depth_var_feat = beta_feat * (1.0 + nu_feat) / (nu_feat * (alpha_feat - 1.0) + self.eps)

        geom_dict.update(
            {
                "depth_mean_feat": depth_mean_feat,
                "depth_gamma_feat": gamma_feat,
                "depth_nu_feat": nu_feat,
                "depth_alpha_feat": alpha_feat,
                "depth_beta_feat": beta_feat,
                "depth_mu_feat": depth_mu_feat,
                "depth_var_feat": depth_var_feat,
            }
        )

        return geom_dict

def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')


def create_dpt_head_uq(net, has_conf=False, nig_channels=4):
    """
    Return PixelwiseTaskWithDPTUQ for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPTUQ(has_conf=has_conf,
                                  feature_dim=feature_dim,
                                  last_dim=last_dim,
                                  hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                  dim_tokens=[ed, dd, dd, dd],
                                  postprocess=postprocess,
                                  depth_mode=net.depth_mode,
                                  conf_mode=net.conf_mode,
                                  head_type='regression')


def create_dpt_head_uq_feat(net, has_conf=False):
    """
    Option B: feature-level UQ head. Uses DPT feature map as input for evidential head.
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2  # kept for consistency
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim

    return PixelwiseTaskWithDPTUQFeat(
        has_conf=has_conf,
        num_channels_geom=3 + int(has_conf),
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2*2//4, l2*3//4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type='regression')
# Example selection:
# if cfg.uq_mode == "geom":
#     head = create_dpt_head_uq(net, has_conf=True)      # Option A (geometry-level UQ)
# elif cfg.uq_mode == "feat":
#     head = create_dpt_head_uq_feat(net, has_conf=True) # Option B (feature-level UQ)
# else:
#     head = create_dpt_head(net, has_conf=True)         # No UQ
