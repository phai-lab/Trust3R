# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# training code for DUSt3R
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser('DUST3R training', add_help=False)
    # model and criterion
    parser.add_argument('--model', default="AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')",
                        type=str, help="string containing the model to build")
    parser.add_argument('--pretrained', default=None, help='path of a starting checkpoint')
    parser.add_argument('--train_criterion', default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
                        type=str, help="train criterion")
    parser.add_argument('--test_criterion', default=None, type=str, help="test criterion")

    # dataset
    parser.add_argument('--train_dataset', required=True, type=str, help="training set")
    parser.add_argument('--test_dataset', default='[None]', type=str, help="testing set")

    # training
    parser.add_argument('--seed', default=0, type=int, help="Random seed")
    parser.add_argument('--batch_size', default=64, type=int,
                        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)")
    parser.add_argument('--epochs', default=800, type=int, help="Maximum number of epochs for the scheduler")
    parser.add_argument('--lambda_evi', type=float, default=1e-3,
                        help='Regularizer weight inside evidential NIG loss')
    parser.add_argument('--lambda_uq', type=float, default=1.0,
                        help='Global weight for evidential depth uncertainty loss')
    parser.add_argument(
        "--lambda_uq_xyz",
        type=float,
        default=0.2,
        help="Mixing weight between base 3D regression loss and 3D evidential xyz NIG loss",
    )
    parser.add_argument(
        "--lambda_evi_xyz",
        type=float,
        default=1e-3,
        help="Evidence regularizer weight inside 3D NIG loss",
    )
    parser.add_argument(
        "--lambda_hetero_xyz",
        type=float,
        default=0.0,
        help="Weight for heteroscedastic Gaussian NLL on 3D xyz",
    )
    parser.add_argument(
        "--lambda_uc_corr",
        type=float,
        default=0.0,
        help="Weight for UC correlation matching loss between teacher conf and NIW xyz uncertainty",
    )
    parser.add_argument(
        "--uc_corr_use",
        type=str,
        default="total",
        choices=["total", "epistemic"],
        help="NIW uncertainty scalar for UC correlation loss: predictive total covariance trace or epistemic covariance trace",
    )
    parser.add_argument(
        "--uc_corr_log",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply log1p to NIW uncertainty scalar before UC correlation matching",
    )
    parser.add_argument(
        "--uc_conf_key",
        type=str,
        default="conf",
        help="Primary key to fetch confidence from model predictions (fallback: conf/confidence)",
    )
    parser.add_argument(
        "--freeze_geom_for_uq",
        action="store_true",
        default=False,
        help="If set, freeze backbone + geometry heads and only train the UQ head(s).",
    )
    parser.add_argument(
        "--gr_residual_mode",
        type=str,
        default="pixel_shuffle",
        choices=["pixel_shuffle", "bilinear"],
        help="Residual-gated upsampling mode (pixel_shuffle for backward compatibility, bilinear for seam fix).",
    )
    parser.add_argument(
        "--gr_gate_mode",
        type=str,
        default="learned",
        choices=["learned", "fixed_one"],
        help="Residual-gated gate mode: learn gate logits or fix gate to 1 everywhere.",
    )
    parser.add_argument(
        "--gr_post_smooth",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable residual depthwise post-smoothing on gate/delta maps.",
    )
    parser.add_argument(
        "--gr_post_ks",
        type=int,
        default=3,
        help="Kernel size for GR post-smoothing depthwise convs.",
    )
    parser.add_argument(
        "--gr_train_mode",
        type=str,
        default="full",
        choices=["full", "post_only", "gr_only"],
        help="Train mode for residual-gated branch: full, post_only, or gr_only.",
    )
    parser.add_argument(
        "--gr_post_lr",
        type=float,
        default=1e-5,
        help="Learning rate for GR post-only finetune.",
    )
    parser.add_argument(
        "--gr_post_wd",
        type=float,
        default=0.0,
        help="Weight decay for GR post-only finetune.",
    )
    parser.add_argument(
        "--lambda_gate_tv",
        type=float,
        default=0.0,
        help="TV regularizer weight on GR gate logits.",
    )
    parser.add_argument(
        "--gate_tv_warmup_steps",
        type=int,
        default=2000,
        help="Warmup steps for GR gate TV regularizer.",
    )
    parser.add_argument(
        "--no_detach_xyz_gamma",
        action="store_true",
        default=False,
        help="If set, backpropagate xyz NIG loss into gamma (mean) by not detaching pts3d_xyz.",
    )
    parser.add_argument(
        "--uq_mode",
        type=str,
        default="geom",
        choices=["geom", "feat", "xyz", "hetero_xyz", "both"],
        help="Which uncertainty path to train (geometry-depth UQ, feature UQ, 3D xyz UQ, hetero xyz UQ, or both)",
    )
    parser.add_argument("--lambda_evi_feat", type=float, default=0.1,
                        help="Weight for feature-level evidential loss")
    parser.add_argument("--lambda_mean_gt", type=float, default=1.0,
                        help="Weight for supervising feature mean against GT depth")
    parser.add_argument("--lambda_mean_distill", type=float, default=1.0,
                        help="Weight for distilling feature mean from geometry teacher")
    parser.add_argument("--lambda_var_distill", type=float, default=0.1,
                        help="Weight for distilling feature variance from geometry teacher")
    parser.add_argument("--phase", type=int, default=1,
                        help="1: train depth_mean_feat only; 2: train evidential FeatUQ + distillation")

    parser.add_argument('--weight_decay', type=float, default=0.05, help="weight decay (default: 0.05)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    parser.add_argument('--amp', type=int, default=0,
                        choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    parser.add_argument("--disable_cudnn_benchmark", action='store_true', default=False,
                        help="set cudnn.benchmark = False")
    # others
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--eval_freq', type=int, default=1, help='Test loss evaluation frequency')
    parser.add_argument('--save_freq', default=1, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-last.pth')
    parser.add_argument('--keep_freq', default=20, type=int,
                        help='frequence (number of epochs) to save checkpoint in checkpoint-%d.pth')
    parser.add_argument('--save_steps', default=0, type=int,
                        help='Save checkpoint every N training steps (0 disables step-level saving)')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='frequence (number of iterations) to print infos while training')

    # output dir
    parser.add_argument('--output_dir', default='./output/', type=str, help="path where to save the output")
    return parser


def train(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    world_size = misc.get_world_size()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print('Building train dataset {:s}'.format(args.train_dataset))
    #  dataset and loader
    data_loader_train = build_dataset(args.train_dataset, args.batch_size, args.num_workers, test=False)

    def _parse_dataset_list(dataset_arg):
        datasets = []
        for raw in dataset_arg.split('+'):
            ds = raw.strip()
            if not ds:
                continue
            ds_clean = ds.strip('[]').strip()
            if ds_clean.lower() == 'none':
                continue
            datasets.append(ds)
        return datasets

    test_datasets = _parse_dataset_list(args.test_dataset)
    if test_datasets:
        print('Building test dataset {:s}'.format(args.test_dataset))
        data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                            for dataset in test_datasets}
    else:
        print('No test dataset requested, skipping evaluation loaders.')
        data_loader_test = {}

    # model
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)

    def configure_residual_gated(model_to_update, args_obj):
        mode = getattr(args_obj, "gr_residual_mode", "pixel_shuffle")
        gate_mode = getattr(args_obj, "gr_gate_mode", "learned")
        post_smooth = bool(getattr(args_obj, "gr_post_smooth", 0))
        post_ks = max(1, int(getattr(args_obj, "gr_post_ks", 3)))
        if post_ks % 2 == 0:
            post_ks += 1

        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_update, head_attr, None)
            if head is None:
                continue
            if getattr(head, "use_residual_gated", False):
                if hasattr(head, "configure_residual_gated"):
                    head.configure_residual_gated(
                        residual_gated_mode=mode,
                        gr_gate_mode=gate_mode,
                        gr_post_smooth=post_smooth,
                        gr_post_ks=post_ks,
                    )
                else:
                    head.residual_gated_mode = mode
                    head.gr_gate_mode = gate_mode
                    head.gr_post_smooth = post_smooth
                    head.use_gr_post_smooth = post_smooth
                    head.gr_post_ks = post_ks
                    if not hasattr(head, "gr_gate_channels"):
                        head.gr_gate_channels = 1
                    if post_smooth:
                        residual = getattr(head, "residual_gated", None)
                        if residual is None:
                            residual = torch.nn.Module()
                            head.residual_gated = residual
                        gate_post = getattr(residual, "gate_post", None)
                        if gate_post is None:
                            residual.gate_post = torch.nn.Conv2d(
                                int(head.gr_gate_channels),
                                int(head.gr_gate_channels),
                                kernel_size=post_ks,
                                padding=post_ks // 2,
                                groups=int(head.gr_gate_channels),
                                bias=True,
                            )
                            torch.nn.init.zeros_(residual.gate_post.weight)
                            torch.nn.init.zeros_(residual.gate_post.bias)
                        delta_post = getattr(residual, "delta_post", None)
                        if delta_post is None:
                            residual.delta_post = torch.nn.Conv2d(
                                3,
                                3,
                                kernel_size=post_ks,
                                padding=post_ks // 2,
                                groups=3,
                                bias=True,
                            )
                            torch.nn.init.zeros_(residual.delta_post.weight)
                            torch.nn.init.zeros_(residual.delta_post.bias)

    configure_residual_gated(model, args)

    if getattr(args, "no_detach_xyz_gamma", False):
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model, head_attr, None)
            if head is None:
                continue
            if hasattr(head, "detach_xyz_gamma"):
                head.detach_xyz_gamma = False

    def _freeze_all(model_to_freeze):
        for p in model_to_freeze.parameters():
            p.requires_grad = False

    def _unfreeze_geom_uq(model_to_freeze):
        """Unfreeze old scalar depth UQ head: depth_evi_head."""
        unfrozen = False
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_freeze, head_attr, None)
            if head is None:
                continue
            if hasattr(head, "depth_evi_head"):
                for p in head.depth_evi_head.parameters():
                    p.requires_grad = True
                unfrozen = True
        return unfrozen

    def _unfreeze_feat_uq(model_to_freeze, phase):
        """Unfreeze feature-level UQ heads: depth_mean_head / depth_evi_head_feat."""
        unfrozen = False
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_freeze, head_attr, None)
            if head is None:
                continue
            if hasattr(head, "depth_mean_head"):
                for p in head.depth_mean_head.parameters():
                    p.requires_grad = True
                unfrozen = True
            if phase >= 2 and hasattr(head, "depth_evi_head_feat"):
                for p in head.depth_evi_head_feat.parameters():
                    p.requires_grad = True
                unfrozen = True
        return unfrozen

    XYZ_UQ_ATTRS = ("head_xyz_evi", "dpt_xyz_evi", "head_xyz_niw", "dpt_xyz_niw")

    def _head_uses_learned_gate(head):
        return getattr(head, "gr_gate_mode", "learned") == "learned"

    def _unfreeze_xyz_uq(model_to_freeze):
        """Unfreeze the 3D XYZ UQ head(s): supports both old MiniConv and new DPT UQ heads."""
        unfrozen = False
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_freeze, head_attr, None)
            if head is None:
                continue

            for attr in XYZ_UQ_ATTRS:
                module = getattr(head, attr, None)
                if module is not None:
                    module.requires_grad_(True)
                    unfrozen = True

            if getattr(head, "use_residual_gated", False):
                if _head_uses_learned_gate(head) and hasattr(head, "gate_mlp") and head.gate_mlp is not None:
                    head.gate_mlp.requires_grad_(True)
                    unfrozen = True
                if hasattr(head, "delta_mlp") and head.delta_mlp is not None:
                    head.delta_mlp.requires_grad_(True)
                    unfrozen = True
                residual = getattr(head, "residual_gated", None)
                if residual is not None:
                    if _head_uses_learned_gate(head) and hasattr(residual, "gate_post") and residual.gate_post is not None:
                        residual.gate_post.requires_grad_(True)
                        unfrozen = True
                    if hasattr(residual, "delta_post") and residual.delta_post is not None:
                        residual.delta_post.requires_grad_(True)
                        unfrozen = True

        return unfrozen


    def _unfreeze_gr_post(model_to_freeze):
        unfrozen = False
        for name, p in model_to_freeze.named_parameters():
            if (
                "residual_gated.delta_post" in name
                or (
                    getattr(args, "gr_gate_mode", "learned") == "learned"
                    and "residual_gated.gate_post" in name
                )
            ):
                p.requires_grad = True
                unfrozen = True
        return unfrozen

    def _unfreeze_gr_all(model_to_freeze):
        unfrozen = False
        for name, p in model_to_freeze.named_parameters():
            if (
                ".delta_mlp." in name
                or "residual_gated.delta_post" in name
                or (
                    getattr(args, "gr_gate_mode", "learned") == "learned"
                    and (
                        ".gate_mlp." in name
                        or "residual_gated.gate_post" in name
                    )
                )
            ):
                p.requires_grad = True
                unfrozen = True
        return unfrozen

    def _post_param_name(name):
        if "residual_gated.delta_post" in name:
            return True
        if getattr(args, "gr_gate_mode", "learned") == "learned" and "residual_gated.gate_post" in name:
            return True
        return False

    def _gr_core_param_name(name):
        if ".delta_mlp." in name:
            return True
        if getattr(args, "gr_gate_mode", "learned") == "learned" and ".gate_mlp." in name:
            return True
        return False

    def _split_named_trainable_params(model_to_split):
        base_named = []
        post_named = []
        for name, p in model_to_split.named_parameters():
            if not p.requires_grad:
                continue
            if _post_param_name(name):
                post_named.append((name, p))
            else:
                base_named.append((name, p))
        return base_named, post_named

    def _split_named_gr_params(model_to_split):
        gr_named = []
        for name, p in model_to_split.named_parameters():
            if not p.requires_grad:
                continue
            if _gr_core_param_name(name) or _post_param_name(name):
                gr_named.append((name, p))
        return gr_named


    def _log_trainable(model_to_log, header):
        if misc.get_rank() != 0:
            return
        trainable = [name for name, p in model_to_log.named_parameters() if p.requires_grad]
        print(header)
        for n in trainable:
            print("  ", n)


    def _unfreeze_hetero_xyz(model_to_freeze):
        """Unfreeze heteroscedastic xyz logvar heads."""
        unfrozen = False
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_freeze, head_attr, None)
            if head is None:
                continue
            if hasattr(head, "dpt_xyz_logvar") and head.dpt_xyz_logvar is not None:
                for p in head.dpt_xyz_logvar.parameters():
                    p.requires_grad = True
                unfrozen = True
        return unfrozen


    def _unfreeze_xyz_gamma(model_to_freeze):
        """Unfreeze the xyz gamma producer (head_pts3d) and disable gamma detaching."""
        unfrozen = False
        for head_attr in ("downstream_head1", "downstream_head2"):
            head = getattr(model_to_freeze, head_attr, None)
            if head is None:
                continue
            if hasattr(head, "head_pts3d"):
                for p in head.head_pts3d.parameters():
                    p.requires_grad = True
                unfrozen = True
            if hasattr(head, "detach_xyz_gamma"):
                head.detach_xyz_gamma = False
        return unfrozen

    def freeze_for_uq(model_to_freeze, args):
        """
        Freeze everything except the requested UQ heads.

        uq_mode:
          - "geom": geometry-level UQ (depth_evi_head)
          - "feat": feature-level UQ (depth_mean_head / depth_evi_head_feat)
          - "xyz": 3D xyz UQ head (head_xyz_evi)
          - "hetero_xyz": heteroscedastic xyz logvar head
          - "both": all available UQ heads
        """
        uq_mode = getattr(args, "uq_mode", "geom")

        if uq_mode == "hetero_xyz":
            _freeze_all(model_to_freeze)
            unfrozen = _unfreeze_hetero_xyz(model_to_freeze)
            if not unfrozen:
                raise RuntimeError("No heteroscedastic xyz head found to unfreeze.")
            return

        has_geom_head = any(
            hasattr(getattr(model_to_freeze, h, None), "depth_evi_head")
            for h in ("downstream_head1", "downstream_head2")
        )
        has_feat_head = any(
            hasattr(getattr(model_to_freeze, h, None), "depth_mean_head")
            or hasattr(getattr(model_to_freeze, h, None), "depth_evi_head_feat")
            for h in ("downstream_head1", "downstream_head2")
        )
        has_xyz_head = any(
            any(hasattr(getattr(model_to_freeze, h, None), attr) for attr in XYZ_UQ_ATTRS)
            for h in ("downstream_head1", "downstream_head2")
        )

        if uq_mode == "geom" and not has_geom_head:
            return
        if uq_mode == "feat" and not has_feat_head:
            return
        if uq_mode == "xyz" and not has_xyz_head:
            if misc.get_rank() == 0:
                print("[UQ] WARNING: uq_mode=xyz but no xyz UQ head found; skip freezing")
            return

        _freeze_all(model_to_freeze)
        unfrozen = False

        if uq_mode in ("geom", "both") and has_geom_head:
            unfrozen = _unfreeze_geom_uq(model_to_freeze) or unfrozen
        if uq_mode in ("feat", "both") and has_feat_head:
            unfrozen = _unfreeze_feat_uq(model_to_freeze, getattr(args, "phase", 1)) or unfrozen
        if uq_mode in ("xyz", "both") and has_xyz_head:
            unfrozen = _unfreeze_xyz_uq(model_to_freeze) or unfrozen
            if getattr(args, "no_detach_xyz_gamma", False):
                unfrozen = _unfreeze_xyz_gamma(model_to_freeze) or unfrozen

        if not unfrozen:
            raise RuntimeError("No UQ head found to unfreeze with the requested configuration.")

    print(f'>> Creating train criterion = {args.train_criterion}')
    train_criterion = eval(args.train_criterion).to(device)
    print(f'>> Creating test criterion = {args.test_criterion or args.train_criterion}')
    test_criterion = eval(args.test_criterion or args.train_criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.pretrained and not args.resume:
        print('Loading pretrained: ', args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    gr_train_mode = getattr(args, "gr_train_mode", "full")
    freeze_geom_for_uq = bool(getattr(args, "freeze_geom_for_uq", False))

    if gr_train_mode == "gr_only":
        if misc.get_rank() == 0:
            print("[GR] Train mode: gr_only (freezing all but gate_mlp/delta_mlp/(optional)post)")
        _freeze_all(model_without_ddp)
        unfrozen = _unfreeze_gr_all(model_without_ddp)
        if not unfrozen:
            raise RuntimeError("[GR] gr_only selected but no trainable GR parameters were found.")
        _log_trainable(model_without_ddp, "Trainable params after GR-only setup:")
    else:
        if freeze_geom_for_uq:
            if misc.get_rank() == 0:
                print(f"[UQ] Freezing geometry and backbone; training only UQ heads for uq_mode={args.uq_mode}")
            freeze_for_uq(model_without_ddp, args)
            if misc.get_rank() == 0:
                print("Trainable params after freeze:")
                trainable = [name for name, p in model_without_ddp.named_parameters() if p.requires_grad]
                for n in trainable:
                    print("  ", n)
                xyz_uq_markers = tuple(XYZ_UQ_ATTRS) + (
                    "gate_mlp",
                    "delta_mlp",
                    "residual_gated.gate_post",
                    "residual_gated.delta_post",
                )
                print("Only xyz UQ modules?", all(any(m in n for m in xyz_uq_markers) for n in trainable))

                if os.getenv("UQ_FREEZE_DEBUG", "").lower() not in ("", "0", "false", "no"):
                    total_params = sum(p.numel() for p in model_without_ddp.parameters())
                    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
                    top_prefixes = sorted({n.split(".", 1)[0] for n in trainable})
                    sub_prefixes = sorted(
                        {".".join(n.split(".")[:2]) if "." in n else n for n in trainable}
                    )
                    print(f"[UQ] total params: {total_params}")
                    print(f"[UQ] trainable params: {trainable_params}")
                    print(f"[UQ] trainable prefixes (top-level): {top_prefixes}")
                    print(f"[UQ] trainable prefixes (2-level): {sub_prefixes}")

                def _module_trainable(head_attr, module_attr):
                    head = getattr(model_without_ddp, head_attr, None)
                    if head is None:
                        return None
                    module = getattr(head, module_attr, None)
                    if module is None:
                        return None
                    return any(p.requires_grad for p in module.parameters())

                xyz_evi_flags_old = []
                xyz_evi_flags_dpt = []
                xyz_niw_flags_old = []
                xyz_niw_flags_dpt = []
                for head_attr in ("downstream_head1", "downstream_head2"):
                    res_old = _module_trainable(head_attr, "head_xyz_evi")
                    if res_old is not None:
                        xyz_evi_flags_old.append(res_old)

                    res_dpt = _module_trainable(head_attr, "dpt_xyz_evi")
                    if res_dpt is not None:
                        xyz_evi_flags_dpt.append(res_dpt)

                    res_niw_old = _module_trainable(head_attr, "head_xyz_niw")
                    if res_niw_old is not None:
                        xyz_niw_flags_old.append(res_niw_old)

                    res_niw_dpt = _module_trainable(head_attr, "dpt_xyz_niw")
                    if res_niw_dpt is not None:
                        xyz_niw_flags_dpt.append(res_niw_dpt)

                if xyz_evi_flags_old:
                    assert any(xyz_evi_flags_old), "[UQ] head_xyz_evi parameters are all frozen"
                if xyz_evi_flags_dpt:
                    assert any(xyz_evi_flags_dpt), "[UQ] dpt_xyz_evi parameters are all frozen"
                if xyz_niw_flags_old:
                    assert any(xyz_niw_flags_old), "[UQ] head_xyz_niw parameters are all frozen"
                if xyz_niw_flags_dpt:
                    assert any(xyz_niw_flags_dpt), "[UQ] dpt_xyz_niw parameters are all frozen"

                print(f"[UQ] head_xyz_evi trainable flags: {xyz_evi_flags_old}")
                print(f"[UQ] dpt_xyz_evi trainable flags: {xyz_evi_flags_dpt}")
                print(f"[UQ] head_xyz_niw trainable flags: {xyz_niw_flags_old}")
                print(f"[UQ] dpt_xyz_niw trainable flags: {xyz_niw_flags_dpt}")

                if getattr(args, "no_detach_xyz_gamma", False):
                    pts3d_flags = []
                    detach_flags = []
                    for head_attr in ("downstream_head1", "downstream_head2"):
                        res = _module_trainable(head_attr, "head_pts3d")
                        if res is not None:
                            pts3d_flags.append(res)
                        head = getattr(model_without_ddp, head_attr, None)
                        if head is not None and hasattr(head, "detach_xyz_gamma"):
                            detach_flags.append(getattr(head, "detach_xyz_gamma"))
                    if pts3d_flags:
                        assert any(pts3d_flags), "[UQ] head_pts3d parameters are all frozen but no_detach_xyz_gamma is set"
                    print(f"[UQ] head_pts3d trainable flags: {pts3d_flags}")
                    print(f"[UQ] detach_xyz_gamma flags: {detach_flags}")

    if gr_train_mode == "post_only":
        if freeze_geom_for_uq:
            if misc.get_rank() == 0:
                print("[GR] Train mode: post_only + freeze_geom_for_uq (base UQ params + GR post params)")
            unfrozen = _unfreeze_gr_post(model_without_ddp)
            if not unfrozen and misc.get_rank() == 0:
                print("[GR] WARNING: no GR post-smooth parameters found to unfreeze.")
        else:
            if misc.get_rank() == 0:
                print("[GR] Train mode: post_only (freezing all but GR post-smooth layers)")
            _freeze_all(model_without_ddp)
            unfrozen = _unfreeze_gr_post(model_without_ddp)
            if not unfrozen and misc.get_rank() == 0:
                print("[GR] WARNING: no GR post-smooth parameters found to unfreeze.")
        _log_trainable(model_without_ddp, "Trainable params after GR post-only setup:")
        if not getattr(args, "gr_post_smooth", 0) and misc.get_rank() == 0:
            print("[GR] WARNING: gr_post_smooth=0 disables post layers; post_only may have no effect.")

    if gr_train_mode == "post_only":
        base_named, post_named = _split_named_trainable_params(model_without_ddp)
        if not post_named:
            raise RuntimeError("[GR] post_only selected but no trainable GR post parameters were found.")
        param_groups = []
        if freeze_geom_for_uq and base_named:
            param_groups.append(
                {
                    "params": [p for _, p in base_named],
                    "weight_decay": float(args.weight_decay),
                    "lr": float(args.lr),
                }
            )
        if post_named:
            param_groups.append(
                {
                    "params": [p for _, p in post_named],
                    "weight_decay": float(args.gr_post_wd),
                    "lr": float(args.gr_post_lr),
                }
            )
        if not param_groups:
            raise RuntimeError("[GR] post_only selected but no trainable parameters were found.")
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    elif gr_train_mode == "gr_only":
        gr_named = _split_named_gr_params(model_without_ddp)
        if not gr_named:
            raise RuntimeError("[GR] gr_only selected but no trainable GR parameters were found.")
        optimizer = torch.optim.AdamW(
            [{"params": [p for _, p in gr_named], "weight_decay": float(args.weight_decay), "lr": float(args.lr)}],
            betas=(0.9, 0.95),
        )
    else:
        # following timm: set wd as 0 for bias and norm layers
        param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
        param_groups = [{**pg, 'params': [p for p in pg['params'] if p.requires_grad]} for pg in param_groups]
        param_groups = [pg for pg in param_groups if pg['params']]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(epoch=epoch, **{f'train_{k}': v for k, v in train_stats.items()})
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update({test_name + '_' + k: v for k, v in test_stats[test_name].items()})

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, fname=fname, best_so_far=best_so_far)

    best_so_far = misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                  optimizer=optimizer, loss_scaler=loss_scaler)
    if best_so_far is None:
        best_so_far = float('inf')
    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}
    for epoch in range(args.start_epoch, args.epochs + 1):

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, 'last', best_so_far)

        # Test on multiple datasets
        new_best = False
        if (epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0):
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(model, test_criterion, testset,
                                       device, epoch, log_writer=log_writer, args=args, prefix=test_name)
                test_stats[test_name] = stats

                # Save best of all
                if stats['loss_med'] < best_so_far:
                    best_so_far = stats['loss_med']
                    new_best = True

        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, 'best', best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model, train_criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / 'checkpoint-final.pth'
    to_save = {
        'args': args,
        'model': model_without_ddp if isinstance(model_without_ddp, dict) else model_without_ddp.cpu().state_dict(),
        'epoch': epoch
    }
    if best_so_far is not None:
        to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    misc.save_on_master(to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Sized, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    args,
                    log_writer=None):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter
    steps_per_epoch = len(data_loader)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        global_step = epoch * steps_per_epoch + data_iter_step
        setattr(args, "global_step", global_step)
        setattr(args, "_global_step", global_step)

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret='loss', args=args)
        loss, loss_details = loss_tuple  # criterion returns two values
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)

        # Optional step-level checkpointing
        if args.save_steps and args.save_steps > 0 and (data_iter_step + 1) % accum_iter == 0:
            global_step = epoch * steps_per_epoch + data_iter_step + 1  # 1-based for readability
            if global_step % args.save_steps == 0:
                misc.save_model(
                    args=args,
                    epoch=epoch,
                    model_without_ddp=model.module if hasattr(model, "module") else model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    fname=f"step-{global_step}",
                    best_so_far=None,
                )
        metric_logger.update(loss=loss_value, **loss_details)

        if (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0:
            loss_value_reduce = misc.all_reduce_mean(loss_value)  # MUST BE EXECUTED BY ALL NODES
            if log_writer is None:
                continue
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int(epoch_f * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_lr', lr, epoch_1000x)
            log_writer.add_scalar('train_iter', epoch_1000x, epoch_1000x)
            for name, val in loss_details.items():
                log_writer.add_scalar('train_' + name, val, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                   data_loader: Sized, device: torch.device, epoch: int,
                   args, log_writer=None, prefix='test'):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = 'Test Epoch: [{}]'.format(epoch)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        global_step = epoch * len(data_loader) + data_iter_step
        setattr(args, "global_step", global_step)
        setattr(args, "_global_step", global_step)
        loss_tuple = loss_of_one_batch(batch, model, criterion, device,
                                       symmetrize_batch=True,
                                       use_amp=bool(args.amp), ret='loss', args=args)
        loss_value, loss_details = loss_tuple  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr) for k, meter in metric_logger.meters.items() for tag, attr in aggs}

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)

    return results
