#!/usr/bin/env python3
# Trust3R — minimal pair inference template.
#
# Loads a Trust3R checkpoint, runs the network on one image pair, and dumps
# the raw output tensors (per-pixel pointmaps + Trust3R evidential UQ tensors).
# Intended as a starting point — extend `summarise_outputs` to project the NIW
# (kappa, nu, Psi) tensors into whatever predictive variance you care about.
#
# Example:
#   python infer.py \
#       --checkpoint checkpoints/trust3r_niw_mast3r_224.pth \
#       --img1 examples/a.jpg --img2 examples/b.jpg \
#       --image-size 512 --output-dir infer_out

import argparse
import os

import torch

from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa: F401  (adds vendored dust3r to sys.path)
from dust3r.inference import loss_of_one_batch
from dust3r.utils.device import to_cpu
from dust3r.utils.image import load_images


def parse_args():
    p = argparse.ArgumentParser(description="Trust3R minimal pair inference")
    p.add_argument("--checkpoint", required=True, help="Path to a Trust3R checkpoint (.pth)")
    p.add_argument("--img1", required=True, help="Path to the first image")
    p.add_argument("--img2", required=True, help="Path to the second image")
    p.add_argument("--image-size", type=int, default=512, help="Resize so the long side equals this")
    p.add_argument("--output-dir", default="infer_out", help="Where to write preds.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def summarise_outputs(name, pred):
    print(f"--- {name} ---")
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Trust3R] device  = {args.device}")
    print(f"[Trust3R] loading = {args.checkpoint}")
    model = AsymmetricMASt3R.from_pretrained(args.checkpoint).to(args.device).eval()

    print(f"[Trust3R] images  = {args.img1!r}, {args.img2!r}  (size={args.image_size})")
    imgs = load_images([args.img1, args.img2], size=args.image_size, verbose=False)
    if len(imgs) != 2:
        raise RuntimeError(f"load_images returned {len(imgs)} entries; expected 2")

    view1, view2 = imgs[0], imgs[1]

    with torch.no_grad():
        res = loss_of_one_batch((view1, view2), model, criterion=None, device=args.device)

    pred1 = to_cpu(res["pred1"])
    pred2 = to_cpu(res["pred2"])

    summarise_outputs("pred1", pred1)
    summarise_outputs("pred2", pred2)

    out_path = os.path.join(args.output_dir, "preds.pt")
    torch.save({"pred1": pred1, "pred2": pred2}, out_path)
    print(f"[Trust3R] wrote raw outputs to {out_path}")
    print("[Trust3R] keys to look at:")
    print("    * pts3d / pts3d_in_other_view   -> per-pixel 3D points")
    print("    * conf                          -> MASt3R-style heuristic confidence")
    print("    * xyz_niw_kappa / xyz_niw_nu / xyz_niw_Lmat or xyz_niw_Psi")
    print("                                    -> NIW evidential parameters (Trust3R)")


if __name__ == "__main__":
    main()
