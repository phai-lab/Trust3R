#!/usr/bin/env python3
import os
import os.path as osp
import re
import glob
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def imread_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def read_depth_png(path, depth_divisor=256.0):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise RuntimeError(f"Cannot read depth png: {path}")
    d = d.astype(np.float32) / float(depth_divisor)
    d[~np.isfinite(d)] = 0.0
    d[d <= 0] = 0.0
    return d


def parse_intrinsics_txt(txt_path):
    try:
        s = open(txt_path, "r", errors="ignore").read()
    except Exception:
        return None
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    vals = [float(x) for x in nums]
    if len(vals) >= 9:
        K = np.array(vals[:9], dtype=np.float32).reshape(3, 3)
        if abs(K[2, 2]) < 1e-6:
            K[2, 2] = 1.0
        return K
    return None


def cam_swap(cam):
    return "03" if cam == "02" else ("02" if cam == "03" else cam)


# -------- filename parsers (IMPORTANT) --------
# image:       <drive>_image_<frame>_image_<cam>.png
IMG_RE = re.compile(r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_image_(?P<frame>\d+)_image_(?P<cam>\d{2})$")

# intrinsics:  <drive>_image_<frame>_image_<cam>.txt
K_RE = IMG_RE

# depth:       <drive>_groundtruth_depth_<frame>_image_<cam>.png
DEP_RE = re.compile(r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)_groundtruth_depth_(?P<frame>\d+)_image_(?P<cam>\d{2})$")


def parse_key(stem, kind):
    if kind == "img":
        m = IMG_RE.match(stem)
    elif kind == "K":
        m = K_RE.match(stem)
    elif kind == "dep":
        m = DEP_RE.match(stem)
    else:
        return None

    if not m:
        return None
    drive = m.group("drive")
    frame = int(m.group("frame"))
    cam = m.group("cam")
    return (drive, frame, cam)


def build_pairs(keys, stride=1, max_pairs=8000):
    """
    keys: list of (drive, frame, cam) that have valid samples
    pairs are by (drive, cam) sequence, consecutive by frame.
    """
    groups = defaultdict(list)
    for (drive, frame, cam) in keys:
        groups[(drive, cam)].append(frame)

    pairs = []
    for (drive, cam), frames in groups.items():
        frames = sorted(set(frames))
        for i in range(0, len(frames) - stride):
            a = (drive, frames[i], cam)
            b = (drive, frames[i + stride], cam)
            pairs.append((a, b))
            if max_pairs and len(pairs) >= max_pairs:
                return pairs
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="val_selection_cropped root")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_pairs", type=int, default=8000)
    ap.add_argument("--depth_divisor", type=float, default=256.0)
    args = ap.parse_args()

    root = args.root
    img_dir = osp.join(root, "image")
    dep_dir = osp.join(root, "groundtruth_depth")
    K_dir = osp.join(root, "intrinsics")

    assert osp.isdir(img_dir), f"missing {img_dir}"
    assert osp.isdir(dep_dir), f"missing {dep_dir}"
    assert osp.isdir(K_dir), f"missing {K_dir}"

    # index: key=(drive,frame,cam) -> path
    img_map = {}
    for p in glob.glob(osp.join(img_dir, "*.png")):
        stem = osp.splitext(osp.basename(p))[0]
        key = parse_key(stem, "img")
        if key:
            img_map[key] = p

    dep_map = {}
    for p in glob.glob(osp.join(dep_dir, "*.png")):
        stem = osp.splitext(osp.basename(p))[0]
        key = parse_key(stem, "dep")
        if key:
            dep_map[key] = p

    K_map = {}
    for p in glob.glob(osp.join(K_dir, "*.txt")):
        stem = osp.splitext(osp.basename(p))[0]
        key = parse_key(stem, "K")
        if key:
            K_map[key] = p

    if not img_map:
        raise RuntimeError("No image files parsed. Check image filename pattern.")
    if not dep_map:
        raise RuntimeError("No depth files parsed. Check depth filename pattern.")

    # first try strict intersection by (drive,frame,cam)
    keys = sorted(set(img_map.keys()) & set(dep_map.keys()))
    swapped = False

    # if strict intersection empty, try cam swap fallback: match (drive,frame,cam) depth with image of swapped cam
    if len(keys) == 0:
        cand = []
        for (drive, frame, cam), dp in dep_map.items():
            img_key = (drive, frame, cam_swap(cam))
            if img_key in img_map:
                cand.append((drive, frame, cam))  # keep depth cam as key; image will be swapped
        keys = sorted(set(cand))
        swapped = True

    if len(keys) == 0:
        raise RuntimeError("No matched (image, depth) pairs found even with cam swap. "
                           "Likely filenames differ more than expected.")

    print(f"[KITTI selection] parsed images={len(img_map)} depths={len(dep_map)} intrinsics={len(K_map)}")
    print(f"[KITTI selection] matched samples={len(keys)} (cam_swapped={swapped})")

    # output
    out_img = osp.join(args.out_dir, "images")
    out_dep = osp.join(args.out_dir, "depths")
    out_cam = osp.join(args.out_dir, "cams")
    ensure_dir(args.out_dir); ensure_dir(out_img); ensure_dir(out_dep); ensure_dir(out_cam)

    pose = np.eye(4, dtype=np.float32)

    kept_keys = []
    for (drive, frame, cam) in keys:
        # depth key
        dep_key = (drive, frame, cam)
        dp = dep_map[dep_key]

        # image key (either same cam or swapped cam)
        img_key = (drive, frame, cam)
        if img_key not in img_map:
            img_key = (drive, frame, cam_swap(cam))
        ip = img_map[img_key]

        # intrinsics should follow the image cam
        kp = K_map.get(img_key, None)

        stem_out = f"{drive}_frame_{frame:010d}_image_{cam}"  # unify stem for output

        jpg_out = osp.join(out_img, stem_out + ".jpg")
        exr_out = osp.join(out_dep, stem_out + ".exr")
        cam_out = osp.join(out_cam, stem_out + ".npz")

        if not (osp.isfile(jpg_out) and osp.isfile(exr_out) and osp.isfile(cam_out)):
            rgb = imread_rgb(ip)
            H, W = rgb.shape[:2]
            Image.fromarray(rgb).save(jpg_out, quality=90)

            depth = read_depth_png(dp, depth_divisor=args.depth_divisor)
            if depth.shape[:2] != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            ok = cv2.imwrite(exr_out, depth.astype(np.float32))
            if not ok:
                raise RuntimeError("OpenCV failed to write EXR. Ensure OpenEXR is enabled in your OpenCV build.")

            K = parse_intrinsics_txt(kp) if kp is not None else None
            if K is None:
                fx = fy = 0.9 * max(H, W)
                cx = W / 2.0
                cy = H / 2.0
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            np.savez(cam_out, intrinsics=K.astype(np.float32), pose=pose.astype(np.float32))

        kept_keys.append((drive, frame, cam))

    # build pairs on kept keys
    pairs_key = build_pairs(kept_keys, stride=args.stride, max_pairs=args.max_pairs)

    # store pairs as output stems
    def key2stem(k):
        drive, frame, cam = k
        return f"{drive}_frame_{frame:010d}_image_{cam}"

    frames_out = sorted([key2stem(k) for k in kept_keys])
    pairs_out = [(key2stem(a), key2stem(b)) for (a, b) in pairs_key]

    np.savez(
        osp.join(args.out_dir, "pairs.npz"),
        frames=np.array(frames_out, dtype=object),
        pairs=np.array(pairs_out, dtype=object),
    )

    print(f"[KITTI selection] wrote frames={len(frames_out)} pairs={len(pairs_out)} to {args.out_dir}")


if __name__ == "__main__":
    main()
