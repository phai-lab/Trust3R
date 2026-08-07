#!/usr/bin/env python3
# python datasets_preprocess/preprocess_tum_rgbd_final.py \
#   --tgzs "/DATA2/EviP3R/data/TUM_rgbd/*.tgz" \
#   --out_root "/DATA2/EviP3R/data/tum_processed_v1" \
#   --work_root "/DATA2/EviP3R/data/_tmp_tum_extract" \
#   --max_dt 0.02 \
#   --pair_deltas "1,2,5,10,20,40" \
#   --pairs_per_delta 5000 \
#   --seed 0 \
#   --cleanup_extract

import os, os.path as osp, argparse, json, tarfile, shutil
import numpy as np
import cv2

def read_stamp_file(txt_path):
    ts, files = [], []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t, fn = line.split()[:2]
            ts.append(float(t))
            files.append(fn)
    return np.asarray(ts, np.float64), files

def read_groundtruth(gt_path):
    ts, Ts = [], []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            t = float(toks[0])
            tx, ty, tz = map(float, toks[1:4])
            qx, qy, qz, qw = map(float, toks[4:8])  # TUM: (qx qy qz qw)
            x, y, z, w = qx, qy, qz, qw
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
            ], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3,:3] = R
            T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
            ts.append(t)
            Ts.append(T)
    return np.asarray(ts, np.float64), np.stack(Ts, 0).astype(np.float32)

def nearest_assoc(tA, tB, max_dt):
    # return list of (iA, iB) with nearest tB within threshold
    idx = np.searchsorted(tB, tA)
    out = []
    for i, j in enumerate(idx):
        cand = []
        if j < len(tB): cand.append(j)
        if j-1 >= 0: cand.append(j-1)
        if not cand: continue
        cand = sorted(cand, key=lambda k: abs(tB[k]-tA[i]))
        k = cand[0]
        if abs(tB[k]-tA[i]) <= max_dt:
            out.append((i, k))
    return out

def intrinsics_for_seq(seq_name):
    # 官方常用的 TUM 内参（按 fr1/fr2/fr3）
    if "freiburg1" in seq_name:
        fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
    elif "freiburg2" in seq_name:
        fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    elif "freiburg3" in seq_name:
        fx, fy, cx, cy = 535.4, 539.2, 320.1, 247.6
    else:
        fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
    K = np.eye(3, dtype=np.float32)
    K[0,0], K[1,1], K[0,2], K[1,2] = fx, fy, cx, cy
    return K

def extract_tgz(tgz_path, extract_root):
    os.makedirs(extract_root, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_root)

def copy_rgb_to_jpg(src_png, dst_jpg):
    img = cv2.imread(src_png, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read: {src_png}")
    os.makedirs(osp.dirname(dst_jpg), exist_ok=True)
    ok = cv2.imwrite(dst_jpg, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to write: {dst_jpg}")

def copy_depth_png(src_png, dst_png):
    dep = cv2.imread(src_png, cv2.IMREAD_UNCHANGED)  # uint16
    if dep is None:
        raise RuntimeError(f"Failed to read depth: {src_png}")
    os.makedirs(osp.dirname(dst_png), exist_ok=True)
    ok = cv2.imwrite(dst_png, dep)
    if not ok:
        raise RuntimeError(f"Failed to write depth: {dst_png}")

def sample_pairs(n, deltas, pairs_per_delta, seed):
    rng = np.random.default_rng(seed)
    pairs = []
    for d in deltas:
        if n <= d: 
            continue
        all_i = np.arange(0, n - d, dtype=np.int64)
        if pairs_per_delta > 0 and len(all_i) > pairs_per_delta:
            sel = rng.choice(all_i, size=pairs_per_delta, replace=False)
            sel.sort()
        else:
            sel = all_i
        for i in sel:
            pairs.append((int(i), int(i + d)))
    return pairs

def main(args):
    os.makedirs(args.out_root, exist_ok=True)
    work_root = args.work_root
    os.makedirs(work_root, exist_ok=True)

    tgzs = [p.strip() for p in args.tgzs.split(",") if p.strip()]
    if len(tgzs) == 1 and ("*" in tgzs[0] or "?" in tgzs[0] or "[" in tgzs[0]):
        import glob
        tgzs = sorted(glob.glob(tgzs[0]))

    assert tgzs, "No tgz found."

    scenes, sceneids, images, intrinsics, trajectories = [], [], [], [], []
    pairs_global = []
    t_rgb_all, t_depth_all, t_gt_all = [], [], []

    offset = 0
    for tgz in tgzs:
        seq_name = osp.splitext(osp.basename(tgz))[0]  # remove .tgz
        scenes.append(seq_name)
        seq_extract = osp.join(work_root, seq_name)
        if osp.isdir(seq_extract):
            shutil.rmtree(seq_extract)
        extract_tgz(tgz, work_root)

        seq_dir = osp.join(work_root, seq_name)
        rgb_txt = osp.join(seq_dir, "rgb.txt")
        depth_txt = osp.join(seq_dir, "depth.txt")
        gt_txt = osp.join(seq_dir, "groundtruth.txt")
        assert osp.isfile(rgb_txt) and osp.isfile(depth_txt) and osp.isfile(gt_txt), f"Bad seq: {seq_dir}"

        t_rgb, f_rgb = read_stamp_file(rgb_txt)
        t_dep, f_dep = read_stamp_file(depth_txt)
        t_gt,  T_gt  = read_groundtruth(gt_txt)

        # associate RGB -> depth, then RGB -> GT (using RGB timestamps)
        rgb_dep = nearest_assoc(t_rgb, t_dep, args.max_dt)
        rgb_idx = np.array([i for (i, _) in rgb_dep], dtype=np.int64)
        t_rgb_sel = t_rgb[rgb_idx]
        rgb_gt = nearest_assoc(t_rgb_sel, t_gt, args.max_dt)
        gt_map = {k_in: j_gt for (k_in, j_gt) in rgb_gt}

        triples = []
        for k, (i_rgb, j_dep) in enumerate(rgb_dep):
            if k not in gt_map:
                continue
            j_gt = gt_map[k]
            triples.append((i_rgb, j_dep, j_gt))

        if len(triples) < 50:
            print(f"[WARN] {seq_name}: too few matched frames ({len(triples)}). skip.")
            continue

        K = intrinsics_for_seq(seq_name)

        # output dirs
        out_scene = osp.join(args.out_root, seq_name)
        out_img_dir = osp.join(out_scene, "images")
        out_dep_dir = osp.join(out_scene, "depth")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_dep_dir, exist_ok=True)

        local_basenames = []
        local_T = []
        local_K = []
        local_trgb, local_tdep, local_tgt = [], [], []

        for (i_rgb, j_dep, j_gt) in triples:
            rgb_rel = f_rgb[i_rgb]     # e.g. rgb/1305031102.175304.png
            dep_rel = f_dep[j_dep]     # e.g. depth/1305031102.160407.png
            stem = osp.splitext(osp.basename(rgb_rel))[0]  # use RGB timestamp as canonical id

            src_rgb = osp.join(seq_dir, rgb_rel)
            src_dep = osp.join(seq_dir, dep_rel)

            dst_rgb = osp.join(out_img_dir, stem + ".jpg")
            dst_dep = osp.join(out_dep_dir, stem + ".png")

            if not osp.isfile(dst_rgb):
                copy_rgb_to_jpg(src_rgb, dst_rgb)
            if not osp.isfile(dst_dep):
                copy_depth_png(src_dep, dst_dep)

            local_basenames.append(stem)
            local_T.append(T_gt[j_gt])
            local_K.append(K)

            local_trgb.append(float(t_rgb[i_rgb]))
            local_tdep.append(float(t_dep[j_dep]))
            local_tgt.append(float(t_gt[j_gt]))

        n = len(local_basenames)
        sid = len(scenes) - 1
        sceneids.extend([sid] * n)
        images.extend(local_basenames)
        intrinsics.extend(local_K)
        trajectories.extend(local_T)
        t_rgb_all.extend(local_trgb)
        t_depth_all.extend(local_tdep)
        t_gt_all.extend(local_tgt)

        # pairs (local -> global)
        deltas = [int(x) for x in args.pair_deltas.split(",") if x.strip()]
        local_pairs = sample_pairs(n, deltas, args.pairs_per_delta, seed=args.seed)
        for (a, b) in local_pairs:
            pairs_global.append((offset + a, offset + b))
        offset += n

        print(f"[TUM] {seq_name}: frames={n}, pairs={len(local_pairs)}")

        # cleanup extracted raw to avoid confusion (optional)
        if args.cleanup_extract:
            shutil.rmtree(seq_dir, ignore_errors=True)

    out_npz = osp.join(args.out_root, "all_metadata.npz")
    np.savez(
        out_npz,
        scenes=np.array(scenes),
        sceneids=np.array(sceneids, np.int64),
        images=np.array(images),
        intrinsics=np.stack(intrinsics, 0).astype(np.float32),
        trajectories=np.stack(trajectories, 0).astype(np.float32),
        pairs=np.array(pairs_global, np.int64),
        timestamps_rgb=np.array(t_rgb_all, np.float64),
        timestamps_depth=np.array(t_depth_all, np.float64),
        timestamps_gt=np.array(t_gt_all, np.float64),
    )

    cfg = dict(
        tgzs=tgzs,
        max_dt=args.max_dt,
        pair_deltas=args.pair_deltas,
        pairs_per_delta=args.pairs_per_delta,
        seed=args.seed,
        depth_scale_note="TUM depth png is uint16 where depth(m)=value/5000; invalid=0",
        rgb_format="jpg(quality=95)",
        depth_format="uint16 png (raw copied, renamed to RGB timestamp)",
    )
    with open(osp.join(args.out_root, "preprocess_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[OK] wrote: {out_npz}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgzs", required=True,
                    help="Comma list or a glob, e.g. '/DATA2/.../TUM_rgbd/*.tgz'")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--work_root", default="/tmp/tum_extract")
    ap.add_argument("--max_dt", type=float, default=0.02)
    ap.add_argument("--pair_deltas", type=str, default="1,2,5,10,20,40")
    ap.add_argument("--pairs_per_delta", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cleanup_extract", action="store_true")
    args = ap.parse_args()
    main(args)
