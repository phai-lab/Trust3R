#!/usr/bin/env python3
import os
import os.path as osp
import re
import glob
import argparse
import shutil
import subprocess
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR


# ---------------------- misc utils ----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_7z_names(archive: str) -> List[str]:
    """List member names inside a .7z archive (prefers system 7z, falls back to py7zr)."""
    exe = (shutil.which("7z") or shutil.which("7zz") or shutil.which("7za") or shutil.which("7zr"))
    if exe is not None:
        # 7z l -ba prints file names; parse lines that look like paths.
        out = subprocess.check_output([exe, "l", "-ba", archive], text=True, errors="ignore")
        names = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Heuristic: path-like and not header/footer
            if "/" in line or "\\" in line:
                names.append(line.replace("\\", "/"))
        return names

    import py7zr
    with py7zr.SevenZipFile(archive, "r") as z:
        return z.getnames()


def extract_7z(archive: str, out_dir: str, members: Optional[List[str]] = None):
    """
    Extract a .7z archive. If members is provided, extract only those members.
    Prefers system 7z if available; else uses py7zr.
    """
    ensure_dir(out_dir)
    exe = (shutil.which("7z") or shutil.which("7zz") or shutil.which("7za") or shutil.which("7zr"))
    if exe is not None:
        if members is None or len(members) == 0:
            subprocess.check_call([exe, "x", archive, f"-o{out_dir}", "-y"])
        else:
            # Extract selected files one-by-one (portable)
            for m in members:
                subprocess.check_call([exe, "x", archive, m, f"-o{out_dir}", "-y"])
        return

    import py7zr
    with py7zr.SevenZipFile(archive, mode="r") as z:
        if members is None or len(members) == 0:
            z.extractall(path=out_dir)
        else:
            z.extract(targets=members, path=out_dir)


def imread_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb_jpg(rgb: np.ndarray, path: str, quality: int = 90):
    Image.fromarray(rgb).save(path, quality=quality)


def resize_with_K(rgb: np.ndarray, K: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Resize image and scale intrinsics accordingly."""
    h0, w0 = rgb.shape[:2]
    sx = out_w / float(w0)
    sy = out_h / float(h0)
    rgb2 = np.array(Image.fromarray(rgb).resize((out_w, out_h), resample=RESAMPLE_BILINEAR))
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] *= sx
    K2[1, 2] *= sy
    return rgb2, K2, sx, sy


# ---------------------- COLMAP parsers ----------------------
def parse_colmap_cameras_txt(path: str) -> Dict[int, dict]:
    """
    cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    """
    cams = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(x) for x in parts[4:]], dtype=np.float32)
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cams


def qvec2rotmat(qw, qx, qy, qz) -> np.ndarray:
    """
    Hamilton convention (Eigen style), as ETH3D documentation states for images.txt.
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    qw, qx, qy, qz = q.tolist()
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    return R.astype(np.float32)


def parse_colmap_images_txt(path: str) -> Dict[str, dict]:
    """
    images.txt:
      IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    followed by a line of 2D observations (ignored).
    Return dict keyed by image basename -> {camera_id, world_to_cam(4x4), name}
    """
    mp = {}
    with open(path, "r", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        i += 1
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 10:
            continue

        img_id = int(parts[0])  # not used
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]  # may include path like dslr_images/DSC_XXXX.JPG

        R = qvec2rotmat(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3:4] = t

        base = osp.basename(name)
        mp[base] = {"image_id": img_id, "camera_id": cam_id, "world_to_cam": w2c, "name": name}

        # skip the next line (points2D), if present
        if i < len(lines) and lines[i] and not lines[i].startswith("#"):
            i += 1

    return mp


def parse_colmap_points3d_txt(path: str) -> List[List[int]]:
    """
    points3D.txt:
      POINT3D_ID X Y Z R G B ERROR TRACK[ (IMAGE_ID, POINT2D_IDX) ... ]
    Return list of image_id tracks.
    """
    tracks: List[List[int]] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            track = parts[8:]
            if len(track) < 2:
                continue
            ids: List[int] = []
            for i in range(0, len(track) - 1, 2):
                try:
                    ids.append(int(track[i]))
                except ValueError:
                    continue
            if len(ids) >= 2:
                tracks.append(ids)
    return tracks


def K_from_camera_params(model: str, params: np.ndarray) -> np.ndarray:
    """
    Return a 3x3 K for PINHOLE or THIN_PRISM_FISHEYE (fx, fy, cx, cy are the first 4 params).
    """
    if model not in ["PINHOLE", "THIN_PRISM_FISHEYE"]:
        raise ValueError(f"Unsupported camera model for this script: {model}")
    fx, fy, cx, cy = params[:4].tolist()
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K


# ---------------------- ETH3D depth (float32 dump with .JPG extension) ----------------------
def read_eth3d_depth_dump(path: str, H: int, W: int) -> np.ndarray:
    """
    ETH3D: depth files in *_depth.7z are 4-byte float binary dumps (row-major),
    invalid pixels are +inf. (Doc: not actually JPEGs even though extension is .JPG)
    """
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.float32)
    if arr.size != H * W:
        raise RuntimeError(f"Depth dump size mismatch: got {arr.size} floats, expected {H*W} (H={H},W={W}). File={path}")
    depth = arr.reshape(H, W)  # row-major, top-to-bottom already
    depth = depth.astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0  # inf -> 0
    depth[depth <= 0] = 0.0
    return depth


# ---------------------- THIN_PRISM_FISHEYE projection (ETH3D official formula) ----------------------
def project_thin_prism_fisheye(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward projection for THIN_PRISM_FISHEYE as described in ETH3D documentation.
    Inputs x,y are normalized coordinates (x=X/Z, y=Y/Z) for some 3D point/ray (Z>0).
    params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]
    Returns pixel coords u,v in the distorted image.
    """
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params.tolist()

    r = np.sqrt(x*x + y*y)
    theta = np.arctan(r)

    th2 = theta*theta
    th4 = th2*th2
    th6 = th4*th2
    th8 = th4*th4
    theta_d = theta * (1 + k1*th2 + k2*th4 + k3*th6 + k4*th8)

    scale = np.ones_like(r, dtype=np.float32)
    m = r > 1e-8
    scale[m] = (theta_d[m] / r[m]).astype(np.float32)

    x_d = x * scale
    y_d = y * scale

    r2 = x_d*x_d + y_d*y_d
    x_dd = x_d + 2*p1*x_d*y_d + p2*(r2 + 2*x_d*x_d) + sx1*(theta_d*theta_d)
    y_dd = y_d + p1*(r2 + 2*y_d*y_d) + 2*p2*x_d*y_d + sy1*(theta_d*theta_d)

    u = fx * x_dd + cx
    v = fy * y_dd + cy
    return u.astype(np.float32), v.astype(np.float32)


def remap_depth_distorted_to_undistorted(
    depth_d: np.ndarray,
    K_u: np.ndarray,
    fisheye_params: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """
    Build undistorted depth by sampling distorted depth along matching rays:
      - For each undistorted pixel (u,v), compute ray direction via PINHOLE K_u
      - Project that ray into distorted image using THIN_PRISM_FISHEYE params
      - Nearest-neighbor sample depth_d at that distorted pixel
    """
    fx, fy, cx, cy = K_u[0, 0], K_u[1, 1], K_u[0, 2], K_u[1, 2]

    # Pixel coordinates in ETH3D are defined with (0,0) at top-left pixel center.
    uu = np.arange(out_w, dtype=np.float32)
    vv = np.arange(out_h, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)

    x = (U - cx) / fx
    y = (V - cy) / fy

    u_d, v_d = project_thin_prism_fisheye(x, y, fisheye_params)

    # nearest sampling
    u_i = np.rint(u_d).astype(np.int32)
    v_i = np.rint(v_d).astype(np.int32)

    H, W = depth_d.shape[:2]
    valid = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)

    out = np.zeros((out_h, out_w), dtype=np.float32)
    out[valid] = depth_d[v_i[valid], u_i[valid]]
    out[~np.isfinite(out)] = 0.0
    out[out <= 0] = 0.0
    return out


# ---------------------- pairs ----------------------
def build_pairs(frames: List[str], stride: int, max_pairs: int) -> List[Tuple[str, str]]:
    frames = list(frames)
    pairs = []
    for i in range(0, len(frames) - stride):
        pairs.append((frames[i], frames[i + stride]))
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


def build_pairs_all(
    frames: List[str],
    min_gap: int,
    max_gap: int,
    max_pairs: int,
) -> List[Tuple[str, str]]:
    frames = list(frames)
    pairs: List[Tuple[str, str]] = []
    n = len(frames)
    for i in range(n):
        j_start = i + min_gap
        if j_start >= n:
            break
        j_end = min(n - 1, i + max_gap)
        for j in range(j_start, j_end + 1):
            pairs.append((frames[i], frames[j]))
    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(0)
        rng.shuffle(pairs)
        pairs = pairs[:max_pairs]
    return pairs


def build_pairs_covis(
    frames: List[str],
    tracks: List[List[int]],
    idx_by_image_id: Dict[int, int],
    min_gap: int,
    max_gap: int,
    min_shared_points: int,
    max_pairs: int,
) -> Tuple[List[Tuple[str, str]], float]:
    shared: Dict[Tuple[int, int], int] = {}
    for track in tracks:
        indices = sorted({idx_by_image_id[i] for i in track if i in idx_by_image_id})
        if len(indices) < 2:
            continue
        for a in range(len(indices) - 1):
            ia = indices[a]
            for b in range(a + 1, len(indices)):
                ib = indices[b]
                gap = ib - ia
                if gap < min_gap or gap > max_gap:
                    continue
                key = (ia, ib)
                shared[key] = shared.get(key, 0) + 1

    candidates = [
        (i, j, cnt) for (i, j), cnt in shared.items() if cnt >= min_shared_points
    ]
    candidates.sort(key=lambda x: (-x[2], x[0], x[1]))
    if max_pairs and len(candidates) > max_pairs:
        candidates = candidates[:max_pairs]

    pairs = [(frames[i], frames[j]) for i, j, _ in candidates]
    counts = [c for _, _, c in candidates]
    median_shared = float(np.median(counts)) if counts else 0.0
    return pairs, median_shared


# ---------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="", help="Folder containing ETH3D *.7z archives")
    ap.add_argument("--root", default="", help="Legacy alias for --raw_dir")
    ap.add_argument("--roots", nargs="*", default=[], help="Alias for --raw_dir (first entry used)")
    ap.add_argument("--out_dir", required=True, help="Output processed root")
    ap.add_argument("--extract_dir", default="", help="Cache extract dir (default: <raw_dir>/_extracted_eth3d)")
    ap.add_argument("--scenes", default="", help="Comma-separated scenes to process (empty=all)")
    ap.add_argument("--use_undistorted", action="store_true", help="Use *_dslr_undistorted.7z (PINHOLE) and remap depth to it")
    ap.add_argument("--max_frames_per_scene", type=int, default=0, help="If >0, keep at most this many frames per scene")
    ap.add_argument("--frame_stride", type=int, default=1, help="Subsample frames (keep every N-th)")
    ap.add_argument("--pair_stride", type=int, default=10, help="Pairs are (i, i+pair_stride)")
    ap.add_argument("--pair_mode", choices=["stride", "all", "covis"], default="stride")
    ap.add_argument("--min_pair_gap", type=int, default=2, help="Minimum index gap for all/covis modes")
    ap.add_argument("--max_pair_gap", type=int, default=999999, help="Maximum index gap for all/covis modes")
    ap.add_argument("--min_shared_points", type=int, default=100, help="Minimum shared points for covis mode")
    ap.add_argument("--max_pairs_per_scene", type=int, default=5000, help="Cap pairs per scene for all/covis (0=unlimited)")
    ap.add_argument("--max_pairs", type=int, default=4000)
    ap.add_argument("--resize_long_edge", type=int, default=0, help="If >0, resize images so long edge == this (scale K accordingly)")
    ap.add_argument("--min_valid_depth_pixels", type=int, default=1, help="Keep frame if depth>0 count >= this (ETH3D depth is sparse)")
    ap.add_argument("--keep_empty_depth", action="store_true", help="Keep frames even if valid depth pixels == 0")
    ap.add_argument("--write_global_pairs", action="store_true", help="Also write <out_dir>/all_pairs.npz")
    ap.add_argument("--force_extract", action="store_true", help="Force re-extract selected files")
    args = ap.parse_args()

    raw_dir = args.raw_dir.strip() if args.raw_dir else ""
    if not raw_dir and args.root:
        raw_dir = args.root.strip()
    if not raw_dir and args.roots:
        roots = []
        for item in args.roots:
            if not item:
                continue
            roots.extend([p.strip() for p in item.split(",") if p.strip()])
        if roots:
            raw_dir = roots[0]
            if len(roots) > 1:
                print(f"[ETH3D] Warning: multiple --roots provided; using first: {raw_dir}")
    if not raw_dir:
        raise RuntimeError("Please provide --raw_dir or --root/--roots")
    out_dir = args.out_dir
    ensure_dir(out_dir)

    extract_base = args.extract_dir.strip() if args.extract_dir.strip() else osp.join(raw_dir, "_extracted_eth3d")
    ensure_dir(extract_base)

    # scene filter
    scene_allow = None
    if args.scenes.strip():
        scene_allow = set([s.strip() for s in args.scenes.split(",") if s.strip()])

    # find scenes from *_dslr_jpg.7z
    jpg_archives = sorted(glob.glob(osp.join(raw_dir, "*_dslr_jpg.7z")), key=natural_key)
    if not jpg_archives:
        raise RuntimeError(f"No *_dslr_jpg.7z found in {raw_dir}")

    global_pairs = []
    global_frames = []

    for jpg7z in jpg_archives:
        scene = osp.basename(jpg7z).replace("_dslr_jpg.7z", "")
        if scene_allow is not None and scene not in scene_allow:
            continue

        depth7z = osp.join(raw_dir, f"{scene}_dslr_depth.7z")
        if not osp.isfile(depth7z):
            print(f"[ETH3D] {scene}: missing {osp.basename(depth7z)} -> skip")
            continue

        und7z = osp.join(raw_dir, f"{scene}_dslr_undistorted.7z")
        if args.use_undistorted and (not osp.isfile(und7z)):
            print(f"[ETH3D] {scene}: requested undistorted but missing {osp.basename(und7z)} -> skip")
            continue

        # cache dirs
        scene_cache = osp.join(extract_base, scene)
        ensure_dir(scene_cache)

        # ---- locate calib files inside archives ----
        def pick_calib_members(
            names: List[str],
            calib_tag: str,
            require_points: bool = False,
        ) -> Tuple[str, str, Optional[str]]:
            cam = [n for n in names if n.endswith(f"{calib_tag}/cameras.txt")]
            img = [n for n in names if n.endswith(f"{calib_tag}/images.txt")]
            pts = [n for n in names if n.endswith(f"{calib_tag}/points3D.txt")]
            if not cam or not img:
                raise RuntimeError(f"[{scene}] cannot find {calib_tag}/(cameras.txt, images.txt) in archive")
            if require_points and not pts:
                raise RuntimeError(f"[{scene}] cannot find {calib_tag}/points3D.txt in archive")
            return cam[0], img[0], (pts[0] if pts else None)

        jpg_names = list_7z_names(jpg7z)
        need_points_jpg = (args.pair_mode == "covis") and (not args.use_undistorted)
        cam_jpg_m, img_jpg_m, pts_jpg_m = pick_calib_members(
            jpg_names, "dslr_calibration_jpg", require_points=need_points_jpg
        )

        und_names = None
        if args.use_undistorted:
            und_names = list_7z_names(und7z)
            need_points_und = (args.pair_mode == "covis") and args.use_undistorted
            cam_und_m, img_und_m, pts_und_m = pick_calib_members(
                und_names, "dslr_calibration_undistorted", require_points=need_points_und
            )

        # Extract calib (small)
        calib_dir = osp.join(scene_cache, "calib")
        ensure_dir(calib_dir)

        cameras_jpg_path = osp.join(calib_dir, "cameras_jpg.txt")
        images_jpg_path = osp.join(calib_dir, "images_jpg.txt")
        points_jpg_path = osp.join(calib_dir, "points_jpg.txt")
        need_extract_jpg = args.force_extract or (not osp.isfile(cameras_jpg_path)) or (
            need_points_jpg and not osp.isfile(points_jpg_path)
        )
        if need_extract_jpg:
            members = [cam_jpg_m, img_jpg_m]
            if pts_jpg_m:
                members.append(pts_jpg_m)
            extract_7z(jpg7z, calib_dir, members)
            # rename to stable names
            # extracted path keeps subdirs, so search
            cams_found = glob.glob(osp.join(calib_dir, "**", "cameras.txt"), recursive=True)
            imgs_found = glob.glob(osp.join(calib_dir, "**", "images.txt"), recursive=True)
            if not cams_found or not imgs_found:
                raise RuntimeError(f"[{scene}] calib extract failed for jpg")
            shutil.copyfile(cams_found[0], cameras_jpg_path)
            shutil.copyfile(imgs_found[0], images_jpg_path)
            if pts_jpg_m:
                pts_found = glob.glob(osp.join(calib_dir, "**", "points3D.txt"), recursive=True)
                pts_jpg = [p for p in pts_found if "dslr_calibration_jpg" in p.replace("\\", "/")]
                if pts_jpg:
                    shutil.copyfile(pts_jpg[0], points_jpg_path)

        if args.use_undistorted:
            cameras_und_path = osp.join(calib_dir, "cameras_und.txt")
            images_und_path = osp.join(calib_dir, "images_und.txt")
            points_und_path = osp.join(calib_dir, "points_und.txt")
            need_extract_und = args.force_extract or (not osp.isfile(cameras_und_path)) or (
                need_points_und and not osp.isfile(points_und_path)
            )
            if need_extract_und:
                members = [cam_und_m, img_und_m]
                if pts_und_m:
                    members.append(pts_und_m)
                extract_7z(und7z, calib_dir, members)
                cams_found = glob.glob(osp.join(calib_dir, "**", "cameras.txt"), recursive=True)
                imgs_found = glob.glob(osp.join(calib_dir, "**", "images.txt"), recursive=True)
                pts_found = glob.glob(osp.join(calib_dir, "**", "points3D.txt"), recursive=True)
                # Note: after extracting both jpg/und, there may be multiple matches; pick the one under undistorted tag
                cams_und = [p for p in cams_found if "dslr_calibration_undistorted" in p.replace("\\", "/")]
                imgs_und = [p for p in imgs_found if "dslr_calibration_undistorted" in p.replace("\\", "/")]
                pts_und = [p for p in pts_found if "dslr_calibration_undistorted" in p.replace("\\", "/")]
                if not cams_und or not imgs_und:
                    raise RuntimeError(f"[{scene}] calib extract failed for undistorted")
                shutil.copyfile(cams_und[0], cameras_und_path)
                shutil.copyfile(imgs_und[0], images_und_path)
                if pts_und:
                    shutil.copyfile(pts_und[0], points_und_path)

        # Parse calib
        cams_jpg = parse_colmap_cameras_txt(cameras_jpg_path)
        imgs_jpg = parse_colmap_images_txt(images_jpg_path)

        if args.use_undistorted:
            cams_und = parse_colmap_cameras_txt(cameras_und_path)
            imgs_und = parse_colmap_images_txt(images_und_path)
        else:
            cams_und, imgs_und = None, None

        points_ref = None
        if args.pair_mode == "covis":
            points_path = points_und_path if args.use_undistorted else points_jpg_path
            if not osp.isfile(points_path):
                raise RuntimeError(f"[{scene}] points3D.txt missing for covis pairing")
            points_ref = parse_colmap_points3d_txt(points_path)

        # ---- build member index for RGB and depth ----
        def index_image_members(names: List[str]) -> Dict[str, str]:
            exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
            mp = {}
            for n in names:
                suf = osp.splitext(n)[1]
                if suf in exts:
                    base = osp.basename(n)
                    if base not in mp:
                        mp[base] = n
            return mp

        # From jpg archive: original distorted images exist there
        img_members_jpg = index_image_members(jpg_names)

        # From und archive: undistorted images exist there
        img_members_und = index_image_members(und_names) if args.use_undistorted else None

        # Depth archive: depth dumps are *.JPG but are float32 dumps; index by basename
        depth_names = list_7z_names(depth7z)
        depth_members = {}
        for n in depth_names:
            if n.lower().endswith(".jpg"):
                base = osp.basename(n)
                if base not in depth_members:
                    depth_members[base] = n

        if len(depth_members) == 0:
            print(f"[ETH3D] {scene}: no depth members ending with .JPG found in depth archive -> skip")
            continue

        # ---- select frames (use undistorted list if requested; else jpg list) ----
        imgs_ref = imgs_und if args.use_undistorted else imgs_jpg
        cams_ref = cams_und if args.use_undistorted else cams_jpg

        # sort by basename
        all_bases = sorted(list(imgs_ref.keys()), key=natural_key)

        # subsample by frame_stride / max_frames
        if args.frame_stride > 1:
            all_bases = all_bases[::args.frame_stride]
        if args.max_frames_per_scene and len(all_bases) > args.max_frames_per_scene:
            all_bases = all_bases[:args.max_frames_per_scene]

        # keep only those that have depth
        sel_bases = [b for b in all_bases if b in depth_members]
        if len(sel_bases) < (args.pair_stride + 1):
            print(f"[ETH3D] {scene}: not enough frames with depth after selection ({len(sel_bases)}), skip")
            continue

        # ---- extract selected RGB + depth to cache ----
        rgb_cache = osp.join(scene_cache, "rgb_und" if args.use_undistorted else "rgb_jpg")
        dep_cache = osp.join(scene_cache, "depth_dump")
        ensure_dir(rgb_cache); ensure_dir(dep_cache)

        # choose correct archive for RGB
        rgb7z = und7z if args.use_undistorted else jpg7z
        rgb_names = und_names if args.use_undistorted else jpg_names
        rgb_members = img_members_und if args.use_undistorted else img_members_jpg

        rgb_to_extract = []
        dep_to_extract = []
        for b in sel_bases:
            if b in rgb_members:
                rgb_to_extract.append(rgb_members[b])
            dep_to_extract.append(depth_members[b])

        # Extract (skip if already exists unless force_extract)
        if args.force_extract:
            # wipe cache (optional)
            pass

        # Extract RGB members (only)
        # To avoid re-extracting too much: check one sample exists.
        sample_rgb_path = glob.glob(osp.join(rgb_cache, "**", sel_bases[0]), recursive=True)
        if args.force_extract or (len(sample_rgb_path) == 0):
            extract_7z(rgb7z, rgb_cache, rgb_to_extract)

        # Extract depth members
        sample_dep_path = glob.glob(osp.join(dep_cache, "**", sel_bases[0]), recursive=True)
        if args.force_extract or (len(sample_dep_path) == 0):
            extract_7z(depth7z, dep_cache, dep_to_extract)

        # Build extracted path maps by basename
        def build_extracted_map(root: str) -> Dict[str, str]:
            mp = {}
            for p in glob.glob(osp.join(root, "**", "*"), recursive=True):
                if osp.isfile(p):
                    mp[osp.basename(p)] = p
            return mp

        rgb_path_map = build_extracted_map(rgb_cache)
        dep_path_map = build_extracted_map(dep_cache)

        # ---- prepare output dirs ----
        out_scene = osp.join(out_dir, scene)
        out_img = osp.join(out_scene, "images")
        out_dep = osp.join(out_scene, "depths")
        out_cam = osp.join(out_scene, "cams")
        ensure_dir(out_img); ensure_dir(out_dep); ensure_dir(out_cam)

        kept_frames = []
        stem_to_base: Dict[str, str] = {}
        valid_total = 0
        valid_kept = 0

        # We'll need fisheye params from JPG camera (distorted) for remapping.
        # Pick per-image camera_id from imgs_jpg.
        # NOTE: ETH3D may use one camera block for all images, but we still do per-image.
        for base in sel_bases:
            if base not in rgb_path_map or base not in dep_path_map:
                continue

            # pose/extrinsics (use the ref model's images.txt)
            meta_ref = imgs_ref[base]
            cam_id_ref = meta_ref["camera_id"]
            cam_ref = cams_ref[cam_id_ref]
            K_ref = K_from_camera_params(cam_ref["model"], cam_ref["params"])

            # read RGB
            rgb = imread_rgb(rgb_path_map[base])
            H_u0, W_u0 = rgb.shape[:2]

            # resize if requested
            if args.resize_long_edge and args.resize_long_edge > 0:
                long0 = max(H_u0, W_u0)
                if long0 != args.resize_long_edge:
                    scale = args.resize_long_edge / float(long0)
                    out_w = int(round(W_u0 * scale))
                    out_h = int(round(H_u0 * scale))
                    rgb, K_ref, _, _ = resize_with_K(rgb, K_ref, out_w, out_h)
            H_u, W_u = rgb.shape[:2]

            # read distorted depth dump (need original distorted H,W from JPG camera block)
            meta_jpg = imgs_jpg.get(base, None)
            if meta_jpg is None:
                # If missing in jpg images.txt, still try best-effort: use any camera block
                cam_j = list(cams_jpg.values())[0]
                H_d0, W_d0 = cam_j["height"], cam_j["width"]
                params_fish = cam_j["params"]
            else:
                cam_id_j = meta_jpg["camera_id"]
                cam_j = cams_jpg[cam_id_j]
                H_d0, W_d0 = cam_j["height"], cam_j["width"]
                params_fish = cam_j["params"]
                if cam_j["model"] != "THIN_PRISM_FISHEYE":
                    raise RuntimeError(f"[{scene}] expected THIN_PRISM_FISHEYE in jpg calib, got {cam_j['model']}")

            depth_d = read_eth3d_depth_dump(dep_path_map[base], H=H_d0, W=W_d0)

            # If using undistorted RGB, remap depth -> undistorted
            if args.use_undistorted:
                depth_u = remap_depth_distorted_to_undistorted(
                    depth_d=depth_d,
                    K_u=K_ref,
                    fisheye_params=params_fish,
                    out_h=H_u,
                    out_w=W_u,
                )
            else:
                # Using original distorted RGB: depth matches it.
                # (We still output K from THIN_PRISM_FISHEYE first 4 params as 3x3.)
                # If resized, resize depth accordingly.
                depth_u = depth_d
                # if RGB was resized, also resize depth to RGB size
                if depth_u.shape[:2] != (H_u, W_u):
                    depth_u = cv2.resize(depth_u, (W_u, H_u), interpolation=cv2.INTER_NEAREST)

            valid_cnt = int((depth_u > 0).sum())
            if (valid_cnt < args.min_valid_depth_pixels) and (not args.keep_empty_depth):
                continue

            valid_total += valid_cnt
            valid_kept += 1

            # write outputs
            stem = osp.splitext(base)[0]  # DSC_0286
            stem_to_base[stem] = base
            jpg_out = osp.join(out_img, stem + ".jpg")
            exr_out = osp.join(out_dep, stem + ".exr")
            cam_out = osp.join(out_cam, stem + ".npz")

            if not (osp.isfile(jpg_out) and osp.isfile(exr_out) and osp.isfile(cam_out)):
                save_rgb_jpg(rgb, jpg_out, quality=90)

                ok = cv2.imwrite(exr_out, depth_u.astype(np.float32))
                if not ok:
                    raise RuntimeError("OpenCV failed to write EXR. Ensure OpenEXR is enabled in your OpenCV build.")

                w2c = meta_ref["world_to_cam"].astype(np.float32)
                c2w = np.linalg.inv(w2c).astype(np.float32)
                np.savez(
                    cam_out,
                    intrinsics=K_ref.astype(np.float32),
                    world_to_cam=w2c,
                    cam_to_world=c2w,
                )

            kept_frames.append(stem)

        kept_frames = sorted(list(set(kept_frames)), key=natural_key)
        avg_valid = (valid_total / valid_kept) if valid_kept else 0.0
        min_needed = (args.pair_stride + 1) if args.pair_mode == "stride" else (args.min_pair_gap + 1)
        if len(kept_frames) < min_needed:
            print(f"[ETH3D] {scene}: not enough kept frames ({len(kept_frames)}), skip pairs")
            continue

        median_shared = None
        if args.pair_mode == "stride":
            pairs = build_pairs(kept_frames, stride=args.pair_stride, max_pairs=args.max_pairs)
        elif args.pair_mode == "all":
            pairs = build_pairs_all(
                kept_frames,
                min_gap=args.min_pair_gap,
                max_gap=args.max_pair_gap,
                max_pairs=args.max_pairs_per_scene,
            )
        else:
            img_id_by_base = {b: meta.get("image_id") for b, meta in imgs_ref.items()}
            idx_by_image_id: Dict[int, int] = {}
            for idx, stem in enumerate(kept_frames):
                base = stem_to_base.get(stem)
                if not base:
                    continue
                img_id = img_id_by_base.get(base)
                if img_id is None:
                    continue
                idx_by_image_id[int(img_id)] = idx
            pairs, median_shared = build_pairs_covis(
                kept_frames,
                tracks=points_ref or [],
                idx_by_image_id=idx_by_image_id,
                min_gap=args.min_pair_gap,
                max_gap=args.max_pair_gap,
                min_shared_points=args.min_shared_points,
                max_pairs=args.max_pairs_per_scene,
            )
        np.savez(
            osp.join(out_scene, "pairs.npz"),
            frames=np.array(kept_frames, dtype=object),
            pairs=np.array(pairs, dtype=object),
        )

        if args.pair_mode == "covis":
            median_val = 0.0 if median_shared is None else median_shared
            print(
                f"[ETH3D] {scene}: kept_frames={len(kept_frames)} pairs={len(pairs)} "
                f"median_shared={median_val:.1f} avg_valid={avg_valid:.1f} "
                f"(use_undistorted={args.use_undistorted})"
            )
        else:
            print(
                f"[ETH3D] {scene}: kept_frames={len(kept_frames)} pairs={len(pairs)} "
                f"avg_valid={avg_valid:.1f} (use_undistorted={args.use_undistorted})"
            )

        if args.write_global_pairs:
            for f in kept_frames:
                global_frames.append((scene, f))
            for a, b in pairs:
                global_pairs.append((scene, a, b))

    if args.write_global_pairs:
        np.savez(
            osp.join(out_dir, "all_pairs.npz"),
            pairs=np.array(global_pairs, dtype=object),
            frames=np.array(global_frames, dtype=object),
        )
        print(f"[ETH3D] wrote global pairs: {osp.join(out_dir,'all_pairs.npz')} num_pairs={len(global_pairs)}")


if __name__ == "__main__":
    main()
