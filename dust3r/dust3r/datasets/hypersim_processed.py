import os
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class HyperSimProcessed(BaseStereoViewDataset):
    """
    Expected layout:
      ROOT/<scene>/<cam_xx>/<frame>_rgb.png
      ROOT/<scene>/<cam_xx>/<frame>_depth.npy
      ROOT/<scene>/<cam_xx>/<frame>_cam.npz   (keys: intrinsics[3,3], pose[4,4])

    We build pairs within same scene+camera by a fixed stride.
    Good enough for quick OOD evaluation / visualization.
    """
    def __init__(
        self,
        *args,
        ROOT,
        cameras=("cam_00",),
        pair_stride=10,
        max_scenes=None,
        scene_whitelist=None,
        max_pairs=None,
        **kwargs
    ):
        self.ROOT = ROOT
        self.cameras = tuple(cameras)
        self.pair_stride = int(pair_stride)
        self.max_scenes = max_scenes
        self.scene_whitelist = set(scene_whitelist) if scene_whitelist else None
        self.max_pairs = max_pairs
        super().__init__(*args, **kwargs)
        self._build_pairs()

    def _build_pairs(self):
        scenes = [d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))]
        scenes = sorted(scenes)
        if self.scene_whitelist:
            scenes = [s for s in scenes if s in self.scene_whitelist]
        if self.max_scenes is not None:
            scenes = scenes[: int(self.max_scenes)]

        pairs = []
        for s in scenes:
            for cam in self.cameras:
                cam_dir = osp.join(self.ROOT, s, cam)
                if not osp.isdir(cam_dir):
                    continue

                rgb_files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_rgb.png")])
                ids = []
                suffix = "_rgb.png"
                for f in rgb_files:
                    stem = f[:-len(suffix)]
                    if stem.isdigit():
                        ids.append((int(stem), stem))
                ids = sorted(ids, key=lambda x: x[0])

                stride = self.pair_stride
                for i in range(0, len(ids) - stride):
                    f1 = ids[i][1]
                    f2 = ids[i + stride][1]
                    pairs.append((s, cam, f1, f2))
                    if self.max_pairs is not None and len(pairs) >= int(self.max_pairs):
                        break
                if self.max_pairs is not None and len(pairs) >= int(self.max_pairs):
                    break
            if self.max_pairs is not None and len(pairs) >= int(self.max_pairs):
                break

        self.pairs = pairs
        print(f"[HyperSimProcessed] scenes={len(scenes)} pairs={len(self.pairs)} stride={self.pair_stride}")

    def __len__(self):
        return len(self.pairs)

    def _load_view(self, scene, cam, fid, resolution, rng):
        fid = str(fid)
        base = osp.join(self.ROOT, scene, cam, fid)
        rgb = imread_cv2(base + "_rgb.png")  # BGR uint8 HWC
        depth = np.load(base + "_depth.npy").astype(np.float32)  # meters, HW

        cam_npz = np.load(base + "_cam.npz")
        intr = cam_npz["intrinsics"].astype(np.float32)
        pose = cam_npz["pose"].astype(np.float32)

        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0

        rgb, depth, intr = self._crop_resize_if_necessary(
            rgb, depth, intr, resolution, rng=rng, info=f"{scene}/{cam}/{fid}"
        )

        return dict(
            img=rgb,
            depthmap=depth.astype(np.float32),
            camera_pose=pose.astype(np.float32),
            camera_intrinsics=intr.astype(np.float32),
            dataset="HyperSim",
            label=f"{scene}_{cam}_{fid}",
            instance=f"{scene}_{cam}_{fid}",
        )

    def _get_views(self, idx, resolution, rng):
        scene, cam, f1, f2 = self.pairs[idx]
        v1 = self._load_view(scene, cam, f1, resolution, rng)
        v2 = self._load_view(scene, cam, f2, resolution, rng)
        return [v1, v2]
