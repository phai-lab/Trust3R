import os
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ETH3DProcessedDust3R(BaseStereoViewDataset):
    """
    Expected layout:
      ROOT/all_pairs.npz (optional; pairs: [scene, f1, f2])
      ROOT/<scene>/images/<frame>.jpg
      ROOT/<scene>/depths/<frame>.exr
      ROOT/<scene>/cams/<frame>.npz (intrinsics[3,3], cam_to_world[4,4] or world_to_cam)
      ROOT/<scene>/pairs.npz (fallback if no all_pairs.npz)
    """
    def __init__(self, *args, ROOT, split="test", max_pairs=None, **kwargs):
        self.ROOT = ROOT
        self.max_pairs = max_pairs
        super().__init__(*args, split=split, **kwargs)
        self.pairs = self._load_pairs()
        if self.max_pairs is not None:
            self.pairs = self.pairs[: int(self.max_pairs)]
        print(f"[ETH3DProcessedDust3R] pairs={len(self.pairs)}")

    def _load_pairs(self):
        pairs_path = osp.join(self.ROOT, "all_pairs.npz")
        pairs = []
        if osp.isfile(pairs_path):
            with np.load(pairs_path, allow_pickle=True) as data:
                arr = data["pairs"]
            for scene, f1, f2 in arr:
                pairs.append((str(scene), str(f1), str(f2)))
            return pairs

        scenes = sorted([d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))])
        for scene in scenes:
            p = osp.join(self.ROOT, scene, "pairs.npz")
            if not osp.isfile(p):
                continue
            with np.load(p, allow_pickle=True) as data:
                arr = data["pairs"]
            for f1, f2 in arr:
                pairs.append((scene, str(f1), str(f2)))
                if self.max_pairs is not None and len(pairs) >= int(self.max_pairs):
                    return pairs
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_view(self, scene, frame, resolution, rng, idx):
        scene_dir = osp.join(self.ROOT, scene)
        img = imread_cv2(osp.join(scene_dir, "images", frame + ".jpg"))
        depth = imread_cv2(osp.join(scene_dir, "depths", frame + ".exr"))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0

        cam_path = osp.join(scene_dir, "cams", frame + ".npz")
        with np.load(cam_path) as cam:
            intr = cam["intrinsics"].astype(np.float32)
            if "cam_to_world" in cam:
                pose = cam["cam_to_world"].astype(np.float32)
            elif "pose" in cam:
                pose = cam["pose"].astype(np.float32)
            elif "world_to_cam" in cam:
                pose = np.linalg.inv(cam["world_to_cam"]).astype(np.float32)
            else:
                pose = np.eye(4, dtype=np.float32)

        img, depth, intr = self._crop_resize_if_necessary(
            img, depth, intr, resolution, rng=rng, info=f"{scene}/{frame}"
        )

        return dict(
            img=img,
            depthmap=depth.astype(np.float32),
            camera_pose=pose.astype(np.float32),
            camera_intrinsics=intr.astype(np.float32),
            dataset="ETH3D",
            label=f"{scene}_{frame}",
            instance=f"{idx}_{frame}",
            scene=scene,
        )

    def _get_views(self, idx, resolution, rng):
        scene, f1, f2 = self.pairs[idx]
        v1 = self._load_view(scene, f1, resolution, rng, idx)
        v2 = self._load_view(scene, f2, resolution, rng, idx)
        return [v1, v2]
