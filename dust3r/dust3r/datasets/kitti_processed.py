import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class KITTIDust3RProcessed(BaseStereoViewDataset):
    """
    Expected layout:
      ROOT/images/<stem>.jpg
      ROOT/depths/<stem>.exr
      ROOT/cams/<stem>.npz (intrinsics[3,3], pose[4,4])
      ROOT/pairs.npz (pairs: [stem1, stem2])
    """
    def __init__(self, *args, ROOT, split="test", max_pairs=None, **kwargs):
        self.ROOT = ROOT
        self.max_pairs = max_pairs
        super().__init__(*args, split=split, **kwargs)
        self.pairs = self._load_pairs()
        if self.max_pairs is not None:
            self.pairs = self.pairs[: int(self.max_pairs)]
        print(f"[KITTIDust3RProcessed] pairs={len(self.pairs)}")

    def _load_pairs(self):
        path = osp.join(self.ROOT, "pairs.npz")
        if not osp.isfile(path):
            raise FileNotFoundError(f"Missing {path}")
        with np.load(path, allow_pickle=True) as data:
            arr = data["pairs"]
        return [(str(a), str(b)) for a, b in arr]

    def __len__(self):
        return len(self.pairs)

    def _scene_from_stem(self, stem):
        if "_frame_" in stem:
            return stem.split("_frame_")[0]
        return "kitti"

    def _load_view(self, stem, resolution, rng, idx):
        img = imread_cv2(osp.join(self.ROOT, "images", stem + ".jpg"))
        depth = imread_cv2(osp.join(self.ROOT, "depths", stem + ".exr"))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0

        cam_path = osp.join(self.ROOT, "cams", stem + ".npz")
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
            img, depth, intr, resolution, rng=rng, info=stem
        )

        scene = self._scene_from_stem(stem)
        return dict(
            img=img,
            depthmap=depth.astype(np.float32),
            camera_pose=pose.astype(np.float32),
            camera_intrinsics=intr.astype(np.float32),
            dataset="KITTI",
            label=stem,
            instance=f"{idx}_{stem}",
            scene=scene,
        )

    def _get_views(self, idx, resolution, rng):
        s1, s2 = self.pairs[idx]
        v1 = self._load_view(s1, resolution, rng, idx)
        v2 = self._load_view(s2, resolution, rng, idx)
        return [v1, v2]
