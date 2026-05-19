import os.path as osp
import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

class TUMRGBD(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        with np.load(osp.join(self.ROOT, "all_metadata.npz")) as data:
            self.scenes = data["scenes"]
            self.sceneids = data["sceneids"]
            self.images = data["images"]
            self.intrinsics = data["intrinsics"].astype(np.float32)
            self.trajectories = data["trajectories"].astype(np.float32)
            self.pairs = data["pairs"][:, :2].astype(int)

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        i1, i2 = self.pairs[idx]
        views = []
        for view_idx in [i1, i2]:
            sid = int(self.sceneids[view_idx])
            scene = self.scenes[sid]
            basename = str(self.images[view_idx])

            scene_dir = osp.join(self.ROOT, scene)
            rgb = imread_cv2(osp.join(scene_dir, "images", basename + ".jpg"))
            depth = imread_cv2(osp.join(scene_dir, "depth", basename + ".png"), cv2.IMREAD_UNCHANGED)
            depth = depth.astype(np.float32) / 5000.0  # TUM depth scale
            depth[~np.isfinite(depth)] = 0.0
            depth[depth <= 0] = 0.0

            K = self.intrinsics[view_idx]
            pose = self.trajectories[view_idx]

            rgb, depth, K = self._crop_resize_if_necessary(rgb, depth, K, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb,
                depthmap=depth.astype(np.float32),
                camera_pose=pose.astype(np.float32),
                camera_intrinsics=K.astype(np.float32),
                dataset="TUMRGBD",
                label=f"{scene}_{basename}",
                instance=f"{idx}_{view_idx}",
            ))
        return views
