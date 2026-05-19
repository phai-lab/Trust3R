# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++
# dataset at https://github.com/scannetpp/scannetpp - non-commercial research and educational purposes
# https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf
# See datasets_preprocess/preprocess_scannetpp.py
# --------------------------------------------------------
import json
import os.path as osp
import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, max_scenes=None, scene_whitelist=None,
                 splits_json=None, split_by_scene=True, train_fraction=0.9,
                 disable_scene_split=False, **kwargs):
        self.ROOT = ROOT
        self.max_scenes = max_scenes
        self.scene_whitelist = scene_whitelist
        self.splits_json = splits_json
        self.split_by_scene = split_by_scene
        self.train_fraction = train_fraction
        self.disable_scene_split = disable_scene_split
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()
        if self.splits_json is not None and self.scene_whitelist is None:
            with open(self.splits_json, "r", encoding="utf-8") as handle:
                splits = json.load(handle)
            scenes = splits.get("scenes") if isinstance(splits, dict) else None
            if not isinstance(scenes, dict) or self.split not in scenes:
                available = sorted(scenes.keys()) if isinstance(scenes, dict) else []
                raise ValueError(
                    f"Split '{self.split}' not found in splits_json scenes. "
                    f"Available: {available}"
                )
            self.scene_whitelist = scenes[self.split]
        self._filter_by_scene()
        if self.disable_scene_split:
            print("[ScanNetpp] disable_scene_split=True: skipping internal train/val scene split")
        elif self.split_by_scene and self.splits_json is None and self.scene_whitelist is None:
            self._split_by_scene(train_fraction=self.train_fraction)

    def _load_data(self):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)

    def _filter_by_scene(self):
        if self.scene_whitelist is None and self.max_scenes is None:
            return

        before_pairs = len(self.pairs)
        before_scenes = len(np.unique(self.sceneids))

        keep_scene_ids = None
        if self.scene_whitelist is not None:
            whitelist = set(self.scene_whitelist)
            keep_scene_ids = [sid for sid, name in enumerate(self.scenes) if name in whitelist]
            preview = sorted(whitelist)[:5]
            mode = f"whitelist(n={len(whitelist)}, preview={preview})"
        else:
            keep_scene_ids = []
            seen = set()
            for sid in self.sceneids:
                if sid not in seen:
                    keep_scene_ids.append(sid)
                    seen.add(sid)
                if len(keep_scene_ids) >= int(self.max_scenes):
                    break
            mode = f"max_scenes={self.max_scenes}"

        keep_scene_ids = set(keep_scene_ids)
        pairs_mask = np.array([(self.sceneids[i] in keep_scene_ids) and (self.sceneids[j] in keep_scene_ids)
                               for i, j in self.pairs], dtype=bool)
        self.pairs = self.pairs[pairs_mask]
        if len(self.pairs) > 0:
            kept_scene_ids = {self.sceneids[idx] for pair in self.pairs for idx in pair}
        else:
            kept_scene_ids = set()
        after_scenes = len(kept_scene_ids)

        print(f"[ScanNetpp] scene filter ({mode}): pairs {before_pairs}->{len(self.pairs)}, "
              f"scenes {before_scenes}->{after_scenes}")

    def _split_by_scene(self, train_fraction=0.9):
        if self.split not in ('train', 'val'):
            return

        if len(self.pairs) == 0:
            return

        # determine unique scenes in the current subset in a stable order
        unique_scene_ids = []
        for pair in self.pairs:
            for idx in pair:
                sid = self.sceneids[idx]
                if sid not in unique_scene_ids:
                    unique_scene_ids.append(sid)

        if len(unique_scene_ids) < 2:
            return

        cutoff = int(len(unique_scene_ids) * train_fraction)
        cutoff = min(max(cutoff, 1), len(unique_scene_ids) - 1)
        if self.split == 'train':
            keep_scenes = set(unique_scene_ids[:cutoff])
            split_label = f"train@{train_fraction:.2f}"
        else:
            keep_scenes = set(unique_scene_ids[cutoff:])
            split_label = f"val@{1 - train_fraction:.2f}"

        before_pairs = len(self.pairs)
        pairs_mask = np.array([(self.sceneids[i] in keep_scenes) and (self.sceneids[j] in keep_scenes)
                               for i, j in self.pairs], dtype=bool)
        self.pairs = self.pairs[pairs_mask]

        after_scenes = len(keep_scenes)
        print(f"[ScanNetpp] split={split_label}: pairs {before_pairs}->{len(self.pairs)}, "
              f"scenes {len(unique_scene_ids)}->{after_scenes}")

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        for view_idx in [image_idx1, image_idx2]:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ScanNetpp(
        split='val',
        ROOT="data/scannetpp_processed",
        resolution=224,
        aug_crop=16,
        disable_scene_split=True,
    )

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
