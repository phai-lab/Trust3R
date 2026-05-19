import os
import os.path as osp
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Tricky24(BaseStereoViewDataset):
    """
    Expected layout (scene folder):
      <scene>/camera_00/im*.png
      <scene>/camera_02/im*.png
      <scene>/disp_00.npy
      <scene>/disp_02.npy
      <scene>/calib_00-02.xml
      <scene>/mask_00.png (optional)
      <scene>/mask_02.png (optional)

    ROOT can be:
      - a parent folder containing many scenes
      - a single scene folder
    """

    def __init__(
        self,
        *args,
        ROOT: str,
        split: Optional[str] = None,
        scenes: Optional[str] = None,
        max_pairs: Optional[int] = None,
        use_mask: bool = True,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.max_pairs = max_pairs
        self.use_mask = bool(use_mask)
        self._calib_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._disp_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._mask_cache: Dict[str, Dict[str, Optional[np.ndarray]]] = {}

        self.scenes = self._discover_scenes(scenes)
        self.scene_paths = {name: path for name, path in self.scenes}

        super().__init__(*args, split=split, **kwargs)
        self.pairs = self._build_pairs()
        if self.max_pairs is not None:
            self.pairs = self.pairs[: int(self.max_pairs)]
        print(f"[Tricky24] scenes={len(self.scenes)} pairs={len(self.pairs)}")

    def _discover_scenes(self, scenes: Optional[str]) -> List[Tuple[str, str]]:
        root = self.ROOT
        cam0 = osp.join(root, "camera_00")
        cam2 = osp.join(root, "camera_02")
        if osp.isdir(cam0) and osp.isdir(cam2):
            name = osp.basename(osp.normpath(root))
            return [(name, root)]

        scene_filter = None
        if scenes:
            scene_filter = {s.strip() for s in scenes.split(",") if s.strip()}

        out = []
        for name in sorted(os.listdir(root)):
            path = osp.join(root, name)
            if not osp.isdir(path):
                continue
            if scene_filter and name not in scene_filter:
                continue
            if osp.isdir(osp.join(path, "camera_00")) and osp.isdir(osp.join(path, "camera_02")):
                out.append((name, path))
        if scene_filter and not out:
            raise FileNotFoundError(f"No scenes matched {scene_filter} under {root}")
        if not out:
            raise FileNotFoundError(f"No scenes found under {root}")
        return out

    def _list_frames(self, cam_dir: str) -> List[str]:
        exts = {".png", ".jpg", ".jpeg"}
        frames = [f for f in os.listdir(cam_dir) if osp.splitext(f)[1].lower() in exts]
        frames.sort()
        return frames

    def _build_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for scene, path in self.scenes:
            cam0 = osp.join(path, "camera_00")
            cam2 = osp.join(path, "camera_02")
            frames0 = set(self._list_frames(cam0))
            frames2 = set(self._list_frames(cam2))
            frames = sorted(frames0 & frames2)
            if not frames:
                raise FileNotFoundError(f"No matching frames in {cam0} and {cam2}")
            for frame in frames:
                pairs.append((scene, frame))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _read_opencv_xml(self, xml_path: str, name: str) -> Optional[np.ndarray]:
        root = ET.parse(xml_path).getroot()
        elem = root.find(name)
        if elem is None:
            return None
        rows = int(elem.find("rows").text)
        cols = int(elem.find("cols").text)
        data = np.fromstring(elem.find("data").text.strip(), sep=" ")
        return data.reshape(rows, cols)

    def _load_calib(self, scene: str) -> Dict[str, np.ndarray]:
        if scene in self._calib_cache:
            return self._calib_cache[scene]

        scene_path = self.scene_paths[scene]
        xml_path = osp.join(scene_path, "calib_00-02.xml")
        if not osp.isfile(xml_path):
            raise FileNotFoundError(f"Missing calib file: {xml_path}")

        P1 = self._read_opencv_xml(xml_path, "proj_matL")
        P2 = self._read_opencv_xml(xml_path, "proj_matR")
        K1 = self._read_opencv_xml(xml_path, "mtxL")
        K2 = self._read_opencv_xml(xml_path, "mtxR")

        if P1 is not None and P1.shape == (3, 4):
            K_left = P1[:, :3].astype(np.float32)
        elif K1 is not None:
            K_left = K1.astype(np.float32)
        else:
            raise ValueError(f"Missing intrinsics in {xml_path}")

        if P2 is not None and P2.shape == (3, 4):
            K_right = P2[:, :3].astype(np.float32)
        elif K2 is not None:
            K_right = K2.astype(np.float32)
        else:
            raise ValueError(f"Missing right intrinsics in {xml_path}")

        baseline = None
        if P1 is not None and P2 is not None:
            fx = float(P1[0, 0])
            baseline = -float(P2[0, 3]) / fx if fx != 0 else None
        if baseline is None:
            raise ValueError(f"Cannot derive baseline from {xml_path}")

        pose_left = np.eye(4, dtype=np.float32)
        pose_right = np.eye(4, dtype=np.float32)
        pose_right[0, 3] = float(baseline)

        calib = {
            "K_left": K_left,
            "K_right": K_right,
            "baseline": np.float32(baseline),
            "pose_left": pose_left,
            "pose_right": pose_right,
        }
        self._calib_cache[scene] = calib
        return calib

    def _load_disparity(self, scene: str, cam_id: str) -> np.ndarray:
        if scene in self._disp_cache and cam_id in self._disp_cache[scene]:
            return self._disp_cache[scene][cam_id]

        scene_path = self.scene_paths[scene]
        disp_path = osp.join(scene_path, f"disp_{cam_id}.npy")
        if not osp.isfile(disp_path):
            raise FileNotFoundError(f"Missing disparity: {disp_path}")
        disp = np.load(disp_path).astype(np.float32)

        if scene not in self._disp_cache:
            self._disp_cache[scene] = {}
        self._disp_cache[scene][cam_id] = disp
        return disp

    def _load_mask(self, scene: str, cam_id: str) -> Optional[np.ndarray]:
        if not self.use_mask:
            return None
        if scene in self._mask_cache and cam_id in self._mask_cache[scene]:
            return self._mask_cache[scene][cam_id]

        scene_path = self.scene_paths[scene]
        mask_path = osp.join(scene_path, f"mask_{cam_id}.png")
        mask = None
        if osp.isfile(mask_path):
            m = imread_cv2(mask_path)
            if m.ndim == 3:
                m = m[..., 0]
            mask = (m > 0)

        if scene not in self._mask_cache:
            self._mask_cache[scene] = {}
        self._mask_cache[scene][cam_id] = mask
        return mask

    def _depth_from_disparity(self, disp: np.ndarray, fx: float, baseline: float) -> np.ndarray:
        depth = np.zeros_like(disp, dtype=np.float32)
        m = disp > 0
        depth[m] = (fx * baseline) / disp[m]
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0
        return depth

    def _load_view(self, scene: str, frame: str, cam_id: str, resolution, rng, idx):
        scene_path = self.scene_paths[scene]
        img_path = osp.join(scene_path, f"camera_{cam_id}", frame)
        img = imread_cv2(img_path)

        calib = self._load_calib(scene)
        disp = self._load_disparity(scene, cam_id)
        fx = float(calib["K_left"][0, 0]) if cam_id == "00" else float(calib["K_right"][0, 0])
        depth = self._depth_from_disparity(disp, fx, float(calib["baseline"]))

        mask = self._load_mask(scene, cam_id)
        if mask is not None:
            depth = depth.copy()
            depth[~mask] = 0.0

        intr = calib["K_left"] if cam_id == "00" else calib["K_right"]
        pose = calib["pose_left"] if cam_id == "00" else calib["pose_right"]

        img, depth, intr = self._crop_resize_if_necessary(
            img, depth, intr, resolution, rng=rng, info=f"{scene}/{frame}/{cam_id}"
        )

        return dict(
            img=img,
            depthmap=depth.astype(np.float32),
            camera_pose=pose.astype(np.float32),
            camera_intrinsics=intr.astype(np.float32),
            dataset="Tricky24",
            label=f"{scene}_{frame}_{cam_id}",
            instance=f"{idx}_{frame}_{cam_id}",
            scene=scene,
        )

    def _get_views(self, idx, resolution, rng):
        scene, frame = self.pairs[idx]
        v1 = self._load_view(scene, frame, "00", resolution, rng, idx)
        v2 = self._load_view(scene, frame, "02", resolution, rng, idx)
        return [v1, v2]
