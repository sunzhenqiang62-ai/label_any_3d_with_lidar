from PIL import Image

from dataset_model.BaseScene import BaseScene
from geometry.lidar_depth import load_calib


class PointCloudScene(BaseScene):
    """Scene with RGB image, world/sensor LiDAR path, and camera calibration."""

    def __init__(self, img_path, pointcloud_path, calib_path):
        super().__init__(Image.open(img_path))
        self.pointcloud_path = pointcloud_path
        self.calib_path = calib_path
        calib = load_calib(calib_path)
        self.K = calib["K"]
        self.c2w = calib["c2w"]
        self.world_to_cam = calib["world_to_cam"]
        self.calib = calib
