"""针孔相机：base_link 前方为 x，左侧为 y，上方为 z。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH = 320
HEIGHT = 240
FX = 260.0
FY = 260.0
PITCH_DEG = 27.0
# 相机在 base_link 下的位置：前方 0.20 m，高 0.55 m。
CAM_X = 0.20
CAM_Y = 0.0
CAM_Z = 0.55


def _pitch_rotation(pitch_deg: float) -> np.ndarray:
    """相机光轴相对水平面向下俯仰。列是相机轴在 base_link 中的方向。"""
    theta = np.deg2rad(pitch_deg)
    s = float(np.sin(theta))
    c = float(np.cos(theta))
    # theta = 0 时：相机 x 向右（base -y），y 向下（base -z），z 向前（base +x）。
    return np.array(
        [
            [0.0, -s, c],
            [-1.0, 0.0, 0.0],
            [0.0, -c, -s],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation_base_from_cam: np.ndarray
    translation_base: np.ndarray

    def ray_directions(self) -> np.ndarray:
        """每个像素一条射线，参数 z 就是相机坐标系下的深度（米）。"""
        us = np.arange(self.width, dtype=np.float64)
        vs = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
        return d_cam @ self.rotation_base_from_cam.T

    def project(self, points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = np.asarray(points_base, dtype=np.float64) - self.translation_base
        cam = delta @ self.rotation_base_from_cam
        z = cam[:, 2]
        u = self.fx * cam[:, 0] / z + self.cx
        v = self.fy * cam[:, 1] / z + self.cy
        return u, v, z

    def backproject(self, u: np.ndarray, v: np.ndarray, z_m: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        cam = np.stack([x, y, z], axis=-1)
        return cam @ self.rotation_base_from_cam.T + self.translation_base


def default_camera() -> Camera:
    return Camera(
        width=WIDTH,
        height=HEIGHT,
        fx=FX,
        fy=FY,
        cx=(WIDTH - 1) / 2.0,
        cy=(HEIGHT - 1) / 2.0,
        rotation_base_from_cam=_pitch_rotation(PITCH_DEG),
        translation_base=np.array([CAM_X, CAM_Y, CAM_Z], dtype=np.float64),
    )
