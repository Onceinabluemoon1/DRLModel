"""把深度图反投影到 base_link，取物体近表面的前向距离。"""

from __future__ import annotations

import numpy as np

from aisle_replay.camera import Camera

# 近表面带宽。距离取这条带的中位数，不用检测框像素高度。
_SURFACE_BAND_M = 0.04
_MIN_POINTS = 8


def range_from_mask(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    camera: Camera,
) -> tuple[float | None, float | None]:
    """返回 (前向 x 米, 横向 y 米)。掩码里没有有效深度时两个都是 None。"""
    if mask is None or mask.shape != depth_mm.shape:
        return None, None
    ys, xs = np.nonzero(mask)
    if xs.size < _MIN_POINTS:
        return None, None
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    valid = z > 0.05
    if int(valid.sum()) < _MIN_POINTS:
        return None, None
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    if z.size > 5000:
        step = int(np.ceil(z.size / 5000))
        xs, ys, z = xs[::step], ys[::step], z[::step]
    pts = camera.backproject(xs, ys, z)
    front = pts[:, 0] > float(camera.translation_base[0])
    if int(front.sum()) < _MIN_POINTS:
        return None, None
    pts = pts[front]
    x = pts[:, 0]
    # 丢掉最近的 2% 像素，避免单点跳到相机跟前。
    x_near = float(np.percentile(x, 2))
    band = pts[x <= x_near + _SURFACE_BAND_M]
    if band.shape[0] < _MIN_POINTS:
        band = pts[x <= np.percentile(x, 15)]
    if band.shape[0] == 0:
        return None, None
    return float(np.median(band[:, 0])), float(np.median(band[:, 1]))
