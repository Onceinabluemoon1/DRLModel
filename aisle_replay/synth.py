"""合成通道 RGB-D。深度是相机测量的 z（毫米），不是检测框高度。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aisle_replay.camera import Camera, default_camera

LABEL_BG = 0
LABEL_PERSON = 1
LABEL_TOTE = 2
LABEL_PALLET = 3
LABEL_WALL = 4
LABEL_DISTRACTOR = 5

YOLO_FROM_LABEL = {
    LABEL_PERSON: 0,
    LABEL_TOTE: 1,
    LABEL_PALLET: 2,
}
CLASS_FROM_LABEL = {
    LABEL_PERSON: "PERSON",
    LABEL_TOTE: "TOTE",
    LABEL_PALLET: "PALLET",
}

# 背景、人、周转箱、托盘、墙、干扰物。干扰物不进标注。
_COLORS = np.array(
    [
        [176, 176, 172],
        [214, 48, 48],
        [232, 144, 28],
        [36, 92, 204],
        [86, 88, 92],
        [52, 140, 72],
    ],
    dtype=np.int16,
)

WALL_X = 7.0


@dataclass(frozen=True)
class Aabb:
    """base_link 下的轴对齐盒子。xmin 是朝向车的前表面。"""

    class_name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def label_id(self) -> int:
        return {
            "PERSON": LABEL_PERSON,
            "TOTE": LABEL_TOTE,
            "PALLET": LABEL_PALLET,
            "DISTRACTOR": LABEL_DISTRACTOR,
        }[self.class_name]

    @property
    def center_y(self) -> float:
        return 0.5 * (self.ymin + self.ymax)


def person_box(front_x: float, y_center: float = 0.0) -> Aabb:
    return Aabb("PERSON", front_x, front_x + 0.28, y_center - 0.22, y_center + 0.22, 0.0, 1.72)


def tote_box(front_x: float, y_center: float = 0.0) -> Aabb:
    return Aabb("TOTE", front_x, front_x + 0.46, y_center - 0.28, y_center + 0.28, 0.0, 0.36)


def pallet_box(front_x: float, y_center: float = 0.0) -> Aabb:
    return Aabb("PALLET", front_x, front_x + 0.60, y_center - 0.42, y_center + 0.42, 0.0, 0.22)


@dataclass
class Rendered:
    rgb: np.ndarray
    depth_mm: np.ndarray
    label: np.ndarray
    objects: list[Aabb]


def _slab_parameter(
    origin: np.ndarray,
    direction: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> np.ndarray:
    """射线 origin + z * direction 与 AABB 的最近正交深度。未击中为 +inf。"""
    shape = direction.shape[:-1]
    tnear = np.full(shape, -np.inf, dtype=np.float64)
    tfar = np.full(shape, np.inf, dtype=np.float64)
    for axis in range(3):
        d = direction[..., axis]
        o = float(origin[axis])
        lo = float(bmin[axis])
        hi = float(bmax[axis])
        parallel = np.abs(d) < 1e-8
        outside = parallel & ((o < lo) | (o > hi))
        d_ok = np.where(parallel, 1.0, d)
        t1 = (lo - o) / d_ok
        t2 = (hi - o) / d_ok
        t_small = np.minimum(t1, t2)
        t_big = np.maximum(t1, t2)
        tnear = np.maximum(tnear, np.where(parallel, -np.inf, t_small))
        tfar = np.minimum(tfar, np.where(parallel, np.inf, t_big))
        tfar = np.where(outside, -np.inf, tfar)
    hit = (tfar >= tnear) & (tfar > 1e-4)
    zhit = np.where(tnear > 1e-4, tnear, tfar)
    return np.where(hit, zhit, np.inf)


def render(
    objects: list[Aabb],
    camera: Camera | None = None,
    rng: np.random.Generator | None = None,
) -> Rendered:
    camera = camera or default_camera()
    direction = camera.ray_directions()
    origin = camera.translation_base
    best_z = np.full(direction.shape[:-1], np.inf, dtype=np.float64)
    best_id = np.zeros(direction.shape[:-1], dtype=np.uint8)

    dz = direction[..., 2]
    floor_ok = dz < -1e-6
    z_floor = np.where(floor_ok, -float(origin[2]) / np.where(floor_ok, dz, 1.0), np.inf)
    floor_p = origin + z_floor[..., None] * direction
    floor_on = (
        floor_ok
        & (floor_p[..., 0] > 0.05)
        & (floor_p[..., 0] < WALL_X)
        & (np.abs(floor_p[..., 1]) < 4.0)
        & (z_floor > 1e-4)
    )
    z_floor = np.where(floor_on, z_floor, np.inf)
    closer = z_floor < best_z
    best_z = np.where(closer, z_floor, best_z)
    best_id = np.where(closer, LABEL_BG, best_id)

    dx = direction[..., 0]
    wall_ok = dx > 1e-6
    z_wall = np.where(wall_ok, (WALL_X - float(origin[0])) / np.where(wall_ok, dx, 1.0), np.inf)
    wall_p = origin + z_wall[..., None] * direction
    wall_on = (
        wall_ok
        & (wall_p[..., 2] >= 0.0)
        & (wall_p[..., 2] <= 3.0)
        & (np.abs(wall_p[..., 1]) < 4.0)
        & (z_wall > 1e-4)
    )
    z_wall = np.where(wall_on, z_wall, np.inf)
    closer = z_wall < best_z
    best_z = np.where(closer, z_wall, best_z)
    best_id = np.where(closer, LABEL_WALL, best_id)

    for obj in objects:
        bmin = np.array([obj.xmin, obj.ymin, obj.zmin], dtype=np.float64)
        bmax = np.array([obj.xmax, obj.ymax, obj.zmax], dtype=np.float64)
        zhit = _slab_parameter(origin, direction, bmin, bmax)
        closer = zhit < best_z
        best_z = np.where(closer, zhit, best_z)
        best_id = np.where(closer, obj.label_id, best_id)

    valid = np.isfinite(best_z)
    depth_m = np.where(valid, best_z, 0.0)
    depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
    label = best_id.copy()
    label[~valid] = LABEL_BG
    # 墙和干扰物只上色，不作为 PERSON / TOTE / PALLET 真值。
    label[(label == LABEL_WALL) | (label == LABEL_DISTRACTOR)] = LABEL_BG

    color = _COLORS[best_id]
    if rng is not None:
        gain = rng.uniform(0.86, 1.14, size=(1, 1, 3)).astype(np.float64)
        color = color.astype(np.float64) * gain
    else:
        color = color.astype(np.float64)
    yy, xx = np.mgrid[0 : camera.height, 0 : camera.width]
    grain = ((xx * 13 + yy * 7) % 17).astype(np.float64) - 8.0
    rgb = np.clip(color + grain[..., None], 0, 255).astype(np.uint8)
    return Rendered(rgb=rgb, depth_mm=depth_mm, label=label, objects=list(objects))
