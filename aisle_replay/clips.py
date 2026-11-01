"""四段回放，外加一段 1 m 周转箱核对。期望原因码来自盒子的几何位置。"""

from __future__ import annotations

from dataclasses import dataclass

from aisle_replay.camera import Camera, default_camera
from aisle_replay.safety_gate import load_rules
from aisle_replay.synth import Aabb, Rendered, pallet_box, person_box, render, tote_box


@dataclass
class Clip:
    name: str
    description: str
    frames: list[Rendered]
    expected_reasons: list[str]

    @property
    def person_enter_frame(self) -> int | None:
        for i, reason in enumerate(self.expected_reasons):
            if reason == "STOP_PERSON":
                return i
        return None


def geometric_reason(objects: list[Aabb]) -> str:
    """用前表面的米制位置写期望原因码，不经过检测器。"""
    rules = load_rules()
    for obj in objects:
        if obj.class_name != "PERSON":
            continue
        if (
            rules.forward_min_m < obj.xmin <= rules.person_stop_m
            and abs(obj.center_y) <= rules.lane_half_width_m
        ):
            return "STOP_PERSON"
    for obj in objects:
        if obj.class_name not in ("TOTE", "PALLET"):
            continue
        if (
            rules.forward_min_m < obj.xmin <= rules.object_stop_m
            and abs(obj.center_y) <= rules.lane_half_width_m
        ):
            return "STOP_DISTANCE"
    return "OK"


def _clip_from_specs(
    name: str,
    description: str,
    specs: list[list[Aabb]],
    camera: Camera | None = None,
) -> Clip:
    frames = [render(objects, camera) for objects in specs]
    reasons = [geometric_reason(objects) for objects in specs]
    return Clip(name, description, frames, reasons)


def build_clip(name: str, camera: Camera | None = None) -> Clip:
    camera = camera or default_camera()
    if name == "empty":
        return _clip_from_specs(
            "empty",
            "空通道",
            [[], [], []],
            camera,
        )
    if name == "person":
        return _clip_from_specs(
            "person",
            "人从停止区外走进 2.5 m 前方停止区",
            [[person_box(front, 0.0)] for front in (3.4, 2.9, 2.2, 1.7)],
            camera,
        )
    if name == "tote":
        return _clip_from_specs(
            "tote",
            "车道内周转箱靠近到 1.0 m 以内；最后一帧箱子在车道外，仍能被看到",
            [
                [tote_box(1.8, 0.0)],
                [tote_box(1.35, 0.0)],
                [tote_box(0.75, 0.0)],
                [tote_box(2.30, 1.15)],
            ],
            camera,
        )
    if name == "pallet":
        return _clip_from_specs(
            "pallet",
            "车道内托盘靠近到 1.0 m 以内；最后一帧托盘在车道外，仍能被看到",
            [
                [pallet_box(1.7, 0.0)],
                [pallet_box(0.80, 0.0)],
                [pallet_box(2.20, -1.15)],
            ],
            camera,
        )
    if name == "tote_1m":
        return _clip_from_specs(
            "tote_1m",
            "周转箱前表面固定在 base_link 前方 1.00 m，用来量距离误差",
            [[tote_box(1.00, 0.0)] for _ in range(5)],
            camera,
        )
    raise KeyError(name)


def all_behavior_clips() -> list[Clip]:
    return [build_clip(name) for name in ("empty", "person", "tote", "pallet")]
