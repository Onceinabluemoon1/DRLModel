"""回放入口。每一帧：检测 → 深度反投影 → 安全门。不连接速度进程。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from aisle_replay.backproject import range_from_mask
from aisle_replay.camera import default_camera
from aisle_replay.clips import Clip, build_clip
from aisle_replay.detect import FixtureDetector, RawDetection, Yolo11SegDetector
from aisle_replay.safety_gate import GateCommand, InstanceRange, SafetyGate


@dataclass
class FrameEvent:
    frame: int
    reason: str
    vx_m_s: float
    wz_rad_s: float
    t_wall: float
    instances: list[dict]

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "reason": self.reason,
            "vx_m_s": self.vx_m_s,
            "wz_rad_s": self.wz_rad_s,
            "t_wall": self.t_wall,
            "instances": self.instances,
        }


def instances_from_detections(
    depth_mm,
    detections: list[RawDetection],
    camera,
) -> list[InstanceRange]:
    instances = []
    for det in detections:
        # 距离只来自掩码里的深度。这里不读取检测框的宽高。
        range_m, lateral_m = range_from_mask(depth_mm, det.mask, camera)
        instances.append(
            InstanceRange(det.class_name, det.score, range_m, lateral_m)
        )
    return instances


def replay_clip(
    clip: Clip,
    detector,
    gate: SafetyGate | None = None,
    frame_sleep_s: float = 0.0,
    release_path: Path | None = None,
) -> list[FrameEvent]:
    gate = gate or SafetyGate()
    camera = default_camera()
    if release_path is not None:
        while not release_path.exists():
            time.sleep(0.02)
    events = []
    for index, frame in enumerate(clip.frames):
        if frame_sleep_s > 0:
            time.sleep(frame_sleep_s)
        detections = detector.detect(frame.rgb, frame.label)
        instances = instances_from_detections(frame.depth_mm, detections, camera)
        command: GateCommand = gate.evaluate(instances)
        events.append(
            FrameEvent(
                frame=index,
                reason=command.reason,
                vx_m_s=command.vx_m_s,
                wz_rad_s=command.wz_rad_s,
                t_wall=time.time(),
                instances=[
                    {
                        "class_name": item.class_name,
                        "score": item.score,
                        "range_m": item.range_m,
                        "lateral_m": item.lateral_m,
                    }
                    for item in instances
                ],
            )
        )
    return events


def write_events(events: list[FrameEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
            handle.flush()


def load_detector(kind: str, weights: str | None, conf: float):
    if kind == "fixture":
        return FixtureDetector()
    if kind == "yolo":
        if not weights:
            raise SystemExit("yolo 检测器需要 --weights")
        return Yolo11SegDetector(weights, conf=conf)
    raise SystemExit(f"未知检测器 {kind}")


def run_replay_cli(args) -> int:
    if args.ready:
        ready = Path(args.ready)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text("ready\n", encoding="utf-8")
    clip = build_clip(args.clip)
    detector = load_detector(args.detector, args.weights, args.conf)
    release = Path(args.release) if args.release else None
    events = replay_clip(
        clip,
        detector,
        frame_sleep_s=args.frame_sleep_ms / 1000.0,
        release_path=release,
    )
    mismatched = [
        event.frame
        for event in events
        if event.reason != clip.expected_reasons[event.frame]
    ]
    payload = {
        "clip": clip.name,
        "description": clip.description,
        "detector": args.detector,
        "expected_reasons": clip.expected_reasons,
        "reasons": [event.reason for event in events],
        "mismatched_frames": mismatched,
        "person_enter_frame": clip.person_enter_frame,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.events:
        write_events(events, Path(args.events))
        summary = Path(args.events).with_suffix(".summary.json")
        summary.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not mismatched else 1
