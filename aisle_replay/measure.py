"""测量留出合成回放上的三个人数：人的召回、1 m 箱子距离误差、检测前向毫秒。"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from aisle_replay.backproject import range_from_mask
from aisle_replay.camera import default_camera
from aisle_replay.clips import build_clip
from aisle_replay.detect import Yolo11SegDetector
from aisle_replay.replay import replay_clip
from aisle_replay.safety_gate import SafetyGate
from aisle_replay.synth import LABEL_PERSON, LABEL_TOTE, person_box, render, tote_box
from aisle_replay.train_synth import DEFAULT_WEIGHTS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "metrics.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_tote_errors_cm() -> list[float]:
    camera = default_camera()
    clip = build_clip("tote_1m", camera)
    errors = []
    for frame in clip.frames:
        mask = frame.label == LABEL_TOTE
        range_m, _lateral = range_from_mask(frame.depth_mm, mask, camera)
        if range_m is None:
            raise RuntimeError("1 m 周转箱的真值掩码没有反投影出距离")
        errors.append(abs(range_m - 1.0) * 100.0)
    return errors


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def held_out_person_frames(count: int = 24, seed: int = 2026):
    """种子和训练集（7 / 8）不同，帧不写入 data/。"""
    rng = np.random.default_rng(seed)
    camera = default_camera()
    frames = []
    while len(frames) < count:
        front = float(rng.uniform(1.4, 4.0))
        y = float(rng.uniform(-0.35, 0.35))
        objects = [person_box(front, y)]
        if float(rng.random()) < 0.5:
            objects.append(tote_box(float(rng.uniform(0.8, 2.5)), float(rng.uniform(-0.4, 0.4))))
        rendered = render(objects, camera)
        if int((rendered.label == LABEL_PERSON).sum()) < 40:
            continue
        frames.append(rendered)
    return frames


def measure(weights: Path | None = None) -> dict:
    geometry = geometry_tote_errors_cm()
    report = {
        "data_source": "aisle_replay/synth.py 生成的合成 RGB-D 回放",
        "vehicle": False,
        "verb": "replay",
        "image_hw": [240, 320],
        "device": "cpu",
        "model": "YOLO11-seg",
        "tote_1m_geometry_mae_cm": float(np.mean(geometry)),
        "tote_1m_geometry_max_cm": float(np.max(geometry)),
        "tote_1m_geometry_frames": len(geometry),
        "missing": [],
    }
    weights = weights or DEFAULT_WEIGHTS
    if not weights.exists():
        report["missing"] = [
            "person_recall",
            "tote_1m_detector_mae_cm",
            "detector_predict_ms",
        ]
        report["missing_reason"] = f"没有权重文件 {weights}"
        return report

    detector = Yolo11SegDetector(str(weights), conf=0.25, imgsz=320)
    camera = default_camera()
    frames = held_out_person_frames()
    hits = 0
    for frame in frames:
        gt = frame.label == LABEL_PERSON
        detections = detector.detect(frame.rgb)
        person_masks = [det.mask for det in detections if det.class_name == "PERSON"]
        if any(_iou(mask, gt) >= 0.5 for mask in person_masks):
            hits += 1
    report["person_recall"] = hits / len(frames)
    report["person_recall_hits"] = hits
    report["person_recall_positives"] = len(frames)
    report["person_recall_iou"] = 0.5
    report["person_recall_seed"] = 2026

    clip = build_clip("tote_1m", camera)
    det_errors = []
    detected = 0
    for frame in clip.frames:
        detections = detector.detect(frame.rgb)
        masks = [det.mask for det in detections if det.class_name == "TOTE"]
        if not masks:
            continue
        detected += 1
        # 一张图里可能有多个 TOTE，取和真值掩码 IoU 最高的那个。
        gt = frame.label == LABEL_TOTE
        mask = max(masks, key=lambda item: _iou(item, gt))
        range_m, _lateral = range_from_mask(frame.depth_mm, mask, camera)
        if range_m is None:
            continue
        det_errors.append(abs(range_m - 1.0) * 100.0)
    report["tote_1m_detector_frames"] = len(clip.frames)
    report["tote_1m_detector_hits"] = detected
    if det_errors:
        report["tote_1m_detector_mae_cm"] = float(np.mean(det_errors))
        report["tote_1m_detector_max_cm"] = float(np.max(det_errors))
    else:
        report["missing"].append("tote_1m_detector_mae_cm")

    sample = frames[0].rgb
    for _ in range(3):
        detector.detect(sample)
    forward_ms = []
    predict_ms = []
    for _ in range(15):
        detector.detect(sample)
        forward_ms.append(detector.last_forward_ms)
        predict_ms.append(detector.last_predict_ms)
    report["detector_forward_ms_median"] = float(np.median(forward_ms))
    report["detector_forward_ms_min"] = float(np.min(forward_ms))
    report["detector_predict_ms_median"] = float(np.median(predict_ms))
    report["detector_predict_repeats"] = len(forward_ms)
    report["detector_forward_includes"] = (
        "Ultralytics 记的单张 inference 毫秒，不含预处理和 NMS。CPU，不是 TensorRT。"
    )
    report["detector_predict_includes"] = "同一次调用的整段 predict 墙钟，含预处理、前向、NMS 和掩码。"

    behavior = {}
    for name in ("empty", "person", "tote", "pallet"):
        clip = build_clip(name, camera)
        events = replay_clip(clip, detector, SafetyGate())
        behavior[name] = {
            "expected": clip.expected_reasons,
            "actual": [event.reason for event in events],
        }
    report["yolo_replay"] = behavior

    person = build_clip("person", camera)
    events = replay_clip(person, detector, SafetyGate())
    enter = person.person_enter_frame
    stop_frames = [event.frame for event in events if event.reason == "STOP_PERSON"]
    report["person_clip_expected"] = person.expected_reasons
    report["person_clip_reasons"] = [event.reason for event in events]
    report["person_enter_frame"] = enter
    if stop_frames and enter is not None:
        report["person_stop_frame"] = min(stop_frames)
        report["person_stop_delay_frames"] = min(stop_frames) - enter
    else:
        report["person_stop_frame"] = None
        report["person_stop_delay_frames"] = None
        report["missing"].append("person_stop_frame")

    report["weights"] = str(weights)
    report["weights_sha256"] = _sha256(weights)
    report["conf"] = 0.25
    return report


def write_report(report: dict, path: Path = RESULTS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(weights: str | None = None) -> int:
    path = Path(weights) if weights else None
    t0 = time.perf_counter()
    report = measure(path)
    report["measure_wall_s"] = round(time.perf_counter() - t0, 3)
    out = write_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0
