"""检测器。YOLO11-seg 是回放用的检测器；fixture 只喂真值掩码，用来测安全门。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from aisle_replay.synth import CLASS_FROM_LABEL, LABEL_PALLET, LABEL_PERSON, LABEL_TOTE

_NAME_TO_CLASS = {
    "person": "PERSON",
    "tote": "TOTE",
    "pallet": "PALLET",
}


@dataclass
class RawDetection:
    class_name: str
    score: float
    mask: np.ndarray


def map_class_name(raw: str) -> str | None:
    """PERSON 只接受独立类名 person。不把别的类并进来，也没有 OTHER。"""
    return _NAME_TO_CLASS.get(raw.strip().lower())


def _mask_from_label(label: np.ndarray, label_id: int) -> np.ndarray:
    return label == label_id


class FixtureDetector:
    """渲染真值。不是神经网络，不能拿它的召回当检测指标。"""

    name = "fixture"

    def detect(self, rgb: np.ndarray, label: np.ndarray) -> list[RawDetection]:
        del rgb
        found = []
        for label_id in (LABEL_PERSON, LABEL_TOTE, LABEL_PALLET):
            mask = _mask_from_label(label, label_id)
            if int(mask.sum()) < 8:
                continue
            found.append(RawDetection(CLASS_FROM_LABEL[label_id], 1.0, mask))
        return found


def rasterize_polygon(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    if polygon is None or len(polygon) < 3:
        return np.zeros((height, width), dtype=bool)
    flat = [(float(x), float(y)) for x, y in np.asarray(polygon)]
    ImageDraw.Draw(mask).polygon(flat, fill=1)
    return np.asarray(mask, dtype=np.uint8).astype(bool)


class Yolo11SegDetector:
    """Ultralytics YOLO11-seg。权重类别必须同时有 person、tote、pallet。"""

    name = "yolo11-seg"

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 320):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.last_forward_ms: float | None = None
        self.last_predict_ms: float | None = None
        names = {int(k): str(v).strip().lower() for k, v in self.model.names.items()}
        self._id_to_class = {}
        for idx, name in names.items():
            mapped = map_class_name(name)
            if mapped is not None:
                self._id_to_class[idx] = mapped
        present = set(self._id_to_class.values())
        missing = {"PERSON", "TOTE", "PALLET"} - present
        if missing:
            have = sorted(names.values())
            raise ValueError(
                "YOLO11-seg 权重类别缺少 "
                + ", ".join(sorted(missing))
                + f"。当前类别：{have}。PERSON 必须单独成类，不能用 COCO 权重顶替周转箱。"
            )

    def detect(self, rgb: np.ndarray, label: np.ndarray | None = None) -> list[RawDetection]:
        del label
        import time

        # predict 把 numpy 当成 BGR。
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        t0 = time.perf_counter()
        results = self.model.predict(
            bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device="cpu",
        )
        self.last_predict_ms = (time.perf_counter() - t0) * 1000.0
        result = results[0]
        speed = getattr(result, "speed", None) or {}
        if "inference" in speed:
            self.last_forward_ms = float(speed["inference"])
        else:
            self.last_forward_ms = self.last_predict_ms
        height, width = rgb.shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        polygons = []
        if result.masks is not None and result.masks.xy is not None:
            polygons = list(result.masks.xy)
        found = []
        classes = result.boxes.cls.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        for i, (cls_id, score) in enumerate(zip(classes, scores)):
            class_name = self._id_to_class.get(int(cls_id))
            if class_name is None:
                continue
            if i < len(polygons):
                mask = rasterize_polygon(polygons[i], height, width)
            else:
                mask = np.zeros((height, width), dtype=bool)
            found.append(RawDetection(class_name, float(score), mask))
        return found
