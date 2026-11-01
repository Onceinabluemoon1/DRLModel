"""在合成 RGB-D 上微调 YOLO11-seg。类别是 person、tote、pallet。"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from aisle_replay.camera import default_camera
from aisle_replay.synth import (
    YOLO_FROM_LABEL,
    Aabb,
    pallet_box,
    person_box,
    render,
    tote_box,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "yolo11n-seg-synth.pt"
_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"


def _edge_points(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    edge = m & ~inner
    ys, xs = np.nonzero(edge if edge.any() else m)
    return np.stack([xs, ys], axis=1)


def _convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def yolo_label_lines(label: np.ndarray) -> list[str]:
    height, width = label.shape
    lines = []
    for label_id, cls in YOLO_FROM_LABEL.items():
        mask = label == label_id
        if int(mask.sum()) < 40:
            continue
        hull = _convex_hull(_edge_points(mask))
        if len(hull) < 3:
            continue
        coords = []
        for x, y in hull:
            coords.append(f"{min(max(x / width, 0.0), 0.999999):.6f}")
            coords.append(f"{min(max(y / height, 0.0), 0.999999):.6f}")
        lines.append(str(cls) + " " + " ".join(coords))
    return lines


def _random_scene(rng: np.random.Generator) -> list[Aabb]:
    roll = float(rng.random())
    if roll < 0.12:
        kinds: list[str] = []
    elif roll < 0.55:
        kinds = [str(rng.choice(["PERSON", "TOTE", "PALLET"]))]
    elif roll < 0.85:
        kinds = [str(k) for k in rng.choice(["PERSON", "TOTE", "PALLET"], size=2, replace=False)]
    else:
        kinds = ["PERSON", "TOTE", "PALLET"]
    objects: list[Aabb] = []
    for kind in kinds:
        if kind == "PERSON":
            objects.append(person_box(float(rng.uniform(1.3, 4.2)), float(rng.uniform(-0.4, 0.4))))
        elif kind == "TOTE":
            objects.append(tote_box(float(rng.uniform(0.72, 3.0)), float(rng.uniform(-0.5, 0.5))))
        else:
            objects.append(pallet_box(float(rng.uniform(0.72, 2.8)), float(rng.uniform(-0.45, 0.45))))
    if float(rng.random()) < 0.35:
        front = float(rng.uniform(1.6, 4.5))
        y = float(rng.choice([-1.3, 1.35]))
        objects.append(Aabb("DISTRACTOR", front, front + 0.4, y - 0.2, y + 0.2, 0.0, 1.3))
    return objects


def write_split(root: Path, split: str, count: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    camera = default_camera()
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        rendered = render(_random_scene(rng), camera, rng=rng)
        Image.fromarray(rendered.rgb, mode="RGB").save(image_dir / f"{index:04d}.png")
        lines = yolo_label_lines(rendered.label)
        (label_dir / f"{index:04d}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _download_base(dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    urllib.request.urlretrieve(_BASE_URL, dest)
    return dest


def train(epochs: int = 40, train_count: int = 64, val_count: int = 16, out: Path | None = None) -> Path:
    out = out or DEFAULT_WEIGHTS
    data_root = ROOT / "data"
    if data_root.exists():
        shutil.rmtree(data_root)
    write_split(data_root, "train", train_count, seed=7)
    write_split(data_root, "val", val_count, seed=8)
    data_yaml = data_root / "aisle.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {data_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "  1: tote",
                "  2: pallet",
                "",
            ]
        ),
        encoding="utf-8",
    )
    from ultralytics import YOLO

    base = _download_base(Path("/tmp/yolo11n-seg.pt"))
    model = YOLO(str(base))
    run_dir = ROOT / "runs"
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=320,
        batch=4,
        device="cpu",
        workers=0,
        project=str(run_dir),
        name="aisle",
        exist_ok=True,
        pretrained=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.25,
        hsv_v=0.25,
        degrees=0.0,
        translate=0.04,
        scale=0.12,
        fliplr=0.5,
        patience=epochs,
        cls=2.0,
        val=True,
        plots=False,
        amp=False,
        cache=True,
        seed=7,
        verbose=True,
        deterministic=False,
    )
    best = run_dir / "aisle" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    return out
