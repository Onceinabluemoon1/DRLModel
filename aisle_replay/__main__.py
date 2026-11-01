"""命令：replay、velocity、prove-kill、train、measure、dump-clips。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aisle_replay.replay import run_replay_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aisle_replay")
    sub = parser.add_subparsers(dest="cmd", required=True)

    replay = sub.add_parser("replay", help="回放一段 RGB-D")
    replay.add_argument("--clip", required=True, choices=["empty", "person", "tote", "pallet", "tote_1m"])
    replay.add_argument("--detector", default="fixture", choices=["fixture", "yolo"])
    replay.add_argument("--weights", default=None)
    replay.add_argument("--conf", type=float, default=0.25)
    replay.add_argument("--events", default=None)
    replay.add_argument("--ready", default=None)
    replay.add_argument("--release", default=None)
    replay.add_argument("--frame-sleep-ms", type=int, default=0)

    velocity = sub.add_parser("velocity", help="速度占位进程")
    velocity.add_argument("--host", default="127.0.0.1")
    velocity.add_argument("--port", type=int, required=True)
    velocity.add_argument("--vx", type=float, default=0.8)

    sub.add_parser("prove-kill", help="杀掉速度进程后人回放仍 STOP_PERSON")

    train = sub.add_parser("train", help="在合成帧上微调 YOLO11-seg")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--out", default=None)

    measure = sub.add_parser("measure", help="测量召回、距离误差、前向毫秒")
    measure.add_argument("--weights", default=None)

    dump = sub.add_parser("dump-clips", help="把四段回放写成 png，便于看画面")
    dump.add_argument("--dir", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "replay":
        return run_replay_cli(args)
    if args.cmd == "velocity":
        from aisle_replay.velocity_node import serve

        serve(args.host, args.port, args.vx)
        return 0
    if args.cmd == "prove-kill":
        from aisle_replay.prove_kill import main as prove_main

        return prove_main()
    if args.cmd == "train":
        from aisle_replay.train_synth import train as train_model

        out = train_model(epochs=args.epochs, out=Path(args.out) if args.out else None)
        print(out)
        return 0
    if args.cmd == "measure":
        from aisle_replay.measure import main as measure_main

        return measure_main(args.weights)
    if args.cmd == "dump-clips":
        from PIL import Image

        from aisle_replay.clips import build_clip

        dest = Path(args.dir)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("empty", "person", "tote", "pallet", "tote_1m"):
            clip = build_clip(name)
            for event_i, frame in enumerate(clip.frames):
                path = dest / f"{name}_{event_i:02d}_{clip.expected_reasons[event_i]}.png"
                Image.fromarray(frame.rgb, mode="RGB").save(path)
                print(path, int((frame.label > 0).sum()))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
