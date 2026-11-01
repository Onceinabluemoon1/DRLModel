"""杀掉速度进程之后，人的回放仍然给出 STOP_PERSON。"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


POLICY_VX = 0.8
CRUISE_VX = 0.6


def _free_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _read_policy(port: int, timeout_s: float = 5.0) -> dict:
    deadline = time.time() + timeout_s
    last_error = "未连接"
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5) as conn:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(256)
                    if not chunk:
                        break
                    data += chunk
            return json.loads(data.decode("utf-8"))
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.05)
    raise RuntimeError(f"速度进程没有吐出候选速度：{last_error}")


def _run_replay(clip: str, events: Path, ready: Path | None, release: Path | None, sleep_ms: int):
    cmd = [
        sys.executable,
        "-m",
        "aisle_replay",
        "replay",
        "--clip",
        clip,
        "--detector",
        "fixture",
        "--events",
        str(events),
        "--frame-sleep-ms",
        str(sleep_ms),
    ]
    if ready is not None:
        cmd.extend(["--ready", str(ready)])
    if release is not None:
        cmd.extend(["--release", str(release)])
    return subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_file(path: Path, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise RuntimeError(f"等待文件超时：{path}")


def prove() -> dict:
    """速度进程先活着做一次空通道回放，再杀掉，然后才开始人的回放。"""
    port = _free_port()
    velocity = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "aisle_replay.velocity_node",
            "--port",
            str(port),
            "--vx",
            str(POLICY_VX),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        policy = _read_policy(port)
        if abs(float(policy["vx"]) - POLICY_VX) > 1e-6:
            raise RuntimeError(f"速度进程输出不是 {POLICY_VX}：{policy}")

        with tempfile.TemporaryDirectory(prefix="aisle-replay-") as tmp:
            root = Path(tmp)
            empty_events = root / "empty.jsonl"
            empty = _run_replay("empty", empty_events, None, None, 0)
            empty_out, empty_err = empty.communicate(timeout=30)
            if empty.returncode != 0:
                raise RuntimeError(f"空通道回放失败：{empty_out}\n{empty_err}")
            if velocity.poll() is not None:
                raise RuntimeError("空通道回放期间速度进程自己退出了")
            empty_rows = [
                json.loads(line)
                for line in empty_events.read_text(encoding="utf-8").splitlines()
            ]
            for row in empty_rows:
                if row["reason"] != "OK" or abs(row["vx_m_s"] - CRUISE_VX) > 1e-6:
                    raise RuntimeError(
                        f"空通道应给出规则巡航 {CRUISE_VX}，实际 {row}。"
                        f"速度进程正在发送 {POLICY_VX}。"
                    )

            ready = root / "ready"
            release = root / "release"
            person_events = root / "person.jsonl"
            person = _run_replay("person", person_events, ready, release, 0)
            try:
                _wait_file(ready, 10)
                os.kill(velocity.pid, 9)
                velocity.wait(timeout=5)
                if velocity.poll() is None:
                    raise RuntimeError("速度进程没有被杀掉")
                release.write_text("go\n", encoding="utf-8")
                person_out, person_err = person.communicate(timeout=30)
            except Exception:
                person.kill()
                person.communicate(timeout=5)
                raise
            if person.returncode != 0:
                raise RuntimeError(f"人回放失败：{person_out}\n{person_err}")
            rows = [
                json.loads(line)
                for line in person_events.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(
                person_events.with_suffix(".summary.json").read_text(encoding="utf-8")
            )
            stops = [row for row in rows if row["reason"] == "STOP_PERSON"]
            if not stops:
                raise RuntimeError(f"杀掉速度进程后没有 STOP_PERSON：{rows}")
            for row in stops:
                if abs(row["vx_m_s"]) > 1e-9:
                    raise RuntimeError(f"STOP_PERSON 时速度不是 0：{row}")
            if summary["mismatched_frames"]:
                raise RuntimeError(f"人回放原因码和几何期望不一致：{summary}")
            enter = summary["person_enter_frame"]
            stop_frames = [row["frame"] for row in stops]
            delay = min(stop_frames) - enter
            if delay < 0 or delay > 1:
                raise RuntimeError(
                    f"STOP_PERSON 不在两个感知帧内：进入帧 {enter}，停车帧 {stop_frames}"
                )
            return {
                "velocity_killed": True,
                "policy_vx_while_alive": POLICY_VX,
                "empty_vx_m_s": CRUISE_VX,
                "person_reasons": [row["reason"] for row in rows],
                "person_enter_frame": enter,
                "stop_person_frames": stop_frames,
                "delay_frames": delay,
            }
    finally:
        if velocity.poll() is None:
            velocity.kill()
        velocity.communicate(timeout=5)


def main() -> int:
    report = prove()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
