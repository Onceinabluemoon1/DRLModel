"""速度占位进程。

它按固定周期吐出候选速度。安全门不连接这个端口。
这里的 vx 故意是 0.8，和 safety/rules.yaml 里的巡航 0.6 不同，
这样从输出就能看出安全门没有采用这里的数。
"""

from __future__ import annotations

import argparse
import json
import socket
import time


def serve(host: str, port: int, vx: float) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    server.settimeout(0.2)
    print(f"velocity listening {host}:{port} vx={vx}", flush=True)
    while True:
        try:
            conn, _addr = server.accept()
        except socket.timeout:
            continue
        line = json.dumps({"vx": vx, "wz": 0.0}) + "\n"
        try:
            conn.sendall(line.encode("utf-8"))
        finally:
            conn.close()
        time.sleep(0.01)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="速度占位进程")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vx", type=float, default=0.8)
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.vx)


if __name__ == "__main__":
    main()
