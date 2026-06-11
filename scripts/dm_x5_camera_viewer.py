"""
腕部相机与触觉传感器可视化脚本
使用方法
需保证x5上的server在正常运行
1.识别全部相机，直接执行 python dm_x5_viewer.py 
2.识别单个夹爪的相机 python dm_x5_viewer.py --ip=对应夹爪ip

"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0)) # 禁止生成core dump文件

import argparse
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import zmq


def normalize_gray(frame):
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return frame


def ensure_bgr(frame):
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)
    return frame


def make_placeholder(width, height, title, subtitle, color=(40, 40, 40)):
    canvas = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.putText(canvas, title, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, subtitle, (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 180, 180), 1)
    return canvas


def pad_to_height(image, target_height, color=(0, 0, 0)):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    if height > target_height:
        return cv2.resize(image, (width, target_height))

    pad_total = target_height - height
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    return cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=color,
    )


def pad_to_width(image, target_width, color=(0, 0, 0)):
    height, width = image.shape[:2]
    if width == target_width:
        return image
    if width > target_width:
        return cv2.resize(image, (target_width, height))

    pad_total = target_width - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return cv2.copyMakeBorder(
        image,
        0,
        0,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )


class WristReader:
    def __init__(
        self,
        ip,
        port=5552,
        recv_timeout_s=2.0,
        expected_width=1920,
        expected_height=1080,
        expected_fps=30,
    ):
        self.ip = ip
        self.port = port
        self.recv_timeout_s = recv_timeout_s
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.expected_fps = expected_fps
        self.context = None
        self.sock = None
        self.running = False
        self.thread = None
        self.frame = None
        self.last_seen = None
        self.lock = threading.Lock()
        self.error = None
        self.frame_count = 0
        self.max_frame_bytes = 2 * 1024 * 1024

    def connect(self):
        self.context = zmq.Context.instance()
        self.sock = self.context.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVTIMEO, int(self.recv_timeout_s * 1000))
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(f"tcp://{self.ip}:{self.port}")
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        while self.running:
            try:
                payload = self.sock.recv()
            except zmq.Again:
                continue
            except Exception as e:
                self.error = str(e)
                break

            try:
                if not payload:
                    continue

                if len(payload) > self.max_frame_bytes:
                    self.error = f"oversized jpeg={len(payload)}"
                    continue

                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                frame = np.ascontiguousarray(frame)
                with self.lock:
                    self.frame = frame
                    self.last_seen = time.time()
                    self.frame_count += 1
            except Exception as e:
                self.error = str(e)
                break

        self.running = False

    def get_latest(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            last_seen = self.last_seen
            frame_count = self.frame_count
        return frame, last_seen, frame_count

    def disconnect(self):
        self.running = False
        if self.sock is not None:
            try:
                self.sock.close(0)
            finally:
                self.sock = None
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class TactileReader:
    def __init__(self, ip, port=39010, topic="", timeout_ms=200):
        self.ip = ip
        self.port = port
        self.topic = topic.encode("utf-8") if isinstance(topic, str) else topic
        self.timeout_ms = timeout_ms
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.devices = OrderedDict()
        self.error = None

    def connect(self):
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        context = None
        sub = None
        while self.running:
            try:
                if sub is None:
                    context = zmq.Context()
                    sub = context.socket(zmq.SUB)
                    sub.connect(f"tcp://{self.ip}:{self.port}")
                    sub.setsockopt(zmq.SUBSCRIBE, self.topic)
                    sub.setsockopt(zmq.RCVTIMEO, self.timeout_ms)

                topic_b, meta_b, raw_b = sub.recv_multipart()
                serial = topic_b.decode("utf-8", errors="ignore").strip()
                if not serial:
                    continue
                meta = json.loads(meta_b.decode("utf-8"))
                raw_info = meta.get("raw", {})
                shape = raw_info.get("shape")
                if not raw_info.get("present") or not shape or not raw_b:
                    continue

                frame = np.frombuffer(raw_b, dtype=np.uint8).reshape(tuple(shape))
                frame = normalize_gray(frame)
                now = time.time()

                with self.lock:
                    item = self.devices.get(serial)
                    if item is None:
                        item = {
                            "serial": serial,
                            "frame": frame,
                            "meta": meta,
                            "first_seen": now,
                            "last_seen": now,
                        }
                        self.devices[serial] = item
                        print(f"[{self.ip}] 发现触觉 serial={serial}")
                    else:
                        item["frame"] = frame
                        item["meta"] = meta
                        item["last_seen"] = now
            except zmq.Again:
                continue
            except Exception as e:
                self.error = str(e)
                break

        if sub is not None:
            try:
                sub.close(0)
            except Exception:
                pass
        if context is not None:
            try:
                context.term()
            except Exception:
                pass
        self.running = False

    def get_latest_devices(self):
        with self.lock:
            result = []
            for serial, item in self.devices.items():
                result.append(
                    {
                        "serial": serial,
                        "frame": None if item["frame"] is None else item["frame"].copy(),
                        "meta": None if item["meta"] is None else dict(item["meta"]),
                        "first_seen": item["first_seen"],
                        "last_seen": item["last_seen"],
                    }
                )
        return result

    def disconnect(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class X5MultiCameraViewer:
    def __init__(
        self,
        ips,
        wrist_port=5552,
        tactile_port=39010,
        tactile_topic="",
        window_name="X5 Multi Camera Viewer",
        wrist_tile_size=(640, 360),
        tactile_tile_size=(640, 480),
        stale_timeout_s=2.0,
    ):
        self.ips = ips
        self.window_name = window_name
        self.wrist_tile_size = wrist_tile_size
        self.tactile_tile_size = tactile_tile_size
        self.stale_timeout_s = stale_timeout_s
        self.running = False
        self.show_overlay = True
        self.groups = []

        for ip in ips:
            self.groups.append(
                {
                    "ip": ip,
                    "wrist": WristReader(
                        ip,
                        port=wrist_port,
                        expected_width=1920,
                        expected_height=1080,
                        expected_fps=30,
                    ),
                    "tactile": TactileReader(ip, port=tactile_port, topic=tactile_topic),
                }
            )

    def connect(self):
        for group in self.groups:
            ip = group["ip"]
            try:
                group["wrist"].connect()
                print(f"[{ip}] 鱼眼相机已连接")
            except Exception as e:
                group["wrist"].error = str(e)
                print(f"[{ip}] 鱼眼相机连接失败: {e}")

            try:
                group["tactile"].connect()
                print(f"[{ip}] 触觉广播已连接")
            except Exception as e:
                group["tactile"].error = str(e)
                print(f"[{ip}] 触觉广播连接失败: {e}")

    def disconnect(self):
        for group in self.groups:
            group["wrist"].disconnect()
            group["tactile"].disconnect()
        cv2.destroyAllWindows()

    def _annotate(self, image, lines, color=(255, 255, 255)):
        if not self.show_overlay:
            return image
        output = image.copy()
        y = 28
        for line in lines:
            cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 1)
            y += 26
        return output

    def _build_wrist_tile(self, group):
        ip = group["ip"]
        width, height = self.wrist_tile_size
        frame, last_seen, frame_count = group["wrist"].get_latest()
        if frame is None:
            msg = group["wrist"].error or "waiting for wrist stream"
            tile = make_placeholder(width, height, f"IP {ip} | Wrist", msg, color=(60, 30, 30))
        else:
            tile = ensure_bgr(frame)
            tile = cv2.resize(tile, (width, height))
            tile = self._annotate(
                tile,
                [
                    f"IP: {ip}",
                    "Stream: 1920x1080 @ 30",
                    f"Frames: {frame_count}",
                ],
                color=(0, 255, 0),
            )
        return tile

    def _build_tactile_tile(self, ip, device, slot_name):
        width, height = self.tactile_tile_size
        if device is None or device["frame"] is None:
            return make_placeholder(width, height, f"IP {ip} | {slot_name}", "waiting for tactile", color=(30, 30, 60))

        frame = ensure_bgr(device["frame"])
        frame = cv2.resize(frame, (width, height))
        fid = device["meta"].get("fid", "N/A") if device["meta"] else "N/A"
        return self._annotate(
            frame,
            [
                f"IP: {ip}",
                f"Serial: {device['serial']}",
                f"FID: {fid}",
            ],
            color=(0, 255, 255),
        )

    def _build_row(self, group):
        ip = group["ip"]
        tiles = []

        wrist_frame, _, _ = group["wrist"].get_latest()
        if wrist_frame is not None:
            tiles.append(self._build_wrist_tile(group))

        tactile_devices = group["tactile"].get_latest_devices()
        tactile_devices = sorted(tactile_devices, key=lambda x: x["first_seen"])[:2]
        for idx, device in enumerate(tactile_devices, start=1):
            if device is not None and device.get("frame") is not None:
                tiles.append(self._build_tactile_tile(ip, device, f"tactile_{idx}"))

        if not tiles:
            row = make_placeholder(
                self.wrist_tile_size[0],
                self.wrist_tile_size[1],
                f"IP {ip}",
                "waiting for any camera stream",
                color=(40, 40, 40),
            )
            return row

        row_height = max(tile.shape[0] for tile in tiles)
        tiles = [pad_to_height(tile, row_height, color=(20, 20, 20)) for tile in tiles]
        row = np.hstack(tiles)
        return row

    def _build_canvas(self):
        rows = [self._build_row(group) for group in self.groups]
        if not rows:
            return make_placeholder(1280, 720, "X5 Multi Camera Viewer", "no IP configured")
        max_width = max(row.shape[1] for row in rows)
        rows = [pad_to_width(row, max_width, color=(20, 20, 20)) for row in rows]
        return np.vstack(rows)

    def print_summary(self):
        print("\n当前识别结果:")
        for group in self.groups:
            wrist_frame, _, _ = group["wrist"].get_latest()
            devices = group["tactile"].get_latest_devices()
            serials = [d["serial"] for d in devices]
            print(f"  {group['ip']}: wrist={'yes' if wrist_frame is not None else 'no'}, tactile={serials if serials else '[]'}")

    def run(self):
        self.running = True
        print("\n开始显示双手相机流...")
        print("控制指令:")
        print("  q / ESC: 退出")
        print("  c: 显示/隐藏信息叠加")
        print("  p: 在终端打印当前识别到的 serial")

        while self.running:
            canvas = self._build_canvas()
            now = datetime.now().strftime("%H:%M:%S")
            cv2.putText(canvas, now, (canvas.shape[1] - 120, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow(self.window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.running = False
            elif key == ord("c"):
                self.show_overlay = not self.show_overlay
                print(f"信息叠加: {'显示' if self.show_overlay else '隐藏'}")
            elif key == ord("p"):
                self.print_summary()

        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Visualize wrist fisheye + tactile cameras from two X5 boards.")
    parser.add_argument("--ip", type=str, default=None, help="Use a single X5 IP only")
    parser.add_argument("--ips", nargs="+", default=None, help="X5 IP list, supports one or two IPs")
    parser.add_argument("--wrist-port", type=int, default=5552, help="Wrist fisheye TCP port")
    parser.add_argument("--tactile-port", type=int, default=39010, help="Tactile ZMQ port")
    parser.add_argument("--tactile-topic", type=str, default="", help="Optional tactile topic prefix filter")
    parser.add_argument("--window-name", type=str, default="X5 Multi Camera Viewer")
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=360)
    parser.add_argument("--tactile-width", type=int, default=640)
    parser.add_argument("--tactile-height", type=int, default=480)
    args = parser.parse_args()

    if args.ip:
        ips = [args.ip]
    elif args.ips:
        ips = args.ips
    else:
        ips = ["192.168.9.10", "192.168.9.11"]

    if len(ips) == 0:
        raise ValueError("At least one X5 IP must be provided.")

    if len(ips) > 2:
        raise ValueError(f"Expected one or two X5 IPs, got {len(ips)}: {ips}")

    viewer = X5MultiCameraViewer(
        ips=ips,
        wrist_port=args.wrist_port,
        tactile_port=args.tactile_port,
        tactile_topic=args.tactile_topic,
        window_name=args.window_name,
        wrist_tile_size=(args.wrist_width, args.wrist_height),
        tactile_tile_size=(args.tactile_width, args.tactile_height),
    )
    viewer.connect()
    try:
        viewer.run()
    finally:
        viewer.disconnect()


if __name__ == "__main__":
    main()
