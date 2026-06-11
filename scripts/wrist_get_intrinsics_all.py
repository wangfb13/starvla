#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISHEYE_HOSTS = ["192.168.9.10", "192.168.9.11"]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_fish_camera_messages():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "camera_proxy_inline.proto"
    file_proto.package = "fish_camera.grpc_test"
    file_proto.syntax = "proto3"

    calibration_request = file_proto.message_type.add()
    calibration_request.name = "CalibrationRequest"
    field = calibration_request.field.add()
    field.name = "device"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    intrinsics_response = file_proto.message_type.add()
    intrinsics_response.name = "IntrinsicsResponse"
    fields = [
        ("device", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, False),
        ("camera_model", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, False),
        ("distortion_model", 3, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, False),
        ("intrinsics", 4, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, True),
        ("camera_matrix", 5, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, True),
        ("distortion_coeffs", 6, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, True),
        ("resolution", 7, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE, True),
        ("sn", 8, descriptor_pb2.FieldDescriptorProto.TYPE_STRING, False),
        ("sn_valid", 9, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL, False),
        ("camera_model_enum", 10, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32, False),
    ]
    for name, number, field_type, repeated in fields:
        field = intrinsics_response.field.add()
        field.name = name
        field.number = number
        field.label = (
            descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
            if repeated
            else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        )
        field.type = field_type

    sn_response = file_proto.message_type.add()
    sn_response.name = "SNResponse"
    fields = [
        ("device", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("sn", 2, descriptor_pb2.FieldDescriptorProto.TYPE_STRING),
        ("valid", 3, descriptor_pb2.FieldDescriptorProto.TYPE_BOOL),
    ]
    for name, number, field_type in fields:
        field = sn_response.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = field_type

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    factory = message_factory.MessageFactory(pool)

    return (
        message_factory.GetMessageClass(pool.FindMessageTypeByName("fish_camera.grpc_test.CalibrationRequest")),
        message_factory.GetMessageClass(pool.FindMessageTypeByName("fish_camera.grpc_test.IntrinsicsResponse")),
        message_factory.GetMessageClass(pool.FindMessageTypeByName("fish_camera.grpc_test.SNResponse")),
    )


CalibrationRequest, IntrinsicsResponse, SNResponse = _build_fish_camera_messages()


def _import_grpc():
    import grpc

    return grpc


class RemoteCameraCapture:
    def __init__(
        self,
        host: str,
        port: int = 50088,
        codec: str = "MJPG",
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        udp_port: int = 0,
        bind_host: str = "0.0.0.0",
        client_ip: str = "",
        device: str = "",
    ):
        self.host = host
        self.port = int(port)
        self.codec = codec
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.udp_port = int(udp_port)
        self.bind_host = bind_host
        self.client_ip = client_ip
        self.device = device

    @staticmethod
    def _ensure_direct_grpc_target_for_host(host: str):
        for key in ("no_proxy", "NO_PROXY"):
            current = os.environ.get(key, "")
            hosts = [item.strip() for item in current.split(",") if item.strip()]
            if host not in hosts:
                hosts.append(host)
            os.environ[key] = ",".join(hosts)

    def _ensure_direct_grpc_target(self):
        self._ensure_direct_grpc_target_for_host(self.host)

    def _calibration_device(self) -> str:
        return self.device or ""

    @staticmethod
    def _format_intrinsics_response(response):
        matrix = list(response.camera_matrix)
        if len(matrix) == 9:
            camera_matrix = [matrix[0:3], matrix[3:6], matrix[6:9]]
        else:
            camera_matrix = []
        return {
            "intrinsics": list(response.intrinsics),
            "camera_matrix": camera_matrix,
            "distortion_coeffs": list(response.distortion_coeffs),
            "resolution": list(response.resolution),
        }

    def get_intrinsics(self, timeout: float | None = None):
        grpc = _import_grpc()
        self._ensure_direct_grpc_target()
        with grpc.insecure_channel(f"{self.host}:{self.port}", options=(("grpc.enable_http_proxy", 0),)) as channel:
            rpc = channel.unary_unary(
                "/fish_camera.grpc_test.CameraProxy/GetIntrinsics",
                request_serializer=CalibrationRequest.SerializeToString,
                response_deserializer=IntrinsicsResponse.FromString,
            )
            response = rpc(CalibrationRequest(device=self._calibration_device()), timeout=timeout)
            return self._format_intrinsics_response(response)

    def get_sn(self, timeout: float | None = None) -> str:
        grpc = _import_grpc()
        self._ensure_direct_grpc_target()
        with grpc.insecure_channel(f"{self.host}:{self.port}", options=(("grpc.enable_http_proxy", 0),)) as channel:
            rpc = channel.unary_unary(
                "/fish_camera.grpc_test.CameraProxy/GetSN",
                request_serializer=CalibrationRequest.SerializeToString,
                response_deserializer=SNResponse.FromString,
            )
            response = rpc(CalibrationRequest(device=self._calibration_device()), timeout=timeout)
            return response.sn

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch wrist fisheye intrinsics from fish_camera service and save camchain YAML files."
    )
    parser.add_argument(
        "hosts",
        nargs="*",
        default=DEFAULT_FISHEYE_HOSTS,
        help=(
            "One or more X5 IP addresses. "
            f"Defaults to: {' '.join(DEFAULT_FISHEYE_HOSTS)}"
        ),
    )
    parser.add_argument("--port", type=int, default=50088, help="fish_camera gRPC port")
    parser.add_argument("--codec", default="MJPG", choices=("MJPG", "HEVC"), help="Requested codec")
    parser.add_argument("--width", type=int, default=1920, help="Expected image width")
    parser.add_argument("--height", type=int, default=1080, help="Expected image height")
    parser.add_argument("--fps", type=int, default=30, help="Requested fps")
    parser.add_argument("--device", default="", help="Optional remote device name")
    parser.add_argument("--udp-port", type=int, default=0, help="Optional local UDP port")
    parser.add_argument("--bind-host", default="0.0.0.0", help="Local bind host for UDP")
    parser.add_argument("--client-ip", default="", help="Optional client IP reported to server")
    parser.add_argument("--timeout", type=float, default=5.0, help="gRPC timeout for intrinsics request")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "run_scripts" / "intrinsics_data"),
        help="Directory used to save YAML files",
    )
    parser.add_argument("--camera-model", default="pinhole", help="camera_model field in YAML")
    parser.add_argument("--distortion-model", default="equidistant", help="distortion_model field in YAML")
    parser.add_argument("--rostopic", default="/camera/image_raw", help="rostopic field in YAML")
    return parser


def sanitize_host_for_filename(host: str) -> str:
    return host.replace(".", "_").replace("/", "_").replace(":", "_").strip()


def format_camchain_yaml(intrinsics_data: dict, resolution_fallback: list[int], args) -> dict:
    raw_intrinsics = intrinsics_data.get("intrinsics")
    camera_matrix = intrinsics_data.get("camera_matrix")
    distortion_coeffs = intrinsics_data.get("distortion_coeffs")
    resolution = intrinsics_data.get("resolution") or resolution_fallback

    if raw_intrinsics and len(raw_intrinsics) == 4:
        fx, fy, cx, cy = [float(v) for v in raw_intrinsics]
    elif camera_matrix and len(camera_matrix) == 3 and all(len(row) == 3 for row in camera_matrix):
        fx = float(camera_matrix[0][0])
        fy = float(camera_matrix[1][1])
        cx = float(camera_matrix[0][2])
        cy = float(camera_matrix[1][2])
    else:
        raise ValueError(f"Invalid intrinsics response: {intrinsics_data}")

    if not distortion_coeffs:
        raise ValueError(f"Missing distortion_coeffs in intrinsics response: {intrinsics_data}")

    if not resolution or len(resolution) != 2:
        raise ValueError(f"Invalid resolution in intrinsics response: {intrinsics_data}")

    return {
        "cam0": {
            "cam_overlaps": [],
            "camera_model": args.camera_model,
            "distortion_coeffs": [float(v) for v in distortion_coeffs],
            "distortion_model": args.distortion_model,
            "intrinsics": [fx, fy, cx, cy],
            "resolution": [int(resolution[0]), int(resolution[1])],
            "rostopic": args.rostopic,
        }
    }


def fetch_and_save(host: str, args) -> Path:
    cap = RemoteCameraCapture(
        host=host,
        port=args.port,
        codec=args.codec,
        width=args.width,
        height=args.height,
        fps=args.fps,
        udp_port=args.udp_port,
        bind_host=args.bind_host,
        client_ip=args.client_ip,
        device=args.device,
    )
    intrinsics_data = cap.get_intrinsics(timeout=args.timeout)
    yaml_data = format_camchain_yaml(intrinsics_data, [args.width, args.height], args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wrist_fishCamera_{sanitize_host_for_filename(host)}.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=False)

    return output_path


def main():
    args = build_argparser().parse_args()

    for host in args.hosts:
        print(f"[info] fetching intrinsics from {host}:{args.port}", flush=True)
        output_path = fetch_and_save(host, args)
        print(f"[ok] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
