#!/usr/bin/env python

from lerobot.cameras.dm_tactile_x5 import DmTactileCameraX5, DmTactileCameraX5Config
from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.wrist_fisheye_x5 import WristFisheyeX5Camera, WirstFisheyeCameraX5Config


def test_make_spark3_camera_backends_from_config():
    cameras = make_cameras_from_configs(
        {
            "wrist": WirstFisheyeCameraX5Config(
                x5_ip="192.168.9.11",
                width=640,
                height=480,
                fps=30,
            ),
            "tactile": DmTactileCameraX5Config(
                x5_ip="192.168.9.10",
                left_serial_number="left",
                right_serial_number="right",
                width=320,
                height=240,
                fps=30,
            ),
            "head": OrbbecCameraConfig(
                serial_number="AUG5630002Z",
                width=640,
                height=480,
                fps=30,
                use_depth=True,
            ),
        }
    )

    assert isinstance(cameras["wrist"], WristFisheyeX5Camera)
    assert isinstance(cameras["tactile"], DmTactileCameraX5)
    assert isinstance(cameras["head"], OrbbecCamera)


def test_dm_tactile_config_builds_pub_addr_and_three_channel_output_contract():
    config = DmTactileCameraX5Config(
        x5_ip="192.168.9.10",
        left_serial_number="left",
        width=320,
        height=240,
        fps=30,
    )

    assert config.pub_addr == "tcp://192.168.9.10:39010"
    assert config.channels == 3
