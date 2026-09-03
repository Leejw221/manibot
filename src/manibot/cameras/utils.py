"""In-repo camera factory.

Mirrors ``lerobot.cameras.utils.make_cameras_from_configs`` but instantiates
manipulation_pipeline's own camera classes (Orbbec + RealSense, ported with the
connection-stability fixes) so data collection runs on repo-owned hardware code
rather than lerobot's. Unknown camera types fall back to lerobot's factory.
"""

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}
    for key, cfg in camera_configs.items():
        if cfg.type == "orbbec":
            from manibot.cameras.orbbec.camera_orbbec import OrbbecCamera

            cameras[key] = OrbbecCamera(cfg)
        elif cfg.type == "intelrealsense":
            from manibot.cameras.realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)
        else:
            # Anything we don't own (opencv, zmq, reachy2, ...) → lerobot's factory.
            from lerobot.cameras.utils import make_cameras_from_configs as _lerobot_make

            cameras[key] = _lerobot_make({key: cfg})[key]
    return cameras
