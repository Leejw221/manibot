# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Configuration class for Orbbec cameras (e.g. Femto Bolt).

    This class provides configuration options for Orbbec depth cameras using
    the pyorbbecsdk library. Devices are identified by serial number or index.

    Example configurations for Orbbec Femto Bolt:
    ```python
    # Basic configuration (auto-detect first device)
    OrbbecCameraConfig(fps=30, width=1280, height=720)

    # With serial number
    OrbbecCameraConfig(serial_number_or_index="CL4K4310079", fps=30, width=1280, height=720)

    # With device index
    OrbbecCameraConfig(serial_number_or_index="0", fps=30, width=640, height=480)
    ```

    Attributes:
        serial_number_or_index: Serial number or device index (as string) to identify the camera.
            Use empty string to auto-detect the first available device.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0, 90, 180, or 270 degrees). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds).

    Note:
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
        - Requires pyorbbecsdk to be installed.
    """

    serial_number_or_index: str = ""
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2
    exposure: int = 170
    gain: int = 10

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
