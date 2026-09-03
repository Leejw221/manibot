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

"""
Provides the OrbbecCamera class for capturing frames from Orbbec cameras (e.g. Femto Bolt).
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

try:
    import pyorbbecsdk as ob
except Exception as e:
    logging.info(f"Could not import pyorbbecsdk: {e}")

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from .configuration_orbbec import OrbbecCameraConfig

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    """
    Manages interactions with Orbbec cameras (e.g. Femto Bolt) for frame capture.

    This class uses the pyorbbecsdk library to interface with Orbbec depth cameras.
    It captures color frames via a background thread for low-latency access.

    Example:
        ```python
        from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig

        config = OrbbecCameraConfig(fps=30, width=1280, height=720)
        camera = OrbbecCamera(config)
        camera.connect()

        # Read 1 frame synchronously (blocking)
        color_image = camera.read()

        # Read 1 frame asynchronously (waits for new frame with a timeout)
        async_image = camera.async_read()

        camera.disconnect()
        ```
    """

    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number_or_index = config.serial_number_or_index
        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self._pipeline = None
        self._ob_config = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number_or_index})"

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Connect to the Orbbec camera and start the background read thread."""
        self._pipeline = ob.Pipeline()

        # If serial number specified, open specific device
        if self.serial_number_or_index:
            ctx = ob.Context()
            devices = ctx.query_devices()
            device = None

            if self.serial_number_or_index.isdigit():
                idx = int(self.serial_number_or_index)
                if idx < devices.get_count():
                    device = devices.get_device_by_index(idx)
            else:
                for i in range(devices.get_count()):
                    dev = devices.get_device_by_index(i)
                    info = dev.get_device_info()
                    if info.get_serial_number() == self.serial_number_or_index:
                        device = dev
                        break

            if device is not None:
                self._pipeline = ob.Pipeline(device)
            else:
                logger.warning(
                    f"Could not find Orbbec device '{self.serial_number_or_index}', using default device."
                )

        self._ob_config = ob.Config()
        self._configure_streams()

        try:
            self._pipeline.start(self._ob_config)
        except Exception as e:
            self._pipeline = None
            self._ob_config = None
            raise ConnectionError(
                f"Failed to start Orbbec pipeline for {self}. "
                f"Ensure the camera is connected and pyorbbecsdk is properly installed."
            ) from e

        self._configure_capture_settings()
        self._configure_exposure_gain()
        self._start_read_thread()

        # Warmup: wait for stable frames
        self.warmup_s = max(self.warmup_s, 1)
        start_time = time.time()
        while time.time() - start_time < self.warmup_s:
            self.async_read(timeout_ms=self.warmup_s * 1000)
            time.sleep(0.1)

        with self.frame_lock:
            if self.latest_color_frame is None:
                raise ConnectionError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def _configure_streams(self) -> None:
        """Configure color stream profile on the pipeline."""
        try:
            color_profiles = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            if self.width and self.height and self.fps:
                color_profile = color_profiles.get_video_stream_profile(
                    self.capture_width, self.capture_height, ob.OBFormat.RGB, self.fps
                )
            else:
                color_profile = color_profiles.get_video_stream_profile(
                    1280, 720, ob.OBFormat.RGB, 30
                )
            self._ob_config.enable_stream(color_profile)
        except Exception as e:
            logger.warning(f"Could not configure requested color profile: {e}. Using default.")
            self._ob_config.enable_stream(
                self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                .get_default_video_stream_profile()
            )

    def _configure_capture_settings(self) -> None:
        """Set fps, width, and height from actual stream if not already configured."""
        # Try to read one frame to determine actual resolution
        try:
            frames = self._pipeline.wait_for_frames(3000)
            if frames is not None:
                color = frames.get_color_frame()
                if color is not None:
                    actual_width = color.get_width()
                    actual_height = color.get_height()

                    if self.width is None or self.height is None:
                        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                            self.width, self.height = actual_height, actual_width
                            self.capture_width, self.capture_height = actual_width, actual_height
                        else:
                            self.width, self.height = actual_width, actual_height
                            self.capture_width, self.capture_height = actual_width, actual_height

                    if self.fps is None:
                        self.fps = 30  # Default FPS for Orbbec
        except Exception as e:
            logger.warning(f"Could not determine capture settings from stream: {e}")
            if self.width is None:
                self.width = 1280
            if self.height is None:
                self.height = 720
            if self.fps is None:
                self.fps = 30
            self.capture_width = self.width
            self.capture_height = self.height

    def _configure_exposure_gain(self) -> None:
        """Configure manual exposure and gain on the color sensor."""
        try:
            device = self._pipeline.get_device()

            device.set_bool_property(ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            logger.info(f"{self} auto exposure disabled.")

            device.set_int_property(ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, self.config.exposure)
            logger.info(f"{self} exposure set to {self.config.exposure}.")

            device.set_int_property(ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT, self.config.gain)
            logger.info(f"{self} gain set to {self.config.gain}.")

        except Exception as e:
            logger.warning(f"{self} failed to configure exposure/gain: {e}")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detects available Orbbec cameras connected to the system."""
        found_cameras = []
        try:
            ctx = ob.Context()
            devices = ctx.query_devices()
            for i in range(devices.get_count()):
                device = devices.get_device_by_index(i)
                info = device.get_device_info()
                camera_info = {
                    "name": info.get_name(),
                    "type": "Orbbec",
                    "id": info.get_serial_number(),
                    "index": i,
                    "pid": info.get_pid(),
                    "vid": info.get_vid(),
                }
                found_cameras.append(camera_info)
        except Exception as e:
            logger.warning(f"Error querying Orbbec devices: {e}")
        return found_cameras

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        """Apply color conversion and rotation to a raw color frame."""
        if self.color_mode == ColorMode.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            image = cv2.rotate(image, self.rotation)

        return image

    def _read_loop(self) -> None:
        """Background thread loop for continuous frame capture."""
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                w = color_frame.get_width()
                h = color_frame.get_height()
                data = color_frame.get_data()
                color_image = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                processed_frame = self._postprocess_image(color_image)

                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_color_frame = processed_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Start the background read thread."""
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signal the background read thread to stop and wait for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """Read a single color frame synchronously (blocking)."""
        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Read the latest available color frame asynchronously."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent color frame captured immediately (non-blocking)."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """Disconnect from the camera, stop the pipeline, and clean up resources."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                logger.warning(f"Error stopping Orbbec pipeline: {e}")
            self._pipeline = None
            self._ob_config = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
