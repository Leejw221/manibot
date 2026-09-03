import time
import numpy as np
import cv2
from manibot.cameras.base_camera import BaseCamera


class OrbbecCamera(BaseCamera):
    """Orbbec Femto Bolt RGB camera capture.

    Uses pyorbbecsdk for camera access. Runs a background thread for
    continuous frame capture to minimize latency.
    """

    def __init__(
        self,
        serial_number_or_index: str = "",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_s: float = 2.0,
    ):
        self.serial_number_or_index = serial_number_or_index
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_s = warmup_s
        self._pipeline = None
        self._connected = False
        self._latest_frame = None
        self._read_thread = None
        self._stop_event = None

    def connect(self):
        import pyorbbecsdk as ob
        import threading

        ctx = ob.Context()

        # Select device
        device = None
        device_list = ctx.query_devices()
        if self.serial_number_or_index:
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                serial = dev.get_device_info().get_serial_number()
                if serial == self.serial_number_or_index or str(i) == self.serial_number_or_index:
                    device = dev
                    break
            if device is None:
                raise RuntimeError(f"Orbbec device '{self.serial_number_or_index}' not found")
        else:
            device = device_list.get_device_by_index(0)

        self._pipeline = ob.Pipeline(device)
        config = ob.Config()

        # Configure color stream
        profile_list = self._pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        profile = profile_list.get_video_stream_profile(self.width, self.height, ob.OBFormat.RGB, self.fps)
        config.enable_stream(profile)

        self._pipeline.start(config)

        # Background read thread
        self._stop_event = threading.Event()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

        # Warmup
        if self.warmup_s > 0:
            time.sleep(self.warmup_s)

        self._connected = True
        serial = device.get_device_info().get_serial_number()
        print(f"OrbbecCamera connected (serial: {serial}, {self.width}x{self.height}@{self.fps}fps)")

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                w = color_frame.get_width()
                h = color_frame.get_height()
                data = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(h, w, 3)
                self._latest_frame = data.copy()
            except Exception:
                continue

    def disconnect(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._connected = False
        print("OrbbecCamera disconnected")

    def get_image(self) -> np.ndarray:
        """Get latest RGB image.

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB.
        """
        if self._latest_frame is None:
            raise RuntimeError("No frame available yet")
        return self._latest_frame.copy()

    def is_connected(self) -> bool:
        return self._connected
