import time
import numpy as np
from manibot.cameras.base_camera import BaseCamera


class RealSenseCamera(BaseCamera):
    """Intel RealSense D405 RGB camera capture.

    Uses pyrealsense2 for camera access.
    """

    def __init__(
        self,
        serial_number: str = "",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_s: float = 2.0,
        rotation: int = 0,
        use_depth: bool = True,
    ):
        # D405 has no dedicated RGB sensor — the color stream is produced by
        # the stereo (depth) module, and its auto-exposure / white-balance
        # engine only engages while the depth stream is also enabled. With
        # depth off, color frames come out noticeably darker and higher-
        # contrast than what lerobot-record (which enables depth) captured at
        # training time, which silently shifts the policy's visual input
        # distribution at eval. Default to True so eval matches collection.
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_s = warmup_s
        self.rotation = rotation
        self.use_depth = use_depth
        self._pipeline = None
        self._connected = False

    def connect(self, max_retries: int = 3):
        import pyrealsense2 as rs

        # Pre-reset to clear any stuck state from previous runs
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if not self.serial_number or dev.get_info(rs.camera_info.serial_number) == str(self.serial_number):
                    dev.hardware_reset()
                    print(f"RealSenseCamera pre-reset done (serial: {self.serial_number})")
                    break
            time.sleep(5)
        except Exception:
            pass

        for attempt in range(1, max_retries + 1):
            self._pipeline = rs.pipeline()
            config = rs.config()

            if self.serial_number:
                config.enable_device(self.serial_number)

            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            if self.use_depth:
                # Depth stream enabled so the stereo module's AE/AWB engine
                # engages; the depth data itself is discarded in get_image().
                config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            self._pipeline.start(config)

            try:
                if self.warmup_s > 0:
                    start = time.time()
                    while time.time() - start < self.warmup_s:
                        self._pipeline.wait_for_frames(timeout_ms=5000)

                self._connected = True
                print(f"RealSenseCamera connected (serial: {self.serial_number}, {self.width}x{self.height}@{self.fps}fps)")
                return
            except RuntimeError as e:
                print(f"RealSenseCamera warmup failed (attempt {attempt}/{max_retries}): {e}")
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None

                if attempt < max_retries:
                    print("Performing hardware reset...")
                    try:
                        ctx = rs.context()
                        for dev in ctx.query_devices():
                            if dev.get_info(rs.camera_info.serial_number) == str(self.serial_number):
                                dev.hardware_reset()
                                break
                    except Exception as reset_err:
                        print(f"Hardware reset failed: {reset_err}")
                    time.sleep(5)
                else:
                    raise RuntimeError(f"RealSenseCamera failed to connect after {max_retries} attempts.") from e

    def disconnect(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._connected = False
        print("RealSenseCamera disconnected")

    def get_image(self) -> np.ndarray:
        """Get latest RGB image (180° rotated for wrist mount).

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB.
        """
        import cv2
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received")
        image = np.asanyarray(color_frame.get_data())
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image

    def is_connected(self) -> bool:
        return self._connected
