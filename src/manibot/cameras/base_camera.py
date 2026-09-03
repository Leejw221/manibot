from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):
    """Base interface for camera capture. All cameras must implement these methods."""

    @abstractmethod
    def connect(self):
        """Connect to the camera and start capture."""
        pass

    @abstractmethod
    def disconnect(self):
        """Stop capture and disconnect."""
        pass

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Capture a single RGB image.

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB image.
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the camera is connected."""
        pass
