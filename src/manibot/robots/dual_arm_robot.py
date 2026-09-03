"""Generic dual-arm wrapper composing two single-arm robots.

Works with any class implementing BaseRobot — write a single-arm class once,
the dual variant comes for free via composition.

Action/state convention:
    Concatenated [left, right]. For PiPER (7D each) → 14D total.
    The wrapper splits in half by default; override `_left_action_dim` if your
    robot has asymmetric DOF.

Usage:
    from manibot.robots.dual_arm_robot import DualArmRobot
    from manibot.robots.piper_robot import PiperRobot

    robot = DualArmRobot(
        single_cls=PiperRobot,
        left_config={"can_name": "can_master", "eval_init": [...]},
        right_config={"can_name": "can_slave",  "eval_init": [...]},
    )
"""

import threading
import numpy as np

from manibot.robots.base_robot import BaseRobot


class DualArmRobot(BaseRobot):
    """Compose two single-arm robot instances into a dual-arm robot."""

    def __init__(self, single_cls, left_config: dict, right_config: dict):
        self.left = single_cls(**left_config)
        self.right = single_cls(**right_config)

    def _both(self, method_name: str, *args, parallel: bool = True, **kwargs):
        """Invoke `method_name` on both arms (parallel by default via threads)."""
        if parallel:
            t_left = threading.Thread(target=getattr(self.left, method_name),
                                      args=args, kwargs=kwargs)
            t_right = threading.Thread(target=getattr(self.right, method_name),
                                       args=args, kwargs=kwargs)
            t_left.start(); t_right.start()
            t_left.join(); t_right.join()
        else:
            getattr(self.left, method_name)(*args, **kwargs)
            getattr(self.right, method_name)(*args, **kwargs)

    def connect(self):
        # Parallel connect: each arm's hardware reset + enable runs concurrently.
        self._both("connect", parallel=True)

    def disconnect(self):
        self._both("disconnect", parallel=True)

    def is_connected(self) -> bool:
        return self.left.is_connected() and self.right.is_connected()

    def set_diag_enabled(self, enabled: bool):
        if hasattr(self.left, "set_diag_enabled"):
            self.left.set_diag_enabled(enabled)
        if hasattr(self.right, "set_diag_enabled"):
            self.right.set_diag_enabled(enabled)

    def get_state(self) -> np.ndarray:
        return np.concatenate([self.left.get_state(), self.right.get_state()])

    def send_action(self, action: np.ndarray, next_action: np.ndarray = None):
        n = len(action) // 2
        next_left = next_action[:n] if next_action is not None else None
        next_right = next_action[n:] if next_action is not None else None
        self.left.send_action(action[:n], next_action=next_left)
        self.right.send_action(action[n:], next_action=next_right)

    def reset_filter(self):
        self.left.reset_filter()
        self.right.reset_filter()

    def move_to_eval_pose(self, duration: float = 5.0):
        # Parallel so both arms reach target at the same time.
        self._both("move_to_eval_pose", duration=duration, parallel=True)
