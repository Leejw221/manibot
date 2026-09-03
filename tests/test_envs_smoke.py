"""시뮬 env 를 만들고 reset/step 1회. robosuite 가 없으면(=[sim] 미설치) 건너뛴다.

MUJOCO_GL=egl 로 실행한다 (오프스크린 렌더).
"""
import numpy as np

OBS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

try:
    import robosuite  # noqa: F401
except ImportError:
    print("  - robosuite 없음 — [sim] 미설치라 건너뛴다")
    raise SystemExit(0)

from manibot.envs.robomimic import make_lowdim_env

for env_name in ["NutAssemblySquare", "DoorCabinet"]:
    try:
        env = make_lowdim_env(env_name, "Panda", OBS)
        obs = env.reset()
        missing = [k for k in OBS if k not in obs]
        assert not missing, f"obs 에 없는 키: {missing}"
        obs, reward, done, info = env.step(np.zeros(env.action_dimension))
        print(f"  ✓ {env_name:20s} action_dim {env.action_dimension} · reward {reward}")
    except Exception as e:
        print(f"  ✗ {env_name:20s} {type(e).__name__}: {e}")

# 커스텀 태스크가 import 부수효과로 robosuite 레지스트리에 올라왔는지
assert "DoorCabinet" in robosuite.ALL_ENVIRONMENTS, "DoorCabinet 이 등록되지 않았다"
print("  ✓ DoorCabinet 레지스트리 등록 확인")
