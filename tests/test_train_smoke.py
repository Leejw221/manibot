"""학습 경로를 끝에서 끝까지 한 번 돌린다.

convert 가 만들어내는 것과 같은 모양(zarr + config.json)의 작은 데이터셋을 만들고
scripts/train.py 를 실제 진입점으로 실행한다. 이식이 여러 단계로 나뉘어 있어서,
뒤 단계가 앞 단계를 깨뜨리면 여기서 잡힌다.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from manibot.datasets.replay_buffer import ReplayBuffer

S, A, HW, FPS = 7, 7, 32, 20
EPISODES = [40, 35, 30]


def build_dataset(root: Path):
    shutil.rmtree(root, ignore_errors=True)
    buf = ReplayBuffer.create_from_path(str(root), mode="a")
    states, actions = [], []
    for i, T in enumerate(EPISODES):
        st = np.random.randn(T, S).astype(np.float32)
        ac = np.random.randn(T, A).astype(np.float32)
        buf.add_episode({
            "observation.state": st,
            "action": ac,
            "observation.images.main": np.random.randint(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "observation.images.wrist": np.random.randint(0, 255, (T, HW, HW, 3), dtype=np.uint8),
            "episode_index": np.full(T, i, dtype=np.int64),
            "timestamp": np.arange(T, dtype=np.float32) / FPS,
        })
        states.append(st), actions.append(ac)

    def stat(a):
        return {"mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    imgs = ["observation.images.main", "observation.images.wrist"]
    json.dump({
        "repo_id": None,
        "stats": {"observation.state": stat(np.concatenate(states)),
                  "action": stat(np.concatenate(actions))},
        "num_frames": int(buf.n_steps), "num_episodes": len(EPISODES),
        "features": {"observation.state": {"dtype": "float32", "shape": [S]},
                     "action": {"dtype": "float32", "shape": [A]},
                     **{k: {"dtype": "image", "shape": [HW, HW, 3]} for k in imgs}},
        "camera_keys": imgs, "video_keys": [], "image_keys": imgs,
        "fps": FPS, "tasks": {0: "fake"},
    }, open(root / "config.json", "w"), indent=2)


def run_train(tmp, root, steps, extra=()):
    cmd = [
        sys.executable, "-m", "manibot.scripts.train",
        "task=piper_cube_stack", "policy=diffusion", "device=cpu",
        f"task.dataset_root={root}", "task.dataset_repo_id=null", f"task.fps={FPS}",
        "resize_shape=[32,32]", "crop_shape=[28,28]",
        f"train.steps={steps}", "train.batch_size=2", "train.num_workers=0",
        "train.log_freq=2", f"train.save_freq={steps}",
        "policy.unet.down_dims=[64,128]", "wandb.enable=false",
        f"output_dir={tmp}/out", f"hydra.run.dir={tmp}", *extra,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout + r.stderr


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    root = tmp / "ds"
    build_dataset(root)
    print(f"  ✓ 합성 데이터셋 {sum(EPISODES)} frames · {len(EPISODES)} episodes")

    out = run_train(tmp, root, steps=4)
    assert "Training completed" in out, out[-2000:]
    assert "loss:" in out and "grad_norm:" in out, "loss/grad_norm 이 로그에 없다"
    ckpts = sorted((tmp / "out" / "checkpoints").glob("step_*"))
    assert ckpts, "체크포인트가 저장되지 않았다"
    print(f"  ✓ 학습 4스텝 · 체크포인트 {ckpts[-1].name}")

    out = run_train(tmp, root, steps=6, extra=("resume=true",))
    assert "Resumed from checkpoint at step 4" in out, out[-2000:]
    print("  ✓ 체크포인트에서 재개 (step 4 -> 6)")
