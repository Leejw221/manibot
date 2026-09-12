"""여러 zarr 데이터셋을 하나로 잇는다 — APO 는 시연과 개입 데이터를 **한 배치에서** 섞어야 한다.

`create_dataset` 이 `task.dataset_root` 하나만 읽으므로, 라운드 학습에 쓸 데이터는 미리
합쳐 둔다. 프레임 축으로 잇고 `episode_index`·`episode_ends` 를 오프셋한다.

`action_mode` 가 없는 데이터셋(시연)은 **LABEL_ROLLOUT 으로 채운다.** 그래도 되는 이유:
개입이 없는 에피소드는 recovery confidence 가 전부 0 이라 `has_intervention=False` 로
`correct` 풀에 들어가고, 거기서 시연과 "개입 없이 끝난 롤아웃" 은 어차피 같은 취급을 받는다
(APO 공개 코드도 둘을 `correct` 한 버킷에 합친다).

사용:
    python -m manibot.scripts.merge_zarr \
        --out data/square_apo_r1_zarr \
        data/square_ph50_zarr data/square_deploy_r1_zarr
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import zarr

from manibot.utils.intervention_labels import LABEL_ROLLOUT

CHUNK = 512


def merge(srcs, out):
    roots = [zarr.open(str(s), "r") for s in srcs]
    fields = set(roots[0]["data"].keys())
    for r in roots[1:]:
        fields &= set(r["data"].keys())
    fields = sorted(fields | {"action_mode"})          # action_mode 는 없으면 채운다
    lens = [r["meta"]["episode_ends"][-1] for r in roots]
    total = int(sum(lens))
    print(f"입력 {len(srcs)}개 · 프레임 {lens} -> {total}\n필드: {fields}")

    dst = zarr.open(str(out), "w")
    d, m = dst.require_group("data"), dst.require_group("meta")

    for f in fields:
        ref = next((r["data"][f] for r in roots if f in r["data"]), None)
        shape = (total,) + tuple(ref.shape[1:])
        chunks = (CHUNK,) + tuple(ref.shape[1:])
        arr = d.zeros(f, shape=shape, chunks=chunks, dtype=ref.dtype, overwrite=True)
        off = 0
        for r, n in zip(roots, lens):
            n = int(n)
            if f in r["data"]:
                src = r["data"][f]
                for i in range(0, n, CHUNK):
                    j = min(i + CHUNK, n)
                    block = src[i:j]
                    if f == "episode_index":
                        block = block + sum(              # 앞 데이터셋들의 에피소드 수만큼 민다
                            len(rr["meta"]["episode_ends"]) for rr in roots[:roots.index(r)])
                    arr[off + i:off + j] = block
            else:
                assert f == "action_mode", f"'{f}' 가 일부 데이터셋에만 있다 — 채울 규칙이 없다"
                arr[off:off + n] = LABEL_ROLLOUT
                print(f"  {Path(str(r.store.path)).name}: action_mode 없음 -> LABEL_ROLLOUT 로 채움")
            off += n

    ends, off, nep = [], 0, 0
    for r, n in zip(roots, lens):
        ends.extend((np.asarray(r["meta"]["episode_ends"]) + off).tolist())
        off += int(n); nep += len(r["meta"]["episode_ends"])
    m.array("episode_ends", np.asarray(ends, dtype=np.int64), overwrite=True)

    # ⚠ config.json 의 **stats 는 첫 데이터셋 것을 그대로** 쓴다.
    #    fine-tuning 은 pi_ref 와 정규화가 같아야 한다 — 병합 데이터로 다시 계산하면
    #    입력·출력 스케일이 달라져 불러온 가중치가 다른 뜻을 갖게 된다.
    #    실측(2026-09-12): 개입 데이터가 시연 min/max 를 벗어나는 비율은 action 0.03%,
    #    state 0.5% 이고 정규화 후 최대 1.20 — 무시할 수준이다.
    cfg = json.load(open(Path(srcs[0]) / "config.json"))
    cfg["num_frames"] = total
    cfg["num_episodes"] = nep
    cfg["merged_from"] = [str(s) for s in srcs]
    cfg["stats_from"] = str(srcs[0])
    json.dump(cfg, open(Path(out) / "config.json", "w"), indent=4)
    print(f"\n저장: {out}   에피소드 {nep} · 프레임 {total}")
    print(f"  ⚠ stats 는 {srcs[0]} 것을 그대로 사용 (pi_ref 와 정규화 일치)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("srcs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    merge(a.srcs, a.out)
