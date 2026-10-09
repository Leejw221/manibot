"""SIRIUS 의 weighted BC — 프레임 라벨 · 시점별 가중.  공개 코드를 그대로 옮긴다.

⚠ **논문과 공개 코드가 다르다. 코드를 따른다.**
  논문 §IV-D : "We set P*(preintv)=0, essentially nullifying ..." · P*(demo)=P(demo)
  코드        : weight_preintv = 0.002 · w_demos = 1 (P*/P 가 아니라 상수)
  [코드 확인 2026-09-28, ~/Downloads/sirius/robomimic/utils/dataset.py:951 _sirius_reweight]

코드 그대로:
    라벨    프레임마다 {-1 demo, 0 rollout, 1 intv, -10 preintv}. preintv = 각 개입 시작
            직전 15 프레임, 앞선 개입을 만나면 멈춘다 (dataset.py:1246 _get_intv_labels)
            -> 우리 relabel_preintv(k=15) 와 프레임 단위로 같다 (r1·r2 불일치 0, 2026-09-28)
    비율    샘플의 **첫 시점 라벨**로 센다 (action_mode_selection=0, dataset.py:892)
    가중    w_demos=1 · w_intvs=0.5/r_intv · w_rollouts=(1-0.5-r_demo-0.002)/r_rollout
            · w_pre_intvs=0.002/r_preintv
    적용    가중 배열이 (N, seq_length) — **시점마다 자기 라벨의 가중** (dataset.py:1047,
            same_weight_for_seq=False). 손실은 action_loss (B,T) *= w ; mean() (bc.py:664)
공식 설정 [exps/sirius/sirius.json]: normalize=false · use_weighted_sampler=false
    -> 정규화 없음 · 자연 분포 샘플링 (finetune.balanced=null).

DP 로 옮기며 바뀐 것은 샘플별 손실뿐이다: -log pi(a_t|s) 대신 시점 t 의 노이즈 예측 MSE
(행동 차원 평균). 가중을 뺀 손실은 LeRobot DP 의 기본 손실(t 하나, 전 원소 MSE 평균)과 같다.

기본값은 **이전 실로봇 실험(PDF)·mani_sim 방식**이다 [사용자 결정 2026-09-29: "그게 잘 되었던 방식"]:
    chunk_mean  청크 16칸 가중의 평균을 샘플 가중으로 (PDF 19쪽 · mani_sim diffusion_trainer.py:507)
    normalize   sum(w*l)/sum(w) — 배치 가중 평균으로 나눈다 (PDF 12쪽 · 같은 파일 :511)
    음수 가중    0 으로 자른다 (PDF 14쪽 "Clipped to 0"). 원문 식은 P(demo) > 0.498 이면 음수가 된다.
둘 다 끄면 SIRIUS 원문 코드(칸별 가중 · 정규화 없음)다.

chunk_label="majority" [사용자 결정 2026-10-05]: 샘플(청크) 하나에 클래스 하나.
    첫 칸으로 세고 칸 평균으로 가중하면 세는 단위와 가중 단위가 달라 P* 가 실제 손실 몫이
    되지 않는다(8차 ① r1 preintv 목표 0.2% -> 실제 2.66%). 16칸 과반으로 정하고, 동률이면
    preintv 가 아닌 쪽. 가중 w(c)=P*(c)/P(c) 를 16칸에 같은 값으로 준다.
    개입 뒤 제어를 돌려주지 않는다고 가정하므로 한 창의 라벨은 많아야 둘(robot|preintv ·
    preintv|intv) — preintv 가 끼지 않은 동률은 생기지 않는다.
p_star_auto [사용자 2026-10-05, ④ "전체 자율 수행 구간에 pre-intervention 가중"]:
    robot 과 preintv 를 한 클래스로 보고 둘이 합쳐 목표 몫 p_star_auto. 남는 몫은 어디에도 주지
    않는다 — normalize 가 demo 와 intv 에 비례로 나눠 demo:intv 비가 ① 과 같게 남는다.
"""

import numpy as np
import torch

from manibot.policies.diffusion_ops import prepare_cond, unet_out

__all__ = ["SiriusLoss", "LABELS", "frame_labels", "action_windows"]

LABELS = {"demo": -1, "robot": 0, "intv": 1, "preintv": -10}


def frame_labels(dataset_root, n_demo, use_preintv=True):
    """(프레임 라벨 (F,), 에피소드 번호 (F,)). 앞 n_demo 에피소드는 demo, 나머지는 action_mode
    (0 robot · 1 intv) 에 개입 시작 직전 15 프레임을 preintv 로 덧씌운다."""
    import zarr

    from manibot.utils.intervention_labels import LABEL_PREINTV, relabel_preintv
    z = zarr.open(str(dataset_root), "r")["data"]
    ep = np.asarray(z["episode_index"]).ravel()
    mode = np.asarray(z["action_mode"]).ravel()
    fl = np.empty(len(ep), dtype=np.int64)
    for e in np.unique(ep):
        s = ep == e
        if e < n_demo:
            fl[s] = LABELS["demo"]
            continue
        # 0·1 밖의 값은 relabel_preintv 의 LABEL_PREINTV(2)와 섞이거나 LABELS 에 없는 클래스가 된다
        if not np.isin(mode[s], (0, 1)).all():
            raise ValueError(f"에피소드 {e} 의 action_mode 에 0·1 밖의 값: {np.unique(mode[s]).tolist()}")
        if not use_preintv:
            fl[s] = mode[s]
        else:
            m = relabel_preintv(mode[s], k=15)   # fixed_preintv_length
            fl[s] = np.where(m == LABEL_PREINTV, LABELS["preintv"], m)
    return fl, ep


def action_windows(dataset, ep):
    """(N, H) 샘플 i 의 액션 창 프레임 인덱스.  데이터셋의 _get_query_indices 를 그대로 쓴다
    (precompute_apo_labels 와 같은 이유 — 직접 재현하면 어긋난다)."""
    from lerobot.utils.constants import ACTION
    return np.stack([np.asarray(dataset._get_query_indices(i, int(ep[i]))[0][ACTION])
                     for i in range(len(dataset))])


class SiriusLoss:
    def __init__(self, frame_label, act_idx, p_star_intv=0.5, p_star_preintv=0.002,
                 chunk_mean=True, normalize=True, p_star_robot=None,
                 chunk_label="first", p_star_auto=None):
        """frame_label: (F,) 프레임 라벨.  act_idx: (N, H) 샘플 i 의 액션 창 프레임 인덱스."""
        assert chunk_label in ("first", "majority"), chunk_label
        fl = np.asarray(frame_label, dtype=np.int64)
        act_idx = np.asarray(act_idx, dtype=np.int64)
        self.lab = fl[act_idx]                                   # (N, H)
        if chunk_label == "majority":
            vals = np.array(list(LABELS.values()))
            cnt = (self.lab[:, :, None] == vals).sum(1).astype(np.float64)   # (N, 4)
            cnt[:, vals == LABELS["preintv"]] -= 0.5             # 동률이면 preintv 가 아닌 쪽
            cls = vals[cnt.argmax(1)]
        else:
            cls = self.lab[:, 0]
        N = len(cls)
        self.n = {c: int((cls == v).sum()) for c, v in LABELS.items()}
        self.P = {c: n / N for c, n in self.n.items()}
        self.w_cls = {
            "demo": 1.0,
            "intv": p_star_intv / self.P["intv"] if self.n["intv"] else 0.0,
            # p_star_robot 를 주면 rollout 도 preintv 처럼 목표 몫을 고정한다 (원문은 나머지 몫)
            "robot": ((p_star_robot if p_star_robot is not None
                       else max(0.0, 1.0 - p_star_intv - self.P["demo"] - p_star_preintv)) / self.P["robot"]
                      if self.n["robot"] else 0.0),
            "preintv": p_star_preintv / self.P["preintv"] if self.n["preintv"] else 0.0,
        }
        if p_star_auto is not None:
            assert p_star_robot is None, "p_star_auto 와 p_star_robot 은 함께 쓰지 않는다"
            p_auto = self.P["robot"] + self.P["preintv"]
            self.w_cls["robot"] = self.w_cls["preintv"] = p_star_auto / p_auto if p_auto else 0.0
        lut = {LABELS[c]: w for c, w in self.w_cls.items()}
        if chunk_label == "majority":
            self.w = np.repeat(np.vectorize(lut.get)(cls)[:, None], self.lab.shape[1], axis=1)
            self.w = self.w.astype(np.float32)
        else:
            self.w = np.vectorize(lut.get)(self.lab).astype(np.float32)
            if chunk_mean:
                self.w = np.repeat(self.w.mean(1, keepdims=True), self.w.shape[1], axis=1)
        self.cls = cls
        self.normalize = bool(normalize)

    def __call__(self, policy, batch):
        m = policy.diffusion
        T = m.noise_scheduler.config.num_train_timesteps
        cond, x0 = prepare_cond(policy, batch)
        idx = batch["dataset_index"].cpu().numpy()
        w = torch.as_tensor(self.w[idx], device=x0.device, dtype=x0.dtype)   # (B, H)

        t = torch.randint(0, T, (x0.shape[0],), device=x0.device, dtype=torch.long)
        eps = torch.randn_like(x0)
        e_the, _, _ = unet_out(m, cond, x0, t, eps)
        l_t = ((eps - e_the) ** 2).mean(dim=2)                  # (B, H) 시점별
        # chunk_mean 이면 w 가 칸마다 같아 (w * l_t) 가 w_i * (청크 MSE) 와 같다
        loss = (w * l_t).sum() / w.sum().clamp_min(1e-8) if self.normalize else (w * l_t).mean()

        out = {"w_mean": float(w.mean()), "l_the_mean": float(l_t.mean().detach())}
        lab = torch.as_tensor(self.lab[idx], device=x0.device)
        for c, v in LABELS.items():
            sel = lab == v
            if sel.any():
                out[f"l_the/{c}"] = float(l_t[sel].median().detach())
                out[f"n/{c}"] = int(sel.sum())
        return loss, out
