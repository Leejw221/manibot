"""색 표식 -> 거울상 쌓기 task 의 장면 구성. collect/probe 가 공유한다.

블록을 RoboCasa 물체 레지스트리(식품 198종)로는 못 만든다 -> 모델 XML 에 직접 주입한다.
카메라 주입과 같은 방식이고, 대신 free joint 가 늘어나므로 state 를 패딩해야 한다.
원본 free joint(obj1/obj2)가 qpos 의 **맨 끝**(109,116 / nq=123)이라, worldbody 끝에
붙인 블록의 qpos 는 그 뒤에 오고 패딩은 단순 append 가 된다 [실측 2026-08-28].

빨강(목표대)은 **고정 body** — 자유 물체로 두면 쌓는 중에 밀려 스크립트 실패 요인만 는다.
표식(cue)도 고정: 보기만 하고 만지지 않는다. 따라서 free joint 는 초록·파랑 2개뿐.
"""
import numpy as np

COUNTER_Z = 0.920          # counter_1_front_group_2_top_* 윗면 [실측]
HALF = 0.022               # 4.4cm 큐브. 그리퍼 열림 8.6cm 이라 여유 2.1cm/측
BASE_HALF = 0.030          # 빨강 목표대는 조금 크게

RED_XY = (6.660, -4.250)   # 로봇(6.77,-3.84) 정면 카운터
DX = 0.150                 # 좌우 대칭 간격. 로봇 시점 왼쪽 = +x [실측: left door x 가 더 큼]
GREEN_XY = (RED_XY[0] + DX, RED_XY[1])
BLUE_XY = (RED_XY[0] - DX, RED_XY[1])

CUE_HALF = 0.050           # 학습 해상도 84px 에서도 몇 픽셀은 되도록 크게
CUE_Z = 1.610              # 캐비닛 1단 선반(z≈1.54) 위에 얹히는 높이
CUE_Y = -4.500             # 앞면 y=-4.45. 문 닫힘 은닉 / 열림 노출을 아래에서 실측한다
# 캐비닛 중앙(힌지 6.177/7.057 의 중점 ≈ 6.617)에서 24cm 씩. 중앙 근처면 반대쪽 문만
# 열어도 보이고(0.30m 이전엔 0.06m 였고 초록 116 누수), 너무 벌리면 문틀에 반쯤 가린다.
# 0.18~0.30m 를 훑어 **0.24m 에서 가장 크게 보였다**(105픽셀 vs 0.30m 72픽셀).
# 누수는 0.18m 까지도 0 이다 [실측 2026-08-29].
CUE_X = {"right": 6.377, "left": 6.857}
COLOR = {"green": "0.1 0.75 0.15 1", "blue": "0.1 0.3 0.9 1", "red": "0.85 0.1 0.1 1"}

DOORVIEW = ('<camera name="doorview" pos="6.62 -3.05 1.95" '
            'xyaxes="-1 0 0  0 -0.36 0.93" fovy="42"/>')
# 작업대 정면 뷰 — 사람이 볼 용도
BENCHVIEW = ('<camera name="benchview" pos="6.66 -3.55 1.45" '
             'xyaxes="-1 0 0  0 -0.514 0.857" fovy="45"/>')
# 정책 관측용 글로벌 뷰. 캐비닛 안 표식과 작업대 블록 3개를 **한 장에** 담아야 한다.
# 비스듬히 보면 한쪽 문만 열어도 반대쪽 안이 보여 task 가 깨진다 -> 캐비닛 **정면**에 둔다.
GLOBALVIEW = ('<camera name="globalview" pos="6.617 -3.15 1.80" '
              'xyaxes="-1 0 0  0 -0.3536 0.9354" fovy="55"/>')


def _geoms(name, half, rgba, mass=None):
    """robosuite 규약대로 충돌(group 0) + 시각(group 1) 을 나눈다.
    렌더는 group 1 만 그리고, `inertiagrouprange="0 0"` 이라 질량·관성은 group 0 에서만
    나온다 — 하나로 합치면 둘 중 하나가 깨진다 [실측 2026-08-28]."""
    m = f' mass="{mass}"' if mass is not None else ""
    return (f'<geom name="{name}_g" type="box" size="{half} {half} {half}" '
            f'group="0" rgba="{rgba}" friction="1 0.05 0.001"{m}/>'
            f'<geom name="{name}_vis" type="box" size="{half} {half} {half}" '
            f'group="1" rgba="{rgba}" contype="0" conaffinity="0" mass="0"/>')


def _static(name, xyz, half, rgba):
    return (f'<body name="{name}" pos="{xyz[0]} {xyz[1]} {xyz[2]}">'
            f'{_geoms(name, half, rgba)}</body>')


def _free(name, xyz, half, rgba):
    return (f'<body name="{name}" pos="{xyz[0]} {xyz[1]} {xyz[2]}">'
            f'<freejoint name="{name}_joint"/>'
            f'{_geoms(name, half, rgba, mass=0.05)}</body>')


def add_door_friction(xml, frictionloss=3.0, target="cab_1_front_group_2"):
    """우리가 쓰는 캐비닛 경첩에만 마찰을 넣는다(주방의 다른 문은 그대로).
    원본은 damping="2" 만 있고 frictionloss 가 없어, 손을 놓으면 문이 스스로 닫혀
    표식이 1.3초밖에 안 보였다 [실측 2026-08-28: 유지 86프레임 중 26프레임만 열림].
    실제 주방 문은 놓아도 그 자리에 있으므로 물리적으로도 이쪽이 자연스럽다."""
    import re

    def patch(m):
        tag = m.group(0)
        if "frictionloss" in tag:
            return tag
        return tag[:-2].rstrip() + f' frictionloss="{frictionloss}" />'

    return re.sub(rf'<joint[^>]*{target}_(left|right)doorhinge[^>]*/>', patch, xml)


def build_xml(xml, cue_side, cue_color):
    """카메라 + 빨강 목표대 + 표식(고정) / 초록 + 파랑(자유) 을 주입.
    자유 body 는 **맨 끝**에 붙여야 qpos 패딩이 append 로 끝난다."""
    xml = add_door_friction(xml)
    i = xml.index("<worldbody>") + len("<worldbody>")
    head = ("\n    " + DOORVIEW + "\n    " + BENCHVIEW + "\n    " + GLOBALVIEW
            + "\n    " + _static("blk_red", (*RED_XY, COUNTER_Z + BASE_HALF),
                                 BASE_HALF, COLOR["red"])
            + "\n    " + _static("cue", (CUE_X[cue_side], CUE_Y, CUE_Z),
                                 CUE_HALF, COLOR[cue_color]))
    j = xml.rindex("</worldbody>")
    tail = ("\n    " + _free("blk_green", (*GREEN_XY, COUNTER_Z + HALF), HALF, COLOR["green"])
            + "\n    " + _free("blk_blue", (*BLUE_XY, COUNTER_Z + HALF), HALF, COLOR["blue"])
            + "\n    ")
    return xml[:i] + head + xml[i:j] + tail + xml[j:]


def pad_state(state, nq_old, nv_old):
    """[time, qpos(nq), qvel(nv)] 에 자유 블록 2개(qpos 7 x2, qvel 6 x2)를 끼워 넣는다.
    값은 아무거나 넣어도 되고(아래에서 sim 이 body pos 로 덮어씀) 쿼터니언만 유효해야 한다."""
    t, qpos, qvel = state[0], state[1:1 + nq_old], state[1 + nq_old:]
    assert len(qvel) == nv_old, (len(qvel), nv_old)
    add_q = np.tile([0, 0, 0, 1, 0, 0, 0], 2).astype(float)
    return np.concatenate([[t], qpos, add_q, qvel, np.zeros(12)])
