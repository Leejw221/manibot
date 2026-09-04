"""AgileX NERO 양팔 플랫폼을 MuJoCo 로 세운다 — ROS 없이.

**왜 이렇게 하나**
  · 벤더 repo(`agilexrobotics/agx_arm_urdf`)에는 **단일 7축 팔**만 있다. 사진의 양팔 플랫폼
    (토르소 + 2팔)은 조립본이 없어 여기서 만든다.
  · xacro 3단(arm -> flange -> gripper)에 **매크로·변수가 하나도 없다**[확인] -> 텍스트로 합칠 수
    있어 ROS/xacro 도구가 필요 없다.
  · 시각 메시가 .dae(COLLADA)라 MuJoCo 가 못 읽는데, **충돌용 STL 이 16k~37k 삼각형으로
    full-detail**이라 재질만 없을 뿐 형상이 같다[실측] -> STL 을 시각·충돌 양쪽에 쓴다.
  · MuJoCo 의 URDF 임포터는 mesh 경로에서 **파일명만** 취한다 -> `meshdir` 를 메시 폴더로 준다.

**기하 [URDF 실측]**
  joint1(요)·joint2(피치)가 base_link 기준 z=0.138 에서 교차 = **어깨축**
  상완 0.31 · 전완 0.27 · 손목~플랜지 0.0235 · 최대 도달 약 0.60 m · 팔 질량 3.52 kg
  그리퍼: prismatic `gripper` 0~0.1 m + 손가락 2개 = **2지 그리퍼**

**사용자 지정**: 어깨축 높이 1.08 m · 2지 그리퍼.
**추정값(실측 필요)**: 두 어깨 사이 간격, 팔 장착 각도. 아래 상수 두 개만 고치면 된다.
"""
import argparse
import os
import xml.etree.ElementTree as ET

# 메시·xacro 는 워크스페이스 밖 형제 디렉토리에 있다. 머신마다 부모 폴더 이름이 달라서
# (pobi = jungwook_ws · 개인 PC = workspace) 절대경로를 박으면 한쪽에서만 돈다.
# 자산은 이 패키지 안에 있다 — manibot repo 하나만 clone 해도 돌아야 하기 때문이다.
NERO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PKG = "package://agx_arm_description/agx_arm_urdf/"

SHOULDER_Z = 1.08          # [사용자 지정 2026-09-01] 어깨축 높이 (m)
SHOULDER_DZ = 0.138        # [URDF 실측] base_link -> 어깨축(joint1·joint2 교차점)

# ⚠️ 아래 둘은 **추정**이다 — 실물에서 재서 고칠 것.
SHOULDER_SPAN = 0.20       # [사용자 지정 2026-09-01] 어깨축 사이 거리(m)
TORSO_W = 0.12             # 몸통 폭(m). **어깨 간격보다 좁아야** 팔이 안 뚫는다 —
                           # span 0.20 에 폭 0.20 을 줬더니 팔이 82mm 침투했다 [실측].
                           # 팔 링크 반경이 약 5cm 라 span - 2*5cm 이하로 둔다.
# ── 실물 몸체 메시 (WeGo_Nero_Body.stl → meshes/body.stl) ───────────────────
# 실측 [메시 실측 2026-09-04]: 팔 장착면 2개가 메시 좌표 x=-0.159 / +0.041 —
# **간격 0.200 으로 SHOULDER_SPAN 과 일치**하고 법선이 ±x(수직)이라 MOUNT_TILT=0 도 맞다.
# 메시의 좌우축이 x 라서 시뮬의 좌우축(y)에 맞추려면 z축으로 +90° 돌린다.
BODY_MESH  = "body.stl"
BODY_SCALE = 0.001                      # STL 단위가 mm
BODY_CTR   = (-0.059, 0.195, 0.109)     # 두 장착면의 중점 (메시 좌표)
BODY_FLOOR = -0.456                     # 바닥 플랜지 밑면 (메시 좌표)
# 충돌은 메시를 쓰지 않는다 — 기둥에 트러스 구멍이 있어 볼록껍질로 근사하면 구멍이
# 메워져 실제보다 뚱뚱해진다. 팔이 몸통에 닿는지만 보면 되므로 박스 두 개가 더 정확하다.
BODY_COLL  = (((-0.119, 0.141, -0.440), (0.000, 0.216, -0.020)),   # 세로 기둥
              ((-0.161, 0.131, -0.020), (0.042, 0.258,  0.195)))   # 상단 헤드

MOUNT_TILT = 0.0           # **0도.** 베이스는 윗면 철판에 평평하게(수직으로) 붙는다 —
                           # 기울여 달면 구조가 안 맞는다 [사용자 지적 2026-09-01]
                           # (사용자 지시 2026-09-01). 25도로 벌렸더니 허수아비 형태가 됐다.
                           # 참고로 벌림은 성능과 거의 무관하다: span 0.20 에서 tilt 를 훑어 고른 값 —
                           # 작업대 도달은 tilt 와 거의 무관하고(각 팔이 y -0.65~+0.45,
                           # 양팔 겹침 0.8~0.9m) **자기충돌률**만 달라진다:
                           # 0도 19.1% · 22도 10.4% · 40도 3.9%(단 손이 1.1m 벌어져
                           # upper body 형태를 벗어난다). 15~30도는 노이즈 안에서 평평 [실측]
# 장착 자세: base 의 +z 를 아래로 뒤집고(roll=pi) 바깥으로 tilt 만큼 벌린다.
# 영자세에서 팔이 위로 뻗던 것을 사진처럼 어깨에서 아래로 내려오게 만든다.


def merged_arm_urdf():
    """arm + flange + 2지 그리퍼를 평문 URDF 하나로 합친다 (xacro 매크로 없음이라 가능)."""
    parts = [
        f"{NERO}/urdf/nero_description.urdf",
        f"{NERO}/urdf/nero_with_gripper_flange_description.xacro",
        f"{NERO}/urdf/nero_with_gripper_description.xacro",
    ]
    out = ET.Element("robot", {"name": "nero"})
    seen = set()
    for p in parts:
        root = ET.fromstring(open(p).read())
        for el in root:
            if el.tag not in ("link", "joint"):
                continue                      # xacro:include 등은 버린다
            key = (el.tag, el.get("name"))
            if key in seen:
                continue
            seen.add(key)
            out.append(el)
    # world 링크와 그 고정 joint 는 우리가 토르소에 붙일 것이므로 제거
    for el in list(out):
        if el.get("name") in ("world", "world_to_base_link"):
            out.remove(el)
    return out


def add_missing_inertial(root, mass=1e-3, inertia=1e-6):
    """질량 없는 링크에 최소 관성을 넣는다.

    그리퍼 xacro 의 `gripper_link` 는 `<link name="gripper_link"/>` 로 **완전히 비어 있는데**
    prismatic joint 가 달려 있다(ROS 에서 흔한 가상 링크). MuJoCo 는 움직이는 body 에
    질량을 요구해 `mass and inertia of moving bodies must be larger than mjMINVAL` 로 죽는다.
    형상·질량이 없는 링크이므로 물리에 영향 없는 최소값을 넣는다."""
    fixed = []
    for link in root.findall("link"):
        if link.find("inertial") is not None:
            continue
        fixed.append(link.get("name"))
        ine = ET.SubElement(link, "inertial")
        ET.SubElement(ine, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(ine, "mass", {"value": str(mass)})
        ET.SubElement(ine, "inertia", dict(ixx=str(inertia), ixy="0", ixz="0",
                                           iyy=str(inertia), iyz="0", izz=str(inertia)))
    return fixed


def fix_meshes(root):
    """package:// 제거 + 시각 메시를 .dae -> .stl 로. MuJoCo 는 파일명만 쓴다."""
    for m in root.iter("mesh"):
        f = m.get("filename", "").replace(PKG, "")
        f = os.path.basename(f)
        if f.endswith(".dae"):
            f = f[:-4] + ".stl"
        m.set("filename", f)
    return root


def build(dual=True, shoulder_z=SHOULDER_Z, span=SHOULDER_SPAN, tilt=MOUNT_TILT,
          mount="side"):
    """mount:
      'side'     = **바닥과 수직인 철판**(몸통 옆면)에 베이스를 붙인다. 베이스 축이 수평으로
                   바깥을 향해, 팔이 옆으로 나갔다가 아래로 꺾인다. 제조사 사진 구조.
      'upright'  = 윗면 철판에 세워 붙임(팔이 위로)
      'inverted' = 윗면 철판에 거꾸로 붙임(팔이 아래로)"""
    arm = merged_arm_urdf()
    empty = add_missing_inertial(arm)
    if empty:
        print(f"  질량 없는 링크에 최소 관성 부여: {empty}")
    arm = fix_meshes(arm)
    robot = ET.Element("robot", {"name": "nero_dual" if dual else "nero_single"})

    mj = ET.SubElement(robot, "mujoco")
    c = ET.SubElement(mj, "compiler")
    c.set("meshdir", f"{NERO}/meshes")
    c.set("balanceinertia", "true")
    c.set("discardvisual", "false")

    # base 를 roll=pi 로 뒤집으므로 어깨 오프셋(+z 0.138)이 **아래**를 향한다
    # -> 어깨를 지정 높이에 두려면 base 를 그만큼 **위로** 올린다. tilt 만큼 짧아진다.
    import math as _m
    # 어깨 오프셋(base 의 +z 로 0.138)이 어느 쪽을 향하느냐에 따라 base 위치를 역산한다.
    # upright  : +z 가 위 -> 어깨는 base 보다 위. roll = ±tilt
    # inverted : +z 가 아래 -> 어깨는 base 보다 아래. roll = pi ∓ tilt
    if mount == "side":
        # 베이스 축이 수평 -> 어깨 오프셋 0.138 이 **옆으로** 간다.
        # 어깨 높이 = 베이스 높이, 어깨 간격 = 베이스 간격 + 2*0.138
        base_z = shoulder_z
        half_y = span / 2
    else:
        sgn = 1.0 if mount == "upright" else -1.0
        base_z = shoulder_z - sgn * SHOULDER_DZ * _m.cos(tilt)
        half_y = span / 2 - sgn * SHOULDER_DZ * _m.sin(tilt)

    # upper body dual-arm — 베이스 **결합면이 바닥과 수직**(몸통 옆면)이어야 한다
    # [사용자 2026-09-01]. 그래야 사람 어깨처럼 몸통 옆에 팔이 달린다.
    # 영자세에서는 팔이 옆으로 뻗지만 joint1 -91°/joint2 -89° 로 몸통 옆에 내려온다
    # (손 z=0.325 y=-0.245) [실측]. 참고: 윗면 장착(upright)은 joint2 한계 ±100° 라
    # 손이 z=0.953m 까지밖에 안 내려가 사람 팔 형태가 불가능하다 [실측].
    # 베이스를 세워 달면(upright) joint2 한계가 ±100° 뿐이라 손이 z=0.953m 까지밖에 안 내려간다
    # (어깨 1.08). 뒤집어 달아야 z=0.325m 로 늘어뜨려진다 [실측 2026-09-01]. 그래서 inverted.
    # 베이스의 결합면이 위를 향하므로 **철판이 베이스 위**에 오고 그 아래에 베이스가 붙는다.
    # 기둥은 철판 아래 가운데. 어깨 간격(20cm)보다 좁아야 팔이 몸통을 안 뚫는다.
    PLATE_T = 0.03
    torso = ET.SubElement(robot, "link", {"name": "torso"})
    ine = ET.SubElement(torso, "inertial")
    ET.SubElement(ine, "origin", {"xyz": f"0 0 {base_z/2:.4f}", "rpy": "0 0 0"})
    ET.SubElement(ine, "mass", {"value": "25"})
    ET.SubElement(ine, "inertia", dict(ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1"))
    body_mesh = os.path.join(NERO, "meshes", BODY_MESH)
    if mount == "side" and os.path.exists(body_mesh):
        # 메시 좌표 p 를 링크 좌표로: Rz(90°)·p + t. 두 장착면의 중점이 (0,0,base_z) 에 오게 t 를 잡는다.
        rot = lambda p: (-p[1], p[0], p[2])
        rc = rot(BODY_CTR)
        t = (-rc[0], -rc[1], base_z - rc[2])
        vis = ET.SubElement(torso, "visual")
        ET.SubElement(vis, "origin", {"xyz": f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}",
                                      "rpy": f"0 0 {_m.pi/2:.6f}"})
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh",
                      {"filename": BODY_MESH,
                       "scale": f"{BODY_SCALE} {BODY_SCALE} {BODY_SCALE}"})
        for lo, hi in BODY_COLL:
            a, b = rot(lo), rot(hi)
            ctr = [(a[i] + b[i]) / 2 + t[i] for i in range(3)]
            size = [abs(b[i] - a[i]) for i in range(3)]
            c = ET.SubElement(torso, "collision")
            ET.SubElement(c, "origin", {"xyz": f"{ctr[0]:.4f} {ctr[1]:.4f} {ctr[2]:.4f}",
                                        "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(c, "geometry"), "box",
                          {"size": f"{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}"})
        print(f"  몸통: 실물 메시 {BODY_MESH} · 바닥 z={t[2] + BODY_FLOOR:.3f} m (= 받침대 높이)")
    else:
        body_w = span if mount == "side" else TORSO_W
        for tag in ("visual", "collision"):
            col = ET.SubElement(torso, tag)                    # 세로 기둥 (바닥 ~ 철판)
            ET.SubElement(col, "origin", {"xyz": f"0 0 {base_z/2:.4f}", "rpy": "0 0 0"})
            ET.SubElement(ET.SubElement(col, "geometry"), "box",
                          {"size": f"0.20 {body_w:.3f} {base_z:.4f}"})
            if mount == "inverted":                            # 베이스 **위**의 어깨 철판
                pl = ET.SubElement(torso, tag)
                ET.SubElement(pl, "origin",
                              {"xyz": f"0 0 {base_z + PLATE_T/2:.4f}", "rpy": "0 0 0"})
                ET.SubElement(ET.SubElement(pl, "geometry"), "box",
                              {"size": f"0.22 {span + 0.10:.3f} {PLATE_T:.3f}"})

    import math
    if mount == "side":
        # roll ±90° : base 의 +z 를 각각 -y / +y (바깥)로 돌린다
        sides = ([("right", -half_y, +math.pi / 2 - tilt),
                  ("left", +half_y, -math.pi / 2 + tilt)] if dual
                 else [("right", 0.0, math.pi / 2)])
    else:
        base_roll = 0.0 if mount == "upright" else math.pi
        s = 1.0 if mount == "upright" else -1.0
        sides = ([("right", -half_y, base_roll - s * tilt), ("left", +half_y, base_roll + s * tilt)]
                 if dual else [("right", 0.0, base_roll)])
    for name, y, roll in sides:
        for el in arm:
            e = ET.fromstring(ET.tostring(el))
            e.set("name", f"{name}_{e.get('name')}")
            for k in ("parent", "child"):
                for sub in e.findall(k):
                    sub.set("link", f"{name}_{sub.get('link')}")
            robot.append(e)
        j = ET.SubElement(robot, "joint",
                          {"name": f"{name}_mount", "type": "fixed"})
        ET.SubElement(j, "origin", {"xyz": f"0 {y:.4f} {base_z:.4f}",
                                    "rpy": f"{roll:.4f} 0 0"})
        ET.SubElement(j, "parent", {"link": "torso"})
        ET.SubElement(j, "child", {"link": f"{name}_base_link"})
    return robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="한 팔만")
    ap.add_argument("--shoulder-z", type=float, default=SHOULDER_Z)
    ap.add_argument("--span", type=float, default=SHOULDER_SPAN, help="어깨축 사이 거리(m)")
    ap.add_argument("--tilt-deg", type=float, default=None, help="팔 벌림(도)")
    ap.add_argument("--mount", choices=("side","upright","inverted"), default="side")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nero.urdf"))
    args = ap.parse_args()

    import math as _mm
    tilt = MOUNT_TILT if args.tilt_deg is None else _mm.radians(args.tilt_deg)
    root = build(dual=not args.single, shoulder_z=args.shoulder_z, span=args.span,
                 tilt=tilt, mount=args.mount)
    ET.indent(root, "  ")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(ET.tostring(root, encoding="unicode"))
    print(f"URDF 작성: {args.out}")

    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(args.out)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    print(f"✅ MuJoCo 로드  body {m.nbody} · joint {m.njnt} · dof {m.nv} · geom {m.ngeom}")
    print(f"   총질량 {sum(m.body_mass):.2f} kg")
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_link1")
           for s in (["right"] if args.single else ["right", "left"])]
    print(f"   어깨축 높이 {d.xpos[ids[0]][2]:.4f} m  (목표 {args.shoulder_z})")
    if len(ids) == 2:
        # span 은 **장착면** 간격이고 어깨축은 거기서 SHOULDER_DZ 만큼 더 바깥이다.
        # 예전엔 이 줄이 "어깨축 간격 (목표 span)"이라 둘을 같은 것처럼 보여 오해를 샀다.
        gap = abs(d.xpos[ids[0]][1] - d.xpos[ids[1]][1])
        print(f"   장착면 간격 {args.span:.4f} m  [메시 실측 0.200 과 일치]")
        print(f"   어깨축 간격 {gap:.4f} m  (= 장착면 {args.span} + 2×{SHOULDER_DZ})")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    print(f"   joints({len(names)}) {names}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# robosuite 자산 (robot.xml · gripper.xml)
#
# robosuite 는 **팔과 그리퍼를 분리**해 다룬다 — 그리퍼는 `{arm}_hand` body 에 붙는다.
# body/site 이름 규약은 Panda·Tiago 의 자산을 읽어 확인했다 [코드 확인 2026-09-04]:
#   로봇  : <body name="base"> ... <site name="{arm}_center"> ... <body name="{arm}_hand">
#   그리퍼: <body name="right_gripper"> + eef · finger + grip_site / ft_frame / ee_x,y,z
# ─────────────────────────────────────────────────────────────────────────────

# link7 기준 그리퍼 부착점 [MJCF 변환 결과에서 실측 2026-09-05].
# gripper_flange(0.031,0,-0.0235) 에 gripper_base_joint(0,0,0.006) 을 합성한 값과 일치한다.
HAND_POS = "0.037 0 -0.0235"
HAND_QUAT = "0.5 -0.5 0.5 -0.5"
# 그리퍼 손가락 [변환된 MJCF 에서 실측]. joint ±0.05 -> 개구 0.10 m (STEP 파트명 "100mm爪夹").
FINGERS = (("leftfinger", "gripper_link1", "gripper_joint1", "0 0 1", "0 0.05"),
           ("rightfinger", "gripper_link2", "gripper_joint2", "0 0 -1", "-0.05 0"))
FINGER_POS = "0 0 0.138"
FINGER_QUAT = ("0 0 0.707107 0.707107", "0.707107 0.707107 0 0")
GRIP_Z = 0.0998          # 두 손가락 사이 중심 = 파지점


def _drop(parent, pred):
    for c in list(parent):
        if pred(c):
            parent.remove(c)


def robot_mjcf(src_mjcf):
    """팔+몸통 MJCF 를 robosuite 규약으로 고친다. 그리퍼는 떼어내고 부착점만 남긴다."""
    root = ET.parse(src_mjcf).getroot()
    wb = root.find("worldbody")

    # 1) 그리퍼를 떼어낸다. gripper_base geom 도 그리퍼 쪽 자산이므로 뺀다.
    for b in wb.iter("body"):
        _drop(b, lambda c: c.tag == "body" and "gripper_link" in (c.get("name") or ""))
        _drop(b, lambda c: c.tag == "geom" and c.get("mesh") == "gripper_base")

    # 2) link7 에 그리퍼 부착점 {arm}_hand 를 만든다.
    for s in ("right", "left"):
        for b in wb.iter("body"):
            if b.get("name") == f"{s}_link7":
                ET.SubElement(b, "body", {"name": f"{s}_hand",
                                          "pos": HAND_POS, "quat": HAND_QUAT})
            if b.get("name") == f"{s}_link1":
                ET.SubElement(b, "site", {"name": f"{s}_center", "pos": "0 0 0",
                                          "size": "0.01", "rgba": "0 0 1 0"})

    # 3) 몸통은 world 에 융합돼 있다(fixed joint). robosuite 는 루트 body 를 요구하므로 감싼다.
    base = ET.Element("body", {"name": "base", "pos": "0 0 0"})
    for c in list(wb):
        if c.tag in ("geom", "body", "site"):
            wb.remove(c)
            base.append(c)
    wb.append(base)

    # 4) 메시 경로: meshdir 을 없애고 파일 경로에 폴더를 넣는다 (gripper_mjcf 의 주석 참조).
    for c in root.iter("compiler"):
        c.attrib.pop("meshdir", None)
    for m in root.iter("mesh"):
        f = m.get("file")
        if f:
            m.set("file", "meshes/" + os.path.basename(f))

    # 4-b) 볼록껍질 artifact 제외. MuJoCo 는 메시 충돌을 볼록껍질로 근사해서 인접·근접
    #      링크의 껍질이 겹친다 — 영자세에서 base↔link1 5mm, link5↔link7 28.5mm(사이 link6
    #      이 짧다)가 잡히는데 실제 간섭이 아니다 [실측 2026-09-01].
    #      ⚠️ link7↔몸통은 **빼지 않는다** — 그건 진짜 제약이라 자세로 피해야 한다.
    con = ET.SubElement(root, "contact")
    for s_ in ("right", "left"):
        ET.SubElement(con, "exclude", {"body1": "base", "body2": f"{s_}_link1"})
        for a, b in ((5, 7), (4, 6), (5, 6), (6, 7)):
            ET.SubElement(con, "exclude", {"body1": f"{s_}_link{a}", "body2": f"{s_}_link{b}"})

    # 5) 관절 물리값은 제조사 MuJoCo 모델(agx_arm_sim/mujoco/agilex_arm/agilex_nero/nero.xml)
    #    을 따른다 — URDF 에는 없는 값이다. damping 은 제조사도 nero.xml 에선 0 이다
    #    (damping 2000 은 위치제어용 nero_arm.xml 쪽) [대조 2026-09-05].
    for j in root.iter("joint"):
        if "gripper" in (j.get("name") or ""):
            continue
        j.set("armature", "0.005")
        j.set("frictionloss", "0.3")

    # 6) 액추에이터는 robosuite 가 토크 제어를 전제하므로 motor 로. 그리퍼 것은 뺀다.
    _drop(root, lambda c: c.tag in ("actuator", "equality"))
    act = ET.SubElement(root, "actuator")
    for s in ("right", "left"):
        for i in range(1, 8):
            ET.SubElement(act, "motor", {"name": f"torq_{s}_j{i}", "joint": f"{s}_joint{i}",
                                         "ctrllimited": "true", "ctrlrange": "-100 100"})
    # 그리퍼 관절도 URDF 에서 딸려온다 — 그리퍼 자산 쪽에서 다시 정의하므로 여기선 제거.
    for b in wb.iter("body"):
        _drop(b, lambda c: c.tag == "joint" and "gripper" in (c.get("name") or ""))
    return root


def gripper_mjcf(meshdir="meshes"):
    """robosuite 규약 그리퍼. Panda 의 gripper.xml 과 같은 body/site 이름을 쓴다.

    ⚠️ `meshdir` 을 쓰지 않는다 — robosuite 는 XML 을 문자열로 다시 조립해
    `from_xml_string` 으로 읽기 때문에 meshdir 의 기준 폴더가 사라진다. Panda 처럼
    파일 경로 자체에 폴더를 넣어야 robosuite 가 절대경로로 바꿔준다 [실측 2026-09-05].
    """
    r = ET.Element("mujoco", {"model": "nero_gripper"})
    ET.SubElement(r, "compiler", {"angle": "radian"})
    a = ET.SubElement(r, "asset")
    for n, f in (("nero_gripper_base", "gripper_base.stl"),
                 ("nero_finger1", "gripper_link1.stl"), ("nero_finger2", "gripper_link2.stl")):
        ET.SubElement(a, "mesh", {"name": n, "file": f"{meshdir}/{f}"})
    act = ET.SubElement(r, "actuator")
    for i, (_, _, jn, _, rng) in enumerate(FINGERS, start=1):
        ET.SubElement(act, "position", {"name": f"gripper_finger_joint{i}", "joint": jn,
                                        "kp": "1000", "ctrlrange": rng, "forcerange": "-20 20"})
    wb = ET.SubElement(r, "worldbody")
    g = ET.SubElement(wb, "body", {"name": "right_gripper", "pos": "0 0 0"})
    ET.SubElement(g, "inertial", {"pos": "0 0 0.03", "mass": "0.45",
                                  "diaginertia": "0.00093 0.00071 0.00039"})
    ET.SubElement(g, "geom", {"type": "mesh", "mesh": "nero_gripper_base", "group": "1",
                              "contype": "0", "conaffinity": "0", "rgba": ".7 .7 .72 1"})
    ET.SubElement(g, "geom", {"type": "mesh", "mesh": "nero_gripper_base", "group": "0"})
    ET.SubElement(g, "site", {"name": "ft_frame", "pos": "0 0 0", "size": "0.01",
                              "rgba": "1 0 0 0", "type": "sphere"})
    eef = ET.SubElement(g, "body", {"name": "eef", "pos": f"0 0 {GRIP_Z}"})
    ET.SubElement(eef, "site", {"name": "grip_site", "pos": "0 0 0", "size": "0.01",
                                "rgba": "1 0 0 0.5", "type": "sphere", "group": "1"})
    ET.SubElement(eef, "site", {"name": "grip_site_cylinder", "pos": "0 0 0",
                                "size": "0.005 10", "rgba": "0 1 0 0.3",
                                "type": "cylinder", "group": "1"})
    for nm, pos in (("ee_x", "0.1 0 0"), ("ee_y", "0 0.1 0"), ("ee_z", "0 0 0.1")):
        ET.SubElement(eef, "site", {"name": nm, "pos": pos, "size": "0.005 .1",
                                    "rgba": "0 0 0 0", "type": "cylinder", "group": "1"})
    # robosuite 는 매 step 마다 말단 힘·토크 센서를 읽는다. 없으면 step 에서 죽는다.
    sen = ET.SubElement(r, "sensor")
    ET.SubElement(sen, "force", {"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sen, "torque", {"name": "torque_ee", "site": "ft_frame"})

    for i, ((bn, _, jn, axis, rng), quat) in enumerate(zip(FINGERS, FINGER_QUAT), start=1):
        b = ET.SubElement(g, "body", {"name": bn, "pos": FINGER_POS, "quat": quat})
        ET.SubElement(b, "inertial", {"pos": "0 -0.049 0.01", "mass": "0.025",
                                      "diaginertia": "7.4e-05 7.8e-06 7.5e-05"})
        # damping·armature·frictionloss 는 Panda 그리퍼와 같은 크기로 둔다 — 이보다 작으면
        # 손가락이 접촉에서 밀려 파지가 안 잡힌다 [panda_gripper.xml 대조 2026-09-05].
        ET.SubElement(b, "joint", {"name": jn, "type": "slide", "axis": axis, "range": rng,
                                   "damping": "100", "armature": "1.0", "frictionloss": "1.0"})
        ET.SubElement(b, "geom", {"name": f"{bn}_visual", "type": "mesh",
                                  "mesh": f"nero_finger{i}", "group": "1",
                                  "contype": "0", "conaffinity": "0", "rgba": ".8 .8 .82 1"})
        # 이름을 붙여야 GripperModel._important_geoms 가 파지 판정에 쓸 수 있다.
        ET.SubElement(b, "geom", {"name": f"{bn}_collision", "type": "mesh",
                                  "mesh": f"nero_finger{i}", "group": "0",
                                  "friction": "1 0.05 0.001", "condim": "4",
                                  "solimp": "0.95 0.99 0.001", "solref": "0.005 1"})
    return r
