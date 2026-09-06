"""robocasa MJCF 자산을 축척 변환한다.

**왜 필요한가**: 전자레인지 실물 크기(폭 0.547 m)는 NERO 의 작업 영역에 안 들어간다 —
문을 여는 동안 손잡이가 x 로 0.345 m 움직이는데 한 팔의 쓸 만한 폭이 그만큼이 안 된다.
축척을 줄이면 이동량도 같은 비율로 줄어 성립한다 [실측 2026-09-06]:
    축척 1.00 (0.55m) 여유  4°  ·  0.85 (0.46m) 10°  ·  0.70 (0.38m) 15°
0.85 배는 시중 컴팩트 전자레인지(0.44~0.46 m) 크기라 축소해도 부자연스럽지 않다.

사용: python scale_asset.py <원본디렉토리> <출력디렉토리> <축척>
"""
import os
import shutil
import sys
import xml.etree.ElementTree as ET


def scale_xml(root, s):
    """길이 차원을 가진 속성을 전부 s 배 한다."""
    for el in root.iter():
        if el.tag == "mesh":
            cur = el.get("scale")
            base = [float(v) for v in cur.split()] if cur else [1.0, 1.0, 1.0]
            el.set("scale", " ".join(f"{v * s:.6f}" for v in base))
            continue
        for key in ("pos", "size", "fromto"):
            v = el.get(key)
            if v is None:
                continue
            # size 는 원기둥·구 등에서도 전부 길이라 그대로 곱하면 된다
            el.set(key, " ".join(f"{float(x) * s:.6f}" for x in v.split()))
    return root


def main():
    src, dst, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
    os.makedirs(dst, exist_ok=True)
    for sub in os.listdir(src):
        p = os.path.join(src, sub)
        if os.path.isdir(p):
            shutil.copytree(p, os.path.join(dst, sub), dirs_exist_ok=True)
    tree = ET.parse(os.path.join(src, "model.xml"))
    scale_xml(tree.getroot(), s)
    tree.write(os.path.join(dst, "model.xml"), encoding="unicode")
    print(f"{src} -> {dst}  (축척 {s})")


if __name__ == "__main__":
    main()
