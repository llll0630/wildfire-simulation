# ===============================================================
#  屋顶形状（参数化）：人字、四坡、单坡、复折、带女儿墙的平屋顶
#  pipeline（buildings_blender.py，给真实建筑加坡屋顶）和素材生成脚本（asset_builders/build_building_parts.py）共用
#  只用标准库
# ===============================================================
import math


def roof_mesh(kind, L, W, pitch, overhang=0.4, parapet=0.9):
    """屋顶几何（墙顶在 z = 0，轮廓中心在原点，长边沿 x）。pitch = 坡度（升高 / 水平距离，6/12 坡 = 0.5）
    kind：gable 人字、hip 四坡、shed 单坡、gambrel 复折、flat 平屋顶（带女儿墙）。返回 (顶点, 面, 面的材质编号)
    材质编号：0 = 屋面，1 = 山墙 / 女儿墙（墙面材质）"""
    hx, hy = L / 2 + overhang, W / 2 + overhang
    rise = (W / 2) * pitch
    V, F, M = [], [], []
    def quad(a, b, c, d, m=0): o = len(V); V.extend([a, b, c, d]); F.append((o, o + 1, o + 2, o + 3)); M.append(m)
    def tri(a, b, c, m=0): o = len(V); V.extend([a, b, c]); F.append((o, o + 1, o + 2)); M.append(m)
    if kind == 'gable':
        quad((-hx, -hy, 0), (hx, -hy, 0), (hx, 0, rise), (-hx, 0, rise))
        quad((hx, hy, 0), (-hx, hy, 0), (-hx, 0, rise), (hx, 0, rise))
        tri((-L / 2, -W / 2, 0), (-L / 2, 0, rise - overhang * pitch * 0), (-L / 2, W / 2, 0), 1)
        tri((L / 2, W / 2, 0), (L / 2, 0, rise), (L / 2, -W / 2, 0), 1)
    elif kind == 'hip':
        r = max(0.1, hx - hy)                                   # 屋脊长度（长减宽）
        quad((-hx, -hy, 0), (hx, -hy, 0), (r, 0, rise), (-r, 0, rise))
        quad((hx, hy, 0), (-hx, hy, 0), (-r, 0, rise), (r, 0, rise))
        tri((hx, -hy, 0), (hx, hy, 0), (r, 0, rise)); tri((-hx, hy, 0), (-hx, -hy, 0), (-r, 0, rise))
    elif kind == 'shed':
        top = W * pitch * 0.5
        quad((-hx, -hy, 0), (hx, -hy, 0), (hx, hy, top), (-hx, hy, top))
        quad((-L / 2, W / 2, 0), (-L / 2, -W / 2, 0), (-L / 2, -W / 2, 0.01), (-L / 2, W / 2, top), 1)
        quad((L / 2, -W / 2, 0), (L / 2, W / 2, 0), (L / 2, W / 2, top), (L / 2, -W / 2, 0.01), 1)
    elif kind == 'gambrel':                                     # 谷仓：下段陡（约 60°）、上段缓
        k = 0.32; z1 = W * 0.5; z2 = z1 + W * 0.18
        for s in (-1, 1):
            a, b = (-hx, s * hy, 0), (hx, s * hy, 0)
            c, d = (hx, s * W * k, z1), (-hx, s * W * k, z1)
            quad(*((a, b, c, d) if s < 0 else (b, a, d, c)))
            quad(*(((-hx, s * W * k, z1), (hx, s * W * k, z1), (hx, 0, z2), (-hx, 0, z2)) if s < 0 else
                   ((hx, s * W * k, z1), (-hx, s * W * k, z1), (-hx, 0, z2), (hx, 0, z2))))
        for x in (-L / 2, L / 2):
            pts = [(x, -W / 2, 0), (x, -W * k, z1), (x, 0, z2), (x, W * k, z1), (x, W / 2, 0)]
            for i in range(1, 4):
                tri(pts[0], pts[i], pts[i + 1], 1) if x < 0 else tri(pts[0], pts[i + 1], pts[i], 1)
    else:                                                       # flat：屋面 + 一圈女儿墙
        quad((-L / 2, -W / 2, 0.02), (L / 2, -W / 2, 0.02), (L / 2, W / 2, 0.02), (-L / 2, W / 2, 0.02))
        t = 0.25
        for (x0, y0, x1, y1) in ((-L / 2, -W / 2, L / 2, -W / 2), (L / 2, -W / 2, L / 2, W / 2),
                                 (L / 2, W / 2, -L / 2, W / 2), (-L / 2, W / 2, -L / 2, -W / 2)):
            quad((x0, y0, 0), (x1, y1, 0), (x1, y1, parapet), (x0, y0, parapet), 1)
            dx, dy = (y1 - y0), -(x1 - x0); n = math.hypot(dx, dy); dx, dy = dx / n * t, dy / n * t
            quad((x0, y0, parapet), (x1, y1, parapet), (x1 + dx, y1 + dy, parapet), (x0 + dx, y0 + dy, parapet), 1)
            quad((x1 + dx, y1 + dy, 0), (x0 + dx, y0 + dy, 0), (x0 + dx, y0 + dy, parapet), (x1 + dx, y1 + dy, parapet), 1)
    return V, F, M
