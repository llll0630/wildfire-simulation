# ===============================================================
#  人造物和其他小素材（三档）—— Blender 5.x
#  这一步只准备素材；pipeline 怎么用（电线杆按 Overture 电力线摆、栅栏按 barrier 线摆……）以后再定。
#
#    电力：木电线杆（带横担、绝缘子；一款带变压器）、钢架输电塔；电线本身以后按每一档的两端生成，这里只给材质和粗细
#    栅栏（一段，沿线重复）：木板隐私栅栏、白色尖桩栅栏、铁丝网栅栏、牧场横杆栅栏、带刺铁丝栅栏
#    绿篱（一段）、玉米（一株）
#    车：轿车、SUV、皮卡（暂时不生成，见 BUILD_CARS）
#    街道：路灯、停车标志、红绿灯、消防栓、信箱、垃圾桶、长椅、设备箱；桥（一段）
#    公园：秋千、滑梯、野餐桌、篮球架；水边：木码头（一段）；道路：波形护栏（一段）、限速牌、让行牌、路名牌、高速指示牌
#  尺寸按美国常见的真实尺寸（米）。
#  三档：低 = 简单几何 + 顶点颜色；中 = 更多细节 + 贴图（铁丝网、玉米叶等用透明贴图）；高 = 再细一些（倒角、更多分段）
#
#  用法：blender -b --factory-startup --python build_props.py -- [--preview]
#  输出：assets_hq/<档位>/props/props_export.blend（每个素材一个 Collection）
#        assets_hq/props_manifest.json、assets_hq/previews/props_<档位>.png
# ===============================================================
import bpy, bmesh, math, random, os, sys, json
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv

ROOT = os.path.dirname(HERE)
HQ = os.path.join(ROOT, 'assets_hq')
TIERS = ('low', 'medium', 'high')
BUILD_CARS = False          # 车暂时不做（生成函数 car() 留着，以后要用改成 True）
CAR_COLORS = {'white': (0.82, 0.82, 0.80), 'black': (0.03, 0.03, 0.035), 'silver': (0.45, 0.46, 0.47),
              'gray': (0.18, 0.19, 0.2), 'red': (0.45, 0.03, 0.03), 'blue': (0.05, 0.12, 0.35)}


class Mesh:
    """简单的拼装：往一个 bmesh 里加盒子、圆柱，每个零件带颜色和材质编号"""
    def __init__(self, tier):
        self.bm = bmesh.new(); self.tier = tier; self.col = {}; self.uv = self.bm.loops.layers.uv.new('UVMap')

    def _paint(self, verts, c, mat):
        for v in verts: self.col[v] = c
        for f in {f for v in verts for f in v.link_faces}:
            f.material_index = mat

    def box(self, cx, cy, z0, sx, sy, sz, c, mat=0, rot=0.0):
        r = bmesh.ops.create_cube(self.bm, size=1.0)
        m = Matrix.Translation((cx, cy, z0 + sz / 2)) @ Matrix.Rotation(rot, 4, 'Z') @ Matrix.Diagonal((sx, sy, sz, 1))
        bmesh.ops.transform(self.bm, matrix=m, verts=r['verts']); self._paint(r['verts'], c, mat)
        return r['verts']

    def cyl(self, p0, p1, r0, r1, c, mat=0, sides=None):
        sides = sides or {'low': 5, 'medium': 8, 'high': 12}[self.tier]
        p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
        r = bmesh.ops.create_cone(self.bm, cap_ends=True, segments=sides, radius1=r0, radius2=r1, depth=d.length)
        m = Matrix.Translation((p0 + p1) / 2) @ d.to_track_quat('Z', 'Y').to_matrix().to_4x4()
        bmesh.ops.transform(self.bm, matrix=m, verts=r['verts']); self._paint(r['verts'], c, mat)
        return r['verts']

    def card(self, p, up, side, w, h, c, mat=1):
        """一张竖直的贴图卡片（铁丝网、玉米叶），UV 0~1"""
        p, up, side = Vector(p), Vector(up).normalized(), Vector(side).normalized()
        vs = [self.bm.verts.new(p + side * sx * w / 2 + up * sy * h) for sx, sy in ((-1, 0), (1, 0), (1, 1), (-1, 1))]
        f = self.bm.faces.new(vs); f.material_index = mat
        for loop, uv in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))): loop[self.uv].uv = uv
        for v in vs: self.col[v] = c

    def finish(self, name, mats):
        me = bpy.data.meshes.new(name)
        order = list(self.bm.verts)
        cols = np.array([self.col.get(v, (1, 1, 1)) for v in order], np.float32)
        self.bm.to_mesh(me); self.bm.free()
        attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
        attr.data.foreach_set('color', np.concatenate([np.clip(cols, 0, 1), np.ones((len(cols), 1))], 1).ravel())
        me.color_attributes.active_color = attr
        for m in mats: me.materials.append(m)
        return me


# ---------------- 贴图 ----------------
def chainlink_texture(S):
    """铁丝网：斜向菱形网格（透明背景）"""
    img = np.zeros((S, S, 4), np.float32)
    yy, xx = np.mgrid[0:S, 0:S] / S
    n = 8
    d1 = np.abs(((xx + yy) * n) % 1 - 0.5); d2 = np.abs(((xx - yy) * n) % 1 - 0.5)
    m = (d1 > 0.5 - 0.035 * max(1, 256 / S)) | (d2 > 0.5 - 0.035 * max(1, 256 / S))
    img[m] = (0.55, 0.56, 0.55, 1)
    return img


def corn_leaf_texture(S):
    """玉米叶：一片长条叶，中间叶脉浅色，边缘稍黄"""
    img = np.zeros((S, S, 4), np.float32)
    yy, xx = np.mgrid[0:S, 0:S] / S
    t = 1 - yy; w = 0.42 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.7)
    m = np.abs(xx - 0.5) < w * 0.5
    edge = np.abs(xx - 0.5) / np.maximum(w * 0.5, 1e-3)
    col = np.stack([0.20 + 0.08 * edge, 0.34 + 0.05 * edge, 0.10 * np.ones_like(edge)], -1)
    col[np.abs(xx - 0.5) < 0.012] = (0.42, 0.48, 0.25)
    img[m, :3] = col[m]; img[m, 3] = 1
    return img


def wire_texture(S):
    """带刺铁丝：一根细线，每隔一段一个刺结"""
    img = np.zeros((S, S, 4), np.float32)
    img[S // 2 - max(1, S // 128):S // 2 + max(1, S // 128) + 1] = (0.35, 0.35, 0.33, 1)
    for k in range(4):
        x = int((k + 0.5) / 4 * S)
        for dy in range(-S // 24, S // 24):
            img[S // 2 + dy, x + dy // 2 - 1:x + dy // 2 + 2] = (0.35, 0.35, 0.33, 1)
    return img


# ---------------- 素材 ----------------
WOOD = (0.30, 0.22, 0.15); WOOD_GRAY = (0.38, 0.34, 0.30); METAL = (0.42, 0.43, 0.44); DARK = (0.06, 0.06, 0.07)
CERAMIC = (0.75, 0.75, 0.72); WHITE = (0.85, 0.85, 0.82)


def utility_pole(m, transformer=False):
    """木电线杆：地上 11 m，顶上一根 2.4 m 横担、3 个绝缘子；transformer 再挂一个变压器桶"""
    m.cyl((0, 0, -0.5), (0, 0, 11.0), 0.15, 0.11, WOOD_GRAY)
    m.box(0, 0, 10.2, 2.4, 0.1, 0.12, WOOD)
    for x in (-1.05, 0.0, 1.05):
        m.cyl((x, 0, 10.32), (x, 0, 10.55), 0.05, 0.04, CERAMIC, sides=6)
    if m.tier != 'low':
        for x in (-0.9, 0.9):
            m.cyl((x * 0.2, 0, 9.4), (x, 0, 10.2), 0.02, 0.02, METAL, sides=4)         # 斜撑
    if transformer:
        m.cyl((0, 0.35, 8.0), (0, 0.35, 9.0), 0.27, 0.27, METAL)
    return {'height_m': 11.0, 'wire_attach_z': 10.55, 'wire_attach_x': [-1.05, 0.0, 1.05]}


def transmission_tower(m):
    """钢架输电塔（简化）：四条腿往上收，几道横撑，顶上两层横担；高 30 m"""
    H = 30.0
    legs = [(sx, sy) for sx in (-1, 1) for sy in (-1, 1)]
    def w(z): return 3.0 * (1 - z / H) + 0.6
    lv = [0, 6, 12, 18, 24, 30] if m.tier != 'low' else [0, 12, 24, 30]
    for sx, sy in legs:
        for a, b in zip(lv[:-1], lv[1:]):
            m.cyl((sx * w(a), sy * w(a), a), (sx * w(b), sy * w(b), b), 0.08, 0.07, METAL, sides=4)
    for z in lv[1:-1]:
        for (ax, ay), (bx, by) in zip(legs, legs[1:] + legs[:1]):
            m.cyl((ax * w(z), ay * w(z), z), (bx * w(z), by * w(z), z), 0.04, 0.04, METAL, sides=4)
        if m.tier == 'high':
            for (ax, ay), (bx, by) in ((legs[0], legs[3]), (legs[1], legs[2])):
                m.cyl((ax * w(z - 6), ay * w(z - 6), z - 6), (bx * w(z), by * w(z), z), 0.03, 0.03, METAL, sides=4)
    for z, half in ((22, 6.0), (27, 4.5)):
        m.box(0, 0, z, 2 * half, 0.4, 0.4, METAL)
        for x in (-half, half):
            m.cyl((x, 0, z - 1.6), (x, 0, z), 0.08, 0.08, CERAMIC, sides=6)
    return {'height_m': H, 'wire_attach_z': [20.4, 25.4], 'wire_attach_x': [-6.0, 6.0, -4.5, 4.5]}


def fence(m, kind, mats):
    """一段栅栏（沿 x 方向，长 L，原点在一端）；返回长度、高度"""
    if kind == 'wood_privacy':
        L, H = 2.4, 1.8
        m.box(0, 0, -0.3, 0.09, 0.09, H + 0.3, WOOD)
        for z in (0.3, 1.5): m.box(L / 2, 0.06, z, L, 0.04, 0.09, WOOD)
        n = 16 if m.tier != 'low' else 1
        for k in range(n):
            w = L / n
            m.box(w * (k + 0.5), 0.1, 0.02, w * 0.97, 0.02, H - 0.02 + (0.03 if k % 2 else 0), WOOD_GRAY if k % 3 else WOOD)
    elif kind == 'picket':
        L, H = 2.4, 1.0
        m.box(0, 0, -0.3, 0.09, 0.09, H + 0.4, WHITE)
        for z in (0.2, 0.75): m.box(L / 2, 0.05, z, L, 0.03, 0.07, WHITE)
        n = 14 if m.tier != 'low' else 0
        for k in range(n):
            x = L * (k + 0.5) / n
            m.box(x, 0.08, 0.05, 0.08, 0.02, H - 0.1, WHITE)
            if m.tier == 'high':
                m.cyl((x, 0.08, H - 0.05), (x, 0.08, H + 0.05), 0.04, 0.005, WHITE, sides=4)
        if n == 0: m.box(L / 2, 0.08, 0.05, L, 0.02, H - 0.1, WHITE)
    elif kind == 'chain_link':
        L, H = 3.0, 1.8
        m.cyl((0, 0, -0.3), (0, 0, H + 0.05), 0.03, 0.03, METAL)
        m.cyl((0, 0, H), (L, 0, H), 0.02, 0.02, METAL, sides=6)
        if m.tier == 'low':
            m.box(L / 2, 0, 0.05, L, 0.01, H - 0.05, (0.45, 0.46, 0.45))
        else:
            m.card((L / 2, 0, 0.05), (0, 0, 1), (1, 0, 0), L, H - 0.05, (1, 1, 1), mat=1)
    elif kind == 'split_rail':
        L, H = 3.0, 1.2
        m.box(0, 0, -0.3, 0.14, 0.14, H + 0.3, WOOD_GRAY)
        for z in (0.45, 0.95):
            m.box(L / 2, 0, z, L, 0.11, 0.11, WOOD_GRAY, rot=0.0)
    else:   # barbed_wire
        L, H = 3.5, 1.2
        m.box(0, 0, -0.4, 0.09, 0.09, H + 0.4, WOOD_GRAY)
        for z in (0.35, 0.7, 1.05):
            if m.tier == 'low':
                m.box(L / 2, 0, z, L, 0.01, 0.01, METAL)
            else:
                m.card((L / 2, 0, z - 0.05), (0, 0, 1), (1, 0, 0), L, 0.1, (1, 1, 1), mat=2)
    return {'length_m': L, 'height_m': H}


def hedge(m, mats):
    """一段绿篱：长 2 m、高 1.4 m、厚 0.9 m 的修剪过的绿篱；中高档外面贴叶片卡片"""
    L, H, D = 2.0, 1.4, 0.9
    m.box(L / 2, 0, 0, L, D * (0.9 if m.tier != 'low' else 1.0), H * (0.95 if m.tier != 'low' else 1.0), (0.10, 0.20, 0.07))
    if m.tier != 'low':
        rng = random.Random(3); n = 80 if m.tier == 'medium' else 200
        for _ in range(n):
            face = rng.choice(('front', 'back', 'top'))
            x = rng.uniform(0.05, L - 0.05)
            if face == 'top':
                p = Vector((x, rng.uniform(-D / 2, D / 2), H * 0.9)); up = Vector((rng.uniform(-.3, .3), rng.uniform(-.3, .3), 1))
            else:
                sgn = 1 if face == 'front' else -1
                p = Vector((x, sgn * D * 0.42, rng.uniform(0.05, H * 0.85))); up = Vector((rng.uniform(-.3, .3), sgn * 0.5, 1))
            s = rng.uniform(0.3, 0.45)
            m.card(p, up, Vector((1, rng.uniform(-.4, .4), 0)), s, s, (0.75, 0.9, 0.7), mat=1)
    return {'length_m': L, 'height_m': H}


def corn(m, mats):
    """一株玉米：2.2 m 高的茎，交错长 10 ~ 12 片下垂的长叶，顶上雄穗，中间一个玉米棒"""
    rng = random.Random(5); H = 2.2
    m.cyl((0, 0, 0), (0, 0, H), 0.018, 0.01, (0.35, 0.42, 0.2), sides={'low': 3, 'medium': 5, 'high': 6}[m.tier])
    n = {'low': 4, 'medium': 10, 'high': 12}[m.tier]
    for k in range(n):
        z = 0.3 + (H - 0.5) * k / n; az = k * math.pi + rng.uniform(-0.3, 0.3)
        d = Vector((math.cos(az), math.sin(az), 0))
        if m.tier == 'low':
            m.box(d.x * 0.35, d.y * 0.35, z, 0.7, 0.06, 0.01, (0.25, 0.4, 0.12), rot=az)
        else:
            m.card(Vector((0, 0, z)), d + Vector((0, 0, 0.35)), d.cross(Vector((0, 0, 1))), 0.12, 0.85, (1, 1, 1), mat=1)
    m.cyl((0, 0, H), (0, 0, H + 0.3), 0.01, 0.004, (0.55, 0.48, 0.25), sides=4)
    m.cyl((0.04, 0, 1.0), (0.08, 0, 1.25), 0.03, 0.02, (0.45, 0.5, 0.25), sides=5)
    return {'height_m': H + 0.3}


def car(m, kind, color):
    """车：车身 + 车窗（深色）+ 四个轮子。尺寸：轿车 4.8 × 1.85 × 1.45 m、SUV 4.9 × 1.95 × 1.75 m、皮卡 5.8 × 2.0 × 1.9 m"""
    L, W, Hh = {'sedan': (4.8, 1.85, 1.45), 'suv': (4.9, 1.95, 1.75), 'pickup': (5.8, 2.0, 1.9)}[kind]
    glass = (0.05, 0.07, 0.09); tire = DARK
    body_h = Hh * 0.5; base = 0.3
    m.box(0, 0, base, L, W, body_h, color)
    if kind == 'pickup':
        m.box(-L * 0.1 + L * 0.18, 0, base + body_h, L * 0.38, W * 0.92, Hh - body_h - base, color)
        m.box(-L * 0.1 + L * 0.18, 0, base + body_h + 0.05, L * 0.34, W * 0.94, (Hh - body_h - base) * 0.75, glass)
        m.box(-L * 0.3, 0, base + body_h, L * 0.38, W, 0.06, color)                     # 车斗边沿
    else:
        cab_l = L * (0.55 if kind == 'sedan' else 0.7)
        cx = -L * 0.03 if kind == 'sedan' else -L * 0.08
        top_k = 0.62 if kind == 'sedan' else 0.85             # 车顶比车窗底部短多少：挡风玻璃、后窗是斜的
        for vs, k in ((m.box(cx, 0, base + body_h, cab_l, W * 0.88, Hh - body_h - base, color), top_k),
                      (m.box(cx, 0, base + body_h + 0.04, cab_l * 1.02, W * 0.9, (Hh - body_h - base) * 0.78, glass), top_k)):
            ztop = max(v.co.z for v in vs)
            for v in vs:
                if v.co.z > ztop - 1e-4:
                    v.co.x = cx + (v.co.x - cx) * k
    for sx in (-1, 1):
        for sy in (-1, 1):
            x = sx * L * 0.33; y = sy * (W / 2 - 0.12)
            m.cyl((x, y - 0.12 * sy, 0.36), (x, y + 0.12 * sy, 0.36), 0.36, 0.36, tire,
                  sides={'low': 6, 'medium': 10, 'high': 16}[m.tier])
    if m.tier == 'high':                                                                 # 车灯
        for sy in (-1, 1):
            m.box(L / 2, sy * W * 0.35, base + body_h * 0.55, 0.04, 0.3, 0.12, (0.9, 0.9, 0.85))
            m.box(-L / 2, sy * W * 0.38, base + body_h * 0.6, 0.04, 0.25, 0.1, (0.6, 0.05, 0.03))
    return {'length_m': L, 'width_m': W, 'height_m': Hh}


def street_light(m):
    """路灯：8 m 灯杆 + 1.8 m 悬臂 + 灯头"""
    m.cyl((0, 0, 0), (0, 0, 8.0), 0.1, 0.07, (0.3, 0.31, 0.32))
    m.cyl((0, 0, 7.8), (1.8, 0, 8.0), 0.04, 0.035, (0.3, 0.31, 0.32))
    m.box(1.9, 0, 7.85, 0.6, 0.28, 0.15, (0.25, 0.25, 0.26))
    return {'height_m': 8.15}


def stop_sign(m):
    """停车标志：2.1 m 高的杆 + 0.75 m 红色八角牌"""
    m.cyl((0, 0, 0), (0, 0, 2.4), 0.03, 0.03, METAL)
    r = bmesh.ops.create_circle(m.bm, cap_ends=True, segments=8, radius=0.39)
    mm = Matrix.Translation((0, -0.04, 2.45)) @ Matrix.Rotation(math.pi / 2, 4, 'X') @ Matrix.Rotation(math.pi / 8, 4, 'Z')
    bmesh.ops.transform(m.bm, matrix=mm, verts=r['verts']); m._paint(r['verts'], (0.55, 0.03, 0.03), 0)
    return {'height_m': 2.85}


def hydrant(m):
    """消防栓：0.8 m 高，红色（美国常见），两侧出水口"""
    red = (0.55, 0.05, 0.03)
    m.cyl((0, 0, 0), (0, 0, 0.1), 0.17, 0.17, red); m.cyl((0, 0, 0.1), (0, 0, 0.65), 0.12, 0.12, red)
    m.cyl((0, 0, 0.65), (0, 0, 0.8), 0.13, 0.05, red)
    for s in (-1, 1):
        m.cyl((0, 0, 0.45), (s * 0.2, 0, 0.45), 0.05, 0.05, red, sides=6)
    return {'height_m': 0.8}


def mailbox(m):
    """路边信箱：木桩 + 拱顶的金属信箱"""
    m.box(0, 0, 0, 0.1, 0.1, 1.05, WOOD)
    m.box(0, 0.05, 1.05, 0.2, 0.5, 0.18, (0.12, 0.12, 0.13))
    m.cyl((0, -0.2, 1.23), (0, 0.3, 1.23), 0.1, 0.1, (0.12, 0.12, 0.13))
    return {'height_m': 1.33}


def bins(m):
    """两个带轮的垃圾桶（灰色垃圾、蓝色回收）"""
    for x, c in ((-0.4, (0.15, 0.16, 0.16)), (0.4, (0.05, 0.15, 0.4))):
        m.box(x, 0, 0.05, 0.6, 0.72, 0.95, c); m.box(x, 0.02, 1.0, 0.64, 0.78, 0.06, c)
        for sx in (-0.22, 0.22):
            m.cyl((x + sx, 0.36, 0.1), (x + sx, 0.42, 0.1), 0.1, 0.1, DARK, sides=8)
    return {'height_m': 1.06}


def bench(m):
    """公园长椅：1.8 m，木条 + 铸铁腿"""
    for x in (-0.75, 0.75):
        m.box(x, 0, 0, 0.06, 0.5, 0.45, (0.12, 0.12, 0.12)); m.box(x, 0.22, 0.45, 0.06, 0.06, 0.45, (0.12, 0.12, 0.12))
    for k in range(4 if m.tier != 'low' else 1):
        m.box(0, -0.18 + k * 0.12, 0.43, 1.8, 0.09 if m.tier != 'low' else 0.45, 0.04, WOOD)
    for k in range(2 if m.tier != 'low' else 1):
        m.box(0, 0.24, 0.6 + k * 0.15, 1.8, 0.04, 0.09, WOOD)
    return {'length_m': 1.8, 'height_m': 0.9}


def traffic_light(m):
    """红绿灯：路边立杆 + 横臂，臂上挂两个三色灯（黄色灯箱）"""
    pole = (0.3, 0.31, 0.32)
    m.cyl((0, 0, 0), (0, 0, 6.0), 0.14, 0.11, pole)
    m.cyl((0, 0, 5.8), (6.0, 0, 5.8), 0.07, 0.05, pole)
    for x in (3.0, 5.6):
        m.box(x, 0, 4.6, 0.35, 0.3, 1.1, (0.75, 0.6, 0.1))
        for k, c in enumerate(((0.6, 0.05, 0.03), (0.7, 0.5, 0.05), (0.05, 0.45, 0.15))):
            m.box(x, -0.16, 5.4 - k * 0.33, 0.22, 0.03, 0.22, c)
    return {'height_m': 6.0, 'arm_m': 6.0}


def utility_box(m):
    """路边的电力 / 通信设备箱（浅绿灰色）"""
    m.box(0, 0, 0, 1.1, 0.6, 1.3, (0.16, 0.2, 0.15)); m.box(0, 0, 1.3, 1.15, 0.65, 0.05, (0.14, 0.18, 0.13))
    return {'height_m': 1.35}


def bridge_segment(m):
    """桥的一段（沿 x 长 12 m，宽 10 m）：桥面 + 两侧护栏 + 一对桥墩；桥面顶在 z = 0"""
    L, W = 12.0, 10.0
    conc = (0.55, 0.54, 0.52)
    m.box(L / 2, 0, -0.9, L, W, 0.9, conc)
    for s in (-1, 1):
        m.box(L / 2, s * (W / 2 - 0.2), 0, L, 0.4, 0.85, conc)
        if m.tier == 'high':
            m.box(L / 2, s * (W / 2 - 0.2), 0.85, L, 0.12, 0.08, (0.5, 0.5, 0.52))
    for s in (-1, 1):
        m.box(L / 2, s * 2.5, -9.0, 1.2, 1.2, 8.1, conc)
    m.box(L / 2, 0, -1.6, 1.6, W - 1.0, 0.7, conc)
    return {'length_m': L, 'width_m': W, 'pier_height_m': 8.1, 'note': '桥墩高度以后按桥下地形截短'}


def swing_set(m):
    """秋千架：A 字形钢架 + 横梁，两个秋千"""
    steel = (0.15, 0.32, 0.18)
    for x in (-1.8, 1.8):
        for sy in (-1, 1):
            m.cyl((x, sy * 1.0, 0), (x, 0, 2.4), 0.04, 0.04, steel, sides=6)
    m.cyl((-1.9, 0, 2.4), (1.9, 0, 2.4), 0.05, 0.05, steel, sides=6)
    for x in (-0.8, 0.8):
        for dx in (-0.22, 0.22):
            m.cyl((x + dx, 0, 2.4), (x + dx, 0, 0.5), 0.008, 0.008, (0.3, 0.3, 0.3), sides=3)
        m.box(x, 0, 0.45, 0.5, 0.18, 0.04, (0.1, 0.1, 0.1))
    return {'length_m': 3.8, 'height_m': 2.45}


def slide(m):
    """滑梯：平台 + 梯子 + 斜滑板（黄色）"""
    m.box(0, 0, 1.4, 1.0, 1.0, 0.08, (0.6, 0.35, 0.05))
    for sx in (-0.45, 0.45):
        for sy in (-0.45, 0.45):
            m.cyl((sx, sy, 0), (sx, sy, 2.3), 0.05, 0.05, (0.1, 0.25, 0.45), sides=6)
    vs = m.box(0, 1.8, 0, 0.6, 2.8, 0.06, (0.75, 0.6, 0.05))
    for v in vs:
        v.co.z += 1.4 * (1 - (v.co.y - 0.4) / 2.8)
    for k in range(5):
        m.box(0, -0.75, 0.28 * k + 0.2, 0.5, 0.05, 0.04, (0.3, 0.3, 0.3))
    m.box(0, -0.6, 0, 0.04, 0.04, 1.5, (0.3, 0.3, 0.3))
    return {'length_m': 4.0, 'height_m': 2.3}


def picnic_table(m):
    wood = (0.32, 0.22, 0.14)
    m.box(0, 0, 0.72, 1.8, 0.75, 0.05, wood)
    for sy in (-0.65, 0.65):
        m.box(0, sy, 0.43, 1.8, 0.28, 0.04, wood)
    for x in (-0.7, 0.7):
        for sy in (-1, 1):
            vs = m.box(x, sy * 0.35, 0, 0.08, 0.08, 0.75, wood)
            for v in vs:
                if v.co.z < 0.01: v.co.y += sy * 0.35
    return {'length_m': 1.8, 'height_m': 0.77}


def basketball_hoop(m):
    m.cyl((0, 0, 0), (0, 0, 3.4), 0.07, 0.07, (0.25, 0.25, 0.26), sides=8)
    m.cyl((0, 0, 3.2), (0, -0.9, 3.2), 0.05, 0.05, (0.25, 0.25, 0.26), sides=6)
    m.box(0, -0.95, 2.9, 1.8, 0.04, 1.05, (0.9, 0.9, 0.88))
    seg = {'low': 6, 'medium': 12, 'high': 16}[m.tier]
    for k in range(seg):
        a, b = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
        m.cyl((0.23 * math.cos(a), -1.2 + 0.23 * math.sin(a), 3.05), (0.23 * math.cos(b), -1.2 + 0.23 * math.sin(b), 3.05),
              0.01, 0.01, (0.75, 0.25, 0.05), sides=3)
    return {'height_m': 3.43}


def dock_segment(m):
    """木码头一段（沿 x 长 4 m、宽 2 m）：桩 + 木板，板面高出水面 0.8 m"""
    wood = (0.32, 0.24, 0.16)
    for x in (0.1, 3.9):
        for y in (-0.9, 0.9):
            m.cyl((x, y, -1.5), (x, y, 0.9), 0.12, 0.12, (0.25, 0.2, 0.14))
    n = {'low': 1, 'medium': 14, 'high': 26}[m.tier]
    for k in range(n):
        m.box(4.0 * (k + 0.5) / n, 0, 0.75, 4.0 / n * 0.92, 2.0, 0.05, tuple(c * (0.9 + 0.1 * (k % 3)) for c in wood))
    for y in (-0.9, 0.9):
        m.box(2.0, y, 0.6, 4.0, 0.1, 0.15, wood)
    return {'length_m': 4.0, 'width_m': 2.0, 'deck_z_m': 0.8}


def guardrail(m):
    """高速公路波形护栏一段（4 m）：钢板 + 两根立柱"""
    steel = (0.55, 0.56, 0.57)
    for x in (0.0, 2.0):
        m.box(x, 0.1, -0.5, 0.15, 0.1, 1.2, steel)
    if m.tier == 'low':
        m.box(2.0, 0, 0.5, 4.0, 0.05, 0.31, steel)
    else:
        for k, z in enumerate((0.55, 0.65, 0.75)):                 # W 形：三道起伏
            m.box(2.0, -0.04 if k == 1 else 0.0, z - 0.05, 4.0, 0.05, 0.11, steel)
    return {'length_m': 4.0, 'height_m': 0.8}


def sign(m, kind):
    """路牌：限速牌（白底）、让行牌（倒三角）、路名牌（绿色）、高速指示牌（大块绿色、两根柱）"""
    pole = (0.45, 0.46, 0.47)
    if kind == 'highway':
        for x in (-2.0, 2.0):
            m.cyl((x, 0, 0), (x, 0, 6.0), 0.12, 0.12, pole)
        m.box(0, -0.15, 4.2, 5.0, 0.06, 2.2, (0.0, 0.25, 0.12))
        m.box(0, -0.19, 4.4, 4.6, 0.01, 0.12, (0.85, 0.85, 0.82)); m.box(0, -0.19, 5.0, 3.8, 0.01, 0.18, (0.85, 0.85, 0.82))
        return {'height_m': 6.4}
    m.cyl((0, 0, 0), (0, 0, 2.6), 0.03, 0.03, pole, sides=6)
    if kind == 'speed':
        m.box(0, -0.05, 1.95, 0.6, 0.02, 0.75, (0.88, 0.88, 0.85)); m.box(0, -0.065, 2.05, 0.4, 0.01, 0.25, (0.05, 0.05, 0.05))
        return {'height_m': 2.7, 'note': '限速牌，数字以后用贴图换'}
    if kind == 'yield':
        bm = m.bm
        vs = [bm.verts.new(p) for p in ((-0.45, -0.05, 2.6), (0.45, -0.05, 2.6), (0, -0.05, 1.82))]
        bm.faces.new(vs); m._paint(vs, (0.6, 0.04, 0.03), 0)
        return {'height_m': 2.6}
    m.box(0, -0.03, 2.55, 0.9, 0.02, 0.18, (0.0, 0.32, 0.15)); m.box(0, 0, 2.75, 0.18, 0.9, 0.18, (0.0, 0.32, 0.15))
    return {'height_m': 2.85, 'note': '路名牌（十字两块），路名以后用贴图换'}


def build(tier):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(HQ, tier, 'props'); tex = os.path.join(base, 'textures')
    vc = bpy.data.materials.new('props_vc'); bs = vc.node_tree.nodes['Principled BSDF']
    n = vc.node_tree.nodes.new('ShaderNodeVertexColor'); n.layer_name = 'Col'
    vc.node_tree.links.new(n.outputs['Color'], bs.inputs['Base Color']); bs.inputs['Roughness'].default_value = 0.7
    px = 256 if tier == 'medium' else 512
    alpha = {}
    if tier != 'low':
        for k, fn in (('chainlink', chainlink_texture), ('corn_leaf', corn_leaf_texture), ('barbed', wire_texture)):
            alpha[k] = hv.export_material(f'props_{k}', hv.make_image(k, fn(px), tex), True)
        leaves = hv.leaf_texture(px, 31); leaves[..., :3] = np.clip(leaves[..., :3] * 1.6 ** (1 / 2.2), 0, 1)
        alpha['hedge'] = hv.export_material('props_hedge', hv.make_image('hedge_leaves', leaves, tex), True)
    mat_for = lambda second=None, third=None: [vc] + ([alpha[second]] if second else []) + ([alpha[third]] if third else [])
    items = []
    def add(name, kind, fn, *args, mats=None):
        m = Mesh(tier); info = fn(m, *args)
        me = m.finish(name, mats or [vc])
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        c.objects.link(bpy.data.objects.new(name, me))
        items.append({'name': name, 'kind': kind, **info, 'triangles': hv.tris(me), 'collection': c})
    add('pole_utility', 'power', utility_pole)
    add('pole_utility_transformer', 'power', utility_pole, True)
    add('tower_transmission', 'power', transmission_tower)
    for k in ('wood_privacy', 'picket', 'chain_link', 'split_rail', 'barbed_wire'):
        mats = None
        if tier != 'low' and k == 'chain_link': mats = mat_for('chainlink')
        if tier != 'low' and k == 'barbed_wire': mats = [vc, vc, alpha['barbed']]
        add(f'fence_{k}', 'fence', fence, k, None, mats=mats)
    add('hedge', 'hedge', hedge, None, mats=mat_for('hedge') if tier != 'low' else None)
    add('crop_corn', 'crop', corn, None, mats=mat_for('corn_leaf') if tier != 'low' else None)
    if BUILD_CARS:
        for kind in ('sedan', 'suv', 'pickup'):
            for cname, rgb in CAR_COLORS.items():
                add(f'car_{kind}_{cname}', 'vehicle', car, kind, rgb)
    add('street_light', 'street', street_light)
    add('stop_sign', 'street', stop_sign)
    add('traffic_light', 'street', traffic_light)
    add('fire_hydrant', 'street', hydrant)
    add('mailbox', 'street', mailbox)
    add('trash_bins', 'street', bins)
    add('bench', 'street', bench)
    add('utility_box', 'street', utility_box)
    add('bridge_segment', 'bridge', bridge_segment)
    add('playground_swing', 'park', swing_set)
    add('playground_slide', 'park', slide)
    add('picnic_table', 'park', picnic_table)
    add('basketball_hoop', 'park', basketball_hoop)
    add('dock_segment', 'waterfront', dock_segment)
    add('guardrail', 'road', guardrail)
    for k in ('speed', 'yield', 'street_name', 'highway'):
        add(f'sign_{k}', 'road', sign, k)
    os.makedirs(base, exist_ok=True)
    bpy.data.libraries.write(os.path.join(base, 'props_export.blend'), {i['collection'] for i in items}, fake_user=True)
    for i in items: i.pop('collection')
    return items


def preview(tier):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    with bpy.data.libraries.load(os.path.join(HQ, tier, 'props', 'props_export.blend'), link=False) as (src, dst):
        dst.collections = list(src.collections)
    order = ['tower_transmission', 'pole_utility', 'pole_utility_transformer', 'street_light', 'stop_sign',
             'fence_wood_privacy', 'fence_picket', 'fence_chain_link', 'fence_split_rail', 'fence_barbed_wire', 'hedge',
             'crop_corn', 'traffic_light', 'fire_hydrant', 'mailbox', 'trash_bins', 'bench', 'utility_box']
    if BUILD_CARS:
        order += [f'car_{k}_{c}' for k in ('sedan', 'suv', 'pickup') for c in ('white', 'red', 'blue')]
    cs = {c.name: c for c in dst.collections}
    x = -26.0
    for name in order:
        e = bpy.data.objects.new(name, None); e.instance_type = 'COLLECTION'; e.instance_collection = cs[name]
        e.location = (x, 0, 0); sc.collection.objects.link(e)
        x += {'tower_transmission': 9, 'fence_wood_privacy': 3.2, 'fence_picket': 3.2, 'fence_chain_link': 3.8,
              'fence_split_rail': 3.8, 'fence_barbed_wire': 4.3, 'hedge': 2.8, 'crop_corn': 1.5}.get(name, 3.0 if 'car' not in name else 2.6)
        if name.startswith('car'):
            e.rotation_euler = (0, 0, math.pi / 2)
    bpy.ops.mesh.primitive_plane_add(size=300)
    g = bpy.context.active_object; gm = bpy.data.materials.new('g'); g.data.materials.append(gm)
    gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.15, 0.17, 0.08, 1)
    w = bpy.data.worlds.new('w'); sc.world = w
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.45, 0.6, 0.85, 1)
    sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 4; so = bpy.data.objects.new('s', sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(-40)); sc.collection.objects.link(so)
    cd = bpy.data.cameras.new('c'); cd.lens = 32; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam)
    cam.location = (10, -62, 14); cam.rotation_euler = (Vector((10, 0, 7)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cam; sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 32
    sc.render.resolution_x, sc.render.resolution_y = 2400, 1000
    sc.view_settings.view_transform = 'AgX'
    sc.render.filepath = os.path.join(HQ, 'previews', f'props_{tier}.png')
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    man = {}
    for tier in TIERS:
        for it in build(tier):
            e = man.setdefault(it['name'], {k: v for k, v in it.items() if k != 'triangles'})
            e.setdefault('triangles', {})[tier] = it['triangles']
    with open(os.path.join(HQ, 'props_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'wire': {'diameter_m': 0.02, 'color': [0.08, 0.08, 0.08], 'sag_ratio': 0.03},
                   'assets': list(man.values())}, f, ensure_ascii=False, indent=1)
    print('props', len(man))
    if '--preview' in argv:
        import build_nature as bn
        for t in TIERS:
            preview(t)
            bn.grid_preview(os.path.join(HQ, t, 'props', 'props_export.blend'), os.path.join(HQ, 'previews', f'props_grid_{t}.png'), cols=8)


if __name__ == '__main__':
    main()
