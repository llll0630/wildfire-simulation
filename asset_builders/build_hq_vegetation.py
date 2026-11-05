# ===============================================================
#  中档、高档植被素材（写实树、灌木、草）—— Blender 5.x
#  做法来自 hero/hero_trees.py：递归生长的枝干 + 带透明贴图的叶片簇 / 针叶簇卡片。
#  hero 原版太重（松树每棵 5 ~ 9 万个三角形），这里按档位减少叶片张数、枝干细分；
#  颜色全部做成"贴图 × 顶点颜色"，GLB 能原样带走（hero 原版用的是 Blender 节点材质，导不出去）。
#
#  低档 = assets/ 里现在的素材，这个脚本不碰它。
#  素材名字和低档一样（tree_oak_01 …），高度也缩放到和低档一样，pipeline 换档时只换文件夹。
#
#  用法：blender -b --factory-startup --python build_hq_vegetation.py -- [high] [medium] [--preview]
#        不写档位就两档都生成；--preview 另外渲染一张 低 / 中 / 高 对比图
#  草：每丛几张交叉的竖直卡片 + 画满草叶的透明贴图（低档是 120 ~ 150 个三角形的实心模型，这里更轻也更像）
#
#  输出：assets_hq/<档位>/trees/trees_export.blend、shrubs/、grass/（贴图打包在里面）
#        assets_hq/<档位>/textures/*.png（同样的贴图，方便查看）
#        assets_hq/<档位>/stats.json（每个素材的三角形数）
#        assets_hq/previews/tiers_compare.png（--preview）
# ===============================================================
import bpy, bmesh, math, random, os, sys, json
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOW = os.path.join(ROOT, 'assets')                 # 低档素材（只读）
OUT = os.path.join(ROOT, 'assets_hq')

# 每档的参数。数字越大越细：
#   tex / bark_tex     叶片、树皮贴图边长（像素）
#   card_rows          每张叶片卡片几段（2 = 中间弯一下，4 个三角形；1 = 平的，2 个三角形）
#   oak_sides          橡树枝干每一级的边数（主干、大枝、中枝、细枝；< 3 就不建这一级的管子，只挂叶子）
#   oak_stride         每一级枝干隔几个点取一段（1 = 每 0.3 m 一段）
#   oak_cards          每个挂叶点平均几张卡片；oak_scale 卡片放大倍数（张数少了就放大一些，树冠不变稀）
#   pine_step          松针卡片沿枝的间距（米）；pine_scale 卡片放大倍数
#   pine_sides / pine_seg   轮生枝的边数、每段长度（米）；pine_twigs 是否给侧枝建管子
#   shrub_cards / shrub_scale   灌木每个挂叶点几张卡片、放大倍数
#   grass_tex / grass_cards / grass_rows   草的贴图边长、每丛几张卡片、每张几段（2 = 中间往外弯一下）
TIERS = {
    'high':   dict(tex=1024, bark_tex=512, card_rows=2,
                   oak_sides=(8, 5, 4, 3), oak_stride=(1, 2, 2, 3), oak_cards=1.4, oak_scale=1.35,
                   pine_step=0.24, pine_scale=1.45, pine_sides=3, pine_seg=0.6, pine_twigs=False,
                   shrub_cards=3.0, shrub_scale=1.15, grass_tex=512, grass_cards=4, grass_rows=2),
    'medium': dict(tex=512, bark_tex=256, card_rows=1,
                   oak_sides=(6, 4, 3, 0), oak_stride=(2, 3, 3, 3), oak_cards=0.8, oak_scale=1.7,
                   pine_step=0.36, pine_scale=1.75, pine_sides=3, pine_seg=0.9, pine_twigs=False,
                   shrub_cards=1.6, shrub_scale=1.45, grass_tex=256, grass_cards=3, grass_rows=1),
}
# 草的颜色（sRGB）：根部 → 叶尖
GRASS_COLORS = {'green': ((0.07, 0.14, 0.04), (0.33, 0.48, 0.15)),
                'dry': ((0.24, 0.19, 0.1), (0.78, 0.68, 0.44))}

# 叶色：(色 A, 色 B) 按每张卡片的随机数在两色之间取，乘在贴图上（和 hero 一样的值）
TINTS = {'OakLeaves': ((1.0, 1.2, 0.7), (1.5, 1.35, 0.85)),
         'PineNeedles': ((1.6, 1.8, 1.4), (2.1, 1.9, 1.3)),
         'OliveLeaves': ((1.05, 1.1, 0.8), (1.45, 1.35, 1.0)),
         'HillBush': ((0.85, 1.1, 0.6), (1.2, 1.25, 0.7))}
BARKS = {'oak': ((0.035, 0.03, 0.025), (0.17, 0.15, 0.12)),          # 线性颜色，和 hero 的树皮材质一样
         'pine': ((0.04, 0.028, 0.02), (0.16, 0.1, 0.07))}


def clampf(v, a=0.0, b=1.0): return a if v < a else b if v > b else v


# ---------------- 贴图（numpy 画，和 hero 一样） ----------------
def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
    return np.hypot(px - ax - t * vx, py - ay - t * vy), t


def leaf_texture(S, seed=1):
    """一小枝橡树叶：中间一根细枝，两边交错长 16 片带浅裂的叶子（sRGB 值，带透明度）"""
    rng = np.random.default_rng(seed)
    img = np.zeros((S, S, 4), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    py, px = 1 - yy, xx
    k = S / 1024                                          # 线宽按 1024 设计，小贴图等比例放粗一点
    twig = lambda v: 0.5 + 0.04 * math.sin(v * 3.0)
    d, _ = seg_dist(px, py, 0.5, 0.0, twig(0.92), 0.92)
    img[d < 0.006 / max(k, 0.5) ** 0.3] = (0.20, 0.14, 0.08, 1)
    n = 16
    for i in range(n):
        v = 0.12 + 0.8 * i / n + rng.uniform(-0.02, 0.02)
        side = 1 if i % 2 else -1
        ang = math.radians(rng.uniform(35, 70)) * side
        L = rng.uniform(0.2, 0.3) * (1.0 - 0.35 * v); W = L * rng.uniform(0.34, 0.42)
        bx, by = twig(v), v
        ux, uy = math.sin(ang), math.cos(ang)
        pet = 0.03
        d, _ = seg_dist(px, py, bx, by, bx + ux * pet, by + uy * pet)
        img[d < 0.004] = (0.2, 0.2, 0.08, 1)
        ox, oy = bx + ux * pet, by + uy * pet
        lx = (px - ox) * ux + (py - oy) * uy
        ly = -(px - ox) * uy + (py - oy) * ux
        t = lx / L
        inside = (t > 0) & (t < 1)
        prof = np.sin(np.pi * np.clip(t, 0, 1) ** 0.85) ** 0.9 * (1 + 0.22 * np.sin(t * np.pi * 9) * (t > 0.15))
        leaf = inside & (np.abs(ly) < W * 0.5 * prof)
        g = rng.uniform(-1, 1)
        base = np.array((0.16 + 0.05 * g, 0.30 + 0.06 * g, 0.07 + 0.02 * g))
        shade = 0.75 + 0.35 * t + 0.25 * (ly / W) * side
        col = base[None, None, :] * shade[..., None]
        rib = np.abs(ly) < 0.0035
        col = np.where(rib[..., None], col * 1.35 + 0.03, col)
        img[leaf, :3] = col[leaf]; img[leaf, 3] = 1
    return img


def needle_texture(S, seed=2):
    """一簇松枝：hero 原版一张卡片只有一根细枝；这里一张画 4 根（中间一根 + 两侧斜出三根），
    卡片张数少很多也能保持树冠密实。针的粗细按贴图大小放宽，小贴图不至于糊没"""
    rng = np.random.default_rng(seed)
    img = np.zeros((S, S, 4), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    py, px = 1 - yy, xx
    wid = 0.0026 * max(1.0, 1024 / S) ** 0.6

    def draw(x0_, y0_, x1_, y1_, w):
        """在 (x0_, y0_) → (x1_, y1_) 的线段附近画一笔（只算包围盒里的像素）"""
        pad = w * 2 + 2 / S
        c0, c1 = int(max(0, min(x0_, x1_) - pad) * S), int(min(1, max(x0_, x1_) + pad) * S) + 1
        r0, r1 = int(max(0, min(y0_, y1_) - pad) * S), int(min(1, max(y0_, y1_) + pad) * S) + 1
        if c1 <= c0 or r1 <= r0: return None, None, None
        sl = (slice(S - r1, S - r0), slice(c0, c1))
        d, t = seg_dist(px[sl], py[sl], x0_, y0_, x1_, y1_)
        return sl, d, t

    def twig(bx, by, ang, length, scale):
        dx, dy = math.sin(ang), math.cos(ang)
        sl, d, _ = draw(bx, by, bx + dx * length, by + dy * length, 0.006 * scale)
        if sl is not None: img[sl][d < 0.006 * scale] = (0.22, 0.15, 0.09, 1)
        v = 0.04
        while v < 0.95:
            for side in (-1, 1):
                a = ang + math.radians(rng.uniform(35, 60) + 15 * v) * side
                L = rng.uniform(0.2, 0.3) * (1 - 0.4 * v) * length * scale
                sx, sy = bx + dx * length * v, by + dy * length * v
                ex, ey = sx + math.sin(a) * L, sy + math.cos(a) * L
                sl, d, t = draw(sx, sy, ex, ey, wid)
                if sl is None: continue
                m = d < wid * (1.2 - 0.6 * t)
                g = rng.uniform(-1, 1)
                col = np.array((0.07 + 0.02 * g, 0.17 + 0.04 * g, 0.08 + 0.02 * g))
                blk = img[sl]
                c = col[None, None, :] * (0.7 + 0.6 * t)[..., None]
                blk[m, :3] = c[m]; blk[m, 3] = 1
            v += rng.uniform(0.008, 0.014) * max(1.0, 1024 / S) ** 0.5 / length ** 0.5

    # 两侧先画（被中间那根压在下面）
    twig(0.5, 0.18, math.radians(-38), 0.5, 0.75)
    twig(0.5, 0.36, math.radians(34), 0.46, 0.72)
    twig(0.5, 0.05, math.radians(48), 0.36, 0.7)
    twig(0.5, 0.0, 0.0, 0.95, 0.85)
    return img


def bark_texture(S, c1, c2, seed):
    """树皮：竖向的长条纹理（随机噪声在频域里做成"竖长横短"），天然可以上下左右无缝重复"""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(S)[:, None]; fx = np.fft.fftfreq(S)[None, :]
    def band(ax, ay):
        f = np.fft.fft2(rng.standard_normal((S, S))) * np.exp(-(fx / ax) ** 2 - (fy / ay) ** 2)
        n = np.real(np.fft.ifft2(f)); return (n - n.mean()) / (n.std() + 1e-9)
    ridge = 1 - np.abs(band(0.035, 0.006))                # 竖着的沟
    fine = band(0.2, 0.08)
    v = np.clip(0.5 + 0.28 * ridge + 0.12 * fine, 0, 1)
    t = np.clip((v - 0.05) / 0.4, 0, 1)[..., None]        # 和 hero 树皮的色带位置一样（0.05 ~ 0.45）
    lin = np.array(c1)[None, None] * (1 - t) + np.array(c2)[None, None] * t
    rgb = np.power(lin, 1 / 2.2)
    return np.concatenate([rgb, np.ones((S, S, 1))], axis=2).astype(np.float32)


def grass_texture(S, colors, seed, blades=70):
    """一丛草：几十片从底部中间长出来、往两边弯的细长叶子，根部窄、颜色深"""
    rng = np.random.default_rng(seed)
    img = np.zeros((S, S, 4), np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    py, px = 1 - yy, xx
    base, tip = np.array(colors[0]), np.array(colors[1])
    wmin = 1.6 / S                                        # 叶尖也至少一个多像素宽，小贴图不断线
    for _ in range(blades):
        x0 = 0.5 + rng.normal(0, 0.06)
        ang = rng.normal(0, 0.16) + (x0 - 0.5) * 1.2
        bend = rng.normal(0, 0.35) + np.sign(ang) * 0.3
        L = rng.uniform(0.55, 0.98); w0 = rng.uniform(0.01, 0.02)
        g = rng.uniform(0.75, 1.15)
        pts = []
        for k in range(9):
            t = k / 8
            a = ang + bend * t * t
            pts.append((x0 + math.sin(a) * L * t * 0.9 if k == 0 else pts[-1][0] + math.sin(a) * L / 8,
                        0.0 if k == 0 else pts[-1][1] + math.cos(a) * L / 8))
        for k in range(8):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            pad = w0 + 2 / S
            c0, c1 = int(max(0, min(ax, bx) - pad) * S), int(min(1, max(ax, bx) + pad) * S) + 1
            r0, r1 = int(max(0, min(ay, by) - pad) * S), int(min(1, max(ay, by) + pad) * S) + 1
            if c1 <= c0 or r1 <= r0: continue
            sl = (slice(S - r1, S - r0), slice(c0, c1))
            d, tt = seg_dist(px[sl], py[sl], ax, ay, bx, by)
            t = (k + tt) / 8                                # 沿叶片 0（根）→ 1（尖）
            m = d < np.maximum(w0 * (1 - t) ** 0.8, wmin)
            col = (base[None, None] * (1 - t[..., None] ** 0.7) + tip[None, None] * t[..., None] ** 0.7) * g
            blk = img[sl]
            blk[m, :3] = col[m]; blk[m, 3] = 1
    return img


def make_image(name, rgba, folder):
    """存一份 PNG 方便看，并打包进 .blend（pipeline 追加素材时贴图一起带过去）"""
    h, w, _ = rgba.shape
    a = rgba[..., 3:4]
    fill = (rgba[..., :3] * a).sum((0, 1)) / max(a.sum(), 1)       # 透明处填平均色，避免 mipmap 黑边
    rgba[..., :3] = np.where(a > 0.01, rgba[..., :3], fill)
    img = bpy.data.images.new(name, w, h, alpha=True)
    img.pixels.foreach_set(np.clip(rgba[::-1], 0, 1).astype(np.float32).ravel())   # Blender 图片从下往上
    os.makedirs(folder, exist_ok=True)
    img.filepath_raw = os.path.join(folder, name + '.png'); img.file_format = 'PNG'; img.save()
    img.pack()
    return img


# ---------------- 材质：贴图 × 顶点颜色，GLB 能带走 ----------------
def export_material(name, img, alpha_clip, single_sided=False):
    """底色 = 贴图 × 顶点颜色 Col；叶子用贴图透明度抠轮廓（导出成 alphaMode MASK）。
    single_sided：只画正面（草的每张卡片正反各做了一份，法线都朝上，网页里不会一面亮一面黑）"""
    m = bpy.data.materials.new(name)
    m.use_backface_culling = single_sided
    try: m.surface_render_method = 'DITHERED'
    except Exception: pass
    N, L = m.node_tree.nodes, m.node_tree.links
    bs = N['Principled BSDF']
    bs.inputs['Roughness'].default_value = 0.8 if alpha_clip else 0.95
    bs.inputs['Specular IOR Level'].default_value = 0.2 if alpha_clip else 0.3
    tex = N.new('ShaderNodeTexImage'); tex.image = img
    vc = N.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'
    mul = N.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    L.new(tex.outputs['Color'], mul.inputs['A']); L.new(vc.outputs['Color'], mul.inputs['B'])
    L.new(mul.outputs['Result'], bs.inputs['Base Color'])
    if alpha_clip:
        rnd = N.new('ShaderNodeMath'); rnd.operation = 'ROUND'            # glTF 导出认这个写法：透明度 0.5 处一刀切
        L.new(tex.outputs['Alpha'], rnd.inputs[0]); L.new(rnd.outputs[0], bs.inputs['Alpha'])
    return m


# ---------------- 几何：枝干管子 + 叶片卡片 ----------------
class TreeBuilder:
    def __init__(self, seed, q):
        self.rng = random.Random(seed)
        self.q = q
        self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new('UVMap')
        self.vattr = []          # 每个顶点的 (是否叶片, 叶色随机数, 树冠内外 0~1)

    def rvec(self):
        r = self.rng
        v = Vector((r.gauss(0, 1), r.gauss(0, 1), r.gauss(0, 1)))
        return v.normalized() if v.length > 1e-6 else Vector((1, 0, 0))

    def tube(self, pts, radii, sides, tile=0.9, stride=1):
        """沿 pts 的管子；树皮贴图横向按周长重复、竖向每 tile 米重复一次。stride > 1 时隔几个点取一个（细枝不用那么多段）"""
        if sides < 3: return
        if stride > 1 and len(pts) > 2:
            idx = list(range(0, len(pts), stride))
            if idx[-1] != len(pts) - 1: idx.append(len(pts) - 1)
            pts, radii = [pts[i] for i in idx], [radii[i] for i in idx]
        bm = self.bm
        n = None; prev = None; vprev = 0.0; vcur = 0.0
        urep = max(1, round(2 * math.pi * radii[0] / tile))
        for i, p in enumerate(pts):
            t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
            n = t.orthogonal().normalized() if n is None else (n - t * n.dot(t)).normalized()
            b = t.cross(n)
            if i: vcur += (p - pts[i - 1]).length / tile
            ring = []
            for s in range(sides):
                a = 2 * math.pi * s / sides
                ring.append(bm.verts.new(p + (n * math.cos(a) + b * math.sin(a)) * radii[i]))
                self.vattr.append((0.0, 0.0, min(1.0, 0.45 + 0.08 * p.z)))     # 树皮：越靠下越暗一点
            if prev:
                for s in range(sides):
                    f = bm.faces.new((prev[s], prev[(s + 1) % sides], ring[(s + 1) % sides], ring[s]))
                    f.material_index = 0; f.smooth = True
                    u0, u1 = s / sides * urep, (s + 1) / sides * urep
                    for loop, uv in zip(f.loops, ((u0, vprev), (u1, vprev), (u1, vcur), (u0, vcur))):
                        loop[self.uv].uv = uv
            prev = ring; vprev = vcur

    def card(self, p, up, w, l, var, shade, bend=0.15):
        """一张叶片卡片：底边在 p，沿 up 方向长 l，宽 w；分两段时中间一排顶点略微下弯"""
        side = up.cross(self.rvec())
        if side.length < 1e-4: side = up.orthogonal()
        side.normalize()
        rows = [(0.0, 0.0), (0.5, bend * 0.4), (1.0, bend)] if self.q['card_rows'] == 2 else [(0.0, 0.0), (1.0, bend * 0.6)]
        vs = []
        for v, drop in rows:
            c = p + up * (l * v) - Vector((0, 0, 1)) * (l * drop)
            vs.append((self.bm.verts.new(c - side * w / 2), self.bm.verts.new(c + side * w / 2)))
            self.vattr += [(1.0, var, shade), (1.0, var, shade)]
        for k in range(len(rows) - 1):
            f = self.bm.faces.new((vs[k][0], vs[k][1], vs[k + 1][1], vs[k + 1][0]))
            f.material_index = 1; f.smooth = True
            v0, v1 = rows[k][0], rows[k + 1][0]
            for loop, uv in zip(f.loops, ((0, v0), (1, v0), (1, v1), (0, v1))):
                loop[self.uv].uv = uv

    def cards(self, avg):
        """平均 avg 张（比如 0.5 就是一半的点挂一张，1.5 就是一张或两张）"""
        k = int(avg); return k + (1 if self.rng.random() < avg - k else 0)

    def finish(self, name):
        me = bpy.data.meshes.new(name); self.bm.to_mesh(me); self.bm.free()
        return me, np.array(self.vattr, dtype=np.float32)


def make_oak(name, H, D, seed, q):
    tb = TreeBuilder(seed, q); rng = tb.rng
    cen = Vector((0, 0, H * 0.6)); rad = Vector((D / 2, D / 2, H * 0.42))
    def env(p):            # 相对树冠椭球的"半径"：<1 在里面
        r = p - cen; return math.sqrt((r.x / rad.x) ** 2 + (r.y / rad.y) ** 2 + (r.z / rad.z) ** 2)
    leaves = []

    def grow(p, d, L, r0, lvl):
        segs = max(3, int(L / 0.3)); step = L / segs
        pts, radii = [p.copy()], [r0 * (1.35 if lvl == 0 else 1.0)]
        cur = p.copy()
        for k in range(segs):
            r = cur - cen
            d = d + tb.rvec() * (0.12 + 0.08 * lvl)
            d.z += (0.06 if lvl < 2 else -0.05)
            if env(cur) > 0.92 and lvl > 0:
                d -= Vector((r.x / rad.x ** 2, r.y / rad.y ** 2, r.z / rad.z ** 2)).normalized() * 0.5
            d.normalize()
            cur = cur + d * step
            pts.append(cur.copy()); radii.append(r0 * (1 - 0.7 * (k + 1) / segs) + 0.004)
            if env(cur) > 1.08 and lvl > 0: break
        tb.tube(pts, radii, q['oak_sides'][lvl], stride=q['oak_stride'][lvl])
        if lvl == 0:
            k = rng.randint(4, 6); az0 = rng.uniform(0, 6.28)
            for i in range(k):
                az = az0 + 2 * math.pi * i / k + rng.uniform(-0.3, 0.3)
                a = math.radians(rng.uniform(30, 62))
                cd = Vector((math.cos(az) * math.sin(a), math.sin(az) * math.sin(a), math.cos(a)))
                grow(pts[-1] - d * 0.2, cd, H * rng.uniform(0.42, 0.55), radii[-1] * 0.75, 1)
            return
        nkids = [0, 8, 6, 0][lvl]
        for i in range(nkids):
            t = rng.uniform(0.3, 0.97)
            idx = min(len(pts) - 2, int(t * (len(pts) - 1)))
            bp = pts[idx]; bd = (pts[idx + 1] - pts[idx]).normalized()
            perp = bd.cross(tb.rvec()).normalized()
            a = math.radians(rng.uniform(35, 60))
            cd = (bd * math.cos(a) + perp * math.sin(a)).normalized()
            grow(bp, cd, L * rng.uniform(0.45, 0.7) * (1.1 - 0.4 * t), radii[idx] * 0.6, lvl + 1)
        if lvl >= 2:
            for i in range(1, len(pts)):
                if lvl == 2 and i < len(pts) * 0.3: continue
                for _ in range(3):                       # hero 原版每点 3 张；这里先记下点，按档位再挑
                    leaves.append((pts[i], (pts[i] - pts[i - 1]).normalized()))

    grow(Vector((0, 0, -0.4)), Vector((rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 1)).normalized(),
         H * rng.uniform(0.3, 0.36), 0.028 * H, 0)
    keep = q['oak_cards'] / 3
    n = 0
    for p, d in leaves:
        if rng.random() > keep: continue
        out = (p - cen); out.z *= 0.6
        up = (d + out.normalized() * 0.6 + Vector((0, 0, 0.3)) + tb.rvec() * 0.5).normalized()
        sh = min(1.0, env(p) ** 2)
        s = rng.uniform(0.95, 1.4) * q['oak_scale']
        tb.card(p + tb.rvec() * 0.25, up, s * 0.9, s, rng.random(), sh, bend=0.12)
        n += 1
    print(f'  {name}: {n} 张叶片')
    return tb.finish(name)


def make_pine(name, H, D, seed, q):
    tb = TreeBuilder(seed, q); rng = tb.rng
    cards = []
    pts, radii = [], []
    n = 16 if q['pine_sides'] >= 4 else 10
    lean = Vector((rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0))
    for k in range(n + 1):
        t = k / n
        pts.append(Vector((0, 0, -0.4 + (H + 0.4) * t)) + lean * t * H + Vector((math.sin(t * 9 + seed) * 0.04, 0, 0)))
        radii.append(0.022 * H * (1 - t) ** 0.9 * (1.4 if k == 0 else 1.0) + 0.012)
    tb.tube(pts, radii, q['pine_sides'] + 3)
    def trunk_at(z):
        t = clampf((z + 0.4) / (H + 0.4)); i = min(n - 1, int(t * n))
        return pts[i].lerp(pts[i + 1], t * n - i), radii[i]
    # 下部几根枯枝桩
    for _ in range(rng.randint(4, 8)):
        z = rng.uniform(0.8, H * 0.3); p, r = trunk_at(z)
        az = rng.uniform(0, 6.28); d = Vector((math.cos(az), math.sin(az), rng.uniform(-0.3, 0.1))).normalized()
        L = rng.uniform(0.3, 0.9)
        tb.tube([p, p + d * L * 0.5, p + d * L], [0.03, 0.02, 0.006], 3)
    # 轮生枝
    z = H * rng.uniform(0.2, 0.26)
    while z < H * 0.97:
        t = z / H
        k = rng.randint(5, 7); az0 = rng.uniform(0, 6.28)
        for i in range(k):
            az = az0 + 2 * math.pi * i / k + rng.uniform(-0.35, 0.35)
            Lb = min(D / 2 * 1.1, (H - z) * 0.55 + 0.3) * rng.uniform(0.8, 1.1)
            droop = rng.uniform(-0.35, -0.05) * (1 - t)
            p, r = trunk_at(z)
            d = Vector((math.cos(az), math.sin(az), droop)).normalized()
            segs = max(2, int(Lb / q['pine_seg'])); bpts, brad = [p], [max(0.012, r * 0.35)]
            cur = p.copy()
            for s in range(segs):
                u = (s + 1) / segs
                d = (d + Vector((0, 0, 0.09 * u * q['pine_seg'] / 0.25)) + tb.rvec() * 0.06).normalized()   # 枝梢上翘（段长了每段多翘一点）
                cur = cur + d * (Lb / segs); bpts.append(cur.copy()); brad.append(brad[0] * (1 - 0.8 * u) + 0.003)
            tb.tube(bpts, brad, q['pine_sides'])
            # 侧枝：只记下位置挂针叶（细枝管子默认不建，针叶卡片能盖住）
            for s in range(1, len(bpts) - 1):
                for _ in range(2 if rng.random() < 0.6 else 1):
                    bd = (bpts[s + 1] - bpts[s]).normalized()
                    side = bd.cross(Vector((0, 0, 1))).normalized() * rng.choice((-1, 1))
                    sd = (bd * 0.6 + side * 0.8 + Vector((0, 0, rng.uniform(-0.1, 0.15)))).normalized()
                    sl = Lb * rng.uniform(0.25, 0.4) * (1 - s / len(bpts) * 0.5) * (q['pine_seg'] / 0.25) ** 0.3
                    sp = [bpts[s], bpts[s] + sd * sl]
                    if q['pine_twigs']: tb.tube(sp, [brad[s] * 0.5, 0.003], 3)
                    cards.append((sp[0], sd, sl, 0.8))
            for s in range(1, len(bpts)):
                cards.append((bpts[s - 1], (bpts[s] - bpts[s - 1]).normalized(), (bpts[s] - bpts[s - 1]).length, 1.0))
        z += rng.uniform(0.26, 0.36) * (1 - 0.3 * t)
    tp, _ = trunk_at(H * 0.9)
    cards.append((tp, Vector((0, 0, 1)), H * 0.12, 1.0))
    cnt = 0
    for p, d, L, sc in cards:
        m = max(1, int(L / q['pine_step'] + rng.random()))
        for j in range(m):
            r = p + d * (L * (j + rng.random()) / m)
            rot = Matrix.Rotation(rng.uniform(-math.pi, math.pi), 3, d)
            up = (rot @ (d.orthogonal().normalized() * 0.55) + d).normalized()
            s = rng.uniform(0.5, 0.72) * sc * q['pine_scale']
            rr = r.to_2d().length / (D / 2 + 1e-3)
            tb.card(r, up, s * 0.95, s, rng.random(), min(1.0, 0.25 + rr), bend=0.18)
            cnt += 1
    print(f'  {name}: {cnt} 张针叶')
    return tb.finish(name)


def make_shrub(name, H, D, seed, q):
    """写实灌木：从根部散开 6~9 根枝，每根再分两三根细枝，外圈挂叶片簇卡片"""
    tb = TreeBuilder(seed, q); rng = tb.rng
    cen = Vector((0, 0, H * 0.45)); rad = Vector((D / 2, D / 2, H * 0.55))
    def env(p):
        r = p - cen; return math.sqrt((r.x / rad.x) ** 2 + (r.y / rad.y) ** 2 + (r.z / rad.z) ** 2)
    spots = []
    def stem(p, d, L, r, lvl):
        segs = 3 if lvl else 4; pts, radii = [p], [r]; cur = p.copy()
        for k in range(segs):
            d = (d + tb.rvec() * 0.25 + Vector((0, 0, 0.05))).normalized()
            cur = cur + d * (L / segs); pts.append(cur.copy()); radii.append(r * (1 - 0.75 * (k + 1) / segs) + 0.003)
            if env(cur) > 1.0: break
        tb.tube(pts, radii, 3, tile=0.5)
        for p2 in pts[1:]:
            if env(p2) > 0.45: spots.append((p2, (p2 - p).normalized()))
        if lvl == 0:
            for _ in range(rng.randint(2, 3)):
                i = rng.randint(1, len(pts) - 1); bd = (pts[i] - pts[i - 1]).normalized()
                cd = (bd + bd.cross(tb.rvec()).normalized() * 0.8).normalized()
                stem(pts[i], cd, L * rng.uniform(0.45, 0.65), radii[i] * 0.6, 1)
    for i in range(rng.randint(6, 9)):
        az = rng.uniform(0, 6.28); a = math.radians(rng.uniform(20, 65))
        d = Vector((math.cos(az) * math.sin(a), math.sin(az) * math.sin(a), math.cos(a)))
        stem(Vector((rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), -0.1)), d, max(H, D / 2) * rng.uniform(0.85, 1.1), 0.035, 0)
    n = 0
    for p, d in spots:
        for _ in range(tb.cards(q['shrub_cards'])):
            up = (d + (p - cen).normalized() * 0.5 + tb.rvec() * 0.6).normalized()
            s = rng.uniform(0.6, 0.85) * q['shrub_scale']
            tb.card(p + tb.rvec() * 0.3, up, s * 0.9, s, rng.random(), min(1.0, env(p) ** 2 + 0.2), bend=0.1)
            n += 1
    print(f'  {name}: {n} 张叶片')
    return tb.finish(name)


def make_grass(name, H, D, seed, q):
    """一丛草：grass_cards 张竖直卡片绕中心均匀转开，上半截往外倒一点。
    每张卡片正反两份（反过来的那份顶点顺序相反），法线大体朝上：草的明暗只跟高度有关，不会因为朝向忽亮忽暗"""
    rng = random.Random(seed)
    bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap')
    col, nrm = [], []                                  # 每个顶点的颜色；每个面角的法线
    n = q['grass_cards']; rows = [0.0, 0.5, 1.0] if q['grass_rows'] == 2 else [0.0, 1.0]
    a0 = rng.uniform(0, math.pi)
    for i in range(n):
        a = a0 + math.pi * i / n + rng.uniform(-0.15, 0.15)
        side = Vector((math.cos(a), math.sin(a), 0)); out = Vector((-side.y, side.x, 0)) * rng.choice((-1, 1))
        w = min(D, H * 1.2) * rng.uniform(0.8, 1.0); lean = rng.uniform(0.03, 0.1) * H
        shade = rng.uniform(0.85, 1.1)
        for front in (True, False):                     # 正反两份各用各的顶点（同一组顶点不能建两个面）
            ring = []
            for v in rows:
                c = out * lean * v * v + Vector((0, 0, H * v * (1 - 0.06 * v)))
                k = (0.55 + 0.45 * v) * shade              # 根部暗（假 AO），叶尖亮
                ring.append([(bm.verts.new(c - side * w / 2), 0.0), (bm.verts.new(c + side * w / 2), 1.0)])
                col += [k, k]
            fn = out if front else -out
            for r in range(len(rows) - 1):
                (a_, ua), (b_, ub) = ring[r]; (c_, uc), (d_, ud) = ring[r + 1]       # 左下、右下；左上、右上
                quad = [(a_, ua, rows[r]), (b_, ub, rows[r]), (d_, ud, rows[r + 1]), (c_, uc, rows[r + 1])]
                if not front: quad = quad[::-1]
                f = bm.faces.new([vv for vv, _, _ in quad])
                for loop, (_, u, v) in zip(f.loops, quad):
                    loop[uv].uv = (u, v)
                    nrm.append((Vector((0, 0, 1)) + fn * 0.35).normalized())
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.normals_split_custom_set([tuple(v) for v in nrm])
    rgb = np.clip(np.array(col, np.float32)[:, None] * np.ones(3), 0, 1)
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set('color', np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32).ravel())
    me.color_attributes.active_color = attr
    return me


# ---------------- 顶点颜色 ----------------
def leaf_gain(tint):
    """叶色乘数最大能到 2.8（hero 的值），顶点颜色存不下 >1 的数：先把贴图提亮 gain 倍，顶点颜色再除以 gain"""
    a, b = tint
    return max(max(a), max(b)) * 1.35


def set_colors(me, va, tint, gain, tree_rand):
    """叶片：两色之间按随机数取色 × 树冠内暗外亮（hero 里在材质节点算，这里直接写进顶点）；枝干：只有明暗"""
    is_leaf, var, shade = va[:, 0], va[:, 1], va[:, 2]
    f = np.clip(var * 0.7 + tree_rand, 0, 1)[:, None]
    a, b = np.array(tint[0]), np.array(tint[1])
    ao = (0.7 + 0.65 * shade)[:, None]                   # 0 → 0.7（树冠里面），1 → 1.35（外面）
    leaf = (a * (1 - f) + b * f) * ao / gain
    bark = np.repeat(shade[:, None], 3, axis=1)
    rgb = np.clip(np.where(is_leaf[:, None] > 0.5, leaf, bark), 0, 1)
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set('color', np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32).ravel())
    me.color_attributes.active_color = attr


def tris(me): return sum(len(p.vertices) - 2 for p in me.polygons)


def low_heights(folder):
    """低档素材的实际高度（包围盒），新素材缩放到一样高，换档后场景里的东西大小不变"""
    with bpy.data.libraries.load(os.path.join(LOW, folder, f'{folder}_export.blend'), link=False) as (src, dst):
        dst.objects = list(src.objects)
    h = {o.name: o.dimensions.z for o in dst.objects if o and o.type == 'MESH'}
    for o in dst.objects:
        if o: me = o.data; bpy.data.objects.remove(o); bpy.data.meshes.remove(me)
    return h


# ---------------- 主程序 ----------------
def build_tier(tier):
    q = TIERS[tier]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(OUT, tier); tex_dir = os.path.join(base, 'textures')
    print(f'== {tier}')
    imgs = {'oak': make_image('oak_leaves', leaf_texture(q['tex'], 1), tex_dir),
            'pine': make_image('pine_needles', needle_texture(q['tex'], 2), tex_dir)}
    gains = {k: leaf_gain(v) for k, v in TINTS.items()}
    # 叶片贴图提亮：同一张贴图给几种叶色共用时取最大的 gain，各自的顶点颜色再除以这个数
    gain_oak = max(gains['OakLeaves'], gains['OliveLeaves'], gains['HillBush'])
    gain_pine = gains['PineNeedles']
    for key, g in (('oak', gain_oak), ('pine', gain_pine)):
        px = np.empty(len(imgs[key].pixels), np.float32); imgs[key].pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        px[:, :3] = np.clip(px[:, :3] * g ** (1 / 2.2), 0, 1)        # 贴图存的是 sRGB 值，线性的 g 倍 ≈ sRGB 的 g^(1/2.2) 倍
        imgs[key].pixels.foreach_set(px.ravel()); imgs[key].update(); imgs[key].pack()
    barks = {k: make_image(f'bark_{k}', bark_texture(q['bark_tex'], c1, c2, 7 + i), tex_dir)
             for i, (k, (c1, c2)) in enumerate(BARKS.items())}
    M = {'bark_oak': export_material('hq_bark_oak', barks['oak'], False),
         'bark_pine': export_material('hq_bark_pine', barks['pine'], False),
         'oak': export_material('hq_oak_leaves', imgs['oak'], True),
         'pine': export_material('hq_pine_needles', imgs['pine'], True)}

    stats = {}
    for folder in ('trees', 'shrubs'):
        with open(os.path.join(LOW, folder, f'{folder}_manifest.json'), encoding='utf-8') as f:
            spec = json.load(f)['assets']
        heights = low_heights(folder)
        colls = []
        for i, a in enumerate(spec):
            name, H, D = a['name'], a['height_m'], a['crown_diameter_m']
            rand = random.Random(name).uniform(0, 0.3)               # 每款素材固定的一点色差
            if a['kind'] == 'tree' and a['species'] == 'oak':
                me, va = make_oak(name, H, D, 10 + i, q); mats, tint, g = (M['bark_oak'], M['oak']), TINTS['OakLeaves'], gain_oak
            elif a['kind'] == 'tree':
                me, va = make_pine(name, H, D, 20 + i, q); mats, tint, g = (M['bark_pine'], M['pine']), TINTS['PineNeedles'], gain_pine
            else:
                tname = 'OliveLeaves' if a['species'] == 'olive' else 'HillBush'
                me, va = make_shrub(name, H, D, 40 + i, q); mats, tint, g = (M['bark_oak'], M['oak']), TINTS[tname], gain_oak
            set_colors(me, va, tint, g, rand)
            for m in mats: me.materials.append(m)
            # 缩放到和低档素材一样高（底部仍在 z = 0 附近）
            zs = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', zs); zs = zs.reshape(-1, 3)
            k = heights[name] / float(zs[:, 2].max() - zs[:, 2].min())
            me.transform(Matrix.Scale(k, 4))
            c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
            ob = bpy.data.objects.new(name, me); c.objects.link(ob)
            colls.append(c)
            stats[name] = {'triangles': tris(me), 'height_m': round(heights[name], 2), 'scale': round(k, 3)}
        os.makedirs(os.path.join(base, folder), exist_ok=True)
        path = os.path.join(base, folder, f'{folder}_export.blend')
        bpy.data.libraries.write(path, set(colls), fake_user=True)
        print('已保存', path)

    # 草：绿草、干草各一张贴图，每款素材一丛
    gm = {k: export_material(f'hq_grass_{k}', make_image(f'grass_{k}', grass_texture(q['grass_tex'], GRASS_COLORS[k], 30 + i), tex_dir),
                             True, single_sided=True) for i, k in enumerate(GRASS_COLORS)}
    with open(os.path.join(LOW, 'grass', 'grass_manifest.json'), encoding='utf-8') as f:
        spec = json.load(f)['assets']
    heights = low_heights('grass')
    colls = []
    for i, a in enumerate(spec):
        name = a['name']
        me = make_grass(name, heights[name], a['crown_diameter_m'] * 1.1, 60 + i, q)
        me.materials.append(gm[a['species']])
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        ob = bpy.data.objects.new(name, me); c.objects.link(ob)
        colls.append(c)
        stats[name] = {'triangles': tris(me), 'height_m': round(heights[name], 2), 'scale': 1.0}
    os.makedirs(os.path.join(base, 'grass'), exist_ok=True)
    path = os.path.join(base, 'grass', 'grass_export.blend')
    bpy.data.libraries.write(path, set(colls), fake_user=True)
    print('已保存', path)

    with open(os.path.join(base, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    for k, v in stats.items(): print(f'  {k:16s} {v["triangles"]:6d} 个三角形')


def preview():
    """低 / 中 / 高 各渲染一张（摆法、机位、灯光和 hero/renders/trees_preview.png 一样），从上到下拼成一张图"""
    panels = []
    for tier, root in (('low', LOW), ('medium', os.path.join(OUT, 'medium')), ('high', os.path.join(OUT, 'high'))):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        sc = bpy.context.scene
        i = 0
        for folder in ('trees', 'shrubs'):
            with bpy.data.libraries.load(os.path.join(root, folder, f'{folder}_export.blend'), link=False) as (src, dst):
                dst.collections = list(src.collections)
            for c in sorted(dst.collections, key=lambda c: c.name):
                e = bpy.data.objects.new(c.name, None)
                e.instance_type = 'COLLECTION'; e.instance_collection = c
                e.location = ((i - 3.5) * 9, 0, 0) if i < 8 else ((i - 10) * 6, -14, 0)
                sc.collection.objects.link(e); i += 1
        # 前面一片草地（每档同样的位置）
        with bpy.data.libraries.load(os.path.join(root, 'grass', 'grass_export.blend'), link=False) as (src, dst):
            dst.collections = list(src.collections)
        gc = sorted(dst.collections, key=lambda c: c.name)
        rng = random.Random(5)
        for k in range(500):
            c = gc[k % len(gc)] if rng.random() < 0.5 else gc[rng.randrange(len(gc))]
            e = bpy.data.objects.new(c.name, None)
            e.instance_type = 'COLLECTION'; e.instance_collection = c
            e.location = (rng.uniform(-34, 34), rng.uniform(-27, -17), 0); e.rotation_euler = (0, 0, rng.uniform(0, 6.28))
            sc.collection.objects.link(e)
        bpy.ops.mesh.primitive_plane_add(size=200)
        g = bpy.context.active_object; gm = bpy.data.materials.new('g'); g.data.materials.append(gm)
        gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.15, 0.17, 0.08, 1)
        w = bpy.data.worlds.new('w'); sc.world = w
        w.node_tree.nodes['Background'].inputs[0].default_value = (0.45, 0.6, 0.85, 1)
        sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 4; so = bpy.data.objects.new('s', sun)
        so.rotation_euler = (math.radians(50), 0, math.radians(-40)); sc.collection.objects.link(so)
        cd = bpy.data.cameras.new('c'); cd.lens = 35; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam)
        cam.location = (0, -66, 8); cam.rotation_euler = (Vector((0, 0, 5)) - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.camera = cam; sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 32
        sc.render.resolution_x, sc.render.resolution_y = 1920, 640
        sc.view_settings.view_transform = 'AgX'
        path = os.path.join(OUT, 'previews', f'{tier}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        panels.append(path)
    rows = []
    for path in panels:
        im = bpy.data.images.load(path); px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px)
        rows.append(px.reshape(im.size[1], im.size[0], 4))
    out = np.concatenate(rows[::-1], axis=0)                # Blender 图片从下往上存，低档放最上面
    im = bpy.data.images.new('tiers_compare', out.shape[1], out.shape[0])
    im.pixels.foreach_set(out.ravel())
    im.filepath_raw = os.path.join(OUT, 'previews', 'tiers_compare.png'); im.file_format = 'PNG'; im.save()
    print('对比图（从上到下：低 / 中 / 高）：', im.filepath_raw)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    tiers = [t for t in argv if t in TIERS] or list(TIERS)
    for t in tiers: build_tier(t)
    if '--preview' in argv: preview()


if __name__ == '__main__':
    main()
