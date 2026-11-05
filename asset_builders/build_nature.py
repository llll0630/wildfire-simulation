# ===============================================================
#  作物、果园和自然小物件（三档）—— Blender 5.x
#    作物（一株）：大豆、小麦、棉花（玉米在 build_props.py）
#    果园 / 葡萄园：苹果树、柑橘树、一段葡萄架（立柱 + 铁丝 + 藤）
#    石头：灰色 / 沙漠红色的大石块、碎石堆，几种大小
#    枯木：站立的枯树（没有叶子）、倒木、树桩（正常森林里都有）
#    岩体：大岩块（6 m、12 m）、一段崖壁（20 m 宽 × 15 m 高）、护岸抛石（一段）、河卵石（一堆）
#    林下：蕨类（两款）、贴地的落枝落叶片
#  这一步只准备素材；pipeline 怎么用以后再定。尺寸按真实尺寸（米）。
#  三档：低 = 简单几何 + 顶点颜色；中、高 = 和分树种植被一样的做法（枝干 + 透明叶片卡片、石头带贴图），高档更细
#
#  用法：blender -b --factory-startup --python build_nature.py -- [--preview]
#  输出：assets_hq/<档位>/nature/nature_export.blend、assets_hq/nature_manifest.json、assets_hq/previews/nature_<档位>.png
#  石头贴图用地面素材库下载的 CC0 贴图（assets_hq/ground/sources/ 的 Rock062、Rock029）
# ===============================================================
import bpy, bmesh, math, random, os, sys, json, zipfile, tempfile
import numpy as np
from mathutils import Vector, Matrix, noise

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv
import build_species as bs
import build_props as bp

ROOT = os.path.dirname(HERE)
HQ = os.path.join(ROOT, 'assets_hq')
TIERS = ('low', 'medium', 'high')
GROUND_SRC = os.path.join(HQ, 'ground', 'sources')


# ---------------- 贴图 ----------------
def tex_wheat(S):
    """小麦：一丛金黄的麦秆，顶上一个个麦穗"""
    rng = np.random.default_rng(41)
    img = hv.grass_texture(S, ((0.42, 0.34, 0.15), (0.82, 0.70, 0.40)), 41, blades=30)
    _, px, py = bs._canvas(S)
    for _ in range(22):
        x0 = 0.5 + rng.normal(0, 0.14); top = rng.uniform(0.7, 0.95); lean = rng.normal(0, 0.05)
        bs._stroke(img, px, py, x0, 0.0, x0 + lean, top, 0.004, (0.70, 0.60, 0.32), S)
        for k in range(8):                                       # 麦穗：一节节小颗粒
            y = top + k * 0.012
            bs._stroke(img, px, py, x0 + lean - 0.008, y, x0 + lean + 0.008, y + 0.01, 0.006, (0.86, 0.74, 0.42), S)
    return img


def tex_cotton(S):
    """棉花：深绿的阔叶，中间点缀白色棉桃"""
    rng = np.random.default_rng(42)
    img = bs._leaf_twig(S, 42, 'lobed', 9, 0.22, (0.14, 0.24, 0.08))
    _, px, py = bs._canvas(S)
    for _ in range(9):
        cx, cy, r = rng.uniform(0.25, 0.75), rng.uniform(0.2, 0.9), rng.uniform(0.03, 0.05)
        m = np.hypot(px - cx, py - cy) < r
        img[m, :3] = (0.92, 0.92, 0.88); img[m, 3] = 1
    return img


def tex_fruit_leaves(S, fruit_col, seed):
    """果树叶片簇：椭圆叶，夹着几个果子"""
    rng = np.random.default_rng(seed)
    img = bs._leaf_twig(S, seed, 'oval', 18, 0.18, (0.15, 0.27, 0.08))
    _, px, py = bs._canvas(S)
    for _ in range(5):
        cx, cy, r = rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.85), rng.uniform(0.035, 0.05)
        d = np.hypot(px - cx, py - cy); m = d < r
        shade = np.clip(1.15 - d / r * 0.5, 0.6, 1.15)[..., None]
        img[m, :3] = (np.array(fruit_col)[None, None] * shade)[m]; img[m, 3] = 1
    return img


def tex_fern(S):
    """蕨类：一片羽状复叶，中轴两侧排满小羽片，越往尖越小"""
    rng = np.random.default_rng(47); img, px, py = bs._canvas(S)
    bs._stroke(img, px, py, 0.5, 0.0, 0.52, 0.97, 0.006, (0.2, 0.3, 0.1), S)
    v = 0.05
    while v < 0.95:
        L = 0.38 * (1 - v) ** 0.7 + 0.03
        for side in (-1, 1):
            ang = side * math.radians(70 - 20 * v)
            x0, y0 = 0.5 + 0.02 * v, v
            for k in range(6):                                   # 每片小羽片再分几个小裂片
                t = (k + 0.5) / 6
                cx, cy = x0 + math.sin(ang) * L * t, y0 + math.cos(ang) * L * t
                bs._stroke(img, px, py, cx, cy, cx + 0.012, cy + 0.02, 0.012 * (1.1 - t), (0.17 + 0.04 * rng.random(), 0.32, 0.10), S)
            bs._stroke(img, px, py, x0, y0, x0 + math.sin(ang) * L, y0 + math.cos(ang) * L, 0.003, (0.22, 0.32, 0.12), S)
        v += 0.035
    return img


def tex_debris(S):
    """林下落枝落叶：一块地上散落的枯枝、落叶、松果（透明背景，平铺在地面上）"""
    rng = np.random.default_rng(48); img, px, py = bs._canvas(S)
    for _ in range(30):                                          # 枯枝
        x, y, a, L = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(0, 3.14), rng.uniform(0.08, 0.35)
        bs._stroke(img, px, py, x, y, x + math.cos(a) * L, y + math.sin(a) * L, rng.uniform(0.003, 0.008),
                   (0.25 + 0.05 * rng.random(), 0.19, 0.12), S)
    for _ in range(220):                                         # 落叶
        x, y, a = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 3.14)
        c = [(0.42, 0.28, 0.12), (0.35, 0.22, 0.10), (0.48, 0.38, 0.18), (0.30, 0.25, 0.12)][rng.integers(4)]
        bs._stroke(img, px, py, x, y, x + math.cos(a) * 0.02, y + math.sin(a) * 0.02, 0.012, c, S)
    return img


FOLIAGE = {'soy': lambda S: bs._leaf_twig(S, 43, 'round', 12, 0.2, (0.18, 0.30, 0.09)),
           'wheat': tex_wheat, 'cotton': tex_cotton,
           'apple': lambda S: tex_fruit_leaves(S, (0.62, 0.08, 0.06), 44),
           'citrus': lambda S: tex_fruit_leaves(S, (0.85, 0.48, 0.05), 45),
           'grape': lambda S: bs._leaf_twig(S, 46, 'lobed', 10, 0.26, (0.20, 0.32, 0.10)),
           'fern': tex_fern, 'debris': tex_debris}
GAIN = {'soy': 1.6, 'wheat': 1.0, 'cotton': 1.4, 'apple': 1.6, 'citrus': 1.5, 'grape': 1.6, 'fern': 1.5, 'debris': 1.0}


def cc0_image(asset, name):
    """从地面素材库的贴图包里取颜色贴图，载入成 Blender 图片（打包进 .blend）"""
    with zipfile.ZipFile(os.path.join(GROUND_SRC, f'{asset}_1K-JPG.zip')) as z:
        data = z.read(f'{asset}_1K-JPG_Color.jpg')
    path = os.path.join(tempfile.gettempdir(), f'{name}.jpg')
    with open(path, 'wb') as f: f.write(data)
    img = bpy.data.images.load(path); img.name = name; img.pack()
    return img


# ---------------- 几何 ----------------
def rock_mesh(name, size, seed, tier, flat=0.6):
    """石块：细分球加噪声起伏，底部压平埋进地里一点；UV 按方块投影（贴图不拉伸）"""
    sub = {'low': 1, 'medium': 2, 'high': 3}[tier]
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
    off = Vector((seed * 7.1, seed * 3.3, seed * 1.7))
    for v in bm.verts:
        n = noise.fractal(v.co * 1.3 + off, 0.6, 2.0, 4)
        v.co *= 1 + 0.35 * n
        v.co.z *= flat
        v.co *= size / 2
        v.co.z = max(v.co.z, -size * 0.15)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, size * 0.1))
    uv = bm.loops.layers.uv.new('UVMap')
    for f in bm.faces:
        n = f.normal; ax = max(range(3), key=lambda i: abs(n[i]))
        for loop in f.loops:
            c = loop.vert.co; a, b = [(1, 2), (0, 2), (0, 1)][ax]
            loop[uv].uv = (c[a] / 2.0, c[b] / 2.0)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = tier != 'low'
    return me


def outcrop_mesh(name, w, h, d, seed, tier):
    """露出地面的大岩体：横向拉长的块状，表面有层理（水平方向的起伏）和裂隙"""
    sub = {'low': 2, 'medium': 3, 'high': 4}[tier]
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=sub * 2, use_grid_fill=True)
    off = Vector((seed * 5.3, seed * 2.1, seed * 8.7))
    for v in bm.verts:
        c = v.co.copy()
        n = noise.fractal(c * 2.2 + off, 0.55, 2.0, 5)
        strata = 0.06 * math.sin(c.z * 18 + noise.noise(c * 3 + off) * 2)           # 层理
        r = 1 + 0.28 * n + strata
        v.co = Vector((c.x * w, c.y * d, (c.z + 0.5) * h)) * 1.0
        v.co.x *= r; v.co.y *= r; v.co.z = max(-0.5, v.co.z * (1 + 0.15 * n))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, -h * 0.15))
    return _box_uv(bm, name, tier, scale=3.0)


def cliff_mesh(name, w, h, seed, tier):
    """一段崖壁（宽 w、高 h、朝 -y）：竖直的一片，表面按噪声前后起伏、有竖向的裂缝和横向的层理；背面平"""
    nx, nz = {'low': (8, 6), 'medium': (24, 18), 'high': (48, 36)}[tier]
    bm = bmesh.new(); rows = []
    off = Vector((seed * 3.1, seed * 7.7, 0))
    for j in range(nz + 1):
        row = []
        for i in range(nx + 1):
            x, z = (i / nx - 0.5) * w, j / nz * h
            p = Vector((x / 6, 0, z / 6)) + off
            depth = 1.6 * noise.fractal(p, 0.6, 2.0, 5) + 0.5 * math.sin(z * 1.3 + noise.noise(p) * 3)
            depth -= 0.8 * max(0.0, noise.ridged_multi_fractal(Vector((x / 3, 0, 0)) + off, 1.0, 2.0, 3, 1.0, 2.0) - 0.6)
            row.append(bm.verts.new((x, -depth - 1.5 * (1 - j / nz) ** 2, z)))
        rows.append(row)
    for j in range(nz):
        for i in range(nx):
            bm.faces.new((rows[j][i], rows[j][i + 1], rows[j + 1][i + 1], rows[j + 1][i]))
    return _box_uv(bm, name, tier, scale=4.0)


def _box_uv(bm, name, tier, scale):
    uv = bm.loops.layers.uv.new('UVMap')
    bm.normal_update()
    for f in bm.faces:
        n = f.normal; ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        for loop in f.loops:
            c = loop.vert.co; loop[uv].uv = (c[a] / scale, c[b] / scale)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = tier != 'low'
    return me


def rock_group(name, n, size_range, seed, tier, spread, flat=0.6, sharp=False):
    """一堆石头：n 块，大小在 size_range 里，散在 spread 的范围（x, y）里；sharp = 有棱角的护岸抛石"""
    rng = random.Random(seed); me = bpy.data.meshes.new(name); bm = bmesh.new()
    for j in range(n):
        part = rock_mesh('tmp', rng.uniform(*size_range), seed * 10 + j, 'low' if sharp else tier, flat=flat)
        part.transform(Matrix.Translation((rng.uniform(-spread[0], spread[0]), rng.uniform(-spread[1], spread[1]), 0))
                       @ Matrix.Rotation(rng.uniform(0, 6.28), 4, 'Z'))
        bm.from_mesh(part); bpy.data.meshes.remove(part)
    bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = not sharp and tier != 'low'
    return me


def debris_patch(tier):
    """林下落枝落叶：一块 4 m × 4 m 贴地的透明贴图片（低档没有，地面贴图里已经有落叶）"""
    m = bp.Mesh(tier)
    for k, (x, y, s_) in enumerate(((0, 0, 4.0), (1.2, -0.8, 2.5), (-1.0, 1.1, 2.0))):
        m.card((x, y - s_ / 2, 0.02), (0, 1, 0), (1, 0, 0), s_, s_, (1, 1, 1), mat=1)
    return m


def snag(sp_seed, H, q, tier):
    """枯立木：主干 + 几根断枝，没有叶子；顶上折断"""
    tb = bs.Builder(sp_seed, {'card_rows': 1}); rng = tb.rng
    n = 8
    pts = [Vector((math.sin(k * 0.7) * 0.15, 0, -0.3 + (H * 0.92) * k / n)) for k in range(n + 1)]
    sides = {'low': 5, 'medium': 7, 'high': 10}[tier]
    tb.tube(pts, [0.28 * (1 - 0.75 * k / n) + 0.04 for k in range(n + 1)], sides, tile=1.0)
    for _ in range({'low': 4, 'medium': 9, 'high': 14}[tier]):
        z = rng.uniform(H * 0.25, H * 0.85); az = rng.uniform(0, 6.28)
        p = Vector((0, 0, z)); d = Vector((math.cos(az), math.sin(az), rng.uniform(-0.4, 0.3))).normalized()
        L = rng.uniform(0.4, 2.0) * (1.2 - z / H)
        tb.tube([p, p + d * L * 0.5, p + d * L], [0.06, 0.04, 0.015], max(3, sides // 2), tile=0.5)
    return tb.finish('snag')


def fallen_log(seed, L, r, tier):
    """倒木：横躺的树干，两头断面、几根断枝桩"""
    tb = bs.Builder(seed, {'card_rows': 1}); rng = tb.rng
    sides = {'low': 6, 'medium': 9, 'high': 12}[tier]
    pts = [Vector((-L / 2 + L * k / 6, math.sin(k) * 0.08, r * 0.85)) for k in range(7)]
    tb.tube(pts, [r * (1 - 0.3 * k / 6) for k in range(7)], sides, tile=1.0)
    for _ in range({'low': 2, 'medium': 4, 'high': 6}[tier]):
        x = rng.uniform(-L / 3, L / 2.5); az = rng.uniform(0, 6.28)
        p = Vector((x, 0, r)); d = Vector((rng.uniform(-0.3, 0.3), math.cos(az), abs(math.sin(az)) + 0.2)).normalized()
        tb.tube([p, p + d * 0.5], [0.06, 0.03], max(3, sides // 2), tile=0.4)
    return tb.finish('log')


def stump(seed, r, tier):
    tb = bs.Builder(seed, {'card_rows': 1})
    sides = {'low': 6, 'medium': 10, 'high': 14}[tier]
    tb.tube([Vector((0, 0, -0.1)), Vector((0, 0, 0.25)), Vector((0, 0, 0.5))], [r * 1.35, r * 1.05, r], sides, tile=0.6)
    return tb.finish('stump')


def vineyard_row(tier, mats):
    """一段葡萄架（6 m）：两根立柱、三道铁丝、藤沿铁丝长满叶片卡片"""
    m = bp.Mesh(tier); L = 6.0
    for x in (0.0, L):
        m.box(x, 0, -0.4, 0.08, 0.08, 2.2, bp.WOOD_GRAY)
    for z in (0.8, 1.3, 1.75):
        m.cyl((0, 0, z), (L, 0, z), 0.004, 0.004, bp.METAL, sides=3)
    for k in range(4):
        x = L * (k + 0.5) / 4
        m.cyl((x, 0, 0), (x, 0, 0.85), 0.04, 0.035, (0.28, 0.2, 0.12), sides=5)
    if tier == 'low':
        m.box(L / 2, 0, 0.75, L, 0.5, 1.1, (0.15, 0.27, 0.08))
    else:
        rng = random.Random(9); n = 60 if tier == 'medium' else 140
        for _ in range(n):
            x = rng.uniform(0.1, L - 0.1); z = rng.uniform(0.75, 1.9)
            up = Vector((rng.uniform(-0.5, 0.5), rng.uniform(-0.6, 0.6), 1))
            s = rng.uniform(0.35, 0.5)
            m.card((x, rng.uniform(-0.15, 0.15), z - s / 2), up, Vector((1, rng.uniform(-0.5, 0.5), 0)), s, s, (1, 1, 1), mat=1)
    return m.finish('vineyard_row', mats), {'length_m': L, 'height_m': 2.0}


# 素材表：名字 → (类别, 生成方法)
def build_tier(tier):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(HQ, tier, 'nature'); tex = os.path.join(base, 'textures')
    vc = bs.vc_material('nature_vc')
    q = bs.Q.get(tier)
    mats = {}
    if tier != 'low':
        for k, fn in FOLIAGE.items():
            rgba = fn(q['tex']); rgba[..., :3] = np.clip(rgba[..., :3] * GAIN[k] ** (1 / 2.2), 0, 1)
            mats[k] = hv.export_material(f'nat_{k}', hv.make_image(k, rgba, tex), True, single_sided=(k == 'wheat'))
        bark = hv.make_image('bark_dead', hv.bark_texture(q['tex'] // 2, (0.03, 0.026, 0.022), (0.14, 0.12, 0.10), 3), tex)
        mats['bark_dead'] = hv.export_material('nat_bark_dead', bark, False)
        bark2 = hv.make_image('bark_fruit', hv.bark_texture(q['tex'] // 2, (0.06, 0.045, 0.035), (0.25, 0.2, 0.15), 4), tex)
        mats['bark_fruit'] = hv.export_material('nat_bark_fruit', bark2, False)
    for key, asset in (('rock_grey', 'Rock062'), ('rock_red', 'Rock029')):
        img = cc0_image(asset, f'{key}_{tier}')
        if tier == 'low':
            mats[key] = vc
        else:
            mats[key] = hv.export_material(f'nat_{key}', img, False)
    items = []
    def keep(name, kind, me, info, mlist):
        for mm in mlist: me.materials.append(mm)
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        ob = bpy.data.objects.new(name, me); c.objects.link(ob)
        d = ob.dimensions
        items.append({'name': name, 'kind': kind, 'height_m': round(d.z, 2), 'width_m': round(max(d.x, d.y), 2),
                      **info, 'triangles': hv.tris(me), 'c': c})

    # 作物、果树：中高档用分树种植被的生成方法，低档用低多边形
    plants = [('crop_soybean', 'crop', 'shrub', dict(H=0.8, D=0.6, stems=6, upright=0.6, cards=3.0, card=0.3, foliage='soy',
                                                     tint=((0.85, 0.95, 0.75), (0.95, 1.0, 0.8)), low=((0.18, 0.3, 0.08), (0.2, 0.25, 0.1)))),
              ('crop_cotton', 'crop', 'shrub', dict(H=1.1, D=0.8, stems=6, upright=0.7, cards=2.5, card=0.35, foliage='cotton',
                                                    tint=((0.9, 0.95, 0.85), (1.0, 1.0, 0.9)), low=((0.3, 0.36, 0.2), (0.2, 0.2, 0.12)))),
              ('crop_wheat', 'crop', 'grass', dict(H=0.9, D=0.6, cards=4, foliage='wheat', low=((0.6, 0.5, 0.25), (0.5, 0.42, 0.2)))),
              ('tree_apple', 'orchard', 'broadleaf', dict(H=4.5, D=5.0, foliage='apple', trunk=0.3, crown_z=0.6, crown_h=0.35,
                                                          spread=0.6, dens=1.3, tint=((0.85, 1.0, 0.75), (1.0, 1.05, 0.8)),
                                                          low=((0.14, 0.26, 0.07), (0.22, 0.17, 0.12)))),
              ('tree_citrus', 'orchard', 'broadleaf', dict(H=4.0, D=4.0, foliage='citrus', trunk=0.18, crown_z=0.55, crown_h=0.42,
                                                           spread=0.4, dens=1.8, tint=((0.7, 0.85, 0.6), (0.85, 0.95, 0.7)),
                                                           low=((0.08, 0.2, 0.06), (0.22, 0.17, 0.12))))]
    for name, kind, arche, sp in plants:
        sp = {**sp, 'name': name, 'arche': arche, 'bark': ((0.06, 0.045, 0.035), (0.25, 0.2, 0.15))}
        seed = sum(map(ord, name))
        if tier == 'low':
            keep(name, kind, bs.lowpoly(sp, sp['H'], sp['D'], seed), {}, [vc])
            continue
        res = bs.ARCHE[arche](sp, sp['H'], sp['D'], seed, q)
        if arche == 'grass':
            keep(name, kind, res, {}, [mats[sp['foliage']]])
        else:
            me, va = res
            bs.card_colors(me, va, sp['tint'], 0.1)
            keep(name, kind, me, {}, [mats['bark_fruit'], mats[sp['foliage']]])
    me, info = vineyard_row(tier, [vc] + ([mats['grape']] if tier != 'low' else []))
    keep('vineyard_row', 'orchard', me, info, [])

    # 石头
    for key, sizes in (('rock_grey', (0.6, 1.4, 3.0)), ('rock_red', (0.6, 1.4, 3.0))):
        for k, s in enumerate(sizes):
            me = rock_mesh(f'{key}_{k + 1:02d}', s, k + (7 if key == 'rock_red' else 0), tier)
            if tier == 'low':
                col = (0.32, 0.31, 0.29) if key == 'rock_grey' else (0.45, 0.25, 0.15)
                attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
                rgb = np.tile(np.array([*col, 1.0], np.float32), len(me.vertices))
                attr.data.foreach_set('color', rgb); me.color_attributes.active_color = attr
            else:
                attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
                attr.data.foreach_set('color', np.ones(len(me.vertices) * 4, np.float32)); me.color_attributes.active_color = attr
            keep(f'{key}_{k + 1:02d}', 'rock', me, {'size_m': s}, [mats[key]])
        # 碎石堆：几块小石头挤在一起
        bm_list = []
        rng = random.Random(3)
        me = bpy.data.meshes.new(f'{key}_pile'); bm = bmesh.new()
        for j in range({'low': 4, 'medium': 8, 'high': 12}[tier]):
            part = rock_mesh('tmp', rng.uniform(0.3, 0.8), j + 20, tier)
            part.transform(Matrix.Translation((rng.uniform(-1, 1), rng.uniform(-1, 1), 0)))
            bm.from_mesh(part); bpy.data.meshes.remove(part)
        bm.to_mesh(me); bm.free()
        attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
        col = (1, 1, 1) if tier != 'low' else ((0.32, 0.31, 0.29) if key == 'rock_grey' else (0.45, 0.25, 0.15))
        attr.data.foreach_set('color', np.tile(np.array([*col, 1.0], np.float32), len(me.vertices))); me.color_attributes.active_color = attr
        keep(f'{key}_pile', 'rock', me, {}, [mats[key]])

    # 大岩体、崖壁、护岸抛石、河卵石
    def vc_white(me, col=(1, 1, 1)):
        attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
        attr.data.foreach_set('color', np.tile(np.array([*col, 1.0], np.float32), len(me.vertices)))
        me.color_attributes.active_color = attr
        return me
    low_col = {'rock_grey': (0.32, 0.31, 0.29), 'rock_red': (0.45, 0.25, 0.15)}
    for key in ('rock_grey', 'rock_red'):
        col = (1, 1, 1) if tier != 'low' else low_col[key]
        for k, (w, h, d) in enumerate(((6.0, 3.5, 4.0), (12.0, 6.0, 7.0))):
            keep(f'{key}_outcrop_{k + 1:02d}', 'rock', vc_white(outcrop_mesh('o', w, h, d, k + 30, tier), col),
                 {'size_m': [w, d, h]}, [mats[key]])
        keep(f'{key}_cliff', 'rock', vc_white(cliff_mesh('c', 20.0, 15.0, 40 if key == 'rock_grey' else 41, tier), col),
             {'size_m': [20.0, 15.0], 'note': '一段崖壁，朝 -y；左右可以并排接着放（起伏是随机的，接缝处用岩块挡）'}, [mats[key]])
    col = (1, 1, 1) if tier != 'low' else low_col['rock_grey']
    keep('riprap_segment', 'rock', vc_white(rock_group('r', {'low': 10, 'medium': 22, 'high': 34}[tier], (0.5, 1.1), 50, tier,
                                                      (3.0, 0.9), flat=0.7, sharp=True), col),
         {'length_m': 6.0, 'note': '护岸抛石一段（沿 x 长 6 m），沿水岸线重复'}, [mats['rock_grey']])
    keep('river_stones', 'rock', vc_white(rock_group('s', {'low': 8, 'medium': 25, 'high': 45}[tier], (0.12, 0.4), 51, tier,
                                                    (1.5, 1.5), flat=0.45), (0.45, 0.43, 0.4) if tier == 'low' else (1.08, 1.05, 1.0)),
         {'note': '河边、溪床的卵石一堆（3 m 见方）'}, [mats['rock_grey']])
    # 蕨类、林下落枝
    sp = {'name': 'fern', 'arche': 'shrub', 'H': 0.9, 'D': 1.2, 'stems': 9, 'upright': 0.3, 'cards': 1.0, 'card': 0.8,
          'foliage': 'fern', 'tint': ((0.85, 0.95, 0.75), (1.0, 1.05, 0.8)), 'low': ((0.12, 0.25, 0.07), (0.15, 0.2, 0.08)),
          'bark': ((0.1, 0.15, 0.05), (0.2, 0.3, 0.1))}
    for k in range(2):
        nm = f'fern_{k + 1:02d}'; sp['name'] = nm
        if tier == 'low':
            keep(nm, 'understory', bs.lowpoly(sp, 0.9, 1.2, 60 + k), {}, [vc])
        else:
            me, va = bs.make_shrub(sp, 0.9 + 0.2 * k, 1.2 + 0.2 * k, 60 + k, q)
            bs.card_colors(me, va, sp['tint'], 0.1)
            keep(nm, 'understory', me, {}, [mats['bark_fruit'], mats['fern']])
    if tier != 'low':
        keep('forest_debris', 'understory', debris_patch(tier).finish('forest_debris', [vc, mats['debris']]),
             {'size_m': 4.0, 'note': '贴地的落枝落叶片，林地里零散铺几块'}, [])

    # 枯木
    for name, me_va in (('snag_01', snag(11, 14.0, q, tier)), ('snag_02', snag(12, 9.0, q, tier)),
                        ('log_01', fallen_log(13, 7.0, 0.25, tier)), ('stump_01', stump(14, 0.3, tier))):
        me, va = me_va
        bs.card_colors(me, va, ((1, 1, 1), (1, 1, 1)), 0)
        if tier == 'low':                                        # 低档：树皮颜色直接写在顶点上
            attr = me.color_attributes['Col']
            c = np.empty(len(me.vertices) * 4, np.float32); attr.data.foreach_get('color', c)
            c = c.reshape(-1, 4); c[:, :3] *= np.array([0.3, 0.27, 0.23]); attr.data.foreach_set('color', c.ravel())
        keep(name, 'deadwood', me, {}, [vc if tier == 'low' else mats['bark_dead']])

    os.makedirs(base, exist_ok=True)
    bpy.data.libraries.write(os.path.join(base, 'nature_export.blend'), {i['c'] for i in items}, fake_user=True)
    for i in items: i.pop('c')
    return items


def grid_preview(blend, out, cols=8, size=320, bg=(0.15, 0.17, 0.08)):
    """一个 .blend 里的每个 Collection 单独取景渲染一张小图，拼成网格（通用，其他素材库也用）"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.collections = list(src.collections)
    obs = []
    for c in dst.collections:
        e = bpy.data.objects.new(c.name, None); e.instance_type = 'COLLECTION'; e.instance_collection = c
        sc.collection.objects.link(e); e.hide_render = True; obs.append((e, c))
    bpy.ops.mesh.primitive_plane_add(size=400)
    g = bpy.context.active_object; gm = bpy.data.materials.new('g'); g.data.materials.append(gm)
    gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*bg, 1)
    w = bpy.data.worlds.new('w'); sc.world = w
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.45, 0.6, 0.85, 1)
    sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 4; so = bpy.data.objects.new('s', sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(-40)); sc.collection.objects.link(so)
    cd = bpy.data.cameras.new('c'); cd.lens = 50; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam)
    sc.camera = cam; sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 16
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.view_settings.view_transform = 'AgX'
    tmp = out + '._tile.png'; tiles = []
    for e, c in obs:
        e.hide_render = False
        lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
        for o in c.all_objects:
            for b in o.bound_box:
                p = o.matrix_world @ Vector(b); lo = Vector(map(min, lo, p)); hi = Vector(map(max, hi, p))
        ctr = (lo + hi) / 2; ext = max(hi - lo) * 1.2 + 0.2
        dist = ext / (2 * math.tan(cd.angle / 2))
        cam.location = ctr + Vector((dist * 0.45, -dist, dist * 0.35))
        cam.rotation_euler = (ctr - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = tmp; bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(tmp); px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px)
        tiles.append(px.reshape(size, size, 4)); bpy.data.images.remove(im)
        e.hide_render = True
    rows = (len(tiles) + cols - 1) // cols
    grid = np.ones((rows * size, cols * size, 4), np.float32)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols); grid[(rows - 1 - r) * size:(rows - r) * size, cc * size:(cc + 1) * size] = t
    im = bpy.data.images.new('grid', grid.shape[1], grid.shape[0]); im.pixels.foreach_set(grid.ravel())
    im.filepath_raw = out; im.file_format = 'PNG'; im.save(); os.remove(tmp)
    print('预览：', out, [c.name for _, c in obs])


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    man = {}
    for tier in TIERS:
        for it in build_tier(tier):
            e = man.setdefault(it['name'], {k: v for k, v in it.items() if k != 'triangles'})
            e.setdefault('triangles', {})[tier] = it['triangles']
    with open(os.path.join(HQ, 'nature_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'assets': list(man.values())}, f, ensure_ascii=False, indent=1)
    print('nature', len(man))
    if '--preview' in argv:
        for t in TIERS:
            grid_preview(os.path.join(HQ, t, 'nature', 'nature_export.blend'), os.path.join(HQ, 'previews', f'nature_{t}.png'))


if __name__ == '__main__':
    main()
