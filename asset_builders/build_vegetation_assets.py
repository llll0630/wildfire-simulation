# ===============================================================
#  植被素材库生成脚本 —— Blender 5.x / 4.x
#  从「山火场景 v3」里提取树和灌木，加上草地测试里的草丛，生成一次并保存成素材库：
#    assets/trees/trees.blend    + trees_manifest.json    树
#    assets/shrubs/shrubs.blend  + shrubs_manifest.json   灌木
#    assets/grass/grass.blend    + grass_manifest.json    草丛
#  每个素材一个 Collection；manifest 记录高度和树冠直径，pipeline 用来算树的数量
#
#  用法（二选一）：
#    1. Blender 里打开 Scripting 面板，粘贴运行
#    2. 命令行：blender --background --factory-startup --python build_vegetation_assets.py
# ===============================================================
import bpy, bmesh, math, random, os, json
from mathutils import Vector, Matrix, noise

SEED = 3                     # 和 v3 场景一致，生成的树长得一样
random.seed(SEED)

# 输出位置：本脚本所在文件夹的上一级 / assets
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = ''
if not HERE or not os.path.isdir(HERE):   # 在 Blender 文本编辑器里运行时 __file__ 不可靠
    HERE = r'D:\cs research\AR\output\asset_builders'
ASSET_DIR = os.path.join(os.path.dirname(HERE), 'assets')
CATEGORY_DIR = {'tree': 'trees', 'shrub': 'shrubs', 'grass': 'grass'}   # 素材种类 → 文件夹名

scene = bpy.context.scene

# ---------------- 清空场景 ----------------
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for c in list(bpy.data.collections):
    bpy.data.collections.remove(c)
for datas in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
    for d in list(datas):
        if d.users == 0:
            datas.remove(d)

# ---------------- 节点构建器（取自 v3） ----------------
class NB:
    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.N = self.mat.node_tree.nodes
        self.L = self.mat.node_tree.links
        self.N.clear()
        self._out = None

    def node(self, t): return self.N.new(t)
    def link(self, a, b): self.L.new(a, b)

    def _set(self, sock, v):
        if isinstance(v, (int, float, tuple)): sock.default_value = v
        else: self.L.new(v, sock)

    def math(self, op, a, b=None, clamp=False):
        n = self.node('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
        self._set(n.inputs[0], a)
        if b is not None: self._set(n.inputs[1], b)
        return n.outputs[0]

    def add(self, a, b, c=False): return self.math('ADD', a, b, c)
    def mul(self, a, b, c=False): return self.math('MULTIPLY', a, b, c)

    def ramp(self, fac, stops):
        r = self.node('ShaderNodeValToRGB'); els = r.color_ramp.elements
        els[0].position = stops[0][0]; els[0].color = (*stops[0][1], 1)
        els[1].position = stops[-1][0]; els[1].color = (*stops[-1][1], 1)
        for p, c in stops[1:-1]:
            e = els.new(p); e.color = (*c, 1)
        self._set(r.inputs['Fac'], fac)
        return r.outputs['Color']

    def noise(self, vec, scale, detail=4.0, rough=0.5, distort=0.0):
        n = self.node('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        n.inputs['Distortion'].default_value = distort
        if vec is not None: self.link(vec, n.inputs['Vector'])
        return n

    def bump(self, h, s=0.5, d=0.1):
        n = self.node('ShaderNodeBump')
        n.inputs['Strength'].default_value = s; n.inputs['Distance'].default_value = d
        self.link(h, n.inputs['Height'])
        return n.outputs['Normal']

    def output(self):
        if self._out is None: self._out = self.node('ShaderNodeOutputMaterial')
        return self._out

    def principled(self, color, rough=0.8, normal=None):
        b = self.node('ShaderNodeBsdfPrincipled')
        b.inputs['Roughness'].default_value = rough
        if isinstance(color, tuple): b.inputs['Base Color'].default_value = (*color, 1)
        else: self.link(color, b.inputs['Base Color'])
        if normal is not None: self.link(normal, b.inputs['Normal'])
        self.link(b.outputs[0], self.output().inputs['Surface'])
        return self.mat

# ---------------- 材质（取自 v3） ----------------
def foliage_mat(name, c1, c2, c3):
    b = NB(name)
    geo = b.node('ShaderNodeNewGeometry'); oi = b.node('ShaderNodeObjectInfo')
    p = geo.outputs['Position']
    n = b.noise(p, 0.6, 3.0).outputs['Fac']
    fac = b.add(b.mul(oi.outputs['Random'], 0.6), b.mul(n, 0.5))
    col = b.ramp(fac, [(0.2, c1), (0.55, c2), (0.95, c3)])
    nb = b.noise(p, 6.0, 6.0, 0.6).outputs['Fac']
    return b.principled(col, 0.85, b.bump(nb, 0.6))

def bark_mat():
    b = NB('Bark'); p = b.node('ShaderNodeNewGeometry').outputs['Position']
    n = b.noise(p, 8.0, 8.0, 0.7).outputs['Fac']
    col = b.ramp(n, [(0.3, (0.05, 0.04, 0.03)), (0.7, (0.14, 0.11, 0.08))])
    return b.principled(col, 0.95, b.bump(n, 0.8))

MAT_BARK = bark_mat()
MAT_OAK = foliage_mat('OakLeaves', (0.03, 0.08, 0.02), (0.06, 0.13, 0.03), (0.11, 0.17, 0.05))
MAT_OLIVE = foliage_mat('OliveLeaves', (0.10, 0.13, 0.08), (0.17, 0.20, 0.13), (0.26, 0.28, 0.19))
MAT_PINE = foliage_mat('PineNeedles', (0.02, 0.05, 0.02), (0.04, 0.08, 0.03), (0.07, 0.11, 0.04))
MAT_BUSH = foliage_mat('HillBush', (0.04, 0.08, 0.025), (0.07, 0.12, 0.04), (0.12, 0.15, 0.06))

# 草叶材质（取自草地测试 grassland_test.py）：根部深、叶尖浅，每丛颜色略有不同
def blade_mat(name, base, mid, tip):
    b = NB(name)
    uv = b.node('ShaderNodeTexCoord').outputs['UV']
    sep = b.node('ShaderNodeSeparateXYZ'); b.link(uv, sep.inputs[0])
    rnd = b.node('ShaderNodeObjectInfo').outputs['Random']
    fac = b.math('MULTIPLY_ADD', rnd, 0.25)
    b.link(sep.outputs['Y'], fac.node.inputs[2])           # Y = 叶片从根(0)到尖(1)
    col = b.ramp(fac, [(0.0, base), (0.55, mid), (1.0, tip)])
    return b.principled(col, 0.8)

MAT_GRASS_GREEN = blade_mat('GrassGreen', (0.06, 0.10, 0.03), (0.18, 0.24, 0.06), (0.36, 0.38, 0.12))
MAT_GRASS_DRY = blade_mat('GrassDry', (0.16, 0.14, 0.05), (0.40, 0.33, 0.13), (0.62, 0.52, 0.26))

# ---------------- 几何工具（取自 v3） ----------------
def smooth(me):
    try: me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))
    except Exception:
        try: me.shade_smooth()
        except Exception: pass

def lumpy_sphere(bm, center, radius, sub=2, amp=0.25, seed=0.0, squash=1.0, mat_index=0):
    res = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=radius,
                                     matrix=Matrix.Translation(center))
    off = Vector((seed * 13.1, seed * 7.7, seed * 3.3))
    for v in res['verts']:
        d = v.co - center
        p = d / radius * 1.4 + off
        d = d * (1 + amp * (noise.noise(p) + 0.5 * noise.noise(p * 2.3)))
        d.z *= squash
        v.co = center + d
    for f in {f for v in res['verts'] for f in v.link_faces}:
        f.material_index = mat_index

def trunk(bm, base, top, r1, r2, mat_index=0):
    d = top - base
    M = Matrix.Translation((base + top) / 2) @ d.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8,
                                radius1=r1, radius2=r2, depth=d.length, matrix=M)
    for f in {f for v in res['verts'] for f in v.link_faces}:
        f.material_index = mat_index

def bm_finish(bm, name, mats, shade_smooth=True):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for m in mats: me.materials.append(m)
    if shade_smooth: smooth(me)
    return me

# ---------------- 植被原型（取自 v3，参数不变） ----------------
def make_tree(name, h, spread, n_blobs, seed, tall=False):
    rnd = random.Random(seed); bm = bmesh.new()
    top = Vector((rnd.uniform(-0.5, 0.5), rnd.uniform(-0.5, 0.5), h * 0.55))
    trunk(bm, Vector((0, 0, -0.5)), top, 0.45 * h / 10, 0.25 * h / 10, 0)
    for k in range(3):
        a = rnd.uniform(0, 2 * math.pi)
        end = top + Vector((math.cos(a) * spread * 0.5, math.sin(a) * spread * 0.5, h * 0.2))
        trunk(bm, top - Vector((0, 0, h * 0.1)), end, 0.18 * h / 10, 0.08 * h / 10, 0)
    zlo, zhi = (0.6, 0.92) if tall else (0.62, 0.85)
    for k in range(n_blobs):
        a = rnd.uniform(0, 2 * math.pi); rr = rnd.uniform(0, spread * 0.6)
        c = Vector((math.cos(a) * rr, math.sin(a) * rr, h * rnd.uniform(zlo, zhi)))
        lumpy_sphere(bm, c, spread * rnd.uniform(0.35, 0.55), 3, 0.3, rnd.random() * 100, 0.75, 1)
    return bm_finish(bm, name, [MAT_BARK, MAT_OAK])

def make_conifer(name, h, seed):
    rnd = random.Random(seed); bm = bmesh.new()
    trunk(bm, Vector((0, 0, -0.5)), Vector((0, 0, h * 0.35)), h * 0.03, h * 0.02, 0)
    tiers = 4
    for k in range(tiers):
        t = k / tiers
        r = h * 0.22 * (1 - t * 0.7)
        z0 = h * (0.15 + t * 0.62)
        depth = h * 0.38 * (1 - t * 0.35)
        res = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=9,
                                    radius1=r, radius2=r * 0.04, depth=depth,
                                    matrix=Matrix.Translation((0, 0, z0 + depth / 2)))
        for v in res['verts']:
            if v.co.z < z0 + depth * 0.6:
                j = rnd.uniform(0.75, 1.15)
                v.co.x *= j; v.co.y *= j
                v.co.z += rnd.uniform(-0.3, 0.3) * r * 0.4
        for f in {f for v in res['verts'] for f in v.link_faces}:
            f.material_index = 1
    return bm_finish(bm, name, [MAT_BARK, MAT_PINE], shade_smooth=False)

def make_shrub(name, seed):
    rnd = random.Random(seed); bm = bmesh.new()
    for k in range(rnd.randint(2, 4)):
        c = Vector((rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(0.6, 1.2)))
        lumpy_sphere(bm, c, rnd.uniform(1.0, 1.6), 2, 0.3, rnd.random() * 100, 0.8, 0)
    return bm_finish(bm, name, [MAT_OLIVE])

def make_bush(name, seed):
    bm = bmesh.new()
    lumpy_sphere(bm, Vector((0, 0, 1.0)), 1.6, 2, 0.35, seed, 0.8, 0)
    return bm_finish(bm, name, [MAT_BUSH])

# 草丛（取自草地测试 grassland_test.py，参数不变）：一圈向外弯的低多边形草叶
def make_clump(name, mat, n_blades, height, spread, width, seed):
    rnd = random.Random(seed)
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new('UVMap')
    SEG = 3
    for b in range(n_blades):
        a = rnd.uniform(0, 2 * math.pi)
        d = Vector((math.cos(a), math.sin(a), 0))           # 叶片弯向的方向
        side = Vector((-d.y, d.x, 0))                        # 叶片宽度方向
        base = d * rnd.uniform(0, spread * 0.25)
        h = height * rnd.uniform(0.6, 1.1)
        lean = spread * rnd.uniform(0.4, 1.0)
        w = width * rnd.uniform(0.8, 1.2)
        rows = []
        for i in range(SEG + 1):
            t = i / SEG
            p = base + d * (lean * t * t) + Vector((0, 0, h * t))
            if i == SEG:
                rows.append([(bm.verts.new(p), 0.5, t)])
            else:
                half = w * (1 - t) / 2
                rows.append([(bm.verts.new(p - side * half), 0.0, t),
                             (bm.verts.new(p + side * half), 1.0, t)])
        for i in range(SEG):
            lo, hi = rows[i], rows[i + 1]
            ring = [lo[0], lo[1], hi[-1], hi[0]] if len(hi) == 2 else [lo[0], lo[1], hi[0]]
            f = bm.faces.new([v for v, _, _ in ring])
            for loop, (_, u, vv) in zip(f.loops, ring):
                loop[uv_layer].uv = (u, vv)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    return me

# 生成顺序和 v3 一致，所以随机出来的树和 v3 里的一模一样
oaks = [make_tree(f'tree_oak_{i + 1:02d}', random.uniform(8.5, 11), random.uniform(4.5, 6),
                  random.randint(7, 10), SEED * 10 + i) for i in range(4)]
pines = [make_conifer(f'tree_pine_{i + 1:02d}', random.uniform(7, 11), SEED * 30 + i) for i in range(4)]
olives = [make_shrub(f'shrub_olive_{i + 1:02d}', SEED * 20 + i) for i in range(3)]
bushes = [make_bush(f'shrub_bush_{i + 1:02d}', i * 17.0) for i in range(2)]
# 草丛：green = 低矮黄绿草，高约 0.5 m；dry = 高枯草，高约 1 m（种子和草地测试一致）
GRASS_SEED = 7
grass_green = [make_clump(f'grass_green_{i + 1:02d}', MAT_GRASS_GREEN, 30, 0.5, 0.5, 0.08, GRASS_SEED * 10 + i)
               for i in range(3)]
grass_dry = [make_clump(f'grass_dry_{i + 1:02d}', MAT_GRASS_DRY, 24, 1.0, 0.6, 0.07, GRASS_SEED * 20 + i)
             for i in range(3)]

# ---------------- 存成素材：每个素材一个 Collection ----------------
root = bpy.data.collections.new('VegetationAssets')
scene.collection.children.link(root)
preview = bpy.data.collections.new('_preview')       # 只用于打开文件时看一眼，pipeline 不用
scene.collection.children.link(preview)

manifest = []
groups = [('tree', 'oak', oaks), ('tree', 'pine', pines),
          ('shrub', 'olive', olives), ('shrub', 'bush', bushes),
          ('grass', 'green', grass_green), ('grass', 'dry', grass_dry)]
for row, (kind, species, meshes) in enumerate(groups):
    for col, me in enumerate(meshes):
        name = me.name
        coll = bpy.data.collections.new(name)
        root.children.link(coll)
        ob = bpy.data.objects.new(name, me)          # 原点 = 树干底部中心，放在 (0,0,0)
        coll.objects.link(ob)
        try: coll.asset_mark()                       # 标记为素材，Asset Browser 里也能看到
        except Exception: pass

        # 预览：把每个素材排成一行一行
        inst = bpy.data.objects.new(f'_preview_{name}', None)
        inst.instance_type = 'COLLECTION'; inst.instance_collection = coll
        inst.location = (col * 16, row * 16, 0)
        preview.objects.link(inst)

        xs = [v.co.x for v in me.vertices]; ys = [v.co.y for v in me.vertices]
        zs = [v.co.z for v in me.vertices]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
        crown_d = (dx + dy) / 2                      # 树冠直径取 x、y 两个方向的平均
        manifest.append({
            'name': name, 'kind': kind, 'species': species,
            'height_m': round(max(zs), 2),
            'crown_diameter_m': round(crown_d, 2),
            'crown_area_m2': round(math.pi * (crown_d / 2) ** 2, 1),
        })

# 按种类分别存成素材库：每个文件只包含这一类素材（Collection 及其模型、材质）
for kind, folder in CATEGORY_DIR.items():
    items = [m for m in manifest if m['kind'] == kind]
    out_dir = os.path.join(ASSET_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)
    blend = os.path.join(out_dir, f'{folder}.blend')
    bpy.data.libraries.write(blend, {bpy.data.collections[m['name']] for m in items}, fake_user=True)
    with open(os.path.join(out_dir, f'{folder}_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'assets': items}, f, ensure_ascii=False, indent=2)
    print('植被素材库已保存：', blend)
    for m in items:
        print(f"  {m['name']:<16} 高 {m['height_m']:>5} m   树冠直径 {m['crown_diameter_m']:>5} m   "
              f"树冠面积 {m['crown_area_m2']:>6} m²")
