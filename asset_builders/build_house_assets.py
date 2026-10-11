# ===============================================================
#  房子素材库生成脚本 —— Blender 5.x / 4.x
#  生成房子并保存成素材库：
#    assets/houses/houses.blend           每栋房子一个 Collection（一个物体），供 pipeline 调用
#    assets/houses/houses_manifest.json   每栋房子的占地尺寸和高度，供 pipeline 摆放时避免重叠
#
#  目前有：house_01 石头小屋（取自「山火场景 v3」，形状和材质都不变）
#          house_02 乡村木板房（照豆包参考图：L 形单层、米色横板墙、深灰沥青瓦、门廊）
#          house_03 郊区两层住宅（照豆包参考图：两层、奶油色横板墙、黑瓦、车库、门廊、红砖烟囱）
#  石墙和屋顶是程序化材质，导出 GLB 前和地面一样需要烘焙；门、窗是纯色，可直接导出
#
#  用法（二选一）：
#    1. Blender 里打开 Scripting 面板，粘贴运行
#    2. 命令行：blender --background --factory-startup --python build_house_assets.py
# ===============================================================
import bpy, bmesh, os, json
from mathutils import Matrix

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = ''
if not HERE or not os.path.isdir(HERE):   # 在 Blender 文本编辑器里运行时 __file__ 不可靠
    HERE = r'D:\cs research\AR\output\asset_builders'
ASSET_DIR = os.path.join(os.path.dirname(HERE), 'assets')
BLEND_PATH = os.path.join(ASSET_DIR, 'houses', 'houses.blend')
MANIFEST_PATH = os.path.join(ASSET_DIR, 'houses', 'houses_manifest.json')

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

# ---------------- 房子材质（取自 v3，参数不变） ----------------
def stone_mat():
    b = NB('Stone'); o = b.node('ShaderNodeTexCoord').outputs['Object']
    vor = b.node('ShaderNodeTexVoronoi'); vor.feature = 'DISTANCE_TO_EDGE'
    vor.inputs['Scale'].default_value = 2.2; b.link(o, vor.inputs['Vector'])
    mortar = b.mul(vor.outputs['Distance'], 12.0, True)
    n = b.noise(o, 4.0, 5.0).outputs['Fac']
    col = b.ramp(b.mul(n, mortar), [(0.0, (0.18, 0.16, 0.13)), (0.35, (0.42, 0.37, 0.30)),
                                    (0.7, (0.58, 0.52, 0.42))])
    return b.principled(col, 0.9, b.bump(mortar, 0.7, 0.05))

def roof_mat():
    b = NB('RoofTiles'); o = b.node('ShaderNodeTexCoord').outputs['Object']
    wave = b.node('ShaderNodeTexWave'); wave.wave_type = 'BANDS'
    try: wave.bands_direction = 'Z'
    except Exception: pass
    wave.inputs['Scale'].default_value = 1.5; b.link(o, wave.inputs['Vector'])
    n = b.noise(o, 3.0, 4.0).outputs['Fac']
    col = b.ramp(n, [(0.3, (0.30, 0.09, 0.04)), (0.7, (0.50, 0.19, 0.09))])
    return b.principled(col, 0.75, b.bump(wave.outputs['Fac'], 0.5))

def plain_mat(name, color, rough=0.8):
    return NB(name).principled(color, rough)

# ---------------- 几何工具：所有部件拼进同一个网格，每个部件用自己的材质编号 ----------------
def add_box(bm, cx, cy, z0, sx, sy, sz, mat_index):
    """底面中心在 (cx, cy, z0) 的长方体"""
    res = bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.transform(bm, verts=res['verts'],
                        matrix=Matrix.Translation((cx, cy, z0 + sz / 2)) @ Matrix.Diagonal((sx, sy, sz, 1)))
    for f in {f for v in res['verts'] for f in v.link_faces}:
        f.material_index = mat_index

def add_poly(bm, verts, faces, mat_index):
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([vs[i] for i in f]).material_index = mat_index

# ---------------- house_01：石头小屋（v3 原样） ----------------
def make_house_01():
    MAT = [stone_mat(),                                           # 0 石墙、烟囱
           roof_mat(),                                            # 1 红瓦屋顶
           plain_mat('DoorWood', (0.08, 0.05, 0.03), 0.7),        # 2 木门
           plain_mat('Window', (0.01, 0.01, 0.01), 0.4)]          # 3 窗
    W, D, H = 8.0, 6.0, 3.4
    bm = bmesh.new()
    add_box(bm, 0, 0, 0, W, D, H, 0)                                     # 墙体
    a, b, c, z0 = W / 2 + 0.7, D / 2 + 0.7, 1.0, H - 0.1                  # 四坡屋顶，屋檐伸出 0.7 m
    add_poly(bm, [(-a, -b, z0), (a, -b, z0), (a, b, z0), (-a, b, z0), (-c, 0, z0 + 2.2), (c, 0, z0 + 2.2)],
             [(0, 1, 5, 4), (1, 2, 5), (2, 3, 4, 5), (3, 0, 4), (3, 2, 1, 0)], 1)
    add_box(bm, 1.5, -D / 2 - 0.05, 0, 1.3, 0.2, 2.1, 2)                  # 门
    add_box(bm, -2.2, -D / 2 - 0.05, 1.6, 0.8, 0.2, 0.8, 3)               # 窗
    add_box(bm, -0.8, 0.3, H + 1.0, 0.6, 0.6, 1.5, 0)                     # 烟囱 1
    add_box(bm, 2.2, -0.5, H + 0.4, 0.6, 0.6, 1.5, 0)                     # 烟囱 2
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('house_01'); bm.to_mesh(me); bm.free()
    for m in MAT: me.materials.append(m)
    return me

# ---------------- house_02 用到的材质（照豆包参考图） ----------------
def obj_coord(b):
    """物体坐标；返回 (水平方向, 高度) 组成的 2D 向量，墙朝 X 或朝 Y 都能用"""
    o = b.node('ShaderNodeTexCoord').outputs['Object']
    sep = b.node('ShaderNodeSeparateXYZ'); b.link(o, sep.inputs[0])
    h = b.math('ADD', sep.outputs['X'], sep.outputs['Y'])
    v = b.node('ShaderNodeCombineXYZ'); b.link(h, v.inputs[0]); b.link(sep.outputs['Z'], v.inputs[1])
    return o, v.outputs[0]

def brick(b, vec, c1, c2, mortar, width, row, mortar_size):
    n = b.node('ShaderNodeTexBrick'); n.offset = 0.5; n.offset_frequency = 2
    b.link(vec, n.inputs['Vector'])
    n.inputs['Color1'].default_value = (*c1, 1); n.inputs['Color2'].default_value = (*c2, 1)
    n.inputs['Mortar'].default_value = (*mortar, 1)
    n.inputs['Scale'].default_value = 1.0
    n.inputs['Mortar Size'].default_value = mortar_size
    n.inputs['Bias'].default_value = 0.0
    n.inputs['Brick Width'].default_value = width
    n.inputs['Row Height'].default_value = row
    return n

def tint(b, col, fac, lo, hi):
    """用 0~1 的 fac 把颜色在 lo~hi 倍之间调暗调亮（木纹、斑驳）"""
    k = b.node('ShaderNodeMapRange'); b.link(fac, k.inputs[0])
    k.inputs[3].default_value = lo; k.inputs[4].default_value = hi
    mix = b.node('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = 1.0
    cc = b.node('ShaderNodeCombineColor')
    for i in range(3): b.link(k.outputs[0], cc.inputs[i])
    b.link(col, mix.inputs['A']); b.link(cc.outputs[0], mix.inputs['B'])
    return mix.outputs['Result']

def siding_mat(name='Siding', c1=(0.66, 0.56, 0.40), c2=(0.74, 0.64, 0.47), mortar=(0.34, 0.28, 0.2)):
    """米色横向木板墙：一条条长木板，板缝略暗，带木纹"""
    b = NB(name); o, v = obj_coord(b)
    br = brick(b, v, c1, c2, mortar, 2.6, 0.2, 0.012)
    grain = b.noise(v, 6.0, 6.0, 0.6, 2.0).outputs['Fac']
    col = tint(b, br.outputs['Color'], grain, 0.88, 1.06)
    return b.principled(col, 0.8, b.bump(b.math('SUBTRACT', 1.0, br.outputs['Fac']), 0.4, 0.02))

def shingle_mat(name='Shingles', c1=(0.045, 0.048, 0.055), c2=(0.085, 0.088, 0.1), flat_along_x=False):
    """深灰色沥青瓦：一片片错缝排列。
    flat_along_x=True 用于沿 X 方向倾斜的平缓屋顶（车库顶、门廊顶）：瓦片的行按 X 方向排"""
    b = NB(name); o, v = obj_coord(b)
    if flat_along_x:
        sep = b.node('ShaderNodeSeparateXYZ'); b.link(o, sep.inputs[0])
        sw = b.node('ShaderNodeCombineXYZ')
        b.link(sep.outputs['Y'], sw.inputs[0]); b.link(sep.outputs['X'], sw.inputs[1])
        v = sw.outputs[0]
    br = brick(b, v, c1, c2, (0.02, 0.02, 0.025), 0.45, 0.25 if flat_along_x else 0.17, 0.012)
    col = tint(b, br.outputs['Color'], b.noise(o, 1.5, 3.0).outputs['Fac'], 0.85, 1.15)
    return b.principled(col, 0.85, b.bump(b.math('SUBTRACT', 1.0, br.outputs['Fac']), 0.6, 0.03))

def porch_shingle_mat():
    """门廊顶的沥青瓦：门廊顶很平，瓦片的行按水平方向（Y）排，而不是按高度排"""
    b = NB('PorchShingles'); o = b.node('ShaderNodeTexCoord').outputs['Object']
    br = brick(b, o, (0.045, 0.048, 0.055), (0.085, 0.088, 0.1), (0.02, 0.02, 0.025), 0.45, 0.25, 0.012)
    col = tint(b, br.outputs['Color'], b.noise(o, 1.5, 3.0).outputs['Fac'], 0.85, 1.15)
    return b.principled(col, 0.85, b.bump(b.math('SUBTRACT', 1.0, br.outputs['Fac']), 0.6, 0.03))

def foundation_mat():
    """灰色毛石地基（和石头小屋同样的做法，换成灰色）"""
    b = NB('FoundationStone'); o = b.node('ShaderNodeTexCoord').outputs['Object']
    vor = b.node('ShaderNodeTexVoronoi'); vor.feature = 'DISTANCE_TO_EDGE'
    vor.inputs['Scale'].default_value = 3.5; b.link(o, vor.inputs['Vector'])
    mortar = b.mul(vor.outputs['Distance'], 12.0, True)
    n = b.noise(o, 4.0, 5.0).outputs['Fac']
    col = b.ramp(b.mul(n, mortar), [(0.0, (0.03, 0.03, 0.03)), (0.35, (0.1, 0.1, 0.1)), (0.7, (0.18, 0.18, 0.17))])
    return b.principled(col, 0.9, b.bump(mortar, 0.7, 0.05))

def wood_mat():
    """棕色木头：门、门廊地板、台阶、窗框、百叶窗"""
    b = NB('Wood'); o, v = obj_coord(b)
    grain = b.noise(v, 4.0, 6.0, 0.6, 3.0).outputs['Fac']
    col = b.ramp(grain, [(0.3, (0.13, 0.06, 0.025)), (0.7, (0.24, 0.12, 0.055))])
    return b.principled(col, 0.7)

# ---------------- house_02：美国乡村单层木板房（照豆包参考图） ----------------
def make_house_02():
    MAT = [siding_mat(),                                          # 0 米色木板墙
           plain_mat('Trim', (0.72, 0.66, 0.52), 0.7),            # 1 白色木条（墙角、门廊柱、栏杆）
           shingle_mat(),                                         # 2 深灰沥青瓦屋顶
           foundation_mat(),                                      # 3 灰色石头地基
           wood_mat(),                                            # 4 棕色木头
           plain_mat('Glass', (0.03, 0.035, 0.045), 0.2),         # 5 窗玻璃
           porch_shingle_mat()]                                   # 6 门廊顶
    SID, TRIM, ROOF, BASE, WOOD, GLASS, PORCH_ROOF = range(7)
    bm = bmesh.new()

    def box(x0, x1, y0, y1, z0, z1, mi):
        add_box(bm, (x0 + x1) / 2, (y0 + y1) / 2, z0, x1 - x0, y1 - y0, z1 - z0, mi)

    def slab(p, t, mi):
        """屋面板：p 是上表面 4 个角，向下加厚 t"""
        top = [bm.verts.new(q) for q in p]
        bot = [bm.verts.new((q[0], q[1], q[2] - t)) for q in p]
        for f in ([0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]):
            vs = (top + bot)
            bm.faces.new([vs[i] for i in f]).material_index = mi

    def on_wall(axis, pos, d0, d1, c0, c1, z0, z1, mi):
        """贴在墙上的东西：axis='x' 是朝 -X 的墙（墙面 x=pos），axis='y' 是朝 -Y 的墙；d0~d1 是凸出墙面的距离"""
        if axis == 'x': box(pos - d1, pos - d0, c0, c1, z0, z1, mi)
        else: box(c0, c1, pos - d1, pos - d0, z0, z1, mi)

    def window(axis, pos, c, z0, z1, w, shutters=False):
        on_wall(axis, pos, 0, 0.06, c - w / 2 - 0.1, c + w / 2 + 0.1, z0 - 0.1, z1 + 0.1, WOOD)   # 窗框
        on_wall(axis, pos, 0, 0.08, c - w / 2 - 0.18, c + w / 2 + 0.18, z0 - 0.18, z0 - 0.1, WOOD)  # 窗台
        on_wall(axis, pos, 0.06, 0.07, c - w / 2, c + w / 2, z0, z1, GLASS)                        # 玻璃
        zm = (z0 + z1) / 2
        on_wall(axis, pos, 0.07, 0.09, c - 0.025, c + 0.025, z0, z1, TRIM)                         # 十字窗棂
        on_wall(axis, pos, 0.07, 0.09, c - w / 2, c + w / 2, zm - 0.025, zm + 0.025, TRIM)
        if shutters:
            for s in (-1, 1):
                cc = c + s * (w / 2 + 0.38)
                on_wall(axis, pos, 0, 0.05, cc - 0.24, cc + 0.24, z0 - 0.1, z1 + 0.1, WOOD)

    H, BASE_H, PITCH = 3.0, 0.45, 2.7 / 3.25          # 墙高、地基高、屋顶坡度（高/水平）
    O, T = 0.45, 0.12                                  # 屋檐伸出、屋面厚度

    # 左翼：x 0~12，y 3.5~10，屋脊沿 X，山墙朝 -X
    box(-0.05, 12.05, 3.45, 10.05, 0, BASE_H, BASE)
    box(0, 12, 3.5, 10, BASE_H, H, SID)
    ym, rise = 6.75, 3.25 * PITCH
    for x in (0, 12):
        add_poly(bm, [(x, 3.5, H), (x, 10, H), (x, ym, H + rise)], [(0, 1, 2)], SID)
    slab([(-O, 3.5 - O, H - O * PITCH + T), (12 + O, 3.5 - O, H - O * PITCH + T),
          (12 + O, ym, H + rise + T), (-O, ym, H + rise + T)], T, ROOF)
    slab([(-O, ym, H + rise + T), (12 + O, ym, H + rise + T),
          (12 + O, 10 + O, H - O * PITCH + T), (-O, 10 + O, H - O * PITCH + T)], T, ROOF)

    # 右翼：x 5~11，y 0~6，屋脊沿 Y，山墙朝 -Y（后端伸进左翼屋顶里）
    box(4.95, 11.05, -0.05, 6, 0, BASE_H, BASE)
    box(5, 11, 0, 6, BASE_H, H, SID)
    xm, rise2 = 8.0, 3.0 * PITCH
    add_poly(bm, [(5, 0, H), (11, 0, H), (xm, 0, H + rise2)], [(0, 1, 2)], SID)
    slab([(5 - O, -O, H - O * PITCH + T), (xm, -O, H + rise2 + T),
          (xm, ym, H + rise2 + T), (5 - O, ym, H - O * PITCH + T)], T, ROOF)
    slab([(xm, -O, H + rise2 + T), (11 + O, -O, H - O * PITCH + T),
          (11 + O, ym, H - O * PITCH + T), (xm, ym, H + rise2 + T)], T, ROOF)

    # 墙角白色木条
    for x, y in ((0, 3.5), (0, 10), (12, 3.5), (12, 10), (5, 0), (11, 0)):
        box(x - 0.08, x + 0.08, y - 0.08, y + 0.08, BASE_H, H, TRIM)

    # 门廊：在两翼夹角里（x 0~5，y 0~3.5）
    box(0.3, 5, 0.3, 3.5, 0, 0.5, WOOD)                                  # 地板
    box(0.7, 2.3, 0.0, 0.3, 0, 0.33, WOOD)                               # 台阶
    box(0.7, 2.3, -0.3, 0.0, 0, 0.17, WOOD)
    box(0.36, 0.54, 0.36, 0.54, 0.5, 2.75, TRIM)                         # 门廊柱
    box(0.3, 5, 0.35, 0.55, 2.55, 2.75, TRIM)                            # 横梁
    box(0.35, 0.55, 0.3, 3.5, 2.55, 2.75, TRIM)
    slab([(-0.2, -0.15, 2.8), (5, -0.15, 2.8), (5, 3.1, 3.08), (-0.2, 3.1, 3.08)], 0.1, PORCH_ROOF)     # 门廊顶（搭在左翼屋檐上面）
    box(2.5, 4.95, 0.37, 0.47, 1.3, 1.38, TRIM)                          # 栏杆扶手
    box(2.5, 4.95, 0.37, 0.47, 0.62, 0.68, TRIM)
    x = 2.6
    while x < 4.95:
        box(x - 0.03, x + 0.03, 0.39, 0.45, 0.5, 1.3, TRIM); x += 0.3
    on_wall('y', 3.5, 0, 0.05, 1.35, 2.65, 0.5, 2.75, TRIM)              # 门框
    on_wall('y', 3.5, 0.05, 0.09, 1.5, 2.5, 0.5, 2.6, WOOD)              # 门
    window('x', 5, 1.9, 1.2, 2.3, 0.9)                                   # 门廊里的窗

    # 窗户
    window('x', 0, 6.75, 1.0, 2.4, 1.2, shutters=True)                   # 左山墙：带百叶窗
    on_wall('x', 0, 0, 0.06, 6.45, 7.05, 4.0, 4.6, WOOD)                 # 左山墙顶的通风口
    window('y', 0, 8.0, 1.0, 2.4, 1.1)                                   # 右山墙

    bmesh.ops.translate(bm, verts=bm.verts, vec=(-6.0, -5.0, 0.0))      # 原点移到占地中心
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('house_02'); bm.to_mesh(me); bm.free()
    for m in MAT: me.materials.append(m)
    return me

# ---------------- house_03 用到的材质（照豆包参考图） ----------------
def chimney_brick_mat():
    """红砖烟囱"""
    b = NB('ChimneyBrick'); o, v = obj_coord(b)
    br = brick(b, v, (0.28, 0.06, 0.03), (0.36, 0.1, 0.05), (0.45, 0.42, 0.38), 0.25, 0.08, 0.008)
    return b.principled(br.outputs['Color'], 0.85, b.bump(b.math('SUBTRACT', 1.0, br.outputs['Fac']), 0.5, 0.02))

def garage_door_mat():
    """白色车库门：横竖分格的嵌板"""
    b = NB('GarageDoor'); o = b.node('ShaderNodeTexCoord').outputs['Object']
    sep = b.node('ShaderNodeSeparateXYZ'); b.link(o, sep.inputs[0])
    v = b.node('ShaderNodeCombineXYZ'); b.link(sep.outputs['Y'], v.inputs[0]); b.link(sep.outputs['Z'], v.inputs[1])
    br = brick(b, v.outputs[0], (0.78, 0.77, 0.73), (0.8, 0.79, 0.75), (0.5, 0.5, 0.48), 0.95, 0.55, 0.03)
    br.offset = 0.0
    return b.principled(br.outputs['Color'], 0.5, b.bump(b.math('SUBTRACT', 1.0, br.outputs['Fac']), 0.6, 0.02))

# ---------------- house_03：美国郊区两层住宅（照豆包参考图） ----------------
def make_house_03():
    MAT = [siding_mat('Siding03', (0.72, 0.66, 0.47), (0.78, 0.72, 0.53), (0.4, 0.36, 0.25)),   # 0 奶油色横板墙
           plain_mat('Trim03', (0.8, 0.79, 0.74), 0.6),                                      # 1 白色木条、窗框、柱子
           shingle_mat('Shingles03', (0.022, 0.022, 0.026), (0.045, 0.045, 0.05)),           # 2 黑色沥青瓦（陡屋顶）
           foundation_mat(),                                                                # 3 灰色石头地基
           wood_mat(),                                                                      # 4 木头（门廊地板、花箱）
           plain_mat('Glass03', (0.1, 0.12, 0.15), 0.15),                                   # 5 窗玻璃
           shingle_mat('FlatShingles03', (0.022, 0.022, 0.026), (0.045, 0.045, 0.05), True),  # 6 平缓屋顶（车库、门廊）
           plain_mat('Shutter03', (0.018, 0.018, 0.02), 0.6),                               # 7 黑色百叶窗
           plain_mat('FrontDoor03', (0.28, 0.06, 0.025), 0.5),                              # 8 红棕色大门
           chimney_brick_mat(),                                                             # 9 红砖烟囱
           plain_mat('Concrete03', (0.42, 0.41, 0.39), 0.9),                                # 10 水泥台阶、烟囱帽
           garage_door_mat(),                                                               # 11 车库门
           plain_mat('Plant03', (0.05, 0.14, 0.03), 0.8)]                                   # 12 花箱里的植物
    SID, TRIM, ROOF, BASE, WOOD, GLASS, FLAT, SHUT, DOOR, BRICK, CONC, GDOOR, PLANT = range(13)
    bm = bmesh.new()

    def box(x0, x1, y0, y1, z0, z1, mi):
        add_box(bm, (x0 + x1) / 2, (y0 + y1) / 2, z0, x1 - x0, y1 - y0, z1 - z0, mi)

    def slab(p, t, mi):
        """屋面板：p 是上表面 4 个角，向下加厚 t"""
        vs = [bm.verts.new(q) for q in p] + [bm.verts.new((q[0], q[1], q[2] - t)) for q in p]
        for f in ([0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]):
            bm.faces.new([vs[i] for i in f]).material_index = mi

    def tri(p, mi):
        add_poly(bm, p, [(0, 1, 2)], mi)

    def on_wall(axis, pos, d0, d1, c0, c1, z0, z1, mi):
        """贴在墙上的东西：axis='x' 是朝 -X 的墙（墙面 x=pos），axis='y' 是朝 -Y 的墙；d0~d1 是凸出墙面的距离"""
        if axis == 'x': box(pos - d1, pos - d0, c0, c1, z0, z1, mi)
        else: box(c0, c1, pos - d1, pos - d0, z0, z1, mi)

    def window(axis, pos, c, z0, z1, w, shutters=False):
        on_wall(axis, pos, 0, 0.06, c - w / 2 - 0.1, c + w / 2 + 0.1, z0 - 0.1, z1 + 0.1, TRIM)   # 白色窗框
        on_wall(axis, pos, 0, 0.09, c - w / 2 - 0.18, c + w / 2 + 0.18, z0 - 0.18, z0 - 0.1, TRIM)  # 窗台
        on_wall(axis, pos, 0.06, 0.07, c - w / 2, c + w / 2, z0, z1, GLASS)                        # 玻璃
        zm = (z0 + z1) / 2
        on_wall(axis, pos, 0.07, 0.09, c - 0.025, c + 0.025, z0, z1, TRIM)                         # 十字窗棂
        on_wall(axis, pos, 0.07, 0.09, c - w / 2, c + w / 2, zm - 0.025, zm + 0.025, TRIM)
        if shutters:
            for s in (-1, 1):
                cc = c + s * (w / 2 + 0.36)
                on_wall(axis, pos, 0, 0.05, cc - 0.22, cc + 0.22, z0 - 0.1, z1 + 0.1, SHUT)

    def vent(axis, pos, c, z0):
        on_wall(axis, pos, 0, 0.06, c - 0.3, c + 0.3, z0, z0 + 0.55, TRIM)

    H1, H2, BASE_H, P = 3.0, 6.0, 0.35, 0.75           # 一层高、两层高、地基高、屋顶坡度（高/水平）
    O, T = 0.4, 0.12                                   # 屋檐伸出、屋面厚度

    # 主楼 B（两层）：x 1.5~8.5，y 1~10，屋脊沿 Y，两端山墙朝 -Y / +Y
    box(1.45, 8.55, 0.95, 10.05, 0, BASE_H, BASE)
    box(1.5, 8.5, 1, 10, BASE_H, H2, SID)
    xm, r = 5.0, 3.5 * P
    tri([(1.5, 1, H2), (8.5, 1, H2), (xm, 1, H2 + r)], SID)
    tri([(8.5, 10, H2), (1.5, 10, H2), (xm, 10, H2 + r)], SID)
    slab([(1.5 - O, 1 - O, H2 - O * P + T), (xm, 1 - O, H2 + r + T), (xm, 10 + O, H2 + r + T),
          (1.5 - O, 10 + O, H2 - O * P + T)], T, ROOF)
    slab([(xm, 1 - O, H2 + r + T), (8.5 + O, 1 - O, H2 - O * P + T), (8.5 + O, 10 + O, H2 - O * P + T),
          (xm, 10 + O, H2 + r + T)], T, ROOF)

    # 左前方凸出的两层 A：x 0~5，y 5~10，屋脊沿 X，山墙朝 -X（屋顶伸进主楼屋顶里）
    box(-0.05, 5, 4.95, 10.05, 0, BASE_H, BASE)
    box(0, 5, 5, 10, BASE_H, H2, SID)
    ym, ra = 7.5, 2.5 * P
    tri([(0, 10, H2), (0, 5, H2), (0, ym, H2 + ra)], SID)
    slab([(-O, 5 - O, H2 - O * P + T), (4.5, 5 - O, H2 - O * P + T), (4.5, ym, H2 + ra + T), (-O, ym, H2 + ra + T)], T, ROOF)
    slab([(-O, ym, H2 + ra + T), (4.5, ym, H2 + ra + T), (4.5, 9.9, H2 + 0.075 + T), (-O, 9.9, H2 + 0.075 + T)], T, ROOF)   # 后侧不出檐，免得从主楼山墙后面露出来

    # 右侧单层侧翼 C：x 2.5~7.5，y -2~1，屋脊沿 Y，山墙朝 -Y
    box(2.45, 7.55, -2.05, 1, 0, BASE_H, BASE)
    box(2.5, 7.5, -2, 1, BASE_H, H1, SID)
    rc = 2.5 * P
    tri([(2.5, -2, H1), (7.5, -2, H1), (xm, -2, H1 + rc)], SID)
    slab([(2.5 - O, -2 - O, H1 - O * P + T), (xm, -2 - O, H1 + rc + T), (xm, 1, H1 + rc + T), (2.5 - O, 1, H1 - O * P + T)], T, ROOF)
    slab([(xm, -2 - O, H1 + rc + T), (7.5 + O, -2 - O, H1 - O * P + T), (7.5 + O, 1, H1 - O * P + T), (xm, 1, H1 + rc + T)], T, ROOF)

    # 车库 G（单层）：x -3~0，y 5~10，车库门朝 -X；平缓的单坡屋顶
    box(-3.05, 0, 4.95, 10.05, 0, 0.15, CONC)
    box(-3, 0, 5, 10, 0.15, H1, SID)
    slab([(-3 - O, 5 - O, H1 - 0.05), (0, 5 - O, 3.65), (0, 10 + O, 3.65), (-3 - O, 10 + O, H1 - 0.05)], 0.1, FLAT)
    on_wall('x', -3, 0, 0.05, 5.45, 9.55, 0.15, 2.65, TRIM)              # 车库门框
    on_wall('x', -3, 0.05, 0.08, 5.6, 9.4, 0.15, 2.5, GDOOR)             # 车库门

    # 墙角白色木条
    for x, y, z1 in ((0, 5, H2), (0, 10, H2), (8.5, 1, H2), (8.5, 10, H2), (1.5, 1, H2),
                     (2.5, -2, H1), (7.5, -2, H1), (-3, 5, H1), (-3, 10, H1)):
        box(x - 0.08, x + 0.08, y - 0.08, y + 0.08, BASE_H, z1, TRIM)

    # 门廊：在主楼和侧翼前面（x -0.8~2.5，y -1~4.2）
    box(-0.8, 2.5, -1.0, 4.2, 0, 0.45, WOOD)                             # 门廊地板
    box(-1.25, -0.8, 2.5, 4.0, 0, 0.3, CONC)                             # 水泥台阶
    box(-1.7, -1.25, 2.5, 4.0, 0, 0.15, CONC)
    for y in (2.3, -0.8):                                                # 白色柱子
        box(-0.7, -0.5, y - 0.1, y + 0.1, 0.45, 2.95, TRIM)
    box(-0.66, -0.54, -0.7, 2.2, 1.3, 1.38, TRIM)                        # 栏杆扶手
    box(-0.66, -0.54, -0.7, 2.2, 0.6, 0.66, TRIM)
    y = -0.6
    while y < 2.2:
        box(-0.63, -0.57, y - 0.03, y + 0.03, 0.45, 1.3, TRIM); y += 0.28
    box(-0.8, -0.4, -1.0, 4.2, 2.75, 2.95, TRIM)                         # 门廊横梁
    slab([(-1.1, -1.3, 3.0), (2.5, -1.3, 3.55), (2.5, 4.5, 3.55), (-1.1, 4.5, 3.0)], 0.1, FLAT)   # 门廊顶
    # 大门上方的小山墙雨棚
    tri([(-0.95, 2.35, 3.0), (-0.95, 4.05, 3.0), (-0.95, 3.2, 3.62)], TRIM)
    slab([(-1.05, 2.25, 2.98 + 0.08), (1.5, 2.25, 2.98 + 0.08), (1.5, 3.2, 3.68), (-1.05, 3.2, 3.68)], 0.08, ROOF)
    slab([(-1.05, 3.2, 3.68), (1.5, 3.2, 3.68), (1.5, 4.15, 2.98 + 0.08), (-1.05, 4.15, 2.98 + 0.08)], 0.08, ROOF)
    on_wall('x', 1.5, 0, 0.05, 2.6, 3.8, 0.45, 2.75, TRIM)               # 门框
    on_wall('x', 1.5, 0.05, 0.09, 2.75, 3.65, 0.45, 2.6, DOOR)           # 红棕色大门
    on_wall('x', 1.5, 0.05, 0.08, 3.9, 4.1, 0.6, 2.5, GLASS)             # 门边的窄玻璃

    # 窗户
    window('x', 0, 7.5, 4.0, 5.3, 1.2, shutters=True)                    # A 二楼正面（带百叶窗）
    on_wall('x', 0, 0, 0.35, 6.9, 8.1, 3.7, 3.92, WOOD)                  # 窗下花箱
    on_wall('x', 0, 0.05, 0.3, 6.95, 8.05, 3.92, 4.05, PLANT)
    vent('x', 0, 7.5, 6.65)                                              # A 山墙通风口
    window('x', 1.5, 2.2, 4.0, 5.2, 0.9)                                 # 主楼二楼正面两扇小窗
    window('x', 1.5, 4.1, 4.0, 5.2, 0.9)
    window('y', 1, 5.0, 4.0, 5.3, 1.2, shutters=True)                    # 主楼右山墙二楼（带百叶窗）
    vent('y', 1, 5.0, 6.95)                                              # 主楼右山墙通风口
    window('y', -2, 5.0, 0.9, 2.3, 1.1, shutters=True)                   # 侧翼山墙（带百叶窗）
    window('x', 2.5, -0.4, 0.9, 2.4, 1.2)                                # 门廊里的窗

    # 红砖烟囱（主楼右后方，高出屋脊）
    box(7.3, 8.1, 3.0, 3.8, 5.5, 9.4, BRICK)
    box(7.22, 8.18, 2.92, 3.88, 9.4, 9.55, CONC)
    box(7.6, 7.8, 3.3, 3.5, 9.55, 9.8, SHUT)

    bmesh.ops.translate(bm, verts=bm.verts, vec=(-2.75, -4.0, 0.0))     # 原点移到占地中心
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new('house_03'); bm.to_mesh(me); bm.free()
    for m in MAT: me.materials.append(m)
    return me

houses = [('house_01', '石头小屋', make_house_01()),
          ('house_02', '乡村木板房', make_house_02()),
          ('house_03', '郊区两层住宅', make_house_03())]

# ---------------- 存成素材：每栋房子一个 Collection ----------------
root = bpy.data.collections.new('HouseAssets')
scene.collection.children.link(root)
preview = bpy.data.collections.new('_preview')       # 只用于打开文件时看一眼，pipeline 不用
scene.collection.children.link(preview)

manifest = []
for i, (name, label, me) in enumerate(houses):
    coll = bpy.data.collections.new(name)
    root.children.link(coll)
    ob = bpy.data.objects.new(name, me)              # 原点 = 房子底面中心，放在 (0,0,0)
    coll.objects.link(ob)
    try: coll.asset_mark()
    except Exception: pass

    inst = bpy.data.objects.new(f'_preview_{name}', None)
    inst.instance_type = 'COLLECTION'; inst.instance_collection = coll
    inst.location = (i * 20, 0, 0)
    preview.objects.link(inst)

    xs = [v.co.x for v in me.vertices]; ys = [v.co.y for v in me.vertices]; zs = [v.co.z for v in me.vertices]
    manifest.append({
        'name': name, 'label': label,
        'footprint_x_m': round(max(xs) - min(xs), 2),    # 占地（含屋檐）
        'footprint_y_m': round(max(ys) - min(ys), 2),
        'height_m': round(max(zs), 2),
    })

lc = bpy.context.view_layer.layer_collection.children['HouseAssets']
lc.exclude = True

os.makedirs(os.path.dirname(BLEND_PATH), exist_ok=True)
with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
    json.dump({'assets': manifest}, f, ensure_ascii=False, indent=2)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)

print('房子素材库已保存：', BLEND_PATH)
for m in manifest:
    print(f"  {m['name']}  {m['label']}  占地 {m['footprint_x_m']} × {m['footprint_y_m']} m  高 {m['height_m']} m")
