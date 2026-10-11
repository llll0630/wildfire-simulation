# ===============================================================
#  地面材质库生成脚本 —— Blender 5.x / 4.x
#  生成地面材质并保存成素材库：assets/ground/ground_materials.blend
#  目前有：mat_grass（取自草地测试 grassland_test.py）
#          mat_litter（取自林地测试 forest_test.py，照豆包参考图制作）
#          mat_farmland（取自农田测试 farmland_test.py）
#          mat_urban（取自城市地面测试 urban_test.py）
#          mat_water（取自水面测试 water_test.py）
#          mat_bare（取自裸地测试 bare_test.py）
#  6 种地面材质已全部完成
#
#  用法（二选一）：
#    1. Blender 里打开 Scripting 面板，粘贴运行
#    2. 命令行：blender --background --factory-startup --python build_ground_materials.py
# ===============================================================
import bpy, os

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = ''
if not HERE or not os.path.isdir(HERE):   # 在 Blender 文本编辑器里运行时 __file__ 不可靠
    HERE = r'D:\cs research\AR\output\asset_builders'
ASSET_DIR = os.path.join(os.path.dirname(HERE), 'assets')
BLEND_PATH = os.path.join(ASSET_DIR, 'ground', 'ground_materials.blend')

scene = bpy.context.scene

# ---------------- 清空场景 ----------------
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
for c in list(bpy.data.collections):
    bpy.data.collections.remove(c)
for datas in (bpy.data.meshes, bpy.data.materials):
    for d in list(datas):
        if d.users == 0:
            datas.remove(d)

def ramp(nt, fac_socket, stops):
    r = nt.nodes.new('ShaderNodeValToRGB'); els = r.color_ramp.elements
    els[0].position = stops[0][0]; els[0].color = (*stops[0][1], 1)
    els[1].position = stops[-1][0]; els[1].color = (*stops[-1][1], 1)
    for p, c in stops[1:-1]:
        e = els.new(p); e.color = (*c, 1)
    nt.links.new(fac_socket, r.inputs['Fac'])
    return r

# ---------------- mat_grass：深绿、黄绿、枯黄成片交错 ----------------
def grass_ground_mat():
    m = bpy.data.materials.new('mat_grass'); m.use_nodes = True
    nt = m.node_tree; N, L = nt.nodes, nt.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial')
    bsdf = N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.95
    pos = N.new('ShaderNodeNewGeometry').outputs['Position']

    big = N.new('ShaderNodeTexNoise')            # 大块：绿色和枯黄色的斑块
    big.inputs['Scale'].default_value = 0.035; big.inputs['Detail'].default_value = 4.0
    L.new(pos, big.inputs['Vector'])
    small = N.new('ShaderNodeTexNoise')          # 小块：细碎的颜色变化
    small.inputs['Scale'].default_value = 0.6; small.inputs['Detail'].default_value = 6.0
    L.new(pos, small.inputs['Vector'])
    mix = N.new('ShaderNodeMath'); mix.operation = 'MULTIPLY_ADD'
    L.new(small.outputs['Fac'], mix.inputs[0])
    mix.inputs[1].default_value = 0.25
    L.new(big.outputs['Fac'], mix.inputs[2])

    col = ramp(nt, mix.outputs[0], [
        (0.35, (0.05, 0.09, 0.02)),   # 深绿
        (0.50, (0.10, 0.14, 0.03)),   # 黄绿
        (0.62, (0.19, 0.18, 0.06)),   # 枯黄
        (0.78, (0.25, 0.21, 0.09)),   # 干草色
    ])
    L.new(col.outputs['Color'], bsdf.inputs['Base Color'])

    bump = N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.4; bump.inputs['Distance'].default_value = 0.05
    L.new(small.outputs['Fac'], bump.inputs['Height'])
    L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

# ---------------- 节点小工具 ----------------
class G:
    def __init__(self, nt): self.nt = nt; self.N = nt.nodes; self.L = nt.links

    def _in(self, sock, v):
        if isinstance(v, (int, float, tuple)): sock.default_value = v
        else: self.L.new(v, sock)

    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.N.new('ShaderNodeMath'); n.operation = op; n.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is not None: self._in(n.inputs[i], v)
        return n.outputs[0]

    def vm(self, op, a, b=None, out=0):
        n = self.N.new('ShaderNodeVectorMath'); n.operation = op
        self._in(n.inputs[0], a)
        if b is not None: self._in(n.inputs[1], b)
        return n.outputs[out]

    def vec(self, x, y, z):
        n = self.N.new('ShaderNodeCombineXYZ')
        for i, v in enumerate((x, y, z)): self._in(n.inputs[i], v)
        return n.outputs[0]

    def map(self, v, a, b, c=0.0, d=1.0):
        n = self.N.new('ShaderNodeMapRange'); n.clamp = True; n.interpolation_type = 'SMOOTHSTEP'
        for i, x in enumerate((v, a, b, c, d)): self._in(n.inputs[i], x)
        return n.outputs[0]

    def sep(self, v):
        n = self.N.new('ShaderNodeSeparateXYZ'); self._in(n.inputs[0], v); return n.outputs

    def sepc(self, c):
        n = self.N.new('ShaderNodeSeparateColor'); self._in(n.inputs[0], c); return n.outputs

    def rgb(self, c):
        n = self.N.new('ShaderNodeRGB'); n.outputs[0].default_value = (*c, 1); return n.outputs[0]

    def mix(self, fac, a, b):
        n = self.N.new('ShaderNodeMix'); n.data_type = 'RGBA'
        self._in(n.inputs['Factor'], fac)
        self.L.new(a, n.inputs['A']); self.L.new(b, n.inputs['B'])
        return n.outputs['Result']

    def darken(self, col, f):
        """颜色乘以一个 0~1 的系数"""
        n = self.N.new('ShaderNodeMix'); n.data_type = 'RGBA'; n.blend_type = 'MULTIPLY'
        n.inputs['Factor'].default_value = 1.0
        self.L.new(col, n.inputs['A']); self.L.new(self.vec(f, f, f), n.inputs['B'])
        return n.outputs['Result']

    def ramp(self, fac, stops):
        r = self.N.new('ShaderNodeValToRGB'); els = r.color_ramp.elements
        els[0].position = stops[0][0]; els[0].color = (*stops[0][1], 1)
        els[1].position = stops[-1][0]; els[1].color = (*stops[-1][1], 1)
        for p, c in stops[1:-1]:
            e = els.new(p); e.color = (*c, 1)
        self._in(r.inputs['Fac'], fac)
        return r.outputs['Color']

    def noise(self, v, scale, detail=3.0):
        n = self.N.new('ShaderNodeTexNoise')
        n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = detail
        self._in(n.inputs['Vector'], v)
        return n.outputs['Fac']

    def cells(self, pos, scale, offset):
        """把地面切成随机小格，每格放一个东西（叶子、松针、细枝）。
        返回：格内局部坐标（随机转了个方向，单位米）、每格 3 个 0~1 随机数"""
        vor = self.N.new('ShaderNodeTexVoronoi'); vor.voronoi_dimensions = '2D'
        vor.inputs['Scale'].default_value = scale
        p = self.vm('ADD', pos, offset)
        self._in(vor.inputs['Vector'], p)
        local = self.vm('SUBTRACT', p, vor.outputs['Position'])
        rnd = self.sepc(vor.outputs['Color'])
        rot = self.N.new('ShaderNodeVectorRotate'); rot.rotation_type = 'Z_AXIS'
        self._in(rot.inputs['Vector'], local)
        self._in(rot.inputs['Angle'], self.m('MULTIPLY', rnd[0], 6.2832))
        return rot.outputs[0], rnd

# ---------------- 地面材质 mat_litter（照参考图）：深色泥土 + 松针 + 细枝 + 落叶 + 苔藓 ----------------
def litter_ground_mat():
    m = bpy.data.materials.new('mat_litter'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); g = G(nt)
    out = g.N.new('ShaderNodeOutputMaterial')
    bsdf = g.N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.9
    pos = g.N.new('ShaderNodeNewGeometry').outputs['Position']
    heights = []                                   # 凹凸高度，最后取最大

    # 1. 泥土：深褐色，带细小颗粒
    grain = g.noise(pos, 80.0, 4.0)
    col = g.ramp(g.m('MULTIPLY_ADD', grain, 0.5, g.noise(pos, 0.5, 3.0)),
                 [(0.55, (0.022, 0.014, 0.009)), (0.85, (0.05, 0.034, 0.022)), (1.0, (0.07, 0.05, 0.035))])

    # 2. 苔藓：零星的小块绿色
    moss_fine = g.noise(pos, 70.0, 2.0)
    moss_mask = g.m('MULTIPLY', g.map(g.noise(pos, 2.5, 3.0), 0.64, 0.7), g.map(moss_fine, 0.35, 0.55))
    col = g.mix(moss_mask, col, g.mix(moss_fine, g.rgb((0.04, 0.07, 0.01)), g.rgb((0.08, 0.14, 0.02))))

    # 3. 松针：铺满地面的细短红褐色线条，三层，方向随机
    for sc, off in ((28.0, (3.1, 7.7, 0.0)), (40.0, (11.3, 2.9, 0.0)), (52.0, (7.9, 5.3, 0.0))):
        loc, rnd = g.cells(pos, sc, off)
        xy = g.sep(loc)
        line = g.m('MINIMUM', g.map(g.m('ABSOLUTE', xy['Y']), 0.0018, 0.0008),
                   g.map(g.m('ABSOLUTE', xy['X']), 0.032, 0.026))
        mask = g.m('MULTIPLY', line, g.m('LESS_THAN', rnd[2], 0.8))
        col = g.mix(mask, col, g.mix(rnd[1], g.rgb((0.045, 0.016, 0.006)), g.rgb((0.11, 0.04, 0.013))))

    # 4. 细枝：稀疏的灰褐色长条，略微弯曲
    loc, rnd = g.cells(pos, 1.3, (5.5, 1.7, 0.0))
    xy = g.sep(loc)
    wob = g.m('ADD', xy['Y'], g.m('MULTIPLY', g.m('SINE', g.m('MULTIPLY', xy['X'], 18.0)), 0.006))
    twig = g.m('MINIMUM', g.map(g.m('ABSOLUTE', wob), 0.011, 0.007),
               g.map(g.m('ABSOLUTE', xy['X']), 0.42, 0.36))
    twig = g.m('MULTIPLY', twig, g.m('LESS_THAN', rnd[2], 0.7))
    col = g.mix(twig, col, g.mix(rnd[1], g.rgb((0.05, 0.035, 0.025)), g.rgb((0.11, 0.08, 0.06))))
    heights.append(g.m('MULTIPLY', twig, 0.6))

    # 5. 落叶：尖椭圆形叶片，边缘偏暗、中间有叶脉，两层叠放
    leaf_cols = [(0.0, (0.06, 0.018, 0.007)), (0.4, (0.12, 0.042, 0.015)),
                 (0.75, (0.17, 0.07, 0.03)), (1.0, (0.22, 0.10, 0.05))]
    for sc, off, dens in ((7.0, (0.0, 0.0, 0.0), 0.7), (9.0, (13.7, 4.1, 0.0), 0.55)):
        loc, rnd = g.cells(pos, sc, off)
        size = g.m('MULTIPLY_ADD', rnd[1], 0.015, 0.035)         # 叶片半长 3.5 – 5 cm
        u = g.vm('DIVIDE', loc, g.vec(size, size, size))          # 归一化：叶尖在 x = ±1 附近
        d1 = g.vm('LENGTH', g.vm('ADD', u, (0.0, 0.55, 0.0)), out=1)
        d2 = g.vm('LENGTH', g.vm('SUBTRACT', u, (0.0, 0.55, 0.0)), out=1)
        inside = g.m('MINIMUM', g.m('SUBTRACT', 1.14, d1), g.m('SUBTRACT', 1.14, d2))
        mask = g.m('MULTIPLY', g.map(inside, 0.0, 0.03), g.m('LESS_THAN', rnd[2], dens))
        edge = g.map(inside, 0.0, 0.2, 0.45, 1.0)                # 边缘偏暗
        vein = g.map(g.m('ABSOLUTE', g.sep(u)['Y']), 0.0, 0.04, 0.75, 1.0)   # 叶脉
        spots = g.map(g.noise(u, 3.0, 2.0), 0.4, 0.7, 0.8, 1.0)  # 斑点
        lc = g.ramp(g.m('MULTIPLY_ADD', rnd[0], 0.5, g.m('MULTIPLY', rnd[1], 0.5)), leaf_cols)
        lc = g.darken(lc, g.m('MULTIPLY', g.m('MULTIPLY', edge, vein), spots))
        col = g.mix(mask, col, lc)
        heights.append(g.m('MULTIPLY', mask, 0.4))

    # 6. 大块明暗变化，避免远看重复
    col = g.darken(col, g.m('MULTIPLY_ADD', g.noise(pos, 0.04, 2.0), 0.6, 0.7))
    g.L.new(col, bsdf.inputs['Base Color'])

    h = g.m('MULTIPLY', grain, 0.3)
    for x in heights: h = g.m('MAXIMUM', h, x)
    bump = g.N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.35; bump.inputs['Distance'].default_value = 0.01
    g.L.new(h, bump.inputs['Height'])
    g.L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    g.L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

# ---------------- 地面材质 mat_farmland：平行的作物垄 + 褐色土沟 ----------------
ROW_SPACING = 0.76          # 垄距（米），美国玉米/大豆常见的 30 英寸行距
PLANT_SPACING = 0.3         # 同一垄里相邻两株的距离（米）

def farmland_ground_mat():
    m = bpy.data.materials.new('mat_farmland'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); g = G(nt)
    out = g.N.new('ShaderNodeOutputMaterial')
    bsdf = g.N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.9
    pos = g.N.new('ShaderNodeNewGeometry').outputs['Position']
    xyz = g.sep(pos)

    # 1. 垄的位置：沿 Y 方向的平行线，用大尺度噪声让垄略微弯曲，不是死板的直线
    bend = g.m('MULTIPLY', g.m('SUBTRACT', g.noise(pos, 0.015, 2.0), 0.5), 0.8)
    t = g.m('DIVIDE', g.m('ADD', xyz['X'], bend), ROW_SPACING)
    d = g.m('ABSOLUTE', g.m('SUBTRACT', g.m('FRACT', t), 0.5))     # 到垄中心的距离：0 ~ 0.5

    # 2. 作物宽度：叶子边缘参差不齐；沿垄方向偶尔稀疏、缺苗
    leafy = g.noise(pos, 7.0, 4.0)
    gaps = g.map(g.noise(pos, 0.6, 3.0), 0.25, 0.4, 0.12, 0.0)    # 缺苗的地方作物变窄
    # 一株一株的植物：沿垄方向每隔 PLANT_SPACING 一株，株与株之间收窄，边缘呈波浪形
    plant = g.m('SINE', g.m('ADD', g.m('MULTIPLY', xyz['Y'], 6.2832 / PLANT_SPACING),
                              g.m('MULTIPLY', g.noise(pos, 1.5, 2.0), 4.0)))
    hw = g.m('MULTIPLY_ADD', leafy, 0.16, 0.2)
    hw = g.m('SUBTRACT', g.m('MULTIPLY_ADD', plant, 0.05, hw), gaps)   # 半宽（以垄距为 1）
    crop = g.map(d, g.m('ADD', hw, 0.02), g.m('SUBTRACT', hw, 0.02))

    # 3. 作物颜色：深浅不一的绿色，成片的地方略微发黄
    cc = g.ramp(g.m('MULTIPLY_ADD', g.noise(pos, 40.0, 3.0), 0.5, g.m('MULTIPLY', g.noise(pos, 12.0, 4.0), 0.5)),
                [(0.35, (0.02, 0.06, 0.008)), (0.5, (0.06, 0.13, 0.018)), (0.65, (0.12, 0.2, 0.04))])
    yellow = g.map(g.noise(pos, 0.04, 3.0), 0.55, 0.7, 0.0, 0.6)
    cc = g.mix(yellow, cc, g.rgb((0.17, 0.16, 0.045)))
    cc = g.darken(cc, g.map(d, 0.0, 0.3, 1.0, 0.7))                  # 垄中间亮、边缘暗

    # 4. 土壤：褐色带颗粒，靠近作物的沟底更暗；整块地有干湿深浅
    grain = g.noise(pos, 60.0, 4.0)
    soil = g.ramp(g.m('MULTIPLY_ADD', grain, 0.4, g.m('MULTIPLY', g.noise(pos, 0.08, 3.0), 0.6)),
                  [(0.35, (0.05, 0.03, 0.016)), (0.6, (0.095, 0.06, 0.032)), (0.85, (0.13, 0.09, 0.052))])
    soil = g.darken(soil, g.map(d, 0.15, 0.5, 0.7, 1.0))

    col = g.mix(crop, soil, cc)
    col = g.darken(col, g.m('MULTIPLY_ADD', g.noise(pos, 0.015, 2.0), 0.4, 0.8))   # 大块明暗
    g.L.new(col, bsdf.inputs['Base Color'])

    # 凹凸：作物凸起 + 叶片起伏 + 土壤颗粒
    h = g.m('ADD', g.m('MULTIPLY', crop, g.m('MULTIPLY_ADD', leafy, 0.5, 0.5)),
            g.m('MULTIPLY', grain, 0.15))
    bump = g.N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.4; bump.inputs['Distance'].default_value = 0.03
    g.L.new(h, bump.inputs['Height'])
    g.L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    g.L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

# ---------------- 地面材质 mat_urban：水泥板 + 沥青 + 接缝裂缝 + 污渍 ----------------
SLAB_SIZE = 3.0             # 水泥板边长（米）
LOT_SIZE = 15.0             # 沥青区块边长（米），是水泥板的整数倍，边缘正好落在板缝上
ASPHALT_SHARE = 0.35        # 大约多少比例的区块铺沥青

def urban_ground_mat():
    m = bpy.data.materials.new('mat_urban'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); g = G(nt)
    out = g.N.new('ShaderNodeOutputMaterial')
    bsdf = g.N.new('ShaderNodeBsdfPrincipled')
    pos = g.N.new('ShaderNodeNewGeometry').outputs['Position']
    xyz = g.sep(pos)
    heights = []

    # 1. 沥青：深灰色，带细小的浅色碎石颗粒
    grit = g.noise(pos, 220.0, 2.0)
    asphalt = g.ramp(g.m('MULTIPLY_ADD', g.noise(pos, 3.0, 4.0), 0.4, g.m('MULTIPLY', grit, 0.6)),
                     [(0.3, (0.018, 0.018, 0.019)), (0.55, (0.035, 0.035, 0.036)), (0.75, (0.08, 0.078, 0.075))])

    # 2. 水泥板：浅灰色方格，每块深浅略有不同，板缝是暗线
    u = g.m('DIVIDE', xyz['X'], SLAB_SIZE); v = g.m('DIVIDE', xyz['Y'], SLAB_SIZE)
    cell = g.vec(g.m('FLOOR', u), g.m('FLOOR', v), 0.0)
    wn = g.N.new('ShaderNodeTexWhiteNoise'); wn.noise_dimensions = '3D'
    g.L.new(cell, wn.inputs['Vector'])
    slab_tone = g.m('MULTIPLY_ADD', wn.outputs['Value'], 0.25, 0.85)          # 每块板 0.85 ~ 1.1
    concrete = g.ramp(g.m('MULTIPLY_ADD', g.noise(pos, 8.0, 5.0), 0.5, g.m('MULTIPLY', g.noise(pos, 60.0, 2.0), 0.5)),
                      [(0.35, (0.13, 0.12, 0.105)), (0.6, (0.19, 0.18, 0.16)), (0.8, (0.24, 0.225, 0.2))])
    concrete = g.darken(concrete, slab_tone)
    ju = g.m('ABSOLUTE', g.m('SUBTRACT', g.m('FRACT', u), 0.5))
    jv = g.m('ABSOLUTE', g.m('SUBTRACT', g.m('FRACT', v), 0.5))
    joint = g.map(g.m('MAXIMUM', ju, jv), 0.492, 0.497)                          # 板缝约 1.5 cm
    concrete = g.darken(concrete, g.m('SUBTRACT', 1.0, g.m('MULTIPLY', joint, 0.7)))

    # 3. 哪里是沥青、哪里是水泥：按方形区块随机决定，边缘是直线（像停车场、路面）
    lot = g.vec(g.m('FLOOR', g.m('DIVIDE', xyz['X'], LOT_SIZE)), g.m('FLOOR', g.m('DIVIDE', xyz['Y'], LOT_SIZE)), 7.0)
    lot_rnd = g.N.new('ShaderNodeTexWhiteNoise'); lot_rnd.noise_dimensions = '3D'
    g.L.new(lot, lot_rnd.inputs['Vector'])
    region = g.m('GREATER_THAN', lot_rnd.outputs['Value'], 1.0 - ASPHALT_SHARE)
    region = g.m('SUBTRACT', 1.0, region)                                  # 1 = 水泥，0 = 沥青
    joint = g.m('MULTIPLY', joint, region)                                 # 沥青上没有板缝
    heights.append(g.m('SUBTRACT', 1.0, joint))
    col = g.mix(region, asphalt, concrete)

    # 4. 裂缝：细而弯曲的暗线，只零星出现
    cn = g.N.new('ShaderNodeTexNoise'); cn.inputs['Scale'].default_value = 1.2
    g.L.new(pos, cn.inputs['Vector'])
    wpos = g.vm('ADD', pos, g.vm('MULTIPLY', g.vm('SUBTRACT', cn.outputs['Color'], (0.5, 0.5, 0.5)), (0.8, 0.8, 0.0)))
    crack_v = g.N.new('ShaderNodeTexVoronoi'); crack_v.feature = 'DISTANCE_TO_EDGE'
    crack_v.inputs['Scale'].default_value = 0.35
    g.L.new(wpos, crack_v.inputs['Vector'])
    crack = g.map(crack_v.outputs['Distance'], 0.012, 0.004)
    crack = g.m('MULTIPLY', crack, g.map(g.noise(pos, 0.15, 2.0), 0.55, 0.62))   # 只保留一部分
    col = g.darken(col, g.m('SUBTRACT', 1.0, g.m('MULTIPLY', crack, 0.75)))
    heights.append(g.m('SUBTRACT', 1.0, crack))

    # 5. 污渍和磨损：深色油渍斑块 + 整体脏旧感
    stain = g.map(g.noise(pos, 0.4, 4.0), 0.6, 0.72, 0.0, 0.45)
    col = g.darken(col, g.m('SUBTRACT', 1.0, stain))
    col = g.darken(col, g.m('MULTIPLY_ADD', g.noise(pos, 0.06, 3.0), 0.35, 0.8))

    # 6. 缝里长出的零星小草
    weed = g.m('MULTIPLY', g.m('MAXIMUM', joint, crack), g.map(g.noise(pos, 1.0, 3.0), 0.6, 0.66))
    col = g.mix(weed, col, g.rgb((0.05, 0.09, 0.02)))
    g.L.new(col, bsdf.inputs['Base Color'])

    # 沥青粗糙、水泥略光滑
    g.L.new(g.m('MULTIPLY_ADD', region, -0.15, 0.9), bsdf.inputs['Roughness'])
    h = g.m('MULTIPLY_ADD', grit, 0.15, 0.0)
    for x in heights: h = g.m('ADD', h, g.m('MULTIPLY', x, 0.5))
    bump = g.N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.3; bump.inputs['Distance'].default_value = 0.01
    g.L.new(h, bump.inputs['Height'])
    g.L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    g.L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

# ---------------- 地面材质 mat_water：平静的湖面，蓝绿色，细小波纹，反光 ----------------
def water_ground_mat():
    m = bpy.data.materials.new('mat_water'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); g = G(nt)
    out = g.N.new('ShaderNodeOutputMaterial')
    bsdf = g.N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['IOR'].default_value = 1.33
    pos = g.N.new('ShaderNodeNewGeometry').outputs['Position']

    # 1. 水色：深处偏蓝、浅处偏绿，大片缓慢变化
    depth = g.noise(pos, 0.02, 3.0)
    col = g.ramp(depth, [(0.35, (0.004, 0.018, 0.03)), (0.55, (0.008, 0.035, 0.045)), (0.75, (0.02, 0.06, 0.055))])
    g.L.new(col, bsdf.inputs['Base Color'])

    # 2. 粗糙度：有风的地方水面皱一些（不那么亮），平静的地方像镜子
    calm = g.noise(pos, 0.05, 2.0)
    g.L.new(g.map(calm, 0.4, 0.65, 0.03, 0.15), bsdf.inputs['Roughness'])

    # 3. 波纹：一层被风吹出的长条波纹 + 一层细碎小波 + 很缓的大涌
    wave = g.N.new('ShaderNodeTexWave'); wave.wave_type = 'BANDS'
    wave.inputs['Scale'].default_value = 0.6; wave.inputs['Distortion'].default_value = 8.0
    wave.inputs['Detail'].default_value = 3.0; wave.inputs['Detail Scale'].default_value = 1.5
    g.L.new(pos, wave.inputs['Vector'])
    small = g.noise(pos, 3.0, 4.0)
    swell = g.noise(pos, 0.15, 2.0)
    h = g.m('ADD', g.m('MULTIPLY', wave.outputs['Fac'], 0.5),
            g.m('ADD', g.m('MULTIPLY', small, 0.35), g.m('MULTIPLY', swell, 0.8)))
    bump = g.N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.45; bump.inputs['Distance'].default_value = 0.03
    g.L.new(h, bump.inputs['Height'])
    g.L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    g.L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

# ---------------- 地面材质 mat_bare：干燥的浅褐色裸土 + 细小裂纹 + 零星碎石 ----------------
def bare_ground_mat():
    m = bpy.data.materials.new('mat_bare'); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear(); g = G(nt)
    out = g.N.new('ShaderNodeOutputMaterial')
    bsdf = g.N.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Roughness'].default_value = 0.95
    pos = g.N.new('ShaderNodeNewGeometry').outputs['Position']
    heights = []

    # 1. 土壤：浅褐色，带细颗粒；大片区域深浅不一
    grain = g.noise(pos, 70.0, 4.0)
    col = g.ramp(g.m('MULTIPLY_ADD', grain, 0.35, g.m('MULTIPLY', g.noise(pos, 0.6, 4.0), 0.65)),
                 [(0.3, (0.13, 0.09, 0.055)), (0.5, (0.2, 0.145, 0.09)), (0.7, (0.27, 0.205, 0.135))])
    col = g.darken(col, g.map(g.noise(pos, 0.05, 3.0), 0.3, 0.6, 0.8, 1.05))

    # 2. 干裂纹：龟裂的细暗线，只在一部分区域出现
    cn = g.N.new('ShaderNodeTexNoise'); cn.inputs['Scale'].default_value = 3.0
    g.L.new(pos, cn.inputs['Vector'])
    wpos = g.vm('ADD', pos, g.vm('MULTIPLY', g.vm('SUBTRACT', cn.outputs['Color'], (0.5, 0.5, 0.5)), (0.12, 0.12, 0.0)))
    cv = g.N.new('ShaderNodeTexVoronoi'); cv.feature = 'DISTANCE_TO_EDGE'
    cv.inputs['Scale'].default_value = 2.5
    g.L.new(wpos, cv.inputs['Vector'])
    crack = g.map(cv.outputs['Distance'], 0.012, 0.004)
    crack = g.m('MULTIPLY', crack, g.map(g.noise(pos, 0.12, 3.0), 0.5, 0.58))
    col = g.darken(col, g.m('SUBTRACT', 1.0, g.m('MULTIPLY', crack, 0.65)))
    heights.append(g.m('SUBTRACT', 0.0, crack))

    # 3. 碎石：小石子较多，大石块零星，灰褐色，中间亮边缘暗
    stone_cols = [(0.0, (0.08, 0.068, 0.055)), (0.5, (0.15, 0.13, 0.105)), (1.0, (0.23, 0.205, 0.17))]
    for sc, off, dens, r0, r1 in ((7.0, (2.3, 8.1, 0.0), 0.35, 0.018, 0.035),
                                  (0.9, (6.7, 1.9, 0.0), 0.18, 0.1, 0.22)):
        loc, rnd = g.cells(pos, sc, off)
        r = g.m('MULTIPLY_ADD', rnd[1], r1 - r0, r0)                  # 石头半径（米）
        xy = g.sep(loc)
        ell = g.vm('LENGTH', g.vec(g.m('DIVIDE', xy['X'], 1.3), xy['Y'], 0.0), out=1)   # 略扁的椭圆
        wob = g.m('MULTIPLY', g.m('SUBTRACT', g.noise(g.vm('DIVIDE', loc, g.vec(r, r, r)), 0.8, 1.0), 0.5), 0.35)  # 边缘不规则
        inside = g.m('SUBTRACT', 1.0, g.m('ADD', g.m('DIVIDE', ell, r), wob))
        mask = g.m('MULTIPLY', g.map(inside, 0.0, 0.04), g.m('LESS_THAN', rnd[2], dens))
        sc_col = g.darken(g.ramp(rnd[0], stone_cols), g.map(inside, 0.0, 0.5, 0.6, 1.1))
        col = g.mix(mask, col, sc_col)
        heights.append(g.m('MULTIPLY', mask, g.map(inside, 0.0, 0.5, 0.3, 1.0)))

    g.L.new(col, bsdf.inputs['Base Color'])
    h = g.m('MULTIPLY', grain, 0.2)
    for x in heights: h = g.m('ADD', h, x)
    bump = g.N.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.4; bump.inputs['Distance'].default_value = 0.02
    g.L.new(h, bump.inputs['Height'])
    g.L.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    g.L.new(bsdf.outputs[0], out.inputs['Surface'])
    return m

materials = [grass_ground_mat(), litter_ground_mat(), farmland_ground_mat(),
             urban_ground_mat(), water_ground_mat(), bare_ground_mat()]

# 材质没有物体使用时，保存文件会被丢掉；加 fake user 让它留在文件里
for m in materials:
    m.use_fake_user = True
    try: m.asset_mark()               # 标记为素材，Asset Browser 里也能看到
    except Exception: pass

os.makedirs(os.path.dirname(BLEND_PATH), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
print('地面材质库已保存：', BLEND_PATH)
for m in materials:
    print('  ', m.name)
