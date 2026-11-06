# ===============================================================
#  建筑三维部件（三档）—— Blender 5.x
#  屋顶形状取决于每栋楼的轮廓，所以这里是一组按参数生成的函数（长、宽、坡度、出檐），
#  素材库里先放一套按 10 m × 8 m 住宅生成的样例；以后 pipeline 按真实轮廓调用同样的函数。
#    屋顶：人字（gable）、四坡（hip）、单坡（shed）、复折（gambrel，谷仓）、带女儿墙的平屋顶
#    附件：烟囱、老虎窗（人字）、前门廊（屋顶 + 立柱 + 台阶 + 栏杆）、阳台、木平台、空调外机、
#          屋顶空调机组、太阳能板阵列、卫星天线
#  贴图用建筑外观素材库（build_building_materials.py 先跑）：中高档用对应档位的贴图，低档用平均颜色。
#  这一步只准备素材；pipeline 怎么用以后再定。
#
#  用法：blender -b --factory-startup --python build_building_parts.py -- [--preview]
#  输出：assets_hq/<档位>/building_parts/building_parts_export.blend、assets_hq/building_parts_manifest.json、
#        assets_hq/previews/building_parts_<档位>.png
# ===============================================================
import bpy, bmesh, math, os, sys, json
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv
import build_props as bp
import build_nature as bn
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'pipeline'))
from roof_shapes import roof_mesh   # 屋顶形状函数在 pipeline 里（pipeline 也用）

ROOT = os.path.dirname(HERE)
HQ = os.path.join(ROOT, 'assets_hq')
BLD = os.path.join(HQ, 'buildings')
TIERS = ('low', 'medium', 'high')
with open(os.path.join(BLD, 'buildings_manifest.json'), encoding='utf-8') as f:
    LIB = json.load(f)


def material(tier, kind, name):
    """建筑素材库里的一种外墙 / 屋顶：中高档 = 贴图（+ 法线）；低档 = 平均颜色。返回 (材质, 一张代表几米)"""
    e = LIB[kind][name]
    key = f'bp_{kind}_{name}'
    if key in bpy.data.materials:
        return bpy.data.materials[key], e['tile_m']
    m = bpy.data.materials.new(key); N, L = m.node_tree.nodes, m.node_tree.links
    bs_ = N['Principled BSDF']; bs_.inputs['Roughness'].default_value = 0.85
    if tier == 'low':
        bs_.inputs['Base Color'].default_value = (*[c ** 2.2 for c in e['avg_rgb']], 1)
    else:
        tx = N.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(os.path.join(BLD, e['files'][tier])); tx.image.pack()
        L.new(tx.outputs['Color'], bs_.inputs['Base Color'])
        npath = os.path.join(BLD, e['files'][tier].replace('.jpg', '_normal.jpg'))
        if os.path.exists(npath):
            ni = N.new('ShaderNodeTexImage'); ni.image = bpy.data.images.load(npath); ni.image.pack()
            ni.image.colorspace_settings.name = 'Non-Color'
            nm = N.new('ShaderNodeNormalMap'); L.new(ni.outputs['Color'], nm.inputs['Color']); L.new(nm.outputs['Normal'], bs_.inputs['Normal'])
    return m, e['tile_m']


def uv_project(me, tiles):
    """按面的朝向方块投影 UV，单位米 / 每种材质一张贴图代表几米（tiles[材质编号]）：贴图不拉伸，大小是真实的"""
    uv = me.uv_layers.new(name='UVMap')
    for p in me.polygons:
        n = p.normal; ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        t = tiles.get(p.material_index, 1.0)
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            if ax == 2:
                uv.data[li].uv = (co[a] / t, co[b] / t)
            else:
                # 斜屋顶：沿坡面方向量长度，瓦片不会被压扁
                slope = math.sqrt(1 - min(1.0, n.z ** 2)) or 1.0
                uv.data[li].uv = (co[a] / t, co[2] / t / max(slope, 0.3))


# ---------------- 屋顶（参数化）：roof_mesh 在 pipeline/roof_shapes.py ----------------
def mesh_from(name, V, F, M, mats, tiles):
    me = bpy.data.meshes.new(name); me.from_pydata(V, [], F); me.update()
    for p, m in zip(me.polygons, M): p.material_index = m
    for m in mats: me.materials.append(m)
    uv_project(me, tiles)
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set('color', np.ones(len(me.vertices) * 4, np.float32)); me.color_attributes.active_color = attr
    return me


# ---------------- 附件（用 build_props 的拼装工具） ----------------
def chimney(m):
    m.box(0, 0, 0, 0.7, 0.7, 2.2, (0.45, 0.2, 0.15)); m.box(0, 0, 2.2, 0.8, 0.8, 0.12, (0.5, 0.48, 0.45))
    return {'height_m': 2.32, 'note': '底部 z = 0 放在屋脊旁；埋进屋顶的部分让屋面挡住'}


def dormer(m, wall_col, roof_col):
    """人字老虎窗：一个小盒子（墙）+ 一个小人字屋顶 + 正面一扇窗；朝 -y，后面插进主屋顶"""
    w, d, h = 1.6, 1.4, 1.3
    m.box(0, d / 2, 0, w, d, h, wall_col)
    rise = w / 2 * 0.9; o = 0.12
    bm = m.bm
    vs = [bm.verts.new(p) for p in ((-w / 2 - o, -o, h), (w / 2 + o, -o, h), (0, -o, h + rise),
                                    (-w / 2 - o, d, h), (w / 2 + o, d, h), (0, d, h + rise))]
    for f in ((vs[0], vs[2], vs[5], vs[3]), (vs[1], vs[4], vs[5], vs[2]), (vs[0], vs[1], vs[2])):
        bm.faces.new(f)
    m._paint(vs, roof_col, 0)
    m.box(0, -0.01, 0.25, 0.8, 0.04, 0.9, (0.12, 0.15, 0.2))           # 窗
    return {'width_m': w, 'depth_m': d, 'height_m': h + rise}


def porch(m, roof_col, post_col=(0.85, 0.85, 0.82), floor_col=(0.4, 0.33, 0.25)):
    W, D, H = 4.0, 2.2, 2.6
    m.box(0, -D / 2, 0, W, D, 0.35, floor_col)
    for k in range(3):
        m.box(0, -D - 0.15 - k * 0.3, 0, 1.6, 0.3, 0.35 - k * 0.12, (0.5, 0.5, 0.48))      # 台阶
    for x in (-W / 2 + 0.15, W / 2 - 0.15):
        m.box(x, -D + 0.15, 0.35, 0.15, 0.15, H - 0.35, post_col)
    m.box(0, -D / 2, H, W + 0.3, D + 0.3, 0.15, roof_col)
    if m.tier != 'low':
        for side in (-1, 1):
            x0 = side * (W / 2 - 0.15); x1 = side * 0.85
            m.box((x0 + x1) / 2, -D + 0.15, 1.2, abs(x1 - x0), 0.06, 0.06, post_col)
            n = 6 if m.tier == 'medium' else 10
            for k in range(n):
                x = x0 + (x1 - x0) * (k + 0.5) / n
                m.box(x, -D + 0.15, 0.35, 0.04, 0.04, 0.85, post_col)
    return {'width_m': W, 'depth_m': D + 0.9, 'height_m': H + 0.15, 'note': '原点 = 门所在墙面的中点，门廊朝 -y'}


def balcony(m):
    W, D = 3.0, 1.2
    m.box(0, -D / 2, 0, W, D, 0.15, (0.6, 0.6, 0.58))
    m.box(0, -D, 0.15, W, 0.05, 0.05 if m.tier == 'low' else 0.05, (0.15, 0.15, 0.16))
    m.box(0, -D, 1.05, W, 0.06, 0.06, (0.15, 0.15, 0.16))
    n = {'low': 2, 'medium': 12, 'high': 24}[m.tier]
    for k in range(n + 1):
        m.box(-W / 2 + W * k / n, -D, 0.15, 0.03, 0.03, 0.9, (0.15, 0.15, 0.16))
    for x in (-W / 2, W / 2):
        m.box(x, -D / 2, 1.05, 0.06, D, 0.06, (0.15, 0.15, 0.16))
    return {'width_m': W, 'depth_m': D, 'height_m': 1.11}


def deck(m):
    W, D, H = 4.0, 3.0, 0.6
    wood = (0.38, 0.27, 0.18)
    n = {'low': 1, 'medium': 14, 'high': 28}[m.tier]
    for k in range(n):
        m.box(-W / 2 + W * (k + 0.5) / n, -D / 2, H - 0.04, W / n * 0.92, D, 0.04, tuple(v * (0.9 + 0.1 * (k % 3)) for v in wood))
    for x in (-W / 2 + 0.1, W / 2 - 0.1):
        for y in (-0.1, -D + 0.1):
            m.box(x, y, 0, 0.12, 0.12, H + 0.95, wood)
    for (x0, y0, x1, y1) in ((-W / 2, -D, W / 2, -D), (-W / 2, -D, -W / 2, 0), (W / 2, -D, W / 2, 0)):
        m.box((x0 + x1) / 2, (y0 + y1) / 2, H + 0.9, max(abs(x1 - x0), 0.06), max(abs(y1 - y0), 0.06), 0.06, wood)
    return {'width_m': W, 'depth_m': D, 'height_m': H + 0.96}


def ac_unit(m):
    m.box(0, 0, 0, 0.9, 0.9, 0.85, (0.72, 0.72, 0.7))
    m.cyl((0, 0, 0.85), (0, 0, 0.87), 0.36, 0.36, (0.15, 0.15, 0.16))
    return {'height_m': 0.87}


def rooftop_hvac(m):
    m.box(0, 0, 0, 2.4, 1.6, 1.3, (0.66, 0.66, 0.64))
    for x in (-0.6, 0.6):
        m.cyl((x, 0, 1.3), (x, 0, 1.34), 0.45, 0.45, (0.2, 0.2, 0.21))
    m.box(1.4, 0, 0.3, 0.4, 0.5, 0.5, (0.55, 0.55, 0.53))
    return {'height_m': 1.34}


def solar_array(m):
    """太阳能板：2 排 × 4 块，每块 1.7 × 1.0 m，平铺（贴在坡屋面上时按屋面坡度转）"""
    for r in range(2):
        for c in range(4):
            x, y = -3.0 + c * 1.75 + 0.875, r * 1.05 + 0.5
            m.box(x, y, 0.05, 1.7, 1.0, 0.04, (0.04, 0.06, 0.12))
            if m.tier == 'high':
                for k in (1, 2):
                    m.box(x - 0.85 + 1.7 * k / 3, y, 0.09, 0.01, 1.0, 0.002, (0.5, 0.5, 0.55))
    return {'width_m': 7.0, 'depth_m': 2.1, 'height_m': 0.09}


def satellite_dish(m):
    m.cyl((0, 0, 0), (0, 0, 0.5), 0.02, 0.02, (0.3, 0.3, 0.3), sides=6)
    m.cyl((0, -0.05, 0.6), (0, -0.2, 0.6), 0.04, 0.38, (0.85, 0.85, 0.84))
    return {'height_m': 0.95}


def build_tier(tier):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(HQ, tier, 'building_parts')
    items = []
    def keep(name, kind, me, info):
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        ob = bpy.data.objects.new(name, me); c.objects.link(ob)
        items.append({'name': name, 'kind': kind, **info, 'triangles': hv.tris(me), 'c': c})
    # 屋顶样例：10 m × 8 m 的住宅，坡度 6/12；谷仓 12 × 9 m；平屋顶 14 × 10 m
    samples = [('roof_gable', 'gable', 10, 8, 0.5, 'shingle_gray', 'siding_white'),
               ('roof_gable_steep', 'gable', 10, 8, 0.75, 'slate', 'brick_red'),
               ('roof_hip', 'hip', 10, 8, 0.5, 'shingle_brown', 'stucco_cream'),
               ('roof_hip_tile', 'hip', 12, 9, 0.4, 'clay_tile', 'stucco_terracotta'),
               ('roof_shed', 'shed', 8, 6, 0.3, 'metal_roof', 'siding_gray'),
               ('roof_gambrel', 'gambrel', 12, 9, 1.0, 'metal_roof', 'siding_gray'),
               ('roof_flat_parapet', 'flat', 14, 10, 0, 'flat_membrane', 'concrete'),
               ('roof_flat_gravel', 'flat', 14, 10, 0, 'flat_gravel', 'brick_tan')]
    for name, kind, L, W, pitch, roof, wall in samples:
        V, F, M = roof_mesh(kind, L, W, pitch)
        rm, rt = material(tier, 'roofs', roof); wm, wt = material(tier, 'walls', wall)
        keep(name, 'roof', mesh_from(name, V, F, M, [rm, wm], {0: rt, 1: wt}),
             {'shape': kind, 'footprint_m': [L, W], 'pitch': pitch, 'roof': roof, 'wall': wall})
    # 附件
    vc = bpy.data.materials.new('parts_vc'); b_ = vc.node_tree.nodes['Principled BSDF']
    n = vc.node_tree.nodes.new('ShaderNodeVertexColor'); n.layer_name = 'Col'
    vc.node_tree.links.new(n.outputs['Color'], b_.inputs['Base Color']); b_.inputs['Roughness'].default_value = 0.75
    roof_c = tuple(LIB['roofs']['shingle_gray']['avg_rgb']); wall_c = tuple(LIB['walls']['siding_white']['avg_rgb'])
    lin = lambda c: tuple(v ** 2.2 for v in c)
    for name, fn, args in (('chimney_brick', chimney, ()), ('dormer_gable', dormer, (lin(wall_c), lin(roof_c))),
                           ('porch_front', porch, (lin(roof_c),)), ('balcony', balcony, ()), ('deck_wood', deck, ()),
                           ('ac_condenser', ac_unit, ()), ('rooftop_hvac', rooftop_hvac, ()),
                           ('solar_array', solar_array, ()), ('satellite_dish', satellite_dish, ())):
        m = bp.Mesh(tier); info = fn(m, *args)
        keep(name, 'fixture', m.finish(name, [vc]), info)
    os.makedirs(base, exist_ok=True)
    bpy.data.libraries.write(os.path.join(base, 'building_parts_export.blend'), {i['c'] for i in items}, fake_user=True)
    for i in items: i.pop('c')
    return items


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    man = {}
    for tier in TIERS:
        for it in build_tier(tier):
            e = man.setdefault(it['name'], {k: v for k, v in it.items() if k != 'triangles'})
            e.setdefault('triangles', {})[tier] = it['triangles']
    with open(os.path.join(HQ, 'building_parts_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'generator': 'pipeline/roof_shapes.py: roof_mesh(kind, L, W, pitch, overhang, parapet)',
                   'assets': list(man.values())}, f, ensure_ascii=False, indent=1)
    print('parts', len(man))
    if '--preview' in argv:
        for t in TIERS:
            bn.grid_preview(os.path.join(HQ, t, 'building_parts', 'building_parts_export.blend'),
                            os.path.join(HQ, 'previews', f'building_parts_{t}.png'), cols=6)


if __name__ == '__main__':
    main()
