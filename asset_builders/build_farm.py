# ===============================================================
#  农场和乡村建筑（三档）—— Blender 5.x
#  美国中西部农田（Iowa、Flint Hills……）里最常见的东西：
#    红色复折屋顶谷仓、金属机库、谷仓筒仓（波纹钢，大小两种）、混凝土青贮塔、圆草捆 / 方草捆堆、
#    中心支轴喷灌机（一跨）、水塔、抽水风车
#  尺寸按美国常见的真实尺寸（米）。外墙、屋顶贴图用建筑外观素材库（中高档贴图、低档平均颜色）。
#  这一步只准备素材；pipeline 怎么用以后再定。
#
#  用法：blender -b --factory-startup --python build_farm.py -- [--preview]
#  输出：assets_hq/<档位>/farm/farm_export.blend、assets_hq/farm_manifest.json、assets_hq/previews/farm_<档位>.png
# ===============================================================
import bpy, bmesh, math, random, os, sys, json
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv
import build_props as bp
import build_nature as bn
import build_building_parts as bbp

ROOT = os.path.dirname(HERE)
HQ = os.path.join(ROOT, 'assets_hq')
TIERS = ('low', 'medium', 'high')


def walls_box(L, W, H):
    """四面墙（不含顶）的顶点和面：长 L（沿 x）、宽 W、高 H，底在 z = 0"""
    hx, hy = L / 2, W / 2
    V = [(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0), (-hx, -hy, H), (hx, -hy, H), (hx, hy, H), (-hx, hy, H)]
    F = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return V, F


def building(name, tier, L, W, H, roof_kind, pitch, wall, roof, door=None):
    """一栋简单的建筑：墙（外墙材质）+ 屋顶（bbp.roof_mesh）+ 正面一扇大门（深色）"""
    wm, wt = bbp.material(tier, 'walls', wall); rm, rt = bbp.material(tier, 'roofs', roof)
    V, F = walls_box(L, W, H); M = [1] * len(F)
    rv, rf, rmi = bbp.roof_mesh(roof_kind, L, W, pitch, overhang=0.5)
    o = len(V); V += [(x, y, z + H) for x, y, z in rv]; F += [tuple(i + o for i in f) for f in rf]
    M += [0 if m == 0 else 1 for m in rmi]
    if door:
        dw, dh = door
        o = len(V); V += [(-dw / 2, -W / 2 - 0.02, 0), (dw / 2, -W / 2 - 0.02, 0), (dw / 2, -W / 2 - 0.02, dh), (-dw / 2, -W / 2 - 0.02, dh)]
        F.append((o, o + 1, o + 2, o + 3)); M.append(2)
    dark = bpy.data.materials.get('farm_door') or bpy.data.materials.new('farm_door')
    dark.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.05, 0.05, 0.05, 1)
    return bbp.mesh_from(name, V, F, M, [rm, wm, dark], {0: rt, 1: wt, 2: 1.0})


def cylinder_building(name, tier, D, H, roof_h, wall, roof, dome=False):
    """圆柱形（谷仓筒仓、青贮塔）：侧面外墙材质，顶上锥顶或圆顶"""
    seg = {'low': 10, 'medium': 20, 'high': 32}[tier]
    wm, wt = bbp.material(tier, 'walls', wall); rm, rt = bbp.material(tier, 'roofs', roof)
    V, F, M = [], [], []
    r = D / 2
    for k in range(seg):
        a = 2 * math.pi * k / seg; V += [(r * math.cos(a), r * math.sin(a), 0), (r * math.cos(a), r * math.sin(a), H)]
    for k in range(seg):
        a, b = 2 * k, 2 * ((k + 1) % seg); F.append((a, b, b + 1, a + 1)); M.append(1)
    rings = 4 if dome else 1
    prev = [2 * k + 1 for k in range(seg)]
    for j in range(1, rings + 1):
        t = j / rings
        if j == rings:
            V.append((0, 0, H + roof_h)); top = len(V) - 1
            for k in range(seg): F.append((prev[k], prev[(k + 1) % seg], top)); M.append(0)
        else:
            rr = r * math.cos(t * math.pi / 2); zz = H + roof_h * math.sin(t * math.pi / 2)
            cur = []
            for k in range(seg):
                a = 2 * math.pi * k / seg; V.append((rr * math.cos(a), rr * math.sin(a), zz)); cur.append(len(V) - 1)
            for k in range(seg): F.append((prev[k], prev[(k + 1) % seg], cur[(k + 1) % seg], cur[k])); M.append(0)
            prev = cur
    me = bbp.mesh_from(name, V, F, M, [rm, wm], {0: rt, 1: wt})
    # 圆柱侧面改成按周长展开的 UV（方块投影在圆柱上会拉伸）
    uv = me.uv_layers['UVMap']
    for p in me.polygons:
        if p.material_index != 1: continue
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (math.atan2(co.y, co.x) * r / wt, co.z / wt)
    return me


def hay_round(m):
    m.cyl((0, -0.6, 0.75), (0, 0.6, 0.75), 0.75, 0.75, (0.36, 0.27, 0.11))
    return {'diameter_m': 1.5, 'width_m': 1.2}


def hay_stack(m):
    rng = random.Random(4)
    for z in range(3):
        for y in range(2):
            for x in range(4 - z):
                m.box(x * 1.0 - 1.5 + z * 0.5, y * 0.48, z * 0.36, 0.95, 0.46, 0.35,
                      tuple(c * rng.uniform(0.9, 1.05) for c in (0.40, 0.31, 0.13)))
    return {'length_m': 4.0, 'height_m': 1.08}


def pivot_span(m):
    """中心支轴喷灌机一跨（50 m）：一根钢管 + 下面的桁架拉杆 + 跨端的 A 字形塔架（两个大轮子）"""
    L, z = 50.0, 3.8; steel = (0.55, 0.56, 0.57)
    m.cyl((0, 0, z), (L, 0, z), 0.08, 0.08, steel)
    n = {'low': 3, 'medium': 8, 'high': 14}[m.tier]
    for k in range(n):
        x0, x1 = L * k / n, L * (k + 1) / n
        m.cyl((x0, 0, z), ((x0 + x1) / 2, 0, z - 1.2), 0.02, 0.02, steel, sides=4)
        m.cyl(((x0 + x1) / 2, 0, z - 1.2), (x1, 0, z), 0.02, 0.02, steel, sides=4)
        if m.tier != 'low':
            m.cyl(((x0 + x1) / 2, 0, z), ((x0 + x1) / 2, 0, z - 1.6), 0.01, 0.01, (0.2, 0.2, 0.2), sides=3)   # 喷头吊管
    for sy in (-1, 1):
        m.cyl((L, 0, z), (L, sy * 1.6, 0.6), 0.06, 0.06, steel, sides=5)
        m.cyl((L, sy * 1.6, 0.6), (L, sy * 1.6 + sy * 0.5, 0.6), 0.6, 0.6, bp.DARK, sides=10 if m.tier != 'low' else 6)
    return {'length_m': L, 'height_m': z, 'note': '原点在支轴一端，沿 +x；一台喷灌机 = 中心塔 + 若干跨首尾相接（一般 7 ~ 9 跨，约 400 m）'}


def water_tower(m):
    """水塔：四条腿撑起的球形水箱（35 m 高）"""
    steel = (0.75, 0.76, 0.74); H = 30.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.cyl((sx * 5, sy * 5, 0), (sx * 3, sy * 3, H), 0.35, 0.3, steel)
    for z in (10, 20):
        for (ax, ay), (bx, by) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            k = 5 - 2 * z / H
            m.cyl((ax * k, ay * k, z), (bx * k, by * k, z), 0.08, 0.08, steel, sides=5)
    m.cyl((0, 0, 0), (0, 0, H), 0.6, 0.6, steel)
    bmv = bmesh.ops.create_uvsphere(m.bm, u_segments={'low': 10, 'medium': 20, 'high': 32}[m.tier], v_segments={'low': 6, 'medium': 12, 'high': 18}[m.tier], radius=7.0)
    bmesh.ops.transform(m.bm, matrix=Matrix.Translation((0, 0, H + 4)) @ Matrix.Diagonal((1, 1, 0.72, 1)), verts=bmv['verts'])
    m._paint(bmv['verts'], steel, 0)
    return {'height_m': H + 9.0, 'tank_diameter_m': 14.0}


def windmill(m):
    """抽水风车：10 m 高的四棱钢架，顶上 18 片叶的风轮 + 尾翼"""
    steel = (0.45, 0.46, 0.47); H = 10.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.cyl((sx * 1.4, sy * 1.4, 0), (sx * 0.25, sy * 0.25, H), 0.05, 0.04, steel, sides=4)
    for z in (2.5, 5.0, 7.5):
        k = 1.4 - 1.15 * z / H
        for (ax, ay), (bx, by) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
            m.cyl((ax * k, ay * k, z), (bx * k, by * k, z), 0.025, 0.025, steel, sides=3)
    n = {'low': 6, 'medium': 12, 'high': 18}[m.tier]
    for k in range(n):
        a = 2 * math.pi * k / n
        m.box(-0.5, math.cos(a) * 1.2, H + 0.4 + math.sin(a) * 1.2, 0.02, 0.35, 1.6, steel, rot=0)
    m.box(0.9, 0, H + 0.4, 1.8, 0.03, 0.8, steel)
    return {'height_m': H + 1.9}


def build_tier(tier):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(HQ, tier, 'farm')
    items = []
    def keep(name, me, info):
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        ob = bpy.data.objects.new(name, me); c.objects.link(ob); d = ob.dimensions
        items.append({'name': name, **info, 'size_m': [round(d.x, 1), round(d.y, 1), round(d.z, 1)], 'triangles': hv.tris(me), 'c': c})
    keep('barn_red', building('barn_red', tier, 18, 12, 4.5, 'gambrel', 1.0, 'siding_red', 'metal_roof', door=(4.5, 4.2)),
         {'kind': 'farm_building'})
    keep('machine_shed', building('machine_shed', tier, 24, 15, 5.0, 'gable', 0.3, 'metal_wall', 'metal_roof', door=(8.0, 4.5)),
         {'kind': 'farm_building'})
    keep('grain_bin_large', cylinder_building('grain_bin_large', tier, 11.0, 8.0, 3.2, 'metal_wall', 'metal_roof'), {'kind': 'farm_storage'})
    keep('grain_bin_small', cylinder_building('grain_bin_small', tier, 6.0, 6.0, 2.0, 'metal_wall', 'metal_roof'), {'kind': 'farm_storage'})
    keep('silo_concrete', cylinder_building('silo_concrete', tier, 5.5, 20.0, 2.6, 'concrete', 'metal_roof', dome=True),
         {'kind': 'farm_storage'})
    vc = bpy.data.materials.new('farm_vc'); b_ = vc.node_tree.nodes['Principled BSDF']
    n = vc.node_tree.nodes.new('ShaderNodeVertexColor'); n.layer_name = 'Col'
    vc.node_tree.links.new(n.outputs['Color'], b_.inputs['Base Color']); b_.inputs['Roughness'].default_value = 0.8
    for name, fn, kind in (('hay_round', hay_round, 'farm_small'), ('hay_stack', hay_stack, 'farm_small'),
                           ('pivot_span', pivot_span, 'irrigation'), ('water_tower', water_tower, 'utility'),
                           ('windmill', windmill, 'farm_small')):
        m = bp.Mesh(tier); info = fn(m)
        keep(name, m.finish(name, [vc]), {'kind': kind, **info})
    os.makedirs(base, exist_ok=True)
    bpy.data.libraries.write(os.path.join(base, 'farm_export.blend'), {i['c'] for i in items}, fake_user=True)
    for i in items: i.pop('c')
    return items


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    man = {}
    for tier in TIERS:
        for it in build_tier(tier):
            e = man.setdefault(it['name'], {k: v for k, v in it.items() if k != 'triangles'})
            e.setdefault('triangles', {})[tier] = it['triangles']
    with open(os.path.join(HQ, 'farm_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'assets': list(man.values())}, f, ensure_ascii=False, indent=1)
    print('farm', len(man))
    if '--preview' in argv:
        for t in TIERS:
            bn.grid_preview(os.path.join(HQ, t, 'farm', 'farm_export.blend'), os.path.join(HQ, 'previews', f'farm_{t}.png'), cols=5)


if __name__ == '__main__':
    main()
