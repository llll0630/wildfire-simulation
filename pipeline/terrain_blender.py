# ===============================================================
#  地形：在 Blender 里搭地形场景（pipeline.py 加了 --terrain 时用）
#  地形网格、地面贴图烘焙、素材按地形高度摆放、中景简化树
#  网页端不能用 Blender 的节点材质，所以：近景地面烘焙成一张图；中景远景用顶点颜色；雾在网页端做
# ===============================================================
import bpy, os, json, math
import numpy as np
import pipeline_rules as rules
import terrain_core as tc

BAKE_SIZE = 2048             # 近景地面贴图边长（像素），覆盖 ±110 m，约 0.1 m 一个像素
GROUND_DIR = os.path.join(rules.ASSET_DIR, 'ground')


# ---------------- 小工具 ----------------
def make_mesh_object(name, x, y, z, faces, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.stack([x, y, z], axis=1).tolist(), [], faces)
    me.update()
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def set_colors(me, name, rgba):
    attr = me.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set('color', np.asarray(rgba, dtype=np.float32).ravel())
    me.color_attributes.active_color_name = name


def mesh_tris(me):
    me.calc_loop_triangles()
    return len(me.loop_triangles)


def ground_textures():
    with open(os.path.join(GROUND_DIR, 'ground_export_manifest.json'), encoding='utf-8') as f:
        gm = json.load(f)
    paths = {k: os.path.join(GROUND_DIR, v['texture']) for k, v in gm.items()}
    tiles = {k: v['tile_m'] for k, v in gm.items()}
    return paths, tiles


def load_tex(path):
    img = bpy.data.images.load(path, check_existing=True)
    px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(img.size[1], img.size[0], 4)[:, :, :3]


def texture_average(path):
    return np.power(load_tex(path).reshape(-1, 3).mean(axis=0), 2.2)      # 转成线性颜色


# ---------------- 材质 ----------------
def vertex_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']
    p.inputs['Roughness'].default_value = 0.95
    vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'base'
    nt.links.new(vc.outputs['Color'], p.inputs['Base Color'])
    return m


def flat_material(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*rgb, 1)
    p.inputs['Roughness'].default_value = 0.9
    return m


def bake_near(t, paths, tiles, avg):
    """近景地面：按地表占比混合 6 张贴图，烘焙成一张图（数值都是贴图原本的 sRGB）"""
    n = BAKE_SIZE; half = tc.NEAR_HALF
    ax = (np.arange(n) + 0.5) / n * 2 * half - half
    x, y = np.meshgrid(ax, ax)
    c = t.cover(x, y)
    # 地表分界不那么规整：权重加一点起伏，再平方让边界更明显
    wob = 1 + 0.5 * np.sin(x * 0.21 + np.sin(y * 0.13) * 2) * np.sin(y * 0.17 + np.sin(x * 0.09) * 2)
    wt = (c * wob) ** 2
    wt = wt / np.maximum(wt.sum(axis=0), 1e-6)

    def tiled(key):
        tx = load_tex(paths[key]); h, w = tx.shape[:2]
        u = np.mod(x / tiles[key], 1.0); v = np.mod(y / tiles[key], 1.0)
        return tx[np.minimum((v * h).astype(int), h - 1), np.minimum((u * w).astype(int), w - 1)]

    # 冰雪暂时没有贴图，并入草地（素材待补）
    groups = (('mat_grass', wt[tc.G] + wt[tc.S] + wt[tc.N]), ('mat_litter', wt[tc.F]), ('mat_urban', wt[tc.D]),
              ('mat_bare', wt[tc.B]), ('mat_farmland', wt[tc.A]), ('mat_water', wt[tc.W]))
    rgb = np.zeros((n, n, 3), dtype=np.float32)
    for key, weight in groups:
        if weight.max() > 0.001:
            rgb += tiled(key) * weight[..., None]
    fade = tc.smoothstep(85, half - 3, np.maximum(np.abs(x), np.abs(y)))[..., None]      # 边缘淡出到中景颜色
    far = np.power(np.clip(tc.cover_colors(t, x, y, avg), 0, 1), 1 / 2.2)
    rgb = rgb * (1 - fade) + far * fade

    img = bpy.data.images.new('near_ground_bake', n, n, alpha=False)
    rgba = np.concatenate([rgb, np.ones((n, n, 1), dtype=np.float32)], axis=2)
    img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    img.pack()
    m = bpy.data.materials.new('near_ground')
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']; p.inputs['Roughness'].default_value = 0.95
    tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = img
    nt.links.new(tx.outputs['Color'], p.inputs['Base Color'])
    return m


def set_uv(me, half):
    uv = me.uv_layers.new(name='UVMap')
    vi = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get('vertex_index', vi)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get('co', co)
    data = (co.reshape(-1, 3)[vi][:, :2] + half) / (2 * half)
    uv.uv.foreach_set('vector', data.ravel())


# ---------------- 实例 ----------------
def scatter(coll, name, src, items):
    """和 pipeline.scatter 一样（同款素材合成一组 GPU 实例），只是每个点带上地形高度"""
    me = bpy.data.meshes.new(f'_pts_{name}')
    me.from_pydata([(t['x'], t['y'], t['z']) for t in items], [], [])
    rot = me.attributes.new('rot', 'FLOAT', 'POINT')
    rot.data.foreach_set('value', [t['rot'] for t in items])
    scl = me.attributes.new('scl', 'FLOAT_VECTOR', 'POINT')
    scl.data.foreach_set('vector', [v for t in items for v in (t['s'], t['s'], t.get('sz', t['s']))])
    ob = bpy.data.objects.new(f'{name}_scatter', me)
    coll.objects.link(ob)

    ng = bpy.data.node_groups.new(f'_scatter_{name}', 'GeometryNodeTree')
    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gi, go = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    inst = N.new('GeometryNodeInstanceOnPoints')
    info = N.new('GeometryNodeObjectInfo'); info.inputs['As Instance'].default_value = True
    info.inputs['Object'].default_value = src
    a_rot = N.new('GeometryNodeInputNamedAttribute'); a_rot.data_type = 'FLOAT'; a_rot.inputs['Name'].default_value = 'rot'
    a_scl = N.new('GeometryNodeInputNamedAttribute'); a_scl.data_type = 'FLOAT_VECTOR'; a_scl.inputs['Name'].default_value = 'scl'
    eul = N.new('ShaderNodeCombineXYZ')
    L.new(a_rot.outputs['Attribute'], eul.inputs['Z'])
    L.new(gi.outputs['Geometry'], inst.inputs['Points'])
    L.new(info.outputs['Geometry'], inst.inputs['Instance'])
    L.new(eul.outputs['Vector'], inst.inputs['Rotation'])
    L.new(a_scl.outputs['Attribute'], inst.inputs['Scale'])
    L.new(inst.outputs['Instances'], go.inputs['Geometry'])
    ob.modifiers.new('scatter', 'NODES').node_group = ng


def simple_tree(name, rings, rgb):
    """中景简化树的模型：几圈六边形，高度 1、半径按比例"""
    sides = 6
    verts, faces = [], []
    for hf, rr in rings:
        if rr == 0.0:
            verts.append((0, 0, hf))
        else:
            verts += [(rr * math.cos(2 * math.pi * k / sides), rr * math.sin(2 * math.pi * k / sides), hf)
                      for k in range(sides)]
    for ring in range(len(rings) - 2):
        a0 = ring * sides; a1 = a0 + sides
        faces += [(a0 + k, a0 + (k + 1) % sides, a1 + (k + 1) % sides, a1 + k) for k in range(sides)]
    top = len(verts) - 1; a0 = top - sides
    faces += [(a0 + k, a0 + (k + 1) % sides, top) for k in range(sides)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces); me.update()
    me.materials.append(flat_material(name + '_mat', rgb))
    return bpy.data.objects.new(name, me)


# ---------------- 搭场景 ----------------
def build(base, t, p, seed, export=False, timing=None):
    """base = pipeline 模块（用它的 clear_scene / make_coll / append_collections）；p = terrain_core.plan() 的结果。
    export=True 时用导出用素材（*_export.blend），树草用 GPU 实例；否则用普通素材，和原 pipeline 一样逐个放"""
    import time
    clock = time.perf_counter
    t0 = clock()
    layout = p['layout']
    base.clear_scene()
    paths, tiles = ground_textures()
    avg = {k: texture_average(v) for k, v in paths.items()}
    c_ter = base.make_coll('Terrain')
    stats = {}

    axis, x, y, mid_z, faces = tc.mid_mesh(t)
    mid = make_mesh_object('terrain_mid_far', x, y, mid_z, faces, c_ter)
    set_colors(mid.data, 'base', np.concatenate([tc.cover_colors(t, x, y, avg), np.ones((len(x), 1))], axis=1))
    mid.data.materials.append(vertex_material('terrain_far'))
    stats['mid_far_vertices'] = len(x)

    ax, x, y, near_z, faces = tc.near_mesh(t)
    x2, y2, z2, f2 = tc.add_skirt(x, y, near_z, faces, len(ax))
    near = make_mesh_object('terrain_near', x2, y2, z2, f2, c_ter)
    set_uv(near.data, tc.NEAR_HALF)
    stats['near_vertices'] = len(x)
    if timing is not None:
        timing['terrain_meshes'] = round(clock() - t0, 2); t0 = clock()
    near.data.materials.append(bake_near(t, paths, tiles, avg))
    if timing is not None:
        timing['bake_ground_texture'] = round(clock() - t0, 2); t0 = clock()

    suffix = '_export' if export else ''
    tris = {'terrain': mesh_tris(mid.data) + mesh_tris(near.data)}
    veg = {}
    for k in ('trees', 'shrubs', 'grass'):
        veg.update(base.append_collections(os.path.join(k, f'{k}{suffix}.blend'), {i['name'] for i in layout[k]}))
    houses = base.append_collections(os.path.join('houses', f'houses{suffix}.blend'), {h['name'] for h in layout['houses']})

    def put(coll, src, it):
        ob = bpy.data.objects.new(it['name'], None)
        ob.instance_type = 'COLLECTION'; ob.instance_collection = src
        ob.location = (it['x'], it['y'], it['z']); ob.rotation_euler = (0, 0, it['rot'])
        s = it.get('s', 1.0)
        ob.scale = (s, s, it.get('sz', s))
        coll.objects.link(ob)

    c_house = base.make_coll('Houses')
    tris['houses'] = 0
    for h in layout['houses']:
        put(c_house, houses[h['name']], h)
        tris['houses'] += sum(mesh_tris(o.data) for o in houses[h['name']].all_objects if o.type == 'MESH')
    for key in ('trees', 'shrubs', 'grass'):
        coll = base.make_coll(key.capitalize())
        tris[key] = 0
        for name in sorted({i['name'] for i in layout[key]}):
            items = [i for i in layout[key] if i['name'] == name]
            src = veg[name]
            if export:
                scatter(coll, name, src.objects[0], items)
            else:
                for it in items:
                    put(coll, src, it)
            tris[key] += sum(mesh_tris(o.data) for o in src.all_objects if o.type == 'MESH') * len(items)
    stats['placed'] = {k: len(layout[k]) for k in ('houses', 'trees', 'shrubs', 'grass')}

    # 中景简化树：两款模型各一组实例
    rng, in_view = p['rng'], p['in_view']
    fx, fy, fz, fk = tc.far_tree_points(t, np.random.default_rng(seed),
                                        lambda x, y: in_view(x, y, 7, 45, 400) & (np.sin(x * 0.09 + np.sin(y * 0.07) * 3) > -0.5))
    pine = simple_tree('mid_pine', [(0.15, 0.27), (1.0, 0.0)], (0.030, 0.078, 0.034))
    oak = simple_tree('mid_oak', [(0.32, 0.24), (0.62, 0.44), (0.9, 0.24), (1.0, 0.0)], (0.060, 0.135, 0.035))
    c_mid = base.make_coll('MidTrees')
    groups = ([], [])
    for px, py, pz, k in zip(fx.tolist(), fy.tolist(), fz.tolist(), fk.tolist()):
        which = int(rng.random() < 0.5)
        s_ = rng.uniform(8.5, 11) * k if which else rng.uniform(7.5, 10) * k
        groups[which].append({'x': px, 'y': py, 'z': pz, 'rot': rng.uniform(0, 6.28), 's': s_})
    if groups[0]:
        scatter(c_mid, 'mid_pine', pine, groups[0])
    if groups[1]:
        scatter(c_mid, 'mid_oak', oak, groups[1])
    tris['mid_trees'] = mesh_tris(pine.data) * len(groups[0]) + mesh_tris(oak.data) * len(groups[1])
    stats['mid_trees'] = len(fx)
    stats['triangles'] = {**tris, 'total': sum(tris.values())}
    if timing is not None:
        timing['place_assets'] = round(clock() - t0, 2)

    checks = tc.run_checks(t, layout, layout['fire'], p['spawn'], near_z, mid_z, axis, (fx, fy))
    return {'stats': stats, 'checks': checks, 'height': {
        'half': tc.NEAR_HALF, 'step': tc.NEAR_STEP, 'n': len(ax), 'data': [round(float(v), 2) for v in near_z.tolist()]}}


def scene_json(t, p, seed, built, inputs):
    """沿用原 pipeline 的 scene.json 格式，火、出生点补上地形高度，另加地面高度表"""
    layout, decision = p['layout'], p['decision']
    js = rules.scene_json(seed, decision, layout)
    fire = layout['fire']
    if fire:
        cx, cy = fire['center']
        js['fire']['center'] = rules.to_three(cx, cy, float(t.height(np.array([cx]), np.array([cy]))[0]))
        js['fire']['burning_trees'] = [{'position': rules.to_three(tr['x'], tr['y'], tr['z']), 'height': round(tr['h'], 2)}
                                       for tr in fire['burning_trees']]
    sx, sy = p['spawn']; dx, dy = p['dir']
    eye = float(t.height(np.array([sx]), np.array([sy]))[0]) + 1.7
    js['spawn'] = {'position': rules.to_three(sx, sy, eye),
                   'look_at': rules.to_three(sx + dx * 100, sy + dy * 100,
                                             eye + 100 * math.tan(math.radians(max(p['skyline_deg'], 0) * 0.5 + 2)))}
    js['height'] = built['height']
    js['aerial'] = {'position': rules.to_three(-dx * 240, -dy * 240, 60), 'look_at': rules.to_three(dx * 300, dy * 300, 40)}
    js['debug'].update({'inputs': inputs, 'placement': p['placement'], 'stats': built['stats'],
                        'checks': built['checks'], 'view': {'best': p['best_view'], 'table': p['view_rows']}})
    return js
