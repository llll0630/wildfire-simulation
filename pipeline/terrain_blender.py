# ===============================================================
#  地形：在 Blender 里搭地形场景（pipeline.py 加了 --terrain 时用）
#  地形网格、地面贴图烘焙、素材按地形高度摆放、中景简化树
#  网页端不能用 Blender 的节点材质，所以：近景地面烘焙成一张图；中景远景用顶点颜色；雾在网页端做
# ===============================================================
import bpy, os, json, math
import numpy as np
import pipeline_rules as rules
import terrain_core as tc
import quality as qa

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


def ground_textures(folder=GROUND_DIR):
    """地面贴图清单：{材质: 路径}、{材质: 一张代表几米}。名字以 _ 开头的不是地面材质（比如细节法线），不算在里面"""
    with open(os.path.join(folder, 'ground_export_manifest.json'), encoding='utf-8') as f:
        gm = json.load(f)
    paths = {k: os.path.join(folder, v['texture']) for k, v in gm.items() if not k.startswith('_')}
    tiles = {k: v['tile_m'] for k, v in gm.items() if not k.startswith('_')}
    return paths, tiles


def downsample(tx, texels):
    """贴图每次缩小一半（4 个像素取平均），直到一张贴图的像素数接近烘焙图上一张贴图占的像素数，避免取样时闪烁、出噪点"""
    while tx.shape[0] / 2 >= texels and tx.shape[0] % 2 == 0 and tx.shape[1] % 2 == 0:
        tx = 0.25 * (tx[0::2, 0::2] + tx[1::2, 0::2] + tx[0::2, 1::2] + tx[1::2, 1::2])
    return tx


def sample_bilinear(tx, u, v):
    """按 0~1 重复的坐标双线性取样（四周接着重复）"""
    h, w = tx.shape[:2]
    fu = u * w - 0.5; fv = v * h - 0.5
    i0 = np.floor(fu).astype(int); j0 = np.floor(fv).astype(int)
    a = (fu - i0)[..., None]; b = (fv - j0)[..., None]
    i0 %= w; j0 %= h; i1 = (i0 + 1) % w; j1 = (j0 + 1) % h
    return (tx[j0, i0] * (1 - a) + tx[j0, i1] * a) * (1 - b) + (tx[j1, i0] * (1 - a) + tx[j1, i1] * a) * b


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


def bake_near(t, paths, tiles, avg, bld=None, n=BAKE_SIZE, detail=None):
    """近景地面：每个像素用哪种地面贴图按 ground_rules.py 的规则（Overture 地面类型 > LANDFIRE 地表 + 植被类型），
    烘焙成一张图（数值都是贴图原本的 sRGB）。有真实建筑数据（bld）时，路、小路、铺装、建筑底下按真实位置换颜色
    n = 烘焙图边长；大图分几段算（每段约 400 万个像素），省内存
    贴图先缩小到烘焙图的精度再双线性取样（写实贴图细节多，直接取样会出噪点）
    detail = (细节法线贴图路径, 每几米重复, 强度)：近景地面材质加法线贴图，用第二套 UV（UVDetail）"""
    import ground_rules as gr
    half = tc.NEAR_HALF
    ax = (np.arange(n) + 0.5) / n * 2 * half - half
    mats = sorted(paths)
    texels = {k: tiles[k] / (2 * half / n) for k in paths}          # 一张贴图在烘焙图上占几个像素
    cache = {}
    def tex(key):
        if key not in cache:
            cache[key] = downsample(load_tex(paths[key]), texels[key])
        return cache[key]
    def sample(key, xs, ys):
        return sample_bilinear(tex(key), np.mod(xs / tiles[key], 1.0), np.mod(ys / tiles[key], 1.0))

    by_class = gr.class_tables(mats)
    if bld is not None:
        import buildings_data as bdm
        by_ground = gr.ground_table(mats, bdm.GROUND)
        water_ids = np.array(sorted(bdm.WATER_GROUND))

    rgb = np.empty((n, n, 3), dtype=np.float32)
    rows = max(1, 2048 * 2048 // n)
    for r0 in range(0, n, rows):
        x, y = np.meshgrid(ax, ax[r0:r0 + rows])
        c = t.cover(x, y)
        # 地表分界不那么规整：权重加一点起伏，再平方让边界更明显
        wob = 1 + 0.5 * np.sin(x * 0.21 + np.sin(y * 0.13) * 2) * np.sin(y * 0.17 + np.sin(x * 0.09) * 2)
        wt = (c * wob) ** 2
        wt = wt / np.maximum(wt.sum(axis=0), 1e-6)
        # 植被类型是 30 m 一格：取的位置抖一抖，格子边界不是直角
        veg = t.veg(x + 9 * np.sin(y * 0.11 + np.sin(x * 0.07) * 2), y + 9 * np.sin(x * 0.13 + np.sin(y * 0.05) * 2))

        part = np.zeros(x.shape + (3,), dtype=np.float32)
        for cl in range(len(gr.CLASS_DEFAULT)):
            w = wt[cl]
            if w.max() <= 0.001:
                continue
            mid = by_class[cl][veg]
            for m in np.unique(mid[w > 0.001]).tolist():
                sel = (mid == m) & (w > 0.001)
                part[sel] += sample(mats[m], x[sel], y[sel]) * w[sel][:, None]
        if bld is not None:
            # Overture 说明了地面是什么的地方换掉：每个像素在周围取 4 个点（再加一点起伏），按占比混合，
            # 1 m 一格的数据边界不会是锯齿状的台阶
            jx = 0.5 * np.sin(y * 0.37 + np.sin(x * 0.23) * 2); jy = 0.5 * np.sin(x * 0.31 + np.sin(y * 0.29) * 2)
            oms = [by_ground[bld.ground(x + jx + ox, y + jy + oy)] for ox, oy in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))]
            g = bld.ground(x + jx, y + jy)
            cover = np.zeros(x.shape, np.float32)
            add = np.zeros(x.shape + (3,), np.float32)
            for m in np.unique(np.concatenate([o[o >= 0] for o in oms])).tolist():
                w = sum((o == m).astype(np.float32) for o in oms) / 4
                sel = w > 0
                add[sel] += sample(mats[m], x[sel], y[sel]) * w[sel][:, None]
                cover += w
            part = part * (1 - cover)[..., None] + add
            # 水边一圈湿沙泥
            shore = bld.shore(x + jx, y + jy, gr.SHORE_M) & ~np.isin(g, water_ids)
            if shore.any():
                part[shore] = 0.15 * part[shore] + 0.85 * sample('shore', x[shore], y[shore])
            import buildings_blender as bb
            part = bb.overlay_ground(bld, x, y, part, lambda key, sel: sample(key, x[sel], y[sel]))
        fade = tc.smoothstep(85, half - 3, np.maximum(np.abs(x), np.abs(y)))[..., None]      # 边缘淡出到中景颜色
        far = np.power(np.clip(tc.cover_colors(t, x, y, avg), 0, 1), 1 / 2.2)
        rgb[r0:r0 + rows] = part * (1 - fade) + far * fade

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
    if detail:
        path, _, strength = detail
        dimg = bpy.data.images.load(path, check_existing=True); dimg.colorspace_settings.name = 'Non-Color'
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'UVDetail'
        dt = nt.nodes.new('ShaderNodeTexImage'); dt.image = dimg
        nm = nt.nodes.new('ShaderNodeNormalMap'); nm.uv_map = 'UVDetail'; nm.inputs['Strength'].default_value = strength
        nt.links.new(uvn.outputs['UV'], dt.inputs['Vector'])
        nt.links.new(dt.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], p.inputs['Normal'])
    return m


def set_detail_uv(me, tile):
    """第二套 UV：按世界坐标每 tile 米重复一次（细节法线贴图用）"""
    uv = me.uv_layers.new(name='UVDetail')
    vi = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get('vertex_index', vi)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get('co', co)
    uv.uv.foreach_set('vector', (co.reshape(-1, 3)[vi][:, :2] / tile).ravel())
    me.uv_layers.active = me.uv_layers['UVMap']


def set_uv(me, half):
    uv = me.uv_layers.new(name='UVMap')
    vi = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get('vertex_index', vi)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32); me.vertices.foreach_get('co', co)
    data = (co.reshape(-1, 3)[vi][:, :2] + half) / (2 * half)
    uv.uv.foreach_set('vector', data.ravel())


def set_ground_normals(me, z, n):
    """法线按地面坡度算，挡边也用它上沿那个点的法线；否则挡边是竖着的面，露出来时是一条暗线"""
    gy, gx = np.gradient(z.reshape(n, n), tc.NEAR_STEP)
    nrm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1).reshape(-1, 3)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    me.normals_split_custom_set_from_vertices(np.concatenate([nrm, nrm[tc.ring_indices(n)]]).tolist())


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


CONIFER_VEG = ('pine_west', 'fir_spruce', 'pine_east', 'pinyon_juniper', 'swamp_forest')
BROADLEAF_VEG = ('hardwood', 'oak_savanna', 'riparian_forest', 'tropical_forest', 'developed_trees')


def veg_no_tree(t, x, y):
    """中景简化树：植被类型里几乎不长树的地方（沙漠灌丛、草原、农田……，见 terrain_core.VEG_KEEP）不放"""
    import veg_types as vt
    table = np.array([tc.VEG_KEEP.get(n, (1.0, 1.0, 1.0))[0] for n in vt.TYPES])
    return table[t.veg(x, y)] < 0.05


def veg_mid_kind(t, x, y):
    """中景简化树用哪种形状：针叶林 → 0（松树形），阔叶林 → 1（阔叶形），其他 → -1（随机）"""
    import veg_types as vt
    table = np.full(len(vt.TYPES), -1)
    table[[vt.ID[n] for n in CONIFER_VEG]] = 0
    table[[vt.ID[n] for n in BROADLEAF_VEG]] = 1
    return table[t.veg(x, y)]


def mid_no_tree(bld, x, y):
    """中景简化树：Overture 地面类型里几乎不长树的地方（水、沙滩、农田、停车场、冰川……）不放"""
    import buildings_data as bdm
    return tc.ground_keep_table(bdm.GROUND)['trees'][bld.ground(x, y)] < 0.05


# ---------------- 搭场景 ----------------
def build(base, t, p, seed, export=False, timing=None, quality='low'):
    """base = pipeline 模块（用它的 clear_scene / make_coll / append_collections）；p = terrain_core.plan() 的结果。
    export=True 时用导出用素材（*_export.blend），树草用 GPU 实例；否则用普通素材，和原 pipeline 一样逐个放。
    quality = 画质档位（quality.py）；中档、高档只有导出用素材"""
    import time
    clock = time.perf_counter
    t0 = clock()
    layout = p['layout']
    base.clear_scene()
    low_paths, _ = ground_textures()
    avg = {k: texture_average(v) for k, v in low_paths.items()}  # 中景远景颜色：按 8 大类的平均色（三档一样）
    cfg = qa.TIERS[quality]
    paths, tiles = ground_textures(qa.ground_dir(quality))       # 近景地面材质库（三档同一套）
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
    set_ground_normals(near.data, near_z, len(ax))
    stats['near_vertices'] = len(x)
    if timing is not None:
        timing['terrain_meshes'] = round(clock() - t0, 2); t0 = clock()
    detail = None
    if cfg['detail']:
        with open(os.path.join(qa.ground_dir(quality), 'ground_export_manifest.json'), encoding='utf-8') as f:
            d = json.load(f)['_detail_normal']
        detail = (os.path.join(qa.ground_dir(quality), d['texture']), d['tile_m'], cfg['detail'])
        set_detail_uv(near.data, d['tile_m'])
    near.data.materials.append(bake_near(t, paths, tiles, avg, p.get('buildings'), n=cfg['bake'], detail=detail))
    if timing is not None:
        timing['bake_ground_texture'] = round(clock() - t0, 2); t0 = clock()

    bld, buildings_info = p.get('buildings'), None
    if bld is not None:
        import buildings_blender as bb
        surf = bb.Surface(ax, near_z, axis, mid_z)
        buildings_info, stats['buildings'] = bb.build(bld, surf, base.make_coll('Buildings'), quality)
        if timing is not None:
            timing['buildings'] = round(clock() - t0, 2); t0 = clock()

    suffix = '_export' if export else ''
    tris = {'terrain': mesh_tris(mid.data) + mesh_tris(near.data)}
    groups = qa.vegetation_groups(quality, layout, export)   # 每档对应的素材（含分树种素材），见 quality.py
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
        for name, src, items in groups[key]:
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
                                        lambda x, y: (in_view(x, y, 7, 45, 400) & (np.sin(x * 0.09 + np.sin(y * 0.07) * 3) > -0.5))
                                        | (bld.blocked(x, y) if bld is not None else False)
                                        | (mid_no_tree(bld, x, y) if bld is not None else False)
                                        | veg_no_tree(t, x, y))
    # 中景、远景：每个树种一张公告板（几张交叉面片贴树的照片），按植被类型选树种（asset_use.py）
    import asset_use as au
    c_mid = base.make_coll('MidTrees')
    table = au.billboard_table(qa.species_manifest())
    mid_kind = veg_mid_kind(t, fx, fy)              # 没有对应树种时：0 松树形、1 阔叶形、-1 随机
    vegs = t.veg(fx, fy)
    groups = {}
    for px, py, pz, k, mk, v in zip(fx.tolist(), fy.tolist(), fz.tolist(), fk.tolist(), mid_kind.tolist(), vegs.tolist()):
        name = au.pick_billboard(table, v, mk, rng)
        groups.setdefault(name, []).append({'x': px, 'y': py, 'z': pz, 'rot': rng.uniform(0, 6.28), 's': rng.uniform(0.85, 1.15) * k})
    bbs = qa.load(qa.lib_path(quality, 'billboards'), groups)
    tris['mid_trees'] = 0
    for name in sorted(groups):
        src = bbs[name].objects[0]
        scatter(c_mid, name, src, groups[name])
        tris['mid_trees'] += mesh_tris(src.data) * len(groups[name])
    stats['mid_trees'] = len(fx)
    stats['mid_billboards'] = {n: len(v) for n, v in sorted(groups.items())}
    # 小物件：石块、电线杆 + 电线、栅栏、路灯（props_blender.py）
    import props_blender as pb
    stats['props'], tris['props'] = pb.place(t, p, quality, base, scatter, mesh_tris, rng)
    if bld is not None:
        tris['buildings_roads'] = sum(stats['buildings']['triangles'].values())
    stats['triangles'] = {**tris, 'total': sum(tris.values())}
    if timing is not None:
        timing['place_assets'] = round(clock() - t0, 2)

    checks = tc.run_checks(t, layout, layout['fire'], p['spawn'], near_z, mid_z, axis, (fx, fy), bld)
    return {'stats': stats, 'checks': checks, 'buildings': buildings_info, 'height': {
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
    bld = p.get('buildings')
    if bld is not None:
        # 真实建筑清单：网站可以用来点选、搜索、飞到某栋楼。center 是屋顶中心（three.js 坐标）
        js['buildings'] = {'status': bld.status, 'source': bld.meta.get('source'), 'release': bld.meta.get('release'),
                           'half': bld.meta.get('half'),
                           'items': [{'id': b['id'], 'type': b['type'], 'name': b['name'], 'height': b['height'],
                                      'height_source': b['height_source'], 'data_source': b['data_source'],
                                      'center': rules.to_three(b['center'][0], b['center'][1], b['roof'])}
                                     for b in built['buildings'] or []]}
        if built['buildings']:
            # 高空视角：相机在场景中心的另一侧高处，朝建筑最多的方向看，场景中心在前景、街区在中间
            c = np.array([b['center'] for b in built['buildings']]).mean(axis=0)
            if np.hypot(*c) > 100:
                ux, uy = c / np.hypot(*c); tx, ty = c
            else:                                     # 建筑都在中心附近：按原来的方向看
                ux, uy = dx, dy; tx, ty = 0.0, 0.0
            px, py = -ux * 450, -uy * 450
            pz = float(t.height(np.array([px]), np.array([py]))[0]) + 350      # 按脚下地形抬高，不会放进山里
            js['aerial'] = {'position': rules.to_three(px, py, pz),
                            'look_at': rules.to_three(tx, ty, float(t.height(np.array([tx]), np.array([ty]))[0]))}
    js['debug'].update({'inputs': inputs, 'placement': p['placement'], 'stats': built['stats'],
                        'checks': built['checks'], 'view': {'best': p['best_view'], 'table': p['view_rows']}})
    return js
