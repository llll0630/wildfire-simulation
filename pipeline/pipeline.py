# ===============================================================
#  火灾三维场景生成 Pipeline —— Blender 5.x / 4.x
#  输入 5 个值 → 做决定（pipeline_rules.py）→ 从素材库取素材摆放 → 输出场景
#
#  用法（命令行）：
#    blender --background --factory-startup --python pipeline.py -- --input 输入.json --out 输出文件夹
#  可选参数：
#    --seed 123      指定随机种子，同样的输入 + 同样的种子 = 同样的场景
#    --render        额外渲染一张预览图 preview.png
#    --ground-only   只生成地面，不放树、草、房子（看场景大小用）
#    --glb           用导出用素材（*_export.blend）搭场景，并导出 scene.glb 给网页用
#                    （需要先运行过 asset_builders/build_export_assets.py）
#    --terrain 地形.npz  用真实地形（terrain_data.py 生成）；不加就是原来的平地场景
#                    地形相关代码在 terrain_core.py（摆放、高度）和 terrain_blender.py（搭场景）
#
#  输出（在输出文件夹里）：
#    scene.blend     生成的 Blender 场景
#    scene.json      给网页端的参数（火、烟、天空、初始位置）+ 调试信息
#    preview.png     预览图（加了 --render 才有）
#    scene.glb       给网页用的三维场景（加了 --glb 才有）
# ===============================================================
import bpy, sys, os, json, math, argparse
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pipeline_rules as rules

ASSET_DIR = rules.ASSET_DIR
GROUND_MAT = {'water': 'mat_water', 'bare': 'mat_bare', 'urban': 'mat_urban', 'farmland': 'mat_farmland',
              'grassland': 'mat_grass', 'shrubland': 'mat_grass', 'forest': 'mat_litter'}


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for datas in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for d in list(datas):
            if d.users == 0:
                datas.remove(d)


def make_coll(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


def append_collections(blend, names):
    """从素材库追加需要的 Collection，返回 {名字: Collection}"""
    if not names:
        return {}
    with bpy.data.libraries.load(os.path.join(ASSET_DIR, blend), link=False) as (src, dst):
        dst.collections = sorted(names)
    return {c.name: c for c in dst.collections}


def instance(coll, name, src, x, y, rot, s=1.0, sz=None):
    ob = bpy.data.objects.new(name, None)
    ob.instance_type = 'COLLECTION'; ob.instance_collection = src
    ob.location = (x, y, 0); ob.rotation_euler = (0, 0, rot)
    ob.scale = (s, s, sz if sz is not None else s)
    coll.objects.link(ob)


def scatter(coll, name, src, items):
    """导出模式用：同一款素材的所有副本放进一个几何节点物体里（点云 + 每个点放一个实例）。
    导出 GLB 时会变成一份模型 + 一串位置（EXT_mesh_gpu_instancing），网页加载和渲染都快得多"""
    me = bpy.data.meshes.new(f'_pts_{name}')
    me.from_pydata([(t['x'], t['y'], 0.0) for t in items], [], [])
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
    info.inputs['Object'].default_value = src.objects[0]
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


def build(decision, layout, ground_only=False, export=False):
    """export=True 时用导出用素材（颜色已烘焙成顶点颜色/贴图），用于导出 GLB"""
    clear_scene()
    size = rules.SCENE_SIZE
    suffix = '_export' if export else ''

    # 地面
    c_ground = make_coll('Ground')
    mat_name = GROUND_MAT[decision['ground']]
    if export:
        with open(os.path.join(ASSET_DIR, 'ground', 'ground_export_manifest.json'), encoding='utf-8') as f:
            gcfg = json.load(f)[mat_name]
        mat_name = gcfg['export_material']
    with bpy.data.libraries.load(os.path.join(ASSET_DIR, 'ground', f'ground_materials{suffix}.blend'), link=False) as (src, dst):
        dst.materials = [mat_name]
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = 'Ground'
    for c in ground.users_collection: c.objects.unlink(ground)
    c_ground.objects.link(ground)
    ground.data.materials.append(dst.materials[0])
    if export:                                      # 贴图按实际大小重复铺满地面
        k = size / gcfg['tile_m']
        for d in ground.data.uv_layers.active.data:
            d.uv = (d.uv[0] * k, d.uv[1] * k)

    if ground_only:
        return

    # 素材：只追加这次用到的
    veg = {}
    for k in ('trees', 'shrubs', 'grass'):          # 每类素材在自己的文件夹里：assets/trees/trees.blend 等
        veg.update(append_collections(os.path.join(k, f'{k}{suffix}.blend'), {i['name'] for i in layout[k]}))
    houses = append_collections(os.path.join('houses', f'houses{suffix}.blend'), {h['name'] for h in layout['houses']})

    c_houses, c_trees = make_coll('Houses'), make_coll('Trees')
    c_shrubs, c_grass = make_coll('Shrubs'), make_coll('Grass')
    for h in layout['houses']:
        instance(c_houses, h['name'], houses[h['name']], h['x'], h['y'], h['rot'])
    if export:                                      # 导出模式：同款素材合并成一个实例组
        for key, coll in (('trees', c_trees), ('shrubs', c_shrubs), ('grass', c_grass)):
            for name in sorted({t['name'] for t in layout[key]}):
                scatter(coll, name, veg[name], [t for t in layout[key] if t['name'] == name])
    else:
        for t in layout['trees']:
            instance(c_trees, t['name'], veg[t['name']], t['x'], t['y'], t['rot'], t['s'], t['sz'])
        for t in layout['shrubs']:
            instance(c_shrubs, t['name'], veg[t['name']], t['x'], t['y'], t['rot'], t['s'])
        for t in layout['grass']:
            instance(c_grass, t['name'], veg[t['name']], t['x'], t['y'], t['rot'], t['s'])

    # 火的位置标记（只在 Blender 里看，不导出）：一个红色圆圈
    if layout['fire'] and not export:
        f = layout['fire']
        bpy.ops.mesh.primitive_circle_add(vertices=48, radius=f['radius'], location=(*f['center'], 0.3))
        ring = bpy.context.active_object
        ring.name = '_fire_marker'
        bpy.ops.object.convert(target='CURVE')
        ring.data.bevel_depth = 0.25
        m = bpy.data.materials.new('_fire_marker'); m.use_nodes = True
        em = m.node_tree.nodes.new('ShaderNodeEmission')
        em.inputs['Color'].default_value = (1.0, 0.15, 0.0, 1); em.inputs['Strength'].default_value = 5
        m.node_tree.links.new(em.outputs[0], m.node_tree.nodes['Material Output'].inputs['Surface'])
        ring.data.materials.append(m)


def export_glb(path, terrain=False):
    """导出 GLB：不含相机和灯光（网页端自己设置），贴图转成 JPEG 减小文件。
    地形场景的中景远景用顶点颜色上色，要导出顶点颜色"""
    extra = {'export_vertex_color': 'ACTIVE', 'export_image_quality': 80} if terrain else {'export_image_quality': 85}
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_cameras=False, export_lights=False,
                              export_image_format='JPEG', export_gpu_instances=True, export_gn_mesh=True, **extra)


def setup_preview(scene_js):
    """预览用的天空、太阳、相机（相机放在用户初始位置稍高处；地形场景放在高空视角）"""
    s = bpy.context.scene
    sun = bpy.data.lights.new('Sun', 'SUN'); sun.energy = 3.0; sun.color = (1.0, 0.93, 0.82)
    so = bpy.data.objects.new('Sun', sun); so.rotation_euler = (math.radians(50), 0, math.radians(35))
    s.collection.objects.link(so)
    w = bpy.data.worlds.new('World'); s.world = w
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.55, 0.7, 0.9, 1)
    cd = bpy.data.cameras.new('Camera'); cd.lens = 28; cd.clip_end = 1000
    cam = bpy.data.objects.new('Camera', cd); s.collection.objects.link(cam); s.camera = cam
    size = rules.SCENE_SIZE
    cam.location = (0, -size * 0.625, size * 0.25)
    target = Vector((0, size * 0.04, 0))
    if 'aerial' in scene_js:                        # three.js 坐标 [x, y, z] → Blender (x, -z, y)
        to_b = lambda v: Vector((v[0], -v[2], v[1]))
        cam.location, target = to_b(scene_js['aerial']['position']), to_b(scene_js['aerial']['look_at'])
        cd.clip_end = 300000
    cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    for eng in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        try: s.render.engine = eng; break
        except TypeError: pass
    s.view_settings.view_transform = 'AgX'
    s.render.resolution_x = 1200; s.render.resolution_y = 675


def frame_viewport():
    """在 Blender 界面里运行时：切到材质预览，并把视角缩放到能看到整个场景"""
    screen = bpy.context.screen
    if screen is None:
        return
    for area in screen.areas:
        if area.type != 'VIEW_3D':
            continue
        space = area.spaces.active
        space.shading.type = 'MATERIAL'
        space.clip_end = 2000
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region:
            with bpy.context.temp_override(area=area, region=region):
                bpy.ops.view3d.view_all()


def generate_terrain(inputs, terrain_path, seed=None, export=False):
    """地形场景：做决定（规则不变）→ 按地形摆放 → 搭场景。返回 (seed, scene_json)"""
    import random, time
    import terrain_core as tc, terrain_blender as tb
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    t0 = time.perf_counter(); timing = {}
    t = tc.Terrain(terrain_path)
    veg, houses = rules.load_manifests()
    p = tc.plan(t, inputs, seed, veg, houses)
    timing['plan'] = round(time.perf_counter() - t0, 2)
    built = tb.build(sys.modules[__name__], t, p, seed, export=export, timing=timing)
    js = tb.scene_json(t, p, seed, built, inputs)
    js['debug']['timing_s'] = timing
    return seed, js


def generate(inputs, seed=None, terrain=None):
    """在当前打开的 Blender 里直接生成场景（粘贴进 Scripting 运行时用）。terrain = 地形 .npz 路径（可选）"""
    if terrain:
        seed, js = generate_terrain(inputs, terrain, seed)
        setup_preview(js)
        frame_viewport()
        print(f"[pipeline] seed={seed} 地形={terrain} 摆放={js['debug']['stats']['placed']} 火={js['fire']['level']}")
        return js
    seed, decision, layout, js = rules.run_rules(inputs, seed)
    build(decision, layout)
    setup_preview(js)
    frame_viewport()
    d = js['debug']
    print(f"[pipeline] seed={seed} 地面={js['ground']} 树={d['counts']['trees']} 灌木={d['counts']['shrubs']} "
          f"草={d['counts']['grass']} 房子={d['counts']['houses']} 火={js['fire']['level']}")
    return js


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--ground-only', action='store_true', help='只生成地面（看场景大小用）')
    ap.add_argument('--glb', action='store_true', help='用导出用素材搭场景，并导出 scene.glb')
    ap.add_argument('--terrain', default=None, help='地形文件 .npz（terrain_data.py 生成）；不加就是平地')
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)          # 渲染输出要用绝对路径，否则会存到别的地方

    with open(a.input, encoding='utf-8') as f:
        inputs = json.load(f)
    if a.terrain:
        main_terrain(a, inputs)
        return
    seed, decision, layout, js = rules.run_rules(inputs, a.seed)
    build(decision, layout, a.ground_only, export=a.glb)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'scene.json'), 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    if a.glb:
        export_glb(os.path.join(a.out, 'scene.glb'))
    setup_preview(js)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, 'scene.blend'))
    if a.render:
        bpy.context.scene.render.filepath = os.path.join(a.out, 'preview.png')
        bpy.ops.render.render(write_still=True)

    d = js['debug']
    print(f"[pipeline] seed={seed} 地面={js['ground']} 树={d['counts']['trees']} 灌木={d['counts']['shrubs']} "
          f"草={d['counts']['grass']} 房子={d['counts']['houses']} 火={js['fire']['level']} "
          f"树冠火={len(js['fire']['burning_trees'])}棵 烟={js['smoke_level']} 天空={js['sky']}")


def main_terrain(a, inputs):
    """命令行：地形场景（--terrain）"""
    import time
    if a.ground_only:
        print('[pipeline] --ground-only 只用于平地场景，地形场景忽略')
    t0 = time.perf_counter()
    seed, js = generate_terrain(inputs, a.terrain, a.seed, export=a.glb)
    os.makedirs(a.out, exist_ok=True)
    if a.glb:
        t1 = time.perf_counter()
        export_glb(os.path.join(a.out, 'scene.glb'), terrain=True)
        js['debug']['timing_s']['export_glb'] = round(time.perf_counter() - t1, 2)
        js['debug']['stats']['glb_mb'] = round(os.path.getsize(os.path.join(a.out, 'scene.glb')) / 1e6, 2)
    js['debug']['timing_s']['total'] = round(time.perf_counter() - t0, 2)
    with open(os.path.join(a.out, 'scene.json'), 'w', encoding='utf-8') as f:
        json.dump(js, f, ensure_ascii=False)
    setup_preview(js)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(a.out, 'scene.blend'))
    if a.render:
        bpy.context.scene.render.filepath = os.path.join(a.out, 'preview.png')
        bpy.ops.render.render(write_still=True)
    d = js['debug']
    print(f"[pipeline] seed={seed} 地形 摆放={d['stats']['placed']} 中景树={d['stats']['mid_trees']} "
          f"火={js['fire']['level']} 用时={d['timing_s']['total']}s")
    print('[pipeline-json] ' + json.dumps({'seed': seed, 'stats': d['stats'], 'checks': d['checks'],
                                           'timing_s': d['timing_s'], 'fire': js['fire']['level']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
