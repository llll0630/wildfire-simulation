# ===============================================================
#  导出用素材转换脚本 —— Blender 5.x / 4.x
#  GLB 格式看不懂 Blender 的程序化材质（用节点公式算出来的颜色），直接导出会全部变白。
#  这个脚本把 assets/ 里的素材转换成 GLB 能保留颜色的版本，另存为 *_export.blend，原素材不动：
#
#    树、灌木、草丛  →  颜色烘焙到模型顶点上（顶点颜色）
#                       assets/trees/trees_export.blend 等
#    地面材质        →  烘焙成一张可以无缝重复铺的贴图
#                       assets/ground/ground_materials_export.blend + textures/*.png
#                       assets/ground/ground_export_manifest.json（每张贴图代表多少米）
#    房子            →  展开 UV，把所有材质烘焙成一张贴图
#                       assets/houses/houses_export.blend + textures/*.png
#
#  素材有改动时（重新运行了其他 build_*.py），再运行一次这个脚本即可。
#  用法：blender --background --factory-startup --python build_export_assets.py
#        只转换某几类：在最后加 -- ground houses（可选 trees shrubs grass ground houses）
# ===============================================================
import bpy, os, json
import numpy as np

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = ''
if not HERE or not os.path.isdir(HERE):   # 在 Blender 文本编辑器里运行时 __file__ 不可靠
    HERE = r'D:\cs research\AR\output\asset_builders'
ASSET_DIR = os.path.join(os.path.dirname(HERE), 'assets')

# 地面贴图：每张贴图代表多少米（重复铺满整个场景）、分辨率、粗糙度
GROUND_TILES = {
    'mat_grass':    {'tile_m': 8.0,  'px': 2048, 'roughness': 0.95},
    'mat_litter':   {'tile_m': 4.0,  'px': 2048, 'roughness': 0.9},
    'mat_farmland': {'tile_m': 7.6,  'px': 2048, 'roughness': 0.9},    # 10 条垄，垄在贴图边缘能接上
    'mat_urban':    {'tile_m': 45.0, 'px': 4096, 'roughness': 0.8, 'seam': 0},   # 3 × 3 个区块，边缘正好是区块边界，不用融合
    'mat_water':    {'tile_m': 8.0,  'px': 2048, 'roughness': 0.08},
    'mat_bare':     {'tile_m': 8.0,  'px': 2048, 'roughness': 0.95},
}
SEAM = 0.15            # 贴图边缘渐变融合的宽度（占贴图边长的比例），用来消除接缝
HOUSE_PX = 2048

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 4
scene.render.bake.margin = 8


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)


def select_only(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def flat_material(name, base_color_node_fn, roughness):
    """新建一个只用 Principled BSDF 的材质，底色由 base_color_node_fn(nodes) 提供"""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.use_backface_culling = False                  # 导出成双面材质（草叶、树叶从背面看也能看到）
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = roughness
    nt.links.new(base_color_node_fn(nt.nodes), bsdf.inputs['Base Color'])
    return m


def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()


# ---------------- 1. 树、灌木、草丛：烘焙顶点颜色 ----------------
def convert_vegetation(folder):
    clear_scene()
    with open(os.path.join(ASSET_DIR, folder, f'{folder}_manifest.json'), encoding='utf-8') as f:
        names = [a['name'] for a in json.load(f)['assets']]
    with bpy.data.libraries.load(os.path.join(ASSET_DIR, folder, f'{folder}.blend'), link=False) as (src, dst):
        dst.collections = names
    out_mat = None
    for c in dst.collections:
        ob = c.objects[0]
        scene.collection.objects.link(ob)          # 烘焙需要物体在场景里
        me = ob.data
        attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'CORNER')
        me.color_attributes.active_color = attr
        select_only(ob)
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target='VERTEX_COLORS')
        if out_mat is None:                         # 这一类素材共用一个读顶点颜色的材质
            def vc(nodes):
                n = nodes.new('ShaderNodeVertexColor'); n.layer_name = 'Col'
                return n.outputs['Color']
            out_mat = flat_material(f'{folder}_vertex_color', vc, 0.85)
        me.materials.clear()
        me.materials.append(out_mat)
        scene.collection.objects.unlink(ob)
        print(f'  {c.name}：顶点颜色已烘焙')
    path = os.path.join(ASSET_DIR, folder, f'{folder}_export.blend')
    bpy.data.libraries.write(path, set(dst.collections), fake_user=True)
    print('已保存：', path)


# ---------------- 2. 地面：烘焙成无缝贴图 ----------------
def make_seamless(px, n, overlap):
    """px：烘焙出的 (n + overlap) 见方的图；返回 n 见方、左右上下都能接上的图。
    做法：贴图的开头一段，用"贴图后面那段"渐变过渡过去，这样首尾的颜色正好接上。"""
    w = np.linspace(0.0, 1.0, overlap)[:, None]
    out = px[:n + overlap, :n + overlap].copy()
    # 先处理行方向（y）
    out[:overlap] = out[n:n + overlap] * (1 - w[:, :, None]) + out[:overlap] * w[:, :, None]
    out = out[:n]
    # 再处理列方向（x）
    wx = w.reshape(1, overlap, 1)
    out[:, :overlap] = out[:, n:n + overlap] * (1 - wx) + out[:, :overlap] * wx
    return out[:, :n]


def convert_ground():
    clear_scene()
    names = list(GROUND_TILES)
    with bpy.data.libraries.load(os.path.join(ASSET_DIR, 'ground', 'ground_materials.blend'), link=False) as (src, dst):
        dst.materials = names
    manifest, export_mats = {}, []
    for mat in dst.materials:
        cfg = GROUND_TILES[mat.name]
        n = cfg['px']; ov = int(n * cfg.get('seam', SEAM)); full = n + ov
        size = cfg['tile_m'] * full / n                         # 烘焙区域比一张贴图大一点，多出来的用来融合接缝
        bpy.ops.mesh.primitive_plane_add(size=size, location=(size / 2, size / 2, 0))
        plane = bpy.context.active_object
        plane.data.materials.append(mat)
        img = bpy.data.images.new(f'_bake_{mat.name}', full, full, float_buffer=True)
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage'); tex.image = img
        mat.node_tree.nodes.active = tex                        # 烘焙结果写进当前选中的贴图节点
        select_only(plane)
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target='IMAGE_TEXTURES')
        mat.node_tree.nodes.remove(tex)

        px = np.array(img.pixels[:], dtype=np.float32).reshape(full, full, 4)
        tile = make_seamless(px, n, ov) if ov else px[:n, :n].copy()
        rgb = np.clip(tile[..., :3], 0, 1)                      # 烘焙结果是线性颜色，存 PNG 前转成 sRGB
        tile[..., :3] = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
        tile[..., 3] = 1.0
        out = bpy.data.images.new(mat.name, n, n)
        out.pixels.foreach_set(tile.ravel())
        save_png(out, os.path.join(ASSET_DIR, 'ground', 'textures', f'{mat.name}.png'))
        bpy.data.images.remove(img)
        bpy.data.objects.remove(plane, do_unlink=True)

        def image_node(nodes, out=out):
            t = nodes.new('ShaderNodeTexImage'); t.image = out
            return t.outputs['Color']
        em = flat_material(f'{mat.name}_export', image_node, cfg['roughness'])
        export_mats.append(em)
        manifest[mat.name] = {'export_material': em.name, 'texture': f'textures/{mat.name}.png', 'tile_m': cfg['tile_m']}
        print(f'  {mat.name}：已烘焙成 {n}×{n} 贴图，每张代表 {cfg["tile_m"]} m')
    path = os.path.join(ASSET_DIR, 'ground', 'ground_materials_export.blend')
    bpy.data.libraries.write(path, set(export_mats), fake_user=True, path_remap='RELATIVE_ALL')
    with open(os.path.join(ASSET_DIR, 'ground', 'ground_export_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('已保存：', path)


# ---------------- 3. 房子：展开 UV，所有材质烘焙成一张贴图 ----------------
def convert_houses():
    clear_scene()
    with open(os.path.join(ASSET_DIR, 'houses', 'houses_manifest.json'), encoding='utf-8') as f:
        names = [a['name'] for a in json.load(f)['assets']]
    with bpy.data.libraries.load(os.path.join(ASSET_DIR, 'houses', 'houses.blend'), link=False) as (src, dst):
        dst.collections = names
    for c in dst.collections:
        ob = c.objects[0]
        scene.collection.objects.link(ob)
        select_only(ob)
        me = ob.data
        if not me.uv_layers:
            me.uv_layers.new(name='UVMap')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
        bpy.ops.object.mode_set(mode='OBJECT')

        img = bpy.data.images.new(f'{c.name}', HOUSE_PX, HOUSE_PX)
        added = []
        for m in me.materials:                                   # 每个材质都放一个指向同一张图的贴图节点
            t = m.node_tree.nodes.new('ShaderNodeTexImage'); t.image = img
            m.node_tree.nodes.active = t
            added.append((m, t))
        bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, target='IMAGE_TEXTURES')
        for m, t in added:
            m.node_tree.nodes.remove(t)
        save_png(img, os.path.join(ASSET_DIR, 'houses', 'textures', f'{c.name}.png'))

        def image_node(nodes, img=img):
            t = nodes.new('ShaderNodeTexImage'); t.image = img
            return t.outputs['Color']
        me.materials.clear()
        me.materials.append(flat_material(f'{c.name}_baked', image_node, 0.8))
        scene.collection.objects.unlink(ob)
        print(f'  {c.name}：已烘焙成 {HOUSE_PX}×{HOUSE_PX} 贴图')
    path = os.path.join(ASSET_DIR, 'houses', 'houses_export.blend')
    bpy.data.libraries.write(path, set(dst.collections), fake_user=True, path_remap='RELATIVE_ALL')
    print('已保存：', path)


import sys
only = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []     # 可以只转换某几类：-- ground houses
for folder in ('trees', 'shrubs', 'grass'):
    if not only or folder in only:
        convert_vegetation(folder)
if not only or 'ground' in only:
    convert_ground()
if not only or 'houses' in only:
    convert_houses()
print('导出用素材全部转换完成')
