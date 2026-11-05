# ===============================================================
#  远景公告板（billboard / impostor）—— Blender 5.x，三档
#  远处（150 m ~ 2.5 km）上万棵树不能用写实模型：每个树种用高档模型从侧面拍一张透明背景的"照片"，
#  远景树只用几张交叉的竖直面片贴这张图，三角形数和现在的六边形简化树差不多，但看得出是什么树。
#  这一步只准备素材；pipeline 怎么用（中景按植被类型选公告板）以后再定。
#
#  素材：assets_hq/high/species 里的树、灌木（build_species.py）+ assets_hq/high/trees 里的 8 种通用树
#  三档：低 128 像素 × 2 张交叉面片；中 256 × 3；高 512 × 3（同一张照片缩小）
#  照片用均匀的天光 + 一点正面阳光拍，颜色接近素材本身的颜色（网页端再打光）
#
#  用法：blender -b --factory-startup --python build_billboards.py
#  输出：assets_hq/<档位>/billboards/billboards_export.blend（每个一个 Collection：bb_<素材名>）
#        assets_hq/<档位>/billboards/textures/bb_<素材名>.png
#        assets_hq/billboards_manifest.json、assets_hq/previews/billboards.png
# ===============================================================
import bpy, bmesh, math, os, sys, json
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv

ROOT = os.path.dirname(HERE)
HQ = os.path.join(ROOT, 'assets_hq')
TIERS = {'low': dict(px=128, planes=2), 'medium': dict(px=256, planes=3), 'high': dict(px=512, planes=3)}
SOURCES = [os.path.join(HQ, 'high', 'species', 'trees', 'trees_export.blend'),
           os.path.join(HQ, 'high', 'species', 'shrubs', 'shrubs_export.blend'),
           os.path.join(HQ, 'high', 'trees', 'trees_export.blend')]


def setup_studio(px):
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 32
    sc.render.film_transparent = True
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'          # 不压色调，拍出来的颜色就是贴图颜色
    w = bpy.data.worlds.new('w'); sc.world = w
    w.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1)
    w.node_tree.nodes['Background'].inputs[1].default_value = 0.9
    sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 1.6
    so = bpy.data.objects.new('s', sun); so.rotation_euler = (math.radians(55), 0, math.radians(-25))
    sc.collection.objects.link(so)
    cd = bpy.data.cameras.new('c'); cd.type = 'ORTHO'
    cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam); sc.camera = cam
    return sc, cam


def photograph(ob, sc, cam, path, px):
    """正侧面正交拍一张：图宽 px，高按包围盒的高宽比取 2 的幂（px/2 ~ 2px），素材正好装满画面。
    返回画面代表的宽、高（米）和底边高度——公告板面片按这个尺寸做，照片里的树和原模型一样大"""
    d = ob.dimensions
    wdt = max(d.x, d.y) * 1.04; h = d.z * 1.02
    res_y = int(min(px * 2, max(px // 2, 2 ** round(math.log2(px * h / wdt)))))
    sc.render.resolution_x, sc.render.resolution_y = px, res_y
    # 正交相机的 ortho_scale 对应画面较长的那一边
    if res_y >= px:
        scale = max(h, wdt * res_y / px); vis_h, vis_w = scale, scale * px / res_y
    else:
        scale = max(wdt, h * px / res_y); vis_w, vis_h = scale, scale * res_y / px
    cam.data.ortho_scale = scale
    zmin = min((ob.matrix_world @ Vector(c)).z for c in ob.bound_box)
    xs = [(ob.matrix_world @ Vector(c)).x for c in ob.bound_box]
    ob.location.x -= (min(xs) + max(xs)) / 2                     # 不对称的素材（约书亚树）先挪到画面正中
    cam.location = (0, -100, zmin + vis_h / 2)
    cam.rotation_euler = (math.radians(90), 0, 0)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return vis_w, vis_h, zmin


def billboard_mesh(name, w, h, z0, planes):
    """planes 张绕竖轴均匀转开的竖直面片，宽 w、高 h，底边在 z0；UV 0~1"""
    bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap')
    for k in range(planes):
        a = math.pi * k / planes
        dx, dy = math.cos(a) * w / 2, math.sin(a) * w / 2
        vs = [bm.verts.new(p) for p in ((-dx, -dy, z0), (dx, dy, z0), (dx, dy, z0 + h), (-dx, -dy, z0 + h))]
        f = bm.faces.new(vs)
        for loop, c in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
            loop[uv].uv = c
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set('color', np.ones(len(me.vertices) * 4, np.float32))
    me.color_attributes.active_color = attr
    return me


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc, cam = setup_studio(512)
    shots = []
    tmp = os.path.join(HQ, 'previews', '_bb')
    os.makedirs(tmp, exist_ok=True)
    for path in SOURCES:
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.collections = list(src.collections)
        for c in dst.collections:
            ob = c.objects[0]
            sc.collection.objects.link(ob)
            for o in sc.collection.objects:
                if o.type == 'MESH': o.hide_render = o is not ob
            out = os.path.join(tmp, f'{c.name}.png')
            w, h, z0 = photograph(ob, sc, cam, out, 512)
            shots.append((c.name, out, w, h, z0))
            sc.collection.objects.unlink(ob)
            print('  拍好', c.name)
    man = {}
    for tier, q in TIERS.items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        base = os.path.join(HQ, tier, 'billboards'); tex = os.path.join(base, 'textures')
        colls = []
        for name, path, w, h, z0 in shots:
            im = bpy.data.images.load(path); W, H = im.size
            px = np.empty(W * H * 4, np.float32); im.pixels.foreach_get(px)
            rgba = px.reshape(H, W, 4)[::-1].copy(); bpy.data.images.remove(im)
            if q['px'] != 512:                                           # 缩小（4 个像素取平均，透明度一起平均）
                f = 512 // q['px']
                rgba = rgba[:H - H % f, :W - W % f].reshape(H // f, f, W // f, f, 4).mean(axis=(1, 3))
            img = hv.make_image(f'bb_{name}', rgba, tex)
            mat = hv.export_material(f'bb_{name}', img, True)
            me = billboard_mesh(f'bb_{name}', w, h, z0, q['planes']); me.materials.append(mat)
            c = bpy.data.collections.new(f'bb_{name}'); bpy.context.scene.collection.children.link(c)
            c.objects.link(bpy.data.objects.new(f'bb_{name}', me)); colls.append(c)
            man.setdefault(f'bb_{name}', {'name': f'bb_{name}', 'source': name, 'width_m': round(w, 2), 'height_m': round(h, 2),
                                         'tiers': {}})['tiers'][tier] = {'px': q['px'], 'planes': q['planes'], 'triangles': 2 * q['planes']}
        os.makedirs(base, exist_ok=True)
        bpy.data.libraries.write(os.path.join(base, 'billboards_export.blend'), set(colls), fake_user=True)
    with open(os.path.join(HQ, 'billboards_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'assets': list(man.values())}, f, ensure_ascii=False, indent=1)
    # 预览：高档照片拼成一张（灰底）
    tiles = []
    for name, path, w, h, z0 in shots:
        im = bpy.data.images.load(path); W, H = im.size
        px = np.empty(W * H * 4, np.float32); im.pixels.foreach_get(px); bpy.data.images.remove(im)
        a = px.reshape(H, W, 4)
        t = np.full((1024, 512, 4), 1.0, np.float32); t[..., :3] = 0.62
        hh = min(H, 1024); t[:hh, :W, :3] = a[:hh, :, :3] * a[:hh, :, 3:] + 0.62 * (1 - a[:hh, :, 3:])
        tiles.append(t[::4, ::4])
    cols = 12; rows = (len(tiles) + cols - 1) // cols
    grid = np.ones((rows * 256, cols * 128, 4), np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols); grid[(rows - 1 - r) * 256:(rows - r) * 256, c * 128:(c + 1) * 128] = t
    g = bpy.data.images.new('g', grid.shape[1], grid.shape[0]); g.pixels.foreach_set(grid.ravel())
    g.filepath_raw = os.path.join(HQ, 'previews', 'billboards.png'); g.file_format = 'PNG'; g.save()
    for name, path, *_ in shots:
        os.remove(path)
    os.rmdir(tmp)
    print('公告板', len(shots), '个')


if __name__ == '__main__':
    main()
