# ===============================================================
#  真实建筑、道路（Blender 端）：把 buildings_data.py 取到的数据放到地形上
#  由 terrain_blender.build 调用（pipeline.py --terrain ... --buildings buildings.npz）
#
#  做法（轮廓和位置真实，细节估计）：
#    建筑：轮廓直接挤出；墙往地下多埋 0.5 m，斜坡上不悬空；地形不压平
#          外观按建筑类别（asset_use.building_look）：墙用立面模块，接近长方形的小住宅加坡屋顶，其余平屋顶
#          屋顶高度 = 轮廓下地面平均高度 + 建筑高度（至少比最高的地面高 2.5 m）
#    道路：按等级给宽度，做成贴着地形网格的条带；路面颜色同时烘焙进近景地面贴图（overlay_ground）
#    高度都按真实网格插值（Surface），不按地形函数：网格之间的地方才不会被地面盖住
# ===============================================================
import math, os, zlib
import numpy as np
import bpy
import buildings_data as bd
import terrain_core as tc

ROAD_LIFT = 0.10             # 路面比地形网格高多少（米）
PATH_LIFT = 0.06
LIFT_PER_M = 0.0015          # 近景以外中景网格粗，离得越远抬得越高
WALL_SINK = 0.5              # 墙往地下多埋多少
ROOF_MIN = 2.5               # 屋顶至少比轮廓下最高的地面高多少
NEAR_SAMPLE = 2.5            # 道路沿线取点间距：近景内和近景网格一样
FAR_SAMPLE = 10.0            # 近景外（中景网格 10 m 起）
ASSET_HALF = 150.0

ROAD_RGB = (0.055, 0.055, 0.06)
PATH_RGB = {'dirt': (0.30, 0.23, 0.15), 'paved': (0.33, 0.33, 0.32), 'gravel': (0.36, 0.34, 0.31)}
UNPAVED_ROAD = {'unpaved': 'dirt', 'dirt': 'dirt', 'earth': 'dirt', 'ground': 'dirt', 'mud': 'dirt', 'sand': 'dirt',
                'gravel': 'gravel', 'fine_gravel': 'gravel', 'compacted': 'gravel', 'pebblestone': 'gravel'}
ROOF_RGB = ((0.16, 0.14, 0.13), (0.22, 0.12, 0.09), (0.12, 0.13, 0.15), (0.25, 0.24, 0.22))
WALL_RGB = ((0.62, 0.58, 0.52), (0.55, 0.50, 0.44), (0.68, 0.66, 0.62), (0.48, 0.40, 0.34), (0.60, 0.55, 0.47))


class Surface:
    """按真实网格双线性插值高度：近景（±110 m，2.5 m 一格）以内用近景网格，以外用中景网格"""
    def __init__(self, near_axis, near_z, mid_axis, mid_z):
        self.na, self.nz = near_axis, near_z.reshape(len(near_axis), len(near_axis))
        self.ma, self.mz = mid_axis, mid_z.reshape(len(mid_axis), len(mid_axis))

    @staticmethod
    def _bl(ax, z, x, y):
        i = np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2)
        j = np.clip(np.searchsorted(ax, y) - 1, 0, len(ax) - 2)
        tx = np.clip((x - ax[i]) / (ax[i + 1] - ax[i]), 0, 1); ty = np.clip((y - ax[j]) / (ax[j + 1] - ax[j]), 0, 1)
        return (z[j, i] * (1 - tx) * (1 - ty) + z[j, i + 1] * tx * (1 - ty)
                + z[j + 1, i] * (1 - tx) * ty + z[j + 1, i + 1] * tx * ty)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        h = self.na[-1]
        inner = (np.abs(x) <= h) & (np.abs(y) <= h)
        return np.where(inner, self._bl(self.na, self.nz, x, y), self._bl(self.ma, self.mz, x, y))


def _material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    p = nt.nodes['Principled BSDF']; p.inputs['Roughness'].default_value = 0.9
    vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'base'
    nt.links.new(vc.outputs['Color'], p.inputs['Base Color'])
    return m


def _mesh(name, verts, faces, colors, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate(clean_customdata=False)
    me.update()
    attr = me.color_attributes.new('base', 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set('color', np.asarray([(*c, 1.0) for c in colors], dtype=np.float32).ravel())
    me.color_attributes.active_color_name = 'base'
    me.materials.append(_material(name + '_mat'))
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _tex_material(name, img_path, normal_path=None, rough=0.85):
    """贴图材质（素材库里的外墙、屋顶、立面模块、路带）；同名的只建一次"""
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    m = bpy.data.materials.new(name); N, L = m.node_tree.nodes, m.node_tree.links
    bs = N['Principled BSDF']; bs.inputs['Roughness'].default_value = rough
    tx = N.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(img_path, check_existing=True)
    L.new(tx.outputs['Color'], bs.inputs['Base Color'])
    if normal_path and os.path.exists(normal_path):
        ni = N.new('ShaderNodeTexImage'); ni.image = bpy.data.images.load(normal_path, check_existing=True)
        ni.image.colorspace_settings.name = 'Non-Color'
        nm = N.new('ShaderNodeNormalMap'); L.new(ni.outputs['Color'], nm.inputs['Color']); L.new(nm.outputs['Normal'], bs.inputs['Normal'])
    return m


class _Slots:
    """一个网格里用到的材质：按 (种类, 名字) 编号"""
    def __init__(self, lib, folder, tier, prefix):
        self.lib, self.folder, self.tier, self.prefix = lib, folder, tier, prefix
        self.index, self.mats = {}, []

    def __call__(self, kind, name):
        key = (kind, name)
        if key not in self.index:
            rel = self.lib[kind][name]['files'][self.tier]
            path = os.path.join(self.folder, rel)
            self.mats.append(_tex_material(f'{self.prefix}_{kind}_{name}_{self.tier}', path, path.replace('.jpg', '_normal.jpg')))
            self.index[key] = len(self.mats) - 1
        return self.index[key]


def _obb(p):
    """轮廓的最小外接矩形：(中心 x, 中心 y, 长边方向角, 长, 宽)"""
    pts = sorted(map(tuple, p))
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for q in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(q)
    for q in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0: hi.pop()
        hi.append(q)
    hull = np.array(lo[:-1] + hi[:-1])
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        ang = math.atan2(e[1], e[0]); c, s_ = math.cos(-ang), math.sin(-ang)
        r = hull @ np.array([[c, s_], [-s_, c]])
        mn, mx = r.min(axis=0), r.max(axis=0); area = float(np.prod(mx - mn))
        if best is None or area < best[0]:
            ctr = (mn + mx) / 2; cx = ctr[0] * math.cos(ang) - ctr[1] * math.sin(ang); cy = ctr[0] * math.sin(ang) + ctr[1] * math.cos(ang)
            w, h = mx - mn
            best = (area, cx, cy, ang if w >= h else ang + math.pi / 2, max(w, h), min(w, h))
    return best[1:]


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def build_buildings(bld, surf, coll, tier='low'):
    """全部建筑合成一个网格：墙用立面模块（底层、上面各一种，按建筑类别选，见 asset_use.building_look），
    接近长方形的小住宅加坡屋顶（roof_shapes），其余平屋顶。返回每栋的信息（给 scene.json 用）和三角形数"""
    import json, roof_shapes as rs, asset_use as au, quality as qa
    folder = qa.buildings_dir()
    with open(os.path.join(folder, 'buildings_manifest.json'), encoding='utf-8') as f:
        lib = json.load(f)
    slot = _Slots(lib, folder, tier, 'bld')
    V, F, M, UV, info = [], [], [], [], []
    def face(pts, uvs, mat):
        o = len(V); V.extend(pts); UV.extend(uvs); F.append(tuple(range(o, o + len(pts)))); M.append(mat)
    for b in bld.buildings:
        p = np.array(b['poly'], dtype=float)
        if len(p) < 3:
            continue
        gz = surf(p[:, 0], p[:, 1])
        ground = float(np.mean(gz))
        top = max(ground + b['height'], float(gz.max()) + ROOF_MIN)
        bottom = float(gz.min()) - WALL_SINK
        look = au.building_look(b)
        pitched = None
        if look['pitched'] and len(p) <= 16:
            cx, cy, ang, L, W = _obb(p)
            if _poly_area(p) <= au.PITCH_MAX_AREA and _poly_area(p) / max(L * W, 1e-6) >= au.PITCH_MIN_FILL and W >= 3:
                pitch = 0.45; rise = W / 2 * pitch
                eave = max(ground + 2.6, top - rise * 0.6)
                pitched = (cx, cy, ang, L, W, pitch, eave)
        eave = pitched[6] if pitched else top
        # 墙：每条边 1 ~ 2 段（底层 + 上面几层），立面模块按真实宽度、层高铺 UV
        floor_z = ground + lib['floor_m']
        bands = [(bottom, min(floor_z, eave), look['ground_module'])]
        if eave > floor_z + 0.5:
            bands.append((floor_z, eave, look['upper_module']))
        s0 = 0.0
        n = len(p)
        for k in range(n):
            (x0, y0), (x1, y1) = p[k], p[(k + 1) % n]
            le = math.hypot(x1 - x0, y1 - y0)
            for za, zb, mod in bands:
                if mod and mod in lib['modules']:
                    mi = slot('modules', mod); tw = lib['modules'][mod]['width_m']; th = lib['floor_m']
                else:
                    mi = slot('walls', look['wall']); tw = lib['walls'][look['wall']]['tile_m']; th = lib['walls'][look['wall']]['tile_h_m']
                u0, u1 = s0 / tw, (s0 + le) / tw; va, vb = (za - ground) / th, (zb - ground) / th
                face([(x0, y0, za), (x1, y1, za), (x1, y1, zb), (x0, y0, zb)], [(u0, va), (u1, va), (u1, vb), (u0, vb)], mi)
            s0 += le
        roof = look['roof']
        rm_i = slot('roofs', roof); rt = lib['roofs'][roof]['tile_m']
        if pitched:
            cx, cy, ang, L, W, pitch, _ = pitched
            kind = 'gable' if L / W > 1.3 else 'hip'
            rv, rf, rmi = rs.roof_mesh(kind, L, W, pitch, overhang=0.35)
            c, s_ = math.cos(ang), math.sin(ang)
            wall_i = slot('walls', look['wall']); wt = lib['walls'][look['wall']]['tile_m']
            for fv, fm in zip(rf, rmi):
                loc = [rv[i] for i in fv]
                pts = [(cx + x * c - y * s_, cy + x * s_ + y * c, eave + z) for x, y, z in loc]
                if fm == 0:       # 坡面：沿屋脊方向是 x，沿坡面量长度是 v
                    uvs = [(x / rt, math.hypot(y, z) / rt) for x, y, z in loc]
                    face(pts, uvs, rm_i)
                else:             # 山墙三角：墙面材质
                    uvs = [(y / wt, z / wt) for x, y, z in loc]
                    face(pts, uvs, wall_i)
        else:
            face([(x, y, top) for x, y in p], [(x / rt, y / rt) for x, y in p], rm_i)
        c_ = p.mean(axis=0)
        info.append({**{k_: b.get(k_) for k_ in ('id', 'type', 'name', 'height', 'height_source', 'data_source')},
                     'center': (float(c_[0]), float(c_[1])), 'ground': round(ground, 2), 'roof': round(top, 2),
                     'look': {'wall': look['wall'], 'roof': roof, 'pitched': bool(pitched)}})
    tris = 0
    if V:
        me = bpy.data.meshes.new('buildings'); me.from_pydata(V, [], F); me.update()
        for poly, m in zip(me.polygons, M): poly.material_index = m
        uv = me.uv_layers.new(name='UVMap')
        loops_v = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', loops_v)
        uv.data.foreach_set('uv', np.array(UV, np.float32)[loops_v].ravel())
        for m in slot.mats: me.materials.append(m)
        coll.objects.link(bpy.data.objects.new('buildings', me))
        tris = sum(len(f) - 2 for f in F)
    return info, tris


def _ribbon(line, width):
    """折线 → 取点后的左右两条边。近景内每 2.5 m 一个点，近景外 10 m（保留原来的拐点）"""
    p = np.array(line, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    p = np.vstack([p[:1], p[1:][seg > 0.05]])
    if len(p) < 2:
        return None
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    ts, cur = [0.0], 0.0
    while cur < s[-1]:
        x_, y_ = np.interp(cur, s, p[:, 0]), np.interp(cur, s, p[:, 1])
        cur += NEAR_SAMPLE if max(abs(x_), abs(y_)) < tc.NEAR_HALF + 10 else FAR_SAMPLE
        ts.append(min(cur, s[-1]))
    ts = np.unique(np.concatenate([ts, s]))
    x = np.interp(ts, s, p[:, 0]); y = np.interp(ts, s, p[:, 1])
    d = np.zeros((len(ts), 2))
    step = np.column_stack([np.diff(x), np.diff(y)])
    d[:-1] += step; d[1:] += step                       # 每个点的方向 = 前后两段的平均
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    nrm = np.column_stack([-d[:, 1], d[:, 0]]) * width / 2
    return x - nrm[:, 0], y - nrm[:, 1], x + nrm[:, 0], y + nrm[:, 1]


def build_roads(bld, surf, coll, tier='low'):
    """道路网格：贴着地形的条带，按道路等级、路面材质用路带贴图（asset_use.road_strip）；
    UV 横向 0 ~ 1 = 路的整个宽度，纵向按真实长度每 12 m 重复一次"""
    import json, asset_use as au, quality as qa
    folder = qa.roads_dir()
    with open(os.path.join(folder, 'roads_manifest.json'), encoding='utf-8') as f:
        lib = json.load(f)
    slot = _Slots({'roads': lib['roads']}, folder, tier, 'road')
    verts, faces, mats, uvs = [], [], [], []
    for r in sorted(bld.roads, key=lambda r: (r['kind'] == 'road', r['width'])):
        rib = _ribbon(r['line'], r['width'])
        if rib is None:
            continue
        lx, ly, rx, ry = rib
        # 等级高（宽）的路略高一点，交叉口不会闪
        lift = PATH_LIFT if r['kind'] == 'path' else ROAD_LIFT + 0.004 * r['width']
        lift = lift + LIFT_PER_M * np.maximum(np.maximum(np.abs((lx + rx) / 2), np.abs((ly + ry) / 2)) - tc.NEAR_HALF, 0)
        lz = surf(lx, ly) + lift; rz = surf(rx, ry) + lift
        mi = slot('roads', au.road_strip(r))
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
        sv = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))]) / lib['length_m']
        o = len(verts); m = len(lx)
        verts += list(zip(lx.tolist(), ly.tolist(), lz.tolist())) + list(zip(rx.tolist(), ry.tolist(), rz.tolist()))
        uvs += [(0.0, v) for v in sv.tolist()] + [(1.0, v) for v in sv.tolist()]
        faces += [(o + k, o + k + 1, o + m + k + 1, o + m + k) for k in range(m - 1)]
        mats += [mi] * (m - 1)
    if verts:
        me = bpy.data.meshes.new('roads'); me.from_pydata(verts, [], faces); me.validate(clean_customdata=False); me.update()
        for poly, mi in zip(me.polygons, mats): poly.material_index = mi
        uv = me.uv_layers.new(name='UVMap')
        loops_v = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', loops_v)
        uv.data.foreach_set('uv', np.array(uvs, np.float32)[loops_v].ravel())
        for mm in slot.mats: me.materials.append(mm)
        coll.objects.link(bpy.data.objects.new('roads', me))
    return 2 * len(faces)


def overlay_ground(bld, x, y, rgb, sample):
    """近景地面烘焙图：路、小路、铺装、建筑底下换成对应贴图（x、y、rgb 是烘焙图每个像素，sRGB）
    sample(材质名, 选中的像素) → 那些像素的贴图颜色。铺装处 Overture 已经细分过（停车场、地砖……）就不再覆盖"""
    c = bld.classes(x, y)
    g = bld.ground(x, y)
    out = rgb.copy()
    paved = (c == bd.PAVED) & (g == 0)
    for sel, key, k in ((paved, 'concrete', 1.0), (c == bd.BUILDING, 'concrete', 0.8), (c == bd.PATH, 'dirt', 0.9),
                        (c == bd.ROAD, 'asphalt', 1.0)):
        if sel.any():
            out[sel] = sample(key, sel) * k
    return out


def build(bld, surf, coll, tier='low'):
    """建筑和道路网格（贴图按画质档位取）。返回 (每栋建筑的信息, 统计)"""
    info, b_tris = build_buildings(bld, surf, coll, tier)
    r_tris = build_roads(bld, surf, coll, tier)
    stats = {'status': bld.status, 'release': bld.meta.get('release'), 'buildings': len(info),
             'roads': len(bld.roads), 'triangles': {'buildings': b_tris, 'roads': r_tris},
             'height_source': {s: sum(1 for b in info if b['height_source'] == s) for s in ('height', 'floors', 'default')}}
    return info, stats
