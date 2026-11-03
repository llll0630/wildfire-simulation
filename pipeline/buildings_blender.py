# ===============================================================
#  真实建筑、道路（Blender 端）：把 buildings_data.py 取到的数据放到地形上
#  由 terrain_blender.build 调用（pipeline.py --terrain ... --buildings buildings.npz）
#
#  做法（轮廓和位置真实，细节估计）：
#    建筑：轮廓直接挤出，平屋顶；墙往地下多埋 0.5 m，斜坡上不悬空；地形不压平
#          屋顶高度 = 轮廓下地面平均高度 + 建筑高度（至少比最高的地面高 2.5 m）
#    道路：按等级给宽度，做成贴着地形网格的条带；路面颜色同时烘焙进近景地面贴图（overlay_ground）
#    高度都按真实网格插值（Surface），不按地形函数：网格之间的地方才不会被地面盖住
# ===============================================================
import math, zlib
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
PATH_RGB = {'dirt': (0.30, 0.23, 0.15), 'paved': (0.33, 0.33, 0.32)}
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


def build_buildings(bld, surf, coll):
    """全部建筑合成一个网格：墙 + 平屋顶。返回每栋的信息（给 scene.json 用）和三角形数"""
    verts, faces, colors, info = [], [], [], []
    tris = 0
    for b in bld.buildings:
        p = np.array(b['poly'], dtype=float)
        if len(p) < 3:
            continue
        gz = surf(p[:, 0], p[:, 1])
        ground = float(np.mean(gz))
        roof = max(ground + b['height'], float(gz.max()) + ROOF_MIN)
        bottom = float(gz.min()) - WALL_SINK
        k = zlib.crc32(b['id'].encode())                # 每栋楼的颜色固定，重新生成也一样
        wall = WALL_RGB[k % len(WALL_RGB)]; roof_c = ROOF_RGB[(k // 7) % len(ROOF_RGB)]
        n = len(p)
        o = len(verts)
        for a in range(n):                              # 墙：每条边单独 4 个点，平面着色
            (x0, y0), (x1, y1) = p[a], p[(a + 1) % n]
            verts += [(x0, y0, bottom), (x1, y1, bottom), (x1, y1, roof), (x0, y0, roof)]
            shade = 0.85 + 0.15 * math.cos(math.atan2(y1 - y0, x1 - x0) - 0.6)      # 各面深浅不同，看得出体积
            colors += [tuple(v * shade for v in wall)] * 4
            faces.append((o + 4 * a, o + 4 * a + 1, o + 4 * a + 2, o + 4 * a + 3))
        o = len(verts)
        verts += [(x, y, roof) for x, y in p]
        colors += [roof_c] * n
        faces.append(tuple(range(o, o + n)))
        tris += 2 * n + n - 2
        c = p.mean(axis=0)
        info.append({**{k_: b.get(k_) for k_ in ('id', 'type', 'name', 'height', 'height_source', 'data_source')},
                     'center': (float(c[0]), float(c[1])), 'ground': round(ground, 2), 'roof': round(roof, 2)})
    if verts:
        _mesh('buildings', verts, faces, colors, coll)
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


def build_roads(bld, surf, coll):
    verts, faces, colors = [], [], []
    for r in sorted(bld.roads, key=lambda r: (r['kind'] == 'road', r['width'])):
        rib = _ribbon(r['line'], r['width'])
        if rib is None:
            continue
        lx, ly, rx, ry = rib
        # 等级高（宽）的路略高一点，交叉口不会闪
        lift = PATH_LIFT if r['kind'] == 'path' else ROAD_LIFT + 0.004 * r['width']
        lift = lift + LIFT_PER_M * np.maximum(np.maximum(np.abs((lx + rx) / 2), np.abs((ly + ry) / 2)) - tc.NEAR_HALF, 0)
        lz = surf(lx, ly) + lift; rz = surf(rx, ry) + lift
        if r['kind'] == 'path':
            rgb = PATH_RGB['paved' if (r.get('surface') or '') in bd.PAVED_SURFACES else 'dirt']
        else:
            rgb = ROAD_RGB
        o = len(verts); m = len(lx)
        verts += list(zip(lx.tolist(), ly.tolist(), lz.tolist())) + list(zip(rx.tolist(), ry.tolist(), rz.tolist()))
        colors += [rgb] * (2 * m)
        faces += [(o + k, o + k + 1, o + m + k + 1, o + m + k) for k in range(m - 1)]
    if verts:
        _mesh('roads', verts, faces, colors, coll)
    return 2 * len(faces)


def overlay_ground(bld, x, y, rgb, tex):
    """近景地面烘焙图：路、小路、铺装、建筑底下换成对应颜色 / 贴图（x、y、rgb 是烘焙图每个像素，sRGB）"""
    c = bld.classes(x, y)
    out = rgb.copy()
    out[c == bd.PAVED] = tex['mat_urban'][c == bd.PAVED]
    out[c == bd.BUILDING] = tex['mat_urban'][c == bd.BUILDING] * 0.8
    out[c == bd.PATH] = tex['mat_bare'][c == bd.PATH] * 0.9
    road = c == bd.ROAD
    out[road] = np.power(np.array(ROAD_RGB), 1 / 2.2) * (0.9 + 0.2 * tex['mat_urban'][road].mean(axis=-1, keepdims=True))
    return out


def build(bld, surf, coll):
    """建筑和道路网格。返回 (每栋建筑的信息, 统计)"""
    info, b_tris = build_buildings(bld, surf, coll)
    r_tris = build_roads(bld, surf, coll)
    stats = {'status': bld.status, 'release': bld.meta.get('release'), 'buildings': len(info),
             'roads': len(bld.roads), 'triangles': {'buildings': b_tris, 'roads': r_tris},
             'height_source': {s: sum(1 for b in info if b['height_source'] == s) for s in ('height', 'floors', 'default')}}
    return info, stats
