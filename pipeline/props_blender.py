# ===============================================================
#  小物件（Blender 端）：石块、电线杆 + 电线、栅栏、路灯，按真实数据放（规则表在 asset_use.py）
#    石块：按植被类型（裸岩、沙漠……）和 Overture 地面类型（岩石）的密度撒
#    电线杆：沿 Overture 电力线，配电线每 45 m 一根木杆，输电线每 300 m 一座塔；杆之间拉下垂的电线
#    栅栏：沿 Overture barrier 线，一段段接着放（按类别选木栅栏、铁丝网、绿篱、护栏）
#    路灯：主要道路每 35 m 一盏，放在路边
#  只放在 ±150 m（真实素材的范围）以内；素材按画质档位从 assets_hq/<档位>/{props,nature}/ 取
# ===============================================================
import math
import numpy as np
import bpy
import asset_use as au
import quality as qa
import veg_types as vt


def _resample(line, step):
    """折线上每 step 米一个点：(x, y, 方向角)"""
    p = np.array(line, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    if len(p) < 2 or seg.sum() < 1e-3:
        return []
    s = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for d in np.arange(0, s[-1] + 1e-6, step):
        i = min(np.searchsorted(s, d, side='right') - 1, len(seg) - 1)
        t = (d - s[i]) / max(seg[i], 1e-9)
        x, y = p[i] + (p[i + 1] - p[i]) * t
        out.append((float(x), float(y), math.atan2(p[i + 1][1] - p[i][1], p[i + 1][0] - p[i][0])))
    return out


def _inside(x, y, half=au.PROPS_HALF):
    return max(abs(x), abs(y)) <= half


def _wires(spans, name, coll):
    """电线：每一档两端挂点之间 12 段的下垂曲线（3 棱的细管）"""
    verts, faces = [], []
    r = 0.015
    for (a, b) in spans:
        a, b = np.array(a), np.array(b); L = float(np.linalg.norm(b[:2] - a[:2]))
        sag = 0.03 * L
        pts = [a + (b - a) * t - np.array([0, 0, 4 * sag * t * (1 - t)]) for t in np.linspace(0, 1, 13)]
        o = len(verts)
        for q in pts:
            verts += [tuple(q + np.array([0, 0, r])), tuple(q + np.array([r, 0, -r])), tuple(q + np.array([-r, 0, -r]))]
        for k in range(12):
            for j in range(3):
                faces.append((o + 3 * k + j, o + 3 * k + (j + 1) % 3, o + 3 * (k + 1) + (j + 1) % 3, o + 3 * (k + 1) + j))
    if not verts:
        return 0
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    m = bpy.data.materials.get('wire') or bpy.data.materials.new('wire')
    m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.03, 0.03, 0.03, 1)
    me.materials.append(m)
    coll.objects.link(bpy.data.objects.new(name, me))
    return 2 * len(faces)


def place(t, p, quality, base, scatter, mesh_tris, rng):
    """把小物件放进场景。base = pipeline 模块（make_coll）；scatter = terrain_blender.scatter。返回统计"""
    bld = p.get('buildings')
    items = {}                                       # 素材名 → [摆放点]
    def add(name, x, y, rot, s=1.0, z=None):
        items.setdefault(name, []).append({'x': x, 'y': y, 'z': z, 'rot': rot, 's': s})
    stats = {}

    # 石块：按植被类型、Overture 地面类型的密度
    half = au.PROPS_HALF
    dens = np.zeros(len(vt.TYPES))
    for k, v in au.ROCK_VEG.items():
        dens[vt.ID[k]] = v
    tries = int((2 * half) ** 2 * max(max(au.ROCK_VEG.values()), max(au.ROCK_GROUND.values())))
    x = rng_np_uniform(rng, -half, half, tries); y = rng_np_uniform(rng, -half, half, tries)
    d = dens[t.veg(x, y)]
    if bld is not None:
        import buildings_data as bdm
        gd = np.zeros(len(bdm.GROUND))
        for k, v in au.ROCK_GROUND.items():
            gd[bdm.G[k]] = v
        d = np.maximum(d, gd[bld.ground(x, y)])
        d = np.where(bld.blocked(x, y), 0, d)
    keep = np.array([rng.random() for _ in range(tries)]) * max(max(au.ROCK_VEG.values()), max(au.ROCK_GROUND.values())) < d
    reds = {vt.ID['desert_barren'], vt.ID['desert_scrub']}
    for xi, yi, vi in zip(x[keep].tolist(), y[keep].tolist(), t.veg(x[keep], y[keep]).tolist()):
        color = 'rock_red' if vi in reds else 'rock_grey'
        r = rng.random()
        name = f'{color}_outcrop_01' if r < 0.04 else f'{color}_pile' if r < 0.25 else f'{color}_0{1 + int(rng.random() * 3)}'
        add(name, xi, yi, rng.uniform(0, 6.28), rng.uniform(0.7, 1.3))
    stats['rocks'] = int(keep.sum())

    spans = []
    if bld is not None:
        # 电线杆 + 电线
        for ln in bld.power['lines']:
            cls = ln['class']
            step = au.TOWER_SPACING if cls == 'power_line' else au.POLE_SPACING.get(cls, 45.0)
            asset = 'tower_transmission' if cls == 'power_line' else 'pole_utility'
            pts = [q for q in _resample(ln['line'], step) if _inside(q[0], q[1])]
            prev = None
            for k_, (px, py, ang) in enumerate(pts):
                name = 'pole_utility_transformer' if asset == 'pole_utility' and k_ % 6 == 3 else asset
                add(name, px, py, ang)
                z0 = float(t.height(np.array([px]), np.array([py]))[0])
                attach = (20.4 if asset == 'tower_transmission' else 10.55)
                arms = (-6.0, 6.0) if asset == 'tower_transmission' else (-1.05, 0.0, 1.05)
                cur = [(px - ax_ * math.sin(ang), py + ax_ * math.cos(ang), z0 + attach) for ax_ in arms]
                if prev is not None and len(prev) == len(cur):
                    spans += list(zip(prev, cur))
                prev = cur
        # 栅栏
        n_f = 0
        for br in bld.barriers:
            asset = au.FENCE_ASSET.get(br['class'], au.FENCE_DEFAULT)
            seg_len = {'guardrail': 4.0, 'hedge': 2.0, 'fence_chain_link': 3.0, 'fence_wood_privacy': 2.4}.get(asset, 3.0)
            for (fx, fy, ang) in _resample(br['line'], seg_len)[:-1]:
                if _inside(fx, fy):
                    add(asset, fx, fy, ang); n_f += 1
        stats['fence_segments'] = n_f
        # 路灯
        n_l = 0
        for r in bld.roads:
            if r['kind'] != 'road' or r['class'] not in au.STREET_LIGHT_CLASSES:
                continue
            for k_, (lx, ly, ang) in enumerate(_resample(r['line'], au.STREET_LIGHT_SPACING)):
                side = 1 if k_ % 2 else -1
                off = r['width'] / 2 + 1.0
                x_, y_ = lx - side * off * math.sin(ang), ly + side * off * math.cos(ang)
                if _inside(x_, y_) and not bool(bld.classes(np.array([x_]), np.array([y_]))[0] == 5):
                    add('street_light', x_, y_, ang + (math.pi / 2 if side > 0 else -math.pi / 2)); n_l += 1
        stats['street_lights'] = n_l
        stats['poles'] = sum(len(v) for k, v in items.items() if k.startswith(('pole', 'tower')))

    if not items:
        return stats, 0
    coll = base.make_coll('Props')
    names = set(items)
    nat = {n for n in names if n.startswith('rock')}
    srcs = {}
    if nat:
        srcs.update(qa.load(qa.lib_path(quality, 'nature'), nat))
    if names - nat:
        srcs.update(qa.load(qa.lib_path(quality, 'props'), names - nat))
    tris = 0
    for name in sorted(items):
        its = items[name]
        zs = t.height(np.array([i['x'] for i in its]), np.array([i['y'] for i in its]))
        for i, z in zip(its, zs.tolist()):
            i['z'] = z - (0.1 if name.startswith('rock') else 0.0)
        src = srcs[name].objects[0]
        scatter(coll, name, src, its)
        tris += mesh_tris(src.data) * len(its)
    tris += _wires(spans, 'power_wires', coll)
    stats['wire_spans'] = len(spans)
    return stats, tris


def rng_np_uniform(rng, a, b, n):
    return np.array([rng.uniform(a, b) for _ in range(n)])
