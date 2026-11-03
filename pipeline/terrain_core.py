# ===============================================================
#  地形：高度函数 + 按地形摆放（不依赖 Blender，只用 numpy，可以直接用 python 测试）
#  输入是 terrain_data.py 生成的 terrain.npz；pipeline.py 加了 --terrain 时才会用到这里
#
#  场景分三层，都从同一个高度函数 Terrain.height() 取高度，所以接得上：
#    近景：±110 m（活动范围 ±100 m），每 2.5 m 一个顶点，真实素材
#    中景：到约 2.5 km，顶点上色的地形 + 简化树
#    远景：到约 90 km，只有地形轮廓（网页端加雾）
#  5 个输入怎么决定场景（pipeline_rules.decide / place）不变；这里只改“东西摆在哪、多高”
# ===============================================================
import math
import numpy as np
import pipeline_rules as rules

CLASSES = ('forest_woodland', 'shrubland', 'grass_herbaceous', 'agriculture',
           'developed', 'open_water', 'sparse_barren', 'snow_ice')
F, S, G, A, D, W, B, N = range(8)

# ---------------- 可调参数 ----------------
NEAR_HALF = 110.0            # 近景网格半边长（比 100 m 活动范围略大）
NEAR_STEP = 2.5              # 近景顶点间距（米）；也是 scene.json 里地面高度表的间距
MID_STEP = 20.0              # 中景网格起始间距
MID_FINE = 10.0              # 300 m 以内中景加密到这个间距，陡坡上和近景接缝更贴合
MID_GROWTH = 1.08            # 往外每一格比上一格大多少
FAR_HALF = 90000.0           # 远景到多远
MAX_HOUSE_SLOPE = 14.0       # 房子只放在坡度小于这个值的地方（度）
BUILDING_CLASS = 5           # buildings_data.py 占地图里“建筑”的编号
ASSET_TREE_HALF = 150.0      # 这个范围内的树都用真实素材（活动范围外，但离得近看得清）
GRASS_RING_HALF = 170.0      # 草丛延伸到这里并逐渐变稀
NEAR_TREE_TRIES = 15000      # 600 m 内简化树的尝试次数（再按森林占比筛）
FAR_TREE_TRIES = 25000       # 600 m ~ 2.5 km 的尝试次数
FAR_TREE_RADIUS = 2500.0     # 简化树到多远
WATER_MAX = 0.4              # 水面占比达到这个值就不放草、灌木、简化树
SKIRT_DROP = 1.5             # 近景网格四周往下垂的边（挡住和中景之间的缝）
EARTH_R = 6371000.0


# ---------------- 插值 ----------------
def sample_cubic(grid, u, v):
    """Catmull-Rom 插值；u 是列（可带小数），v 是行"""
    h, w = grid.shape
    u = np.clip(u, 0, w - 1); v = np.clip(v, 0, h - 1)
    i = np.floor(u).astype(int); j = np.floor(v).astype(int)
    tx = u - i; ty = v - j

    def weights(t):
        return (-0.5 * t ** 3 + t ** 2 - 0.5 * t, 1.5 * t ** 3 - 2.5 * t ** 2 + 1,
                -1.5 * t ** 3 + 2 * t ** 2 + 0.5 * t, 0.5 * t ** 3 - 0.5 * t ** 2)
    wx, wy = weights(tx), weights(ty)
    out = np.zeros_like(u, dtype=float)
    for a in range(4):
        jj = np.clip(j + a - 1, 0, h - 1)
        for b in range(4):
            ii = np.clip(i + b - 1, 0, w - 1)
            out += wy[a] * wx[b] * grid[jj, ii]
    return out


def sample_linear(grid, u, v):
    h, w = grid.shape
    u = np.clip(u, 0, w - 1); v = np.clip(v, 0, h - 1)
    i = np.minimum(np.floor(u).astype(int), w - 2); j = np.minimum(np.floor(v).astype(int), h - 2)
    tx = u - i; ty = v - j
    return (grid[j, i] * (1 - tx) * (1 - ty) + grid[j, i + 1] * tx * (1 - ty)
            + grid[j + 1, i] * (1 - tx) * ty + grid[j + 1, i + 1] * tx * ty)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------- 高度函数 ----------------
class Terrain:
    """整个场景只有这一个高度函数。坐标：Blender 米，原点 = 地形数据中心，x 朝东、y 朝北（EPSG:5070 网格方向）"""
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.le, self.lc = d['local_elev'], d['local_cover']
        self.fe, self.fc = d['far_elev'], d['far_cover']
        self.fr = d['far_relief']
        self.lox, self.loy, self.lcs = float(d['local_ox']), float(d['local_oy']), float(d['local_cs'])
        self.fox, self.foy, self.fcs = float(d['far_ox']), float(d['far_oy']), float(d['far_cs'])
        self.half = self.le.shape[0] * self.lcs / 2
        rng = np.random.default_rng(12345)
        # 小起伏：数据里没有 30 m 以下的细节，这里只为了看起来不那么光滑，不代表真实地形
        self.waves = [(rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), 2 * np.pi / wl)
                      for wl in (17, 29, 43, 71, 113)]
        # 远山轮廓：3 km 的均值太平滑，按每格真实高差加上山脊状起伏。幅度来自数据，具体形状是生成的
        self.ridges = [(rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi), 2 * np.pi / wl, amp)
                       for wl, amp in ((12000, .5), (6100, .32), (3100, .2), (1700, .12), (900, .07))]
        # 小起伏的幅度跟着真实坡度走：平地上几乎没有，不能让它挡住地平线
        gy_, gx_ = np.gradient(self.le, self.lcs)
        self.lslope = np.degrees(np.arctan(np.hypot(gx_, gy_)))
        self.base = 0.0
        self.pads = []              # 房子底下压平的地方 (x, y, 高度)
        self.base = float(self.raw(np.array([0.0]), np.array([0.0]))[0])

    def _uv(self, x, y, ox, oy, cs):
        return (x - ox) / cs - 0.5, (oy - y) / cs - 0.5

    def blend(self, x, y):
        return smoothstep(0, 400, self.half - np.maximum(np.abs(x), np.abs(y)))

    def raw(self, x, y):
        """不含房子压平的高度（相对场景中心，米）"""
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        lu, lv = self._uv(x, y, self.lox, self.loy, self.lcs)
        fu, fv = self._uv(x, y, self.fox, self.foy, self.fcs)
        w = self.blend(x, y)
        z = w * sample_cubic(self.le, lu, lv) + (1 - w) * sample_cubic(self.fe, fu, fv)
        dist = np.hypot(x, y)
        amp = (0.3 + 1.2 * smoothstep(120, 500, dist)) * (1 - smoothstep(2500, 6000, dist))
        ripple = sum(np.sin((x * np.cos(a) + y * np.sin(a)) * k + p) for a, p, k in self.waves) / len(self.waves)
        z = z + amp * ripple * 1.6 * np.clip(sample_linear(self.lslope, lu, lv) / 10, 0.03, 1)
        u = x + 900 * np.sin(y * 2 * np.pi / 7300 + 1.3); v = y + 900 * np.sin(x * 2 * np.pi / 6100 + 0.7)
        ridge = sum(am * (1 - 2 * np.abs(np.sin((u * np.cos(a) + v * np.sin(a)) * k + p)))
                    for a, p, k, am in self.ridges)
        z = z + sample_linear(self.fr, fu, fv) * 0.5 * ridge * smoothstep(1500, 4500, dist)
        return z - self.base - 0.87 * dist ** 2 / (2 * EARTH_R)      # 远处按地球曲率往下压

    def height(self, x, y):
        """最终高度：房子底下压平"""
        z = self.raw(x, y)
        for hx, hy, hz in self.pads:
            d = np.hypot(np.asarray(x) - hx, np.asarray(y) - hy)
            z = hz + (z - hz) * smoothstep(8, 15, d)
        return z

    def slope(self, x, y, e=6.0):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        dx = (self.raw(x + e, y) - self.raw(x - e, y)) / (2 * e)
        dy = (self.raw(x, y + e) - self.raw(x, y - e)) / (2 * e)
        return np.degrees(np.arctan(np.hypot(dx, dy)))

    def cover(self, x, y):
        """返回 8 类地表占比，形状 (8, ...)"""
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        lu, lv = self._uv(x, y, self.lox, self.loy, self.lcs)
        fu, fv = self._uv(x, y, self.fox, self.foy, self.fcs)
        w = self.blend(x, y)
        lc = np.nan_to_num(self.lc)
        return np.stack([w * sample_linear(lc[i], lu, lv) + (1 - w) * sample_linear(self.fc[i], fu, fv)
                         for i in range(8)])


# ---------------- 网格（顶点和面，不建 Blender 物体） ----------------
def grid_mesh(xs, ys, keep_quad=None):
    xx, yy = np.meshgrid(xs, ys)
    nx, ny = len(xs), len(ys)
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if keep_quad and not keep_quad(xs[i], xs[i + 1], ys[j], ys[j + 1]):
                continue
            faces.append((j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i))
    return xx.ravel(), yy.ravel(), faces


def near_axis():
    return np.arange(-NEAR_HALF, NEAR_HALF + 0.01, NEAR_STEP)


def mid_axis():
    """中景 + 远景的坐标轴：中间密、外面稀；边界正好落在近景边缘，中间挖空给近景"""
    v = [i * MID_STEP for i in range(0, int(300 / MID_STEP) + 1)]
    step = MID_STEP
    while v[-1] < FAR_HALF:
        step *= MID_GROWTH
        v.append(v[-1] + step)
    v = np.array(v)
    return np.unique(np.concatenate([-v[:0:-1], v, np.arange(-300, 301, MID_FINE), [-NEAR_HALF, NEAR_HALF]]))


def mid_mesh(t):
    axis = mid_axis()
    inner = NEAR_HALF + 0.01
    x, y, faces = grid_mesh(axis, axis, lambda x0, x1, y0, y1: not (
        abs(x0) < inner and abs(x1) < inner and abs(y0) < inner and abs(y1) < inner))
    return axis, x, y, t.height(x, y), faces


def ring_indices(n):
    """n x n 网格最外一圈顶点的序号（按顺序绕一圈）"""
    return ([i for i in range(n)] + [j * n + n - 1 for j in range(1, n)]
            + [(n - 1) * n + i for i in range(n - 2, -1, -1)] + [j * n for j in range(n - 2, 0, -1)])


def add_skirt(x, y, z, faces, n, drop=SKIRT_DROP):
    """近景网格四周加一圈往下垂的边，中景比近景低一点时也不会漏出缝（中景比近景高的情况由 match_mid_edge 处理）"""
    ring = ring_indices(n)
    base_i = len(x)
    x2 = np.concatenate([x, x[ring]]); y2 = np.concatenate([y, y[ring]]); z2 = np.concatenate([z, z[ring] - drop])
    m = len(ring)
    faces = list(faces) + [(ring[k], ring[(k + 1) % m], base_i + (k + 1) % m, base_i + k) for k in range(m)]
    return x2, y2, z2, faces


def _edge_point(x, y):
    """每个点投到最近的近景边上：返回边上的点、沿边的坐标、是哪条边"""
    h = NEAR_HALF
    on_x = np.abs(x) >= np.abs(y)
    ex = np.where(on_x, np.sign(x) * h, x); ey = np.where(on_x, y, np.sign(y) * h)
    side = np.where(on_x, np.where(x >= 0, 0, 1), np.where(y >= 0, 2, 3))
    return ex, ey, np.where(on_x, ey, ex), side


def mid_edge_height(t, x, y, axis=None):
    """中景网格在近景边线上的高度：中景沿边线只有每个 axis 点的高度，中间是直线"""
    axis = mid_axis() if axis is None else axis
    h = NEAR_HALF
    ex, ey, along, side = _edge_point(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    out = np.empty_like(along)
    for k, (vx, vy) in enumerate(((np.full_like(axis, h), axis), (np.full_like(axis, -h), axis),
                                  (axis, np.full_like(axis, h)), (axis, np.full_like(axis, -h)))):
        m = side == k
        out[m] = np.interp(along[m], axis, t.height(vx, vy))
    return out


def match_mid_edge(t, x, y, z, blend=10.0):
    """近景边缘的高度改成和中景在交界线上完全一样，往里 10 m 内逐渐过渡回原高度。
    近景边缘每 2.5 m 一个点、中景 10 m 一个点，有的地方中景会高出近景零点几米，
    往下垂的挡边盖不住这种缝，远处会透过缝看到一条黑线"""
    ex, ey, _, _ = _edge_point(x, y)
    delta = mid_edge_height(t, x, y) - t.height(ex, ey)
    w = 1 - smoothstep(0, blend, NEAR_HALF - np.maximum(np.abs(x), np.abs(y)))
    return z + delta * w


def near_mesh(t):
    ax = near_axis()
    x, y, faces = grid_mesh(ax, ax)
    z = match_mid_edge(t, x, y, t.height(x, y))
    return ax, x, y, z, faces


def cover_colors(t, x, y, avg):
    """中景、远景的顶点颜色（线性）：按地表占比混合，再加一点斑驳感。avg = 各地面贴图的平均颜色"""
    c = t.cover(x, y)
    pal = np.array([
        (0.035, 0.075, 0.030),      # 森林（从远处看到的是树冠）
        (0.16, 0.17, 0.08),         # 灌丛
        avg['mat_grass'],
        avg['mat_farmland'],
        avg['mat_urban'],
        avg['mat_water'],
        (0.20, 0.17, 0.13),         # 裸地
        (0.85, 0.88, 0.92),         # 冰雪
    ])
    rgb = np.einsum('i...,ij->...j', c, pal)
    # 有简化树的范围内，森林的地面是林下地面的颜色；再往外没有树了，才用树冠的深绿色
    d = smoothstep(FAR_TREE_RADIUS - 600, FAR_TREE_RADIUS, np.hypot(x, y))[..., None]
    rgb = rgb + c[F][..., None] * (1 - d) * (np.array((0.075, 0.085, 0.035)) - pal[F])
    a, b = 0.013, 0.047
    mottle = (np.sin(x * a + np.sin(y * a * 1.7) * 2.1) + np.sin(y * b + np.sin(x * b * 0.8) * 1.9)
              + np.sin((x + y) * 0.11)) / 3
    return rgb * (1 + 0.22 * mottle)[..., None]


# ---------------- 摆放：数量按原规则，位置按地形和地表重新筛 ----------------
def relocate(t, layout, rng):
    half = rules.SCENE_SIZE / 2
    report = {'houses_wanted': len(layout['houses'])}

    def cover_at(x, y):
        return t.cover(np.array([x]), np.array([y]))[:, 0]

    # 房子：坡度够缓、不在水里，优先放在建成区
    houses = []
    for h in layout['houses']:
        for _ in range(400):
            x, y = rng.uniform(-half + h['r'], half - h['r']), rng.uniform(-half + h['r'], half - h['r'])
            c = cover_at(x, y)
            if c[W] > 0.2 or float(t.slope(np.array([x]), np.array([y]))[0]) > MAX_HOUSE_SLOPE:
                continue
            if rng.random() > 0.15 + 0.85 * c[D]:
                continue
            if all((x - o['x']) ** 2 + (y - o['y']) ** 2 > (h['r'] + o['r'] + 1) ** 2 for o in houses):
                houses.append({**h, 'x': x, 'y': y})
                break
    layout['houses'] = houses
    report['houses_placed'] = len(houses)
    # 房子底下压平，再算其余东西的高度
    t.pads = [(h['x'], h['y'], float(t.raw(np.array([h['x']]), np.array([h['y']]))[0])) for h in houses]

    def clear(x, y, r):
        return all((x - h['x']) ** 2 + (y - h['y']) ** 2 > (r + h['r']) ** 2 for h in houses)

    # 树：总量按原规则（树冠覆盖），位置按森林占比
    trees = []
    for tr in layout['trees']:
        for _ in range(60):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            c = cover_at(x, y)
            if c[W] < 0.3 and rng.random() < c[F] + 0.25 * c[S] + 0.1 * c[G] + 0.1 * c[D] and clear(x, y, tr['r'] * 0.5):
                trees.append({**tr, 'x': x, 'y': y})
                break
    report['trees'] = (len(layout['trees']), len(trees))
    layout['trees'] = trees

    def thin(items, weight, r):
        if not items:
            return items
        xs = np.array([i['x'] for i in items]); ys = np.array([i['y'] for i in items])
        cv = t.cover(xs, ys)
        keep = []
        for k, it in enumerate(items):
            if rng.random() < weight(cv[:, k]) and clear(it['x'], it['y'], r):
                keep.append(it)
        return keep

    report['shrubs'] = len(layout['shrubs'])
    layout['shrubs'] = thin(layout['shrubs'], lambda c: (c[S] + 0.5 * c[F] + 0.3 * c[G]) * (c[W] < WATER_MAX), 1.5)
    report['grass'] = len(layout['grass'])
    layout['grass'] = thin(layout['grass'], lambda c: (c[G] + c[S] + 0.45 * c[F] + 0.25 * c[D]) * (c[W] < WATER_MAX), 0.5)
    report['after'] = {k: len(layout[k]) for k in ('shrubs', 'grass')}
    return report


def extend_near(t, layout, veg_m, rng, blocked):
    """活动范围外一圈继续用真实素材，这样走到边上也看不到简化树；密度和范围内一样"""
    half = rules.SCENE_SIZE / 2
    trees_m = [a for a in veg_m if a['kind'] == 'tree']
    green = [a for a in veg_m if a['kind'] == 'grass' and a['species'] == 'green']
    dry = [a for a in veg_m if a['kind'] == 'grass' and a['species'] == 'dry']

    def ring(hmax, tries):
        x = np.array([rng.uniform(-hmax, hmax) for _ in range(tries)])
        y = np.array([rng.uniform(-hmax, hmax) for _ in range(tries)])
        return x, y, np.maximum(np.abs(x), np.abs(y)), t.cover(x, y)

    area = (2 * ASSET_TREE_HALF) ** 2
    density = max(len(layout['trees']), 1) / rules.SCENE_SIZE ** 2
    x, y, d, c = ring(ASSET_TREE_HALF, int(area * density))
    for i in range(len(x)):
        if d[i] > half and not blocked(x[i], y[i]) and c[W, i] < 0.3 and rng.random() < c[F, i] + 0.2 * c[S, i] + 0.08 * c[G, i]:
            tm = rng.choice(trees_m); s_ = rng.uniform(0.8, 1.2)
            layout['trees'].append({'name': tm['name'], 'x': float(x[i]), 'y': float(y[i]), 'rot': rng.uniform(0, 6.28),
                                    's': s_, 'sz': s_ * rng.uniform(0.9, 1.15), 'r': tm['crown_diameter_m'] * s_ / 2})
    x, y, d, c = ring(GRASS_RING_HALF, 9000)
    fade = 1 - smoothstep(half, GRASS_RING_HALF, d)
    for i in range(len(x)):
        if d[i] > half and c[W, i] < WATER_MAX and rng.random() < fade[i] * (c[G, i] + c[S, i] + 0.45 * c[F, i]):
            pool = dry if rng.random() < 0.45 else green
            layout['grass'].append({'name': rng.choice(pool)['name'], 'x': float(x[i]), 'y': float(y[i]),
                                    'rot': rng.uniform(0, 6.28), 's': rng.uniform(1.0, 1.8)})


def place_fire(t, decision, layout, rng, avoid=None):
    """火的中心：优先“能烧的地表、不在水里、离房子够远”，都找不到时退而求其次，尽量不放进水里。
    avoid(x, y)：退而求其次时也不放的地方（真实建筑里）"""
    if not layout.get('fire'):
        return None
    half = rules.SCENE_SIZE / 2
    radius = layout['fire']['radius']
    houses = layout['houses']

    def clear(x, y):
        return all((x - h['x']) ** 2 + (y - h['y']) ** 2 > (rules.FIRE_HOUSE_GAP + h['r']) ** 2 for h in houses)
    x = np.array([rng.uniform(-half * 0.8, half * 0.8) for _ in range(3000)])
    y = np.array([rng.uniform(-half * 0.8, half * 0.8) for _ in range(3000)])
    c = t.cover(x, y)
    burnable = c[F] + c[S] + c[G] + 0.5 * c[A]
    ok_house = np.array([clear(a, b) for a, b in zip(x.tolist(), y.tolist())])
    center = None
    not_avoided = ~avoid(x, y) if avoid is not None else np.ones(len(x), dtype=bool)
    for mask in ((c[W] < 0.1) & (burnable > 0.5) & ok_house, (c[W] < 0.1) & ok_house, (c[W] < 0.1) & not_avoided, c[W] < 0.1):
        if mask.any():
            k = int(np.argmax(mask))
            center = (float(x[k]), float(y[k])); break
    if center is None:                       # 整个范围都是水：放在水最少的地方
        k = int(np.argmin(c[W]))
        center = (float(x[k]), float(y[k]))
    burning = []
    if decision['crown_fire']:
        burning = [tr for tr in layout['trees'] if (tr['x'] - center[0]) ** 2 + (tr['y'] - center[1]) ** 2 < radius ** 2]
    return {'center': center, 'radius': radius, 'burning_trees': burning}


def far_tree_points(t, rng_np, keep_clear):
    """中景简化树的位置：600 m 内密度和近景一致，再往外变稀、变大（k = 放大倍数）"""
    def pick(r0, r1, tries):
        x = rng_np.uniform(-r1, r1, tries); y = rng_np.uniform(-r1, r1, tries)
        d = np.maximum(np.abs(x), np.abs(y))
        c = t.cover(x, y)
        ok = ((d > ASSET_TREE_HALF) & (d >= r0) & (d < r1) & (rng_np.random(tries) < c[F] + 0.08 * c[S])
              & (c[W] < WATER_MAX) & ~keep_clear(x, y))
        return x[ok], y[ok]
    x1, y1 = pick(0, 600, NEAR_TREE_TRIES)
    x2, y2 = pick(600, FAR_TREE_RADIUS, FAR_TREE_TRIES)
    x = np.concatenate([x1, x2]); y = np.concatenate([y1, y2])
    k = np.concatenate([np.ones(len(x1)), np.full(len(x2), 2.4)])
    return x, y, t.height(x, y), k


# ---------------- 视线和出生点 ----------------
def view_table(t, x0=0.0, y0=0.0):
    """沿 36 个方向算：站在这里，最远能看到多远的地形、那里在视线上方几度"""
    z0 = float(t.raw(np.array([x0]), np.array([y0]))[0]) + 1.7
    ds = np.geomspace(30, 85000, 260)
    rows = []
    for deg in range(0, 360, 10):
        a = math.radians(deg)
        z = t.raw(x0 + ds * math.cos(a), y0 + ds * math.sin(a))
        ang = np.degrees(np.arctan2(z - z0, ds))
        seen = ang >= np.maximum.accumulate(ang)                  # 比前面所有地形都高，才看得见
        rows.append({'deg': deg, 'visible_km': round(float(ds[seen].max()) / 1000, 1),
                     'skyline_deg': round(float(ang[seen][-1]), 2)})
    return rows


def best_direction(t):
    """出生时朝哪看：优先能看得远、地形又确实高出地平线的方向（不改地形，只选朝向）"""
    rows = view_table(t)
    best = max(rows, key=lambda r: min(r['visible_km'], 60) * max(min(r['skyline_deg'], 6), 0.05))
    return math.radians(best['deg']), best['skyline_deg'], best, rows


def pick_spawn(t, dx, dy, avoid=None):
    """出生点：在活动范围内找一块缓坡、树少、不在水里、朝选定方向看得开的地方。avoid(x, y) 为真的点不选（如建筑里）"""
    best, best_score = (0.0, 0.0), -1e9
    for gx in range(-90, 91, 10):
        for gy in range(-90, 91, 10):
            x, y = np.array([float(gx)]), np.array([float(gy)])
            if avoid is not None and avoid(float(gx), float(gy)):
                continue
            sl = float(t.slope(x, y)[0])
            c = t.cover(x, y)[:, 0]
            ds = np.array([40.0, 80.0, 150.0, 300.0, 600.0])
            rise = t.raw(gx + dx * ds, gy + dy * ds) - float(t.raw(x, y)[0])
            block = np.degrees(np.arctan2(rise, ds)).max()
            score = -abs(sl - 4) * 0.6 - c[F] * 12 - c[W] * 200 - max(block - 6, 0) * 1.5 - 0.02 * (gx * dx + gy * dy)
            if score > best_score:
                best, best_score = (float(gx), float(gy)), score
    return best


def building_circles(bld, half):
    """真实建筑当成等面积的圆（火离房子的距离用），只取 ±half 以内的"""
    out = []
    for b in bld.buildings:
        p = np.array(b['poly'], dtype=float)
        cx, cy = p.mean(axis=0)
        if abs(cx) < half and abs(cy) < half:
            area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
            out.append({'x': float(cx), 'y': float(cy), 'r': math.sqrt(area / math.pi)})
    return out


def footprint_distance(bld, x, y):
    """点到最近的真实建筑轮廓的距离（米）；在建筑里面时为 0"""
    a = np.concatenate([np.array(b['poly'], dtype=float) for b in bld.buildings]) if bld.buildings else None
    if a is None:
        return 999.0
    b_ = np.concatenate([np.roll(np.array(b['poly'], dtype=float), -1, axis=0) for b in bld.buildings])
    ab = b_ - a
    tt = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(axis=1), 1e-9), 0, 1)
    d = float(np.hypot(a[:, 0] + tt * ab[:, 0] - x, a[:, 1] + tt * ab[:, 1] - y).min())
    return 0.0 if int(bld.classes(np.array([x]), np.array([y]))[0]) == BUILDING_CLASS else d


def plan(t, inputs, seed, veg_m, house_m, bld=None):
    """完整的“做决定”：5 个输入 → 决定 → 按地形摆放 → 出生点、火。返回给 Blender 端搭场景用的东西
    bld = buildings_data.load(...) 的结果（可选）：有它时不再随机放房子（没有数据也不放），树草让开真实建筑和道路"""
    import random
    rng = random.Random(seed)
    tree_h = {m['name']: m['height_m'] for m in veg_m if m['kind'] == 'tree'}
    decision = rules.decide(inputs, rng)
    layout = rules.place(decision, rng, veg_m, house_m)
    if bld is not None:
        layout['houses'] = []                    # 房子全部换成真实建筑，地形不压平
    placement = relocate(t, layout, rng)

    ang, hz, best_view, view_rows = best_direction(t)
    dx, dy = math.cos(ang), math.sin(ang)
    avoid = None
    if bld is not None:
        ring = np.array([(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)], dtype=float)     # 出生点 4 m 内不能有建筑
        avoid = lambda x, y: bool((bld.classes(x + ring[:, 0], y + ring[:, 1]) == BUILDING_CLASS).any())
    sx, sy = pick_spawn(t, dx, dy, avoid)

    def in_view(x, y, near_half, far_half, length):
        """出生点视线前方的扇形，里面少放树，免得一进场景就被挡住"""
        rx, ry = np.asarray(x) - sx, np.asarray(y) - sy
        along = rx * dx + ry * dy
        across = np.abs(-rx * dy + ry * dx)
        return (along > -6) & (along < length) & (across < near_half + (far_half - near_half) * np.clip(along, 0, None) / length)

    layout['trees'] = [i for i in layout['trees'] if not in_view(i['x'], i['y'], 7, 45, 130)]
    layout['shrubs'] = [i for i in layout['shrubs'] if not in_view(i['x'], i['y'], 3, 10, 40)]
    extend_near(t, layout, veg_m, rng, lambda x, y: in_view(x, y, 7, 45, 400))
    if bld is not None:
        placement['buildings_removed'] = {}
        for k in ('trees', 'shrubs', 'grass'):
            before = len(layout[k])
            if layout[k]:
                keep = ~bld.blocked(np.array([i['x'] for i in layout[k]]), np.array([i['y'] for i in layout[k]]))
                layout[k] = [i for i, ok in zip(layout[k], keep.tolist()) if ok]
            placement['buildings_removed'][k] = before - len(layout[k])
    for tr in layout['trees']:
        tr.setdefault('h', tree_h[tr['name']] * tr.get('s', 1.0))
    if bld is not None:
        layout['houses'] = building_circles(bld, ASSET_TREE_HALF)   # 只给选火的位置用
        layout['fire'] = place_fire(t, decision, layout, rng, lambda x, y: bld.classes(x, y) == BUILDING_CLASS)
        layout['houses'] = []
    else:
        layout['fire'] = place_fire(t, decision, layout, rng)
    for key in ('houses', 'trees', 'shrubs', 'grass'):
        if layout[key]:
            zs = t.height(np.array([i['x'] for i in layout[key]]), np.array([i['y'] for i in layout[key]]))
            for it, zv in zip(layout[key], zs.tolist()):
                it['z'] = zv
    return {'rng': rng, 'decision': decision, 'layout': layout, 'placement': placement,
            'dir': (dx, dy), 'skyline_deg': hz, 'best_view': best_view, 'view_rows': view_rows,
            'spawn': (sx, sy), 'in_view': in_view, 'buildings': bld}


# ---------------- 自动检查（结果写进 scene.json 的 debug.checks） ----------------
def seam_gap(t, axis):
    """地形数据在近景边缘和中景网格（线性插值）之间的高度差；网格上已由 match_mid_edge 对齐，实际的缝见 near_mid_crack_m"""
    e = near_axis()
    worst = 0.0
    for ex, ey in ((np.full_like(e, NEAR_HALF), e), (np.full_like(e, -NEAR_HALF), e),
                   (e, np.full_like(e, NEAR_HALF)), (e, np.full_like(e, -NEAR_HALF))):
        i = np.clip(np.searchsorted(axis, ex) - 1, 0, len(axis) - 2)
        j = np.clip(np.searchsorted(axis, ey) - 1, 0, len(axis) - 2)
        x0, x1, y0, y1 = axis[i], axis[i + 1], axis[j], axis[j + 1]
        tx = (ex - x0) / (x1 - x0); ty = (ey - y0) / (y1 - y0)
        m = (t.height(x0, y0) * (1 - tx) * (1 - ty) + t.height(x1, y0) * tx * (1 - ty)
             + t.height(x0, y1) * (1 - tx) * ty + t.height(x1, y1) * tx * ty)
        worst = max(worst, float(np.abs(t.height(ex, ey) - m).max()))
    return round(worst, 2)


def run_checks(t, layout, fire, spawn_xy, near_z, mid_z, axis, mid_tree_xy, bld=None):
    def cov(items):
        if not items:
            return np.zeros((8, 0))
        return t.cover(np.array([i['x'] for i in items]), np.array([i['y'] for i in items]))

    hs = layout['houses']
    hc = cov(hs)
    overlap = sum(1 for i in range(len(hs)) for j in range(i + 1, len(hs))
                  if (hs[i]['x'] - hs[j]['x']) ** 2 + (hs[i]['y'] - hs[j]['y']) ** 2 < (hs[i]['r'] + hs[j]['r']) ** 2)
    sx, sy = spawn_xy
    sc = t.cover(np.array([sx]), np.array([sy]))[:, 0]
    fx, fy = mid_tree_xy
    checks = {
        'houses_in_water': int((hc[W] > 0.2).sum()) if hs else 0,
        'houses_max_slope_deg': round(float(t.slope(np.array([h['x'] for h in hs]), np.array([h['y'] for h in hs])).max()), 1) if hs else None,
        'houses_overlapping_pairs': overlap,
        'trees_in_water': int((cov(layout['trees'])[W] > 0.5).sum()),
        'grass_in_water': int((cov(layout['grass'])[W] > 0.5).sum()),
        'mid_trees_in_water': int((t.cover(fx, fy)[W] > 0.5).sum()) if len(fx) else 0,
        'spawn_slope_deg': round(float(t.slope(np.array([sx]), np.array([sy]))[0]), 1),
        'spawn_water_fraction': round(float(sc[W]), 2),
        'spawn_inside_walk_area': bool(abs(sx) <= 100 and abs(sy) <= 100),
        'terrain_has_nan': bool(np.isnan(near_z).any() or np.isnan(mid_z).any()),
        'near_mid_seam_max_m': seam_gap(t, axis),
    }
    ax = near_axis()
    n = len(ax); zz = near_z.reshape(n, n)
    xx, yy = np.meshgrid(ax, ax)
    ring = ring_indices(n)
    edge = mid_edge_height(t, xx.ravel()[ring], yy.ravel()[ring], axis)
    checks['near_mid_crack_m'] = round(float(np.abs(near_z[ring] - edge).max()), 3)     # 两张网格实际的缝，应为 0
    walk = np.ix_(np.abs(ax) <= 100, np.abs(ax) <= 100)
    gy, gx = np.gradient(zz, NEAR_STEP)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))[walk]
    checks['walk_slope_mean_deg'] = round(float(sl.mean()), 1)
    checks['walk_slope_max_deg'] = round(float(sl.max()), 1)
    checks['walk_relief_m'] = round(float(np.ptp(zz[walk])), 1)
    if fire:
        cx, cy = fire['center']
        fc = t.cover(np.array([cx]), np.array([cy]))[:, 0]
        checks['fire_water_fraction'] = round(float(fc[W]), 2)
        checks['fire_burnable_fraction'] = round(float(fc[F] + fc[S] + fc[G] + 0.5 * fc[A]), 2)
        checks['fire_min_house_gap_m'] = round(min([math.hypot(cx - h['x'], cy - h['y']) - h['r'] for h in hs] or [999]), 1)
    if bld is not None:
        def on_blocked(items):
            return int(bld.blocked(np.array([i['x'] for i in items]), np.array([i['y'] for i in items])).sum()) if items else 0
        checks['buildings_status'] = bld.status
        checks['plants_on_buildings_or_roads'] = sum(on_blocked(layout[k]) for k in ('trees', 'shrubs', 'grass'))
        checks['mid_trees_on_buildings_or_roads'] = int(bld.blocked(fx, fy).sum()) if len(fx) else 0
        ring = np.array([(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4)], dtype=float)
        checks['spawn_in_building'] = bool((bld.classes(sx + ring[:, 0], sy + ring[:, 1]) == BUILDING_CLASS).any())
        if fire:
            cx, cy = fire['center']
            checks['fire_min_building_gap_m'] = round(footprint_distance(bld, cx, cy), 1)
            checks['fire_in_building'] = checks['fire_min_building_gap_m'] == 0
    return checks


CHECK_LIMITS = {          # 超过就算没通过（tests/terrain_regression.py 用）
    'houses_in_water': 0, 'houses_overlapping_pairs': 0, 'trees_in_water': 0, 'grass_in_water': 0,
    'mid_trees_in_water': 0, 'near_mid_seam_max_m': 0.6, 'near_mid_crack_m': 0.01, 'spawn_slope_deg': 15,
    'plants_on_buildings_or_roads': 0, 'mid_trees_on_buildings_or_roads': 0, 'spawn_in_building': False,
    'fire_in_building': False,
}
