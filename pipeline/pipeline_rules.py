# ===============================================================
#  Pipeline 规则：输入 5 个值 → 决定场景里的一切（对照表的代码版）
#  不依赖 Blender，可以直接用 python 运行和测试：
#    python pipeline_rules.py            随便生成一个场景的决定并打印
#  规则对应规格文档「生成规则」一节，数字都在下面的「可调参数」里
# ===============================================================
import json, math, os, random

# ---------------- 可调参数 ----------------
SCENE_SIZE = 200.0                   # 场景边长（米）
AREA = SCENE_SIZE ** 2               # 场景面积（m²）
# 数量都按面积算，改场景大小时自动跟着变（数值参照 120 m 测试场景的效果）
MAX_HOUSES = round(30 / 120 ** 2 * AREA)                   # 房子上限：120 m 时 30 栋
FIRE_PROB = {'low': 0.15, 'medium': 0.40, 'high': 0.75, 'very_high': 0.95}
RISK_EDGES = (0.02, 0.078, 0.171)    # 风险档界线：低 | 中 | 较高 | 很高
FRP_EDGES = (4.33, 24.8, 120.8)      # 火势档界线（MW）：小 | 中 | 大 | 极端
FRP_DRAW = {'small': 0.50, 'medium': 0.40, 'large': 0.09, 'extreme': 0.01}   # 没有 FRP 时的抽取概率
FIRE_RADIUS = {'small': 4.0, 'medium': 8.0, 'large': 15.0, 'extreme': 25.0}
FIRE_HOUSE_GAP = 5.0                 # 火离房子至少多远（米）
GRASS_PER_M2 = {'grassland': 7000 / 120 ** 2, 'shrubland': 3500 / 120 ** 2}   # 每平方米几丛草
SHRUB_PER_M2 = {'shrubland': 200 / 120 ** 2, 'forest': 60 / 120 ** 2}       # 每平方米几丛灌木
SPAWN = (0.0, -SCENE_SIZE / 2 + 5, 1.7)   # 用户初始位置（Blender 坐标：场景南边往里 5 m，人眼高度）

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')

# ---------------- 对照表 ----------------
GROUND_BY_FUEL = {98: 'water', 92: 'bare', 99: 'bare', 91: 'urban', 93: 'farmland'}
GROUND_BY_FUEL.update({c: 'grassland' for c in range(101, 109)})
GROUND_BY_FUEL.update({c: 'shrubland' for c in list(range(121, 125)) + list(range(141, 150))})
GROUND_BY_FUEL.update({c: 'forest' for c in list(range(161, 166)) + list(range(181, 190)) + list(range(201, 204))})
BURNABLE = {'grassland', 'shrubland', 'forest'}
CANOPY_TIER = {0: 'none', 15: 'sparse', 25: 'sparse', 35: 'medium', 45: 'medium', 55: 'medium',
               65: 'dense', 75: 'dense', 85: 'dense', 95: 'dense'}
COVER_RANGE = {'none': (0.0, 0.0), 'sparse': (0.1, 0.3), 'medium': (0.3, 0.6), 'dense': (0.6, 1.0)}
FIRE_LOCATION = {'grassland': 'grass', 'shrubland': 'shrub', 'forest': 'forest_floor'}
SMOKE = {'none': 'none', 'small': 'light', 'medium': 'medium', 'large': 'heavy', 'extreme': 'extreme'}
SKY = {'none': 'normal', 'small': 'normal', 'medium': 'normal', 'large': 'orange', 'extreme': 'dark_orange'}


def _num(v, lo=None, hi=None):
    """合法数字就返回 float，否则返回 None（缺失、不是数字、超出范围都当作缺失）"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        return None
    return v


def _tier(v, edges, names):
    for e, n in zip(edges, names):
        if v < e:
            return n
    return names[-1]


def load_manifests(asset_dir=ASSET_DIR):
    def read(folder):
        with open(os.path.join(asset_dir, folder, f'{folder}_manifest.json'), encoding='utf-8') as f:
            return json.load(f)['assets']
    veg = read('trees') + read('shrubs') + read('grass')
    return veg, read('houses')


# ---------------- 第一部分：决定场景里有什么（步骤 1 – 6） ----------------
def decide(inputs, rng):
    density = _num(inputs.get('housing_density'), 0)
    fuel = _num(inputs.get('fuel_fbfm40'))
    canopy = _num(inputs.get('canopy_cover'))
    risk = _num(inputs.get('occupancy_probability'), 0, 1)
    frp = _num(inputs.get('fire_radiative_power'), 0)

    # 步骤 1：地面
    ground = GROUND_BY_FUEL.get(int(fuel) if fuel is not None and fuel == int(fuel) else None, 'grassland')

    # 步骤 2：树的覆盖率
    canopy_tier = CANOPY_TIER.get(int(canopy) if canopy is not None and canopy == int(canopy) else None, 'none')
    if ground == 'water':
        canopy_tier = 'none'
    lo, hi = COVER_RANGE[canopy_tier]
    tree_cover = rng.uniform(lo, hi) if hi > 0 else 0.0

    # 步骤 3：房子数量
    expected = (density or 0.0) * (SCENE_SIZE / 1000.0) ** 2
    house_count = int(expected) + (1 if rng.random() < expected - int(expected) else 0)
    house_count = 0 if ground == 'water' else min(house_count, MAX_HOUSES)

    # 步骤 4：有没有火
    risk_tier = 'low' if risk is None else _tier(risk, RISK_EDGES, ('low', 'medium', 'high', 'very_high'))
    fire_prob = FIRE_PROB[risk_tier] if ground in BURNABLE else 0.0
    fire = rng.random() < fire_prob

    # 步骤 5：火势
    level, frp_source = 'none', None
    if fire:
        if frp is not None:
            level, frp_source = _tier(frp, FRP_EDGES, ('small', 'medium', 'large', 'extreme')), 'observed'
        else:
            level = rng.choices(list(FRP_DRAW), weights=list(FRP_DRAW.values()))[0]
            frp_source = 'sampled'
    crown_fire = fire and level in ('large', 'extreme') and canopy_tier in ('medium', 'dense')

    return {
        'ground': ground, 'tree_cover': tree_cover, 'house_count': house_count,
        'grass_count': round(GRASS_PER_M2.get(ground, 0) * AREA), 'shrub_count': round(SHRUB_PER_M2.get(ground, 0) * AREA),
        'fire': fire, 'fire_level': level, 'fire_location': FIRE_LOCATION.get(ground, 'none') if fire else 'none',
        'crown_fire': crown_fire, 'smoke_level': SMOKE[level], 'sky': SKY[level],
        'debug': {'inputs': inputs, 'canopy_tier': canopy_tier, 'expected_houses': round(expected, 3),
                  'risk_tier': risk_tier, 'fire_probability': fire_prob, 'frp_source': frp_source},
    }


# ---------------- 第二部分：决定每样东西摆在哪 ----------------
class _Smooth2D:
    """简单的平滑噪声（几组随机方向的正弦叠加），用来让草有疏有密、枯草成片"""
    def __init__(self, rng, scale):
        self.waves = [(rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.4) * scale)
                      for _ in range(4)]

    def __call__(self, x, y):   # 返回 0 ~ 1
        s = sum(math.sin((x * math.cos(a) + y * math.sin(a)) * k + p) for a, p, k in self.waves)
        return 0.5 + s / 8


def place(decision, rng, veg_manifest, house_manifest):
    half = SCENE_SIZE / 2
    trees_m = [a for a in veg_manifest if a['kind'] == 'tree']
    shrubs_m = [a for a in veg_manifest if a['kind'] == 'shrub']
    green_m = [a for a in veg_manifest if a['kind'] == 'grass' and a['species'] == 'green']
    dry_m = [a for a in veg_manifest if a['kind'] == 'grass' and a['species'] == 'dry']
    houses, trees, shrubs, grass = [], [], [], []

    def clear_of_houses(x, y, r):
        return all((x - h['x']) ** 2 + (y - h['y']) ** 2 > (r + h['r']) ** 2 for h in houses)

    # 1. 房子：互不重叠（用外接圆近似占地，留 1 m 间隙）
    tries = 0
    while len(houses) < decision['house_count'] and tries < 5000:
        tries += 1
        hm = rng.choice(house_manifest)
        r = math.hypot(hm['footprint_x_m'], hm['footprint_y_m']) / 2 + 0.5
        x, y = rng.uniform(-half + r, half - r), rng.uniform(-half + r, half - r)
        if clear_of_houses(x, y, r + 0.5):
            houses.append({'name': hm['name'], 'x': x, 'y': y, 'rot': rng.uniform(0, 2 * math.pi), 'r': r})

    # 2. 树：放到树冠面积加起来达到 场景面积 × 覆盖率
    target = SCENE_SIZE ** 2 * decision['tree_cover']
    covered, tries = 0.0, 0
    while covered < target and tries < 30000:
        tries += 1
        tm = rng.choice(trees_m)
        s = rng.uniform(0.8, 1.2)
        r = tm['crown_diameter_m'] * s / 2
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if not clear_of_houses(x, y, r * 0.5):
            continue
        if any((x - t['x']) ** 2 + (y - t['y']) ** 2 < ((r + t['r']) * 0.6) ** 2 for t in trees):
            continue
        trees.append({'name': tm['name'], 'x': x, 'y': y, 'rot': rng.uniform(0, 2 * math.pi), 's': s,
                      'sz': s * rng.uniform(0.9, 1.15), 'r': r, 'h': tm['height_m'] * s})
        covered += math.pi * r * r

    # 3. 灌木
    tries = 0
    while len(shrubs) < decision['shrub_count'] and tries < decision['shrub_count'] * 20:
        tries += 1
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if clear_of_houses(x, y, 1.5):
            shrubs.append({'name': rng.choice(shrubs_m)['name'], 'x': x, 'y': y,
                           'rot': rng.uniform(0, 2 * math.pi), 's': rng.uniform(0.6, 1.1)})

    # 4. 草丛：噪声控制疏密，枯草成片
    density, dryness = _Smooth2D(rng, 0.08), _Smooth2D(rng, 0.06)
    tries = 0
    while len(grass) < decision['grass_count'] and tries < decision['grass_count'] * 6:
        tries += 1
        x, y = rng.uniform(-half, half), rng.uniform(-half, half)
        if rng.random() > 0.35 + density(x, y) * 0.65 or not clear_of_houses(x, y, 0.5):
            continue
        pool = dry_m if dryness(x, y) > 0.55 else green_m
        grass.append({'name': rng.choice(pool)['name'], 'x': x, 'y': y,
                      'rot': rng.uniform(0, 2 * math.pi), 's': rng.uniform(1.0, 1.8)})

    # 5. 火：中心离房子至少 FIRE_HOUSE_GAP 米；树冠火 = 范围内的树
    fire = None
    if decision['fire']:
        radius = FIRE_RADIUS[decision['fire_level']]
        center = None
        for _ in range(2000):
            x, y = rng.uniform(-half * 0.8, half * 0.8), rng.uniform(-half * 0.8, half * 0.8)
            if clear_of_houses(x, y, FIRE_HOUSE_GAP):
                center = (x, y); break
        if center is None:          # 房子太多找不到空地：退而求其次，放在离房子最远的随机点
            center = max(((rng.uniform(-half, half), rng.uniform(-half, half)) for _ in range(500)),
                         key=lambda p: min([(p[0] - h['x']) ** 2 + (p[1] - h['y']) ** 2 for h in houses] or [0]))
        burning = []
        if decision['crown_fire']:
            burning = [t for t in trees if (t['x'] - center[0]) ** 2 + (t['y'] - center[1]) ** 2 < radius ** 2]
        fire = {'center': center, 'radius': radius, 'burning_trees': burning}

    return {'houses': houses, 'trees': trees, 'shrubs': shrubs, 'grass': grass, 'fire': fire}


# ---------------- 第三部分：写给网页端的 scene.json ----------------
def to_three(x, y, z=0.0):
    """Blender 坐标（Z 朝上）→ three.js / glTF 坐标（Y 朝上）"""
    return [round(x, 2), round(z, 2), round(-y, 2)]


def scene_json(seed, decision, layout):
    f = layout['fire']
    fire = {'enabled': False, 'level': 'none', 'location': 'none', 'center': None, 'radius': 0,
            'crown_fire': False, 'burning_trees': []}
    if f:
        fire = {'enabled': True, 'level': decision['fire_level'], 'location': decision['fire_location'],
                'center': to_three(*f['center']), 'radius': f['radius'], 'crown_fire': decision['crown_fire'],
                'burning_trees': [{'position': to_three(t['x'], t['y']), 'height': round(t['h'], 2)}
                                  for t in f['burning_trees']]}
    debug = dict(decision['debug'])
    debug['counts'] = {k: len(layout[k]) for k in ('trees', 'shrubs', 'grass', 'houses')}
    debug['tree_cover'] = round(decision['tree_cover'], 3)
    return {
        'seed': seed, 'scene_size': SCENE_SIZE, 'ground': decision['ground'], 'fire': fire,
        'smoke_level': decision['smoke_level'], 'sky': decision['sky'],
        'spawn': {'position': to_three(*SPAWN), 'look_at': to_three(0, 0, SPAWN[2])},
        'debug': debug,
    }


def run_rules(inputs, seed=None):
    """完整的「做决定」：返回 (seed, decision, layout, scene_json)"""
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2 ** 31)
    rng = random.Random(seed)
    veg, houses = load_manifests()
    decision = decide(inputs, rng)
    layout = place(decision, rng, veg, houses)
    return seed, decision, layout, scene_json(seed, decision, layout)


if __name__ == '__main__':
    demo = {'housing_density': 36, 'fuel_fbfm40': 186, 'canopy_cover': 75,
            'occupancy_probability': 0.25, 'fire_radiative_power': 60}
    seed, d, lay, js = run_rules(demo, seed=1)
    print(json.dumps(js, ensure_ascii=False, indent=2)[:1500])
