# ===============================================================
#  素材怎么用：每个地方用素材库里的哪个素材（只是规则表，Blender 里外都能导入，只用 numpy）
#  素材本身在 assets_hq/（asset_builders/ 生成，见 素材库说明_2026-11.md）；档位只决定从哪个档位的文件夹取。
#    远景公告板：按植被类型选树种（MID_*）
#    建筑外观：按建筑类别选外墙、屋顶、立面模块（BUILDING_*）
#    道路：按道路等级、路面材质选路带（ROAD_*）
#    林下：幼树、蕨类（UNDERSTORY_*）
#    小物件：石块、电线杆、栅栏、路灯（PROPS_*）
# ===============================================================
import zlib
import numpy as np
import veg_types as vt

# ---------------- 远景公告板 ----------------
GENERIC_PINE = [f'bb_tree_pine_0{i}' for i in range(1, 5)]
GENERIC_OAK = [f'bb_tree_oak_0{i}' for i in range(1, 5)]


def billboard_table(species):
    """{植被类型编号: [公告板名, …]}：分树种素材里用在这个植被类型的树，每款一个公告板"""
    out = {}
    for a in species:
        if a['kind'] != 'tree':
            continue
        for v in a['veg_types']:
            out.setdefault(vt.ID[v], []).append('bb_' + a['name'])
    return out


def pick_billboard(table, veg, mid_kind, rng):
    """一棵远景树用哪个公告板：有对应树种就用树种的；没有就按形状用通用松树 / 阔叶树（mid_kind：0 松、1 阔叶、-1 随机）"""
    names = table.get(int(veg))
    if not names:
        kind = mid_kind if mid_kind >= 0 else int(rng.random() < 0.5)
        names = GENERIC_OAK if kind == 1 else GENERIC_PINE
    return names[int(rng.random() * len(names)) % len(names)]


# ---------------- 建筑外观 ----------------
# Overture 建筑类别 → 用途
BUILDING_USE = {
    'residential': {'house', 'detached', 'residential', 'semidetached_house', 'terrace', 'bungalow', 'cabin', 'apartments',
                    'dormitory', 'farm', 'static_caravan', 'houseboat', 'hut'},
    'garage': {'garage', 'garages', 'carport', 'shed', 'roof', 'shelter', 'greenhouse', 'kiosk'},
    'industrial': {'industrial', 'warehouse', 'factory', 'manufacture', 'storage_tank', 'silo', 'barn', 'farm_auxiliary',
                   'cowshed', 'stable', 'hangar', 'service', 'transportation'},
}
# 每种用途可选的外墙（按建筑 id 固定选一种，重新生成也一样）。种类越多 GLB 里的贴图越多，所以每种用途只挑几种常见的
RES_WALLS = ('siding_white', 'siding_gray', 'siding_blue', 'siding_beige', 'siding_sage', 'brick_red', 'brick_tan',
             'stucco_white', 'stucco_cream', 'stucco_tan')
COM_WALLS = ('brick_red', 'brick_tan', 'concrete', 'stucco_white', 'stone', 'stucco_cream')
TOWER_WALLS = ('highrise', 'glass_curtain')
PITCHED_ROOFS = {'stucco_white': ('clay_tile',), 'stucco_cream': ('clay_tile', 'shingle_brown'), 'stucco_tan': ('clay_tile',)}
DEFAULT_PITCHED = ('shingle_gray', 'shingle_brown', 'shingle_black', 'shingle_gray')
PITCH_MAX_AREA = 400.0          # 这么大以内、接近长方形的住宅用坡屋顶（m²）
PITCH_MIN_FILL = 0.82           # 轮廓面积 / 外接矩形面积 ≥ 这个才算"接近长方形"
TOWER_HEIGHT = 30.0             # 这么高以上的商业楼用高层立面 / 玻璃幕墙


def building_look(b):
    """一栋楼：{'use', 'wall', 'ground_module', 'upper_module', 'roof', 'pitched'}。
    ground_module / upper_module = 立面模块名（buildings_manifest 里的 <外墙>__<模块>）；None = 整面墙用外墙材质本身"""
    k = zlib.crc32(b['id'].encode())
    t = (b.get('type') or '').lower()
    use = next((u for u, s in BUILDING_USE.items() if t in s), 'commercial' if t else 'residential')
    h = b['height']
    if use == 'residential' and h > 14:
        use = 'commercial'                      # 高的住宅楼（公寓）按商业楼的样子
    if use == 'residential':
        wall = RES_WALLS[k % len(RES_WALLS)]
        roof = PITCHED_ROOFS.get(wall, DEFAULT_PITCHED)[(k // 11) % len(PITCHED_ROOFS.get(wall, DEFAULT_PITCHED))]
        return {'use': use, 'wall': wall, 'ground_module': f'{wall}__window', 'upper_module': f'{wall}__window',
                'roof': roof, 'pitched': True}
    if use == 'garage':
        wall = RES_WALLS[k % 5]
        return {'use': use, 'wall': wall, 'ground_module': f'{wall}__garage', 'upper_module': None,
                'roof': DEFAULT_PITCHED[k % 4], 'pitched': True}
    if use == 'industrial':
        wall = ('metal_wall', 'concrete')[k % 2]
        return {'use': use, 'wall': wall, 'ground_module': f'{wall}__rollup', 'upper_module': f'{wall}__high_window',
                'roof': ('metal_roof', 'flat_membrane')[k % 2], 'pitched': False}
    if h >= TOWER_HEIGHT:
        wall = TOWER_WALLS[k % 2]
        return {'use': use, 'wall': wall, 'ground_module': None, 'upper_module': None,
                'roof': ('flat_gravel', 'flat_membrane')[k % 2], 'pitched': False}
    wall = COM_WALLS[k % len(COM_WALLS)]
    return {'use': use, 'wall': wall, 'ground_module': f'{wall}__storefront', 'upper_module': f'{wall}__office',
            'roof': ('flat_gravel', 'flat_membrane')[k % 2], 'pitched': False}


# ---------------- 道路 ----------------
ROAD_STRIP = {'motorway': 'highway', 'trunk': 'four_lane', 'primary': 'four_lane', 'secondary': 'two_lane',
              'tertiary': 'two_lane'}
ROAD_UNPAVED = {'unpaved': 'dirt_road', 'dirt': 'dirt_road', 'earth': 'dirt_road', 'ground': 'dirt_road', 'mud': 'dirt_road',
                'sand': 'dirt_road', 'gravel': 'gravel_road', 'fine_gravel': 'gravel_road', 'compacted': 'gravel_road',
                'pebblestone': 'gravel_road'}
PATH_PAVED = {'paved', 'asphalt', 'concrete', 'paving_stones', 'cobblestone', 'metal', 'wood'}


def road_strip(r):
    """一条路用哪张路带贴图（roads_manifest）"""
    surface = r.get('surface') or ''
    if r['kind'] == 'path':
        if r['class'] == 'track':
            return 'dirt_road'
        return 'sidewalk' if surface in PATH_PAVED or r['class'] in ('footway', 'pedestrian') and surface != 'dirt' else 'trail'
    if surface in ROAD_UNPAVED or r['class'] == 'track':
        return ROAD_UNPAVED.get(surface, 'dirt_road')
    return ROAD_STRIP.get(r['class'], 'residential')


# ---------------- 林下 ----------------
# 大树旁边长幼树：{树种: (幼树素材名前缀, 每棵大树旁边平均几棵)}
UNDERSTORY_YOUNG = {'tree_fir': ('tree_fir_young', 0.35), 'tree_hemlock': ('tree_fir_young', 0.3),
                    'tree_ponderosa': ('tree_pine_young', 0.25), 'tree_whitepine': ('tree_pine_young', 0.25),
                    'tree_longleaf': ('tree_pine_young', 0.2), 'tree_maple': ('tree_maple_young', 0.3),
                    'tree_aspen': ('tree_maple_young', 0.2), 'tree_birch': ('tree_maple_young', 0.2)}

# ---------------- 小物件 ----------------
ROCK_VEG = {'rock_barren': 0.004, 'desert_barren': 0.0015, 'sparse_other': 0.0015, 'alpine_shrub': 0.0015,
            'desert_scrub': 0.0006}           # 每平方米几块大石头（按植被类型）
ROCK_GROUND = {'rock': 0.006}                   # 按 Overture 地面类型
POLE_SPACING = {'power_line': 0, 'minor_line': 45.0, 'cable': 45.0}     # 电线杆间距（米）；power_line 用输电塔
TOWER_SPACING = 300.0
FENCE_ASSET = {'fence': 'fence_wood_privacy', 'wall': 'fence_wood_privacy', 'hedge': 'hedge', 'guard_rail': 'guardrail',
               'chain_link': 'fence_chain_link'}
FENCE_DEFAULT = 'fence_split_rail'
STREET_LIGHT_SPACING = 35.0
STREET_LIGHT_CLASSES = {'primary', 'secondary', 'tertiary', 'trunk', 'residential'}
PROPS_HALF = 150.0               # 小物件只放在这个范围内（真实素材的范围）
