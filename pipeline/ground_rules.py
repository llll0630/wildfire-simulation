# ===============================================================
#  地面材质规则：每个地方用哪种地面贴图（材质名见 asset_builders/build_hq_ground.py 的材质库）
#    1. Overture 说明了地面是什么（草坪、沙滩、停车场、冰川、水……）→ 按它
#    2. 否则按 LANDFIRE 地表大类（森林、灌木、草地……），再按植被类型（veg_types.py）细分：
#       针叶林 → 松针地面、阔叶林 → 落叶地面、沙漠灌丛 → 砾石滩、干草原 → 枯黄草地……
#    3. 没有植被类型（示例数据、远景）→ 每个大类的默认材质
#  三档用同一套规则；只用 numpy，Blender 里外都能导入
# ===============================================================
import numpy as np
import veg_types as vt

# LANDFIRE 8 大类（terrain_core 的 F, S, G, A, D, W, B, N 顺序）的默认材质
CLASS_DEFAULT = ('litter_broadleaf', 'scrub_soil', 'grass_green', 'farmland', 'concrete', 'water_lake', 'bare', 'snow')

# 按植被类型换材质：{大类序号: {植被类型: 材质}}
BY_VEG = {
    0: {'pine_west': 'litter_conifer', 'fir_spruce': 'litter_conifer', 'pine_east': 'litter_conifer',
        'pinyon_juniper': 'scrub_soil', 'hardwood': 'litter_broadleaf', 'oak_savanna': 'grass_dry',
        'mixed_forest': 'litter_broadleaf', 'riparian_forest': 'litter_broadleaf', 'swamp_forest': 'wetland',
        'tropical_forest': 'litter_broadleaf', 'developed_trees': 'lawn'},
    1: {'sagebrush': 'scrub_soil', 'desert_scrub': 'desert_gravel', 'chaparral': 'scrub_soil',
        'riparian_shrub': 'meadow', 'tropical_shrub': 'grass_green', 'alpine_shrub': 'meadow', 'shrub_other': 'scrub_soil'},
    2: {'grass_dry': 'grass_dry', 'prairie_tall': 'grass_green', 'meadow': 'meadow', 'wetland_herb': 'wetland',
        'dune_grass': 'beach_sand', 'grass_other': 'grass_green'},
    3: {'crop_row': 'farmland', 'crop_close': 'farmland', 'orchard_vineyard': 'grass_green', 'pasture': 'meadow'},
    4: {'developed_open': 'lawn', 'developed': 'concrete', 'road': 'asphalt'},
    6: {'desert_barren': 'desert_sand', 'rock_barren': 'rock_grey', 'quarry': 'gravel', 'sparse_other': 'bare'},
}

# Overture 地面类型（buildings_data.GROUND）→ 材质；None = Overture 只说了大概（森林、沙漠），按 LANDFIRE
GROUND_MAT = {
    'lawn': 'lawn', 'meadow': 'meadow', 'farmland': 'farmland', 'orchard': 'grass_green', 'forest': None,
    'scrub': 'scrub_soil', 'sand': 'beach_sand', 'rock': 'rock_grey', 'wetland': 'wetland', 'glacier': 'ice',
    'desert': None, 'bare_soil': 'garden_soil', 'gravel': 'gravel', 'concrete': 'concrete', 'commercial': 'concrete',
    'campus': 'lawn', 'paving_stones': 'paving', 'wood_deck': 'wood', 'sports_court': 'sports', 'parking': 'parking',
    'water_river': 'water_river', 'water_lake': 'water_lake', 'water_ocean': 'water_ocean', 'water_pool': 'water_pool',
    'asphalt': 'asphalt'}

SHORE_M = 2.5              # 水边多宽画成湿沙泥（米）


def class_tables(materials):
    """{大类序号: 按植被类型编号查材质序号的数组}；materials = 材质名列表（序号就是它在列表里的位置）"""
    idx = {m: i for i, m in enumerate(materials)}
    out = {}
    for c, default in enumerate(CLASS_DEFAULT):
        table = np.full(len(vt.TYPES), idx[default], np.int32)
        for veg, mat in BY_VEG.get(c, {}).items():
            table[vt.ID[veg]] = idx[mat]
        out[c] = table
    return out


def ground_table(materials, ground_names):
    """按 Overture 地面类型编号查材质序号的数组；-1 = 不覆盖"""
    idx = {m: i for i, m in enumerate(materials)}
    return np.array([idx[GROUND_MAT[g]] if GROUND_MAT.get(g) else -1 for g in ground_names], np.int32)


def used_materials():
    """规则里会用到的全部材质名"""
    names = set(CLASS_DEFAULT) | {m for d in BY_VEG.values() for m in d.values()} | {m for m in GROUND_MAT.values() if m}
    return sorted(names | {'shore', 'dirt'})
