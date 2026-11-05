# ===============================================================
#  植被类型：LANDFIRE EVT（约 1000 种现有植被类型）→ 素材要用的 30 来种
#  地形文件只有 8 大类（森林、灌木、草地……）时分不出松树还是橡树、山艾还是石炭酸灌木；
#  这里按 EVT 的生活型（EVT_LF）、植被外貌（EVT_PHYS）、类群名（EVT_GP_N）和名称（EVT_NAME）归类。
#  terrain_data.py 读原始栅格时用它，把每格的类型存进地形文件（local_veg）；摆放、选素材时按类型取。
#  只用标准库，Blender 里也能导入。
#
#  检查归类：python pipeline/veg_types.py <LF20_EVT_220.csv>   （列出每个类型包含哪些 EVT，以及没归进细类的）
# ===============================================================
import csv, re, sys

# 编号固定（存进地形文件），只能往后加，不能改已有的
TYPES = (
    'unknown',              # 0  没有数据（示例数据包、远景补的格子）：按 8 大类用通用素材
    # 树
    'pine_west',            # 1  西部松林：西黄松、扭叶松、柔枝松、亚高山疏林
    'fir_spruce',           # 2  冷杉、云杉、铁杉、花旗松、红杉、雪松
    'pine_east',            # 3  东部、东南部松林：长叶松、短叶松、平铺松林、北美油松、红松、白松、北美短叶松
    'pinyon_juniper',       # 4  矮松、刺柏林
    'hardwood',             # 5  阔叶林：橡、山核桃、枫、山毛榉、桦、杨
    'oak_savanna',          # 6  稀树橡树林、活橡树（西部、草原边缘）
    'mixed_forest',         # 7  针阔混交林
    'riparian_forest',      # 8  河岸阔叶林：三角叶杨、柳、悬铃木、桤木
    'swamp_forest',         # 9  沼泽林：落羽杉、紫树、湿地阔叶林
    'tropical_forest',      # 10 热带林：硬木群落、红树林（佛罗里达南部）
    'developed_trees',      # 11 城镇里的树
    # 灌木
    'sagebrush',            # 12 山艾、盐生灌丛（灰绿色，大盆地）
    'desert_scrub',         # 13 沙漠灌丛：石炭酸灌木、黑灌木、牧豆树、肉质植物
    'chaparral',            # 14 查帕拉尔、海岸鼠尾草灌丛、灌木橡树、山桃花心木
    'riparian_shrub',       # 15 河岸、湿地灌木（柳灌丛）
    'tropical_shrub',       # 16 热带灌丛
    'alpine_shrub',         # 17 高山、苔原矮灌丛
    'shrub_other',          # 18 其他灌丛（落叶灌丛、过渡植被、外来灌木、城镇灌木）
    # 草本
    'grass_dry',            # 19 干草原：矮草、混合草原、草原草甸、外来一年生草（西部常见的金黄色）
    'prairie_tall',         # 20 高草草原、东部草原
    'meadow',               # 21 山地草甸、高山草甸、苔原
    'wetland_herb',         # 22 湿地草本：沼泽、湿草甸、泥炭地、盐沼
    'dune_grass',           # 23 海岸沙丘草
    'grass_other',          # 24 其他草地（外来多年生草、城镇草地）
    # 农业
    'crop_row',             # 25 行播作物（玉米、大豆、棉花、甘蔗）
    'crop_close',           # 26 密植作物（小麦等）
    'orchard_vineyard',     # 27 果园、葡萄园、浆果
    'pasture',              # 28 牧草地、休耕地
    # 其他
    'developed_open',       # 29 城镇开敞空间（草坪、公园、高尔夫）
    'developed',            # 30 城镇建成区（低、中、高密度）
    'road',                 # 31 道路
    'quarry',               # 32 采石场、矿坑、井场
    'desert_barren',        # 33 沙漠、沙丘、盐碱滩、劣地
    'rock_barren',          # 34 岩石、悬崖、碎石坡、高山裸地
    'sparse_other',         # 35 其他稀疏植被
    'snow_ice',             # 36 冰雪
    'water',                # 37 水面
)
ID = {t: i for i, t in enumerate(TYPES)}


def _has(text, *words):
    return any(re.search(w, text, re.I) for w in words)


def classify(row):
    """一行 EVT 对照表 → 类型名。先看类群名（EVT_GP_N），类群名说不清时再看具体名称"""
    t = _classify(row, row['EVT_GP_N'])
    return t if t is not None else _classify(row, f"{row['EVT_GP_N']} | {row['EVT_NAME']}", final=True)


def _classify(row, t, final=False):
    lf, phys = row['EVT_LF'], row['EVT_PHYS']
    tropical = _has(t, 'Caribbean', "Hawai", 'Pacific Islands', 'Polynesian', 'Mangrove', 'Tropical', 'Hammock')
    if lf == 'Water' or phys == 'Open Water':
        return 'water'
    if lf == 'Snow-Ice':
        return 'snow_ice'
    if phys.startswith('Quarries'):
        return 'quarry'
    if phys == 'Developed-Roads':
        return 'road'
    if phys == 'Developed-Open Space':
        return 'developed_open'
    if phys.startswith('Developed-') or lf == 'Developed':
        return 'developed'
    if phys == 'Agricultural' or lf == 'Agriculture':
        if _has(t, 'Aquaculture'): return 'water'
        if _has(t, 'Orchard', 'Vineyard', 'Bush fruit'): return 'orchard_vineyard'
        if _has(t, 'Pasture', 'Hay', 'Fallow', 'Idle'): return 'pasture'
        if _has(t, 'Close Grown', 'Wheat') and not _has(t, 'Row Crop'): return 'crop_close'
        return 'crop_row'
    if lf in ('Sparse', 'Barren'):
        if _has(t, 'Desert', 'Playa', 'Dune', 'Badland', 'Salt Flat', 'Sand'): return 'desert_barren'
        if _has(t, 'Cliff', 'Scree', 'Talus', 'Rock', 'Bedrock', 'Alpine', 'Volcanic', 'Barren'): return 'rock_barren'
        return 'sparse_other' if final else None

    if lf == 'Tree':
        if phys == 'Developed': return 'developed_trees'
        if tropical: return 'tropical_forest'
        if phys == 'Riparian':
            if _has(t, 'Cypress', 'Swamp', 'Tupelo', 'Peatland', 'Pocosin', 'Depressional', 'Hardwood Flatwoods',
                    'Atlantic Swamp', 'Bottomland', 'Floodplain Forests'): return 'swamp_forest'
            if _has(t, 'Pine Flatwoods', 'Pitch Pine'): return 'pine_east'
            if _has(t, 'Spruce', 'Fir', 'Hemlock', 'Red-cedar'): return 'fir_spruce'
            return 'riparian_forest'
        if phys == 'Conifer-Hardwood' or _has(t, 'Pine-Hemlock-Hardwood', 'Spruce-Fir-Hardwood', 'Conifer-Oak',
                                               'Chestnut Oak-Virginia Pine', 'Shortleaf Pine-Oak'):
            return 'mixed_forest'
        if _has(t, 'Pinyon', 'Juniper', 'Mountain Mahogany'): return 'pinyon_juniper'
        if _has(t, 'Mesquite'): return 'desert_scrub'
        if phys == 'Conifer' or phys == 'Agricultural':
            if _has(t, 'Longleaf', 'Shortleaf', 'Slash', 'Loblolly', 'Pine Flatwoods', 'Pitch Pine', 'Virginia Pine',
                    'Pond Pine', 'Sand Pine', 'Red Pine', 'White Pine', 'Jack Pine'): return 'pine_east'
            if _has(t, 'Fir', 'Spruce', 'Hemlock', 'Red-cedar', 'Redwood', 'Sitka', 'Cedar', 'Mixed Evergreen',
                    'Mixed Conifer'): return 'fir_spruce'
            if _has(t, 'Ponderosa', 'Lodgepole', 'Limber', 'Jeffrey', 'Whitebark', 'Bristlecone', 'Subalpine Woodland',
                    'Larch', 'Pine'): return 'pine_west'
            if _has(t, 'Scrub Oak', 'Chaparral'): return 'chaparral'
            return 'fir_spruce' if final else None
        if _has(t, 'Savanna', 'Live Oak', 'Western Oak', 'Post Oak', 'Bur Oak', 'Black Oak Woodland'):
            return 'oak_savanna'
        if _has(t, 'Pine'):                          # 阔叶外貌但类群里带松（如平铺松林里的阔叶型）
            return 'mixed_forest'
        return 'hardwood' if final else None

    if lf == 'Shrub':
        if tropical: return 'tropical_shrub'
        if phys == 'Riparian' or _has(t, 'Shrub Swamp', 'Willow'): return 'riparian_shrub'
        if _has(t, 'Sagebrush', 'Greasewood', 'Salt Desert', 'Bitterbrush', 'Rabbitbrush'): return 'sagebrush'
        if _has(t, 'Chaparral', 'Coastal Scrub', 'Coastal Sage', 'Scrub Oak', 'Mountain Mahogany', 'Manzanita',
                'Juniper-Oak', 'Pinyon-Juniper'): return 'chaparral'
        if _has(t, 'Creosote', 'Desert Scrub', 'Blackbrush', 'Succulent', 'Mesquite', 'Bursage', 'Joshua'):
            return 'desert_scrub'
        if _has(t, 'Alpine', 'Tundra', 'Dwarf', 'Fell'): return 'alpine_shrub'
        return 'shrub_other' if final else None

    if lf == 'Herb':
        if tropical: return 'grass_other'
        if phys == 'Riparian' or _has(t, 'Marsh', 'Wetland', 'Wet Meadow', 'Peatland', 'Swamp'): return 'wetland_herb'
        if _has(t, 'Dune', 'Littoral', 'Strand', 'Beach'): return 'dune_grass'
        if phys == 'Agricultural': return 'pasture'
        if phys == 'Developed': return 'grass_other'
        if _has(t, 'Alpine', 'Tundra', 'Fell', 'Subalpine', 'Montane', 'Meadow'): return 'meadow'
        if _has(t, 'Tallgrass', 'Prairies and Barrens', 'Inland Marshes and Prairies', 'Glades', 'Alvar'):
            return 'prairie_tall'
        if _has(t, 'Shortgrass', 'Mixedgrass', 'Steppe', 'Sand Prairie', 'Annual Grassland', 'Savanna', 'Semi-Desert',
                'Desert', 'Great Plains', 'Great Basin', 'Columbia', 'Inter-Mountain', 'California', 'Rocky Mountain',
                'Southwestern', 'Chihuahuan', 'Sonoran', 'Mojave', 'Colorado Plateau', 'Western', 'Plains'):
            return 'grass_dry'
        return 'grass_other' if final else None
    return 'unknown'


def lookup(csv_path):
    """读 EVT 对照表 → {EVT 值: 类型编号}"""
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        return {int(r['VALUE']): ID[classify(r)] for r in csv.DictReader(f) if int(r['VALUE']) >= 0}


def main():
    with open(sys.argv[1], newline='', encoding='utf-8-sig') as f:
        rows = [r for r in csv.DictReader(f) if int(r['VALUE']) >= 0]
    by = {}
    for r in rows:
        by.setdefault(classify(r), []).append(r)
    for t in TYPES:
        rs = by.get(t, [])
        print(f'== {ID[t]:2d} {t}: {len(rs)} 种')
        for g in sorted({r["EVT_GP_N"] for r in rs}):
            print('     ', g)


if __name__ == '__main__':
    main()
