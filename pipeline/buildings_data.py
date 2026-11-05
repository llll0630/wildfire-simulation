# ===============================================================
#  取真实建筑、道路：terrain.npz（取它的场景中心）→ buildings.npz（给 pipeline.py --buildings 用）
#  用普通 Python 运行（和 terrain_data.py 一样，依赖见 requirements-terrain.txt），不用 Blender
#    python buildings_data.py --terrain terrain.npz --out buildings.npz
#
#  数据：Overture Maps（OVERTURE_RELEASE 固定版本，每月升级一次，步骤见 建筑道路更新说明_2026-11.md）
#    建筑   buildings/building       轮廓 + 高度（OSM、Microsoft、Esri 等合并）
#    道路   transportation/segment   中心线 + 等级（内容来自 OSM）
#    铺装   base/land_use 的 pedestrian、base/infrastructure 的 parking
#    公园   base/land_use 的 park、recreation 等（只记在占地图里）
#    地面   base/land_use、base/land（自然地物）、base/water、base/infrastructure 的停车场、码头，加上它们的 surface
#           → ground：1 m 一格的地面类型图（草坪、草甸、沙、岩石、湿地、冰川、水……，编号见 GROUND）
#    电力线、栅栏  base/infrastructure 的 power、barrier（记在 meta 里，给以后的电线杆、栅栏素材用）
#  范围：场景中心 ±HALF 米。下载不需要 key，从 Overture 的公开 S3 按范围读取
#
#  输出 npz：
#    meta：json 字符串（status、版本、建筑和道路清单，坐标是场景坐标：x 朝东、y 朝北，单位米）
#    cls：1 m 一格的占地图（0 无 / 1 公园 / 2 铺装 / 3 小路 / 4 道路 / 5 建筑），地面贴图用
#    block：清掉树草的范围（建筑外扩 1.5 m，道路两侧各多 1 m）
#    ground：1 m 一格的地面类型（GROUND 的编号；0 = 数据里没有，用 LANDFIRE）
#  下载失败或这里没有数据时也会输出文件（status 不是 ok），场景照常生成，只是没有建筑，不会改用随机房子
#
#  Blender 端只用 load() 和 Buildings（只依赖 numpy）；下载部分用到的库在函数里才导入
# ===============================================================
import argparse, json, os, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('FIRE_TERRAIN_DATA', os.path.join(os.path.dirname(HERE), 'terrain_data'))
CACHE = os.path.join(DATA, 'buildings_cache')

FORMAT = 2                             # 输出格式版本：加了 ground、电力线、栅栏；旧缓存不再用（缓存文件名带这个数）
OVERTURE_RELEASE = '2026-10-21.0'      # 固定版本；升级方法见更新说明（Overture 只保留最近 2 ~ 3 个版本，最晚 2 个月要升一次）
HALF = 1000.0                          # 取数据的半边长（米）
CELL = 1.0                             # 占地图格子（米）
TIMEOUT_S = 60

# 占地图的类别（数字大的优先）
NONE, PARK, PAVED, PATH, ROAD, BUILDING = 0, 1, 2, 3, 4, 5

# 道路宽度（米），按 Overture 的道路等级；没有列出的不要
ROAD_WIDTH = {'motorway': 14, 'trunk': 12, 'primary': 10, 'secondary': 9, 'tertiary': 8,
              'residential': 6.5, 'unclassified': 6, 'living_street': 5, 'service': 4, 'unknown': 5}
LINK_WIDTH = 6
PATH_WIDTH = {'footway': 1.8, 'path': 1.5, 'cycleway': 2.0, 'bridleway': 1.5, 'steps': 1.8, 'track': 3.0,
              'pedestrian': 4.0}
PAVED_SURFACES = {'paved', 'asphalt', 'concrete', 'paving_stones', 'cobblestone', 'metal', 'wood'}
# 没有高度和层数时按建筑类型估计（米）
DEFAULT_HEIGHT = {'house': 7, 'detached': 7, 'residential': 7, 'semidetached_house': 7, 'terrace': 8,
                  'bungalow': 5, 'cabin': 4, 'garage': 3, 'garages': 3, 'shed': 2.5, 'carport': 3, 'hut': 3,
                  'roof': 4, 'shelter': 3, 'apartments': 12, 'dormitory': 12, 'hotel': 15, 'commercial': 9,
                  'retail': 7, 'office': 12, 'industrial': 9, 'warehouse': 8, 'school': 9, 'university': 12,
                  'college': 12, 'church': 12, 'hospital': 15, 'public': 9, 'civic': 9}
DEFAULT_HEIGHT_OTHER = 6.0
LEVEL_M = 3.2
PARK_LAND_USE = {'park', 'recreation', 'horticulture', 'managed', 'cemetery'}

# 地面类型（编号固定，只能往后加）。0 = Overture 里没有，地面按 LANDFIRE 的地表类型
GROUND = ('none', 'lawn', 'meadow', 'farmland', 'orchard', 'forest', 'scrub', 'sand', 'rock', 'wetland',
          'glacier', 'desert', 'bare_soil', 'gravel', 'concrete', 'commercial', 'campus', 'paving_stones',
          'wood_deck', 'sports_court', 'parking', 'water_river', 'water_lake', 'water_ocean', 'water_pool', 'asphalt')
G = {k: i for i, k in enumerate(GROUND)}
WATER_GROUND = {G['water_river'], G['water_lake'], G['water_ocean'], G['water_pool']}
# surface（表面材质）比用途更具体：有就按它
SURFACE_GROUND = {'asphalt': 'asphalt', 'paved': 'concrete', 'concrete': 'concrete', 'concrete_plates': 'concrete',
                  'paving_stones': 'paving_stones', 'sett': 'paving_stones', 'cobblestone': 'paving_stones',
                  'gravel': 'gravel', 'fine_gravel': 'gravel', 'compacted': 'gravel', 'pebblestone': 'gravel',
                  'dirt': 'bare_soil', 'earth': 'bare_soil', 'ground': 'bare_soil', 'unpaved': 'bare_soil',
                  'woodchips': 'bare_soil', 'grass': 'lawn', 'recreation_grass': 'lawn', 'sand': 'sand',
                  'recreation_sand': 'sand', 'wood': 'wood_deck', 'tartan': 'sports_court', 'rubber': 'sports_court',
                  'recreation_paved': 'sports_court'}
# land_use：按 class 先找，找不到再按 subtype。只表示"大范围"的（国家公园、保护区、军事区……）不用
LAND_USE_CLASS = {'farmland': 'farmland', 'farmyard': 'farmland', 'allotments': 'farmland', 'meadow': 'meadow',
                  'orchard': 'orchard', 'vineyard': 'orchard', 'plant_nursery': 'orchard', 'greenhouse_horticulture': 'concrete',
                  'animal_keeping': 'meadow', 'rough': 'meadow', 'flowerbed': 'bare_soil', 'garden': 'lawn',
                  'playground': 'sand', 'pitch': 'lawn', 'track': 'sports_court', 'plaza': 'concrete', 'pedestrian': 'concrete',
                  'construction': 'bare_soil', 'brownfield': 'bare_soil', 'landfill': 'bare_soil', 'quarry': 'rock',
                  'greenfield': 'meadow', 'railway': 'gravel', 'garages': 'concrete', 'salt_pond': 'water_lake',
                  'water_hazard': 'water_lake', 'lateral_water_hazard': 'water_lake', 'logging': 'bare_soil',
                  'grave_yard': 'lawn', 'cemetery': 'lawn', 'village_green': 'lawn', 'grass': 'lawn',
                  'golf_course': 'lawn', 'fairway': 'lawn', 'green': 'lawn', 'tee': 'lawn', 'driving_range': 'lawn',
                  'recreation_ground': 'lawn', 'park': 'lawn', 'dog_park': 'lawn', 'schoolyard': 'concrete',
                  'stadium': 'sports_court'}
LAND_USE_SUBTYPE = {'residential': 'lawn', 'park': 'lawn', 'recreation': 'lawn', 'managed': 'lawn', 'cemetery': 'lawn',
                    'golf': 'lawn', 'horticulture': 'lawn', 'agriculture': 'farmland', 'pedestrian': 'concrete',
                    'developed': 'commercial', 'education': 'campus', 'medical': 'campus', 'religious': 'campus',
                    'construction': 'bare_soil', 'landfill': 'bare_soil', 'resource_extraction': 'rock'}
# land（自然地物，来自 OSM natural）
LAND_CLASS = {'forest': 'forest', 'wood': 'forest', 'shrub': 'scrub', 'scrub': 'scrub', 'heath': 'scrub',
              'shrubbery': 'scrub', 'grass': 'meadow', 'grassland': 'meadow', 'meadow': 'meadow', 'fell': 'meadow',
              'tundra': 'meadow', 'sand': 'sand', 'beach': 'sand', 'dune': 'sand', 'shingle': 'gravel',
              'bare_rock': 'rock', 'rock': 'rock', 'scree': 'rock', 'stone': 'rock', 'cliff': 'rock',
              'wetland': 'wetland', 'glacier': 'glacier', 'desert': 'desert'}
WATER_SUBTYPE = {'river': 'water_river', 'stream': 'water_river', 'canal': 'water_river', 'drain': 'water_river',
                 'ditch': 'water_river', 'ocean': 'water_ocean', 'physical': 'water_ocean', 'lake': 'water_lake',
                 'pond': 'water_lake', 'reservoir': 'water_lake', 'water': 'water_lake', 'human_made': 'water_lake',
                 'wastewater': 'water_lake', 'spring': 'water_lake'}
WATER_POOL = {'swimming_pool', 'fountain'}
WATER_LINE_WIDTH = {'river': 12.0, 'canal': 8.0, 'stream': 3.0, 'drain': 2.0, 'ditch': 1.5}   # 只有中心线的水（米）
POWER_LINES = {'power_line', 'minor_line', 'cable'}
POWER_POINTS = {'tower', 'pole'}


def ground_type(kind, subtype, cls, surface):
    """一个 Overture 多边形 → 地面类型名（None = 不说明地面是什么）"""
    if kind == 'water':
        return 'water_pool' if cls in WATER_POOL else WATER_SUBTYPE.get(subtype)
    if surface in SURFACE_GROUND:
        return SURFACE_GROUND[surface]
    if kind == 'land_use':
        return LAND_USE_CLASS.get(cls) or LAND_USE_SUBTYPE.get(subtype)
    if kind == 'land':
        return LAND_CLASS.get(cls)
    if kind == 'infrastructure':
        return {'parking': 'parking', 'pier': 'wood_deck', 'boardwalk': 'wood_deck'}.get(cls)
    return None


# ---------------- Blender 端读取用（只依赖 numpy） ----------------
class Buildings:
    """buildings.npz 的内容。blocked / classes 按场景坐标查占地图"""
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(d['meta']))
        self.cls, self.blk = d['cls'], d['block']
        self.ground_map = d['ground'] if 'ground' in d.files else np.zeros_like(self.cls)     # 旧文件没有地面类型
        self.half, self.cell = float(d['half']), float(d['cell'])
        self.buildings, self.roads = self.meta['buildings'], self.meta['roads']
        self.power = self.meta.get('power', {'lines': [], 'poles': []})
        self.barriers = self.meta.get('barriers', [])
        self.status = self.meta['status']

    def _ij(self, x, y):
        x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
        i = np.floor((x + self.half) / self.cell).astype(int)
        j = np.floor((self.half - y) / self.cell).astype(int)
        n = self.cls.shape[0]
        inside = (i >= 0) & (j >= 0) & (i < n) & (j < n)
        return np.clip(i, 0, n - 1), np.clip(j, 0, n - 1), inside

    def blocked(self, x, y):
        i, j, inside = self._ij(x, y)
        return inside & self.blk[j, i]

    def classes(self, x, y):
        i, j, inside = self._ij(x, y)
        return np.where(inside, self.cls[j, i], NONE)

    def ground(self, x, y):
        """地面类型编号（GROUND）；范围外、没有数据是 0"""
        i, j, inside = self._ij(x, y)
        return np.where(inside, self.ground_map[j, i], 0)

    def shore(self, x, y, width):
        """离 Overture 水面 width 米以内（含水面本身）"""
        if getattr(self, '_shore', None) is None or self._shore[0] != width:
            water = np.isin(self.ground_map, list(WATER_GROUND))
            near = water.copy()
            for _ in range(max(1, int(round(width / self.cell)))):      # 每次往外扩一格
                grown = near.copy()
                grown[1:] |= near[:-1]; grown[:-1] |= near[1:]; grown[:, 1:] |= near[:, :-1]; grown[:, :-1] |= near[:, 1:]
                near = grown
            self._shore = (width, near)
        i, j, inside = self._ij(x, y)
        return inside & self._shore[1][j, i]


def load(path):
    return Buildings(path)


# ---------------- 下载和整理（普通 Python） ----------------
def origin_from_terrain(path):
    d = np.load(path, allow_pickle=False)
    if 'origin_x' not in d:
        raise SystemExit(f'{path} 里没有场景中心坐标：请用新版 terrain_data.py 重新生成地形文件')
    return float(d['origin_x']), float(d['origin_y'])


def _download(kind, bbox, release):
    from overturemaps.core import record_batch_reader
    r = record_batch_reader(kind, bbox=bbox, release=release, stac=True,
                            connect_timeout=TIMEOUT_S, request_timeout=TIMEOUT_S)
    if r is None:
        raise RuntimeError(f'Overture {kind} 下载失败')
    return r.read_all()


def _rule_value(rules, key='value'):
    """Overture 的 *_rules / road_surface 等字段：取对整段都适用的值（between 为空）"""
    for r in rules or []:
        if r.get('between') is None:
            return r.get(key)
    return None


def _ranges(flags, name):
    """road_flags 里某个标记覆盖的范围（0~1，按线长比例）"""
    out = []
    for f in flags or []:
        if name in (f.get('values') or []):
            out.append(tuple(f['between']) if f.get('between') else (0.0, 1.0))
    return out


def building_height(h, floors, cls):
    if h and 1.5 < h < 600:
        return float(h), 'height'
    if floors and 0 < floors < 200:
        return floors * LEVEL_M + 1.0, 'floors'
    return float(DEFAULT_HEIGHT.get(cls, DEFAULT_HEIGHT_OTHER)), 'default'


def fetch(cx, cy, half=HALF, release=OVERTURE_RELEASE):
    """下载并整理成场景坐标。返回 (meta, cls, block)"""
    import shapely
    from shapely.geometry import LineString, Polygon, box as make_box
    from shapely.ops import substring
    from pyproj import Transformer
    from rasterio import features
    from affine import Affine

    to_ll = Transformer.from_crs(5070, 4326, always_xy=True)
    to_xy = Transformer.from_crs(4326, 5070, always_xy=True)
    corners = [to_ll.transform(cx + sx * half, cy + sy * half) for sx in (-1, 1) for sy in (-1, 1)]
    bbox = (min(c[0] for c in corners), min(c[1] for c in corners), max(c[0] for c in corners), max(c[1] for c in corners))
    area = make_box(-half, -half, half, half)

    def local(table):
        g = shapely.from_wkb(table.column('geometry').to_pylist())
        return shapely.transform(g, lambda a: np.column_stack(to_xy.transform(a[:, 0], a[:, 1])) - [cx, cy])

    def col(table, name):
        return table.column(name).to_pylist() if name in table.column_names else [None] * table.num_rows

    timing = {}
    t0 = time.perf_counter()
    # 六类同时下载：land_use、infrastructure 的文件大，单独要 10 ~ 25 秒，一起下载总时间约等于最慢的那个
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        tb, tr, tl, ti, tland, tw = ex.map(lambda k: _download(k, bbox, release),
                                           ('building', 'segment', 'land_use', 'infrastructure', 'land', 'water'))
    timing['download'] = round(time.perf_counter() - t0, 2)

    shapes = {PARK: [], PAVED: [], PATH: [], ROAD: [], BUILDING: []}
    block = []
    buildings = []
    for g, bid, h, fl, cl, nm, src, under in zip(local(tb), col(tb, 'id'), col(tb, 'height'), col(tb, 'num_floors'),
                                                  col(tb, 'class'), col(tb, 'names'), col(tb, 'sources'), col(tb, 'is_underground')):
        if g is None or under or not g.is_valid or not g.intersects(area):
            continue
        height, how = building_height(h, fl, cl)
        datasets = '+'.join(sorted({s['dataset'] for s in (src or [])}))
        for part in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if part.geom_type != 'Polygon' or part.area < 4:
                continue
            part = part.simplify(0.15)
            ring = np.array(part.exterior.coords)[:-1]
            if not part.exterior.is_ccw:
                ring = ring[::-1]
            buildings.append({'id': bid, 'type': cl, 'name': (nm or {}).get('primary'), 'height': round(height, 1),
                              'height_source': how, 'data_source': datasets,
                              'poly': [[round(float(x), 2), round(float(y), 2)] for x, y in ring]})
            shapes[BUILDING].append(part)
            block.append(part.buffer(1.5))

    roads = []
    for g, st, cl, flags, surf, wr, nm in zip(local(tr), col(tr, 'subtype'), col(tr, 'class'), col(tr, 'road_flags'),
                                              col(tr, 'road_surface'), col(tr, 'width_rules'), col(tr, 'names')):
        if g is None or st != 'road' or g.geom_type != 'LineString':
            continue
        if cl in PATH_WIDTH:
            kind, w = 'path', PATH_WIDTH[cl]
        elif cl in ROAD_WIDTH:
            kind = 'road'
            w = LINK_WIDTH if _ranges(flags, 'is_link') else ROAD_WIDTH[cl]
        else:
            continue
        given = _rule_value(wr)
        if given and 1 <= given <= 40:
            w = given
        # 隧道部分去掉；桥目前贴着地面（已知问题）
        keep = [(0.0, 1.0)]
        for a, b in _ranges(flags, 'is_tunnel'):
            keep = [seg for k0, k1 in keep for seg in ((k0, min(k1, a)), (max(k0, b), k1)) if seg[1] - seg[0] > 1e-3]
        surface = _rule_value(surf)
        for a, b in keep:
            part = g if (a, b) == (0.0, 1.0) else substring(g, a, b, normalized=True)
            part = part.intersection(area.buffer(50))
            pieces = [part] if part.geom_type == 'LineString' else [p for p in getattr(part, 'geoms', []) if p.geom_type == 'LineString']
            for piece in pieces:
                if piece.is_empty or piece.length < 1:
                    continue
                roads.append({'class': cl, 'kind': kind, 'width': round(float(w), 1), 'surface': surface,
                              'name': (nm or {}).get('primary'), 'bridge': bool(_ranges(flags, 'is_bridge')),
                              'line': [[round(float(x), 2), round(float(y), 2)] for x, y in piece.coords]})
                shapes[ROAD if kind == 'road' else PATH].append(piece.buffer(w / 2, cap_style='flat'))
                block.append(piece.buffer(w / 2 + 1.0, cap_style='flat'))

    for g, st in zip(local(tl), col(tl, 'subtype')):
        if g is None or g.geom_type not in ('Polygon', 'MultiPolygon') or not g.intersects(area):
            continue
        if st == 'pedestrian':
            shapes[PAVED].append(g); block.append(g)
        elif st in PARK_LAND_USE:
            shapes[PARK].append(g)
    for g, st, cl in zip(local(ti), col(ti, 'subtype'), col(ti, 'class')):
        if g is not None and cl == 'parking' and g.geom_type in ('Polygon', 'MultiPolygon') and g.intersects(area):
            shapes[PAVED].append(g); block.append(g)

    # 地面类型：所有说明"地面是什么"的多边形，大的先画、小的后画（小的更具体：公园里的沙坑、住宅区里的池塘），水最后画
    ground_shapes = []
    def add_ground(table, kind):
        for g, st, cl, sf in zip(local(table), col(table, 'subtype'), col(table, 'class'), col(table, 'surface')):
            if g is None or not g.intersects(area):
                continue
            gt = ground_type(kind, st, cl, sf)
            if gt is None:
                continue
            if g.geom_type in ('LineString', 'MultiLineString'):
                if kind != 'water' or st not in WATER_LINE_WIDTH:
                    continue
                g = g.buffer(WATER_LINE_WIDTH[st] / 2)
            elif g.geom_type not in ('Polygon', 'MultiPolygon'):
                continue
            g = g.intersection(area.buffer(5))
            if not g.is_empty:
                ground_shapes.append((G[gt] in WATER_GROUND, -g.area, g, G[gt]))
    for table, kind in ((tl, 'land_use'), (tland, 'land'), (ti, 'infrastructure'), (tw, 'water')):
        add_ground(table, kind)

    # 电力线、电线杆、栅栏（以后的素材用）
    power = {'lines': [], 'poles': []}
    barriers = []
    for g, st, cl in zip(local(ti), col(ti, 'subtype'), col(ti, 'class')):
        if g is None or not g.intersects(area.buffer(50)):
            continue
        if st == 'power' and cl in POWER_LINES and g.geom_type == 'LineString':
            power['lines'].append({'class': cl, 'line': [[round(float(x), 2), round(float(y), 2)] for x, y in g.coords]})
        elif st == 'power' and cl in POWER_POINTS and g.geom_type == 'Point':
            power['poles'].append({'class': cl, 'xy': [round(float(g.x), 2), round(float(g.y), 2)]})
        elif st == 'barrier' and g.geom_type == 'LineString':
            barriers.append({'class': cl, 'line': [[round(float(x), 2), round(float(y), 2)] for x, y in g.coords]})

    n = int(round(2 * half / CELL))
    tf = Affine(CELL, 0, -half, 0, -CELL, half)
    cls = np.zeros((n, n), dtype=np.uint8)
    for c in (PARK, PAVED, PATH, ROAD, BUILDING):          # 后画的覆盖先画的
        geoms = [(g, c) for g in shapes[c] if not g.is_empty]
        if geoms:
            features.rasterize(geoms, out=cls, transform=tf, merge_alg=features.MergeAlg.replace)
    geoms = [(g, 1) for g in block if not g.is_empty]
    blk = features.rasterize(geoms, out_shape=(n, n), transform=tf, dtype=np.uint8) if geoms else np.zeros((n, n), np.uint8)
    ground = np.zeros((n, n), dtype=np.uint8)
    ground_shapes.sort(key=lambda s: (s[0], s[1]))
    if ground_shapes:
        features.rasterize([(g, k) for _, _, g, k in ground_shapes], out=ground, transform=tf,
                           merge_alg=features.MergeAlg.replace)
    timing['total'] = round(time.perf_counter() - t0, 2)

    meta = {'status': 'ok' if buildings or roads else 'no_data', 'release': release, 'half': half,
            'origin_epsg5070': [cx, cy], 'source': 'Overture Maps Foundation, © OpenStreetMap contributors (ODbL)',
            'buildings': buildings, 'roads': roads, 'power': power, 'barriers': barriers, 'format': FORMAT,
            'ground_types': {GROUND[k]: round(float(v), 4) for k, v in
                             enumerate(np.bincount(ground.ravel(), minlength=len(GROUND)) / ground.size) if k and v > 0.001},
            'timing_s': timing}
    return meta, cls, blk.astype(bool), ground


def save(out, meta, cls=None, blk=None, half=HALF, ground=None):
    n = int(round(2 * half / CELL))
    cls = np.zeros((n, n), np.uint8) if cls is None else cls
    blk = np.zeros((n, n), bool) if blk is None else blk
    ground = np.zeros((n, n), np.uint8) if ground is None else ground
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, meta=np.array(json.dumps(meta, ensure_ascii=False)), cls=cls, block=blk, ground=ground,
                        half=half, cell=CELL)


def summary(meta):
    b = meta['buildings']
    km = lambda kind: round(sum(float(np.hypot(*np.diff(np.array(r['line']), axis=0).T).sum())
                                for r in meta['roads'] if r['kind'] == kind) / 1000, 1)
    return {'status': meta['status'], 'release': meta['release'], 'buildings': len(b),
            'height_source': {s: sum(1 for x in b if x['height_source'] == s) for s in ('height', 'floors', 'default')},
            'road_km': km('road'), 'path_km': km('path'), 'ground_types': meta.get('ground_types'),
            'power_lines': len(meta.get('power', {}).get('lines', [])), 'barriers': len(meta.get('barriers', [])),
            'timing_s': meta.get('timing_s')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terrain', help='terrain_data.py 生成的地形文件（取它的场景中心）')
    ap.add_argument('--lon', type=float); ap.add_argument('--lat', type=float)
    ap.add_argument('--out', required=True)
    ap.add_argument('--release', default=OVERTURE_RELEASE, help='Overture 版本（默认用代码里固定的版本）')
    ap.add_argument('--half', type=float, default=HALF)
    ap.add_argument('--no-cache', action='store_true', help='不用、也不写缓存（缓存在 terrain_data/buildings_cache/<版本>/）')
    a = ap.parse_args()
    if a.terrain:
        cx, cy = origin_from_terrain(a.terrain)
    elif a.lon is not None and a.lat is not None:
        from pyproj import Transformer
        cx, cy = Transformer.from_crs(4326, 5070, always_xy=True).transform(a.lon, a.lat)
    else:
        ap.error('需要 --terrain 或 --lon/--lat')

    key = f'{zlib.crc32(f"{cx:.1f},{cy:.1f},{a.half:.0f}".encode()):08x}-v{FORMAT}'
    cached = os.path.join(CACHE, a.release, f'{key}.npz')
    if not a.no_cache and os.path.exists(cached):
        import shutil
        shutil.copyfile(cached, a.out)
        meta = load(a.out).meta
        print('[buildings] 用缓存', cached)
    else:
        try:
            meta, cls, blk, ground = fetch(cx, cy, a.half, a.release)
            save(a.out, meta, cls, blk, a.half, ground)
            if not a.no_cache and (meta['status'] == 'ok' or ground.any()):
                os.makedirs(os.path.dirname(cached), exist_ok=True)
                save(cached, meta, cls, blk, a.half, ground)
        except Exception as e:                        # 下载失败：照常输出（没有建筑），场景生成不受影响
            meta = {'status': 'download_failed', 'error': f'{type(e).__name__}: {e}', 'release': a.release,
                    'half': a.half, 'origin_epsg5070': [cx, cy], 'source': None, 'buildings': [], 'roads': []}
            save(a.out, meta, half=a.half)
            print('[buildings] 下载失败，场景里不会有建筑：', meta['error'], file=sys.stderr)
    print('[buildings] ' + json.dumps(summary(meta), ensure_ascii=False))


if __name__ == '__main__':
    main()
