# ===============================================================
#  取地形：经纬度 → terrain.npz（给 pipeline.py --terrain 用）。用普通 Python 运行，不用 Blender
#  依赖见 requirements-terrain.txt；数据先用 setup_terrain_data.py 准备好
#
#  三种用法：
#    1. 正式（原始栅格，每 30 m 一格）：
#         python terrain_data.py --lon -105.28 --lat 39.99 --out terrain.npz
#       需要 terrain_data/source_paths.json 写好 4 个 LANDFIRE 文件的路径（格式见 source_paths.example.json）
#    2. 测试（数据包自带的 12 个示例地点，每 60 m 一格，不需要栅格）：
#         python terrain_data.py --example boulder_foothills --out terrain.npz
#    3. 粗略（任意坐标，只用全国 3 km 数据，近景很粗，只用来看远景）：
#         python terrain_data.py --lon -81.78 --lat 24.555 --coarse --out terrain.npz
#
#  数据来源：
#    美国本土：LANDFIRE 2020（地表类型、海拔、坡度），通过 fire-eye-landscape-context v1.0.1 数据包读取
#    本土以外（海、加拿大、墨西哥）：NOAA ETOPO1 高度 + Natural Earth 海陆/湖泊范围（build_world_fill.py）
#
#  输出 npz 的内容（坐标：EPSG:5070 米，相对场景中心）：
#    local_*：近景 + 中景（默认 6 km 见方）；far_*：远景（约 ±100 km，每 3 km 一格）
#    local_veg：每格的植被类型（veg_types.py 的编号，比 8 大类细：松 / 冷杉 / 橡树、山艾 / 石炭酸灌木……）。
#      只有原始栅格才有；示例地点和粗略模式全是 0（不知道），场景里用 8 大类的通用素材
#    缺高度或缺地表的格子（海边、国境、数据边缘）用远景数据补，补了多少记在 local_missing / local_no_cover
# ===============================================================
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import veg_types as vt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('FIRE_TERRAIN_DATA', os.path.join(os.path.dirname(HERE), 'terrain_data'))
BUNDLE = os.path.join(DATA, 'landscape_context', 'FireEye-CONUS-Landscape-Context-v1.0.1')
WORLD_FILL = os.path.join(DATA, 'world', 'world_fill_3km.npz')
SOURCES = os.path.join(DATA, 'source_paths.json')

CLASSES = ('forest_woodland', 'shrubland', 'grass_herbaceous', 'agriculture',
           'developed', 'open_water', 'sparse_barren', 'snow_ice')
FAR_RADIUS_CELLS = 34        # 远景取中心周围多少个 3 km 格子（约 100 km）
LOCAL_WIDTH_M = 6000         # 近景 + 中景用 30 m 数据的范围（边长）


def fill_nearest(a):
    """缺失的格子用最近的有效值补上"""
    a = a.copy()
    missing = np.isnan(a)
    count = int(missing.sum())
    while missing.any():
        grown = np.full(a.shape, np.nan)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown = np.where(np.isnan(grown), np.roll(np.roll(a, dr, 0), dc, 1), grown)
        a = np.where(missing, grown, a)
        missing = np.isnan(a)
    return a, count


def far_arrays(cx, cy):
    """远景：中心周围约 100 km，每格 3 km。
    美国本土以内用全国数据；以外用补充地形：海是 0 m 的水面，湖是水面，
    加拿大/墨西哥用 ETOPO1 高度、地表颜色取窗口内美国部分的平均。海岸线上的格子按占比混合"""
    with open(os.path.join(BUNDLE, 'manifest.json'), encoding='utf-8') as f:
        g = json.load(f)['grid']
    atlas = np.load(os.path.join(BUNDLE, 'conus_context.npz'), allow_pickle=False)
    wf = np.load(WORLD_FILL, allow_pickle=False)
    P = int(wf['pad'])
    gx, gy = g['origin_upper_left_m']
    size = g['cell_size_m']
    H, W = wf['elev'].shape
    col = int((cx - gx) // size) + P; row = int((gy - cy) // size) + P      # 在扩边后的网格里的位置
    r = FAR_RADIUS_CELLS
    r0, r1 = max(row - r, 0), min(row + r + 1, H)
    c0, c1 = max(col - r, 0), min(col + r + 1, W)

    def win(name):
        a = atlas[name]
        pad = [(0, 0)] * (a.ndim - 2) + [(P, P), (P, P)]
        return np.pad(a, pad)[..., r0:r1, c0:c1].astype(float)

    dem_count, dem_sum = win('dem_count'), win('dem_sum')
    counts = win('counts')
    covered = counts.sum(axis=0)
    land = wf['land_frac'][r0:r1, c0:c1].astype(float)
    lake = wf['lake_frac'][r0:r1, c0:c1].astype(float)

    # 高度：本土用 LANDFIRE；以外的陆地用 ETOPO1，海按 0 m，湖先空着再用周围补
    outside = np.where(land >= 0.5, wf['elev'][r0:r1, c0:c1], np.where(lake >= 0.5, np.nan, 0.0))
    far_elev = np.where(dem_count > 0, dem_sum / np.maximum(dem_count, 1), outside)
    far_elev, filled = fill_nearest(far_elev)
    far_relief = np.where(dem_count > 0, win('dem_max') - win('dem_min'),
                          np.where(land >= 0.5, wf['relief'][r0:r1, c0:c1], 0.0))

    # 地表：本土部分按真实占比；以外部分 = 陆地(美国部分的平均) + 水
    inside = counts / np.maximum(covered, 1)
    regional = counts.sum(axis=(1, 2))
    regional = regional / regional.sum() if regional.sum() > 0 else np.eye(8)[2]
    water = np.zeros(8); water[CLASSES.index('open_water')] = 1
    land_part = np.clip(land, 0, 1)
    out_cover = land_part[None] * regional[:, None, None] + (1 - land_part[None]) * water[:, None, None]
    in_frac = np.clip(covered / 10000, 0, 1)[None]
    far_cover = in_frac * inside + (1 - in_frac) * out_cover
    return {'far_elev': far_elev, 'far_relief': far_relief, 'far_cover': far_cover,
            'far_ox': gx + (c0 - P) * size - cx, 'far_oy': gy - (r0 - P) * size - cy, 'far_cs': size,
            'far_filled': filled, 'far_outside_cells': int((dem_count == 0).sum())}


def sample_far(far, cx, cy, x, y):
    """在远景网格上按 EPSG:5070 坐标双线性取值，返回 (高度, 地表占比)"""
    fe = far['far_elev']
    u = np.clip((x - (far['far_ox'] + cx)) / far['far_cs'] - 0.5, 0, fe.shape[1] - 1.001)
    v = np.clip(((far['far_oy'] + cy) - y) / far['far_cs'] - 0.5, 0, fe.shape[0] - 1.001)
    i0, j0 = u.astype(int), v.astype(int); tx, ty = u - i0, v - j0

    def bl(a):
        return (a[..., j0, i0] * (1 - tx) * (1 - ty) + a[..., j0, i0 + 1] * tx * (1 - ty)
                + a[..., j0 + 1, i0] * (1 - tx) * ty + a[..., j0 + 1, i0 + 1] * tx * ty)
    return bl(fe), bl(far['far_cover'])


def save_terrain(out, local_elev, local_cover, ox, oy, cs, cx, cy, local_veg=None):
    """保存地形文件。近景缺高度或缺地表的格子（海边、国境、数据边缘）用远景数据补；local_veg 是每格植被类型（没有就全 0）"""
    far = far_arrays(cx, cy)
    n = local_elev.shape[0]
    jj, ii = np.mgrid[0:n, 0:n]
    x = ox + (ii + 0.5) * cs; y = oy - (jj + 0.5) * cs                # 格子中心（EPSG:5070）
    fe, fc = sample_far(far, cx, cy, x, y)
    missing = np.isnan(local_elev)
    no_cover = np.isnan(local_cover).any(axis=0) | (np.nan_to_num(local_cover).sum(axis=0) == 0)
    local_elev = np.where(missing, fe, local_elev)
    local_cover = np.where(no_cover[None], fc, np.nan_to_num(local_cover))
    if local_veg is None:
        local_veg = np.zeros((n, n), np.uint8)
    local_veg = np.where(no_cover, 0, local_veg).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, local_elev=local_elev, local_cover=local_cover,
                        local_ox=ox - cx, local_oy=oy - cy, local_cs=cs,
                        origin_x=cx, origin_y=cy,                            # 场景中心（EPSG:5070），buildings_data.py 用
                        local_missing=int(missing.sum()), local_no_cover=int(no_cover.sum()),
                        local_veg=local_veg, veg_types=np.array(vt.TYPES),
                        **{k: v for k, v in far.items() if k not in ('far_filled', 'far_outside_cells')})
    return {'local_grid': f'{n}x{n}', 'local_cell_m': cs, 'local_missing_height': int(missing.sum()),
            'local_missing_cover': int(no_cover.sum()), 'far_outside_conus_cells': far['far_outside_cells']}


class Outside(Exception):
    pass


def open_reader(sources_path=SOURCES):
    """打开原始栅格（数据包自带的 context.SourceReader，作者用原始栅格验证过）"""
    sys.path.insert(0, BUNDLE)
    import context
    with open(sources_path, encoding='utf-8') as f:
        sources = json.load(f)
    sources.setdefault('boundary', os.path.join(BUNDLE, 'lower_48.geojson'))
    return context, context.SourceReader(sources)


def from_rasters(context, reader, lon, lat, out, width_m=LOCAL_WIDTH_M, cell_m=30):
    """读出以 (lon, lat) 为中心、边长 width_m 的方块，按 cell_m 合并后保存"""
    lon, lat = context.coordinates(lon, lat)
    if not reader.ll_prepared.covers(context.Point(lon, lat)):
        raise Outside('坐标不在美国本土 48 州范围内')
    factor = max(1, int(round(cell_m / 30)))
    n = int(math.ceil(width_m / 30 / factor)) * factor
    x, y = context.TO_XY.transform(lon, lat)
    col = int(math.floor((x - reader.transform.c) / 30 - n / 2))
    row = int(math.floor((reader.transform.f - y) / 30 - n / 2))
    t0 = time.perf_counter()
    d = reader.read(row, col, n, n)
    g = context.aggregate(d, factor)
    read_s = time.perf_counter() - t0
    counts = g['counts'].astype(float)
    covered = counts.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        cover = np.where(covered > 0, counts / covered, np.nan)
        elev = np.where(g['dem_count'] > 0, g['dem_sum'] / g['dem_count'], np.nan)
    t = d['transform']
    ox, oy, cs = t.c, t.f, factor * 30
    veg = veg_grid(reader, row, col, n, d['domain'], factor)
    info = save_terrain(out, elev, cover, ox, oy, cs, ox + n * 30 / 2, oy - n * 30 / 2, veg)
    return {**info, 'read_seconds': round(read_s, 3),
            'veg_types': {vt.TYPES[k]: round(float(v), 3) for k, v in enumerate(np.bincount(veg.ravel(), minlength=len(vt.TYPES)) / veg.size) if v > 0.005}}


def veg_grid(reader, row, col, n, domain, factor):
    """每格的植被类型：原始 EVT 栅格（和 reader.read 同一个窗口）按对照表归类，合并格子时取最多的那种"""
    from rasterio.windows import Window
    lut = np.zeros(65536, np.uint8)
    for value, k in vt.lookup(reader.paths['lookup']).items():
        lut[value] = k
    dr, dc = reader.offsets['vegetation']
    v = reader.sources['vegetation'].read(1, window=Window(col + dc, row + dr, n, n), boundless=True, masked=True)
    codes = np.where(~np.ma.getmaskarray(v) & (v.data >= 0), v.data, 0).astype(np.uint16)
    veg = np.where(domain, lut[codes], 0)
    if factor == 1:
        return veg.astype(np.uint8)
    m = n // factor
    blocks = veg.reshape(m, factor, m, factor).transpose(0, 2, 1, 3).reshape(m, m, -1)
    counts = np.stack([(blocks == k).sum(axis=-1) for k in range(len(vt.TYPES))])
    counts[0] = 0                                        # 有已知类型时不选"不知道"
    out = counts.argmax(axis=0).astype(np.uint8)
    return np.where(counts.max(axis=0) > 0, out, 0).astype(np.uint8)


def from_example(site, out):
    """数据包自带的示例地点（3 km 见方、每 60 m 一格）"""
    with open(os.path.join(BUNDLE, 'examples', f'{site}.local.json'), encoding='utf-8') as f:
        L = json.load(f)['layout']
    n, cs = L['width'], L['cell_size_m']
    ox, oy = L['origin_upper_left_m']
    to_arr = lambda v: np.array(v, dtype=float)      # null 变成 nan
    return save_terrain(out, to_arr(L['elevation_mean_m']), np.stack([to_arr(L['cover_fraction'][c]) for c in CLASSES]),
                        ox, oy, cs, ox + n * cs / 2, oy - n * cs / 2)


def coarse(lon, lat, out):
    """没有栅格时的粗略版本：近景也只用 3 km 数据"""
    from pyproj import Transformer
    cx, cy = Transformer.from_crs(4326, 5070, always_xy=True).transform(lon, lat)
    n, cs = 50, 60
    return save_terrain(out, np.full((n, n), np.nan), np.full((8, n, n), np.nan),
                        cx - n * cs / 2, cy + n * cs / 2, cs, cx, cy)


def main():
    ap = argparse.ArgumentParser(description='经纬度 → terrain.npz')
    ap.add_argument('--lon', type=float)
    ap.add_argument('--lat', type=float)
    ap.add_argument('--example', help='用数据包自带的示例地点（不需要栅格）')
    ap.add_argument('--coarse', action='store_true', help='只用全国 3 km 数据（不需要栅格）')
    ap.add_argument('--sources', default=SOURCES, help='原始栅格路径 json')
    ap.add_argument('--width', type=int, default=LOCAL_WIDTH_M, help='读取范围边长（米）')
    ap.add_argument('--cell', type=int, default=30, help='格子大小（米，30 的倍数）')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.example:
        info = from_example(a.example, a.out)
    elif a.lon is None or a.lat is None:
        ap.error('需要 --lon --lat，或者 --example')
    elif a.coarse:
        info = coarse(a.lon, a.lat, a.out)
    else:
        context, reader = open_reader(a.sources)
        try:
            info = from_rasters(context, reader, a.lon, a.lat, a.out, a.width, a.cell)
        except Outside as e:
            print(json.dumps({'status': 'outside_conus', 'reason': str(e)}, ensure_ascii=False))
            sys.exit(2)
        finally:
            reader.close()
    print(json.dumps({'status': 'ok', 'seconds': round(time.perf_counter() - t0, 2), **info}, ensure_ascii=False))


if __name__ == '__main__':
    main()
