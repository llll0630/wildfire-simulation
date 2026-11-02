# ===============================================================
#  补充地形：美国本土边界以外（海、加拿大、墨西哥）的远景数据。只需要运行一次
#    python build_world_fill.py          （setup_terrain_data.py 会自动调用）
#  输入（terrain_data/world/ 里，都是公开免费数据，setup_terrain_data.py 负责下载）：
#    etopo1/etopo1_conus_region.nc      NOAA ETOPO1 全球高度（每 1 弧分 ≈ 1.8 km），只截了本土周边
#    natural_earth/ne_10m_land.zip       Natural Earth 1:1000 万 陆地范围
#    natural_earth/ne_10m_minor_islands.zip  小岛
#    natural_earth/ne_10m_lakes.zip      大湖泊（陆地范围不扣湖，要单独扣掉）
#  输出：terrain_data/world/world_fill_3km.npz
#    和全国 3 km 网格对齐，并向四周多扩 PAD 格（约 120 km），这样靠近边缘的地点远景也完整
# ===============================================================
import io, json, os, zipfile
import numpy as np
import shapefile
import shapely
from shapely.geometry import shape, box
from shapely.ops import unary_union
from pyproj import Transformer
from scipy.io import netcdf_file

from terrain_data import BUNDLE, DATA
WORLD = os.path.join(DATA, 'world')
PAD = 40
SUB = 3                      # 每个 3 km 格子里取 3x3 个点，算陆地占比和高差


def load_polygons(name, bbox):
    z = zipfile.ZipFile(os.path.join(WORLD, 'natural_earth', f'{name}.zip'))
    r = shapefile.Reader(shp=io.BytesIO(z.read(f'{name}.shp')), shx=io.BytesIO(z.read(f'{name}.shx')),
                         dbf=io.BytesIO(z.read(f'{name}.dbf')))
    geoms = [shape(s.__geo_interface__) for s in r.shapes()]
    geoms = [g.intersection(bbox) for g in geoms if g.intersects(bbox)]
    return unary_union([g for g in geoms if not g.is_empty])


def bilinear(grid, lat0, lon0, step, lat, lon):
    v = (lat - lat0) / step; u = (lon - lon0) / step
    h, w = grid.shape
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    i, j = u.astype(int), v.astype(int); tx, ty = u - i, v - j
    return (grid[j, i] * (1 - tx) * (1 - ty) + grid[j, i + 1] * tx * (1 - ty)
            + grid[j + 1, i] * (1 - tx) * ty + grid[j + 1, i + 1] * tx * ty)


def main():
    with open(os.path.join(BUNDLE, 'manifest.json'), encoding='utf-8') as f:
        g = json.load(f)['grid']
    s = g['cell_size_m']
    W, H = g['width'] + 2 * PAD, g['height'] + 2 * PAD
    x0 = g['origin_upper_left_m'][0] - PAD * s
    y0 = g['origin_upper_left_m'][1] + PAD * s

    f = netcdf_file(os.path.join(WORLD, 'etopo1', 'etopo1_conus_region.nc'), 'r', mmap=False)
    lat_ax = f.variables['latitude'][:].astype(float)
    lon_ax = f.variables['longitude'][:].astype(float)
    elev = f.variables['altitude'][:].astype(float)
    elev[elev == 32767] = np.nan
    step = lat_ax[1] - lat_ax[0]
    f.close()

    bbox = box(lon_ax[0], lat_ax[0], lon_ax[-1], lat_ax[-1])
    land = unary_union([load_polygons('ne_10m_land', bbox), load_polygons('ne_10m_minor_islands', bbox)])
    lakes = load_polygons('ne_10m_lakes', bbox)
    shapely.prepare(land); shapely.prepare(lakes)

    to_ll = Transformer.from_crs(5070, 4326, always_xy=True)
    offs = (np.arange(SUB) + 0.5) / SUB * s
    land_frac = np.zeros((H, W)); lake_frac = np.zeros((H, W))
    z_min = np.full((H, W), np.inf); z_max = np.full((H, W), -np.inf); z_sum = np.zeros((H, W))
    cols = np.arange(W); rows = np.arange(H)
    C, R = np.meshgrid(cols, rows)
    for oy in offs:
        for ox in offs:
            X = x0 + C * s + ox; Y = y0 - R * s - oy
            lon, lat = to_ll.transform(X, Y)
            is_land = shapely.contains_xy(land, lon, lat)
            is_lake = shapely.contains_xy(lakes, lon, lat)
            z = bilinear(elev, lat_ax[0], lon_ax[0], step, lat, lon)
            land_frac += is_land & ~is_lake; lake_frac += is_lake
            z_sum += z; z_min = np.minimum(z_min, z); z_max = np.maximum(z_max, z)
    n = SUB * SUB
    land_frac /= n; lake_frac /= n
    out = os.path.join(WORLD, 'world_fill_3km.npz')
    np.savez_compressed(out, elev=(z_sum / n).astype(np.float32), relief=(z_max - z_min).astype(np.float32),
                        land_frac=land_frac.astype(np.float32), lake_frac=lake_frac.astype(np.float32),
                        pad=PAD, origin_x=x0, origin_y=y0, cell_size=s)
    print(f'{W}x{H} cells, land {land_frac.mean():.2f}, lake {lake_frac.mean():.3f}, saved {out}')


if __name__ == '__main__':
    main()
