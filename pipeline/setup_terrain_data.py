# ===============================================================
#  一次性准备地形数据（在运行 pipeline 的机器上执行一次）
#    pip install -r requirements-terrain.txt
#    python setup_terrain_data.py
#  会下载到 terrain_data/（仓库根目录下，不提交到 git；可用环境变量 FIRE_TERRAIN_DATA 改位置）：
#    landscape_context/   地形数据包 v1.0.1（约 78 MB，校验 SHA256）：全国 3 km 数据、读取代码、12 个示例地点
#    world/               ETOPO1 本土周边高度（约 18 MB）+ Natural Earth 陆地/小岛/湖泊（约 6 MB），
#                         然后生成 world_fill_3km.npz（约 13 MB）
#  原始栅格（LANDFIRE 2020，约 13.7 GB）不在这里下载，放好后在 terrain_data/source_paths.json 里写路径，
#  格式见 terrain_data/landscape_context/.../source_paths.example.json
# ===============================================================
import hashlib, os, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from terrain_data import DATA, BUNDLE, WORLD_FILL

RELEASE = 'https://github.com/LabRAI/fire-eye-landscape-context/releases/download/v1.0.1/'
ZIP_NAME = 'FireEye-CONUS-Landscape-Context-v1.0.1.zip'
ETOPO = ('https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?'
         'altitude%5B(20.0):1:(55.0)%5D%5B(-132.0):1:(-60.0)%5D')
NATURAL_EARTH = 'https://naciscdn.org/naturalearth/10m/physical/{}.zip'


def download(url, path):
    if os.path.exists(path):
        print('已有', path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print('下载', url)
    with urllib.request.urlopen(url, timeout=600) as r, open(path + '.part', 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + '.part', path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main():
    lc = os.path.join(DATA, 'landscape_context')
    zip_path = os.path.join(lc, ZIP_NAME)
    download(RELEASE + ZIP_NAME, zip_path)
    download(RELEASE + 'SHA256SUMS', os.path.join(lc, 'SHA256SUMS'))
    with open(os.path.join(lc, 'SHA256SUMS'), encoding='utf-8') as f:
        expected = f.read().split()[0]
    if sha256(zip_path) != expected:
        raise SystemExit(f'{ZIP_NAME} 校验失败，删掉后重新运行')
    if not os.path.exists(os.path.join(BUNDLE, 'conus_context.npz')):
        print('解压', ZIP_NAME)
        zipfile.ZipFile(zip_path).extractall(lc)

    world = os.path.join(DATA, 'world')
    download(ETOPO, os.path.join(world, 'etopo1', 'etopo1_conus_region.nc'))
    for name in ('ne_10m_land', 'ne_10m_minor_islands', 'ne_10m_lakes'):
        download(NATURAL_EARTH.format(name), os.path.join(world, 'natural_earth', f'{name}.zip'))
    if not os.path.exists(WORLD_FILL):
        print('生成', WORLD_FILL)
        import build_world_fill
        build_world_fill.main()
    print('完成。正式运行还需要原始栅格，并在', os.path.join(DATA, 'source_paths.json'), '里写好路径')


if __name__ == '__main__':
    main()
