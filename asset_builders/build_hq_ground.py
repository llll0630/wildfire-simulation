# ===============================================================
#  地面材质库（普通 Python：numpy + Pillow，不用 Blender）
#  地形场景三档共用这一套（档位只差细节法线的强度，见 pipeline/quality.py）。
#  贴图都来自 ambientCG（https://ambientcg.com，CC0，可商用、不用署名）。
#
#  哪种地面用在哪里由真实数据决定（pipeline/ground_rules.py）：
#    Overture 的土地用途、自然地物、表面材质、水体 → 草坪、沙滩、停车场、冰川、湿地……
#    没有 Overture 数据的地方按 LANDFIRE 地表类型 + 植被类型 → 针叶林地面、干草原、沙漠砾石……
#
#  颜色：原来就有的 6 种（草地、林地、裸地、城市、农田、水）平均颜色调成和低档 assets/ground 一样，
#  远处地形（按低档平均色上色）和近处接得上；新加的按照片原色，个别调色的写在 MATS 里。
#
#  准备：把用到的贴图包（MATS 里的名字）下载到 assets_hq/ground/sources/（不提交到 git）：
#    https://ambientcg.com/get?file=<名字>_1K-JPG.zip
#  用法：python asset_builders/build_hq_ground.py
#  输出：assets_hq/ground/textures/*.png、assets_hq/ground/ground_export_manifest.json
# ===============================================================
import os, json, zipfile, io
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOW = os.path.join(ROOT, 'assets', 'ground')
OUT = os.path.join(ROOT, 'assets_hq', 'ground')
SRC = os.path.join(OUT, 'sources')

# 材质名: (ambientCG 贴图, 一张代表几米, 颜色处理)
#   颜色处理：None = 原色；'low:<低档材质>' = 平均颜色对齐低档；(r, g, b) = 平均颜色调成这个（sRGB 0~1）；
#             ('ramp', 暗色, 亮色) = 只取照片的明暗，重新上色（原图颜色和目标差太远时用，比如绿草改成枯草）
MATS = {
    # 草
    'grass_green':      ('Grass004', 2.0, 'low:mat_grass'),       # 野草地（湿润地区）
    'grass_dry':        ('Grass007', 2.5, ('ramp', (0.30, 0.24, 0.12), (0.80, 0.70, 0.46))),   # 干草原：西部常见的金黄色
    'meadow':           ('Grass008', 2.5, (0.42, 0.47, 0.24)),    # 草甸、牧草地：比草坪暗、发黄
    'lawn':             ('Grass005', 2.0, None),                  # 修剪草坪（住宅、公园、校园）
    # 林地地面
    'litter_broadleaf': ('ScatteredLeaves009', 2.5, 'low:mat_litter'),
    'litter_conifer':   ('Ground077', 2.5, (0.40, 0.29, 0.19)),   # 松针：照片偏橙，压暗
    'scrub_soil':       ('Ground079S', 3.0, None),                # 灌丛地面：砂土 + 碎石
    # 裸地、沙漠、岩石
    'bare':             ('Ground109', 3.0, 'low:mat_bare'),
    'desert_sand':      ('Ground097', 4.0, None),
    'desert_gravel':    ('Ground110', 3.0, (0.55, 0.49, 0.42)),   # 沙漠砾石滩：照片偏灰黑，提亮偏黄
    'playa':            ('Ground102', 4.0, None),                 # 盐碱滩、压实的干土
    'beach_sand':       ('Ground093A', 4.0, None),
    'rock_grey':        ('Rock062', 4.0, None),
    'rock_red':         ('Rock029', 4.0, None),
    'garden_soil':      ('Ground036', 2.5, None),                 # 花坛、工地翻动的土
    # 冰雪、湿地、水边
    'snow':             ('Snow014', 4.0, None),
    'ice':              ('Snow005', 6.0, (0.78, 0.84, 0.90)),     # 冰川：调亮、偏蓝
    'wetland':          ('Ground083', 3.0, (0.33, 0.33, 0.20)),   # 湿地：照片偏黄，压暗成湿的褐绿色
    'shore':            ('Ground094C', 3.0, None),                # 水边一圈湿沙泥（带波纹）
    # 人工地面
    'concrete':         ('Concrete047A', 3.0, None),
    'paving':           ('PavingStones136', 2.0, None),
    'gravel':           ('Gravel043', 2.0, None),
    'dirt':             ('Ground081', 3.0, None),                 # 土路、小径
    'wood':             ('Planks037A', 2.0, None),                # 木栈道、码头
    'sports':           ('Rubber001', 2.0, (0.58, 0.27, 0.20)),   # 塑胶跑道：调成常见的砖红色
    'asphalt':          ('Road012A', 3.0, None),                  # 柏油路面
    'parking':          ('Asphalt031', 3.0, 'low:mat_urban'),     # 停车场柏油
}
WATER = {   # 水面颜色（sRGB，乘在低档水面贴图上；反光、波纹是网页端的事）
    'water_lake': (0.16, 0.30, 0.34), 'water_river': (0.27, 0.31, 0.25),
    'water_ocean': (0.10, 0.25, 0.38), 'water_pool': (0.35, 0.75, 0.85)}
ROW_SPACING = 0.76          # 农田：和低档一样的垄距（米）
FARM_ROWS = 10              # 一张农田贴图 10 垄 = 7.6 m（和低档一样）
SOIL_REPEAT = 3             # 农田贴图里土的贴图重复 3 × 3 次（约 2.5 m 一次）
DETAIL = ('Ground037', 2.5, 512)      # 细节法线：来源、每几米重复一次、边长（像素）


def cc0(asset, kind='Color'):
    """从下载的 zip 里读一张图（只读需要的那张，zip 里别的文件不解开）"""
    path = os.path.join(SRC, f'{asset}_1K-JPG.zip')
    if not os.path.exists(path):
        raise SystemExit(f'缺少 {path}，下载地址：https://ambientcg.com/get?file={asset}_1K-JPG.zip')
    with zipfile.ZipFile(path) as z:
        return Image.open(io.BytesIO(z.read(f'{asset}_1K-JPG_{kind}.jpg'))).convert('RGB')


def to_lin(im): return np.power(np.asarray(im, np.float32) / 255, 2.2)
def to_img(lin): return Image.fromarray((np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255 + 0.5).astype(np.uint8))


def low_mean(name, pick=None):
    """低档贴图的平均颜色（线性）；pick 只算挑出来的像素"""
    lin = to_lin(Image.open(os.path.join(LOW, 'textures', name + '.png')).convert('RGB')).reshape(-1, 3)
    return (lin[pick(lin)] if pick else lin).mean(axis=0)


def match(lin, target):
    """每个通道乘一个数，让平均颜色等于 target（线性）；细节、明暗变化保留"""
    return lin * (target / np.maximum(lin.reshape(-1, 3).mean(axis=0), 1e-6))


def colored(asset, how):
    lin = to_lin(cc0(asset))
    if how is None:
        return lin
    if isinstance(how, str) and how.startswith('low:'):
        return match(lin, low_mean(how[4:]))
    if how[0] == 'ramp':
        lum = lin.mean(axis=-1)
        lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
        t = np.clip((lum - lo) / (hi - lo), 0, 1)[..., None]
        a, b = (np.power(np.array(c, np.float32), 2.2) for c in how[1:])
        return a * (1 - t) + b * t
    return match(lin, np.power(np.array(how, np.float32), 2.2))


def farmland():
    """作物垄（绿）+ 土沟（褐）：颜色分别对齐低档农田贴图里绿的部分和褐的部分"""
    green = lambda p: p[:, 1] > p[:, 0] * 1.15
    soil_c = low_mean('mat_farmland', lambda p: ~green(p)); crop_c = low_mean('mat_farmland', green)
    n = 340 * SOIL_REPEAT                                  # 1020 像素：土的贴图正好重复整数次，没有接缝
    soil = to_lin(cc0('Ground103').resize((n // SOIL_REPEAT,) * 2, Image.LANCZOS))
    soil = match(np.tile(soil, (SOIL_REPEAT, SOIL_REPEAT, 1)), soil_c)
    crop = to_lin(cc0('Grass004').resize((n // SOIL_REPEAT,) * 2, Image.LANCZOS))
    crop = match(np.tile(crop, (SOIL_REPEAT, SOIL_REPEAT, 1)), crop_c)
    yy, xx = np.mgrid[0:n, 0:n] / n * ROW_SPACING * FARM_ROWS             # 米
    rng = np.random.default_rng(3)
    wob = sum(a * np.sin(2 * np.pi * (k * yy / (ROW_SPACING * FARM_ROWS) + rng.uniform())) for k, a in ((1, 0.03), (3, 0.012)))
    d = np.abs(np.mod((xx + wob) / ROW_SPACING, 1.0) - 0.5) * ROW_SPACING              # 到垄中心的距离（米）
    # 垄的宽度沿垄方向有起伏（一株一株），边缘软一点
    width = 0.2 + 0.05 * np.sin(2 * np.pi * yy / 0.3 + 3 * np.sin(xx * 7.0))
    m = np.clip((width - d) / 0.05, 0, 1)[..., None]
    return to_img(crop * m + soil * (1 - m) * (1 - 0.25 * m))


def main():
    os.makedirs(os.path.join(OUT, 'textures'), exist_ok=True)
    with open(os.path.join(LOW, 'ground_export_manifest.json'), encoding='utf-8') as f:
        low = json.load(f)
    man = {}
    for name, (asset, tile, how) in MATS.items():
        to_img(colored(asset, how)).save(os.path.join(OUT, 'textures', name + '.png'))
        man[name] = {'texture': f'textures/{name}.png', 'tile_m': tile, 'source': f'ambientCG {asset} (CC0)'}
    farmland().save(os.path.join(OUT, 'textures', 'farmland.png'))
    man['farmland'] = {'texture': 'textures/farmland.png', 'tile_m': ROW_SPACING * FARM_ROWS,
                       'source': 'ambientCG Ground103 + Grass004 (CC0)'}
    water = to_lin(Image.open(os.path.join(LOW, low['mat_water']['texture'])).convert('RGB'))
    for name, rgb in WATER.items():
        to_img(match(water, np.power(np.array(rgb, np.float32), 2.2))).save(os.path.join(OUT, 'textures', name + '.png'))
        man[name] = {'texture': f'textures/{name}.png', 'tile_m': low['mat_water']['tile_m'], 'source': 'low water texture, tinted'}
    asset, tile, px = DETAIL
    cc0(asset, 'NormalGL').resize((px, px), Image.LANCZOS).save(os.path.join(OUT, 'textures', 'detail_normal.png'))
    man['_detail_normal'] = {'texture': 'textures/detail_normal.png', 'tile_m': tile, 'source': f'ambientCG {asset} (CC0)'}
    # 旧版的贴图（mat_*）不再用，删掉免得混淆
    for f in os.listdir(os.path.join(OUT, 'textures')):
        if f.startswith('mat_'):
            os.remove(os.path.join(OUT, 'textures', f))
    with open(os.path.join(OUT, 'ground_export_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(len(man), 'materials ->', OUT)


if __name__ == '__main__':
    main()
