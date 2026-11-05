# ===============================================================
#  道路素材库（普通 Python：numpy + Pillow）—— 三档
#  这一步只准备素材；pipeline 怎么用（按道路等级、路面材质、车道数选）以后再定。
#
#  每种是一条"路带"贴图：横向 = 路的整个宽度（米数写在清单里），纵向沿路方向每 length_m 米重复一次；
#  标线按美国标准（MUTCD）：中线双黄线（各 0.1 m 宽、间隔 0.1 m）、车道虚线白线（3 m 画 9 m 空）、边线白实线、
#  人行横道白色条纹、停车位白线（车位 3 m 宽、5.5 m 深）
#  路面贴图用地面素材库里已经下载的 CC0 贴图（ambientCG：Road012A 柏油、Ground081 土、Gravel043 砾石、
#  Grass004 草、Concrete047A 混凝土），放在 assets_hq/ground/sources/
#
#  三档：只差分辨率（低 每米 20 像素 / 中 50 / 高 100）；中高档另有法线贴图（标线略凸、裂缝、车辙）
#  用法：python asset_builders/build_road_materials.py
#  输出：assets_hq/roads/<档位>/*.jpg（法线 *_normal.jpg）、assets_hq/roads/roads_manifest.json、
#        assets_hq/previews/roads_<档位>.png
# ===============================================================
import os, json, zipfile, io
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'assets_hq', 'roads')
SRC = os.path.join(ROOT, 'assets_hq', 'ground', 'sources')
TIERS = {'low': dict(ppm=20, normal=False), 'medium': dict(ppm=50, normal=True), 'high': dict(ppm=100, normal=True)}
LENGTH_M = 12.0              # 沿路方向一张贴图代表几米（车道虚线 3 + 9 m 正好一个周期）

WHITE, YELLOW = (0.86, 0.86, 0.82), (0.88, 0.70, 0.18)
# 路带：名字: (路面, 宽（米）, 标线列表, 用在哪)
#   标线：('solid' | 'dashed', 中心离路中线多远（米，负 = 左）, 线宽, 颜色)
ROADS = {
    'residential':   ('asphalt', 7.0, [], 'Overture residential / living_street / service，美国住宅街道多数没有标线'),
    'two_lane':      ('asphalt', 9.0, [('solid', -0.1, 0.1, YELLOW), ('solid', 0.1, 0.1, YELLOW),
                                       ('solid', -4.2, 0.12, WHITE), ('solid', 4.2, 0.12, WHITE)],
                      'tertiary / secondary：双黄中线 + 白边线'),
    'four_lane':     ('asphalt', 15.0, [('solid', -0.1, 0.1, YELLOW), ('solid', 0.1, 0.1, YELLOW),
                                        ('dashed', -3.75, 0.12, WHITE), ('dashed', 3.75, 0.12, WHITE),
                                        ('solid', -7.2, 0.15, WHITE), ('solid', 7.2, 0.15, WHITE)],
                      'primary / trunk：双向四车道'),
    'highway':       ('asphalt', 12.0, [('solid', -5.6, 0.15, YELLOW), ('dashed', -1.85, 0.15, WHITE),
                                        ('dashed', 1.85, 0.15, WHITE), ('solid', 5.6, 0.15, WHITE)],
                      'motorway 单向三车道（左黄边线、右白边线）'),
    'parking_lot':   ('asphalt', 17.0, [('stalls', 0, 0.1, WHITE)], 'transit/parking：两排车位（各深 5.5 m）+ 中间 6 m 通道'),
    'crosswalk':     ('asphalt', 9.0, [('zebra', 0, 0.6, WHITE)], '路口人行横道（贴在路口）'),
    'dirt_road':     ('dirt', 4.0, [('ruts', 0, 0, None)], 'road_surface = dirt / earth / unpaved：两道车辙 + 中间一条草'),
    'gravel_road':   ('gravel', 5.0, [('ruts', 0, 0, None)], 'road_surface = gravel / compacted'),
    'sidewalk':      ('concrete', 1.8, [('joints', 0, 0, None)], '人行道：每 1.5 m 一道伸缩缝'),
    'curb':          ('concrete', 0.3, [], '路缘石（立面，高 0.15 m）'),
    'trail':         ('dirt', 1.5, [], 'footway / path / track 的土路小径'),
}
SURFACE = {'asphalt': ('Road012A', 3.0), 'dirt': ('Ground081', 3.0), 'gravel': ('Gravel043', 2.0),
           'concrete': ('Concrete047A', 3.0), 'grass': ('Grass004', 2.0)}


def cc0(asset, kind='Color'):
    path = os.path.join(SRC, f'{asset}_1K-JPG.zip')
    if not os.path.exists(path):
        raise SystemExit(f'缺少 {path}，下载地址：https://ambientcg.com/get?file={asset}_1K-JPG.zip')
    with zipfile.ZipFile(path) as z:
        n = next((x for x in z.namelist() if x.endswith(f'_{kind}.jpg')), None)
        return np.asarray(Image.open(io.BytesIO(z.read(n))).convert('RGB'), np.float32) / 255 if n else None


def tiled(key, w_m, ppm, normal=False):
    """路面贴图按真实尺寸铺满 w_m × LENGTH_M 米"""
    asset, tile = SURFACE[key]
    img = cc0(asset, 'NormalGL' if normal else 'Color')
    W, H = max(4, int(round(w_m * ppm))), int(round(LENGTH_M * ppm))
    t = max(4, int(round(tile * ppm)))
    small = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((t, t), Image.LANCZOS), np.float32) / 255
    return np.tile(small, (H // t + 2, W // t + 2, 1))[:H, :W].copy()


def paint(col, nrm, x0, x1, y0, y1, c, w_m, ppm, wear=0.15, rng=None):
    """在路带上画一块标线（米坐标：x 从左边缘，y 沿路）；标线有点磨损（wear），法线略凸"""
    H, W = col.shape[:2]
    a, b = int(max(0, x0) * ppm), int(min(w_m, x1) * ppm) + 1
    t, u = int(y0 * ppm), int(y1 * ppm) + 1
    a, b, t, u = max(a, 0), min(b, W), max(t, 0), min(u, H)
    if b <= a or u <= t:
        return
    m = rng.random((u - t, b - a)) > wear
    blk = col[t:u, a:b]; blk[m] = blk[m] * 0.15 + np.array(c) * 0.85
    if nrm is not None:
        e = max(1, (b - a) // 6)
        nrm[t:u, a:a + e] = (0.4, 0.5, 0.95); nrm[t:u, b - e:b] = (0.6, 0.5, 0.95)


def build_road(name, spec, ppm, normal, rng):
    surf, w_m, marks, _ = spec
    col = tiled(surf, w_m, ppm); nrm = tiled(surf, w_m, ppm, True) if normal else None
    H, W = col.shape[:2]
    mid = w_m / 2
    for kind, off, lw, c in marks:
        if kind == 'solid':
            paint(col, nrm, mid + off - lw / 2, mid + off + lw / 2, 0, LENGTH_M, c, w_m, ppm, rng=rng)
        elif kind == 'dashed':
            paint(col, nrm, mid + off - lw / 2, mid + off + lw / 2, 1.0, 4.0, c, w_m, ppm, rng=rng)
        elif kind == 'stalls':                      # 车位：中间一条通道，两边各一排车位线（3 m 一个，12 m 正好 4 个）
            for x0, x1 in ((0.0, 5.5), (w_m - 5.5, w_m)):
                y = 0.0
                while y < LENGTH_M - 1e-6:
                    paint(col, nrm, x0, x1, y, y + lw, c, w_m, ppm, rng=rng)
                    y += 3.0
        elif kind == 'zebra':                       # 人行横道：沿路 3 m 宽的一段画横向条纹（每条 0.6 m，间隔 0.6 m）
            x = 0.3
            while x < w_m - 0.3:
                paint(col, nrm, x, x + lw, 4.5, 7.5, c, w_m, ppm, wear=0.1, rng=rng)
                x += lw * 2
        elif kind == 'ruts':                        # 车辙：两条压实、偏暗的带；中间（土路）一条草
            xs = (np.arange(W) + 0.5) / ppm
            for cx in (mid - 0.85, mid + 0.85):
                k = np.exp(-((xs - cx) / 0.3) ** 2)[None, :, None]
                col[:] = col * (1 - 0.25 * k)
            if surf == 'dirt':
                grass = tiled('grass', w_m, ppm) * np.array([0.8, 0.85, 0.7])
                gk = np.clip(1 - np.abs(xs - mid) / 0.45, 0, 1)[None, :, None] ** 0.7
                wob = (0.6 + 0.4 * np.sin(np.arange(H) / ppm * 2.1 + rng.uniform(0, 6)))[:, None, None]
                col[:] = col * (1 - gk * wob) + grass * gk * wob
            # 路边渐变成路面颜色偏暗（两侧被雨水冲刷、有碎草）
            edge = np.clip(1 - np.minimum(xs, w_m - xs) / 0.35, 0, 1)[None, :, None]
            col[:] = col * (1 - 0.3 * edge)
        elif kind == 'joints':                      # 人行道伸缩缝：每 1.5 m 一道细暗线
            y = 0.0
            while y < LENGTH_M:
                t = int(y * ppm); col[t:t + max(1, ppm // 30)] *= 0.55
                if nrm is not None:
                    nrm[t:t + max(1, ppm // 30)] = (0.5, 0.3, 0.9)
                y += 1.5
    return col, nrm


def save(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=90)


def main():
    man = {'length_m': LENGTH_M, 'tiers': {k: v['ppm'] for k, v in TIERS.items()}, 'roads': {}}
    for tier, q in TIERS.items():
        rng = np.random.default_rng(3)
        for name, spec in ROADS.items():
            col, nrm = build_road(name, spec, q['ppm'], q['normal'], rng)
            save(col, os.path.join(OUT, tier, f'{name}.jpg'))
            if nrm is not None:
                save(nrm, os.path.join(OUT, tier, f'{name}_normal.jpg'))
            e = man['roads'].setdefault(name, {'surface': spec[0], 'width_m': spec[1], 'use': spec[3], 'files': {}})
            e['files'][tier] = f'{tier}/{name}.jpg'
    with open(os.path.join(OUT, 'roads_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    # 预览：每档一张，路带并排（同一比例）
    for tier, q in TIERS.items():
        s = 22 / q['ppm']
        ims = [Image.open(os.path.join(OUT, tier, f'{n}.jpg')) for n in ROADS]
        ims = [im.resize((max(2, int(im.width * s)), int(im.height * s))) for im in ims]
        sheet = Image.new('RGB', (sum(im.width for im in ims) + 20 * len(ims) + 20, ims[0].height + 40), 'white')
        d = ImageDraw.Draw(sheet); x = 20
        for n, im in zip(ROADS, ims):
            sheet.paste(im, (x, 10)); d.text((x, ims[0].height + 14), n, fill='black'); x += im.width + 20
        sheet.save(os.path.join(ROOT, 'assets_hq', 'previews', f'roads_{tier}.png'))
    print('roads done')


if __name__ == '__main__':
    main()
