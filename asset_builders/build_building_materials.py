# ===============================================================
#  建筑外观素材库（普通 Python：numpy + Pillow，不用 Blender）—— 三档
#  这一步只准备素材；pipeline 怎么用（按建筑类别、地区选外墙、屋顶，按层数排窗户）以后再定。
#
#  三类素材：
#    1. 外墙材质（可以无缝重复铺）：砖、石、木板墙（几种常见漆色）、木瓦墙、灰泥（几种颜色）、混凝土、波纹钢板、
#       玻璃幕墙、高层立面
#    2. 屋顶材质：沥青瓦（美国最常见，程序生成，三种颜色）、陶瓦、石板瓦、金属屋顶、平屋顶砾石、平屋顶防水卷材
#    3. 立面模块：一层楼高（3.2 m）的一段墙 + 窗 / 门 / 车库门 / 橱窗，左右可以无缝拼接；
#       一栋楼的一面墙 = 每层若干个模块横着排。外墙 × 模块的组合按用途（住宅 / 商业 / 工业）
#  CC0 贴图来自 ambientCG（https://ambientcg.com）；下载到 assets_hq/buildings/sources/（不提交到 git）：
#    https://ambientcg.com/get?file=<名字>_1K-JPG.zip
#
#  三档：只差分辨率和有没有法线贴图
#    低：每米 40 像素，只有颜色（凹凸用 AO 画进颜色里）
#    中：每米 100 像素，颜色 + 法线
#    高：每米 200 像素，颜色 + 法线
#
#  用法：python asset_builders/build_building_materials.py
#  输出：assets_hq/buildings/<档位>/{walls,roofs,modules}/*.jpg（法线贴图是 *_normal.jpg）
#        assets_hq/buildings/buildings_manifest.json（每种素材：尺寸（米）、用途、平均颜色、来源、每档文件）
#        assets_hq/previews/buildings_<档位>.png
# ===============================================================
import os, json, zipfile, io, math
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'assets_hq', 'buildings')
SRC = os.path.join(OUT, 'sources')
TIERS = {'low': dict(ppm=40, normal=False), 'medium': dict(ppm=100, normal=True), 'high': dict(ppm=200, normal=True)}
FLOOR_M = 3.2                 # 一层楼高（米）

# 外墙：名字: (贴图, 一张贴图横向代表几米, 颜色（None = 原色；(r, g, b) = 只取明暗重新上色，sRGB）, 用途)
WALLS = {
    'brick_red':       ('Bricks101', 1.6, None, ('residential', 'commercial')),
    'brick_tan':       ('Bricks105', 2.4, None, ('residential', 'commercial')),
    'stone':           ('Bricks113A', 2.5, None, ('residential', 'commercial')),
    'siding_white':    ('WoodSiding009', 2.4, (0.88, 0.88, 0.85), ('residential',)),
    'siding_gray':     ('WoodSiding009', 2.4, (0.62, 0.64, 0.64), ('residential',)),
    'siding_blue':     ('WoodSiding009', 2.4, (0.52, 0.60, 0.68), ('residential',)),
    'siding_beige':    ('WoodSiding009', 2.4, (0.80, 0.74, 0.62), ('residential',)),
    'siding_sage':     ('WoodSiding009', 2.4, (0.62, 0.67, 0.56), ('residential',)),
    'wood_shingle':    ('WoodSiding013', 1.8, None, ('residential',)),
    'stucco_white':    ('Plaster001', 3.0, (0.90, 0.89, 0.86), ('residential', 'commercial')),
    'stucco_cream':    ('Plaster001', 3.0, (0.88, 0.82, 0.70), ('residential', 'commercial')),
    'stucco_terracotta': ('Plaster001', 3.0, (0.80, 0.58, 0.45), ('residential', 'commercial')),
    'stucco_tan':      ('Plaster001', 3.0, (0.76, 0.68, 0.55), ('residential', 'commercial')),
    'concrete':        ('Concrete034', 4.0, None, ('commercial', 'industrial')),
    'metal_wall':      ('CorrugatedSteel005', 2.5, None, ('industrial',)),
    'glass_curtain':   ('Facade001', 6.0, None, ('commercial',)),          # 本身就是整面玻璃，不再加窗
    'highrise':        ('Facade018A', 12.0, None, ('commercial',)),        # 本身带窗格，不再加窗
}
SELF_WINDOWED = {'glass_curtain', 'highrise'}
# 屋顶：名字: (贴图或 'shingle:<颜色>' / 'membrane', 一张代表几米, 颜色)
ROOFS = {
    'shingle_gray':  ('shingle', 2.0, (0.38, 0.38, 0.38)),
    'shingle_brown': ('shingle', 2.0, (0.33, 0.26, 0.20)),
    'shingle_black': ('shingle', 2.0, (0.17, 0.17, 0.18)),
    'clay_tile':     ('RoofingTiles012A', 2.0, None),
    'slate':         ('RoofingTiles003', 2.0, None),
    'metal_roof':    ('CorrugatedSteel009', 2.0, None),
    'flat_gravel':   ('Gravel023', 3.0, None),
    'flat_membrane': ('membrane', 6.0, (0.62, 0.62, 0.60)),
}
ROOF_USE = {'shingle_gray': 'pitched', 'shingle_brown': 'pitched', 'shingle_black': 'pitched', 'clay_tile': 'pitched',
            'slate': 'pitched', 'metal_roof': 'pitched', 'flat_gravel': 'flat', 'flat_membrane': 'flat'}
# 立面模块：名字: (宽（米）, 用途, 画什么)
MODULES = {
    'window':       (3.0, 'residential', [('window', 1.0, 1.45, 0.5, 0.95)]),            # (种类, 中心 x（相对宽）, ……)
    'window_pair':  (4.2, 'residential', [('window', 0.3, 1.45, 0.5, 0.95), ('window', 0.7, 1.45, 0.5, 0.95)]),
    'door':         (3.0, 'residential', [('door', 0.5, 0, 0, 0)]),
    'garage':       (3.4, 'residential', [('garage', 0.5, 0, 0, 0)]),
    'blank':        (3.0, 'any', []),
    'storefront':   (4.5, 'commercial', [('storefront', 0.5, 0, 0, 0)]),
    'office':       (4.0, 'commercial', [('ribbon', 0.5, 0, 0, 0)]),
    'rollup':       (5.0, 'industrial', [('rollup', 0.5, 0, 0, 0)]),
    'high_window':  (5.0, 'industrial', [('clerestory', 0.5, 0, 0, 0)]),
}


# ---------------- 读贴图 ----------------
def cc0(asset, kind='Color'):
    path = os.path.join(SRC, f'{asset}_1K-JPG.zip')
    if not os.path.exists(path):
        raise SystemExit(f'缺少 {path}，下载地址：https://ambientcg.com/get?file={asset}_1K-JPG.zip')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        n = next((x for x in names if x.endswith(f'_{kind}.jpg')), None)
        return Image.open(io.BytesIO(z.read(n))).convert('RGB') if n else None


def to_lin(im): return np.power(np.asarray(im, np.float32) / 255, 2.2)
def to_img(lin): return Image.fromarray((np.power(np.clip(lin, 0, 1), 1 / 2.2) * 255 + 0.5).astype(np.uint8))


def recolor(lin, rgb):
    """只取明暗，按目标颜色重新上色（墙漆、灰泥的颜色）"""
    lum = lin.mean(axis=-1)
    t = lum / max(float(lum.mean()), 1e-6)
    return np.clip(np.power(np.array(rgb, np.float32), 2.2)[None, None] * np.clip(t, 0.3, 1.8)[..., None], 0, 1)


def source_maps(asset, color):
    """(颜色（线性，已乘 AO）, 法线（0~1）)"""
    col = to_lin(cc0(asset))
    ao = cc0(asset, 'AmbientOcclusion')
    if ao is not None:
        col = col * (0.55 + 0.45 * to_lin(ao.resize(cc0(asset).size)).mean(axis=-1, keepdims=True) ** 0.6)
    if color is not None:
        col = recolor(col, color)
    nrm = cc0(asset, 'NormalGL')
    nrm = np.asarray(nrm, np.float32) / 255 if nrm is not None else None
    return col, nrm


def shingle_maps(rgb, seed=5, S=1024):
    """沥青瓦：一排排错开的长方形瓦片（每排 3 片宽、上下排错开半片），颗粒感 + 每片颜色略有不同；返回颜色、法线"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:S, 0:S] / S
    rows = 8
    r = np.floor(yy * rows).astype(int)
    fx = (xx * 3 + (r % 2) * 0.5) % 1
    tab = np.floor(xx * 3 + (r % 2) * 0.5).astype(int) % 3
    fy = (yy * rows) % 1
    tone = rng.uniform(0.82, 1.18, (rows, 3))[r, tab]
    grain = rng.normal(1, 0.12, (S, S))
    h = np.clip(fy * 1.2, 0, 1) * (1 - np.exp(-np.minimum(fx, 1 - fx) * 60))       # 每片上端高、片与片之间有缝
    shade = 0.55 + 0.45 * h
    col = np.power(np.array(rgb, np.float32), 2.2)[None, None] * (tone * grain * shade)[..., None]
    gy, gx = np.gradient(h * 0.03 * S / 32)
    n = np.stack([-gx, gy, np.ones_like(gx)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return np.clip(col, 0, 1), n * 0.5 + 0.5


def membrane_maps(rgb, S=1024):
    """平屋顶防水卷材：浅灰，每隔一段有一道接缝"""
    rng = np.random.default_rng(6)
    yy, xx = np.mgrid[0:S, 0:S] / S
    seam = np.exp(-((xx * 6) % 1 - 0.0) ** 2 * 4000) + np.exp(-(((xx * 6) % 1) - 1) ** 2 * 4000)
    col = np.power(np.array(rgb, np.float32), 2.2)[None, None] * (rng.normal(1, 0.04, (S, S)) * (1 - 0.25 * seam))[..., None]
    n = np.zeros((S, S, 3), np.float32); n[..., 2] = 1
    return np.clip(col, 0, 1), n * 0.5 + 0.5


def tile_to(img, w_m, h_m, tile_m, ppm, src_aspect):
    """把一张贴图按真实尺寸铺满 w_m × h_m 米（ppm 像素每米），返回数组"""
    W, H = max(8, int(round(w_m * ppm))), max(8, int(round(h_m * ppm)))
    th = tile_m * src_aspect                                  # 贴图纵向代表几米
    tw_px, th_px = max(4, int(round(tile_m * ppm))), max(4, int(round(th * ppm)))
    t = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tw_px, th_px), Image.LANCZOS),
                   np.float32) / 255
    reps = (H // th_px + 2, W // tw_px + 2, 1)
    return np.tile(t, reps)[:H, :W]


# ---------------- 窗、门（程序画，单位米） ----------------
FRAME = (0.92, 0.92, 0.90)            # 窗框、门框（白）
GLASS_TOP, GLASS_BOT = (0.36, 0.44, 0.52), (0.10, 0.13, 0.17)


def draw_openings(col, nrm, spec, w_m, ppm, rng):
    """在一个模块（宽 w_m，高一层楼）上画窗 / 门；col 是 sRGB 0~1 数组，nrm 法线（可以是 None）"""
    H, W = col.shape[:2]
    def box(x0, y0, x1, y1, c, depth=0.0):
        """米坐标（x 从左，y 从地面往上）的长方形；depth > 0 时法线做出凹进去的边"""
        a, b = int(x0 * ppm), int(x1 * ppm); top, bot = H - int(y1 * ppm), H - int(y0 * ppm)
        a, b, top, bot = max(a, 0), min(b, W), max(top, 0), min(bot, H)
        if b <= a or bot <= top:
            return
        if callable(c):
            col[top:bot, a:b] = c(bot - top, b - a)
        else:
            col[top:bot, a:b] = c
        if nrm is not None and depth > 0:
            e = max(1, int(0.03 * ppm))
            nrm[top:top + e, a:b] = (0.5, 0.15, 0.85); nrm[bot - e:bot, a:b] = (0.5, 0.85, 0.85)
            nrm[top:bot, a:a + e] = (0.85, 0.5, 0.85); nrm[top:bot, b - e:b] = (0.15, 0.5, 0.85)
    def glass(h, w):
        t = np.linspace(0, 1, h)[:, None, None]
        g = np.array(GLASS_TOP) * (1 - t) + np.array(GLASS_BOT) * t
        streak = (np.linspace(0, 1, w)[None, :, None] * 3 + np.linspace(0, 1, h)[:, None, None]) % 1
        return np.broadcast_to(g + 0.06 * (streak < 0.15), (h, w, 3))
    def window(cx, y0, ww, hh, mull=True):
        box(cx - ww / 2 - 0.09, y0 - 0.12, cx + ww / 2 + 0.09, y0 + hh + 0.09, FRAME, 0.05)     # 窗套
        box(cx - ww / 2, y0, cx + ww / 2, y0 + hh, glass, 0.06)
        if mull:
            box(cx - ww / 2, y0 + hh / 2 - 0.025, cx + ww / 2, y0 + hh / 2 + 0.025, FRAME)      # 上下推拉窗的中横框
            box(cx - 0.02, y0, cx + 0.02, y0 + hh, FRAME)
        box(cx - ww / 2 - 0.13, y0 - 0.16, cx + ww / 2 + 0.13, y0 - 0.1, (0.8, 0.8, 0.78), 0.03)  # 窗台
    for kind, fx, a1, a2, a3 in spec:
        cx = fx * w_m if kind != 'window' else (fx * w_m if fx < 1 else w_m / 2)
        if kind == 'window':
            window(cx, 0.95, 0.95 if w_m > 3.5 else a2 * 2, 1.45)
        elif kind == 'door':
            door_c = [(0.35, 0.12, 0.10), (0.15, 0.22, 0.32), (0.85, 0.85, 0.82), (0.25, 0.25, 0.25)][rng.integers(4)]
            box(cx - 0.6, 0, cx + 0.6, 2.35, FRAME, 0.05)
            box(cx - 0.48, 0, cx + 0.48, 2.2, door_c, 0.06)
            for py in (0.35, 1.25):
                box(cx - 0.36, py, cx - 0.04, py + 0.7, tuple(v * 0.85 for v in door_c), 0.02)
                box(cx + 0.04, py, cx + 0.36, py + 0.7, tuple(v * 0.85 for v in door_c), 0.02)
            box(cx - 0.3, 1.85, cx + 0.3, 2.1, glass)
        elif kind == 'garage':
            box(cx - 1.45, 0, cx + 1.45, 2.3, FRAME, 0.05)
            box(cx - 1.35, 0, cx + 1.35, 2.2, (0.86, 0.86, 0.84), 0.06)
            for k in range(4):
                box(cx - 1.35, 0.55 * k + 0.25, cx + 1.35, 0.55 * k + 0.28, (0.7, 0.7, 0.68))
        elif kind == 'storefront':
            box(0.15, 0, w_m - 0.15, 2.9, (0.2, 0.2, 0.22), 0.05)
            box(0.25, 0.3, w_m - 0.25, 2.8, glass, 0.05)
            for k in range(1, 3):
                box(w_m * k / 3 - 0.03, 0.3, w_m * k / 3 + 0.03, 2.8, (0.2, 0.2, 0.22))
            box(0, 2.95, w_m, 3.2, (0.25, 0.27, 0.3))                                           # 招牌带
        elif kind == 'ribbon':
            box(0, 0.9, w_m, 2.6, glass, 0.04)
            for k in range(4):
                box(w_m * k / 4 - 0.03, 0.9, w_m * k / 4 + 0.03, 2.6, (0.55, 0.57, 0.6))
        elif kind == 'rollup':
            box(cx - 1.9, 0, cx + 1.9, 3.0, (0.5, 0.5, 0.48), 0.05)
            for k in range(30):
                box(cx - 1.8, k * 0.1, cx + 1.8, k * 0.1 + 0.02, (0.42, 0.42, 0.4))
        elif kind == 'clerestory':
            for k in range(3):
                x = w_m * (k + 0.5) / 3
                box(x - 0.6, 2.3, x + 0.6, 2.95, (0.5, 0.52, 0.55), 0.03); box(x - 0.55, 2.35, x + 0.55, 2.9, glass)


def lin_to_srgb(a): return np.power(np.clip(a, 0, 1), 1 / 2.2)


# ---------------- 主程序 ----------------
def save(arr, path, srgb=True):
    """存成 JPEG（这些贴图都不透明；PNG 高档要 90 MB，JPEG 小 5 倍以上）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=90)


def main():
    man = {'floor_m': FLOOR_M, 'tiers': {k: v['ppm'] for k, v in TIERS.items()}, 'walls': {}, 'roofs': {}, 'modules': {}}
    walls, roofs = {}, {}
    for name, (asset, tile_m, color, use) in WALLS.items():
        col, nrm = source_maps(asset, color)
        walls[name] = (lin_to_srgb(col), nrm, tile_m, col.shape[0] / col.shape[1])
        man['walls'][name] = {'source': f'ambientCG {asset} (CC0)', 'tile_m': tile_m,
                              'tile_h_m': round(tile_m * col.shape[0] / col.shape[1], 2), 'use': list(use),
                              'avg_rgb': [round(float(v), 3) for v in lin_to_srgb(col.reshape(-1, 3).mean(axis=0))]}
    for name, (asset, tile_m, color) in ROOFS.items():
        if asset == 'shingle':
            col, nrm = shingle_maps(color, seed=hash(name) % 100)
        elif asset == 'membrane':
            col, nrm = membrane_maps(color)
        else:
            col, nrm = source_maps(asset, color)
        roofs[name] = (lin_to_srgb(col), nrm, tile_m, col.shape[0] / col.shape[1])
        man['roofs'][name] = {'source': 'procedural' if asset in ('shingle', 'membrane') else f'ambientCG {asset} (CC0)',
                              'tile_m': tile_m, 'use': ROOF_USE[name],
                              'avg_rgb': [round(float(v), 3) for v in lin_to_srgb(col.reshape(-1, 3).mean(axis=0))]}

    for tier, q in TIERS.items():
        base = os.path.join(OUT, tier); ppm = q['ppm']
        for kind, lib in (('walls', walls), ('roofs', roofs)):
            for name, (col, nrm, tile_m, asp) in lib.items():
                px = max(64, int(round(tile_m * ppm)))
                px = min(1024, 1 << int(round(math.log2(px))))           # 2 的整数次幂，便于 mipmap
                save(np.asarray(Image.fromarray((col * 255).astype(np.uint8)).resize((px, max(16, int(px * asp))), Image.LANCZOS)) / 255,
                     os.path.join(base, kind, f'{name}.jpg'))
                if q['normal'] and nrm is not None:
                    save(np.asarray(Image.fromarray((nrm * 255).astype(np.uint8)).resize((px, max(16, int(px * asp))), Image.LANCZOS)) / 255,
                         os.path.join(base, kind, f'{name}_normal.jpg'))
        # 立面模块：外墙 × 用途相同的模块
        rng = np.random.default_rng(7)
        for wname, (col, nrm, tile_m, asp) in walls.items():
            if wname in SELF_WINDOWED:
                continue
            wuse = WALLS[wname][3]
            for mname, (w_m, muse, spec) in MODULES.items():
                if muse != 'any' and muse not in wuse:
                    continue
                c = tile_to(col, w_m, FLOOR_M, tile_m, ppm, asp).copy()
                n = tile_to(nrm, w_m, FLOOR_M, tile_m, ppm, asp).copy() if (q['normal'] and nrm is not None) else None
                draw_openings(c, n, spec, w_m, ppm, rng)
                key = f'{wname}__{mname}'
                save(c, os.path.join(base, 'modules', f'{key}.jpg'))
                if n is not None:
                    save(n, os.path.join(base, 'modules', f'{key}_normal.jpg'))
                man['modules'][key] = {'wall': wname, 'module': mname, 'width_m': w_m, 'height_m': FLOOR_M, 'use': muse}
        print(tier, 'done')
    for tier in TIERS:
        for kind in ('walls', 'roofs', 'modules'):
            for name in man[kind]:
                man[kind][name].setdefault('files', {})[tier] = f'{tier}/{kind}/{name}.jpg'
    with open(os.path.join(OUT, 'buildings_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    preview()


def preview():
    """每档一张：上面外墙、屋顶，下面拼一栋两层的住宅、一栋商业楼正面"""
    with open(os.path.join(OUT, 'buildings_manifest.json'), encoding='utf-8') as f:
        man = json.load(f)
    for tier in TIERS:
        base = os.path.join(OUT, tier); T = 150
        names = [('walls', n) for n in man['walls']] + [('roofs', n) for n in man['roofs']]
        cols = 13
        sheet = Image.new('RGB', (cols * T, ((len(names) + cols - 1) // cols) * (T + 16) + 460), 'white')
        d = ImageDraw.Draw(sheet)
        for i, (kind, n) in enumerate(names):
            im = Image.open(os.path.join(base, kind, f'{n}.jpg')).convert('RGB').resize((T, T))
            x, y = (i % cols) * T, (i // cols) * (T + 16)
            sheet.paste(im, (x, y)); d.text((x + 2, y + T + 2), n, fill='black')
        y0 = ((len(names) + cols - 1) // cols) * (T + 16) + 10
        def facade(row_specs, x0, scale):
            y = y0 + 440
            for row in row_specs:
                x = x0; hgt = 0
                for key in row:
                    im = Image.open(os.path.join(base, 'modules', f'{key}.jpg')).convert('RGB')
                    im = im.resize((int(im.width * scale), int(im.height * scale)))
                    sheet.paste(im, (x, y - im.height)); x += im.width; hgt = im.height
                y -= hgt
        s = 130 / (FLOOR_M * TIERS[tier]['ppm'])
        facade([['siding_blue__garage', 'siding_blue__window', 'siding_blue__door', 'siding_blue__window_pair'],
                ['siding_blue__window', 'siding_blue__window_pair', 'siding_blue__window', 'siding_blue__window']], 20, s)
        facade([['brick_red__storefront', 'brick_red__storefront', 'brick_red__door', 'brick_red__storefront'],
                ['brick_red__office', 'brick_red__office', 'brick_red__office', 'brick_red__office'],
                ['brick_red__office', 'brick_red__office', 'brick_red__office', 'brick_red__office']], 720, s)
        facade([['stucco_cream__window', 'stucco_cream__door', 'stucco_cream__window_pair']], 1400, s)
        os.makedirs(os.path.join(ROOT, 'assets_hq', 'previews'), exist_ok=True)
        sheet.save(os.path.join(ROOT, 'assets_hq', 'previews', f'buildings_{tier}.png'))
    print('previews saved')


if __name__ == '__main__':
    main()
