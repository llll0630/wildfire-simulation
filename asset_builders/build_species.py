# ===============================================================
#  分树种的植被素材（三档）—— Blender 5.x
#  按 LANDFIRE 植被类型（pipeline/veg_types.py）和 Overture 地面类型选用：
#    西黄松、冷杉云杉、长叶松、刺柏、落羽杉、棕榈、三角叶杨、活橡树、枫树（树）
#    山艾、石炭酸灌木、查帕拉尔、柳灌丛、锯齿棕、约书亚树、仙人掌、高山矮灌丛（灌木）
#    芦苇香蒲、海滨沙丘草、草甸野花（草本）
#
#  中档、高档：和 build_hq_vegetation.py 一样的做法（递归枝干 + 透明贴图叶片卡片），叶片 / 针叶 / 棕榈叶等贴图用 numpy 画；
#  低档：低多边形 + 顶点颜色（和 assets/ 里原来的低档素材同一种风格）。三档同名、尺寸一样。
#  尺寸按真实树种（米）；每种素材用在哪些植被类型 / 地面类型写在 SPECIES 里，存进清单给 pipeline 选素材用。
#
#  用法：blender -b --factory-startup --python build_species.py -- [low] [medium] [high] [--preview]
#  输出：assets_hq/<档位>/species/{trees,shrubs,grass}/<类>_export.blend（贴图打包在里面）
#        assets_hq/<档位>/species/textures/*.png
#        assets_hq/species_manifest.json（素材清单：名字、种类、尺寸、用在哪、每档三角形数）
#        assets_hq/previews/species_<档位>.png（--preview）
# ===============================================================
import bpy, bmesh, math, random, os, sys, json
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_hq_vegetation as hv          # 贴图工具、导出材质（同一套做法）

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'assets_hq')
TIERS = ('low', 'medium', 'high')
# 中高档的精细程度：贴图边长、卡片分段、卡片密度、枝干边数
Q = {'medium': dict(tex=512, card_rows=1, density=0.55, scale=1.3, sides=(5, 3, 3), stride=2),
     'high':   dict(tex=1024, card_rows=2, density=1.0, scale=1.0, sides=(8, 4, 3), stride=1)}

# ---------------- 素材表 ----------------
# 每种：kind（trees / shrubs / grass）、archetype（生成方法）、H 高、D 冠幅（米）、veg（植被类型）、ground（Overture 地面类型）、
#       foliage（叶片贴图）、bark（树皮颜色：暗, 亮，线性）、tint（叶色乘数，两色之间随机）、低档颜色，以及各生成方法自己的参数
SPECIES = {
    # ---- 针叶树 ----
    'tree_ponderosa': dict(kind='trees', arche='conifer', variants=3, H=(17, 20, 24), D=(7, 8, 9.5),
                           veg=['pine_west', 'mixed_forest'], foliage='needles_long', bark=((0.09, 0.04, 0.02), (0.36, 0.17, 0.08)),
                           tint=((0.62, 0.72, 0.5), (0.78, 0.85, 0.58)), clear=0.45, shape='round_cone', whorl=0.8,
                           skip=0.15, droop=0.1, tips=0.65, card=1.5, dens=1.6, low=((0.10, 0.20, 0.08), (0.30, 0.16, 0.08))),
    'tree_fir':       dict(kind='trees', arche='conifer', variants=3, H=(18, 22, 27), D=(5, 6, 7),
                           veg=['fir_spruce'], foliage='needles_short', bark=((0.05, 0.04, 0.035), (0.22, 0.18, 0.15)),
                           tint=((0.7, 0.85, 0.85), (0.85, 0.95, 0.9)), clear=0.12, shape='cone', whorl=0.5,
                           skip=0.0, droop=0.35, tips=1.0, card=1.25, dens=1.8, low=((0.04, 0.11, 0.07), (0.17, 0.13, 0.1))),
    'tree_longleaf':  dict(kind='trees', arche='conifer', variants=2, H=(22, 26), D=(7, 8),
                           veg=['pine_east'], foliage='needles_long', bark=((0.08, 0.05, 0.03), (0.30, 0.20, 0.13)),
                           tint=((0.7, 0.8, 0.55), (0.82, 0.9, 0.6)), clear=0.66, shape='clumps', whorl=0.9,
                           skip=0.2, droop=-0.05, tips=0.55, card=1.6, dens=1.6, low=((0.12, 0.22, 0.08), (0.28, 0.2, 0.13))),
    'tree_juniper':   dict(kind='trees', arche='conifer', variants=2, H=(4.5, 6), D=(3.5, 4.5),
                           veg=['pinyon_juniper'], foliage='scale', bark=((0.10, 0.08, 0.06), (0.35, 0.30, 0.25)),
                           tint=((0.8, 0.9, 0.8), (0.95, 1.0, 0.85)), clear=0.05, shape='bush', whorl=0.35,
                           skip=0.1, droop=-0.1, tips=1.0, card=0.9, low=((0.13, 0.19, 0.12), (0.3, 0.26, 0.21))),
    'tree_cypress':   dict(kind='trees', arche='conifer', variants=2, H=(18, 22), D=(6, 7.5),
                           veg=['swamp_forest'], foliage='feather', bark=((0.10, 0.05, 0.03), (0.38, 0.22, 0.14)),
                           tint=((0.95, 1.05, 0.7), (1.05, 1.1, 0.75)), clear=0.4, shape='flat_top', whorl=0.8,
                           skip=0.1, droop=0.15, tips=1.0, card=1.4, dens=2.4, buttress=1.0, low=((0.16, 0.25, 0.1), (0.3, 0.18, 0.12))),
    # ---- 阔叶树 ----
    'tree_cottonwood': dict(kind='trees', arche='broadleaf', variants=2, H=(17, 20), D=(13, 15),
                            veg=['riparian_forest'], foliage='leaf_round', bark=((0.10, 0.09, 0.08), (0.42, 0.40, 0.36)),
                            tint=((0.95, 1.05, 0.7), (1.1, 1.15, 0.75)), trunk=0.35, crown_z=0.62, crown_h=0.4, spread=0.55,
                            low=((0.2, 0.32, 0.08), (0.35, 0.33, 0.3))),
    'tree_liveoak':   dict(kind='trees', arche='broadleaf', variants=2, H=(10, 12), D=(17, 20),
                           veg=['oak_savanna'], foliage='leaf_oval', bark=((0.05, 0.045, 0.04), (0.20, 0.18, 0.15)),
                           tint=((0.65, 0.8, 0.55), (0.8, 0.9, 0.6)), trunk=0.28, crown_z=0.6, crown_h=0.38, spread=0.7, dens=2.0,
                           low=((0.07, 0.15, 0.05), (0.2, 0.18, 0.15))),
    'tree_maple':     dict(kind='trees', arche='broadleaf', variants=2, H=(15, 18), D=(10, 12),
                           veg=['hardwood', 'developed_trees', 'mixed_forest', 'tropical_forest'], foliage='leaf_lobed',
                           bark=((0.06, 0.055, 0.05), (0.26, 0.24, 0.21)), tint=((0.85, 1.0, 0.7), (1.0, 1.1, 0.75)),
                           trunk=0.3, crown_z=0.62, crown_h=0.4, spread=0.5, low=((0.12, 0.25, 0.06), (0.25, 0.23, 0.2))),
    'tree_palm':      dict(kind='trees', arche='palm', variants=2, H=(9, 12), D=(5, 5.5),
                           veg=['tropical_forest'], foliage='palm_fan', bark=((0.12, 0.11, 0.09), (0.40, 0.36, 0.30)),
                           tint=((0.75, 0.9, 0.7), (0.9, 1.0, 0.75)), fronds=16, low=((0.14, 0.27, 0.1), (0.36, 0.32, 0.26))),
    # ---- 更多树种、幼树（只准备素材：veg 为空，pipeline 不会自动选它们；cand = 以后打算用在哪些植被类型） ----
    'tree_aspen':     dict(kind='trees', arche='broadleaf', variants=2, H=(12, 15), D=(5, 6), veg=[], cand=['hardwood'],
                           foliage='leaf_round', bark=((0.30, 0.30, 0.27), (0.72, 0.71, 0.66)),
                           tint=((0.95, 1.1, 0.7), (1.1, 1.15, 0.7)), trunk=0.45, crown_z=0.7, crown_h=0.3, spread=0.2,
                           low=((0.22, 0.36, 0.1), (0.6, 0.6, 0.56))),
    'tree_birch':     dict(kind='trees', arche='broadleaf', variants=2, H=(12, 14), D=(6, 7), veg=[], cand=['hardwood'],
                           foliage='leaf_oval', bark=((0.35, 0.34, 0.32), (0.80, 0.79, 0.76)),
                           tint=((0.9, 1.05, 0.7), (1.05, 1.1, 0.75)), trunk=0.4, crown_z=0.65, crown_h=0.32, spread=0.3,
                           low=((0.2, 0.33, 0.09), (0.65, 0.64, 0.6))),
    'tree_sycamore':  dict(kind='trees', arche='broadleaf', variants=2, H=(20, 24), D=(15, 17), veg=[], cand=['riparian_forest'],
                           foliage='leaf_lobed', bark=((0.25, 0.23, 0.19), (0.68, 0.65, 0.57)),
                           tint=((0.9, 1.0, 0.7), (1.05, 1.1, 0.75)), trunk=0.32, crown_z=0.62, crown_h=0.38, spread=0.6,
                           low=((0.16, 0.28, 0.08), (0.55, 0.52, 0.45))),
    'tree_mesquite':  dict(kind='trees', arche='broadleaf', variants=2, H=(5, 7), D=(6, 8), veg=[], cand=['desert_scrub'],
                           foliage='leaf_small', bark=((0.06, 0.045, 0.035), (0.25, 0.19, 0.14)),
                           tint=((0.45, 0.55, 0.28), (0.55, 0.62, 0.32)), trunk=0.15, crown_z=0.6, crown_h=0.35, spread=0.9,
                           low=((0.18, 0.24, 0.1), (0.22, 0.17, 0.12))),
    'tree_mangrove':  dict(kind='trees', arche='broadleaf', variants=2, H=(5, 7), D=(6, 7), veg=[], cand=['tropical_forest'],
                           foliage='leaf_oval', bark=((0.08, 0.07, 0.05), (0.30, 0.27, 0.2)),
                           tint=((0.6, 0.75, 0.5), (0.72, 0.85, 0.55)), trunk=0.15, crown_z=0.6, crown_h=0.38, spread=0.6,
                           low=((0.1, 0.2, 0.07), (0.25, 0.22, 0.16))),
    'tree_whitepine': dict(kind='trees', arche='conifer', variants=2, H=(22, 26), D=(9, 10), veg=[], cand=['pine_east', 'mixed_forest'],
                           foliage='needles_long', bark=((0.06, 0.05, 0.04), (0.24, 0.20, 0.16)),
                           tint=((0.6, 0.75, 0.65), (0.72, 0.85, 0.7)), clear=0.3, shape='round_cone', whorl=0.7,
                           skip=0.05, droop=0.0, tips=1.0, card=1.7, dens=2.2, low=((0.08, 0.17, 0.09), (0.22, 0.18, 0.14))),
    'tree_hemlock':   dict(kind='trees', arche='conifer', variants=2, H=(20, 24), D=(7, 8), veg=[], cand=['mixed_forest', 'fir_spruce'],
                           foliage='needles_short', bark=((0.07, 0.05, 0.04), (0.28, 0.20, 0.15)),
                           tint=((0.65, 0.8, 0.65), (0.78, 0.9, 0.7)), clear=0.1, shape='cone', whorl=0.5,
                           skip=0.0, droop=0.5, tips=1.0, card=1.35, dens=1.1, low=((0.05, 0.13, 0.07), (0.2, 0.15, 0.11))),
    'tree_redwood':   dict(kind='trees', arche='conifer', variants=2, H=(38, 45), D=(9, 10), veg=[], cand=['fir_spruce'],
                           foliage='feather', bark=((0.12, 0.05, 0.03), (0.42, 0.20, 0.12)),
                           tint=((0.75, 0.9, 0.65), (0.85, 0.95, 0.7)), clear=0.35, shape='cone', whorl=0.9,
                           skip=0.05, droop=0.2, tips=1.0, card=1.9, dens=2.0, buttress=1.0,
                           low=((0.07, 0.16, 0.07), (0.32, 0.16, 0.1))),
    'tree_fir_young':  dict(kind='trees', arche='conifer', variants=2, H=(3.0, 4.5), D=(1.8, 2.2), veg=[], cand=['fir_spruce'],
                            foliage='needles_short', bark=((0.05, 0.04, 0.035), (0.22, 0.18, 0.15)),
                            tint=((0.7, 0.85, 0.85), (0.85, 0.95, 0.9)), clear=0.03, shape='cone', whorl=0.35,
                            skip=0.0, droop=0.2, tips=1.0, card=0.9, dens=1.1, low=((0.04, 0.11, 0.07), (0.17, 0.13, 0.1))),
    'tree_pine_young': dict(kind='trees', arche='conifer', variants=2, H=(3.0, 4.5), D=(2.0, 2.4), veg=[], cand=['pine_west', 'pine_east'],
                            foliage='needles_long', bark=((0.09, 0.04, 0.02), (0.36, 0.17, 0.08)),
                            tint=((0.62, 0.72, 0.5), (0.78, 0.85, 0.58)), clear=0.08, shape='round_cone', whorl=0.4,
                            skip=0.05, droop=0.0, tips=0.8, card=1.0, dens=1.6, low=((0.1, 0.2, 0.08), (0.3, 0.16, 0.08))),
    'tree_maple_young': dict(kind='trees', arche='broadleaf', variants=2, H=(3.5, 5.0), D=(2.0, 2.8), veg=[], cand=['hardwood'],
                             foliage='leaf_lobed', bark=((0.06, 0.055, 0.05), (0.26, 0.24, 0.21)),
                             tint=((0.85, 1.0, 0.7), (1.0, 1.1, 0.75)), trunk=0.4, crown_z=0.62, crown_h=0.4, spread=0.3,
                             low=((0.12, 0.25, 0.06), (0.25, 0.23, 0.2))),
    # ---- 灌木 ----
    'shrub_sage':     dict(kind='shrubs', arche='shrub', variants=3, H=(0.9, 1.1, 1.3), D=(1.2, 1.4, 1.6),
                           veg=['sagebrush'], foliage='leaf_small', bark=((0.08, 0.07, 0.06), (0.25, 0.22, 0.19)),
                           tint=((0.78, 0.84, 0.76), (0.9, 0.94, 0.82)), stems=9, upright=0.35, cards=4.5, card=0.45,
                           low=((0.2, 0.23, 0.18), (0.2, 0.18, 0.15))),
    'shrub_creosote': dict(kind='shrubs', arche='shrub', variants=3, H=(1.5, 1.8, 2.2), D=(1.8, 2.1, 2.4),
                           veg=['desert_scrub'], foliage='leaf_small', bark=((0.07, 0.06, 0.05), (0.25, 0.22, 0.18)),
                           tint=((0.55, 0.62, 0.3), (0.66, 0.72, 0.36)), stems=14, upright=0.75, cards=1.6, card=0.55,
                           low=((0.2, 0.25, 0.1), (0.2, 0.17, 0.13))),
    'shrub_chaparral': dict(kind='shrubs', arche='shrub', variants=3, H=(1.8, 2.2, 2.6), D=(2.2, 2.6, 3.0),
                            veg=['chaparral', 'shrub_other'], ground=['scrub'], foliage='leaf_small',
                            bark=((0.08, 0.05, 0.04), (0.30, 0.17, 0.12)), tint=((0.40, 0.52, 0.26), (0.5, 0.6, 0.3)),
                            stems=10, upright=0.45, cards=4.0, card=0.6, low=((0.1, 0.17, 0.07), (0.25, 0.14, 0.1))),
    'shrub_willow':   dict(kind='shrubs', arche='shrub', variants=2, H=(2.8, 3.5), D=(2.5, 3.0),
                           veg=['riparian_shrub'], foliage='leaf_narrow', bark=((0.10, 0.07, 0.04), (0.35, 0.25, 0.15)),
                           tint=((0.8, 0.95, 0.7), (0.95, 1.05, 0.75)), stems=16, upright=0.9, cards=3.0, card=0.6,
                           low=((0.18, 0.3, 0.1), (0.3, 0.22, 0.14))),
    'shrub_alpine':   dict(kind='shrubs', arche='shrub', variants=2, H=(0.35, 0.5), D=(1.0, 1.4),
                           veg=['alpine_shrub'], foliage='leaf_small', bark=((0.07, 0.06, 0.05), (0.22, 0.2, 0.17)),
                           tint=((0.52, 0.64, 0.32), (0.62, 0.7, 0.36)), stems=12, upright=0.1, cards=4.0, card=0.35,
                           low=((0.17, 0.24, 0.1), (0.2, 0.18, 0.15))),
    'shrub_palmetto': dict(kind='shrubs', arche='palmetto', variants=2, H=(1.2, 1.5), D=(2.0, 2.4),
                           veg=['tropical_shrub', 'pine_east'], foliage='palm_fan', bark=((0.1, 0.09, 0.07), (0.3, 0.27, 0.22)),
                           tint=((0.7, 0.9, 0.75), (0.85, 1.0, 0.8)), fronds=12, low=((0.16, 0.3, 0.13), (0.3, 0.27, 0.22))),
    'shrub_joshua':   dict(kind='shrubs', arche='joshua', variants=2, H=(4.5, 6.5), D=(3, 4),
                           veg=['desert_scrub'], foliage='yucca', bark=((0.10, 0.08, 0.06), (0.32, 0.27, 0.2)),
                           tint=((0.75, 0.85, 0.6), (0.85, 0.95, 0.65)), share=0.06, low=((0.25, 0.32, 0.14), (0.3, 0.25, 0.18))),
    'shrub_cholla':   dict(kind='shrubs', arche='cactus', variants=2, H=(1.2, 1.6), D=(1.0, 1.3),
                           veg=['desert_scrub'], share=0.15, low=((0.42, 0.48, 0.3), (0.3, 0.3, 0.2))),
    # ---- 草本 ----
    'grass_reed':     dict(kind='grass', arche='grass', variants=2, H=(1.7, 2.1), D=(0.7, 0.8),
                           veg=['wetland_herb', 'swamp_forest'], ground=['wetland'], foliage='reed', cards=4,
                           low=((0.28, 0.33, 0.14), (0.4, 0.38, 0.2))),
    'grass_dune':     dict(kind='grass', arche='grass', variants=2, H=(0.7, 0.9), D=(0.8, 0.9),
                           veg=['dune_grass'], ground=['sand'], foliage='dune', cards=4,
                           low=((0.45, 0.45, 0.28), (0.6, 0.58, 0.4))),
    'grass_flower':   dict(kind='grass', arche='grass', variants=2, H=(0.5, 0.6), D=(0.8, 0.9),
                           veg=['meadow', 'prairie_tall'], ground=['meadow'], foliage='flower', cards=4,
                           low=((0.2, 0.32, 0.1), (0.35, 0.42, 0.18))),
}


# ---------------- 贴图（numpy；sRGB 值，带透明度） ----------------
def _stroke(img, px, py, ax, ay, bx, by, w, col, S):
    """画一笔线段（只算包围盒里的像素），颜色可以是 (r, g, b) 或按沿线位置 t 返回颜色的函数"""
    pad = w + 2 / S
    c0, c1 = int(max(0, min(ax, bx) - pad) * S), int(min(1, max(ax, bx) + pad) * S) + 1
    r0, r1 = int(max(0, min(ay, by) - pad) * S), int(min(1, max(ay, by) + pad) * S) + 1
    if c1 <= c0 or r1 <= r0:
        return
    sl = (slice(S - r1, S - r0), slice(c0, c1))
    d, t = hv.seg_dist(px[sl], py[sl], ax, ay, bx, by)
    m = d < w
    blk = img[sl]
    c = col(t) if callable(col) else np.broadcast_to(np.array(col, np.float32), t.shape + (3,))
    blk[m, :3] = c[m]; blk[m, 3] = 1


def _canvas(S):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / S
    return np.zeros((S, S, 4), np.float32), xx, 1 - yy


def tex_needles_long(S, seed):
    """长针松：一簇簇从枝端散开的长松针（西黄松、长叶松）"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    wid = 0.0022 * max(1.0, 1024 / S) ** 0.6
    _stroke(img, px, py, 0.5, 0.0, 0.5, 0.55, 0.012, (0.25, 0.17, 0.1), S)
    for cx, cy, sc in ((0.5, 0.55, 1.0), (0.38, 0.32, 0.65), (0.63, 0.38, 0.7)):
        _stroke(img, px, py, 0.5, cy * 0.5, cx, cy, 0.008, (0.25, 0.17, 0.1), S)
        for _ in range(int(130 * sc)):
            a = rng.normal(0, 1.1); L = rng.uniform(0.28, 0.45) * sc
            ex, ey = cx + math.sin(a) * L, cy + math.cos(a) * L * 0.9
            g = rng.uniform(-1, 1)
            _stroke(img, px, py, cx, cy, ex, ey, wid,
                    lambda t, g=g: np.stack([0.11 + 0.03 * g + 0.08 * t, 0.19 + 0.04 * g + 0.08 * t, 0.07 + 0.02 * t], -1), S)
    return img


def tex_feather(S, seed):
    """落羽杉：羽毛状的扁平小枝，两侧排满短而粗的叶（比冷杉的针叶粗，远处缩小后也不会看不见）"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    wid = 0.006 * max(1.0, 1024 / S) ** 0.5
    for b in range(5):
        a = rng.uniform(-0.5, 0.5); x0 = 0.5 + rng.uniform(-0.08, 0.08)
        L = rng.uniform(0.6, 0.9); x1, y1 = x0 + math.sin(a) * L, math.cos(a) * L
        _stroke(img, px, py, x0, 0.0, x1, y1, 0.005, (0.3, 0.22, 0.12), S)
        v = 0.08
        while v < 0.97:
            cx, cy = x0 + (x1 - x0) * v, y1 * v
            for side in (-1, 1):
                aa = a + side * math.radians(rng.uniform(55, 75)); l = 0.075 * (1.1 - 0.5 * v)
                g = rng.uniform(0.85, 1.15)
                _stroke(img, px, py, cx, cy, cx + math.sin(aa) * l, cy + math.cos(aa) * l, wid,
                        (0.16 * g, 0.30 * g, 0.10 * g), S)
            v += 0.03
    return img


def tex_scale(S, seed):
    """刺柏：鳞片状小枝，一团团圆鼓鼓的灰绿色"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    for _ in range(26):
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.15, 0.85); r = rng.uniform(0.08, 0.16)
        d = np.hypot((px - cx) / r, (py - cy) / r)
        n = 0.15 * np.sin(px * 90 + rng.uniform(0, 6)) * np.sin(py * 85 + rng.uniform(0, 6))
        m = d + n < 1
        g = rng.uniform(-1, 1)
        shade = np.clip(1.1 - 0.4 * d, 0.6, 1.1)
        col = np.stack([(0.16 + 0.03 * g) * shade, (0.24 + 0.04 * g) * shade, (0.17 + 0.03 * g) * shade], -1)
        img[m, :3] = col[m]; img[m, 3] = 1
    return img


def _leaf_twig(S, seed, shape, n, size, base_col):
    """一小枝阔叶：中间细枝，两边交错长叶子。shape：oval（椭圆）、round（三角圆形，杨树）、lobed（掌状浅裂，枫树）、narrow（柳叶）、small（细碎小叶）"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    _stroke(img, px, py, 0.5, 0.0, 0.5, 0.93, 0.006, (0.22, 0.16, 0.09), S)
    for i in range(n):
        v = 0.08 + 0.85 * i / n + rng.uniform(-0.02, 0.02)
        side = 1 if i % 2 else -1
        ang = math.radians(rng.uniform(30, 70)) * side
        L = size * rng.uniform(0.8, 1.15) * (1.0 - 0.3 * v); ux, uy = math.sin(ang), math.cos(ang)
        ox, oy = 0.5 + ux * 0.02, v + uy * 0.02
        lx = (px - ox) * ux + (py - oy) * uy; ly = -(px - ox) * uy + (py - oy) * ux
        t = lx / L; inside = (t > 0) & (t < 1)
        if shape == 'round':
            prof = np.sin(np.pi * np.clip(t, 0, 1) ** 0.6) * 0.95
        elif shape == 'lobed':
            ang_ = np.arctan2(ly, lx - L * 0.45); rad = np.hypot(lx - L * 0.45, ly) / (L * 0.55)
            m = (rad < 0.75 + 0.3 * np.cos(5 * ang_)) & (lx > 0)
            prof = None
        elif shape == 'narrow':
            prof = np.sin(np.pi * np.clip(t, 0, 1)) * 0.22
        elif shape == 'small':
            prof = np.sin(np.pi * np.clip(t, 0, 1)) * 0.55
        else:
            prof = np.sin(np.pi * np.clip(t, 0, 1) ** 0.85) * 0.42
        if prof is not None:
            m = inside & (np.abs(ly) < L * 0.5 * prof)
        g = rng.uniform(-1, 1)
        col = np.array(base_col) * (1 + 0.15 * g)
        shade = (0.75 + 0.35 * np.clip(t, 0, 1) + 0.2 * np.sign(ly) * side)[..., None]
        img[m, :3] = (col[None, None] * shade)[m]; img[m, 3] = 1
        _stroke(img, px, py, 0.5, v, ox, oy, 0.003, (0.2, 0.2, 0.08), S)
    return img


def tex_small_leaves(S, seed):
    """细碎小叶：很多小枝上的小叶（山艾、石炭酸灌木、查帕拉尔；颜色由顶点颜色区分）"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    for b in range(7):
        a = rng.uniform(-0.6, 0.6); x0 = 0.5 + rng.uniform(-0.1, 0.1)
        x1, y1 = x0 + math.sin(a) * 0.8, 0.05 + math.cos(a) * 0.85
        _stroke(img, px, py, x0, 0.0, x1, y1, 0.004, (0.3, 0.27, 0.22), S)
        for _ in range(45):
            t = rng.uniform(0.1, 1.0); cx, cy = x0 + (x1 - x0) * t, (y1) * t
            r = rng.uniform(0.012, 0.026)
            q = rng.uniform(0, 6.3)
            ex, ey = cx + math.cos(q) * r * 2, cy + math.sin(q) * r * 2
            g = rng.uniform(0.8, 1.15)
            _stroke(img, px, py, cx, cy, ex, ey, r * 0.7, (0.60 * g, 0.62 * g, 0.56 * g), S)
    return img


def tex_palm_fan(S, seed):
    """扇形棕榈叶（菜棕、锯齿棕）：从叶柄一点放射出 30 来条带状裂片，末端分叉下垂"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    cx, cy = 0.5, 0.06
    n = 32
    for i in range(n):
        a = (i / (n - 1) - 0.5) * math.radians(200)
        L = 0.78 * (1 - 0.25 * abs(i / (n - 1) - 0.5) * 2) * rng.uniform(0.92, 1.05)
        pts = [(cx, cy)]
        for k in range(1, 9):
            t = k / 8
            aa = a * (1 + 0.15 * t)
            pts.append((cx + math.sin(aa) * L * t, cy + math.cos(aa) * L * t - 0.06 * t * t * abs(math.sin(a))))
        g = rng.uniform(0.85, 1.1)
        for k in range(8):
            w = 0.014 * (1 - 0.6 * k / 8) + 0.002
            col = (0.20 * g, 0.36 * g, 0.18 * g) if k < 6 else (0.24 * g, 0.34 * g, 0.18 * g)
            _stroke(img, px, py, *pts[k], *pts[k + 1], w, col, S)
    _stroke(img, px, py, 0.5, 0.0, cx, cy, 0.01, (0.3, 0.3, 0.15), S)
    return img


def tex_yucca(S, seed):
    """丝兰 / 约书亚树的叶丛：从中心放射的细长尖叶"""
    rng = np.random.default_rng(seed); img, px, py = _canvas(S)
    cx, cy = 0.5, 0.08
    for _ in range(55):
        a = rng.normal(0, 0.75); L = rng.uniform(0.45, 0.8)
        g = rng.uniform(0.85, 1.1)
        _stroke(img, px, py, cx, cy, cx + math.sin(a) * L, cy + math.cos(a) * L, 0.012,
                lambda t, g=g: np.stack([(0.25 + 0.12 * t) * g, (0.34 + 0.1 * t) * g, (0.17 + 0.05 * t) * g], -1), S)
    return img


def tex_reed(S, seed):
    """芦苇 / 香蒲：细长直立的叶片，几根带褐色穗"""
    rng = np.random.default_rng(seed); img = hv.grass_texture(S, ((0.12, 0.17, 0.07), (0.45, 0.50, 0.25)), seed, blades=45)
    _, px, py = _canvas(S)
    for _ in range(4):
        x0 = 0.5 + rng.normal(0, 0.12); top = rng.uniform(0.75, 0.95)
        _stroke(img, px, py, x0, 0.0, x0 + rng.normal(0, 0.03), top, 0.005, (0.35, 0.36, 0.2), S)
        _stroke(img, px, py, x0, top - 0.16, x0, top - 0.04, 0.018, (0.28, 0.17, 0.08), S)   # 香蒲穗
    return img


def tex_flower(S, seed):
    """草甸：草叶里夹着几种颜色的小野花"""
    rng = np.random.default_rng(seed); img = hv.grass_texture(S, ((0.08, 0.15, 0.05), (0.32, 0.46, 0.16)), seed, blades=55)
    _, px, py = _canvas(S)
    for col in ((0.95, 0.85, 0.2), (0.85, 0.35, 0.6), (0.95, 0.95, 0.9), (0.45, 0.45, 0.9)):
        for _ in range(4):
            x0 = 0.5 + rng.normal(0, 0.15); top = rng.uniform(0.45, 0.85)
            _stroke(img, px, py, x0, 0.0, x0 + rng.normal(0, 0.04), top, 0.003, (0.2, 0.3, 0.1), S)
            d = np.hypot(px - x0, py - top); m = d < 0.025
            img[m, :3] = col; img[m, 3] = 1
    return img


# 叶片贴图提亮倍数（线性）：画的是叶片本身的颜色，阳光下看起来要亮得多（hero 的树叶色乘数是 1.5 ~ 2.8）
GAIN = {'needles_long': 2.0, 'needles_short': 1.9, 'scale': 1.6, 'feather': 1.7, 'leaf_round': 1.7, 'leaf_oval': 1.8, 'leaf_lobed': 1.7,
        'leaf_narrow': 1.6, 'leaf_small': 1.0, 'palm_fan': 1.6, 'yucca': 1.4, 'reed': 1.3, 'dune': 1.0, 'flower': 1.3}
FOLIAGE = {   # 贴图名 → 生成函数
    'needles_long': lambda S: tex_needles_long(S, 11),
    'needles_short': lambda S: hv.needle_texture(S, 12),
    'scale': lambda S: tex_scale(S, 13),
    'feather': lambda S: tex_feather(S, 24),
    'leaf_round': lambda S: _leaf_twig(S, 14, 'round', 14, 0.22, (0.24, 0.36, 0.10)),
    'leaf_oval': lambda S: _leaf_twig(S, 15, 'oval', 20, 0.17, (0.16, 0.26, 0.08)),
    'leaf_lobed': lambda S: _leaf_twig(S, 16, 'lobed', 10, 0.28, (0.19, 0.33, 0.09)),
    'leaf_narrow': lambda S: _leaf_twig(S, 17, 'narrow', 22, 0.24, (0.25, 0.36, 0.14)),
    'leaf_small': lambda S: tex_small_leaves(S, 18),
    'palm_fan': lambda S: tex_palm_fan(S, 19),
    'yucca': lambda S: tex_yucca(S, 20),
    'reed': lambda S: tex_reed(S, 21),
    'dune': lambda S: hv.grass_texture(S, ((0.30, 0.28, 0.16), (0.78, 0.74, 0.52)), 22, blades=40),
    'flower': lambda S: tex_flower(S, 23),
}


# ---------------- 几何：中高档（枝干管子 + 叶片卡片） ----------------
class Builder(hv.TreeBuilder):
    """和 build_hq_vegetation 的 TreeBuilder 一样；顶点属性 (是否叶片, 叶色随机数, 明暗)"""
    def blob_cards(self, center, radius, n, size, shade_fn, outward=0.6):
        """在一个球形范围里撒 n 张卡片（刺柏的一团团、灌木）"""
        for _ in range(n):
            p = center + self.rvec() * radius * self.rng.random() ** 0.5
            up = ((p - center).normalized() * outward + self.rvec() * 0.6 + Vector((0, 0, 0.3))).normalized()
            s = size * self.rng.uniform(0.8, 1.2)
            self.card(p, up, s * 0.9, s, self.rng.random(), shade_fn(p), bend=0.1)


def make_conifer(sp, H, D, seed, q):
    """针叶树：主干 + 一层层轮生枝 + 枝上的针叶卡片。
    shape：cone（尖塔，冷杉）、round_cone（上部圆锥，西黄松）、clumps（只在顶上几团，长叶松）、flat_top（平顶，落羽杉）、bush（刺柏）"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    R = D / 2; z0 = H * sp['clear']
    n = 14 if q['sides'][0] >= 6 else 8
    pts, radii = [], []
    for k in range(n + 1):
        t = k / n
        pts.append(Vector((math.sin(t * 7 + seed) * 0.05 * H / 10, 0, -0.4 + (H + 0.4) * t)))
        r = (0.018 * H + 0.05) * (1 - t) ** 0.8 + 0.02
        if sp.get('buttress') and t < 0.12:
            r *= 1 + 2.2 * (1 - t / 0.12) ** 2            # 落羽杉树干底部膨大
        radii.append(r)
    tb.tube(pts, radii, q['sides'][0] + 2, tile=1.2)

    def trunk_at(z):
        t = hv.clampf((z + 0.4) / (H + 0.4)); i = min(n - 1, int(t * n))
        return pts[i].lerp(pts[i + 1], t * n - i), radii[i]

    def crown_r(u):          # u：在树冠里的相对高度 0（底）~ 1（顶）
        s = sp['shape']
        if s == 'cone': return R * (1 - u) ** 0.95
        if s == 'round_cone': return R * (1 - u) ** 0.7 * (0.6 + 0.4 * math.sin(math.pi * min(1, u * 2)))
        if s == 'clumps': return R * math.sin(math.pi * u) ** 0.6
        if s == 'flat_top': return R * (0.55 + 0.45 * math.sin(math.pi * u ** 0.6)) * (1 - u ** 6)
        return R * math.sin(math.pi * (0.15 + 0.85 * u)) ** 0.5           # bush
    cards = []
    z = z0
    while z < H * 0.97:
        u = (z - z0) / max(H - z0, 1e-3)
        if rng.random() > sp['skip']:
            k = rng.randint(4, 6); az0 = rng.uniform(0, 6.28)
            for i in range(k):
                az = az0 + 2 * math.pi * i / k + rng.uniform(-0.4, 0.4)
                Lb = max(0.3, crown_r(u) * rng.uniform(0.75, 1.1))
                p, r = trunk_at(z)
                d = Vector((math.cos(az), math.sin(az), -sp['droop'] * (1 - u) + rng.uniform(-0.1, 0.15))).normalized()
                segs = max(2, int(Lb / (0.5 if q['sides'][1] > 3 else 0.9)))
                bp, br = [p], [max(0.015, r * 0.4)]
                cur = p.copy()
                for s in range(segs):
                    uu = (s + 1) / segs
                    d = (d + Vector((0, 0, 0.12 * uu)) + tb.rvec() * 0.07).normalized()
                    cur = cur + d * (Lb / segs); bp.append(cur.copy()); br.append(br[0] * (1 - 0.8 * uu) + 0.004)
                tb.tube(bp, br, q['sides'][1], tile=0.6)
                tip0 = 1 - sp['tips']
                for s in range(1, len(bp)):
                    if (s / len(bp)) >= tip0:
                        cards.append((bp[s - 1], (bp[s] - bp[s - 1]).normalized(), (bp[s] - bp[s - 1]).length))
                cards.append((bp[-1], d, Lb * 0.25))
        z += sp['whorl'] * rng.uniform(0.8, 1.2) * (H / 20) ** 0.5
    tp, _ = trunk_at(H * 0.93); cards.append((tp, Vector((0, 0, 1)), H * 0.07))
    step = 0.16 / (q['density'] * sp.get('dens', 1.0))
    for p, d, L in cards:
        m = max(1, int(L / step + rng.random()))
        for j in range(m):
            r = p + d * (L * (j + rng.random()) / m)
            rot = Matrix.Rotation(rng.uniform(-math.pi, math.pi), 3, d)
            up = (rot @ (d.orthogonal().normalized() * 0.6) + d).normalized()
            s = rng.uniform(0.55, 0.8) * sp['card'] * q['scale'] * (H / 20) ** 0.25
            shade = min(1.0, 0.3 + r.to_2d().length / (R + 1e-3))
            tb.card(r, up, s * 0.95, s, rng.random(), shade, bend=0.15)
    return tb.finish(sp['name'])


def make_broadleaf(sp, H, D, seed, q):
    """阔叶树：主干分出几根大枝，再分细枝，细枝上挂叶片簇卡片（和 hero 的橡树一样，树冠形状按树种）"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    cen = Vector((0, 0, H * sp['crown_z'])); rad = Vector((D / 2, D / 2, H * sp['crown_h']))
    def env(p):
        r = p - cen; return math.sqrt((r.x / rad.x) ** 2 + (r.y / rad.y) ** 2 + (r.z / rad.z) ** 2)
    leaves = []
    sides = (q['sides'][0] + 2, q['sides'][1] + 1, q['sides'][2], q['sides'][2])

    def grow(p, d, L, r0, lvl):
        step = 0.35 if lvl < 2 else 0.5
        segs = max(2, int(L / step)); st = L / segs
        pts, radii = [p.copy()], [r0]
        cur = p.copy()
        for k in range(segs):
            r = cur - cen
            d = (d + tb.rvec() * (0.12 + 0.08 * lvl) + Vector((0, 0, 0.05 if lvl < 2 else -0.04))).normalized()
            if env(cur) > 0.9 and lvl > 0:
                d = (d - Vector((r.x / rad.x ** 2, r.y / rad.y ** 2, r.z / rad.z ** 2)).normalized() * 0.5).normalized()
            cur = cur + d * st
            pts.append(cur.copy()); radii.append(r0 * (1 - 0.7 * (k + 1) / segs) + 0.005)
            if env(cur) > 1.08 and lvl > 0: break
        tb.tube(pts, radii, sides[lvl], tile=0.9, stride=1 if lvl < 2 else q['stride'])
        if lvl == 0:
            k = rng.randint(4, 6); az0 = rng.uniform(0, 6.28)
            for i in range(k):
                az = az0 + 2 * math.pi * i / k + rng.uniform(-0.3, 0.3)
                a = math.radians(rng.uniform(30, 50) + 22 * sp['spread'])
                cd = Vector((math.cos(az) * math.sin(a), math.sin(az) * math.sin(a), math.cos(a)))
                grow(pts[-1], cd, max(D / 2, H * 0.4) * rng.uniform(0.75, 0.95), radii[-1] * 0.7, 1)
            return
        for i in range([0, 7, 5, 0][lvl]):
            t = rng.uniform(0.3, 0.97)
            idx = min(len(pts) - 2, int(t * (len(pts) - 1)))
            bd = (pts[idx + 1] - pts[idx]).normalized()
            cd = (bd + bd.cross(tb.rvec()).normalized() * 0.9).normalized()
            grow(pts[idx], cd, L * rng.uniform(0.45, 0.65), radii[idx] * 0.6, lvl + 1)
        if lvl >= 2:
            for i in range(1, len(pts)):
                leaves.append((pts[i], (pts[i] - pts[i - 1]).normalized()))
    grow(Vector((0, 0, -0.4)), Vector((rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06), 1)).normalized(),
         H * sp['trunk'], 0.022 * H + 0.06, 0)
    per = 2.2 * q['density'] * sp.get('dens', 1.0)
    for p, d in leaves:
        for _ in range(tb.cards(per)):
            out = (p - cen); out.z *= 0.6
            up = (d + out.normalized() * 0.6 + Vector((0, 0, 0.3)) + tb.rvec() * 0.5).normalized()
            s = rng.uniform(0.9, 1.3) * q['scale'] * (H / 15) ** 0.3
            tb.card(p + tb.rvec() * 0.3, up, s * 0.9, s, rng.random(), min(1.0, env(p) ** 2), bend=0.12)
    return tb.finish(sp['name'])


def _fronds(tb, top, n, L, droop, q, rng, dead=0):
    """棕榈叶：叶柄（细管）+ 扇形叶片卡片；dead > 0 时下面再挂几片下垂的枯叶"""
    for i in range(n + dead):
        is_dead = i >= n
        az = rng.uniform(0, 6.28) if is_dead else 2 * math.pi * i / n + rng.uniform(-0.2, 0.2)
        el = rng.uniform(-1.2, -0.9) if is_dead else rng.uniform(-0.3, 0.9) - droop * rng.random()
        d = Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el))).normalized()
        pl = L * 0.45
        tip = top + d * pl
        tb.tube([top, top + d * pl * 0.5, tip], [0.035, 0.03, 0.02], max(3, q['sides'][2]), tile=0.5)
        side = d.cross(Vector((0, 0, 1)))
        if side.length < 1e-3: side = Vector((1, 0, 0))
        up = (d + Vector((0, 0, 0.2))).normalized()
        s = L * rng.uniform(0.85, 1.05)
        tb.card(tip - up * s * 0.05, up, s, s, rng.random(), 0.25 if is_dead else 0.8, bend=0.25)


def make_palm(sp, H, D, seed, q):
    """菜棕：细长、略弯的树干，顶上一圈扇形叶，下面几片枯叶"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    n = 12 if q['sides'][0] >= 6 else 7
    lean = Vector((rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0))
    pts = [Vector((0, 0, -0.3)) + lean * (k / n) ** 2 + Vector((0, 0, (H * 0.9 + 0.3) * k / n)) for k in range(n + 1)]
    tb.tube(pts, [0.24 - 0.06 * k / n for k in range(n + 1)], q['sides'][0] + 2, tile=0.6)
    _fronds(tb, pts[-1], sp['fronds'], D * 0.72, 0.6, q, rng, dead=int(6 * q['density']))
    return tb.finish(sp['name'])


def make_palmetto(sp, H, D, seed, q):
    """锯齿棕：贴地的短茎上长出一丛扇形叶"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    for c in range(3):
        base = Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.1))
        _fronds(tb, base, int(sp['fronds'] / 3 + 1), H * 0.85, -0.6, q, rng)
    return tb.finish(sp['name'])


def make_shrub(sp, H, D, seed, q):
    """灌木：根部散开 stems 根枝（upright 越大越直立），外圈挂叶片卡片"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    cen = Vector((0, 0, H * 0.5)); rad = Vector((D / 2, D / 2, H * 0.55))
    def env(p):
        r = p - cen; return math.sqrt((r.x / rad.x) ** 2 + (r.y / rad.y) ** 2 + (r.z / rad.z) ** 2)
    spots = []
    for i in range(sp['stems']):
        az = rng.uniform(0, 6.28); a = math.radians(rng.uniform(15, 75) * (1.2 - sp['upright']))
        d = Vector((math.cos(az) * math.sin(a), math.sin(az) * math.sin(a), math.cos(a)))
        p = Vector((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), -0.05)); pts = [p]; cur = p.copy()
        L = max(H, D / 2) * rng.uniform(0.8, 1.1)
        for k in range(4):
            d = (d + tb.rvec() * 0.25 + Vector((0, 0, 0.08 * sp['upright']))).normalized()
            cur = cur + d * (L / 4); pts.append(cur.copy())
            if env(cur) > 1.0: break
        tb.tube(pts, [0.025 * (1 - k / len(pts)) + 0.004 for k in range(len(pts))], 3, tile=0.4)
        spots += [(q_, (q_ - p).normalized()) for q_ in pts[1:] if env(q_) > 0.3]
    for p, d in spots:
        for _ in range(tb.cards(sp['cards'] * q['density'])):
            up = (d + (p - cen).normalized() * 0.5 + tb.rvec() * 0.6).normalized()
            s = sp['card'] * rng.uniform(0.8, 1.2) * q['scale']
            tb.card(p + tb.rvec() * 0.15 * D, up, s * 0.9, s, rng.random(), min(1.0, env(p) ** 2 + 0.25), bend=0.1)
    return tb.finish(sp['name'])


def make_joshua(sp, H, D, seed, q):
    """约书亚树：粗糙的主干分叉成几根粗枝，每个枝端一团尖叶"""
    tb = Builder(seed, {'card_rows': q['card_rows']}); rng = tb.rng
    tips = []
    def arm(p, d, L, r, lvl):
        pts = [p]; cur = p.copy()
        for k in range(3):
            d = (d + tb.rvec() * 0.2 + Vector((0, 0, 0.15))).normalized(); cur = cur + d * (L / 3); pts.append(cur.copy())
        tb.tube(pts, [r, r * 0.9, r * 0.8, r * 0.75], q['sides'][1] + 2, tile=0.5)
        if lvl < 2:
            for _ in range(rng.randint(2, 3)):
                az = rng.uniform(0, 6.28); cd = Vector((math.cos(az), math.sin(az), rng.uniform(0.5, 1.2))).normalized()
                arm(pts[-1], cd, L * rng.uniform(0.6, 0.8), r * 0.75, lvl + 1)
        else:
            tips.append(pts[-1])
    arm(Vector((0, 0, -0.2)), Vector((0, 0, 1)), H * 0.42, 0.26, 0)
    for p in tips:
        for _ in range(max(3, int(9 * q['density']))):
            up = (Vector((0, 0, 0.6)) + tb.rvec()).normalized()
            s = 0.7 * rng.uniform(0.85, 1.15)
            tb.card(p - up * 0.15, up, s, s, rng.random(), 0.8, bend=0.05)
        for _ in range(max(2, int(5 * q['density']))):          # 枝端下面挂着的枯叶
            up = (Vector((0, 0, -0.8)) + tb.rvec() * 0.5).normalized()
            tb.card(p, up, 0.6, 0.6, rng.random(), 0.2, bend=0.02)
    return tb.finish(sp['name'])


def make_cactus(sp, H, D, seed, q):
    """圆柱仙人掌（cholla）：短主干上一节节分叉的短圆柱，大多朝上，像一棵小树；没有叶片卡片"""
    tb = Builder(seed, {'card_rows': 1}); rng = tb.rng
    sides = 7 if q.get('sides', (8,))[0] >= 6 else 5
    def seg(p, d, L, r, lvl):
        end = p + d * L
        tb.tube([p, p.lerp(end, 0.5) + tb.rvec() * 0.01, end], [r, r * 1.08, r * 0.9], sides, tile=0.3)
        if lvl < 4:
            for _ in range(rng.randint(1, 2) if lvl else 3):
                az = rng.uniform(0, 6.28)
                cd = (Vector((math.cos(az), math.sin(az), 0)) * rng.uniform(0.4, 0.9) + Vector((0, 0, 1))).normalized()
                seg(end, cd, L * rng.uniform(0.75, 0.95), r * 0.9, lvl + 1)
    seg(Vector((0, 0, -0.05)), Vector((0, 0, 1)), H * 0.3, 0.06, 0)
    return tb.finish(sp['name'])


def make_grass(sp, H, D, seed, q):
    return hv.make_grass(sp['name'], H, D, seed, {'grass_cards': sp['cards'] if q['card_rows'] == 2 else sp['cards'] - 1,
                                                    'grass_rows': q['card_rows']})


ARCHE = {'conifer': make_conifer, 'broadleaf': make_broadleaf, 'palm': make_palm, 'palmetto': make_palmetto,
         'shrub': make_shrub, 'joshua': make_joshua, 'cactus': make_cactus, 'grass': make_grass}


# ---------------- 几何：低档（低多边形 + 顶点颜色） ----------------
def lowpoly(sp, H, D, seed):
    """低档：树干一根六棱柱，树冠用几个变形的多面体（针叶树叠几层圆锥），颜色写在顶点上"""
    rng = random.Random(seed)
    bm = bmesh.new(); col = []
    leaf_c, bark_c = (np.array(c) for c in sp['low'])
    def add(geom_verts, c):
        for v in geom_verts:
            col.append(c * rng.uniform(0.85, 1.1))
    def cyl(r0, r1, z0, z1, sides=6, c=bark_c, ox=0.0, oy=0.0):
        ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=sides, radius1=r0, radius2=r1, depth=z1 - z0)
        bmesh.ops.translate(bm, verts=ret['verts'], vec=(ox, oy, (z0 + z1) / 2))
        add(ret['verts'], c)
    def blob(cx, cy, cz, rx, rz, c, sub=1):
        ret = bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        for v in ret['verts']:
            v.co = Vector((cx + v.co.x * rx * rng.uniform(0.85, 1.1), cy + v.co.y * rx * rng.uniform(0.85, 1.1), cz + v.co.z * rz))
        add(ret['verts'], c)
    a = sp['arche']
    if a == 'conifer':
        z0 = H * sp['clear']
        cyl(0.012 * H + 0.08, 0.04, -0.3, H * 0.95)
        layers = 4 if sp['shape'] in ('cone',) else 3
        for k in range(layers):
            u0 = k / layers
            zb = z0 + (H - z0) * u0 * 0.9
            r = D / 2 * ((1 - u0) if sp['shape'] == 'cone' else (0.9 - 0.3 * u0))
            if sp['shape'] in ('bush', 'flat_top', 'round_cone', 'clumps'):
                blob(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), zb + (H - z0) / layers * 0.5, r, (H - z0) / layers * 0.7, leaf_c)
            else:
                cyl(r, r * 0.25, zb, zb + (H - z0) / layers * 1.3, sides=7, c=leaf_c)
    elif a == 'broadleaf':
        cyl(0.02 * H + 0.08, 0.06, -0.3, H * sp['crown_z'])
        for k in range(5):
            az = rng.uniform(0, 6.28); rr = D / 2 * rng.uniform(0.25, 0.5)
            blob(math.cos(az) * rr, math.sin(az) * rr, H * sp['crown_z'] + rng.uniform(-0.15, 0.25) * H * sp['crown_h'] * 2,
                 D * rng.uniform(0.25, 0.33), H * sp['crown_h'] * rng.uniform(0.55, 0.75), leaf_c)
    elif a in ('palm', 'palmetto'):
        top = H * 0.9 if a == 'palm' else 0.2
        if a == 'palm':
            cyl(0.22, 0.18, -0.3, top)
        for k in range(sp['fronds'] // 2):
            az = 2 * math.pi * k / (sp['fronds'] // 2)
            ret = bmesh.ops.create_cone(bm, cap_ends=False, segments=3, radius1=0.45, radius2=0.02, depth=D * 0.5)
            m = Matrix.Translation((math.cos(az) * D * 0.25, math.sin(az) * D * 0.25, top)) @ Matrix.Rotation(az, 4, 'Z') \
                @ Matrix.Rotation(math.radians(70 if a == 'palm' else 50), 4, 'Y')
            bmesh.ops.transform(bm, matrix=m, verts=ret['verts'])
            add(ret['verts'], leaf_c)
    elif a == 'shrub':
        for k in range(3):
            az = rng.uniform(0, 6.28); rr = D * 0.15
            blob(math.cos(az) * rr, math.sin(az) * rr, H * 0.45, D * 0.35, H * 0.5, leaf_c)
    elif a in ('joshua', 'cactus'):
        # 主干 + 几根斜着伸出去的枝（从主干顶上长出来）；约书亚树枝端一团叶，仙人掌全是绿色的
        trunk_c = bark_c if a == 'joshua' else leaf_c
        r0 = 0.2 if a == 'joshua' else 0.08
        top = H * (0.5 if a == 'joshua' else 0.4)
        cyl(r0, r0 * 0.85, -0.2, top, c=trunk_c)
        for k in range(3):
            az = 2 * math.pi * k / 3 + rng.uniform(-0.4, 0.4)
            tip = Vector((math.cos(az) * D * 0.35, math.sin(az) * D * 0.35, H * 0.85))
            v = tip - Vector((0, 0, top)); L = v.length
            ret = bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=r0 * 0.7, radius2=r0 * 0.55, depth=L)
            m = Matrix.Translation(Vector((0, 0, top)) + v / 2) @ v.to_track_quat('Z', 'Y').to_matrix().to_4x4()
            bmesh.ops.transform(bm, matrix=m, verts=ret['verts']); add(ret['verts'], trunk_c)
            if a == 'joshua':
                blob(tip.x, tip.y, tip.z + 0.2, 0.55, 0.45, leaf_c, sub=1)
    else:                                                 # grass：几片竖直的细三角
        for k in range(5):
            az = rng.uniform(0, 3.14); w = D * 0.2
            v0 = bm.verts.new((math.cos(az) * w, math.sin(az) * w, 0)); v1 = bm.verts.new((-math.cos(az) * w, -math.sin(az) * w, 0))
            v2 = bm.verts.new((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), H))
            bm.faces.new((v0, v1, v2)); col += [leaf_c * 0.6, leaf_c * 0.6, leaf_c * 1.1]
    me = bpy.data.meshes.new(sp['name']); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = False
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    rgb = np.clip(np.array(col, np.float32), 0, 1)
    attr.data.foreach_set('color', np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32).ravel())
    me.color_attributes.active_color = attr
    return me


def vc_material(name):
    """低档材质：只读顶点颜色"""
    m = bpy.data.materials.new(name)
    bs = m.node_tree.nodes['Principled BSDF']; bs.inputs['Roughness'].default_value = 0.85
    vc = m.node_tree.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'
    m.node_tree.links.new(vc.outputs['Color'], bs.inputs['Base Color'])
    m.use_backface_culling = False
    return m


def card_colors(me, va, tint, rand):
    """中高档顶点颜色：叶片 = 两种叶色之间取 × 树冠内暗外亮；枝干 = 明暗"""
    is_leaf, var, shade = va[:, 0], va[:, 1], va[:, 2]
    a, b = np.array(tint[0]), np.array(tint[1])
    f = np.clip(var * 0.7 + rand, 0, 1)[:, None]
    leaf = (a * (1 - f) + b * f) * (0.6 + 0.45 * shade)[:, None]
    bark = np.repeat((0.55 + 0.45 * shade)[:, None], 3, axis=1)
    rgb = np.clip(np.where(is_leaf[:, None] > 0.5, leaf, bark), 0, 1)
    attr = me.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
    attr.data.foreach_set('color', np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1).astype(np.float32).ravel())
    me.color_attributes.active_color = attr


# ---------------- 主程序 ----------------
def names():
    """每种 variants 款：tree_fir_01、tree_fir_02……"""
    out = []
    for key, sp in SPECIES.items():
        for v in range(sp['variants']):
            out.append((f'{key}_{v + 1:02d}', key, v))
    return out


def build_tier(tier, manifest):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    base = os.path.join(OUT, tier, 'species'); tex_dir = os.path.join(base, 'textures')
    print(f'== {tier}')
    mats = {}
    if tier == 'low':
        lowmat = vc_material('species_vertex_color')
    else:
        q = Q[tier]
        for k, fn in FOLIAGE.items():
            rgba = fn(q['tex'])
            rgba[..., :3] = np.clip(rgba[..., :3] * GAIN.get(k, 1.0) ** (1 / 2.2), 0, 1)
            img = hv.make_image(f'{k}', rgba, tex_dir)
            mats[k] = hv.export_material(f'sp_{k}', img, True, single_sided=k in ('reed', 'dune', 'flower'))
    colls = {'trees': [], 'shrubs': [], 'grass': []}
    for name, key, v in names():
        sp = {**SPECIES[key], 'name': name}
        H, D = sp['H'][v], sp['D'][v] if 'D' in sp else 1.0
        seed = 1000 + sum(map(ord, name))
        if tier == 'low':
            me = lowpoly(sp, H, D, seed)
            me.materials.append(lowmat)
        else:
            q = Q[tier]
            res = ARCHE[sp['arche']](sp, H, D, seed, q)
            if sp['arche'] == 'grass':
                me = res
                me.materials.append(mats[sp['foliage']])
            else:
                me, va = res
                if sp['arche'] == 'cactus':
                    bark = hv.make_image(f'skin_cactus', hv.bark_texture(q['tex'] // 2, (0.13, 0.17, 0.08), (0.35, 0.4, 0.22), 9), tex_dir)
                    m_b = mats.setdefault('cactus', hv.export_material('sp_cactus', bark, False))
                    card_colors(me, va, ((1, 1, 1), (1, 1, 1)), 0)
                    me.materials.append(m_b)
                else:
                    bk = f'bark_{key}'
                    if bk not in mats:
                        img = hv.make_image(bk, hv.bark_texture(q['tex'] // 2, sp['bark'][0], sp['bark'][1], seed), tex_dir)
                        mats[bk] = hv.export_material(f'sp_{bk}', img, False)
                    card_colors(me, va, sp['tint'], random.Random(name).uniform(0, 0.3))
                    me.materials.append(mats[bk]); me.materials.append(mats[sp['foliage']])
        zs = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', zs); zs = zs.reshape(-1, 3)[:, 2]
        k = H / float(zs.max() - max(zs.min(), -0.5))
        me.transform(Matrix.Scale(k, 4))              # 三档都缩放到设定的真实高度
        c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c)
        ob = bpy.data.objects.new(name, me); c.objects.link(ob)
        colls[sp['kind']].append(c)
        dims = ob.dimensions
        e = manifest.setdefault(name, {'name': name, 'kind': sp['kind'][:-1] if sp['kind'] != 'grass' else 'grass',
                                       'species': key, 'veg_types': sp.get('veg', []), 'ground_types': sp.get('ground', []),
                                       'candidate_veg_types': sp.get('cand', []),
                                       'share': sp.get('share', 1.0), 'triangles': {}})
        if tier == 'high' or 'height_m' not in e:
            e['height_m'] = round(H, 2); e['crown_diameter_m'] = round(max(dims.x, dims.y), 2)
            e['crown_area_m2'] = round(math.pi * (max(dims.x, dims.y) / 2) ** 2, 1)
        e['triangles'][tier] = hv.tris(me)
        print(f'  {name:22s} {hv.tris(me):6d} 个三角形  {dims.z:5.1f} m')
    for kind, cs in colls.items():
        os.makedirs(os.path.join(base, kind), exist_ok=True)
        bpy.data.libraries.write(os.path.join(base, kind, f'{kind}_export.blend'), set(cs), fake_user=True)


def preview(tier):
    """一个档位的全部素材：每个单独取景（按包围盒）渲染一张小图，拼成网格，图下面没有字，顺序同 SPECIES"""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    base = os.path.join(OUT, tier, 'species')
    obs = []
    for kind in ('trees', 'shrubs', 'grass'):
        with bpy.data.libraries.load(os.path.join(base, kind, f'{kind}_export.blend'), link=False) as (src, dst):
            dst.collections = list(src.collections)
        order = {n: i for i, (n, _, _) in enumerate(names())}
        for c in sorted(dst.collections, key=lambda c: order.get(c.name, 999)):
            e = bpy.data.objects.new(c.name, None); e.instance_type = 'COLLECTION'; e.instance_collection = c
            sc.collection.objects.link(e); e.hide_render = True; obs.append((e, c.objects[0]))
    bpy.ops.mesh.primitive_plane_add(size=200)
    g = bpy.context.active_object; gm = bpy.data.materials.new('g'); g.data.materials.append(gm)
    gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.15, 0.17, 0.08, 1)
    w = bpy.data.worlds.new('w'); sc.world = w
    w.node_tree.nodes['Background'].inputs[0].default_value = (0.45, 0.6, 0.85, 1)
    sun = bpy.data.lights.new('s', 'SUN'); sun.energy = 4; so = bpy.data.objects.new('s', sun)
    so.rotation_euler = (math.radians(50), 0, math.radians(-40)); sc.collection.objects.link(so)
    cd = bpy.data.cameras.new('c'); cd.lens = 50; cam = bpy.data.objects.new('c', cd); sc.collection.objects.link(cam)
    sc.camera = cam; sc.render.engine = 'BLENDER_EEVEE'; sc.eevee.taa_render_samples = 16
    sc.render.resolution_x = sc.render.resolution_y = 360
    sc.view_settings.view_transform = 'AgX'
    tiles = []
    tmp = os.path.join(OUT, 'previews', '_tile.png')
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    for e, ob in obs:
        e.hide_render = False
        h = max(ob.dimensions.z, 0.3); wdt = max(ob.dimensions.x, ob.dimensions.y, 0.3)
        size = max(h, wdt) * 1.15
        dist = size / (2 * math.tan(cd.angle / 2)) * 1.1
        tgt = Vector((0, 0, h * 0.5))
        cam.location = tgt + Vector((0, -dist, h * 0.12))
        cam.rotation_euler = (tgt - cam.location).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = tmp
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(tmp); px = np.empty(len(im.pixels), np.float32); im.pixels.foreach_get(px)
        tiles.append(px.reshape(360, 360, 4)); bpy.data.images.remove(im)
        e.hide_render = True
    cols = 9
    rows_ = (len(tiles) + cols - 1) // cols
    grid = np.ones((rows_ * 360, cols * 360, 4), np.float32)
    for i, t_ in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[(rows_ - 1 - r) * 360:(rows_ - r) * 360, c * 360:(c + 1) * 360] = t_        # Blender 图片从下往上
    im = bpy.data.images.new('grid', grid.shape[1], grid.shape[0]); im.pixels.foreach_set(grid.ravel())
    im.filepath_raw = os.path.join(OUT, 'previews', f'species_{tier}.png'); im.file_format = 'PNG'; im.save()
    os.remove(tmp)
    print('预览：', im.filepath_raw)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    tiers = [t for t in argv if t in TIERS] or list(TIERS)
    path = os.path.join(OUT, 'species_manifest.json')
    manifest = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            manifest = {a['name']: a for a in json.load(f)['assets']}
    for t in tiers:
        build_tier(t, manifest)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'assets': list(manifest.values())}, f, ensure_ascii=False, indent=1)
    if '--preview' in argv:
        for t in tiers:
            preview(t)


if __name__ == '__main__':
    main()
