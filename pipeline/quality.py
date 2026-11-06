# ===============================================================
#  画质档位（pipeline.py --quality low|medium|high）：每档用哪套素材
#  低档 = assets/ 里原来的素材，走原来的代码，输出和不加 --quality 时逐字节一致
#  中档、高档 = assets_hq/<档位>/（asset_builders/build_hq_vegetation.py 生成，只有导出版 *_export.blend）
#  用户先选档位再生成场景；换档就重新生成，不要求两档的场景一样
# ===============================================================
import os, json
import bpy
import pipeline_rules as rules

ROOT = os.path.dirname(rules.ASSET_DIR)
DEFAULT = 'low'
KINDS = ('trees', 'shrubs', 'grass')

# 每档每类素材从哪个文件夹取（相对仓库根目录）
#   far / far_kinds / near_half：高档的树只在用户周围 ±60 m 里用高档模型，再远（到 ±150 m）换成中档模型——
#   远处看不清细节；森林地点高档模型全放到 ±110 m 时一个场景接近 1000 万个三角形
#   ground：地面贴图文件夹（asset_builders/build_hq_ground.py 生成）；bake：近景地面烘焙图边长（像素，覆盖 ±110 m）
#     试过高档用 4096：近看和 2048 几乎看不出差别（近处的细节主要来自细节法线），生成却慢 40 秒，所以都用 2048
#   detail：近景地面加一张每 2.5 m 重复的细节法线贴图（第二套 UV），值是强度；0 = 不加
#   只对地形场景（--terrain）有效；平地场景的地面中高档也用低档的
TIERS = {
    'low':    {'trees': 'assets', 'shrubs': 'assets', 'grass': 'assets', 'ground': 'assets_hq', 'bake': 2048, 'detail': 0},
    'medium': {'trees': 'assets_hq/medium', 'shrubs': 'assets_hq/medium', 'grass': 'assets_hq/medium',
               'ground': 'assets_hq', 'bake': 2048, 'detail': 0.5},
    'high':   {'trees': 'assets_hq/high', 'shrubs': 'assets_hq/high', 'grass': 'assets_hq/high',
               'far': 'medium', 'far_kinds': ('trees',), 'near_half': 60.0,
               'ground': 'assets_hq', 'bake': 2048, 'detail': 0.8},
}


SPECIES_MANIFEST = os.path.join(ROOT, 'assets_hq', 'species_manifest.json')


def species_manifest():
    """分树种素材清单（asset_builders/build_species.py 生成）：[{name, kind, species, veg_types, ground_types, ...}]"""
    if not os.path.exists(SPECIES_MANIFEST):
        return []
    with open(SPECIES_MANIFEST, encoding='utf-8') as f:
        return json.load(f)['assets']


def species_path(tier, kind):
    return os.path.join(ROOT, 'assets_hq', tier, 'species', kind, f'{kind}_export.blend')


def lib_path(tier, lib):
    """素材库文件：lib = billboards / nature / props / farm / building_parts（assets_hq/<档位>/<lib>/<lib>_export.blend）"""
    return os.path.join(ROOT, 'assets_hq', tier, lib, f'{lib}_export.blend')


def buildings_dir():
    return os.path.join(ROOT, 'assets_hq', 'buildings')


def roads_dir():
    return os.path.join(ROOT, 'assets_hq', 'roads')


def ground_dir(tier):
    return os.path.join(ROOT, TIERS[tier]['ground'], 'ground')


def blend_path(tier, kind):
    return os.path.join(ROOT, TIERS[tier][kind], kind, f'{kind}_export.blend')


def load(path, names):
    """追加 Collection，按要的名字返回（同名素材从两个文件各追加一次时，第二份会被 Blender 改名成 xxx.001）"""
    names = sorted(names)
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = list(names)               # 传一份拷贝：加载后 Blender 会把这个列表里的名字换成 Collection
    return dict(zip(names, dst.collections))


def is_near(cfg, it):
    return max(abs(it['x']), abs(it['y'])) <= cfg['near_half']


def vegetation_groups(tier, layout, export=True):
    """返回 {类别: [(实例组名, Collection, 这一组的摆放点), …]}。
    分树种的素材（species_manifest.json 里有的名字）从 assets_hq/<档位>/species/ 取；其余按 TIERS。
    export=False（不导出 GLB）时低档的通用素材用原来的 *.blend（分树种素材只有导出版）"""
    cfg = TIERS[tier]
    sp_names = {a['name'] for a in species_manifest()}
    out = {}
    for k in KINDS:
        if k in cfg.get('far_kinds', ()):
            bands = [('', tier, lambda it: is_near(cfg, it)), ('_far', cfg['far'], lambda it: not is_near(cfg, it))]
        else:
            bands = [('', tier, lambda it: True)]
        out[k] = []
        for sfx, t, pick in bands:
            items = {}
            for it in layout[k]:
                if pick(it): items.setdefault(it['name'], []).append(it)
            base = {n: v for n, v in items.items() if n not in sp_names}
            spec = {n: v for n, v in items.items() if n in sp_names}
            colls = {}
            if base:
                path = blend_path(t, k)
                if not export and TIERS[t][k] == 'assets':
                    path = os.path.join(ROOT, 'assets', k, f'{k}.blend')
                colls.update(load(path, base))
            if spec:
                colls.update(load(species_path(t, k), spec))
            out[k] += [(name + sfx, colls[name], items[name]) for name in sorted(items)]
    return out
