# ===============================================================
#  原始栅格放到服务器后，先跑这个确认数据和读取都对：
#    python tests/check_rasters.py [--sources terrain_data/source_paths.json]
#  做法：数据包里的 12 个示例地点就是作者用同一套原始栅格生成的。
#  用栅格按同样的范围（3 km、60 m 一格）重新读一遍，结果应该和示例逐格一致（误差 0）。
#  不一致说明栅格版本不对（必须是 LANDFIRE 2020：LC20_EVT_220、LF2020_Elev、LF2020_SlpD）或路径写错。
#  另外会测一次正式读取（6 km、30 m）的用时。
# ===============================================================
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
import terrain_data as td


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', default=td.SOURCES)
    ap.add_argument('--sites', default=None, help='逗号分隔；默认 12 个全部')
    a = ap.parse_args()
    out = os.path.join(ROOT, 'tests', 'output', 'check_rasters')
    os.makedirs(out, exist_ok=True)
    sites = a.sites.split(',') if a.sites else sorted(f[:-len('.local.json')] for f in os.listdir(os.path.join(td.BUNDLE, 'examples'))
                                                      if f.endswith('.local.json'))
    context, reader = td.open_reader(a.sources)
    bad = 0
    try:
        for site in sites:
            with open(os.path.join(td.BUNDLE, 'examples', f'{site}.local.json'), encoding='utf-8') as f:
                ex = json.load(f)
            lon, lat = ex['selection']['longitude'], ex['selection']['latitude']
            ref = os.path.join(out, f'{site}.example.npz'); got = os.path.join(out, f'{site}.raster.npz')
            td.from_example(site, ref)
            td.from_rasters(context, reader, lon, lat, got, width_m=3000, cell_m=60)
            r, g = np.load(ref), np.load(got)
            same_grid = abs(float(r['local_ox']) - float(g['local_ox'])) < 1e-6 and abs(float(r['local_oy']) - float(g['local_oy'])) < 1e-6
            de = float(np.nanmax(np.abs(r['local_elev'] - g['local_elev'])))
            dc = float(np.nanmax(np.abs(r['local_cover'] - g['local_cover'])))
            t0 = time.perf_counter()
            td.from_rasters(context, reader, lon, lat, os.path.join(out, f'{site}.6km.npz'))
            secs = time.perf_counter() - t0
            ok = same_grid and de < 0.01 and dc < 1e-6
            bad += not ok
            print(f"{site:30} {'一致' if ok else '不一致'}  高度最大差 {de:.3f} m  地表最大差 {dc:.4f}  6 km 读取 {secs:.2f} s")
    finally:
        reader.close()
    print(f'\n{len(sites)} 个地点，{bad} 个不一致')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
