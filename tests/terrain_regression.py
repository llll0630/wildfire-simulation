# ===============================================================
#  地形回归测试：18 个地点（12 个示例 + 6 个海边/国境），每个生成一次场景并检查
#    python tests/terrain_regression.py --blender "D:\blender.exe"
#  需要先运行 pipeline/setup_terrain_data.py；不需要原始栅格（示例地点用数据包里的数据，
#  海边/国境用全国 3 km 粗数据，所以这 6 个地点的近景很粗，出生点和房子的检查只作参考）
#  检查项和上限见 pipeline/terrain_core.py 的 CHECK_LIMITS；结果在 tests/output/regression.jsonl
# ===============================================================
import argparse, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(ROOT, 'pipeline')
OUT = os.path.join(ROOT, 'tests', 'output')
sys.path.insert(0, PIPE)
EXAMPLES = ['tallahassee_urban_woodland', 'boulder_foothills', 'yosemite_valley_area', 'flint_hills_mixed_landscape',
            'central_iowa', 'memphis_riverfront', 'new_york_central_park_area', 'mojave_desert_area',
            'everglades_area', 'mount_rainier_area', 'great_smoky_mountains_area', 'death_valley_area']
COARSE = {'key_west': (-81.78, 24.5551), 'cape_hatteras': (-75.526, 35.25), 'montauk': (-71.945, 41.035),
          'san_diego_border': (-117.03, 32.58), 'glacier_border': (-114.0, 48.95), 'detroit_border': (-83.05, 42.33)}
INPUTS = {'housing_density': 200, 'fuel_fbfm40': 122, 'canopy_cover': 35,      # 占位输入，和测试时一样
          'occupancy_probability': 0.2, 'fire_radiative_power': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blender', required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--glb', action='store_true', help='同时导出 GLB（慢，用来测时间和文件大小）')
    a = ap.parse_args()
    from terrain_core import CHECK_LIMITS
    os.makedirs(OUT, exist_ok=True)
    inputs = os.path.join(OUT, 'inputs.json')
    with open(inputs, 'w', encoding='utf-8') as f:
        json.dump(INPUTS, f)
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    results, failed = [], 0
    for site in EXAMPLES + list(COARSE):
        npz = os.path.join(OUT, f'{site}.npz')
        cmd = [sys.executable, os.path.join(PIPE, 'terrain_data.py'), '--out', npz]
        cmd += ['--example', site] if site in EXAMPLES else ['--lon', str(COARSE[site][0]), '--lat', str(COARSE[site][1]), '--coarse']
        subprocess.run(cmd, check=True, capture_output=True, env=env)
        cmd = [a.blender, '-b', '--factory-startup', '--python', os.path.join(PIPE, 'pipeline.py'), '--',
               '--input', inputs, '--terrain', npz, '--seed', str(a.seed), '--out', os.path.join(OUT, site)]
        if a.glb:
            cmd.append('--glb')
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
        line = next((l for l in r.stdout.splitlines() if l.startswith('[pipeline-json] ')), None)
        if line is None:
            failed += 1
            print(f'{site:28} 运行失败\n' + '\n'.join((r.stdout + r.stderr).splitlines()[-12:]))
            continue
        d = json.loads(line[len('[pipeline-json] '):])
        c = d['checks']
        bad = [f'{k}={c[k]}' for k, lim in CHECK_LIMITS.items() if c.get(k) is not None and c[k] > lim]
        coarse_note = '（粗数据）' if site in COARSE else ''
        if bad and site in EXAMPLES:
            failed += 1
        results.append({'site': site, **d})
        print(f"{site:28} 摆放 {d['stats']['placed']} 中景树 {d['stats']['mid_trees']:5} 用时 {d['timing_s']['total']:5.1f}s "
              f"{'通过' if not bad else '未通过' + coarse_note + ': ' + '; '.join(bad)}")
    with open(os.path.join(OUT, 'regression.jsonl'), 'w', encoding='utf-8') as f:
        for r_ in results:
            f.write(json.dumps(r_, ensure_ascii=False) + '\n')
    print(f'\n{len(results)} 个地点完成，{failed} 个示例地点未通过')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
