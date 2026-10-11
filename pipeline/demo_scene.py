# 用已有素材生成一个 200 m 的静态场景（粘贴进 Blender 的 Scripting 运行；会清空当前场景）
import sys, importlib
sys.path.insert(0, r'D:\cs research\AR\output\pipeline')
import pipeline_rules, pipeline
importlib.reload(pipeline_rules); importlib.reload(pipeline)

pipeline.generate({
    'housing_density': 300,         # 住宅密度：200 m 场景里约 12 栋房子
    'fuel_fbfm40': 102,             # 燃料类型：草地
    'canopy_cover': 25,             # 树冠覆盖：稀疏的树
    'occupancy_probability': 0.2,   # 火灾风险：很高（95% 有火，红圈是火的位置）
    'fire_radiative_power': None,   # 没有 FRP，随机抽火势
}, seed=7)
