# -*- coding: utf-8 -*-
"""ci_test.py — CI 轻量回归（GitHub Actions / 本地通用，v0.27）

覆盖（不依赖完整项目/python_env，纯窗口仓库内容）：
  1. 语法：全部窗口 .py 文件 py_compile
  2. merit 自检（12 操作数一致性）
  3. 内置 GA smoke（进程池评估，pop6×gens3 单调收敛）
  4. 真实星图数据（猎户座 M42 11 颗全在视场）
  5. 星场渲染（mode='real'）
  6. 2D 布局渲染

用法: python ci_test.py        # 退出码 0 = 全部通过
"""
import matplotlib
matplotlib.use('Agg')   # CI 无显示环境
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('WORKBENCH_DATA', os.path.join(HERE, 'data'))
os.environ.setdefault('GA_PROJECT_ROOT', '')   # CI 无完整项目（近轴引擎跳过）


def main():
    fails = []
    step = lambda name: print(f'== {name}')

    # 1. 语法检查
    step('语法检查')
    import py_compile
    n = 0
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.streamlit')]
        for fn in files:
            if fn.endswith('.py'):
                py_compile.compile(os.path.join(root, fn), doraise=True)
                n += 1
    print(f'[PASS] 语法 {n} 个文件')

    # 2. merit 自检
    step('merit 自检')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'core', 'merit.py')],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace')
    if r.returncode != 0:
        fails.append('merit 自检')
        print('[FAIL] merit 自检'); print(r.stderr[-2000:])
    else:
        print('[PASS] merit 自检:', (r.stdout.strip().splitlines() or [''])[-1])

    # 3. 内置 GA smoke
    step('内置 GA smoke')
    from core.ga_engine import run_ga
    from core.merit import merit_from_preset
    t0 = time.time()
    try:
        bi, bs, hist = run_ga(n_groups=6,
                              merit=merit_from_preset('深空 APO 推荐'),
                              pop=6, gens=3, seed_templates=['tessar'], seed=7)
        assert all(hist[i][1] <= hist[i - 1][1] + 1e-9
                   for i in range(1, len(hist))), '收敛不单调'
        print(f'[PASS] GA smoke（{time.time() - t0:.1f}s best={hist[-1][1]:.2f} 单调）')
    except Exception as e:
        fails.append('GA smoke')
        print(f'[FAIL] GA smoke: {e}')

    # 4. 真实星图数据
    step('真实星图数据')
    from core.starfield import _gen_real, REAL_ORION
    try:
        stars = _gen_real()
        assert len(stars) == len(REAL_ORION), f'{len(stars)} != {len(REAL_ORION)}'
        assert all(abs(hx) <= 1.2 and abs(hy) <= 1.2 for hx, hy, _ in stars)
        print(f'[PASS] 真实星图 {len(stars)} 颗全在视场')
    except Exception as e:
        fails.append('真实星图')
        print(f'[FAIL] 真实星图: {e}')

    # 5. 星场渲染（真实星图模式）
    step('星场渲染')
    from core.starfield import render_starfield
    try:
        fig = render_starfield(bs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                               mode='real', scale=12.0, annotate=False)
        assert fig is not None
        print('[PASS] 星场渲染（真实星图）')
    except Exception as e:
        fails.append('星场渲染')
        print(f'[FAIL] 星场渲染: {e}')

    # 6. 2D 布局渲染
    step('2D 布局渲染')
    from core.layout import render_layout
    try:
        fig2 = render_layout(bs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8), efl=200.0)
        assert fig2 is not None
        print('[PASS] 2D 布局渲染')
    except Exception as e:
        fails.append('2D 布局')
        print(f'[FAIL] 2D 布局: {e}')

    print('=' * 56)
    if fails:
        print(f'CI_FAIL: {len(fails)} 项失败 -> {fails}')
        sys.exit(1)
    print('CI_OK: 全部通过 ✅')
    sys.exit(0)


if __name__ == '__main__':
    main()
