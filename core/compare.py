# -*- coding: utf-8 -*-
"""core/compare.py — 多精英对比（指标表 + 并排 2D 图）"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd

from core.lens_io import load_elite_specs
from core.eval import evaluate_specs


def load_multi(fpath, idxs):
    """从精英文件载入多个序号 → [{label, specs}]（跳过解析失败的）"""
    items = []
    for ix in idxs:
        try:
            specs = load_elite_specs(fpath, int(ix))
        except Exception:
            specs = None
        if specs:
            items.append({'label': f'#{int(ix)}', 'specs': specs})
    return items


def compare_table(items, epd=40.0):
    """对比指标表：精英 / EFFL / AXCL / 总长 / RSCE"""
    rows = []
    for it in items:
        m = evaluate_specs(it['specs'], epd=epd)
        rows.append({
            '精英': it['label'],
            'EFFL (mm)': round(m['efl'], 2) if m['valid'] else '—',
            'AXCL (mm)': round(m['axcl'], 4) if m['valid'] else '—',
            '总长 (mm)': round(m['total'], 1) if m['valid'] else '—',
            'RSCE (µm)': round(m['rsce_um'], 0) if m['valid'] else '—',
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'data', 'samples', 'ga_elite.txt')
    items = load_multi(f, [55, 54, 1])
    print('载入:', [it['label'] for it in items])
    df = compare_table(items)
    print(df.to_string(index=False))
    print('COMPARE: OK')
