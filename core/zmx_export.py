# -*- coding: utf-8 -*-
"""core/zmx_export.py — 表面规格 → Zemax .zmx 文本文件
格式逐行对齐 scripts/initial_6lens.zmx 模板（VERS/MODE/ENPD/WAVM/SURF 块）
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np


def _curv(R):
    """曲率 = 1/R；平面 → 0"""
    if not np.isfinite(R) or abs(R) < 1e-12:
        return 0.0
    return 1.0 / R


def _disz(t):
    if not np.isfinite(t):
        return 'INFINITY'
    return f'{t:.12g}'


def _semi(s):
    if not np.isfinite(s) or s <= 0:
        return '0'
    return f'{s:.12g}'


def _surface_block(idx, R, t, glass, semi, is_stop=False, is_image=False, slab=None):
    lines = [f'SURF {idx}', f'  SSID {idx}']
    if is_stop:
        lines.append('  STOP')
    if is_image:
        lines.append('  IMA')
    lines += ['  TYPE STANDARD', '  FIMP ', '',
              f'  CURV {_curv(R):.17g} 0 0 0 0 ""',
              '  HIDE 0 0 0 0 0 0 0 0 0 0 0 0',
              '  MIRR 2 0',
              f'  SLAB {slab if slab is not None else idx}',
              f'  DISZ {_disz(t)}']
    if glass:
        lines.append(f'  GLAS {glass} 0 0 1.5 40 0 0 0 0 0 0 ')
    lines += [f'  DIAM {_semi(semi)} 0 0 0 1 ""',
              f'  MEMA {_semi(semi)} 0 0 0 1 ""',
              '  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0']
    return '\n'.join(lines)


def specs_to_zmx(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8), name='design'):
    """生成完整 .zmx 文本（Zemax 序列模式，可直接打开）"""
    n = len(specs)
    wavs = (0.48613, 0.58756, 0.65627)
    head = [
        'VERS 231205 2763 20120530 20120530',
        'MODE SEQ',
        f'NAME {name}',
        '',
        'AUTH ',
        '',
        'IWDP 0',
        'PFIL 0 0 0',
        'LANG 2',
        'UNIT MM X W X CM MR CPMM',
        f'ENPD {epd:g}',
        'ENVD 20 1 0',
        'GFAC 0 0',
        'GCAT SCHOTT HIKARI CDGM GBJ ',
        'RAIM 0 0 1 1 0 0 0 0 0 1',
        'PUSH 0 0 0 0 0 0',
        'SDMA 0 1 0',
        'OMMA 1 1',
        'FTYP 0 0 5 3 0 0 0 5',
        'ROPD 2',
        'HYPR 0',
        'PICB 1',
        'XFLN 0 0 0 0 0',
        'YFLN 0 0 ' + ' '.join(f'{y:.6g}' for y in fields),
        'FWGN ' + ' '.join(['1'] * (len(fields) + 2)),
        'VDXN 0 0 0 0 0',
        'VDYN 0 0 0 0 0',
        'VCXN 0 0 0 0 0',
        'VCYN 0 0 0 0 0',
        'VANN 0 0 0 0 0',
    ]
    for i, w in enumerate(wavs, 1):
        head.append(f'WAVM {i} {w:.14g} 1')
    head.append('PWAV 1')
    head.append('POLS 1 0 1 0 0 1 0')
    head.append('GLRS 1 0')
    head.append('GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1')
    head.append('NSCD 100 500 0 0.001 10 9.9999999999999995e-07 0 0 0 0 0 0 1000000 0 2')
    head.append('COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"')
    head.append('')

    blocks = []
    for i, s in enumerate(specs):
        blocks.append(_surface_block(
            s['idx'], s['R'], s['t'], s['glass'], s['semi'],
            is_stop=s['is_stop'], is_image=s['is_image'], slab=i))
    return '\n'.join(head + blocks + [''])


if __name__ == '__main__':
    from core.lens_io import elite_to_specs
    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    airs = [39.9, 32.3, 27.6, 9.6, 3.0]
    specs = elite_to_specs(gp, airs)
    txt = specs_to_zmx(specs, epd=40.0, name='elite55')
    out = 'C:/Users/Administrator/Desktop/GA/窗口/_test_elite55.zmx'
    open(out, 'w', encoding='utf-8').write(txt)
    lines = txt.splitlines()
    n_surf = sum(1 for l in lines if l.startswith('SURF '))
    n_glas = sum(1 for l in lines if l.startswith('  GLAS '))
    print(f'zmx 生成: {len(lines)} 行, SURF {n_surf} 个, GLAS {n_glas} 个')
    assert n_surf == 18 and n_glas == 9, f'表面/玻璃数不对: {n_surf}/{n_glas}'
    print('ZMX_EXPORT: OK')
    print(txt.splitlines()[60:70])
