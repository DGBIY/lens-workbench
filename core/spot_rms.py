# -*- coding: utf-8 -*-
"""core/spot_rms.py — 点阵图 + RMS（表面规格版，optiland 真实追迹）"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from core.lens_io import build_lens_from_specs
from optiland.analysis import SpotDiagram

_WL_COLORS = ['b', 'r', 'g']  # F/d/C


def compute_spot(specs, fields=(0.0, 2.0, 4.06, 5.8), epd=40.0,
                 wavelengths=None):
    """点阵图 + 每场 RMS（µm）。
    返回 (fig, rms_list) 或 (None, []) 失败。
    rms_list: [(field_norm, rms_um_max, rms_um_primary)]
    """
    try:
        lens = build_lens_from_specs(specs, epd=epd, fields=fields,
                                     wavelengths=wavelengths)
        if lens is None:
            return None, []
        sd = SpotDiagram(lens, fields='all', wavelengths='all',
                         num_rings=6, reference='chief_ray')
    except Exception:
        return None, []

    try:
        rms_all = np.asarray(sd.rms_spot_radius())   # (n_field, n_wl)，单位 mm
        data = sd.data
        nf = len(data)
    except Exception:
        return None, []

    rms_list = []
    fig, axes = plt.subplots(1, nf, figsize=(4.5 * nf, 4))
    if nf == 1:
        axes = [axes]
    for fi, (ax, field_data) in enumerate(zip(axes, data)):
        for wi, wl_data in enumerate(field_data):
            x = np.asarray(wl_data.x, dtype=float) * 1e3   # mm → µm
            y = np.asarray(wl_data.y, dtype=float) * 1e3
            ax.scatter(x, y, s=3, alpha=0.55, color=_WL_COLORS[wi % 3],
                       label=f'λ{wi}' if fi == 0 else None)
        rms_um_all = float(np.max(rms_all[fi]) * 1e3)
        rms_um_pri = float(rms_all[fi][0] * 1e3)
        field_norm = float(sd.fields[fi].coord[1]) if hasattr(sd.fields[fi], 'coord') else float(fi)
        rms_list.append((field_norm, rms_um_all, rms_um_pri))
        # 艾里斑圈（Zemax 同款：1.22λF#）
        try:
            _wl0 = float(lens.wavelengths[0].value) * 1e3
            _f2 = float(lens.paraxial.f2())
            _fno = epd / _f2 if _f2 > 0 else 0.0
            if _fno > 0:
                ax.add_patch(plt.Circle((0, 0), 1.22 * _wl0 * _fno, fill=False,
                                        color='k', lw=0.8, ls='--'))
        except Exception:
            pass
        ax.set_aspect('equal')
        ax.set_title(f'场 {field_norm:.2f} (norm)\nRMS {rms_um_all:.1f} µm')
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.grid(alpha=0.3)
    if nf > 1:
        axes[0].legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    return fig, rms_list


if __name__ == '__main__':
    import config as CFG
    from core.lens_io import elite_to_specs

    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    airs = [39.9, 32.3, 27.6, 9.6, 3.0]
    specs = elite_to_specs(gp, airs)
    fig, rms = compute_spot(specs, epd=CFG.ENPD)
    print('RMS 每场 (norm, max_um, primary_um):', rms)
    if fig:
        fig.savefig('C:/Users/Administrator/Desktop/GA/窗口/_test_spot.png', dpi=100)
        print('spot(specs) 自检 OK')
    else:
        print('spot(specs) 自检失败')
