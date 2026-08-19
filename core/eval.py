# -*- coding: utf-8 -*-
"""core/eval.py — 表面规格系统的快速指标（optiland paraxial + 真实追迹）
EFFL(d) / BFL / AXCL(F−C) / 总长 / RSCE —— 全在网页实时计算
定义与 evaluate_optiland 完全一致（lens.paraxial.f2 / _bfl_from_paraxial / spot_rms）
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

from core.lens_io import build_lens_from_specs
from core._bridge import spot_rms, _bfl_from_paraxial


def paraxial_trace(specs, h0, wl=0.58756):
    """近轴边缘光线逐面追迹（兜底用；主路径用 optiland lens.paraxial.marginal_ray）
    返回 (zs, ys, u_out)：每面位置/光线高度/出射角（近轴近似）
    """
    def _n_at(glass, nd, vd, lam):
        if not glass:
            return 1.0
        if nd <= 1.0:
            nd, vd = 1.5, 60.0
        # Abbe 近似（d 波长精度足够画图用）
        lF, ld, lC = 0.48613, 0.58756, 0.65627
        if abs(vd) < 1e-6:
            return nd
        nF = nd + (nd - 1.0) / vd * 0.5
        nC = nd - (nd - 1.0) / vd * 0.5
        if abs(lam - lC) < 1e-9:
            return nC
        return nC + (nF - nC) * (lF ** 2 * (lam ** 2 - lC ** 2)) / (lam ** 2 * (lF ** 2 - lC ** 2))

    n = len(specs)
    z, y, u = 0.0, float(h0), 0.0
    zs, ys = [0.0], [y]
    for i in range(1, n):
        t_prev = specs[i - 1]['t']
        if np.isfinite(t_prev) and t_prev != float('inf'):
            y += u * t_prev
            z += t_prev
        zs.append(z)
        ys.append(y)
        R = specs[i]['R']
        if np.isfinite(R) and abs(R) > 1e-9:
            n_prev = _n_at(specs[i - 1]['glass'], specs[i - 1]['nd'], specs[i - 1]['vd'], wl)
            n_next = _n_at(specs[i]['glass'], specs[i]['nd'], specs[i]['vd'], wl)
            u += y * (n_prev - n_next) / (n_next * R)
    return zs, ys, u


def evaluate_specs(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                   wavelengths=None):
    """快速指标（定义与 evaluate_optiland 一致）：
    efl(主波长=索引0) / bfl / axcl(=BFL_末−BFL_首) / total / rsce_um / valid
    """
    if not specs:
        return {'efl': np.nan, 'bfl': np.nan, 'axcl': np.nan,
                'total': np.nan, 'rsce_um': np.nan, 'valid': False}
    try:
        lens = build_lens_from_specs(specs, epd, fields, wavelengths=wavelengths)
        if lens is None:
            return {'efl': np.nan, 'bfl': np.nan, 'axcl': np.nan,
                    'total': np.nan, 'rsce_um': np.nan, 'valid': False}
        n_wl = len(lens.wavelengths)
        efls, bfls = [], []
        for i in range(n_wl):
            lens.wavelengths.primary_index = i
            efls.append(float(lens.paraxial.f2()))
            bfls.append(_bfl_from_paraxial(lens))
        lens.wavelengths.primary_index = 0
        axcl = bfls[-1] - bfls[0]
        total = 0.0
        for s in specs[:-1]:
            if np.isfinite(s['t']) and s['t'] != float('inf'):
                total += s['t']
        rsce_mm = spot_rms(lens)
        return {'efl': float(efls[0]), 'bfl': float(bfls[0]), 'axcl': float(axcl),
                'total': float(total), 'rsce_um': float(rsce_mm) * 1e3,
                'valid': True}
    except Exception:
        return {'efl': np.nan, 'bfl': np.nan, 'axcl': np.nan,
                'total': np.nan, 'rsce_um': np.nan, 'valid': False}


def _bfl_efl_at(specs, epd=40.0, wl=0.58756, fields=(0.0,)):
    """单波长近轴 BFL/EFL（色差曲线数据源）"""
    try:
        lens = build_lens_from_specs(specs, epd=epd, fields=fields,
                                     wavelengths=[(wl, 1.0)])
        if lens is None:
            return None
        return (_bfl_from_paraxial(lens), float(lens.paraxial.f2()))
    except Exception:
        return None


if __name__ == '__main__':
    import config as CFG
    from core.lens_io import elite_to_specs

    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    airs = [39.9, 32.3, 27.6, 9.6, 3.0]
    specs = elite_to_specs(gp, airs)
    m = evaluate_specs(specs, epd=CFG.ENPD, fields=(0.0, 2.0, 4.06, 5.8))
    print('evaluate_specs:', m)
    assert m['valid'] and 195 < m['efl'] < 205, f"EFFL 不合理: {m['efl']}"
    assert m['rsce_um'] > 0, 'RSCE 无效'
    from core._bridge import build_optiland, spot_rms as _srm
    lens_ev = build_optiland(gp, airs, epd=CFG.ENPD, fields=(0.0, 2.0, 4.06, 5.8))
    ev_efl = float(lens_ev.paraxial.f2())
    ev_rsce = float(_srm(lens_ev)) * 1e3
    print(f'桥构建: EFFL={ev_efl:.3f} RSCE={ev_rsce:.2f} µm')
    d1 = abs(m['efl'] - ev_efl)
    print(f'EFFL 差: {d1:.4f}')
    assert d1 < 0.5, 'EFFL 与桥构建偏差过大'
    print('EVAL_SPECS_MATCH: OK（新路径一致）')
