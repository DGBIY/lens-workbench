# -*- coding: utf-8 -*-
"""core/merit.py — 评价函数系统（v0.24，Zemax MFE 风格）

操作数（每项：目标值 + 权重）：
  EFFL 焦距(mm) | AXCL 轴向色差(mm) | FCGT/FCGS 子午/弧矢场曲(mm)
  RSCE 星点RMS(mm) | DIST 畸变(%) | WFE RMS波前(um) | TOTR 总长(mm)
  BFL 后焦(mm) | MTF 高斯近似(30lp/mm) | THF 离焦近似RMS(mm) | RELIL 边缘相对照度(cos^4)

MFE = sqrt(Σ w·Δ² / Σ w)，Δ = 实际 − 目标（Zemax 惯例）
评估全部 optiland 真实追迹，零 Zemax 依赖。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core.lens_io import build_lens_from_specs
from core._bridge import spot_rms, _bfl_from_paraxial
from optiland.analysis import FieldCurvature, Distortion

OPS = ['EFFL', 'AXCL', 'FCGT', 'FCGS', 'RSCE', 'DIST', 'WFE',
       'TOTR', 'BFL', 'MTF', 'THF', 'RELIL']

PRESETS = {
    '深空 APO 推荐': [('EFFL', 200.0, 10), ('AXCL', 0.1, 20),
                    ('FCGT', 0.2, 10), ('FCGS', 0.2, 10), ('RSCE', 0.05, 10)],
    '仅焦距': [('EFFL', 200.0, 10)],
    '星点优先': [('EFFL', 200.0, 10), ('RSCE', 0.05, 30)],
    '完整（含畸变）': [('EFFL', 200.0, 10), ('AXCL', 0.1, 20),
                    ('FCGT', 0.2, 10), ('FCGS', 0.2, 10),
                    ('RSCE', 0.05, 10), ('DIST', 1.0, 5)],
}


def merit_from_preset(name, target_efl=200.0):
    """预设 → 操作数 dict 列表（EFFL 目标用用户输入覆盖）"""
    return [{'op': op, 'target': (target_efl if op == 'EFFL' else t),
             'weight': w} for op, t, w in PRESETS.get(name, PRESETS['深空 APO 推荐'])]


def compute_operands(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                     wavelengths=None, include_heavy=False):
    """全部操作数实际值 dict（mm / % / 无量纲；计算失败项 = nan）
    include_heavy=True 时才计算 WFE（波前分析极慢，仅"完整"预设用）"""
    out = {}
    try:
        lens = build_lens_from_specs(specs, epd, fields, wavelengths=wavelengths)
        if lens is None:
            return out
        n_wl = len(lens.wavelengths)
        bfls = []
        for i in range(n_wl):
            lens.wavelengths.primary_index = i
            bfls.append(_bfl_from_paraxial(lens))
        lens.wavelengths.primary_index = 0
        out['EFFL'] = float(lens.paraxial.f2())
        out['AXCL'] = float(bfls[-1] - bfls[0])
        out['BFL'] = float(bfls[0])
        total = 0.0
        for s in specs[:-1]:
            if np.isfinite(s['t']) and s['t'] != float('inf'):
                total += s['t']
        out['TOTR'] = total
        try:
            fc = FieldCurvature(lens, wavelengths='all', num_points=15)
            fc_d = np.asarray(fc.data)
            # (n_wl, 2[子午,弧矢], n_pts)
            out['FCGT'] = float(np.max(np.abs(fc_d[:, 0, :])))
            out['FCGS'] = float(np.max(np.abs(fc_d[:, 1, :])))
        except Exception:
            out['FCGT'] = out['FCGS'] = float('nan')
        rsc_mm = float(spot_rms(lens))
        out['RSCE'] = rsc_mm
        try:
            d_ = Distortion(lens)
            out['DIST'] = float(np.max(np.abs(np.asarray(d_.data))))
        except Exception:
            out['DIST'] = float('nan')
        if include_heavy:
            try:
                from optiland.analysis import RmsWavefrontErrorVsField
                wfe = RmsWavefrontErrorVsField(lens)
                out['WFE'] = float(np.max(np.abs(np.asarray(wfe.data)))) * 1e3  # um
            except Exception:
                out['WFE'] = float('nan')
        # MTF 高斯近似：MTF(f) = exp(-2π²σ²f²)，σ=RSCE(mm)，f=30 lp/mm
        out['MTF'] = float(np.exp(-2 * np.pi ** 2 * rsc_mm ** 2 * 900.0))
        # THF 近似：离焦 ±2λF#² 处 RMS（几何离焦圆叠加）
        fno = epd / out['EFFL'] if out['EFFL'] > 0 else 5.0
        dz = 2 * 0.00055 * fno * fno
        out['THF'] = float(np.sqrt(rsc_mm ** 2 + (dz / (2 * fno)) ** 2))
        # RELIL：cos⁴ 边缘照度（最大半视场角）
        max_f = max((f[0] if isinstance(f, (tuple, list)) else f) for f in fields)
        out['RELIL'] = float(np.cos(np.radians(max_f)) ** 4)
    except Exception:
        pass
    return out


def merit_value(ops, merit):
    """MFE = sqrt(Σ w·Δ² / Σ w)；任一操作数缺失/无效 → 1e9"""
    s = 0.0
    wsum = 0.0
    for m in merit:
        v = ops.get(m['op'], float('nan'))
        if not np.isfinite(v):
            return 1e9
        w = float(m['weight'])
        d = v - float(m['target'])
        s += w * d * d
        wsum += w
    return float(np.sqrt(s / wsum)) if wsum > 0 else 1e9


if __name__ == '__main__':
    from core.lens_io import elite_to_specs
    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    specs = elite_to_specs(gp, [39.9, 32.3, 27.6, 9.6, 3.0])
    ops = compute_operands(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8))
    ok = {k: (round(v, 4) if np.isfinite(v) else None) for k, v in ops.items()}
    print('operands:', ok)
    mfe = merit_value(ops, merit_from_preset('深空 APO 推荐'))
    print('MFE(深空APO推荐):', round(mfe, 4), '（Zemax 定稿 0.2587 对照）')
    assert 195 < ops['EFFL'] < 205, ops['EFFL']
    assert abs(ops['AXCL']) < 1.0, ops['AXCL']
    assert 0.0 < ops['RSCE'] < 0.5, ops['RSCE']
    assert 0.0 < ops['FCGT'] < 1.0 and 0.0 < ops['FCGS'] < 1.0
    assert 0.0 <= ops['RELIL'] <= 1.0
    assert 0.0 < mfe < 1.0, mfe
    print('MERIT: OK')
