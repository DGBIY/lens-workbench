# -*- coding: utf-8 -*-
"""core/analysis.py — 像差分析 + 优化 + 玻璃工具（v0.9）
场曲/畸变 · RayFan · RMS-vs-场 · EE · 畸变网格 · 透过焦 · RMS波前 ·
MTF vs 频率（PSF FFT）· 轴向色差 · 局部优化（含曲率/厚度变量）· 玻璃搜索
"""
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
from optiland.analysis import (FieldCurvature, Distortion, RayFan,
                               RmsSpotSizeVsField, EncircledEnergy,
                               GridDistortion, ThroughFocusSpotDiagram,
                               ThroughFocusMTF, RmsWavefrontErrorVsField,
                               IncoherentIrradiance)

DEFAULT_WAVS = [(0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)]
DEFAULT_FIELDS = (0.0, 2.0, 4.06, 5.8)

KINDS = {
    'field_curvature': '场曲 Field Curvature',
    'distortion': '畸变 Distortion',
    'ray_fan': '光线扇形图 Ray Fan',
    'rms_vs_field': 'RMS vs 视场',
    'encircled_energy': '能量集中度 Encircled Energy',
    'grid_distortion': '畸变网格 Grid Distortion',
    'through_focus_spot': '透过焦 Spot（对焦敏感度）',
    'through_focus_mtf': '透过焦 MTF',
    'rms_wavefront': 'RMS 波前误差 vs 场',
    'mtf': 'MTF vs 空间频率（星点锐度）',
}


def analysis_fig(specs, kind='field_curvature', epd=40.0,
                 fields=DEFAULT_FIELDS, wavelengths=DEFAULT_WAVS):
    """九种像差分析 → matplotlib fig（失败返回 None）"""
    lens = build_lens_from_specs(specs, epd=epd, fields=fields,
                                 wavelengths=wavelengths)
    if lens is None:
        return None
    try:
        if kind == 'field_curvature':
            obj = FieldCurvature(lens, wavelengths='all', num_points=25)
        elif kind == 'distortion':
            obj = Distortion(lens, wavelengths='all', num_points=64)
        elif kind == 'ray_fan':
            obj = RayFan(lens, fields='all', wavelengths='all', num_points=128)
        elif kind == 'rms_vs_field':
            obj = RmsSpotSizeVsField(lens, num_fields=24, wavelengths='all',
                                     num_rings=6)
        elif kind == 'encircled_energy':
            obj = EncircledEnergy(lens, fields='all', wavelength='primary',
                                  num_rays=20000, num_points=64)
        elif kind == 'grid_distortion':
            obj = GridDistortion(lens, wavelength='primary', num_points=16)
        elif kind == 'through_focus_spot':
            obj = ThroughFocusSpotDiagram(lens, delta_focus=0.1, num_steps=7,
                                          fields='all', wavelengths='all',
                                          num_rings=6)
        elif kind == 'through_focus_mtf':
            obj = ThroughFocusMTF(lens, spatial_frequency=30.0,
                                  delta_focus=0.1, num_steps=7,
                                  fields='all', wavelength='primary',
                                  num_rays=64)
        elif kind == 'rms_wavefront':
            obj = RmsWavefrontErrorVsField(lens, num_fields=24,
                                           wavelengths='all', num_rays=12)
        elif kind == 'mtf':
            return mtf_fig(specs, epd=epd, fields=fields,
                           wavelengths=wavelengths)
        else:
            return None
        fig, _ax = obj.view(show=False)
        return fig
    except Exception:
        return None


def mtf_fig(specs, epd=40.0, fields=DEFAULT_FIELDS,
            wavelengths=DEFAULT_WAVS, grid=128, max_freq=80.0):
    """MTF vs 空间频率（几何 PSF FFT，各视场 + 衍射极限虚线）
    SpotDiagram 点列 → 2D 直方图（mm 标定）→ FFT → 归一化 MTF
    """
    from optiland.analysis import SpotDiagram
    lens = build_lens_from_specs(specs, epd=epd, fields=fields,
                                 wavelengths=wavelengths)
    if lens is None:
        return None
    try:
        sd = SpotDiagram(lens, fields='all', wavelengths='all',
                         num_rings=10, reference='chief_ray')
        data = sd.data
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(data)))
        for fi, field_data in enumerate(data):
            xs, ys = [], []
            for wl_data in field_data:
                xs.extend(float(v) for v in wl_data.x)
                ys.extend(float(v) for v in wl_data.y)
            xs = np.asarray(xs) - np.asarray([v for wl in field_data
                                              for v in wl.x]).mean()
            ys = np.asarray(ys) - np.asarray([v for wl in field_data
                                              for v in wl.y]).mean()
            rmax = float(max(np.abs(xs).max(), np.abs(ys).max(), 1e-6)) * 1.2
            H, _, _ = np.histogram2d(ys, xs, bins=grid,
                                     range=[[-rmax, rmax], [-rmax, rmax]])
            mtf = np.abs(np.fft.fftshift(np.fft.fft2(H)))
            mtf = mtf / (mtf.max() + 1e-12)
            dx = 2.0 * rmax / grid   # mm/像素
            freqs = np.abs(np.fft.fftshift(np.fft.fftfreq(grid, d=dx)))
            center = grid // 2
            half = grid // 2
            ax.plot(freqs[center:center + half], mtf[center, center:center + half],
                    color=colors[fi], lw=1.4,
                    label=f'场 {fi} ({data[fi][0] and ""})' if len(data) > 1 else 'MTF')
        # 衍射极限：MTF = (2/π)(acos(u) - u·sqrt(1-u²))，u = f/fc
        wl0 = float(lens.wavelengths[0].value)
        fno = float(lens.paraxial.FNO())
        fc = 1.0 / (wl0 * 1e-3 * fno)   # lp/mm
        ff = np.linspace(0, min(max_freq, fc * 0.999), 200)
        u = ff / fc
        mtf_diff = (2.0 / np.pi) * (np.arccos(u) - u * np.sqrt(1 - u ** 2))
        ax.plot(ff, mtf_diff, 'k--', lw=1.2,
                label=f'衍射极限 (fc={fc:.0f} lp/mm)')
        ax.set_xlim(0, max_freq)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('空间频率 (lp/mm)')
        ax.set_ylabel('MTF')
        ax.set_title('MTF vs 空间频率（几何 PSF FFT）')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception:
        return None


def axial_color(specs, epd=40.0, fields=DEFAULT_FIELDS,
                wavelengths=DEFAULT_WAVS):
    """各波长 BFL/EFL → [(λ, bfl, efl)]"""
    from core.eval import _bfl_efl_at
    out = []
    for wl, _w in wavelengths:
        r = _bfl_efl_at(specs, epd=epd, wl=wl,
                        fields=(fields[0] if fields else 0.0,))
        if r:
            out.append((wl, r[0], r[1]))
    return out


def axial_color_fig(specs, epd=40.0, fields=DEFAULT_FIELDS,
                    wavelengths=DEFAULT_WAVS):
    """轴向色差图：ΔBFL vs 波长（APO 核心指标）"""
    ac = axial_color(specs, epd=epd, fields=fields, wavelengths=wavelengths)
    if len(ac) < 2:
        return None
    wls = [a[0] for a in ac]
    bfls = [a[1] for a in ac]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(wls, [b - bfls[0] for b in bfls], 'o-', lw=1.6)
    ax.axhline(0, color='gray', lw=0.7, ls='--')
    for w, b in zip(wls, bfls):
        ax.annotate(f'{b - bfls[0]:+.3f}', (w, b - bfls[0]),
                    textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel('波长 (µm)')
    ax.set_ylabel('ΔBFL (mm)')
    ax.set_title('轴向色差（后焦距差）')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ============================================================
# 玻璃搜索（按 nd/vd 找库中最接近的玻璃）
# ============================================================
def search_glass(nd_target, vd_target, top=8):
    """|Δnd|/0.01 + |Δvd|/5 加权排序 → [(玻璃名, nd, vd, score)]"""
    from core.lens_io import glass_catalog
    cat = glass_catalog()
    out = []
    for g, (nd, vd) in cat.items():
        score = abs(nd - nd_target) / 0.01 + abs(vd - vd_target) / 5.0
        out.append((g, nd, vd, score))
    out.sort(key=lambda x: x[3])
    return [(g, n, v, s) for g, n, v, s in out[:top]]


# ============================================================
# 局部优化（v0.9：空气间隔 + 后焦 + 可选厚度/曲率变量）
# ============================================================
def _air_indices(specs):
    idx = []
    for i, s in enumerate(specs):
        if i > 1 and not s['is_stop'] and not s['is_image']:
            if s['glass'] == '' and np.isfinite(s['t']) and s['t'] != float('inf'):
                idx.append(i)
    return idx


def _lens_thick_indices(specs):
    """玻璃面厚度索引（可选变量）"""
    return [i for i, s in enumerate(specs)
            if 1 < i < len(specs) - 1 and s['glass'] and s['t'] < 100]


def _lens_curv_indices(specs):
    """玻璃面曲率索引（可选变量，c=1/R 域）"""
    return [i for i, s in enumerate(specs)
            if 1 < i < len(specs) - 1 and s['glass']
            and np.isfinite(s['R']) and abs(s['R']) > 1e-9]


def _mfe_local(x, specs, epd, fields, wavs, target_efl, include_rsce,
               i_air, i_thick, i_curv, bounds_list=None):
    """局部 MFE（快速路径 num_rings=3；include_rsce=False 纯近轴）
    i_curv 的变量是 R；bounds_list 用于软惩罚（Nelder-Mead 无约束）"""
    try:
        from core.eval import _bfl_from_paraxial
        from core._bridge import spot_rms
        out = [dict(s) for s in specs]
        k = 0
        for i in i_air:
            out[i]['t'] = float(x[k]); k += 1
        for i in i_thick:
            out[i]['t'] = float(x[k]); k += 1
        for i in i_curv:
            out[i]['R'] = float(x[k]); k += 1   # 变量是 R（mm）
        out[-2]['t'] = float(x[k])   # 后焦
        lens = build_lens_from_specs(out, epd=epd, fields=fields,
                                     wavelengths=wavs)
        if lens is None:
            return 1e6
        bfls = []
        for i in range(len(lens.wavelengths)):
            lens.wavelengths.primary_index = i
            bfls.append(_bfl_from_paraxial(lens))
        lens.wavelengths.primary_index = 0
        efl = float(lens.paraxial.f2())
        axcl = bfls[-1] - bfls[0]
        if not np.isfinite(efl) or not np.isfinite(axcl):
            return 1e6
        total = sum(float(s['t']) for s in out[:-1]
                    if np.isfinite(s['t']) and s['t'] != float('inf'))
        mfe = ((efl - target_efl) ** 2 + 1000.0 * axcl ** 2
               + 0.01 * max(0.0, total - 215.0) ** 2)
        if include_rsce:
            rsce_um = spot_rms(lens, num_rings=3) * 1e3
            mfe += (rsce_um / 300.0) ** 2
        if bounds_list:
            for j, (lo, hi) in enumerate(bounds_list):
                if j < len(x):
                    if x[j] < lo:
                        mfe += 1e4 * (lo - x[j]) ** 2
                    elif x[j] > hi:
                        mfe += 1e4 * (x[j] - hi) ** 2
        return mfe
    except Exception:
        return 1e6


def optimize_local(specs, epd=40.0, fields=DEFAULT_FIELDS,
                   wavelengths=DEFAULT_WAVS, maxiter=25, target_efl=200.0,
                   include_rsce=True, vars_thick=False, vars_curv=False):
    """Nelder-Mead 局部优化（纯 Python 实现，稳定）：
    变量 = 全部空气面（除后焦面）+ 后焦 +（可选）玻璃厚度 +（可选）曲率
    曲率用 R 域（同号 ±50% 软惩罚边界，防面型翻转）
    返回 (new_specs, history_mfe, before_mfe, after_mfe)
    """
    from scipy.optimize import minimize
    i_air = [i for i in _air_indices(specs) if i != len(specs) - 2]
    i_thick = _lens_thick_indices(specs) if vars_thick else []
    i_curv = _lens_curv_indices(specs) if vars_curv else []
    if not i_air:
        return specs, [], 1e6, 1e6
    x0 = [float(specs[i]['t']) for i in i_air]
    x0 += [float(specs[i]['t']) for i in i_thick]
    x0 += [float(specs[i]['R']) for i in i_curv]
    x0.append(float(specs[-2]['t']))
    b_curv = []
    for i in i_curv:
        r0 = float(specs[i]['R'])
        lo, hi = (0.5 * r0, 1.5 * r0) if r0 > 0 else (1.5 * r0, 0.5 * r0)
        b_curv.append((lo, hi))
    bounds = ([(3.0, 60.0)] * len(i_air) + [(0.5, 30.0)] * len(i_thick)
              + b_curv + [(40.0, 80.0)])
    hist = []

    def cb(xk):
        hist.append(float(_mfe_local(xk, specs, epd, fields, wavelengths,
                                     target_efl, include_rsce,
                                     i_air, i_thick, i_curv, bounds)))

    before = _mfe_local(x0, specs, epd, fields, wavelengths, target_efl,
                        include_rsce, i_air, i_thick, i_curv, bounds)
    try:
        res = minimize(_mfe_local, np.asarray(x0, dtype=float),
                       args=(specs, epd, fields, wavelengths, target_efl,
                             include_rsce, i_air, i_thick, i_curv, bounds),
                       method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-3,
                                'fatol': 1e-4})
        xf = res.x
    except Exception:
        return specs, hist, before, before
    after = _mfe_local(xf, specs, epd, fields, wavelengths, target_efl,
                       include_rsce, i_air, i_thick, i_curv, bounds)
    new = [dict(s) for s in specs]
    k = 0
    for i in i_air:
        new[i]['t'] = float(xf[k]); k += 1
    for i in i_thick:
        new[i]['t'] = float(xf[k]); k += 1
    for i in i_curv:
        new[i]['R'] = float(xf[k]); k += 1
    new[-2]['t'] = float(xf[k])
    return new, hist, before, after


# 兼容旧名（app v0.8 及之前调用）
optimize_airs = optimize_local


if __name__ == '__main__':
    from core.lens_io import elite_to_specs
    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    airs = [39.9, 32.3, 27.6, 9.6, 3.0]
    specs = elite_to_specs(gp, airs)
    import time
    for k in KINDS:
        t0 = time.time()
        fig = analysis_fig(specs, kind=k)
        print(f'{k}: {"OK" if fig else "失败"} ({time.time()-t0:.0f}s)')
        if fig:
            plt.close(fig)
    ac = axial_color(specs)
    print('色差:', [(round(w, 4), round(b, 2)) for w, b, _ in ac])
    res = search_glass(1.5163, 64.1)
    print('玻璃搜索:', [(g, round(n, 4), round(v, 1)) for g, n, v, _ in res[:3]])
    print('ANALYSIS_V09: OK')
