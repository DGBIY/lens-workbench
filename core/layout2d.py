# -*- coding: utf-8 -*-
"""core/layout2d.py — Zemax 风格 2D Layout（v0.7：多视场 × 多波长真实光线）
- 镜片 sag 轮廓 + 前后表面浅色填充（玻璃体）
- 每视场 × 每波长 N 条子午真实光线（LineY 光瞳分布：py=-1..+1，px=0）
- 面编号标注 + 光阑标记（STOP）+ 每波长焦点线（轴向色差可视化）+ 像面
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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from core.lens_io import build_lens_from_specs
from core._bridge import _bfl_from_paraxial
from optiland.distribution import LineYDistribution

# F 蓝 / d 绿 / C 红（Zemax 默认配色近似）
_WL_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
_N_RAYS = 5   # 每视场每波长光线数（子午面 py=-1..+1 均匀）


def _sag(R, rho):
    """球面矢高（顶点 0，正=边缘靠右；平面返回 0）"""
    R = float(R)
    if not np.isfinite(R) or abs(R) < 1e-9:
        return np.zeros_like(np.asarray(rho, dtype=float))
    rho = np.asarray(rho, dtype=float)
    if abs(R) < rho.max():
        rho = np.clip(rho, 0, abs(R) * 0.999)
    return R - np.sign(R) * np.sqrt(R * R - rho * rho)


def _z_positions(specs):
    """每面的 z 坐标（物面 inf 跳过）"""
    zs = [0.0]
    z = 0.0
    for s in specs[:-1]:
        t = s['t']
        if np.isfinite(t) and t != float('inf'):
            z += t
        zs.append(z)
    return zs


def _lens_fill(ax, z1, R1, z2, R2, semi):
    """镜片体填充：前表面 sag 与后表面 sag 围成的多边形（浅蓝半透明）"""
    rho = np.linspace(0, semi, 60)
    s1 = z1 + _sag(R1, rho)
    s2 = z2 + _sag(R2, rho)
    poly = plt.Polygon(np.vstack([
        np.column_stack([s1, rho]),
        np.column_stack([s2[::-1], rho[::-1]]),
        np.column_stack([s2, -rho]),
        np.column_stack([s1[::-1], -rho[::-1]]),
    ]), closed=True, color='#9db8e8', alpha=0.30, lw=0)
    ax.add_patch(poly)


def plot_layout(specs, epd=40.0, fields=None, wavelengths=None, figsize=(13, 5)):
    """2D Layout（Zemax 风格）：
    - 镜片 sag 轮廓 + 玻璃体浅色填充
    - 每视场 × 每波长 N 条子午真实光线（上/下边缘 + 主光线 + 中间采样）
    - 面编号 + 光阑标记 + 每波长焦点线 + 像面
    wavelengths: [(λ, weight), ...]（默认 F/d/C）；fields: [(y, weight), ...] 或 [y, ...]
    """
    fig, ax = plt.subplots(figsize=figsize)
    zs = _z_positions(specs)
    z_total = zs[-1] if zs else 0.0
    wl_list = list(wavelengths) if wavelengths else [
        (0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)]
    field_list = [f if isinstance(f, tuple) else (f, 1.0) for f in (fields or [(0.0, 1.0)])]

    # ---- 参考高度 ----
    ymax_all = [float(epd) / 2.0]
    for s in specs[1:-1]:
        semi = s['semi']
        if np.isfinite(semi) and semi > 0:
            ymax_all.append(float(semi))
    ymax_ref = max(ymax_all) if ymax_all else float(epd) / 2.0

    # ---- 表面行（z, R, semi, t, glass）----
    rows = []
    for s in specs[1:-1]:
        z = zs[s['idx']] if s['idx'] < len(zs) else zs[-1]
        rows.append((z, s['R'], s['semi'], s['t'], s['glass']))

    # ---- 玻璃体填充（玻璃面行 + 下一行后表面围成）----
    for k, (z1, R1, semi1, t1, glass1) in enumerate(rows):
        if not glass1:
            continue
        semi = float(semi1) if np.isfinite(semi1) and semi1 > 0 else ymax_ref * 1.05 + 2.0
        z2 = z1 + (float(t1) if np.isfinite(t1) else 0.0)
        R2 = rows[k + 1][1] if k + 1 < len(rows) else float('inf')
        _lens_fill(ax, z1, R1, z2, R2, semi)

    # ---- 表面轮廓 ----
    for (z, R, semi, t, glass) in rows:
        semi = float(semi) if np.isfinite(semi) and semi > 0 else ymax_ref * 1.05 + 2.0
        if np.isfinite(R) and abs(R) > 1e-9:
            xs = np.linspace(0, semi, 80)
            sag = _sag(R, xs)
            ax.plot(z + sag, xs, 'k-', lw=1.1)
            ax.plot(z + sag, -xs, 'k-', lw=1.1)
        else:
            ax.plot([z, z], [-semi, semi], 'k-', lw=1.1)

    # ---- 真实光线（每视场 × 每波长 N 条子午光线，optiland 真实追迹）----
    try:
        lens = build_lens_from_specs(specs, epd=epd, fields=field_list, wavelengths=wl_list)
        if lens is not None:
            dist = LineYDistribution()
            dist.generate_points(_N_RAYS)
            nf = min(len(field_list), len(lens.fields))
            for fi in range(nf):
                hx = float(lens.fields[fi].x)
                hy = float(lens.fields[fi].y)
                for wi in range(len(wl_list)):
                    c = _WL_COLORS[wi % len(_WL_COLORS)]
                    try:
                        rays = lens.trace(Hx=hx, Hy=hy, wavelength=wi,
                                          num_rays=_N_RAYS, distribution=dist)
                        xs = np.asarray(rays.z)   # (N, nsurf)
                        ys = np.asarray(rays.y)
                        for k in range(_N_RAYS):
                            xk, yk = xs[k], ys[k]
                            ok = np.isfinite(xk) & np.isfinite(yk)
                            if ok.sum() > 2:
                                ax.plot(xk[ok], yk[ok], color=c, lw=1.0, alpha=0.7)
                    except Exception:
                        continue
                    if fi == 0:
                        ax.plot([], [], color=c, lw=1.2,
                                label=f'λ{wi+1} {wl_list[wi][0]:.4f} um')
            # 焦点标记（每波长：轴向色差可视化）
            z_last_opt = z_total
            if len(specs) > 1 and np.isfinite(specs[-2]['t']):
                z_last_opt -= float(specs[-2]['t'])
            for i in range(len(wl_list)):
                lens.wavelengths.primary_index = i
                try:
                    bfl = _bfl_from_paraxial(lens)
                    if np.isfinite(bfl):
                        ax.axvline(z_last_opt + bfl,
                                   color=_WL_COLORS[i % len(_WL_COLORS)],
                                   ls=':', lw=1.0, alpha=0.8)
                except Exception:
                    continue
            lens.wavelengths.primary_index = 0
    except Exception:
        pass

    # ---- 面编号（物面 0 与光阑重叠，跳过）+ 光阑标记 ----
    ylab = -ymax_ref * 1.12
    for i, z in enumerate(zs):
        if i == 0:
            continue
        ax.annotate(str(i), (z, ylab), fontsize=7, color='#666666',
                    ha='center', va='top')
    for s in specs[1:-1]:
        if s.get('is_stop'):
            z = zs[s['idx']] if s['idx'] < len(zs) else zs[-1]
            ax.axvline(z, color='k', lw=2.2, alpha=0.85)
            ax.annotate('STOP', (z, ymax_ref * 1.05), fontsize=8, color='k',
                        ha='center', fontweight='bold')

    # ---- 光轴 / 像面 / 布局 ----
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.axvline(z_total, color='r', lw=1.5, ls='--', label='像面')
    ax.set_xlim(-2, z_total * 1.02 + 2)
    ax.set_ylim(ylab * 1.15, ymax_ref * 1.3)
    ax.set_xlabel('Z (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(f'2D Layout（{len(field_list)} 视场 × {len(wl_list)} 波长 × {_N_RAYS} 条子午光线 + 面编号 + 光阑）')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.6)
    ax.grid(True, ls='--', lw=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    from core.lens_io import elite_to_specs
    _specs = elite_to_specs([(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)],
                            [39.9, 32.3, 27.6, 9.6, 3.0])
    _fig = plot_layout(_specs, epd=40.0,
                       fields=[(0.0, 1.0), (2.0, 1.0), (4.06, 1.0), (5.8, 1.0)],
                       wavelengths=[(0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)])
    _n = len(_fig.axes[0].lines)
    _p = len(_fig.axes[0].patches)
    print(f'线条数: {_n} | 填充: {_p}')
    _fig.savefig(r'C:\Users\Administrator\Documents\DeepChat\l2d_v2.png', dpi=110)
    print('L2D_SELFTEST:', 'OK' if _n > 30 else 'FAIL')
