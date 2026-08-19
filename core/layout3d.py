# -*- coding: utf-8 -*-
"""core/layout3d.py — 3D Layout（旋转体表面 + 光线）"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

from core.lens_io import build_lens_from_specs


def plot_layout_3d(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                   wavelengths=None, n_rings=10, n_theta=36,
                   max_fields=3, figsize=(11, 4.5)):
    """3D Layout：每个面画旋转体网格（玻璃蓝/空气灰/光阑红/像面绿）+ 光线（YZ 平面）"""
    if wavelengths is None:
        wavelengths = [(0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)]
    lens = build_lens_from_specs(specs, epd=epd, fields=fields,
                                 wavelengths=wavelengths)
    if lens is None:
        return None
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        zs = []
        for s in lens.surfaces:
            z = np.asarray(s.z).flatten()
            zs.append(float(z[0]) if z.size else 0.0)
        # ---- 表面旋转体 ----
        for i, s in enumerate(lens.surfaces):
            R = float(s.geometry.radius)
            semi = 20.0
            try:
                semi = float(s.geometry.semi_aperture)
            except Exception:
                pass
            if not np.isfinite(semi) or semi <= 0:
                semi = 20.0
            r = np.linspace(0, semi, n_rings)
            if np.isfinite(R) and abs(R) > 1e-9:
                with np.errstate(invalid='ignore'):
                    sag = R - np.sqrt(R ** 2 - r ** 2)
                sag = np.nan_to_num(sag, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                sag = np.zeros_like(r)
            th = np.linspace(0, 2 * np.pi, n_theta)
            TH, RR = np.meshgrid(th, r)
            X = RR * np.cos(TH)
            Y = RR * np.sin(TH)
            Z = sag[:, None] + zs[i]
            if s.is_stop:
                color, alpha = '#D62728', 0.9   # 光阑：红环
            elif i == len(lens.surfaces) - 1:
                color, alpha = '#2CA02C', 0.55  # 像面：绿盘
            elif getattr(s.geometry, 'glass', None) is not None:
                color, alpha = '#4C78A8', 0.55  # 玻璃：蓝
            else:
                color, alpha = '#BBBBBB', 0.35  # 空气：灰
            ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                            rstride=1, cstride=1, linewidth=0, shade=True)
        # ---- 光线（主波长，前 max_fields 个视场 × 3 条子午光线）----
        lens.wavelengths.primary_index = 0
        wl0 = float(lens.wavelengths[0].value)
        max_field = float(lens.fields.max_field)
        cmap = plt.cm.turbo
        nf = min(max_fields, len(lens.fields))
        for fi in range(nf):
            hy = float(lens.fields[fi].y) / max_field if max_field > 0 else 0.0
            for pi, py in enumerate((-0.7, 0.0, 0.7)):
                lens.trace_generic(np.array([0.0]), np.array([hy]),
                                   np.array([0.0]), np.array([py]),
                                   wavelength=wl0)
                xs, ys, zs2 = [], [], []
                for s in lens.surfaces:
                    xa, ya, za = (np.asarray(s.x).flatten(),
                                  np.asarray(s.y).flatten(),
                                  np.asarray(s.z).flatten())
                    xs.append(float(xa[0]) if xa.size else 0.0)
                    ys.append(float(ya[0]) if ya.size else 0.0)
                    zs2.append(float(za[0]) if za.size else 0.0)
                ax.plot(xs, ys, zs2, color=cmap(fi / max(1, nf - 1)),
                        lw=1.1, alpha=0.85)
        # ---- 轴与视角 ----
        all_z = np.asarray(zs)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Layout（EPD {epd:.0f} mm | 视场 {max_fields} 个 | 主波长 {wl0:.4f} um）',
                     fontsize=10)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.6))
        except Exception:
            pass
        fig.tight_layout()
        return fig
    except Exception:
        return None
