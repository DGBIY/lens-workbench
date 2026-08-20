# -*- coding: utf-8 -*-
"""core/starfield.py — 模拟星场渲染器（像差可视化，v0.21）

把当前镜头当成望远镜，用 optiland 真实光线追迹渲染"模拟星空"：
- 每颗星的形状 = 该视场的真实点列图（三波长 F/d/C 叠加）
  → 中心圆点(球差) / 边缘彗星尾(彗差) / 四角椭圆(像散) / 彩边(横向色差)
- 离焦对比：像面轴向平移（光线外推），模拟不同调焦位置的星场
- 定量标注：每颗星 RMS 光斑半径（µm）
- 曝光模拟照片：像素级噪声叠加（单张 vs N 张叠加）
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

from optiland.distribution import create_distribution
from core.lens_io import build_lens_from_specs
from core.astro_tools import (star_photons, sky_photons_per_px,
                              limiting_magnitude, stacked_limit, pixel_scale)

_WL = [(0.48613, '#3b6fd4'), (0.58756, '#2e9e4f'), (0.65627, '#d43b3b')]
_D_IDX = 1  # d 波长 = 共同中心（主波长）
_RINGS = 6  # hexapolar 环数（1 + 6×Σ(1..6) = 127 光线）

_BODY_DEG = {  # 天体角直径（度），供外部 UI 使用
    '月亮（31′）': 31.0 / 60.0,
    '太阳（32′）': 32.0 / 60.0,
    '木星（40″）': 40.0 / 3600.0,
    '土星（18″）': 18.0 / 3600.0,
    '火星（17″）': 17.0 / 3600.0,
    '金星（12″）': 12.0 / 3600.0,
}
_SKY_MAG = {'暗空 21.5': 21.5, '乡村 20.5': 20.5, '城郊 19.0': 19.0, '城市 17.0': 17.0}


def extrapolate_to_z(rays, z_target):
    """把像面处光线沿传播方向外推到 z_target（mm）
    RealRays.x/y/z = 每光线最新状态（像面）；L/M/N = 方向余弦"""
    xs = np.asarray(rays.x, dtype=float)
    ys = np.asarray(rays.y, dtype=float)
    zs = np.asarray(rays.z, dtype=float)
    Ls = np.asarray(rays.L, dtype=float)
    Ms = np.asarray(rays.M, dtype=float)
    Ns = np.asarray(rays.N, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (z_target - zs) / Ns
    return xs + Ls * t, ys + Ms * t


def _stats_xy(xs, ys):
    """落点统计：质心 + RMS（µm）；失败返回 None"""
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 3:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    rms = float(np.sqrt(np.mean((xs - cx) ** 2 + (ys - cy) ** 2)))
    return {'rms_um': rms * 1000.0, 'cx': cx, 'cy': cy, 'n': int(len(xs))}


def _gen_grid(n_side=5):
    """5×5 视场网格（归一化场坐标 -1..1）"""
    pts = np.linspace(-1.0, 1.0, n_side)
    return [(float(hx), float(hy)) for hy in pts for hx in pts]


def _gen_random(n, seed=42):
    """随机星场：圆内分布 + 亮度（8-17 等）"""
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.random(n)) * 0.95
    th = rng.random(n) * 2 * np.pi
    return [(float(r[i] * np.cos(th[i])), float(r[i] * np.sin(th[i])),
             float(rng.uniform(8.0, 17.0))) for i in range(n)]


def _build(specs, epd, fields, wavelengths):
    if wavelengths is None:
        wavelengths = [(w, 1.0) for w, _ in _WL]
    return build_lens_from_specs(specs, epd=epd, fields=fields or (0.0, 2.0, 4.06, 5.8),
                                 wavelengths=wavelengths)


def render_starfield(specs, epd=40.0, fields=None, wavelengths=None,
                     mode='grid', n_stars=25, scale=12.0, seed=42,
                     defocus_mm=0.0, annotate=False, figsize=(12, 8)):
    """渲染模拟星场 → matplotlib fig
    mode: 'grid' 网格演示 / 'random' 随机星空
    scale: 像差形态放大倍数（落点相对 d 波长质心的偏移 ×scale）
    defocus_mm: 像面轴向偏移（mm，正=远离镜头）
    annotate: 每颗星标注 RMS（µm）
    """
    lens = _build(specs, epd, fields, wavelengths)
    if lens is None:
        return None
    fig, ax = plt.subplots(figsize=figsize, facecolor='#05070d')
    ax.set_facecolor('#05070d')
    stars = _gen_grid(int(round(n_stars ** 0.5))) if mode == 'grid' else _gen_random(n_stars, seed)
    dist = create_distribution('hexapolar')
    dist.generate_points(_RINGS)
    z_img = None
    d_idx = min(_D_IDX, len(lens.wavelengths) - 1) if len(lens.wavelengths) > 1 else 0
    for st in stars:
        hx, hy = st[0], st[1]
        mag = st[2] if len(st) > 2 else 12.0
        center, rms_um = None, None
        try:
            r0 = lens.trace(Hx=hx, Hy=hy, wavelength=d_idx, num_rays=64, distribution=dist)
            if z_img is None:
                z_img = float(np.asarray(r0.z)[0])
            xs, ys = extrapolate_to_z(r0, z_img + defocus_mm)
            st0 = _stats_xy(xs, ys)
            if st0 is not None and st0['n'] > 5:
                center = (st0['cx'], st0['cy'])
                rms_um = st0['rms_um']
        except Exception:
            center = None
        if center is None:
            continue
        for wi, (wlv, col) in enumerate(_WL):
            try:
                r = lens.trace(Hx=hx, Hy=hy, wavelength=wi, num_rays=64, distribution=dist)
                xs, ys = extrapolate_to_z(r, z_img + defocus_mm)
                ok = np.isfinite(xs) & np.isfinite(ys)
                if int(ok.sum()) < 5:
                    continue
                xd = center[0] + (xs[ok] - center[0]) * scale
                yd = center[1] + (ys[ok] - center[1]) * scale
                s = 5.0 if mag < 11 else 3.2 if mag < 14 else 2.0
                al = 0.9 if mag < 11 else 0.65 if mag < 14 else 0.45
                ax.scatter(xd, yd, s=s, c=col, alpha=al, lw=0)
            except Exception:
                continue
        if annotate and rms_um is not None:
            ax.annotate(f'{rms_um:.0f}μm', (center[0], center[1] + 1.0),
                        color='#ffd27f', fontsize=7, ha='center')
    ax.set_aspect('equal')
    ax.set_xlabel('像面 X（mm × 像差放大）', color='#999')
    ax.set_ylabel('像面 Y（mm × 像差放大）', color='#999')
    dz_txt = f' · 离焦 {defocus_mm * 1000:+.0f}μm' if abs(defocus_mm) > 1e-9 else ''
    ax.set_title(f'模拟星场（{("网格演示" if mode == "grid" else "随机星空")}'
                 f' · F/d/C 三波长 · 像差放大 ×{scale:.0f}{dz_txt}）', color='white')
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_color('#333333')
    ax.tick_params(colors='#999999')
    fig.tight_layout()
    return fig


def best_focus_offset(specs, epd=40.0, fields=None, wavelengths=None,
                      z_range=0.4, steps=21, stars=None):
    """扫描像面位置 → 中心+边缘星总 RMS 最小处 = 最佳对焦偏移（mm）
    返回 (best_dz_mm, rms_curve_list)"""
    lens = _build(specs, epd, fields, wavelengths)
    if lens is None:
        return 0.0, []
    if stars is None:
        stars = [(0.0, 0.0), (0.7, 0.0), (0.0, 0.7), (0.7, 0.7)]
    dist = create_distribution('hexapolar')
    dist.generate_points(_RINGS)
    z_img = None
    curve = []
    for dz in np.linspace(-z_range, z_range, steps):
        tot = 0.0
        for hx, hy in stars:
            try:
                r = lens.trace(Hx=hx, Hy=hy, wavelength=_D_IDX, num_rays=40, distribution=dist)
                if z_img is None:
                    z_img = float(np.asarray(r.z)[0])
                xs, ys = extrapolate_to_z(r, z_img + dz)
                st0 = _stats_xy(xs, ys)
                if st0 is not None:
                    tot += st0['rms_um'] ** 2
            except Exception:
                continue
        curve.append((dz, tot ** 0.5))
    best = min(curve, key=lambda c: c[1]) if curve else (0.0, 0.0)
    return best[0], curve


def render_exposure_stars(specs, epd=40.0, fields=None, wavelengths=None,
                          t_sec=120.0, n_stack=10, sky_mag=21.5, qe=0.6,
                          read_noise=3.0, dark=0.05, n_stars=60, seed=42,
                          zoom=1, pixel_um=3.76, figsize=(13, 5)):
    """像素级曝光模拟照片：单张 vs 叠加 N 张
    - 每颗星：真实追迹 RMS → 高斯 PSF；强度 = 光子统计（star_photons）
    - 噪声：天光背景 + 读出 + 暗电流（单张 σ；叠加 σ/√N）
    zoom: 1=整幅(24×16mm) / 2 / 4（中心放大）
    """
    lens = _build(specs, epd, fields, wavelengths)
    if lens is None:
        return None
    sensor_w, sensor_h = 24.0, 16.0
    view_w, view_h = sensor_w / zoom, sensor_h / zoom
    W, H = 480, 320
    px_mm = view_w / W
    rng = np.random.default_rng(seed)
    # 传感器内随机布星（归一化传感器坐标）
    xu = rng.uniform(0.04, 0.96, n_stars)
    yu = rng.uniform(0.04, 0.96, n_stars)
    mags = rng.uniform(8.0, 17.0, n_stars)
    dist = create_distribution('hexapolar')
    dist.generate_points(_RINGS)
    z_img = None
    f_mm = None
    try:
        f_mm = float(lens.paraxial.f2())
    except Exception:
        f_mm = 200.0
    ps_as = pixel_scale(f_mm, pixel_um)
    img1 = np.zeros((H, W))
    img2 = np.zeros((H, W))
    star_mags_vis = []
    for i in range(n_stars):
        x_mm = (xu[i] - 0.5) * sensor_w
        y_mm = (yu[i] - 0.5) * sensor_h
        # 视图坐标（中心放大时裁掉视图外的星）
        px = (x_mm - (sensor_w - view_w) / 2.0) / px_mm + W / 2.0
        py = (y_mm - (sensor_h - view_h) / 2.0) / px_mm + H / 2.0
        if not (8 <= px < W - 8 and 8 <= py < H - 8):
            continue
        # 视场归一化坐标（±1 → 传感器边缘）
        hx = (x_mm / (sensor_w / 2.0)) * 0.92
        hy = (y_mm / (sensor_h / 2.0)) * 0.92
        rms_um = 10.0
        try:
            r = lens.trace(Hx=float(hx), Hy=float(hy), wavelength=_D_IDX,
                           num_rays=40, distribution=dist)
            if z_img is None:
                z_img = float(np.asarray(r.z)[0])
            xs, ys = extrapolate_to_z(r, z_img)
            st0 = _stats_xy(xs, ys)
            if st0 is not None:
                rms_um = min(max(st0['rms_um'], 3.0), 300.0)
        except Exception:
            pass
        sigma_px = max(rms_um * 0.001 / px_mm, 0.4)
        # 信号电子：峰值强度 = S_e / (2πσ²)
        s_e = star_photons(epd, mags[i], t_sec, qe)
        i_peak = s_e / (2.0 * np.pi * sigma_px ** 2)
        # 高斯写入（3σ 窗口）
        rad = int(max(2, 3 * sigma_px))
        yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
        g = i_peak * np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_px ** 2))
        y0, x0 = int(round(py)), int(round(px))
        ysl = slice(max(0, y0 - rad), min(H, y0 + rad + 1))
        xsl = slice(max(0, x0 - rad), min(W, x0 + rad + 1))
        gsl = g[max(0, rad - y0):rad + 1 + min(0, H - 1 - y0),
                max(0, rad - x0):rad + 1 + min(0, W - 1 - x0)]
        if gsl.shape == (ysl.stop - ysl.start, xsl.stop - xsl.start):
            img1[ysl, xsl] += gsl
            img2[ysl, xsl] += gsl
            star_mags_vis.append(mags[i])
    # 噪声（电子/显示px）：天光按像元电子数换算到显示像素面积
    b_e = sky_photons_per_px(epd, sky_mag, ps_as, t_sec, qe)
    b_disp = b_e * (px_mm * 1000.0 / pixel_um) ** 2  # 天光电子 per display px
    sigma_n = float(np.sqrt(b_disp + read_noise ** 2 + dark * t_sec))
    img1 = img1 + rng.normal(0.0, sigma_n, img1.shape)
    img2 = img2 + rng.normal(0.0, sigma_n / np.sqrt(max(n_stack, 1)), img2.shape)
    m_lim = limiting_magnitude(epd, t_sec, qe, sky_mag, ps_as, read_noise, dark)
    m_st = stacked_limit(m_lim, n_stack)
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='#0a0c12')
    for ax, img, tt in ((axes[0], img1, f'单张 {t_sec:.0f}s · 极限 {m_lim:.1f} 等'),
                        (axes[1], img2, f'叠加 {n_stack} 张 · 极限 {m_st:.1f} 等')):
        ax.set_facecolor('#0a0c12')
        ax.imshow(np.sqrt(np.clip(img, 0, None)), cmap='gray', origin='upper')
        ax.set_title(tt, color='white', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#333333')
    fig.suptitle(f'曝光模拟（口径 {epd:.0f}mm · 天光 {sky_mag:.1f} mag/arcsec^2 · '
                 f'QE {qe:.0f}% · 显示 {"整幅" if zoom == 1 else f"中心 {zoom}×"}）',
                 color='#cccccc', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


if __name__ == '__main__':
    from core.lens_io import elite_to_specs
    # 内置默认结构（精英55 内联，data/samples 可能为空）
    spec = elite_to_specs(
        [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)],
        [39.9, 32.3, 27.6, 9.6, 3.0])
    assert spec is not None, '内置结构构建失败'
    f1 = render_starfield(spec, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                          mode='grid', n_stars=25, scale=12.0)
    assert f1 is not None and len(f1.axes[0].collections) > 0
    f2 = render_starfield(spec, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                          mode='random', n_stars=40, scale=18.0, seed=7,
                          defocus_mm=-0.1, annotate=True)
    assert f2 is not None and len(f2.axes[0].collections) > 0
    bz, curve = best_focus_offset(spec, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8))
    assert abs(bz) <= 0.4 and len(curve) == 21, bz
    f3 = render_exposure_stars(spec, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                               t_sec=120.0, n_stack=10, n_stars=50, seed=3)
    assert f3 is not None and len(f3.axes) == 2
    print(f'STARFIELD: OK（网格/随机/离焦对比/最佳焦点 {bz * 1000:+.0f}um/曝光模拟 自检全过）')
