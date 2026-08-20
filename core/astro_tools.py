# -*- coding: utf-8 -*-
"""core/astro_tools.py — 天文摄影光学工具（深空 DIY）

标准天文公式：像元尺度 / 视场 / 衍射极限 / 星点 FWHM / 焦深 / 极限星等 / 滤镜 / 减焦 / 温漂。
单位：焦距 mm、口径 mm、像元 µm、传感器 mm、波长 nm。
"""
import numpy as np


def pixel_scale(f_mm, pixel_um):
    """像元角尺度（arcsec/pixel）：206.265 × 像元µm / 焦距mm"""
    return 206.265 * pixel_um / f_mm if f_mm > 0 else float('nan')


def fov(sensor_w, sensor_h, f_mm):
    """传感器视场角（度）→ (宽, 高)"""
    if f_mm <= 0:
        return float('nan'), float('nan')
    return (2 * np.degrees(np.arctan(sensor_w / 2.0 / f_mm)),
            2 * np.degrees(np.arctan(sensor_h / 2.0 / f_mm)))


def diffraction_limit(D_mm, lam_nm=550.0):
    """衍射极限角分辨率（arcsec）：1.22λ/D（=138.4/D）"""
    return 206265.0 * 1.22 * lam_nm * 1e-6 / D_mm if D_mm > 0 else float('nan')


def airy_diameter(fno, lam_nm=550.0):
    """艾里斑直径（µm）：2.44λF#"""
    return 2.44 * lam_nm * 1e-3 * fno


def fwhm_arcsec(rms_um, f_mm):
    """星点 FWHM（arcsec）≈ 2.355 × RMS（高斯近似）"""
    return 2.355 * rms_um * 0.206265 / f_mm if f_mm > 0 else float('nan')


def sampling_ratio(fwhm_as, ps):
    """采样率：FWHM / 像元尺度（Nyquist 合理区间 1.5-4）"""
    return fwhm_as / ps if ps > 0 else float('nan')


def critical_focus_depth(fno, lam_nm=550.0):
    """临界焦深 ±2λF#²（mm）——对焦精度需求"""
    return 2.0 * lam_nm * 1e-6 * fno * fno


def limit_magnitude(D_mm):
    """极限星等（近似：人眼 7mm 基准 6.5 等）"""
    return 6.5 + 5.0 * np.log10(D_mm / 7.0) if D_mm > 0 else float('nan')


def light_gain(D_mm):
    """集光力（×人眼 7mm）"""
    return (D_mm / 7.0) ** 2 if D_mm > 0 else float('nan')


def plate_shift(t_mm, n):
    """平行平板（滤镜/窗口）焦点位移（mm）：t(1-1/n)"""
    return t_mm * (1.0 - 1.0 / n) if n > 1 else 0.0


def reducer_effect(f_mm, fno, k):
    """减焦(k<1)/增倍(k>1) → (合成焦距, 合成F#, FOV倍率)"""
    return f_mm * k, fno * k, 1.0 / k


def thermal_shift(L_mm, dT, alpha=23e-6):
    """铝镜筒热漂移（mm）：α·L·ΔT"""
    return alpha * L_mm * dT


# ============================================================
# v0.19：月面/行星像比例 + 曝光/极限星等 SNR 模型 + 减焦镜设计
# ============================================================

def image_scale(f_mm, angular_deg):
    """天体在焦平面的像直径（mm）：f × tan(角直径)
    angular_deg: 天体角直径（度）——月亮 31′≈0.5167、太阳 32′≈0.5333、
    行星按角秒转度（40″≈0.01111）"""
    if f_mm <= 0:
        return float('nan')
    return f_mm * np.tan(np.radians(angular_deg))


def pixels_on_body(diam_mm, pixel_um):
    """天体像直径占像素数：像直径 mm → px"""
    if pixel_um <= 0:
        return float('nan')
    return diam_mm * 1000.0 / pixel_um


# ---- 曝光 / 极限星等：550nm 宽带光子统计模型（深空天光主导近似）----
# F0: m=0 星 550nm 光子通量 ≈ 1e8 photons/m²/s/nm
#   （Vega 光谱辐照度 3.6e-11 W/m²/nm ÷ 光子能量 hc/λ=3.61e-19 J ≈ 1e8）
# 简化：单色近似 + 固定带宽 bw_nm；点源全部光子进一个像元（未分摊）

_F0 = 1.0e8


def star_photons(D_mm, mag, t_sec, qe, bw_nm=150.0):
    """目标星点信号电子数：F0·10^(-0.4m)·A·Δλ·t·QE"""
    if D_mm <= 0 or t_sec <= 0:
        return 0.0
    area = np.pi * (D_mm * 1e-3) ** 2 / 4.0
    return _F0 * 10.0 ** (-0.4 * mag) * area * bw_nm * t_sec * qe


def sky_photons_per_px(D_mm, sky_mag_as2, pixel_scale_as, t_sec, qe, bw_nm=150.0):
    """天光背景电子/像元：天光亮度(mag/arcsec²) × 像元角面积 × 口径 × t × QE"""
    if D_mm <= 0 or t_sec <= 0:
        return 0.0
    area = np.pi * (D_mm * 1e-3) ** 2 / 4.0
    return _F0 * 10.0 ** (-0.4 * sky_mag_as2) * area * bw_nm * (pixel_scale_as ** 2) * t_sec * qe


def snr(D_mm, mag, t_sec, qe, sky_mag_as2, pixel_scale_as,
        read_noise=3.0, dark=0.05, bw_nm=150.0):
    """点源信噪比：SNR = S / sqrt(S + B + R² + D·t)"""
    S = star_photons(D_mm, mag, t_sec, qe, bw_nm)
    B = sky_photons_per_px(D_mm, sky_mag_as2, pixel_scale_as, t_sec, qe, bw_nm)
    return S / np.sqrt(S + B + read_noise ** 2 + dark * t_sec)


def limiting_magnitude(D_mm, t_sec, qe, sky_mag_as2, pixel_scale_as,
                       read_noise=3.0, dark=0.05, target_snr=10.0, bw_nm=150.0):
    """单张极限星等（二分法扫 SNR=target）"""
    lo, hi = 0.0, 26.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        s = snr(D_mm, mid, t_sec, qe, sky_mag_as2, pixel_scale_as,
                read_noise, dark, bw_nm)
        if s > target_snr:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def stacked_limit(m_single, n_frames):
    """N 张叠加极限星等提升：2.5·log10(√N)（天光噪声主导，信号恒定）"""
    if n_frames <= 0:
        return m_single
    return m_single + 2.5 * np.log10(np.sqrt(n_frames))


# ---- 平场镜（减焦镜）薄透镜组合设计 ----

def combined_focal(f1, f2, d):
    """两薄透镜组合焦距：1/f = 1/f1 + 1/f2 - d/(f1·f2)
    f 正=会聚；d=透镜间距（主镜→减焦镜）"""
    if f1 == 0 or f2 == 0:
        return float('nan')
    inv = 1.0 / f1 + 1.0 / f2 - d / (f1 * f2)
    if abs(inv) < 1e-12:
        return float('inf')
    return 1.0 / inv


def reducer_design(f_main, f_reducer, d):
    """减焦率：组合焦距 / 主镜焦距（<1 = 减焦）"""
    if f_main <= 0:
        return float('nan')
    f_comb = combined_focal(f_main, f_reducer, d)
    return f_comb / f_main


def reducer_focal_for(f_main, ratio, d):
    """反推：目标减焦率 → 所需减焦镜焦距 f₂
    由 1/(ratio·f1) = 1/f1 + 1/f2 - d/(f1·f2) 解得：
    f2 = (1 - d/f1)·f1 / (1/ratio - 1)"""
    if f_main <= 0 or abs(ratio - 1.0) < 1e-6:
        return float('inf')
    return (1.0 - d / f_main) * f_main / (1.0 / ratio - 1.0)


if __name__ == '__main__':
    ps = pixel_scale(200, 3.76)
    fw, fh = fov(36, 24, 200)
    dl = diffraction_limit(40)
    ad = airy_diameter(5)
    fw_as = fwhm_arcsec(50, 200)
    sr = sampling_ratio(fw_as, ps)
    fd = critical_focus_depth(5)
    lm = limit_magnitude(40)
    lg = light_gain(40)
    psh = plate_shift(2, 1.52)
    rf, rn, rk = reducer_effect(200, 5, 0.8)
    th = thermal_shift(200, 10)
    assert abs(ps - 3.878) < 0.01, ps
    assert abs(fw - 10.29) < 0.02, fw
    assert abs(fh - 6.87) < 0.02, fh
    assert abs(dl - 3.46) < 0.01, dl
    assert abs(ad - 6.71) < 0.01, ad
    assert abs(fw_as - 0.1214) < 0.002, fw_as
    assert abs(sr - 0.0313) < 0.002, sr
    assert abs(fd - 0.0275) < 0.001, fd
    assert abs(lm - 10.28) < 0.05, lm
    assert abs(lg - 32.65) < 0.1, lg
    assert abs(psh - 0.684) < 0.01, psh
    assert abs(rf - 160) < 1e-6 and abs(rn - 4) < 1e-6 and abs(rk - 1.25) < 1e-6
    assert abs(th - 0.046) < 0.001, th
    # v0.19 新增
    ims = image_scale(200, 31 / 60)
    assert abs(ims - 1.804) < 0.01, ims
    assert abs(pixels_on_body(ims, 3.76) - 479.8) < 1
    sp = star_photons(100, 15, 120, 0.6)
    assert 8.0e3 < sp < 9.0e3, sp
    sk = sky_photons_per_px(100, 21.5, 3.878, 120, 0.6)
    assert 280 < sk < 360, sk
    s = snr(100, 15, 120, 0.6, 21.5, 3.878, 3.0, 0.05)
    assert 80 < s < 100, s
    ml = limiting_magnitude(100, 120, 0.6, 21.5, 3.878, 3.0, 0.05)
    assert 18.0 < ml < 20.0, ml
    assert abs(stacked_limit(21.0, 10) - 22.25) < 0.01
    cf = combined_focal(200, 300, 40)
    assert abs(cf - 130.4) < 0.5, cf
    assert abs(reducer_design(200, 300, 40) - 0.652) < 0.01
    rf2 = reducer_focal_for(200, 0.8, 40)
    assert abs(rf2 - 640) < 5, rf2
    print('ASTRO_TOOLS: OK（22 项数值自检全过）')
