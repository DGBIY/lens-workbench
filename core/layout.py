# -*- coding: utf-8 -*-
"""core/layout.py — 2D 布局图（Zemax 风格镜头结构侧视图 + 近轴光线追踪）

用法:
    fig = render_layout(specs, epd=40.0, fields=(0.0, 0.7, 1.0))
    fig.savefig('layout.png')

specs 格式与 merit/lens_io 一致（elite_to_specs 输出）：
    {idx, R, t, glass, nd, vd, semi, is_stop, is_image}
近轴光线传播：逐面 (y, u) 折射，n'u' = nu + y(n - n')/r（标准近轴公式）
"""
import numpy as np
import matplotlib.pyplot as plt
import os

_LENS_COLOR = '#7fb3d5'   # 镜片填充
_LENS_EDGE = '#2e86c1'    # 镜片描边
_AXIS = '#888888'
_FIELD_COLORS = ('#e74c3c', '#f39c12', '#9b59b6', '#16a085', '#c0392b')


def setup_cjk_font():
    """注册系统中文字体（SimHei / 微软雅黑），避免 CJK 方块"""
    import matplotlib
    from matplotlib import font_manager
    for cand in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyh.ttf',
                 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc'):
        if os.path.exists(cand):
            try:
                font_manager.fontManager.addfont(cand)
                name = font_manager.FontProperties(fname=cand).get_name()
                matplotlib.rcParams['font.sans-serif'] = [name] + list(matplotlib.rcParams['font.sans-serif'])
                matplotlib.rcParams['axes.unicode_minus'] = False
                return name
            except Exception:
                continue
    return None


# 模块导入时尝试注册（幂等）
try:
    _cjk = setup_cjk_font()
except Exception:
    _cjk = None


def _surface_z(specs):
    """每面 z 坐标：specs[i]['t'] = 面 i 到下个面的距离（物面 t=inf 跳过）"""
    zs = [0.0]
    for i, s in enumerate(specs[:-1]):
        t = float(s['t'])
        if not np.isfinite(t) or t < 0:
            t = 0.0
        zs.append(zs[-1] + t)
    return np.asarray(zs)


def _arc_xy(zc, rc, semi, n=50):
    """球面弧线：顶点 (zc, 0)，曲率半径 rc（带符号，正=球心右侧）
    面形 z(y) = zc + rc - sign(rc) * sqrt(rc² - y²)
    """
    semi = max(float(semi), 0.01)
    rc = float(rc)
    if not np.isfinite(rc) or abs(rc) > 1e6 or abs(rc) < 1e-9:
        return np.array([zc, zc]), np.array([-semi, semi])
    th = np.linspace(-1.0, 1.0, n)
    y = th * semi
    sgn = 1.0 if rc > 0 else -1.0
    z = zc + rc - sgn * np.sqrt(np.maximum(rc * rc - y * y, 0.0))
    return z, y


def render_layout(specs, epd=40.0, fields=(0.0, 0.7, 1.0),
                  wavelengths=None, efl=None, z_pad=0.25, ax=None):
    """镜头 2D 侧视图：镜片轮廓 + 每视场（主光线/上下边缘）近轴光线 + 像面
    fields: 半视场角（度）；wavelengths: [(λ, weight)]，仅用于标注（示意用 d 线）
    efl: 有效焦距标注（None 时用后焦面距离代替）
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.6), facecolor='#12121c')
    else:
        fig = ax.figure

    zs = _surface_z(specs)
    z_max = float(zs[-1])
    z_img = z_max

    # ---- 镜片轮廓：玻璃区间着色 + 面弧线描边 ----
    n = len(specs)
    for i, s in enumerate(specs):
        if i >= n - 1:
            break
        semi = float(s['semi']) if s.get('semi') is not None and np.isfinite(s.get('semi', 0.0)) else 0.0
        if semi <= 0:
            continue
        zc = float(zs[i])
        zc_next = float(zs[i + 1])
        rc = float(s['R'])
        z, y = _arc_xy(zc, rc, semi)
        ax.plot(z, y, color=_LENS_EDGE, lw=1.4, zorder=3)
        ax.plot(z, -y, color=_LENS_EDGE, lw=1.4, zorder=3)
        # 玻璃区间着色：面 i 到下个面（平面四边形填充示意）
        if s.get('glass'):
            ax.fill_between([zc, zc_next], -semi, semi, color=_LENS_COLOR,
                            alpha=0.35, lw=0, zorder=1)

    # ---- 近轴光线追踪 ----
    def _glass_nd(i):
        """面 i 之后介质的折射率（面 i 是玻璃前表面）"""
        if i < n and specs[i].get('glass') and specs[i].get('nd'):
            return float(specs[i]['nd'])
        return 1.0

    for fi, fdeg in enumerate(fields):
        u = np.tan(np.radians(float(fdeg)))
        color = _FIELD_COLORS[fi % len(_FIELD_COLORS)]
        for h in (0.0, epd / 2.0, -epd / 2.0):
            y = h
            uz = u
            pts = [(0.0, h + u * (-zs[1]))]   # 物方到面 1 的直段
            for i in range(1, n - 1):
                zc = float(zs[i])
                y = y + uz * (zc - pts[-1][0])
                # 折射：n'u' = nu + y(n - n')/r
                r = float(specs[i]['R'])
                n1 = _glass_nd(i - 1) if i - 1 >= 0 else 1.0
                n2 = _glass_nd(i)
                if np.isfinite(r) and abs(r) > 1e-9 and abs(r) < 1e6:
                    uz = (n1 * uz + y * (n1 - n2) / r) / n2
                else:
                    uz = n1 * uz / n2
                pts.append((zc, y))
            # 最后一小段到像面
            y_end = y + uz * (z_img - pts[-1][0])
            pts.append((z_img, y_end))
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, color=color, lw=0.9, alpha=0.75, zorder=2)

    # ---- 像面 ----
    ax.axvline(z_img, color='#c0392b', lw=2.0, ls='--', zorder=4)
    ax.text(z_img, -0.02, '像面', color='#c0392b', fontsize=9,
            ha='center', va='top', transform=ax.get_yaxis_transform())

    # ---- 装饰 ----
    z_lo = 0.0
    z_hi = z_img * (1 + z_pad)
    ax.set_xlim(z_lo, z_hi)
    y_lim = max(float(epd) / 2.0 * 1.15, max((float(s.get('semi') or 0) for s in specs), default=10) * 1.15)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_aspect('equal')
    ax.set_xlabel('光轴 Z（mm）', color='#bbb')
    ax.set_ylabel('Y（mm）', color='#bbb')
    efl = float(efl) if efl else (specs[-2]['t'] if n >= 2 else 0.0)
    fno = efl / epd if epd > 0 else 0.0
    ax.set_title(f'2D 布局图 · F/{fno:.2f} · EFL={efl:.1f}mm · EPD={epd:.1f}mm'
                 f' · 视场 {", ".join(str(f) for f in fields)}°',
                 color='white', fontsize=12)
    ax.set_facecolor('#0d0d16')
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.tick_params(colors='#999999')
    # 图例：视场
    handles = [plt.Line2D([0], [0], color=_FIELD_COLORS[fi % len(_FIELD_COLORS)], lw=2,
                          label=f'{f}°') for fi, f in enumerate(fields)]
    ax.legend(handles=handles, loc='upper right', framealpha=0.3, fontsize=8,
              labelcolor='#cccccc')
    fig.tight_layout()
    return fig
