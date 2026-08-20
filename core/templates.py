# -*- coding: utf-8 -*-
"""core/templates.py — 镜头样板库（经典 / 现代 / 常用构型，v0.22）

两种生成模式：
① 镜片库凑（library）：按槽位（单片/双胶合 + 光焦度符号 + 最小口径）从镜片库随机匹配，
   结构近似经典构型——每次生成不同组合（可反复点击换组合）
② 完全复刻（replica）：按样板示例参数直接生成 specs——
   - 示例参数以 f=100mm 为基准，自动缩放到目标焦距
   - 玻璃按 nd/vd 自动匹配内置 81 种 AGF 玻璃表
   - 参数为"经典结构示意值"（结构 + 玻璃类型正确），生成后可配合 ⚡ 优化微调
"""
import os
import sys
import random

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

from core._library import get_default_library
from core.lens_io import elite_to_specs, library_rows

_lib = get_default_library()

TEMPLATES = {
    'petzval': {
        'label': '匹兹伐 Petzval（1859 · 经典）',
        'desc': '前组三胶合 + 大间隔 + 后组双胶合×2——深空摄影 / 投影经典构型（≈ 默认精英55 结构）',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 55},
                   {'type': 'd', 'sign': '+', 'min_diam': 50},
                   {'type': 'd', 'sign': '+', 'min_diam': 45},
                   {'type': 's', 'sign': '+', 'min_diam': 40},
                   {'type': 'd', 'sign': '-', 'min_diam': 35},
                   {'type': 'd', 'sign': '+', 'min_diam': 30}],
        'airs': [39.9, 32.3, 27.6, 9.6, 3.0],
        'params': [  # f=100mm 基准（精英55 参数 / 2；玻璃 nd/vd 自动匹配内置表）
            (109.4, 5.15, 1.4970, 81.6),
            (-90.15, 2.25, 1.6056, 43.9),
            (-600.0, 19.95, 0.0, 0.0),
            (49.3, 4.1, 1.5168, 64.1),
            (-31.19, 1.7, 1.6166, 36.6),
            (-1798.5, 16.15, 0.0, 0.0),
            (134.25, 1.5, 1.6727, 32.2),
            (-30.13, 1.5, 1.6727, 32.2),
            (-134.25, 13.8, 0.0, 0.0),
            (-27.1, 1.4, 1.5168, 64.1),
            (-90.0, 1.4, 1.5168, 64.1),
            (-60.0, 0.0, 0.0, 0.0),
        ],
    },
    'double_gauss': {
        'label': '双高斯 Double Gauss（1896 · 经典）',
        'desc': '对称结构：正负胶合 | 胶合负正，中间大空气——现代镜头基础构型，像质均衡',
        'groups': [{'type': 's', 'sign': '+', 'min_diam': 40},
                   {'type': 's', 'sign': '-', 'min_diam': 35},
                   {'type': 'd', 'sign': '-', 'min_diam': 32},
                   {'type': 'd', 'sign': '+', 'min_diam': 32},
                   {'type': 's', 'sign': '-', 'min_diam': 30},
                   {'type': 's', 'sign': '+', 'min_diam': 28}],
        'airs': [10, 28, 28, 10, 10],
        'params': [
            (45.0, 5.0, 1.5168, 64.2),
            (-90.0, 10.0, 0.0, 0.0),
            (-40.0, 3.0, 1.6166, 36.6),
            (-80.0, 28.0, 0.0, 0.0),
            (60.0, 4.0, 1.5168, 64.2),
            (-45.0, 3.0, 1.6166, 36.6),
            (-120.0, 12.0, 0.0, 0.0),
            (35.0, 5.0, 1.5168, 64.2),
            (-70.0, 0.0, 0.0, 0.0),
        ],
    },
    'tessar': {
        'label': '天塞 Tessar（1902 · 经典）',
        'desc': '正 + 正 + 负胶合（4 片 3 组）——蔡司经典，小巧高素质',
        'groups': [{'type': 's', 'sign': '+', 'min_diam': 30},
                   {'type': 's', 'sign': '+', 'min_diam': 28},
                   {'type': 'd', 'sign': '-', 'min_diam': 25}],
        'airs': [2, 4],
        'params': [
            (46.0, 6.0, 1.5168, 64.2),
            (-90.0, 2.0, 0.0, 0.0),
            (38.0, 5.0, 1.5168, 64.2),
            (-58.0, 3.0, 0.0, 0.0),
            (-30.0, 3.0, 1.6166, 36.6),
            (50.0, 4.0, 1.5168, 64.2),
            (-70.0, 0.0, 0.0, 0.0),
        ],
    },
    'sonnar': {
        'label': '松纳 Sonnar（1929 · 大口径）',
        'desc': '胶合 + 胶合 + 单片——大口径人像镜经典（f/1.5-2.8 时代）',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 40},
                   {'type': 'd', 'sign': '-', 'min_diam': 35},
                   {'type': 's', 'sign': '+', 'min_diam': 30}],
        'airs': [6, 10],
        'params': [
            (40.0, 6.0, 1.5168, 64.2),
            (-35.0, 3.0, 1.6166, 36.6),
            (-120.0, 6.0, 0.0, 0.0),
            (-50.0, 3.0, 1.6166, 36.6),
            (45.0, 5.0, 1.5168, 64.2),
            (-90.0, 10.0, 0.0, 0.0),
            (30.0, 6.0, 1.5168, 64.2),
            (-100.0, 0.0, 0.0, 0.0),
        ],
    },
    'cooke': {
        'label': '库克三片 Cooke Triplet（1893 · 经典）',
        'desc': '正 + 负 + 正（3 片）——近代镜头的"最小完全解"，衍生无数构型',
        'groups': [{'type': 's', 'sign': '+', 'min_diam': 30},
                   {'type': 's', 'sign': '-', 'min_diam': 28},
                   {'type': 's', 'sign': '+', 'min_diam': 25}],
        'airs': [12, 16],
        'params': [
            (52.0, 6.0, 1.5168, 64.2),
            (-150.0, 12.0, 0.0, 0.0),
            (-52.0, 3.0, 1.6166, 36.6),
            (80.0, 16.0, 0.0, 0.0),
            (40.0, 6.0, 1.5168, 64.2),
            (-120.0, 0.0, 0.0, 0.0),
        ],
    },
    'telephoto': {
        'label': '望远 Telephoto（远摄 · 常用）',
        'desc': '正胶合 + 大间隔 + 负胶合——焦距 > 镜筒长度，远摄/长焦常用',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 35},
                   {'type': 'd', 'sign': '-', 'min_diam': 25}],
        'airs': [40],
        'params': [
            (80.0, 6.0, 1.5168, 64.2),
            (-50.0, 3.0, 1.6166, 36.6),
            (-150.0, 40.0, 0.0, 0.0),
            (-45.0, 3.0, 1.6166, 36.6),
            (-35.0, 5.0, 1.5168, 64.2),
            (-90.0, 0.0, 0.0, 0.0),
        ],
    },
    'retrofocus': {
        'label': '反望远 Retrofocus（现代广角）',
        'desc': '负胶合 + 大间隔 + 正胶合——广角/无反法兰距匹配常用',
        'groups': [{'type': 'd', 'sign': '-', 'min_diam': 30},
                   {'type': 'd', 'sign': '+', 'min_diam': 30}],
        'airs': [25],
        'params': [
            (-60.0, 3.0, 1.6166, 36.6),
            (-30.0, 5.0, 1.5168, 64.2),
            (-40.0, 25.0, 0.0, 0.0),
            (60.0, 6.0, 1.5168, 64.2),
            (-45.0, 3.0, 1.6166, 36.6),
            (-150.0, 0.0, 0.0, 0.0),
        ],
    },
    'achromat': {
        'label': '消色差双胶合 Achromat（1758 · 基础）',
        'desc': '正冕 + 负火石双胶合——最基础的色差校正单元，望远镜/物镜常用',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 25}],
        'airs': [],
        'params': [
            (60.0, 8.0, 1.5168, 64.2),
            (-43.0, 4.0, 1.6166, 36.6),
            (-130.0, 0.0, 0.0, 0.0),
        ],
    },
    'heliar': {
        'label': '海利亚 Heliar（1900 · 经典）',
        'desc': '胶合 + 单片 + 胶合（5 片 3 组）——柔美焦外，人像/风光经典',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 35},
                   {'type': 's', 'sign': '-', 'min_diam': 30},
                   {'type': 'd', 'sign': '+', 'min_diam': 28}],
        'airs': [8, 8],
        'params': [
            (42.0, 6.0, 1.5168, 64.2),
            (-38.0, 3.0, 1.6166, 36.6),
            (-100.0, 8.0, 0.0, 0.0),
            (-35.0, 3.0, 1.6166, 36.6),
            (-120.0, 8.0, 0.0, 0.0),
            (50.0, 5.0, 1.5168, 64.2),
            (-45.0, 3.0, 1.6166, 36.6),
            (-90.0, 0.0, 0.0, 0.0),
        ],
    },
    'dagor': {
        'label': '达戈 Dagor（1892 · 对称）',
        'desc': '对称双胶合对（4 片 2 组）——早期对称构型，像场平坦',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 35},
                   {'type': 'd', 'sign': '+', 'min_diam': 30}],
        'airs': [30],
        'params': [
            (55.0, 5.0, 1.5168, 64.2),
            (-40.0, 3.0, 1.6166, 36.6),
            (-150.0, 30.0, 0.0, 0.0),
            (150.0, 5.0, 1.5168, 64.2),
            (-55.0, 3.0, 1.6166, 36.6),
            (-80.0, 0.0, 0.0, 0.0),
        ],
    },
    # ============ 天文模板（v0.23）============
    'apo_triplet': {
        'label': 'APO 复消色差物镜（三片式 · 深空）',
        'desc': 'ED 双胶合（FK61 超低色散 + 火石）+ 单片冕——复消色差主流构型，深空摄影主力',
        'groups': [{'type': 'd', 'sign': '+', 'min_diam': 30},
                   {'type': 's', 'sign': '+', 'min_diam': 25}],
        'airs': [2],
        'params': [
            (100.0, 6.0, 1.4970, 81.6),
            (-75.0, 2.5, 1.6727, 32.2),
            (-250.0, 2.0, 0.0, 0.0),
            (130.0, 6.0, 1.5168, 64.1),
            (-400.0, 0.0, 0.0, 0.0),
        ],
    },
    'newtonian': {
        'label': '牛顿反射 Newtonian（1668 · 反射）',
        'desc': '抛物面主镜（k=-1，消球差）——经典入门反射望远镜；简化模型（省略 45° 副镜折叠，像面在主镜前）',
        'reflective': True,
        'groups': [],
        'airs': [],
        'params': [
            (-200.0, -100.0, -1.0, 0.0, -1.0),
        ],
    },
    'cassegrain': {
        'label': '卡塞格林 Cassegrain（1672 · 反射）',
        'desc': '抛物面主镜 + 双曲面副镜（k=-9）——紧凑两镜系统（像面在主镜前，等效焦距 ≈0.96×目标）',
        'reflective': True,
        'groups': [],
        'airs': [],
        'params': [
            (-200.0, -80.0, -1.0, 0.0, -1.0),
            (160.0, 16.0, -1.0, 0.0, -9.0),
        ],
    },
    'ritchey_chretien': {
        'label': '里奇-克雷蒂安 RC（1928 · 反射）',
        'desc': '双曲面主镜（k=-1.1）+ 双曲面副镜（k=-5.4）——天文台主流，消球差+消彗差',
        'reflective': True,
        'groups': [],
        'airs': [],
        'params': [
            (-200.0, -80.0, -1.0, 0.0, -1.1),
            (160.0, 16.0, -1.0, 0.0, -5.4),
        ],
    },
    'schmidt_cassegrain': {
        'label': '施密特-卡塞格林 SCT（1940 · 折反射）',
        'desc': 'EVENASPH 非球面校正板 + 球面主副镜——普及型天文望远镜主力（校正板 A4 可调，建议 ⚡ 优化）',
        'reflective': True,
        'groups': [],
        'airs': [],
        'params': [
            (200.0, 3.0, 1.5168, 64.1, 0.0, [5e-7, 0.0, 0.0, 0.0]),
            (-200.0, 15.0, 0.0, 0.0, 0.0, []),
            (-200.0, -80.0, -1.0, 0.0, 0.0),
            (120.0, 8.0, -1.0, 0.0, 0.0),
        ],
    },
    'maksutov': {
        'label': '马克苏托夫-卡塞格林 Maksutov（1941 · 折反射）',
        'desc': '弯月校正镜 + 球面主副镜——全球面可制造，焦外柔美（真实追迹聚焦 rms<0.1mm）',
        'reflective': True,
        'groups': [],
        'airs': [],
        'params': [
            (-120.0, 3.0, 1.5168, 64.1),
            (-130.0, 15.0, 0.0, 0.0),
            (-200.0, -80.0, -1.0, 0.0, 0.0),
            (200.0, 18.0, -1.0, 0.0, 0.0),
        ],
    },
}


def _single_power_sign(L):
    """单片光焦度符号：φ=(n-1)(1/R1-1/R2)；库 r2 为正数绝对曲率 → R2_actual=-r2"""
    nd = float(L['nd'])
    r1 = float(L['r1'])
    r2 = float(L['r2'])
    c1 = 1.0 / r1 if np.isfinite(r1) and r1 != 0 else 0.0
    c2 = 1.0 / r2 if np.isfinite(r2) and r2 != 0 else 0.0
    p = (nd - 1.0) * (c1 + c2)
    if abs(p) < 1e-12:
        return 0
    return 1 if p > 0 else -1


def _power_sign(L):
    """镜片组光焦度符号（双胶合 = 两片合成）"""
    if isinstance(L, tuple):
        s = 0
        for one in L:
            s += _single_power_sign(one)
        return 1 if s > 0 else (-1 if s < 0 else 0)
    return _single_power_sign(L)


def _group_diam(L):
    """镜片组口径（双胶合取小者）"""
    return min(L[0]['diam'], L[1]['diam']) if isinstance(L, tuple) else L['diam']


def _match_group(grp, rng):
    """从镜片库匹配一个组 → (库类型, 行号)；无匹配返回 None"""
    lts = (5, 6) if grp['type'] == 'd' else (1, 2, 3, 4)
    cands = []
    for lt in lts:
        lo, hi, _cnt = library_rows(lt)
        for row in range(lo, hi + 1):
            L = _lib.get_lens(lt, row)
            if L is None:
                continue
            if _power_sign(L) != grp['sign']:
                continue
            if _group_diam(L) < grp['min_diam']:
                continue
            cands.append((lt, row))
    if not cands:  # 宽松回退：只按类型 + 口径 0.8×
        for lt in lts:
            lo, hi, _cnt = library_rows(lt)
            for row in range(lo, hi + 1):
                L = _lib.get_lens(lt, row)
                if L is None:
                    continue
                if _group_diam(L) >= grp['min_diam'] * 0.8:
                    cands.append((lt, row))
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def _match_glass(nd, vd):
    """nd/vd → 内置 AGF 玻璃表最近玻璃名（加权：Δnd/0.01 + Δvd/5）"""
    from core.lens_io import glass_catalog
    cat = glass_catalog()
    best, best_s = 'D-K9', 1e18
    for g, (n, v) in cat.items():
        s = abs(n - nd) / 0.01 + abs(v - vd) / 5.0
        if s < best_s:
            best_s, best = s, g
    return best


def _replica_specs(tpl, f_mm, back_focus):
    """样板示例参数 → specs
    params 元组：(R, t, nd, vd[, conic[, coeffs]])；nd<0 = 反射面（MIRROR）
    最后一行空气 → t=back_focus（折射后焦）；最后一行镜面 → t 保留（到像面距离，反射系统）"""
    scale = f_mm / 100.0
    specs = [
        {'idx': 0, 'R': np.inf, 't': np.inf, 'glass': '',
         'nd': 0.0, 'vd': 0.0, 'semi': np.nan, 'is_stop': False, 'is_image': False},
        {'idx': 1, 'R': np.inf, 't': 0.0, 'glass': '',
         'nd': 0.0, 'vd': 0.0, 'semi': np.nan, 'is_stop': True, 'is_image': False},
    ]
    idx = 2
    params = tpl['params']
    n = len(params)
    last_is_air = n > 0 and params[-1][2] == 0
    for i, p in enumerate(params):
        R, t, nd, vd = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        conic = float(p[4]) if len(p) > 4 else 0.0
        coeffs = [float(c) for c in (p[5] if len(p) > 5 else []) if c is not None]
        if nd == 0:
            tv = back_focus if (last_is_air and i == n - 1) else t * scale
        else:
            tv = t * scale
        Rv = np.inf if not np.isfinite(R) else R * scale
        glass, ndv, vdv = '', 0.0, 0.0
        if nd > 0:
            glass = _match_glass(nd, vd)
            ndv, vdv = nd, vd
        elif nd < 0:
            glass = 'MIRROR'
        specs.append({'idx': idx, 'R': Rv, 't': tv, 'glass': glass,
                      'nd': ndv, 'vd': vdv, 'semi': 15.0 * scale,
                      'is_stop': False, 'is_image': False,
                      'conic': conic, 'coeffs': coeffs})
        idx += 1
    specs.append({'idx': idx, 'R': np.inf, 't': 0.0, 'glass': '',
                  'nd': 0.0, 'vd': 0.0, 'semi': np.nan,
                  'is_stop': False, 'is_image': True})
    return specs


def build_template(key, mode='library', f_mm=200.0, back_focus=55.0, seed=None):
    """生成样板 specs
    mode: 'library' 镜片库凑（每次随机组合）/ 'replica' 完全复刻（示例参数缩放）
    返回 specs 列表或 None（库凑无匹配时）"""
    tpl = TEMPLATES.get(key)
    if tpl is None:
        return None
    if mode != 'library':
        return _replica_specs(tpl, f_mm, back_focus)
    if tpl.get('reflective'):
        return None  # 反射构型：镜片库为折射玻璃，仅支持完全复刻
    rng = random.Random(seed)
    pairs, airs = [], []
    for grp in tpl['groups']:
        m = _match_group(grp, rng)
        if m is None:
            return None
        pairs.append(m)
    for i in range(len(pairs) - 1):
        airs.append(tpl['airs'][i] if i < len(tpl['airs']) else 5.0)
    return elite_to_specs(pairs, airs, back_focus)


if __name__ == '__main__':
    from core.lens_io import build_lens_from_specs
    from optiland.distribution import create_distribution

    def _mirror_rms(specs, epd=30.0):
        """反射系统真实追迹光斑 RMS（mm）——paraxial 对反射不可靠，用光斑验证聚焦"""
        lens = build_lens_from_specs(specs, epd=epd, fields=(0.0,),
                                     wavelengths=[(0.58756, 1.0)])
        if lens is None:
            return None
        dist = create_distribution('hexapolar')
        dist.generate_points(4)
        try:
            r = lens.trace(Hx=0.0, Hy=0.0, wavelength=0, num_rays=37, distribution=dist)
        except Exception:
            return None
        xs = np.asarray(r.x)
        ys = np.asarray(r.y)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if int(ok.sum()) < 5:
            return None
        return float(np.sqrt(np.mean(xs[ok] ** 2 + ys[ok] ** 2)))

    n_ok = 0
    for key, tpl in TEMPLATES.items():
        # 复刻模式：生成 + 追迹
        sp = build_template(key, mode='replica', f_mm=200.0, back_focus=55.0)
        assert sp is not None and len(sp) >= 4, key
        lens = build_lens_from_specs(sp, epd=30.0, fields=(0.0, 2.0),
                                     wavelengths=[(0.58756, 1.0)])
        assert lens is not None, f'{key} replica build 失败'
        if tpl.get('reflective'):
            # 反射构型：仅复刻（库为折射玻璃）；paraxial 不可靠 → 真实追迹光斑验证
            assert build_template(key, mode='library') is None, f'{key} 反射不应库凑'
            rms_m = _mirror_rms(sp)
            assert rms_m is not None, f'{key} 反射追迹失败'
            lim = 8.0 if key == 'schmidt_cassegrain' else 1.0
            assert rms_m < lim, f'{key} 光斑 {rms_m:.3f}mm 未聚焦'
            continue
        efl = float(lens.paraxial.f2())
        assert np.isfinite(efl) and efl > 50.0, f'{key} EFFL={efl}'
        got = False
        for sd in (1, 2, 3):
            sp2 = build_template(key, mode='library', f_mm=200.0, back_focus=55.0, seed=sd)
            if sp2 is not None:
                got = True
                break
        if got:
            n_ok += 1
    assert n_ok >= 8, f'库凑成功 {n_ok}/10 < 8'
    assert _match_glass(1.5168, 64.1) in ('BK7', 'D-K9', 'H-K9L'), _match_glass(1.5168, 64.1)
    # 天文模板数值抽查（物理合理性）
    def _efl(key, f):
        sp_ = build_template(key, mode='replica', f_mm=f, back_focus=55.0)
        assert sp_ is not None, key
        return float(build_lens_from_specs(sp_, epd=30.0, fields=(0.0,),
                                           wavelengths=[(0.58756, 1.0)]).paraxial.f2())
    ea = _efl('apo_triplet', 100.0)
    assert 60 < ea < 150, ea
    # 反射模板：真实追迹光斑 + 焦距（|主镜后距| + 副镜后距）
    for key, tag, lim in (('newtonian', '牛', 1.0), ('cassegrain', '卡', 1.0),
                          ('ritchey_chretien', 'RC', 1.0),
                          ('schmidt_cassegrain', 'SCT', 8.0),
                          ('maksutov', '马', 1.0)):
        sp_ = build_template(key, mode='replica', f_mm=200.0, back_focus=55.0)
        rms_m = _mirror_rms(sp_)
        assert rms_m is not None and rms_m < lim, (key, rms_m)
        mir = [i for i, s in enumerate(sp_) if s['glass'] == 'MIRROR']
        assert mir, (key, '无反射面')
        fe = abs(float(sp_[mir[0]]['t'])) + (float(sp_[mir[1]]['t']) if len(mir) > 1 else 0.0)
        assert 150 < fe < 260, (key, fe)
    sp_cas = build_template('cassegrain', mode='replica', f_mm=200.0, back_focus=55.0)
    assert sp_cas[2]['glass'] == 'MIRROR' and sp_cas[2]['conic'] == -1.0, 'CAS 主镜'
    assert sp_cas[3]['conic'] == -9.0, 'CAS 副镜 k'
    sp_sct = build_template('schmidt_cassegrain', mode='replica', f_mm=200.0, back_focus=55.0)
    assert any(s['glass'] == 'MIRROR' for s in sp_sct), 'SCT 缺反射面'
    assert sp_sct[2]['coeffs'], 'SCT 校正板缺非球面系数'
    print(f'TEMPLATES: OK（{len(TEMPLATES)} 样板 × 复刻/库凑（{n_ok}/10）'
          f' 天文抽查 APO {ea:.0f} + 5 反射聚焦全过）')
