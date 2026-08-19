# -*- coding: utf-8 -*-
"""core/lens_io.py — 表面规格(SurfaceSpec)模型：Zemax 壳子核心
================================================================
从"库引用(gene_pairs)"升级为"自由表面规格"：
  18 表面 × {idx, R, t, glass, nd, vd, semi, is_stop, is_image}
LDE 表格可编辑任意列 → df_to_specs → build_lens_from_specs → 实时重绘。
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import json
import numpy as np
import pandas as pd

from core._library import get_default_library
from core._bridge import build_optiland, _agf_params
from optiland.optic import Optic
from optiland.materials import AbbeMaterial

_lib = get_default_library()

# ============================================================
# 1. 精英 → 表面规格
# ============================================================
def _parse_elites(fpath):
    """解析精英文件（自包含版）：
    块格式：'精英N: EFFL=... MFE=...' / '  镜片: [(5, 259), ...]' / "  空气: ['35.75', ...]"
    替代完整项目的 decode_design.parse_elites（分享版无外部依赖）
    """
    import re as _re
    out = []
    if not os.path.exists(fpath):
        return out
    with open(fpath, encoding='utf-8') as f:
        lines = f.read().splitlines()
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip()
        if ln.startswith('精英'):
            pairs, airs = [], []
            j = i + 1
            while j < n:
                l2 = lines[j].strip()
                if l2.startswith('镜片'):
                    mm = _re.findall(r'\((\d+),\s*(\d+)\)', l2)
                    pairs = [(int(a), int(b)) for a, b in mm]
                elif l2.startswith('空气'):
                    mm = _re.findall(r'\d+\.?\d*', l2.split(':', 1)[1])
                    airs = [float(x) for x in mm]
                elif l2.startswith('精英'):
                    break
                j += 1
            if len(pairs) >= 6:
                gene = ([p[0] for p in pairs[:6]] + [p[1] for p in pairs[:6]] + airs[:5])
                out.append({'gene_pairs': pairs[:6], 'gene': gene})
            i = j
        else:
            i += 1
    return out


def load_elite(fpath, idx=1):
    """从精英文件载入结构 → (gene_pairs, airs)；失败返回 (None, None)"""
    import config as _cfg
    if not os.path.isabs(fpath):
        fpath = os.path.join(_cfg.SAMPLES_DIR, fpath)
    elites = _parse_elites(fpath)
    if not elites:
        return None, None
    idx = min(max(int(idx), 1), len(elites))
    e = elites[idx - 1]
    gene_pairs = e['gene_pairs']
    airs = [float(a) for a in e['gene'][12:17]]
    return gene_pairs, airs


def elite_to_specs(gene_pairs, airs, back_focus=55.0):
    """库引用结构 → 18 表面规格（与 build_optiland 槽位完全一致）
    玻璃 nd/vd 用 AGF 实测（_agf_params，与 optiland 桥接一致）
    """
    specs = [
        {'idx': 0, 'R': np.inf, 't': np.inf, 'glass': '',
         'nd': 0.0, 'vd': 0.0, 'semi': np.nan, 'is_stop': False, 'is_image': False},
        {'idx': 1, 'R': np.inf, 't': 0.0, 'glass': '',
         'nd': 0.0, 'vd': 0.0, 'semi': np.nan, 'is_stop': True, 'is_image': False},
    ]
    idx = 2
    for i, (lt, row) in enumerate(gene_pairs):
        L = _lib.get_lens(lt, row)
        if L is None:
            return None
        air_gap = airs[i] if i < 5 else back_focus
        if isinstance(L, tuple):
            la, lb = L
            nd_a, vd_a = _agf_params(la['glass'], la['nd'], la['vd'])
            nd_b, vd_b = _agf_params(lb['glass'], lb['nd'], lb['vd'])
            d_half = min(la['diam'], lb['diam']) / 2.0
            specs += [
                {'idx': idx, 'R': float(la['r1']), 't': float(la['thick']),
                 'glass': la['glass'], 'nd': float(nd_a), 'vd': float(vd_a),
                 'semi': d_half, 'is_stop': False, 'is_image': False},
                {'idx': idx + 1, 'R': float(-la['r2']), 't': float(lb['thick']),
                 'glass': lb['glass'], 'nd': float(nd_b), 'vd': float(vd_b),
                 'semi': d_half, 'is_stop': False, 'is_image': False},
                {'idx': idx + 2, 'R': float(-lb['r2']), 't': float(air_gap),
                 'glass': '', 'nd': 0.0, 'vd': 0.0,
                 'semi': d_half, 'is_stop': False, 'is_image': False},
            ]
            idx += 3
        else:
            nd_g, vd_g = _agf_params(L['glass'], L['nd'], L['vd'])
            d_half = L['diam'] / 2.0
            specs += [
                {'idx': idx, 'R': float(L['r1']), 't': float(L['thick']),
                 'glass': L['glass'], 'nd': float(nd_g), 'vd': float(vd_g),
                 'semi': d_half, 'is_stop': False, 'is_image': False},
                {'idx': idx + 1, 'R': float(-L['r2']), 't': float(air_gap),
                 'glass': '', 'nd': 0.0, 'vd': 0.0,
                 'semi': d_half, 'is_stop': False, 'is_image': False},
            ]
            idx += 2
    specs.append({'idx': idx, 'R': np.inf, 't': 0.0, 'glass': '',
                  'nd': 0.0, 'vd': 0.0, 'semi': np.nan,
                  'is_stop': False, 'is_image': True})
    return specs


def load_elite_specs(fpath, idx=1, back_focus=55.0):
    gp, airs = load_elite(fpath, idx)
    if gp is None:
        return None
    return elite_to_specs(gp, airs, back_focus)


# ============================================================
# 2. 规格 → optiland 系统（自由表面，支持自定义 R/t/玻璃）
# ============================================================
_CURRENT_FIELD_TYPE = 'angle'
_CURRENT_PRIMARY = 0


def build_lens_from_specs(specs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
                          wavelengths=None, field_type=None, primary_index=None):
    """表面规格 → optiland Optic（玻璃材料用 AGF nd/vd）
    fields: 每项 float(视场角) 或 (y, weight)；wavelengths: [(λ, weight)]，首个为主波长
    field_type/primary_index: None 时用全局桥（app 侧边栏设置）
    物面物距：specs[0]['t']（inf=无限远，有限值=物距 mm）
    """
    if field_type is None:
        field_type = _CURRENT_FIELD_TYPE
    if primary_index is None:
        primary_index = _CURRENT_PRIMARY
    try:
        lens = Optic()
        obj_t = float(specs[0]['t']) if specs else float('inf')
        if not np.isfinite(obj_t) or obj_t <= 0:
            obj_t = float('inf')
        lens.surfaces.add(index=0, thickness=obj_t)
        lens.surfaces.add(index=1, thickness=0.0, is_stop=True)
        for s in specs[2:-1]:
            kwargs = {'index': s['idx'], 'radius': float(s['R']),
                      'thickness': float(s['t'])}
            if s['glass']:
                nd, vd = _agf_params(s['glass'], float(s['nd']), float(s['vd']))
                kwargs['material'] = AbbeMaterial(nd, vd, 'buchdahl')
            lens.surfaces.add(**kwargs)
        lens.surfaces.add(index=specs[-1]['idx'])
        lens.set_aperture(aperture_type='EPD', value=float(epd))
        lens.fields.set_type(field_type)
        if not fields:
            fields = (0.0, 2.0, 4.06, 5.8)
        for f in fields:
            if isinstance(f, (tuple, list)):
                lens.fields.add(y=float(f[0]), weight=float(f[1]))
            else:
                lens.fields.add(y=float(f))
        if wavelengths:
            for i, (wl, w) in enumerate(wavelengths):
                lens.wavelengths.add(value=float(wl), weight=float(w),
                                     is_primary=(i == primary_index))
        else:
            lens.wavelengths.add(value=0.48613, is_primary=True)
            lens.wavelengths.add(value=0.58756)
            lens.wavelengths.add(value=0.65627)
        return lens
    except Exception:
        return None


# ============================================================
# 3. 规格 ↔ LDE 表格
# ============================================================
def specs_to_df(specs):
    """规格 → 展示 DataFrame（Zemax LDE 风格列名）"""
    rows = []
    for s in specs:
        r_val = np.nan if not np.isfinite(s['R']) else round(float(s['R']), 4)
        t_val = np.nan if not np.isfinite(s['t']) else round(float(s['t']), 4)
        rows.append({
            'Surf': s['idx'],
            'Radius': r_val,
            'Thick': t_val,
            'Glass': s['glass'] or '',
            'ND': round(float(s['nd']), 5) if s['nd'] else np.nan,
            'VD': round(float(s['vd']), 3) if s['vd'] else np.nan,
            'Semi-Dia': round(float(s['semi']), 2) if np.isfinite(s['semi']) else np.nan,
            '备注': '光阑' if s['is_stop'] else ('像面' if s['is_image'] else ''),
        })
    return pd.DataFrame(rows)


def df_to_specs(df):
    """编辑后的 DataFrame → 规格（NaN R/t → inf；玻璃名匹配 AGF 自动带 nd/vd）"""
    n = len(df)
    out = []
    for i in range(n):
        row = df.iloc[i]
        def _f(v):
            try:
                x = float(v)
                return x if np.isfinite(x) else np.inf
            except (TypeError, ValueError):
                return np.inf
        R = _f(row.get('Radius', row.get('曲率半径 R', np.nan)))
        t = _f(row.get('Thick', row.get('厚度 t', np.nan)))
        if i > 1 and i < n - 1 and not np.isfinite(t):
            t = 5.0   # 白纸作画：LDE dynamic 新行默认厚度
        glass = str(row.get('Glass', row.get('玻璃', ''))).strip()
        nd, vd = 0.0, 0.0
        if glass:
            g_nd = row.get('ND', row.get('nd', np.nan))
            g_vd = row.get('VD', row.get('vd', np.nan))
            try:
                nd = float(g_nd) if np.isfinite(float(g_nd)) else 0.0
            except (TypeError, ValueError):
                nd = 0.0
            try:
                vd = float(g_vd) if np.isfinite(float(g_vd)) else 0.0
            except (TypeError, ValueError):
                vd = 0.0
            # 玻璃名在库目录 → 用 AGF 实测值（改玻璃名自动带出参数）
            try:
                cat = glass_catalog()
                if glass in cat:
                    nd, vd = cat[glass]
            except Exception:
                pass
            if nd <= 1.0 or vd <= 1.0:
                nd, vd = 1.5, 60.0
        semi = np.nan
        try:
            sv = float(row.get('Semi-Dia', row.get('半口径', np.nan)))
            if np.isfinite(sv):
                semi = sv
        except (TypeError, ValueError):
            pass
        _sv = row.get('Surf', row.get('面号', i))
        if _sv is None or (isinstance(_sv, float) and not np.isfinite(_sv)):
            _sv = i
        out.append({
            'idx': int(_sv),
            'R': R, 't': t, 'glass': glass, 'nd': nd, 'vd': vd,
            'semi': semi,
            'is_stop': i == 1, 'is_image': i == n - 1,
        })
    return out


# ============================================================
# 4. 设计持久化（json）+ Zemax 文本导出
# ============================================================
def save_design(specs, name, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8), dpath=None):
    dpath = dpath or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'designs')
    os.makedirs(dpath, exist_ok=True)
    fn = name if name.endswith('.json') else name + '.json'
    fn = os.path.join(dpath, fn)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump({'specs': specs, 'epd': epd, 'fields': list(fields)}, f,
                  ensure_ascii=False, indent=1)
    return fn


def load_design(fname, dpath=None):
    dpath = dpath or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'designs')
    with open(os.path.join(dpath, fname), encoding='utf-8') as f:
        d = json.load(f)
    return d['specs'], float(d.get('epd', 40.0)), tuple(d.get('fields', (0.0, 2.0, 4.06, 5.8)))


def list_designs(dpath=None):
    dpath = dpath or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'designs')
    if not os.path.isdir(dpath):
        return []
    return sorted(f for f in os.listdir(dpath) if f.endswith('.json'))


def specs_to_zemax_text(specs, epd=40.0, name='design'):
    """生成 Zemax LDE 文本（可直接粘贴到 Zemax LDE）"""
    lines = [f'! {name}  (由镜头设计工作台导出)',
             f'! EPD = {epd:g} mm',
             'OBJ  STANDARD    Infinity    Infinity',
             'STO  STANDARD    Infinity    0']
    for s in specs[2:-1]:
        r = 'Infinity' if not np.isfinite(s['R']) else f'{s["R"]:.6g}'
        t = 'Infinity' if not np.isfinite(s['t']) else f'{s["t"]:.6g}'
        g = s['glass'] if s['glass'] else ''
        semi = f'{s["semi"]:.4g}' if np.isfinite(s['semi']) else '0'
        line = f'{s["idx"]:<4d} STANDARD    {r:<12s} {t:<12s} {g:<10s} {semi}'
        lines.append(line)
    lines.append('IMA  STANDARD    Infinity    0')
    return '\n'.join(lines)


def auto_semi(specs, epd=40.0):
    """自动通光：optiland 近轴边缘光线每面高度 → 半口径
    semi = max(通光*1.05 + 0.5, 原值)；光阑面跳过"""
    ys = None
    try:
        lens = build_lens_from_specs(specs, epd=epd, fields=(0.0,))
        if lens is not None:
            y_arr, _ = lens.paraxial.marginal_ray()
            ys = [float(v) for v in np.asarray(y_arr).flatten()]
    except Exception:
        pass
    if ys is None:
        return specs
    out = [dict(s) for s in specs]
    for i, s in enumerate(out):
        if s['is_stop'] or s['is_image'] or i >= len(ys) - 1:
            continue
        ymax = max(abs(ys[i]), abs(ys[i + 1]))
        if ymax > 0:
            new_semi = ymax * 1.05 + 0.5
            if not np.isfinite(s['semi']) or new_semi > s['semi']:
                s['semi'] = new_semi
    return out


def set_back_focus(specs, bf):
    """设置后焦（最后一个非像面、非光阑的厚度）"""
    out = [dict(s) for s in specs]
    for i in range(len(out) - 2, -1, -1):
        if not out[i]['is_stop'] and not out[i]['is_image']:
            out[i]['t'] = float(bf)
            break
    return out


# ============================================================
# 自由增删表面（Zemax Insert/Delete Surface）
# ============================================================
def reindex_specs(specs):
    """重排 idx（0..n-1）并维护光阑/像面标记（位置推断）"""
    out = []
    n = len(specs)
    for i, s in enumerate(specs):
        s = dict(s)
        s['idx'] = i
        s['is_stop'] = (i == 1)
        s['is_image'] = (i == n - 1)
        out.append(s)
    return out


def insert_surface(specs, pos):
    """在面 pos 之后插入一个新表面（默认 R=inf, t=5, 空气）
    pos: 面号（索引）；不能是最后一个（像面）"""
    out = [dict(s) for s in specs]
    pos = int(pos)
    if pos < 0 or pos >= len(out) - 1:
        return specs
    new = {'idx': pos + 1, 'R': np.inf, 't': 5.0, 'glass': '',
           'nd': 0.0, 'vd': 0.0, 'semi': np.nan,
           'is_stop': False, 'is_image': False}
    out.insert(pos + 1, new)
    return reindex_specs(out)


def insert_lens_group(specs, pos, glass='D-K9'):
    """在面 pos 之后插入一组镜片（玻璃面 + 空气面），默认 D-K9"""
    out = [dict(s) for s in specs]
    pos = int(pos)
    if pos < 0 or pos >= len(out) - 1:
        return specs
    try:
        nd, vd = _agf_params(glass, 1.5, 60.0)
    except Exception:
        nd, vd = 1.5, 60.0
    g1 = {'idx': 0, 'R': 100.0, 't': 3.0, 'glass': glass,
          'nd': float(nd), 'vd': float(vd), 'semi': 15.0,
          'is_stop': False, 'is_image': False}
    g2 = {'idx': 0, 'R': -100.0, 't': 5.0, 'glass': '',
          'nd': 0.0, 'vd': 0.0, 'semi': 15.0,
          'is_stop': False, 'is_image': False}
    out[pos + 1:pos + 1] = [g1, g2]
    return reindex_specs(out)


# ============================================================
# 4.5 从镜片库挑选插入（库类型 1-6：pybl1-4 单镜片 / pybl5-6 双胶合）
# ============================================================
def library_rows(lt):
    """库类型有效行号区间 → (min, max, 有效数)（UI 行号范围用）"""
    global _lib
    try:
        rows = _lib.valid_rows(int(lt)) if hasattr(_lib, 'valid_rows') else []
        if rows:
            return min(rows), max(rows), len(rows)
    except Exception:
        pass
    return 0, 0, 0


def library_pick(lt, row):
    """从镜片库按（类型, 行号）取镜片 → 预览 dict；无效返回 None"""
    global _lib
    try:
        L = _lib.get_lens(int(lt), int(row))
    except Exception:
        return None
    if L is None:
        return None
    if isinstance(L, tuple):
        la, lb = L
        nd_a, vd_a = _agf_params(la['glass'], la['nd'], la['vd'])
        nd_b, vd_b = _agf_params(lb['glass'], lb['nd'], lb['vd'])
        return {'type': '双胶合',
                'text': (f'{la["glass"]} ({nd_a:.5f}/{vd_a:.1f}) + {lb["glass"]} '
                         f'({nd_b:.5f}/{vd_b:.1f}) | R1={la["r1"]:.1f} R2={-lb["r2"]:.1f} '
                         f'厚 {la["thick"]:.1f}+{lb["thick"]:.1f}')}
    nd, vd = _agf_params(L['glass'], L['nd'], L['vd'])
    return {'type': '单镜片',
            'text': (f'{L["glass"]} ({nd:.5f}/{vd:.1f}) | R1={L["r1"]:.1f} '
                     f'R2={-L["r2"]:.1f} 厚 {L["thick"]:.1f}')}


def library_lens_group(specs, pos, lt, row, t2=5.0, semi=15.0):
    """从镜片库插入一组镜片到面 pos 之后（与 add_lens_group 同构）
    单镜片：玻璃面 + 空气面；双胶合：玻璃面 + 胶合面 + 空气面
    双胶合展开与 elite_to_specs / build_optiland 完全一致（胶合面无独立空气行）"""
    global _lib
    try:
        L = _lib.get_lens(int(lt), int(row))
    except Exception:
        return specs
    if L is None:
        return specs
    out = [dict(s) for s in specs]
    pos = int(pos)
    if pos < 0 or pos >= len(out) - 1:
        return specs
    if isinstance(L, tuple):
        la, lb = L
        nd_a, vd_a = _agf_params(la['glass'], la['nd'], la['vd'])
        nd_b, vd_b = _agf_params(lb['glass'], lb['nd'], lb['vd'])
        rows = [
            {'idx': 0, 'R': float(la['r1']), 't': float(la['thick']), 'glass': la['glass'],
             'nd': float(nd_a), 'vd': float(vd_a), 'semi': float(semi),
             'is_stop': False, 'is_image': False},
            {'idx': 0, 'R': float(-la['r2']), 't': float(lb['thick']), 'glass': lb['glass'],
             'nd': float(nd_b), 'vd': float(vd_b), 'semi': float(semi),
             'is_stop': False, 'is_image': False},
            {'idx': 0, 'R': float(-lb['r2']), 't': float(t2), 'glass': '',
             'nd': 0.0, 'vd': 0.0, 'semi': float(semi),
             'is_stop': False, 'is_image': False},
        ]
    else:
        nd, vd = _agf_params(L['glass'], L['nd'], L['vd'])
        rows = [
            {'idx': 0, 'R': float(L['r1']), 't': float(L['thick']), 'glass': L['glass'],
             'nd': float(nd), 'vd': float(vd), 'semi': float(semi),
             'is_stop': False, 'is_image': False},
            {'idx': 0, 'R': float(-L['r2']), 't': float(t2), 'glass': '',
             'nd': 0.0, 'vd': 0.0, 'semi': float(semi),
             'is_stop': False, 'is_image': False},
        ]
    out[pos + 1:pos + 1] = rows
    return reindex_specs(out)


def delete_surface(specs, pos):
    """删除面 pos（物面 0 / 光阑 1 / 像面 不可删）"""
    out = [dict(s) for s in specs]
    pos = int(pos)
    if pos <= 1 or pos >= len(out) - 1:
        return specs
    del out[pos]
    return reindex_specs(out)


def blank_system(back_focus=55.0):
    """新建空白镜头：物面 / 光阑 / 像面（3 面，从零设计起点）"""
    return [
        {'idx': 0, 'R': np.inf, 't': np.inf, 'glass': '', 'nd': 0.0, 'vd': 0.0,
         'semi': np.nan, 'is_stop': False, 'is_image': False},
        {'idx': 1, 'R': np.inf, 't': 0.0, 'glass': '', 'nd': 0.0, 'vd': 0.0,
         'semi': np.nan, 'is_stop': True, 'is_image': False},
        {'idx': 2, 'R': np.inf, 't': 0.0, 'glass': '', 'nd': 0.0, 'vd': 0.0,
         'semi': np.nan, 'is_stop': False, 'is_image': True},
    ]


def add_lens_group(specs, pos, glass, r1, r2, t1, t2=5.0, semi=15.0):
    """表单式添加镜片组（玻璃面 + 空气面），玻璃名自动带 AGF nd/vd"""
    out = [dict(s) for s in specs]
    pos = int(pos)
    if pos < 0 or pos >= len(out) - 1:
        return specs
    try:
        nd, vd = _agf_params(glass, 1.5, 60.0)
    except Exception:
        nd, vd = 1.5, 60.0
    g1 = {'idx': 0, 'R': float(r1), 't': float(t1), 'glass': glass,
          'nd': float(nd), 'vd': float(vd), 'semi': float(semi),
          'is_stop': False, 'is_image': False}
    g2 = {'idx': 0, 'R': float(r2), 't': float(t2), 'glass': '',
          'nd': 0.0, 'vd': 0.0, 'semi': float(semi),
          'is_stop': False, 'is_image': False}
    out[pos + 1:pos + 1] = [g1, g2]
    return reindex_specs(out)


# ============================================================
# 4.6 结构校验（基础纠错提醒：重叠/无效面/过弯/缺参数）
# ============================================================
def validate_specs(specs):
    """基础结构校验 → 问题列表（空 = 结构正常）

    检查项：
    - 负厚度（面/镜片重叠、光路倒转）
    - 玻璃体厚度 ≤ 0（胶合面除外——胶合面无玻璃厚度 0 行）
    - |R| < 半口径（镜片过弯，追迹易异常）
    - 玻璃名存在但 nd/vd 无效（未识别玻璃）
    """
    issues = []
    for i, s in enumerate(specs):
        if i == 0 or s.get('is_image'):
            continue
        t = s.get('t')
        t_ok = t is not None and np.isfinite(t)
        if s.get('is_stop'):
            if t_ok and t < 0:
                issues.append(f'面 {i}（光阑）厚度 {t:.1f} < 0')
            continue
        if t_ok and t < 0:
            issues.append(f'面 {i} 厚度 {t:.1f} < 0 —— 表面/镜片会重叠！')
        R = s.get('R')
        semi = s.get('semi')
        if R is not None and np.isfinite(R) and abs(R) > 1e-9:
            if semi is not None and np.isfinite(semi) and semi > 0 and abs(R) < semi:
                issues.append(f'面 {i} |R|={abs(R):.1f} < 半口径 {semi:.1f} —— 镜片过弯')
        if s.get('glass') and t_ok and t <= 0:
            issues.append(f'面 {i} 玻璃厚度 {t:.1f} ≤ 0 —— 玻璃体无效')
        if s.get('glass'):
            nd = s.get('nd')
            vd = s.get('vd')
            if (nd is None or not np.isfinite(nd) or nd <= 0 or
                    vd is None or not np.isfinite(vd) or vd <= 0):
                issues.append(f'面 {i} 玻璃 {s["glass"]} 缺少有效 nd/vd')
    return issues


# ============================================================
# 5. 玻璃集合（库玻璃名 → AGF (nd, vd)）
# ============================================================
_CAT = None


def glass_catalog():
    """从库全部行收集玻璃名 → (nd, vd) AGF 实测；结果缓存"""
    global _CAT
    if _CAT is not None:
        return _CAT
    cat = {}
    for lt in range(1, 7):   # 库类型 pybl1-6
        try:
            rows = _lib.valid_rows(lt) if hasattr(_lib, 'valid_rows') else []
        except Exception:
            rows = []
        for r in rows:
            try:
                L = _lib.get_lens(lt, r)
            except Exception:
                continue
            for piece in (L if isinstance(L, tuple) else (L,)):
                g = piece.get('glass')
                if g and g not in cat:
                    nd, vd = _agf_params(g, piece.get('nd', 1.5), piece.get('vd', 50.0))
                    cat[g] = (round(nd, 5), round(vd, 2))
    _CAT = cat
    return cat


if __name__ == '__main__':
    # 自检：elite→specs→optiland 与 build_optiland 一致
    import config as CFG
    gp = [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)]
    airs = [39.9, 32.3, 27.6, 9.6, 3.0]
    specs = elite_to_specs(gp, airs)
    print('specs 数量:', len(specs))
    assert len(specs) == 18

    # 与 build_optiland 对比 RSCE
    from core._bridge import spot_rms
    lens_old = build_optiland(gp, airs, back_focus=55.0, epd=CFG.ENPD, fields=(0.0, 2.0, 4.06, 5.8))
    lens_new = build_lens_from_specs(specs, epd=CFG.ENPD, fields=(0.0, 2.0, 4.06, 5.8))
    r_old = spot_rms(lens_old)
    r_new = spot_rms(lens_new)
    print(f'RSCE 旧路径: {r_old}  |  新路径: {r_new}')
    assert abs(r_old - r_new) < 1e-6, 'RSCE 不一致！'
    print('SPECS_BUILD_MATCH: OK（新旧路径一致）')

    # df 往返
    df = specs_to_df(specs)
    specs2 = df_to_specs(df)
    r2 = spot_rms(build_lens_from_specs(specs2, epd=CFG.ENPD, fields=(0.0, 2.0, 4.06, 5.8)))
    print(f'df 往返后 RSCE: {r2}')
    assert abs(r2 - r_old) < 1e-6, 'df 往返不一致！'
    print('DF_ROUNDTRIP: OK')

    # 玻璃目录
    cat = glass_catalog()
    print('玻璃目录条目:', len(cat))
    print('  样例:', {k: cat[k] for k in list(cat)[:8]})

    # Zemax 导出
    txt = specs_to_zemax_text(specs, epd=CFG.ENPD, name='elite55')
    print()
    print('=== Zemax 导出预览 ===')
    print(txt[:400])
