# -*- coding: utf-8 -*-
"""
library.py — 镜片库加载与约束
================================
【2026-08-18 设计修订：浮动直径策略】
  之前：全部镜片直径 ≥ 54mm 一刀切 → 95% 镜片被淘汰
  现在：全局下限仅 25mm（DIY 制造可行性），
        位置1（光阑）≥54mm 由评估引擎保证（MIN_DIAM_POS1），
        位置2-6 按实际光线高度浮动判定（光锥多宽就要多大），
        允许镜片直径大于需求值（不强制用满）。
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from config import PYBL_DIR

LIB_DIR = PYBL_DIR

MIN_DIAM = 25.0   # 全局下限：DIY 制造可行性（不再按口径一刀切）
MIN_R = 27.0      # 最小曲率半径（镜片强度/加工）

# 【2026-08-18 前大后小浮动阈值 — v2 修订】
# v1 阶梯 [60,58,56,54,...42] 的问题：静态阈值与用户"浮动动态"意图冲突，
#   位置2-10 本可用小镜片（光锥已收窄）却被阶梯强制淘汰。
# v2：位置1（光阑）保底 54mm（通光保证），位置2-10 只用全局下限 25mm，
#   实际通光需求由评估引擎按光线高度动态判定（needed=2×(ymax+余量)），
#   "前大后小"由物理光锥自然涌现，不再人为硬编码。
POS_DIAM_MIN = [54.0, 25.0, 25.0, 25.0, 25.0, 25.0]  # 6 片时代（位置1 保底 54，其余 25）

# 玻璃名映射（CDGM.AGF 中不存在，知识库 KNOWLEDGE.md 4.4 节）
GLASS_MAP = {
    'H-K9': 'D-K9',        # H-K9 是笔误，实际为 D-K9 (nd 1.5164/vd 64.08)
    'H-ZF7L': 'H-ZF7LA',   # 笔误，实际为 H-ZF7LA
    'H-ZK9': 'H-ZK9B',     # 笔误，实际为 H-ZK9B
    # 【2026-08-18 补充】缺失玻璃 → AGF 等价（glass_mapping.py 校准，Δnd≈0）
    # 之前这些玻璃走 nd/vd 近似；映射后使用 AGF 真实色散，色差精度提升。
    'H-LAK7': 'H-LAK7A',   # Δnd=0.0e-3 Δvd=0.1（同系列）
    'LZ_LK3': 'J-FK5',     # Δnd=0.0e-3 Δvd=0.2
    'S-BSM16': 'SK16',     # Δnd=0.0e-3 Δvd=0.0（完全等价）
    'S-TIL6': 'E-LLF6',    # Δnd=0.0e-3 Δvd=0.0（完全等价）
}

# 类型名
LIB_NAMES = {
    1: '双凹', 2: '弯月凸左', 3: '双凸', 4: '弯月凸右',
    5: '双片', 6: '双片反向',
}


def _parse_lib_file(fname):
    """解析单片库文件：r1 r2 thick diam glass nd vd"""
    rows = []
    with open(os.path.join(LIB_DIR, fname), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                rows.append({
                    'r1': float(parts[0]),
                    'r2': float(parts[1]),
                    'thick': float(parts[2]),
                    'diam': float(parts[3]),
                    'glass': parts[4],
                    'nd': float(parts[5]),
                    'vd': float(parts[6]),
                })
    return rows


def _parse_doublet_file(fname, single_libs):
    """解析双片库文件：libA rowA libB rowB（引用单片库）"""
    rows = []
    with open(os.path.join(LIB_DIR, fname), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                libA, rowA = int(parts[0]), int(parts[1])
                libB, rowB = int(parts[2]), int(parts[3])
                la = single_libs[libA - 1][rowA - 1] if 0 <= libA - 1 < len(single_libs) and 0 <= rowA - 1 < len(single_libs[libA - 1]) else None
                lb = single_libs[libB - 1][rowB - 1] if 0 <= libB - 1 < len(single_libs) and 0 <= rowB - 1 < len(single_libs[libB - 1]) else None
                rows.append((la, lb))
    return rows


class LensLibrary:
    """镜片库（加载 + 约束过滤）"""

    def __init__(self, lib_dir=LIB_DIR, min_diam=MIN_DIAM, min_r=MIN_R):
        self.lib_dir = lib_dir
        self.min_diam = min_diam
        self.min_r = min_r
        self._single = None   # [pybl1..pybl4] 原始行
        self._single_valid = None
        self._doublet = None  # [pybl5, pybl6]
        self._doublet_valid = None
        self.load()

    def load(self):
        self._single = [
            _parse_lib_file('pybl1.txt'),
            _parse_lib_file('pybl2.txt'),
            _parse_lib_file('pybl3.txt'),
            _parse_lib_file('pybl4.txt'),
        ]
        self._doublet = [
            _parse_doublet_file('pybl5.txt', self._single),
            _parse_doublet_file('pybl6.txt', self._single),
        ]
        self._build_valid()

    def _lens_ok(self, L):
        if L is None:
            return False
        return (L['diam'] >= self.min_diam and
                abs(L['r1']) >= self.min_r and
                abs(L['r2']) >= self.min_r)

    def _build_valid(self):
        self._single_valid = [
            [i + 1 for i, L in enumerate(lib) if self._lens_ok(L)]
            for lib in self._single
        ]
        self._doublet_valid = [
            [i + 1 for i, (la, lb) in enumerate(lib)
             if la is not None and lb is not None and
             self._lens_ok(la) and self._lens_ok(lb)]
            for lib in self._doublet
        ]

    def valid_rows(self, lt):
        """有效行号列表（1-based）"""
        if 1 <= lt <= 4:
            return self._single_valid[lt - 1]
        if 5 <= lt <= 6:
            return self._doublet_valid[lt - 5]
        return []

    def valid_rows_at(self, lt, pos):
        """位置感知有效行号：在 valid_rows 基础上按该位置最小直径过滤
        前大后小：位置1 最严（通光+装配余量），后面逐级放宽
        """
        rows = self.valid_rows(lt)
        if not rows:
            return []
        th = POS_DIAM_MIN[pos - 1]
        out = []
        for row in rows:
            L = self.get_lens(lt, row)
            if L is None:
                continue
            d = min(L[0]['diam'], L[1]['diam']) if isinstance(L, tuple) else L['diam']
            if d >= th:
                out.append(row)
        return out

    def get_lens(self, lt, row):
        """获取镜片（row 1-based）。单片返回 dict，双片返回 tuple(dict, dict)"""
        try:
            if 1 <= lt <= 4:
                L = self._single[lt - 1][row - 1]
                if L is None or not self._lens_ok(L):
                    return None
                L = dict(L)
                L['glass'] = GLASS_MAP.get(L['glass'], L['glass'])
                return L
            if 5 <= lt <= 6:
                la, lb = self._doublet[lt - 5][row - 1]
                if la is None or lb is None:
                    return None
                # 【BUG-1 修复 2026-08-18】双片两片都要过完整约束
                # （直径 + 曲率半径），与单片同等严格，防止 r<27mm
                # 的不可加工镜片混入。
                if not self._lens_ok(la) or not self._lens_ok(lb):
                    return None
                la = dict(la)
                lb = dict(lb)
                la['glass'] = GLASS_MAP.get(la['glass'], la['glass'])
                lb['glass'] = GLASS_MAP.get(lb['glass'], lb['glass'])
                return (la, lb)
        except (IndexError, TypeError):
            return None
        return None

    def stats(self):
        total = sum(len(v) for v in self._single_valid) + \
                sum(len(v) for v in self._doublet_valid)
        detail = [f'pybl{i+1}={len(self._single_valid[i])} 有效'
                  for i in range(4)]
        detail += [f'pybl{i+5}={len(self._doublet_valid[i])} 有效'
                   for i in range(2)]
        return total, detail


# 全局单例
_default = None


def get_default_library():
    global _default
    if _default is None:
        _default = LensLibrary()
    return _default
