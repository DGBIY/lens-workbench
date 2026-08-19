# -*- coding: utf-8 -*-
"""core/_bridge.py — 自包含桥接层（分享版：无完整项目/Zemax 依赖）

从完整项目 src/optiland_bridge.py 内嵌的核心函数：
  _agf_params         玻璃名 → (nd, vd)（查内置玻璃表，无 Zemax Glasscat 依赖）
  spot_rms            点列图 RMS（optiland 真实追迹，对应 Zemax RSCE）
  _bfl_from_paraxial  后焦距（最后光学表面 → 焦点）
  build_optiland      从库行号（基因）构建 optiland 系统（GA 精英解码用）
"""
import numpy as np

from optiland.optic import Optic
from optiland.materials import AbbeMaterial
from optiland.analysis import SpotDiagram

from core._glass_table import GLASS_TABLE


def _agf_params(glass, fallback_nd, fallback_vd):
    """玻璃名 → (AGF 实测 nd, vd)；内置表无此玻璃时回退库文件值"""
    return GLASS_TABLE.get(str(glass), (float(fallback_nd), float(fallback_vd)))


def spot_rms(lens, num_rings=6):
    """点列图 RMS（全视场×全波长最大，对应 Zemax RSCE）
    sd.data[field_idx][wl_idx] = SpotData(x, y, intensity)
    相对质心 RMS（与 Zemax RSCE 同义，边缘视场像高不参与）
    """
    try:
        sd = SpotDiagram(lens, num_rings=num_rings)
        rms_all = []
        for field_data in sd.data:
            for spot in field_data:
                x = np.asarray(spot.x, dtype=float)
                y = np.asarray(spot.y, dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 10:
                    continue
                cx = x[valid].mean()
                cy = y[valid].mean()
                rms_all.append(float(np.sqrt(np.mean((x[valid] - cx) ** 2 +
                                                     (y[valid] - cy) ** 2))))
        return max(rms_all) if rms_all else float('nan')
    except Exception:
        return float('nan')


def _bfl_from_paraxial(lens):
    """后焦距（最后光学表面→焦点）。与近轴引擎/Zemax BFL 定义一致"""
    try:
        y, u = lens.paraxial.marginal_ray()
        y_last = float(np.asarray(y)[-2][0])
        u_last = float(np.asarray(u)[-2][0])
        if abs(u_last) < 1e-12:
            return 0.0
        return -y_last / u_last
    except Exception:
        return float('nan')


def build_optiland(gene, airs, back_focus=55.0, epd=58.0,
                   fields=(0.0, 4.1, 5.8)):
    """从库行号（基因）构建 optiland 系统（GA 精英解码用）
    gene: [(lt, row) × 6]；airs: 5 个空气间隔
    """
    from core._library import get_default_library
    lib = get_default_library()
    lens = Optic()
    lens.surfaces.add(index=0, thickness=float('inf'))
    lens.surfaces.add(index=1, thickness=0.0, is_stop=True)
    idx = 2
    for i in range(6):
        lt, row = gene[i]
        L = lib.get_lens(lt, row)
        if L is None:
            return None
        air_gap = airs[i] if i < 5 else back_focus
        if isinstance(L, tuple):
            la, lb = L
            nd_a, vd_a = _agf_params(la['glass'], la['nd'], la['vd'])
            nd_b, vd_b = _agf_params(lb['glass'], lb['nd'], lb['vd'])
            lens.surfaces.add(index=idx, radius=la['r1'], thickness=la['thick'],
                              material=AbbeMaterial(nd_a, vd_a, 'buchdahl'))
            lens.surfaces.add(index=idx + 1, radius=-la['r2'], thickness=lb['thick'],
                              material=AbbeMaterial(nd_b, vd_b, 'buchdahl'))
            lens.surfaces.add(index=idx + 2, radius=-lb['r2'], thickness=air_gap)
            idx += 3
        else:
            nd_g, vd_g = _agf_params(L['glass'], L['nd'], L['vd'])
            lens.surfaces.add(index=idx, radius=L['r1'], thickness=L['thick'],
                              material=AbbeMaterial(nd_g, vd_g, 'buchdahl'))
            lens.surfaces.add(index=idx + 1, radius=-L['r2'], thickness=air_gap)
            idx += 2
    lens.surfaces.add(index=idx)
    lens.set_aperture(aperture_type='EPD', value=epd)
    lens.fields.set_type('angle')
    for y in fields:
        lens.fields.add(y=y)
    lens.wavelengths.add(value=0.48613, is_primary=True)
    lens.wavelengths.add(value=0.58756)
    lens.wavelengths.add(value=0.65627)
    return lens
