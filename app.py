# -*- coding: utf-8 -*-
"""app.py — 镜头设计工作台 v0.10（集成中心）
侧边栏 = 分类导航（📂 数据 / ⚙️ 系统 / 🚀 运行 / 🛠 设计），一次一类，顶部固定系统摘要
优化移入主区工作区（Zemax Optimize 独立菜单语义）；工具栏瘦身（新建/重置入侧边栏）
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as CFG
if CFG.GA_PROJECT_ROOT:
    for p in (CFG.GA_PROJECT_ROOT, os.path.join(CFG.GA_PROJECT_ROOT, 'scripts')):
        if p not in sys.path:
            sys.path.insert(0, p)
ELITES_DIR = os.path.join(CFG.GA_PROJECT_ROOT, 'results') if CFG.GA_PROJECT_ROOT else CFG.SAMPLES_DIR

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='镜头设计工作台', layout='wide', page_icon='🔭')

# ============ Zemax 风格 UI（CSS）：分析图窗口可拖拽缩放等 ============
st.markdown("""
<style>
/* 分析图"窗口"：右下角可拖拽缩放（Zemax 窗口感） */
[data-testid="stImage"] {
    resize: both;
    overflow: auto;
    min-width: 320px;
    min-height: 220px;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    background: #fbfdff;
    padding: 6px;
}
[data-testid="stImage"] img { width: 100%; height: auto; }
/* 指标卡片卡片化 */
[data-testid="stMetric"] {
    background: #f6f8fa; border: 1px solid #eaeef2;
    border-radius: 8px; padding: 8px 12px;
}
/* 侧边栏分区标题紧凑化 */
section[data-testid="stSidebar"] h3 { font-size: 14px; margin-top: 14px; }
/* 分段控件下留白 */
[data-testid="stSegmentedControl"] { margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

from core.lens_io import (elite_to_specs, load_elite_specs, specs_to_df,
                          df_to_specs, save_design, load_design, list_designs,
                          specs_to_zemax_text, auto_semi, set_back_focus,
                          glass_catalog, insert_surface, delete_surface,
                          blank_system, add_lens_group, library_pick,
                          library_lens_group, library_rows, validate_specs)
from core.layout2d import plot_layout
from core.layout3d import plot_layout_3d
from core.spot_rms import compute_spot
from core.eval import evaluate_specs
from core.zmx_export import specs_to_zmx
from core import run_control
from core.compare import compare_table, load_multi
from core.analysis import (analysis_fig, axial_color_fig, optimize_local,
                           search_glass, mtf_fig, KINDS)
from core.astro_tools import (pixel_scale, fov, diffraction_limit, airy_diameter,
                              fwhm_arcsec, sampling_ratio, critical_focus_depth,
                              limit_magnitude, light_gain, plate_shift,
                              reducer_effect, thermal_shift,
                              image_scale, pixels_on_body, star_photons,
                              sky_photons_per_px, snr, limiting_magnitude,
                              stacked_limit, combined_focal, reducer_design,
                              reducer_focal_for)
from core.starfield import (render_starfield, render_exposure_stars,
                            best_focus_offset, _SKY_MAG, _BODY_DEG)
from core.templates import build_template, TEMPLATES
from core.merit import PRESETS as MERIT_PRESETS, merit_from_preset

DEFAULT_FILE = 'run_20260818_213729/ga_elite.txt'
DEFAULT_IDX = 55
DEFAULT_FIELDS = (0.0, 2.0, 4.06, 5.8)
DEFAULT_WAVS = [(0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)]

if 'specs' not in st.session_state:
    st.session_state.specs = load_elite_specs(
        os.path.join(ELITES_DIR, DEFAULT_FILE), DEFAULT_IDX) or elite_to_specs(
        [(5, 259), (5, 158), (5, 152), (4, 217), (2, 244), (2, 199)],
        [39.9, 32.3, 27.6, 9.6, 3.0])
    st.session_state.epd = 40.0
    st.session_state.src_label = f'{DEFAULT_FILE} #{DEFAULT_IDX}'
    st.session_state.msg = None
    st.session_state.panel_log = None
    st.session_state.wavs = list(DEFAULT_WAVS)
    st.session_state.fields = [(y, 1.0) for y in DEFAULT_FIELDS]
    st.session_state.stop_surf = 1


def _apply_stop(specs):
    """把 session 指定的光阑面号应用到 specs（白纸作画：光阑位置可移动）"""
    _st = int(st.session_state.get('stop_surf', 1))
    _st = max(1, min(_st, len(specs) - 2))
    st.session_state.stop_surf = _st
    for i, s in enumerate(specs):
        s['is_stop'] = (i == _st)
    return specs

# ============ 侧边栏：集成中心（分类导航） ============
with st.sidebar:
    st.title('🎛 集成中心')
    st.caption(f'📋 {st.session_state.src_label} | EPD {float(st.session_state.epd):.0f} mm | '
               f'主波长 λ1 | {len(st.session_state.specs)} 面')
    st.divider()

    section = st.radio('功能区', ['📂 数据', '⚙️ 系统', '🚀 运行', '🛠 设计'],
                       key='hub_section')

    if section == '📂 数据':
        with st.expander('📥 精英载入', expanded=True):
            fname = st.text_input('精英文件（相对 results/）', DEFAULT_FILE, key='fname')
            idx = st.number_input('精英序号', 1, 300, DEFAULT_IDX, step=1, key='eidx')
            if st.button('载入精英', use_container_width=True, key='load_elite'):
                s0 = load_elite_specs(os.path.join(ELITES_DIR, fname), int(idx))
                if s0:
                    st.session_state.pop('lde_table', None)
                    st.session_state.specs = _apply_stop(s0)
                    st.session_state.src_label = f'{fname} #{idx}'
                    st.session_state.msg = ('ok', f'已载入 {fname} #{idx}')
                else:
                    st.session_state.msg = ('err', f'无法解析 {fname} #{idx}')
        with st.expander('💾 设计存取'):
            dname = st.text_input('设计名', 'my_design', key='dname')
            if st.button('保存设计', use_container_width=True, key='save_d'):
                fn = save_design(st.session_state.specs, dname,
                                 epd=float(st.session_state.epd), fields=DEFAULT_FIELDS)
                st.session_state.msg = ('ok', f'已保存: {os.path.basename(fn)}')
            designs = list_designs()
            sel_d = st.selectbox('已保存设计', designs if designs else ['（无）'], key='sel_d')
            if st.button('载入设计', use_container_width=True, key='load_d',
                         disabled=not designs):
                s0, e0, _f = load_design(sel_d)
                st.session_state.pop('lde_table', None)
                st.session_state.specs = _apply_stop(s0)
                st.session_state.epd = e0
                st.session_state.src_label = f'设计: {sel_d}'
                st.session_state.msg = ('ok', f'已载入设计 {sel_d}')

    elif section == '⚙️ 系统':
        with st.expander('🔧 口径', expanded=True):
            st.session_state.epd = st.slider('EPD (mm)', 10.0, 80.0,
                                             float(st.session_state.epd), step=1.0,
                                             key='epd_slider')
            st.session_state.fig_scale = st.slider(
                '图窗口尺寸', 0.6, 1.6, float(st.session_state.get('fig_scale', 1.0)),
                step=0.05, key='fig_scale_slider',
                help='所有分析图窗口的缩放倍率（配合窗口右下角拖拽使用）')
            cur_bf = 55.0
            for _i in range(len(st.session_state.specs) - 2, 0, -1):
                _s = st.session_state.specs[_i]
                if not _s['is_stop'] and not _s['is_image']:
                    cur_bf = float(_s['t'])
                    break
            new_bf = st.number_input('后焦 BFL (mm)', 1.0, 150.0, cur_bf, step=0.5,
                                     key='bf')
            if abs(new_bf - cur_bf) > 1e-9:
                st.session_state.specs = set_back_focus(st.session_state.specs, new_bf)
                st.session_state.msg = ('ok', f'后焦 → {new_bf:.1f} mm')
            _od_inf = st.checkbox('无限远物距', value=(not np.isfinite(st.session_state.specs[0]['t'])),
                                  key='od_inf')
            if not _od_inf:
                _cur_od = float(st.session_state.specs[0]['t'])
                if not np.isfinite(_cur_od) or _cur_od <= 0:
                    _cur_od = 1000.0
                _od_val = st.number_input('物距 (mm)', 50.0, 100000.0, _cur_od,
                                          step=50.0, key='od_val')
                if abs(_od_val - float(st.session_state.specs[0]['t'])) > 1e-6:
                    st.session_state.specs[0]['t'] = float(_od_val)
            elif np.isfinite(st.session_state.specs[0]['t']):
                st.session_state.specs[0]['t'] = float('inf')
        with st.expander('🌈 波长', expanded=True):
            wavs_ui = []
            for i in range(3):
                c1, c2 = st.columns([3, 1])
                with c1:
                    wl = st.number_input(f'λ{i+1}', 0.38, 0.75,
                                         float(st.session_state.wavs[i][0]),
                                         step=0.005, format='%.5f', key=f'wl_{i}')
                with c2:
                    wt = st.number_input(f'w{i+1}', 0.0, 1.0,
                                         float(st.session_state.wavs[i][1]),
                                         step=0.1, key=f'ww_{i}')
                wavs_ui.append((wl, wt))
            st.session_state.wavs = wavs_ui
            _pw = st.radio('主波长', ['λ1', 'λ2', 'λ3'],
                           index=int(st.session_state.get('primary_wl', 0)),
                           horizontal=True, key='pw_sel')
            st.session_state.primary_wl = ['λ1', 'λ2', 'λ3'].index(_pw)
        with st.expander('🎯 视场', expanded=True):
            _ft_opts = ['angle', 'object_height', 'paraxial_image_height', 'real_image_height']
            _ft_lbl = {'angle': '角度 Angle', 'object_height': '物高 Object Height',
                       'paraxial_image_height': '近轴像高 Paraxial Img Ht',
                       'real_image_height': '真实像高 Real Img Ht'}
            _ft = st.selectbox('视场类型', _ft_opts,
                               index=_ft_opts.index(st.session_state.get('field_type', 'angle')),
                               format_func=lambda k: _ft_lbl[k], key='ft_sel')
            st.session_state.field_type = _ft
            n_f = st.selectbox('视场数', [1, 2, 3, 4, 5], index=3, key='n_f')
            fields_ui = []
            _dft = (0.0, 2.0, 4.06, 5.8)
            for i in range(n_f):
                c1, c2 = st.columns([3, 1])
                with c1:
                    yv = st.number_input(
                        f'Y{i+1} (°)', -90.0, 90.0,
                        float(st.session_state.fields[i][0]) if i < len(st.session_state.fields) else _dft[i],
                        step=0.1, key=f'fy_{i}')
                with c2:
                    wv = st.number_input(
                        f'W{i+1}', 0.0, 1.0,
                        float(st.session_state.fields[i][1]) if i < len(st.session_state.fields) else 1.0,
                        step=0.1, key=f'fw_{i}')
                fields_ui.append((yv, wv))
            st.session_state.fields = fields_ui

    elif section == '🚀 运行':
        if not run_control.SCRIPTS:
            st.info('🔌 GA 运行控制未配置——可选功能（需要完整 GA 项目环境 + Zemax COM）\n\n'
                    '设置环境变量 GA_PROJECT_ROOT 指向完整项目后启用，详见 README')
        else:
            with st.expander('▶ 任务控制', expanded=True):
                mode = st.selectbox('任务', list(run_control.SCRIPTS.keys()),
                                    format_func=lambda k: run_control.SCRIPTS[k]['label'],
                                    key='rc_mode')
                params = {}
                for arg in run_control.SCRIPTS[mode]['args']:
                    key, typ, default, flag = arg[:4]
                    rest = arg[4:]
                    help_ = rest[0] if rest else key
                    if typ == 'int':
                        lo, hi = (rest[1:3] if len(rest) >= 3 else (0, 100000))
                        params[key] = st.number_input(help_, int(lo), int(hi),
                                                      int(default), step=1, key=f'rc_{key}')
                    elif typ == 'select':
                        opts = rest[1] if len(rest) >= 2 else [default]
                        params[key] = st.selectbox(help_, opts,
                                                   index=opts.index(default) if default in opts else 0,
                                                   key=f'rcs_{key}')
                    else:
                        params[key] = st.text_input(help_, str(default), key=f'rct_{key}')
                if run_control.SCRIPTS[mode]['note']:
                    st.caption(run_control.SCRIPTS[mode]['note'])
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button('▶ 启动', use_container_width=True, key='rc_go'):
                        ok, msg, logp = run_control.launch(mode, params)
                        st.session_state.panel_log = logp
                        (st.success if ok else st.error)(msg)
                with rc2:
                    if st.button('⏹ 停止', use_container_width=True, key='rc_stop'):
                        ok, msg = run_control.stop()
                        (st.success if ok else st.warning)(msg)
        with st.expander('🧬 optiland GA（免 Zemax · v0.24）'):
            st.caption('工作台内置遗传算法：optiland 真实追迹评估（零 Zemax 依赖）。'
                       '需要 Zemax 引擎请用上方"▶ 任务控制"（引擎开关，可两者并用）')
            gg1, gg2, gg3 = st.columns(3)
            with gg1:
                _ga_ng = st.number_input('镜片组数', 4, 10, 6, step=1, key='ga_ng',
                                         help='基因长度：N 组镜片 + N-1 空气间隔（推荐 6，可调）')
                _ga_pop = st.number_input('种群', 8, 60, 16, step=4, key='ga_pop')
                _ga_gen = st.number_input('代数', 3, 100, 12, step=1, key='ga_gen')
            with gg2:
                _ga_preset = st.selectbox('评价函数预设', list(MERIT_PRESETS.keys()),
                                          key='ga_preset')
                _ga_efl = st.number_input('目标焦距 (mm)', 50.0, 500.0, 200.0,
                                          step=10.0, key='ga_efl')
                _ga_seeds = st.multiselect('模板种子（初始种群·天文模板联动）',
                                           [k for k in TEMPLATES
                                            if not TEMPLATES[k].get('reflective')],
                                           default=['tessar'],
                                           format_func=lambda k: TEMPLATES[k]['label'],
                                           key='ga_seeds')
            with gg3:
                _ga_eng = st.radio('评估引擎',
                                   ['近轴 GA · CPU 向量化（6 组）',
                                    '近轴 GA · GPU（6 组）',
                                    '内置 optiland（任意组数）'],
                                   index=0, key='ga_eng',
                                   help='近轴引擎复用完整项目 ga_fast（KNN/重启/爬坡/去重，'
                                        'CPU 向量化 500×20 代 ≈ 1s；GPU 需 N 卡 + python_env）')
                if int(_ga_ng) != 6 and _ga_eng.startswith('近轴'):
                    st.warning('⚠️ 近轴引擎固定 6 组基因——其他组数请用"内置 optiland"')
                _ga_mr = st.slider('变异率（内置模式）', 0.05, 0.9, 0.3, 0.05, key='ga_mr')
                _ga_refine = st.checkbox('跑完自动局部精修（Nelder-Mead 快速）',
                                         value=True, key='ga_refine')
            with st.expander('📋 评价函数明细（操作数/目标/权重）'):
                _gd = pd.DataFrame([{'操作数': op, '目标': t, '权重': w}
                                    for op, t, w in MERIT_PRESETS[_ga_preset]])
                st.dataframe(_gd, use_container_width=True, hide_index=True)
                st.caption('MFE = √(Σw·Δ²/Σw)（Zemax 惯例）；12 种操作数可换预设，'
                           '或改权重/目标后重跑')
            if st.button('▶ 启动 GA（跑完自动回填 LDE + 星场验证）',
                         use_container_width=True, key='ga_go'):
                from core import ga_engine
                _bar = st.progress(0.0, text='GA 进化中...')

                def _prog(g, f):
                    _bar.progress(min(1.0, g / max(1, int(_ga_gen))),
                                  text=f'第 {g}/{int(_ga_gen)} 代 · best {f:.3f}')

                _seed_now = int(st.session_state.get('ga_seed_ctr', 0)) + 1
                st.session_state.ga_seed_ctr = _seed_now
                _hist = []
                _bs = None
                if _ga_eng.startswith('近轴'):
                    if int(_ga_ng) != 6:
                        st.error('近轴引擎固定 6 组基因——其他组数请用"内置 optiland"')
                    else:
                        with st.spinner('近轴 GA 运行中（向量化，很快）...'):
                            _genes, _hist = ga_engine.run_ga_remote(
                                engine='gpu' if 'GPU' in _ga_eng else 'cpu',
                                pop=int(_ga_pop), gens=int(_ga_gen), seed=_seed_now,
                                target_efl=float(_ga_efl), progress=_prog)
                        _bs = elite_to_specs(
                            [(int(_genes[i]), int(_genes[6 + i])) for i in range(6)],
                            [float(x) for x in _genes[12:17]])
                else:
                    with st.spinner('内置 optiland GA 运行中（约 30-120s）...'):
                        _bi, _bs, _hist = ga_engine.run_ga(
                            n_groups=int(_ga_ng),
                            merit=merit_from_preset(_ga_preset, float(_ga_efl)),
                            epd=epd, fields=fields_ui, wavs=wavs_ui,
                            pop=int(_ga_pop), gens=int(_ga_gen),
                            seed_templates=list(_ga_seeds), back_focus=55.0,
                            progress=_prog)
                _bar.progress(1.0, text='完成')
                if _bs:
                    if _ga_refine:
                        with st.spinner('局部精修（Nelder-Mead 快速）...'):
                            _nr, _hr, _b0, _a0 = optimize_local(
                                _bs, epd=epd, fields=fields_ui,
                                wavelengths=wavs_ui, include_rsce=False)
                            if _a0 < _b0:
                                _bs = _nr
                                st.caption(f'✅ 精修改善 MFE {_b0:.4f} → {_a0:.4f}')
                    st.session_state.pop('lde_table', None)
                    st.session_state.pop('bf', None)
                    st.session_state.specs = _apply_stop(_bs)
                    st.session_state.epd = epd
                    st.session_state.src_label = f'GA best（{_ga_eng.split("（")[0]}）'
                    st.session_state.msg = ('ok', f'GA 完成（best {_hist[-1][1]:.3f}），'
                                           '已自动回填 LDE')
                    _gm = evaluate_specs(_bs, epd=epd, fields=fields_ui,
                                         wavelengths=wavs_ui)
                    st.success(f'✅ 回填：EFFL {_gm["efl"]:.1f} | AXCL '
                               f'{_gm["axcl"]:+.3f} | RSCE {_gm["rsce_um"]:.0f}µm')
                    with st.spinner('渲染星场验证...'):
                        _sf = render_starfield(_bs, epd=epd, fields=fields_ui,
                                               wavelengths=wavs_ui, mode='grid',
                                               n_stars=25, scale=15.0, annotate=True)
                    if _sf is not None:
                        st.pyplot(_sf)
                        st.caption('↑ GA 结果星场验证——不满意可改引擎/预设/种子重跑')
                    if _hist:
                        st.line_chart(pd.DataFrame(
                            {'gen': [h[0] for h in _hist],
                             'best_mfe': [h[1] for h in _hist]}).set_index('gen'))
                else:
                    st.error('GA 失败（无有效个体）——检查镜片库或减小组数')

            with st.expander('📊 运行状态'):
                _ri = run_control.get_run_info()
                st.caption(f'状态: {"⏳ 运行中 (PID " + str(_ri["pid"]) + ")" if _ri["running"] else "⏸ 空闲"}')
                if _ri['running'] and _ri['started_at'] and _ri['elapsed_s']:
                    _es = int(_ri['elapsed_s'])
                    st.caption(f'已运行 {_es // 60} 分 {_es % 60} 秒 | 数据点 {_ri["data_points"]}'
                               + (f' | best {_ri["latest_best"]:.4f}'
                                  if _ri['latest_best'] is not None else ''))
                    _hs = run_control.list_history_csv()
                    if _hs:
                        _g_, _b_ = run_control.read_history_csv(_hs[0])
                        if _g_:
                            st.line_chart(pd.DataFrame({'gen': _g_, 'best_mfe': _b_})
                                          .set_index('gen'), height=140)

    else:  # 🛠 设计
        with st.expander('🆕 画布管理', expanded=True):
            if st.button('✨ 新建空白镜头', use_container_width=True, key='new_blank'):
                st.session_state.pop('bf', None)
                st.session_state.pop('lde_table', None)
                st.session_state.specs = _apply_stop(blank_system())
                st.session_state.epd = 40.0
                st.session_state.src_label = '新建空白镜头'
                st.session_state.msg = ('ok', '已新建空白系统——直接在下表"追加行"开始画镜头')
            if st.button('↩️ 重置默认精英55', use_container_width=True, key='reset_dft'):
                st.session_state.pop('bf', None)
                st.session_state.pop('lde_table', None)
                st.session_state.specs = _apply_stop(
                    load_elite_specs(os.path.join(ELITES_DIR, DEFAULT_FILE), DEFAULT_IDX)
                    or st.session_state.specs)
                st.session_state.epd = 40.0
                st.session_state.src_label = f'{DEFAULT_FILE} #{DEFAULT_IDX}'
                st.session_state.msg = ('ok', '已重置为默认精英55')
        with st.expander('⚙️ 光阑设置'):
            _smax = max(1, len(st.session_state.specs) - 2)
            _scur = int(st.session_state.get('stop_surf', 1))
            _stop_new = st.number_input('光阑面号', 1, _smax, _scur, step=1,
                                        key='stop_surf_in')
            if st.button('设为光阑', use_container_width=True, key='set_stop'):
                st.session_state.stop_surf = min(int(_stop_new), _smax)
                st.session_state.pop('lde_table', None)
                st.session_state.specs = _apply_stop(st.session_state.specs)
                st.session_state.msg = ('ok', f'光阑 → 面 {int(_stop_new)}')
        st.divider()
        st.info('✏️ LDE 直接追加/删除行（白纸作画）\n📐 自动通光 / ⚡ 优化在"🛠 设计"页')

    st.divider()
    if st.session_state.msg:
        ok, m = st.session_state.msg
        (st.success if ok == 'ok' else st.error)(m)

# ============ 顶部：指标 + 工具栏 ============
# ---- 全局桥：视场类型 / 主波长（所有 build 调用自动跟随）----
import core.lens_io as _lens_io_mod
_lens_io_mod._CURRENT_FIELD_TYPE = str(st.session_state.get('field_type', 'angle'))
_lens_io_mod._CURRENT_PRIMARY = int(st.session_state.get('primary_wl', 0))

st.title('🔭 镜头设计工作台')
st.caption('流程：新建/载入 → 编辑 LDE → 分析（2D/Spot/像差）→ 优化 → 导出 .zmx')

specs = st.session_state.specs
epd = float(st.session_state.epd)
fields_ui = st.session_state.fields
wavs_ui = st.session_state.wavs

m = evaluate_specs(specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui)
mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
_has_mirror = any(str(s.get('glass', '')).upper() == 'MIRROR' for s in specs)
mc1.metric('近轴 EFFL (mm)',
           ('— 反射系统' if _has_mirror else (f"{m['efl']:.1f}" if np.isfinite(m['efl']) else '—')))
mc2.metric('AXCL (mm)', f"{m['axcl']:+.3f}" if np.isfinite(m['axcl']) else '—')
mc3.metric('总长 (mm)', f"{m['total']:.1f}" if np.isfinite(m['total']) else '—')
mc4.metric('RSCE (µm)', f"{m['rsce_um']:.0f}" if np.isfinite(m['rsce_um']) else '—')
mc5.metric('F/#', f"{epd / m['efl']:.2f}" if np.isfinite(m['efl']) and m['efl'] > 0 else '—')
_bfl_v = float(specs[-2]['t']) if len(specs) > 2 else float('nan')
mc6.metric('后焦 BFL (mm)', f"{_bfl_v:.1f}" if np.isfinite(_bfl_v) else '—')

_issues = validate_specs(specs)
if _issues:
    st.warning('⚠️ 结构校验：' + '；'.join(_issues[:4]) +
               (f'…（共 {len(_issues)} 项）' if len(_issues) > 4 else ''))

tc1, tc2, tc3 = st.columns(3)
with tc1:
    st.download_button('📤 导出 .zmx',
                       specs_to_zmx(specs, epd=epd, fields=DEFAULT_FIELDS, name='design'),
                       file_name='design.zmx', use_container_width=True)
with tc2:
    st.download_button('📄 LDE 文本', specs_to_zemax_text(specs, epd=epd),
                       file_name='design_lde.txt', use_container_width=True)
with tc3:
    if st.button('📐 自动通光', use_container_width=True):
        st.session_state.specs = auto_semi(st.session_state.specs, epd=epd)
        st.session_state.msg = ('ok', '半口径已按通光自动更新')

# ============ 主区 ============
tab_work, tab_cmp, tab_ga, tab_astro = st.tabs(['🛠 设计', '📊 多精英对比', '📈 GA 收敛 + 运行', '🔭 天文工具'])

with tab_work:
    # ---- LDE 全宽 ----
    st.subheader('📋 Lens Data Editor')
    lde_h = st.slider('表格高度 (px)', 320, 900, 560, 20, key='lde_h',
                      help='LDE 表格窗口高度（可调）')
    cur_df = specs_to_df(st.session_state.specs)
    edited = st.data_editor(
        cur_df, num_rows='dynamic', key='lde_table', hide_index=True,
        use_container_width=True, height=lde_h,
        disabled={'Surf': True, '备注': True,
                  'Radius': [0, len(cur_df) - 1],
                  'Thick': [0, len(cur_df) - 1],
                  'Glass': [0, len(cur_df) - 1],
                  'ND': [0, len(cur_df) - 1],
                  'VD': [0, len(cur_df) - 1],
                  'Conic': [0, len(cur_df) - 1],
                  'A4': [0, len(cur_df) - 1],
                  'A6': [0, len(cur_df) - 1],
                  'A8': [0, len(cur_df) - 1]},
        column_config={
            'Surf': st.column_config.NumberColumn('Surf', width='small'),
            'Radius': st.column_config.NumberColumn('Radius', width='medium', format='%.3f'),
            'Thick': st.column_config.NumberColumn('Thick', width='medium', format='%.3f'),
            'Glass': st.column_config.TextColumn('Glass', width='small'),
            'ND': st.column_config.NumberColumn('ND', width='small', format='%.5f'),
            'VD': st.column_config.NumberColumn('VD', width='small', format='%.1f'),
            'Semi-Dia': st.column_config.NumberColumn('Semi-Dia', width='small', format='%.2f'),
            'Conic': st.column_config.NumberColumn('Conic', width='small', format='%.3f',
                                                   help='圆锥常数 k（EVENASPH：0=球面）'),
            'A4': st.column_config.NumberColumn('A4 (ρ⁴)', width='small', format='%.3e',
                                                help='偶次非球面 ρ⁴ 系数（Zemax A4）'),
            'A6': st.column_config.NumberColumn('A6 (ρ⁶)', width='small', format='%.3e',
                                                help='偶次非球面 ρ⁶ 系数（Zemax A6）'),
            'A8': st.column_config.NumberColumn('A8 (ρ⁸)', width='small', format='%.3e',
                                                help='偶次非球面 ρ⁸ 系数（Zemax A8）'),
            '备注': st.column_config.TextColumn('Comment', width='medium'),
        })
    if edited is not None and not edited.equals(cur_df):
        ns = df_to_specs(edited)
        if ns:
            st.session_state.specs = _apply_stop(ns)
    specs = st.session_state.specs
    st.caption('➕ 点击表格底部"添加行"追加表面（玻璃名填入自动带 nd/vd；留空为空气面）；🖱 选中行可删除')

    # ---- 编辑操作 ----
    st.subheader('✏️ 编辑操作')
    _max_op = max(2, len(specs) - 2)
    _op = st.number_input('操作面号', 1, _max_op, 2, step=1, key='op_surf')
    _op = min(int(_op), _max_op)
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        if st.button('➕ 在面后插入', use_container_width=True, key='ins_surf'):
            st.session_state.specs = _apply_stop(insert_surface(st.session_state.specs, _op))
            st.session_state.msg = ('ok', f'已在面 {_op} 后插入表面')
    with oc2:
        if st.button('🗑 删除面', use_container_width=True, key='del_surf'):
            st.session_state.specs = _apply_stop(delete_surface(st.session_state.specs, _op))
            st.session_state.msg = ('ok', f'已删除面 {_op}')
    with oc3:
        st.caption(f'当前 {len(specs)} 面（物面/光阑/像面保护）')

    st.markdown('**➕ 添加镜片**（手动定义 / 从镜片库挑选）')
    _fmax = max(1, len(specs) - 2)
    fpos = st.number_input('插入位置（面 N 后）', 1, _fmax, _fmax,
                           step=1, key='fpos')
    ins_src = st.radio('镜片来源', ['手动定义', '从镜片库挑选'], horizontal=True,
                       key='ins_src')
    if ins_src == '手动定义':
        _gls = sorted(glass_catalog().keys())
        fgl = st.selectbox('玻璃（AGF 实测 nd/vd）', _gls, key='fgl')
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            fr1 = st.number_input('R1 (mm)', -1000.0, 1000.0, 100.0, key='fr1')
        with fc2:
            fr2 = st.number_input('R2 (mm)', -1000.0, 1000.0, -100.0, key='fr2')
        with fc3:
            ft = st.number_input('厚度 (mm)', 0.5, 50.0, 3.0, key='ft')
        fa1, fa2 = st.columns(2)
        with fa1:
            fa_t2 = st.number_input('后空气 (mm)', 0.5, 60.0, 5.0, key='fa_t2')
        with fa2:
            fa_semi = st.number_input('半口径 (mm)', 2.0, 100.0, 15.0, key='fa_semi')
        if st.button('插入镜片', use_container_width=True, key='add_lens'):
            st.session_state.specs = _apply_stop(add_lens_group(
                st.session_state.specs, int(fpos), fgl, fr1, fr2, ft,
                t2=float(fa_t2), semi=float(fa_semi)))
            st.session_state.msg = ('ok', f'已在面 {int(fpos)} 后添加 {fgl} 镜片')
    else:
        lc1, lc2 = st.columns(2)
        with lc1:
            ins_lt = st.selectbox('库类型', [1, 2, 3, 4, 5, 6],
                                  format_func=lambda t: f'库{t}（{"单镜片" if t <= 4 else "双胶合"}）',
                                  key='ins_lt')
        with lc2:
            _lo, _hi, _cnt = library_rows(int(ins_lt))
            ins_row = st.number_input(f'行号（有效 {_lo}-{_hi}）', _lo or 1,
                                      max(_hi, _lo or 1), _lo or 1, step=1, key='ins_row')
        _prev = library_pick(int(ins_lt), int(ins_row))
        if _prev:
            st.caption(f'🔍 预览[{_prev["type"]}]: {_prev["text"]}')
        else:
            st.caption(f'⚠️ 行 {ins_row} 无有效镜片（库{ins_lt} 有效行 {_lo}-{_hi}，共 {_cnt} 个）')
        lb1, lb2 = st.columns(2)
        with lb1:
            ins_t2 = st.number_input('后空气 (mm)', 0.5, 60.0, 5.0, key='ins_t2')
        with lb2:
            ins_semi = st.number_input('半口径 (mm)', 2.0, 100.0, 15.0, key='ins_semi')
        if st.button('插入库镜片', use_container_width=True, key='ins_lib'):
            _n0 = len(st.session_state.specs)
            _ns = library_lens_group(st.session_state.specs, int(fpos),
                                     int(ins_lt), int(ins_row),
                                     t2=float(ins_t2), semi=float(ins_semi))
            if len(_ns) == _n0:
                st.session_state.msg = ('err', f'库{ins_lt} 行{ins_row} 无效，未插入')
            else:
                st.session_state.specs = _apply_stop(_ns)
                st.session_state.msg = ('ok', f'已从库{ins_lt} 行{ins_row} 插入镜片（面 {int(fpos)} 后）')
            st.session_state.pop('lde_table', None)

    # ---- 📚 镜头样板（经典 / 现代 / 常用构型）----
    st.divider()
    with st.expander('📚 镜头样板（经典 / 现代 / 常用构型）'):
        st.caption('一键载入经典镜头构型：① 镜片库凑（结构近似，每次随机组合，可多点几次换组合）'
                   '② 完全复刻（样板示例参数，f=100mm 基准自动缩放 + 玻璃自动匹配内置 AGF 表）')
        k1, k2 = st.columns([2, 1])
        with k1:
            _tpl = st.selectbox('构型', list(TEMPLATES.keys()),
                                format_func=lambda k: ('🪞 ' if TEMPLATES[k].get('reflective') else '🔭 ')
                                + TEMPLATES[k]['label'], key='tpl_sel')
        with k2:
            _tpl_refl = bool(TEMPLATES[_tpl].get('reflective'))
            _tpl_opts = ['② 完全复刻'] if _tpl_refl else ['① 镜片库凑', '② 完全复刻']
            _tpl_mode = st.radio('生成模式', _tpl_opts, horizontal=True, key='tpl_mode',
                                 help=('🪞 反射构型：镜片库为折射玻璃，仅支持完全复刻'
                                       if _tpl_refl else '① 从镜片库凑出结构近似；② 按样板参数直接生成'))
        st.caption(('🪞 ' if _tpl_refl else '💡 ') + TEMPLATES[_tpl]['desc']
                   + ('｜反射构型仅支持完全复刻' if _tpl_refl else ''))
        p1, p2, p3 = st.columns(3)
        with p1:
            _tpl_f = st.number_input('目标焦距 (mm)', 50.0, 500.0, 200.0, step=10.0, key='tpl_f')
        with p2:
            _tpl_bf = st.number_input('后焦 (mm)', 10.0, 120.0, 55.0, step=1.0, key='tpl_bf')
        with p3:
            _tpl_epd = st.number_input('口径 EPD (mm)', 10.0, 120.0, 40.0, step=1.0, key='tpl_epd')
        if st.button('✨ 生成样板镜头（载入主编辑区）', use_container_width=True, key='tpl_go'):
            _tpl_specs = build_template(_tpl,
                                        mode='library' if _tpl_mode == '① 镜片库凑' else 'replica',
                                        f_mm=float(_tpl_f), back_focus=float(_tpl_bf))
            if _tpl_specs:
                st.session_state.pop('lde_table', None)
                st.session_state.pop('bf', None)
                st.session_state.specs = _apply_stop(_tpl_specs)
                st.session_state.epd = float(_tpl_epd)
                st.session_state.src_label = f'样板: {TEMPLATES[_tpl]["label"]}'
                st.session_state.msg = ('ok', f'已载入样板 {TEMPLATES[_tpl]["label"]}'
                                       f'（{len(_tpl_specs)} 面 · {_tpl_mode}）——可继续编辑/优化')
            else:
                st.session_state.msg = ('err', '样板生成失败——库中无匹配镜片，可切换"② 完全复刻"模式')
        st.caption('复刻模式参数为经典结构示意值（结构/玻璃类型正确）；载入后用 ⚡ 优化 + 模拟星场验证像质')

    # ---- 优化（Zemax Optimize 独立菜单 → 工作区）----
    with st.expander('⚡ 优化（局部）'):
        _mv = evaluate_specs(specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui)
        st.caption(f'当前: EFFL {_mv["efl"]:.1f}（目标 200）| AXCL {_mv["axcl"]:+.3f}（目标 0.05）'
                   f' | RSCE {_mv["rsce_um"]:.0f} µm')
        st.caption('变量：全部空气间隔 + 后焦；可选厚度/曲率 | Nelder-Mead')
        opt_fast = st.checkbox('快速模式（近轴，~10s）', value=False, key='opt_fast',
                               help='不含 RSCE；标准模式含星点 1-3 分钟')
        opt_thick = st.checkbox('包含镜片厚度', value=False, key='opt_thick',
                                help='玻璃面厚度也作为变量')
        opt_curv = st.checkbox('包含曲率', value=False, key='opt_curv',
                               help='玻璃面曲率也作为变量（同号 ±50% 边界）')
        if st.button('开始优化', use_container_width=True, key='opt_go'):
            with st.spinner('优化中（快速 ~10s / 标准 1-3 分钟）...'):
                ns, _h, _b, _a = optimize_local(
                    specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui,
                    include_rsce=not opt_fast,
                    vars_thick=opt_thick, vars_curv=opt_curv)
                if _a < _b:
                    st.session_state.specs = _apply_stop(ns)
                    specs = ns
                    st.session_state.msg = ('ok', f'优化完成 MFE {_b:.4f} → {_a:.4f}（改善 {100*(_b-_a)/_b:.0f}%）')
                else:
                    st.session_state.msg = ('warn', f'优化未改善（{_b:.4f} → {_a:.4f}），保持原结构')
                if _h:
                    st.line_chart(pd.DataFrame({'MFE 历史': _h}))

    # ---- 分析视图（segmented 窗口切换，缓存图避免重算）----
    st.divider()
    if hasattr(st, 'segmented_control'):
        view = st.segmented_control('分析视图', ['2D Layout', '3D', 'Spot', '像差分析', '总览'],
                                    default='2D Layout', selection_mode='single')
    else:
        view = st.radio('分析视图', ['2D Layout', '3D', 'Spot', '像差分析', '总览'], horizontal=True)

    def _dl_png(fig, label, fname):
        """分析图下载 PNG"""
        import io as _io
        _buf = _io.BytesIO()
        fig.savefig(_buf, format='png', dpi=100)
        _buf.seek(0)
        st.download_button(f'⬇ {label}', _buf, file_name=fname, mime='image/png')

    def _cache_fig(key, maker):
        """按结构+参数缓存图：切换视图/重复交互不重算，结构变化自动失效
        返回前按全局"图窗口尺寸"滑块缩放（原始尺寸记录在缓存，缩放不累积）"""
        if st.session_state.get('fig_cache_key') != key:
            st.session_state.fig_cache_key = key
            st.session_state.fig_cache_val = maker()
            st.session_state.fig_cache_size = None
        fig = st.session_state.fig_cache_val
        if fig is not None:
            try:
                _fig0 = fig[0] if isinstance(fig, tuple) else fig
                if st.session_state.get('fig_cache_size') is None:
                    st.session_state.fig_cache_size = _fig0.get_size_inches()
                _w, _h = st.session_state.fig_cache_size
                _s = float(st.session_state.get('fig_scale', 1.0))
                if abs(_s - 1.0) > 1e-3:
                    _fig0.set_size_inches(_w * _s, _h * _s)
                    _fig0.tight_layout()
            except Exception:
                pass
        return fig

    _skey = (tuple((s['idx'], s['R'], s['t'], s['glass'], s['semi']) for s in specs),
             epd, tuple(fields_ui), tuple(wavs_ui))
    if view == '2D Layout':
        st.subheader('🔭 2D Layout（多波长 F/d/C + 焦点标记）')
        with st.spinner(''):
            _fig2 = _cache_fig(('2d', _skey), lambda: plot_layout(
                specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
        _dl_png(_fig2, '下载 2D Layout', 'layout_2d.png')
        st.pyplot(_fig2)
    elif view == '3D':
        st.subheader('🧊 3D Layout（旋转体 + 光线）')
        with st.spinner(''):
            _fig3 = _cache_fig(('3d', _skey), lambda: plot_layout_3d(
                specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
        if _fig3 is not None:
            _dl_png(_fig3, '下载 3D Layout', 'layout_3d.png')
            st.pyplot(_fig3)
        else:
            st.warning('3D Layout 生成失败')
    elif view == 'Spot':
        st.subheader('⭕ Spot Diagram（optiland 真实追迹，6 环）')
        fig_s, rms = _cache_fig(('spot', _skey), lambda: compute_spot(
            specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
        if fig_s is not None:
            _dl_png(fig_s, '下载 Spot', 'spot.png')
            st.pyplot(fig_s)
            rdf = pd.DataFrame(rms, columns=['视场 (norm)', 'RMS max (µm)', 'RMS 主波长 (µm)'])
            st.dataframe(rdf, use_container_width=True, hide_index=True)
        else:
            st.warning('点阵图追迹失败（结构可能无效）')
    elif view == '总览':
        st.subheader('🪟 总览（2×2 多窗口 · Zemax 平铺风格）')
        st.caption('四个分析窗口并行平铺；每个窗口右下角可拖拽缩放；"图窗口尺寸"滑块整体缩放')
        o1, o2 = st.columns(2)
        with o1:
            with st.spinner(''):
                _fo2 = _cache_fig(('2d', _skey), lambda: plot_layout(
                    specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
            if _fo2 is not None:
                st.caption('🔭 2D Layout')
                st.pyplot(_fo2)
        with o2:
            with st.spinner(''):
                _fos, _rms_o = _cache_fig(('spot', _skey), lambda: compute_spot(
                    specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
            if _fos is not None:
                st.caption('⭕ Spot Diagram')
                st.pyplot(_fos)
        o3, o4 = st.columns(2)
        with o3:
            with st.spinner(''):
                _fom = _cache_fig(('ana', 'mtf', _skey), lambda: analysis_fig(
                    specs, kind='mtf', epd=epd, fields=fields_ui, wavelengths=wavs_ui))
            if _fom is not None:
                st.caption('📈 MTF vs 空间频率')
                st.pyplot(_fom)
        with o4:
            with st.spinner(''):
                _fof = _cache_fig(('ana', 'field_curvature', _skey), lambda: analysis_fig(
                    specs, kind='field_curvature', epd=epd, fields=fields_ui, wavelengths=wavs_ui))
            if _fof is not None:
                st.caption('📉 场曲 Field Curvature')
                st.pyplot(_fof)
    else:
        st.subheader('📉 像差分析')
        ak = st.selectbox('分析类型', list(KINDS.keys()), format_func=lambda k: KINDS[k])
        with st.spinner(''):
            _ana = _cache_fig(('ana', ak, _skey), lambda: analysis_fig(
                specs, kind=ak, epd=epd, fields=fields_ui, wavelengths=wavs_ui))
        if _ana is not None:
            _dl_png(_ana, '下载像差图', 'aberration.png')
            st.pyplot(_ana)
        else:
            st.warning('该分析类型计算失败（结构可能无效）')
        with st.expander('🌈 轴向色差（APO）'):
            with st.spinner(''):
                fig_ac = axial_color_fig(specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui)
            if fig_ac:
                st.pyplot(fig_ac)
            else:
                st.info('色差曲线不可用')
    with st.expander('🔬 玻璃库（AGF nd/vd）'):
        cat = glass_catalog()
        gdf = pd.DataFrame([{'玻璃': k, 'nd': v[0], 'vd': v[1]} for k, v in sorted(cat.items())])
        st.dataframe(gdf, use_container_width=True, hide_index=True)
        st.caption(f'共 {len(cat)} 种玻璃')
        st.markdown('**🔎 玻璃搜索**（按 nd/vd 找最接近玻璃）')
        gc1, gc2, gc3 = st.columns([2, 2, 1])
        with gc1:
            _g_nd = st.number_input('目标 nd', 1.3, 2.2, 1.5163, step=0.005,
                                    format='%.4f', key='g_nd')
        with gc2:
            _g_vd = st.number_input('目标 vd', 15.0, 95.0, 64.1, step=0.5,
                                    key='g_vd')
        with gc3:
            if st.button('搜索', use_container_width=True, key='g_search'):
                st.session_state.g_res = search_glass(float(_g_nd), float(_g_vd))
        if 'g_res' in st.session_state and st.session_state.g_res:
            rdf = pd.DataFrame(
                [{'玻璃': g, 'nd': round(n, 5), 'vd': round(v, 2), '距离': round(s, 2)}
                 for g, n, v, s in st.session_state.g_res])
            st.dataframe(rdf, use_container_width=True, hide_index=True)

with tab_cmp:
    st.subheader('多精英对比')
    cc1, cc2, cc3 = st.columns([3, 2, 1])
    with cc1:
        cmp_file = st.text_input('对比精英文件', DEFAULT_FILE)
    with cc2:
        cmp_idxs = st.text_input('精英序号（逗号分隔）', '55,1,2')
    with cc3:
        if st.button('载入对比', use_container_width=True):
            try:
                idxs = [int(x.strip()) for x in cmp_idxs.split(',') if x.strip()]
                items = load_multi(os.path.join(ELITES_DIR, cmp_file), idxs)
                st.session_state.cmp_items = items
                st.session_state.msg = ('ok', f'对比载入 {len(items)} 个精英')
            except Exception as e:
                st.session_state.msg = ('err', f'序号解析失败: {e}')
    if 'cmp_items' in st.session_state and st.session_state.cmp_items:
        items = st.session_state.cmp_items
        st.dataframe(compare_table(items, epd=epd), use_container_width=True, hide_index=True)
        cols = st.columns(len(items))
        for col, it in zip(cols, items):
            with col:
                st.caption(f'精英 {it["label"]}')
                st.pyplot(plot_layout(it['specs'], epd=epd, figsize=(5, 2.6)))
        labels = [f'精英 {it["label"]}' for it in items]
        pick = st.selectbox('载入主编辑区', labels)
        if st.button('载入所选'):
            it = items[labels.index(pick)]
            st.session_state.specs = it['specs']
            st.session_state.src_label = f'{cmp_file} {it["label"]}'
            st.session_state.msg = ('ok', f'已载入 {it["label"]}')
    else:
        st.info('输入精英文件 + 序号（如 55,1,2），点击"载入对比"')

with tab_ga:
    st.subheader('📈 GA 收敛 + 运行状态')
    st.caption('运行中的 GA 每 log_every 代实时写入 ga_history.csv；本区每 5 秒自动刷新。')

    @st.fragment(run_every=5)
    def _ga_curve():
        _info = run_control.get_run_info()
        st.markdown('**⏱ 运行状态**')
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric('状态', '⏳ 运行中' if _info['running'] else '⏸ 空闲')
        c2.metric('PID', str(_info['pid']) if _info['pid'] else '—')
        if _info['started_at'] and _info['elapsed_s']:
            _es = int(_info['elapsed_s'])
            c3.metric('已运行', f'{_es // 60} 分 {_es % 60} 秒')
        else:
            c3.metric('已运行', '—')
        c4.metric('收敛数据点', str(_info['data_points']))
        c5.metric('最新 best', f'{_info["latest_best"]:.4f}' if _info['latest_best'] is not None else '—')
        if _info['gen_str']:
            st.caption(f'日志进度：第 {_info["gen_str"]} 代（若日志格式支持）')
        hists = run_control.list_history_csv()
        if hists:
            sel = st.selectbox('收敛曲线（results/…/ga_history.csv）', hists, key='ga_hist_sel')
            res = run_control.read_history_csv(sel)
            if res:
                g, b = res
                _df = pd.DataFrame({'gen': g, 'best_mfe': b})
                _df['最优保持'] = _df['best_mfe'].cummin()
                _df['平滑(5代)'] = _df['best_mfe'].rolling(5, min_periods=1).mean()
                st.line_chart(_df.set_index('gen'), height=340)
                st.caption(f'共 {len(g)} 个数据点 | best {b[-1]:.4f}（第 {g[-1]:.0f} 代）'
                           f' | 相对首代提升 {b[0] / b[-1]:.1f}×'
                           + (' | ⏳ 仍在运行' if _info['running'] else ''))
                with st.expander('📋 最近 20 代明细'):
                    st.dataframe(_df.tail(20).round(4), use_container_width=True,
                                 hide_index=True)
            else:
                st.info('该 CSV 暂无数据')
        else:
            st.info('暂无 ga_history.csv——先在侧边栏"🚀 运行"页跑一次 GA（未配置 GA 环境时不可用）')
        log_path = _info['log'] if _info['log'] else st.session_state.get('panel_log')
        if log_path and os.path.exists(log_path):
            with st.expander('任务日志尾部'):
                st.code(run_control.tail_log(log_path, 100), language='text')

    _ga_curve()

# ============ 🔭 天文工具 ============
with tab_astro:
    st.subheader('🔭 天文摄影工具')
    st.caption('深空摄影光学计算（基于当前镜头实时数据 + 标准天文公式）')

    _am = evaluate_specs(specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui)
    _af = float(_am['efl']) if np.isfinite(_am['efl']) else 200.0
    _arsce = float(_am['rsce_um']) if np.isfinite(_am['rsce_um']) else 0.0

    # ---- 1. 成像系统计算器 ----
    with st.expander('📷 成像系统计算器', expanded=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            _sens = st.selectbox('传感器', ['全画幅 36×24mm', 'APS-C 23.5×15.6mm',
                                           'M4/3 17.3×13mm', '1″ 13.2×8.8mm', '自定义'],
                                 key='astro_sens')
            _sdim = {'全画幅 36×24mm': (36.0, 24.0), 'APS-C 23.5×15.6mm': (23.5, 15.6),
                     'M4/3 17.3×13mm': (17.3, 13.0), '1″ 13.2×8.8mm': (13.2, 8.8),
                     '自定义': (36.0, 24.0)}[_sens]
        with s2:
            _px = st.number_input('像元尺寸 (µm)', 1.0, 20.0, 3.76, step=0.01, key='astro_px')
        with s3:
            _fuse = st.number_input('焦距 (mm，默认=当前镜头)', 10.0, 3000.0,
                                    float(_af), step=1.0, key='astro_f')
        _fno = _fuse / epd if epd > 0 else 0.0
        _ps = pixel_scale(_fuse, _px)
        _fw, _fh = fov(*_sdim, _fuse)
        _dl = diffraction_limit(epd)
        _airy = airy_diameter(_fno)
        _fwhm = fwhm_arcsec(_arsce, _fuse)
        _samp = sampling_ratio(_fwhm, _ps)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('像元尺度', f'{_ps:.2f}″/px')
        m2.metric('视场', f'{_fw:.2f}° × {_fh:.2f}°')
        m3.metric('衍射极限', f'{_dl:.2f}″')
        m4.metric('艾里斑直径', f'{_airy:.1f} µm')
        m5, m6, m7, m8 = st.columns(4)
        m5.metric('星点 FWHM（当前设计）', f'{_fwhm:.2f}″',
                  help='≈2.355×RSCE（全视场最大 RMS）；中心星点以 Spot 图场 0 为准')
        m6.metric('采样率 FWHM/px', f'{_samp:.1f}' if np.isfinite(_samp) else '—')
        m7.metric('极限星等', f'{limit_magnitude(epd):.1f} 等')
        m8.metric('集光力', f'{light_gain(epd):.0f}× 人眼')
        if np.isfinite(_samp):
            if _samp < 1.5:
                st.warning(f'⚠️ 欠采样：FWHM {_fwhm:.2f}″ < 1.5 像元（{_ps:.2f}″/px）——星点细节丢失，建议更小像元或增倍镜')
            elif _samp > 4.0:
                st.info(f'💡 过采样：FWHM 跨 {_samp:.1f} 像元——可考虑减焦镜提升视场/信噪比')
            else:
                st.success('✅ 采样合理（1.5-4 像元/FWHM，Nyquist 附近）')
        st.caption(f'参考：衍射极限 138.4/D = 138.4/{epd:.0f}mm；艾里斑 2.44λF#；极限星等按人眼 7mm 基准 6.5 等近似')

    # ---- 3. 月面 / 行星像比例 ----
    with st.expander('🪐 月面 / 行星像比例'):
        st.caption('天体像直径 = f·tan(角直径)；行星按冲日典型值——规划增倍镜 / 行星相机像元用')
        b1, b2 = st.columns(2)
        with b1:
            _bd = st.selectbox('天体', list(_BODY_DEG.keys()), key='astro_body')
        with b2:
            _fb = st.number_input('焦距 (mm)', 10.0, 3000.0, float(_af), key='astro_fb')
        _diam = image_scale(_fb, _BODY_DEG[_bd])
        _pxn = pixels_on_body(_diam, _px)
        q1, q2, q3 = st.columns(3)
        q1.metric('像直径', f'{_diam:.2f} mm')
        q2.metric(f'占像素（{_px:.2f}µm 像元）', f'{_pxn:.0f} px')
        q3.metric('占画幅宽（36mm）', f'{_diam / 36 * 100:.1f}%')
        st.caption(f'参考：{_bd} 角直径 {_BODY_DEG[_bd] * 3600:.0f}″；'
                   '月亮 31′ ≈ 0.517°——200mm 下满月像直径约 1.8mm（全画幅的 5%）')

    # ---- 4. 曝光 / 极限星等（SNR 光子统计）----
    with st.expander('🌌 曝光 / 极限星等（SNR 模型）'):
        st.caption('550nm 宽带光子统计：SNR = S/√(S + 天光 + 读出² + 暗电流·t)；'
                   '天光主导时极限星等 ≈ 天光亮度限制（暗空 100mm/120s ≈ 18.9 等）')
        e1, e2, e3 = st.columns(3)
        with e1:
            _em = st.number_input('目标星等 m', 8.0, 22.0, 15.0, step=0.5, key='astro_m')
            _et = st.number_input('单张曝光 (s)', 10.0, 1200.0, 120.0, step=10.0, key='astro_t')
        with e2:
            _eq = st.number_input('量子效率 QE', 0.2, 0.95, 0.6, step=0.05, key='astro_qe')
            _sky = st.selectbox('天光（mag/″²）', list(_SKY_MAG.keys()), key='astro_sky')
        with e3:
            _er = st.number_input('读出噪声 (e⁻)', 1.0, 15.0, 3.0, step=0.5, key='astro_r')
            _ed = st.number_input('暗电流 (e⁻/s/px)', 0.0, 1.0, 0.05, step=0.01, key='astro_d')
            _nfr = st.number_input('叠加张数', 1, 300, 20, step=1, key='astro_nfr')
        _snr_v = snr(epd, _em, _et, _eq, _SKY_MAG[_sky], _ps, _er, _ed)
        _mlim = limiting_magnitude(epd, _et, _eq, _SKY_MAG[_sky], _ps, _er, _ed)
        _mlim_n = stacked_limit(_mlim, _nfr)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(f'{_em:.1f} 等星 SNR', f'{_snr_v:.0f}')
        r2.metric('单张极限（SNR=10）', f'{_mlim:.1f} 等')
        r3.metric(f'{_nfr} 张叠加极限', f'{_mlim_n:.1f} 等')
        r4.metric('像元尺度', f'{_ps:.2f}″/px')
        if _snr_v < 10:
            st.warning('⚠️ SNR < 10：目标太暗或曝光不足——加曝光/叠加/减噪')
        elif _snr_v < 50:
            st.info('💡 SNR 10-50：可探测但噪声明显，建议增加叠加')
        else:
            st.success('✅ SNR ≥ 50：信噪比充足')

    # ---- 2. 星点 / 视宁度 / 焦深 ----
    with st.expander('🔭 星点 / 视宁度 / 焦深'):
        _see = st.slider('视宁度 seeing (″)', 0.5, 6.0, 2.5, 0.1, key='astro_see')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('衍射极限', f'{_dl:.2f}″')
        with c2:
            st.metric('视宁度', f'{_see:.1f}″')
        with c3:
            st.metric('有效星点下限', f'{max(_dl, _see):.2f}″')
        if _dl < _see:
            st.info(f'💡 口径 {epd:.0f}mm 衍射极限 {_dl:.2f}″ < 视宁度 {_see:.1f}″——像质受视宁度限制，加大口径增益有限')
        else:
            st.success(f'✅ 口径 {epd:.0f}mm 衍射极限 {_dl:.2f}″ ≥ 视宁度——口径可发挥全部潜力')
        _fdepth = critical_focus_depth(_fno)
        _dt = st.slider('温度漂移 ΔT (K)', 0, 40, 10, key='astro_dt')
        _tlen = sum(float(s['t']) for s in specs[:-1]
                    if np.isfinite(s['t']) and s['t'] != float('inf'))
        _th = thermal_shift(_tlen, _dt)
        d1, d2 = st.columns(2)
        with d1:
            st.metric('临界焦深 ±2λF#²', f'±{_fdepth * 1000:.0f} µm',
                      help='对焦精度需求（超过则星点明显变大）')
        with d2:
            st.metric(f'铝筒热漂移（ΔT={_dt}K）', f'{_th * 1000:.0f} µm',
                      help='α=23e-6/K × 筒长 × ΔT')
        if _th > _fdepth:
            st.warning(f'⚠️ 温漂 {_th * 1000:.0f}µm 超过焦深 {_fdepth * 1000:.0f}µm——建议电动调焦/温补')

    # ---- 3. 配件计算 ----
    with st.expander('🛠 配件计算（滤镜 / 减焦 / 卡口）'):
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('**🧪 滤镜 / 平窗焦点位移**')
            _ft = st.number_input('平板厚度 (mm)', 0.5, 10.0, 2.0, step=0.1, key='astro_ft')
            _fn = st.number_input('折射率 n', 1.4, 1.9, 1.52, step=0.01, key='astro_fn')
            _psh = plate_shift(_ft, _fn)
            st.metric('焦点后移', f'{_psh:.2f} mm',
                      help='安装滤镜后焦点向像面移动 t(1-1/n)，需重新对焦')
        with f2:
            st.markdown('**🔧 减焦 / 增倍镜**')
            _k = st.number_input('系数（<1 减焦，>1 增倍）', 0.5, 2.0, 0.8, step=0.05, key='astro_k')
            _rf, _rn, _rk = reducer_effect(_fuse, _fno, _k)
            st.metric('合成系统', f'焦距 {_rf:.0f}mm | F{_rn:.2f}',
                      help=f'FOV ×{_rk:.2f}（{"增大" if _k < 1 else "缩小"}）')
        st.markdown('**📷 相机卡口法兰距参考**')
        _cam = st.selectbox('相机', ['天文相机 T 环（55mm）', '佳能 EF（44mm）',
                                    '索尼 E（18mm）', '尼康 Z（16mm）', '佳能 RF（20mm）'],
                            key='astro_cam')
        _fl = {'天文相机 T 环（55mm）': 55.0, '佳能 EF（44mm）': 44.0, '索尼 E（18mm）': 18.0,
               '尼康 Z（16mm）': 16.0, '佳能 RF（20mm）': 20.0}[_cam]
        _bfl_cur = float(specs[-2]['t']) if len(specs) > 2 and np.isfinite(specs[-2]['t']) else float('nan')
        if np.isfinite(_bfl_cur):
            st.caption(f'当前镜头后焦 BFL = {_bfl_cur:.1f}mm；{_cam.split("（")[0]} 法兰距 {_fl:.0f}mm'
                       + (' —— ✅ 匹配（后焦 ≥ 法兰距，可直连）' if _bfl_cur >= _fl
                          else f' —— ⚠️ 后焦差 {_fl - _bfl_cur:.1f}mm，需转接环/延长筒'))

    # ---- 6. 平场镜（减焦镜）薄透镜设计 ----
    with st.expander('🔧 平场镜（减焦镜）薄透镜设计'):
        st.caption('薄透镜组合：1/f = 1/f₁ + 1/f₂ − d/(f₁f₂)；真实减焦镜是双胶合组——先定规格，再用 LDE 设计验证')
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('**正向：主镜 + 减焦镜 → 合成系统**')
            _f2 = st.number_input('减焦镜焦距 f₂ (mm)', 50.0, 3000.0, 300.0, key='astro_f2')
            _dd = st.number_input('间距 d (mm，主镜→减焦镜)', 5.0, 300.0, 40.0, key='astro_dd')
            _fc = combined_focal(_fuse, _f2, _dd)
            _rt = reducer_design(_fuse, _f2, _dd)
            st.metric('合成焦距', f'{_fc:.0f} mm' if np.isfinite(_fc) else '∞')
            st.metric('减焦率', f'{_rt:.2f}×' if np.isfinite(_rt) else '—')
        with g2:
            st.markdown('**反向：目标减焦率 → 所需 f₂（设计规格）**')
            _rtg = st.number_input('目标减焦率', 0.5, 1.0, 0.8, step=0.05, key='astro_rtg')
            _ddg = st.number_input('间距 d (mm)', 5.0, 300.0, 40.0, key='astro_ddg')
            _f2g = reducer_focal_for(_fuse, _rtg, _ddg)
            st.metric('所需减焦镜焦距 f₂', f'{_f2g:.0f} mm' if np.isfinite(_f2g) else '—')
            st.metric('合成 F#', f'{_fc / epd:.2f}' if np.isfinite(_fc) and epd > 0 else '—')
        st.caption('提示：间距 d 越大减焦越强；f₂ 正=会聚组（减焦），负=发散组（增倍镜方向）')

    # ---- 7. 模拟星场（像差可视化）----
    with st.expander('🌠 模拟星场（像差可视化）'):
        st.caption('真实光线追迹：每颗星 = 该视场点列图（三波长 F/d/C 叠加）。'
                   '中心圆点=球差/色差｜边缘彗星尾=彗差｜四角椭圆=像散｜星点彩边=横向色差')
        sf1, sf2, sf3 = st.columns(3)
        with sf1:
            _smode = st.radio('模式', ['网格演示', '随机星空'], horizontal=True, key='astro_smode')
            _sann = st.checkbox('标注 RMS (µm)', value=False, key='astro_sann')
        with sf2:
            _sn = st.slider('星数', 9, 81, 25, step=4, key='astro_sn')
            _ss = st.slider('像差放大', 3, 60, 15, step=3, key='astro_ss')
        with sf3:
            _seed = st.number_input('随机种子', 1, 999, 42, key='astro_seed')
            _sd = st.slider('离焦（像面偏移 µm）', -500, 500, 0, 50, key='astro_sd')
        _sf_key = (('starfield', _smode, _sn, _ss, _seed, _sd, _sann) + _skey)
        with st.spinner('追迹渲染中（网格 25 星约 2s）...'):
            _sfig = _cache_fig(_sf_key, lambda: render_starfield(
                specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui,
                mode='grid' if _smode == '网格演示' else 'random',
                n_stars=int(_sn), scale=float(_ss), seed=int(_seed),
                defocus_mm=float(_sd) / 1000.0, annotate=bool(_sann)))
        if _sfig is not None:
            _dl_png(_sfig, '下载星场', 'starfield.png')
            st.pyplot(_sfig)
        else:
            st.warning('星场渲染失败（结构可能无效）')
        if st.checkbox('🎞 离焦对比（5 帧：-400/-200/0/+200/+400 µm）', value=False,
                       key='astro_sfcmp'):
            _dzs = [-0.4, -0.2, 0.0, 0.2, 0.4]
            cols = st.columns(5)
            for ci, dz in enumerate(_dzs):
                with cols[ci]:
                    st.caption(f'{dz * 1000:+.0f} µm')
                    with st.spinner(''):
                        _cf = _cache_fig(
                            ('starfield', 'cmp', ci, _smode, _sn, _ss, _seed) + _skey,
                            lambda dz=dz: render_starfield(
                                specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui,
                                mode='grid' if _smode == '网格演示' else 'random',
                                n_stars=int(_sn), scale=float(_ss), seed=int(_seed),
                                defocus_mm=dz, annotate=False))
                    if _cf is not None:
                        st.pyplot(_cf)
        if st.button('🎯 测算最佳对焦位置（扫描 ±0.4mm）', use_container_width=True,
                     key='astro_bf'):
            with st.spinner('扫描中（21 步 × 4 星）...'):
                _bz, _bc = best_focus_offset(specs, epd=epd, fields=fields_ui,
                                             wavelengths=wavs_ui)
                st.session_state.sf_bz = _bz
        if 'sf_bz' in st.session_state and st.session_state.sf_bz is not None:
            _bzv = float(st.session_state.sf_bz)
            if abs(_bzv) <= 0.05:
                st.success(f'🎯 最佳对焦 ≈ {_bzv * 1000:+.0f} µm——当前像面在最佳焦点附近')
            else:
                st.warning(f'🎯 最佳对焦 = {_bzv * 1000:+.0f} µm（相对当前像面；'
                           '负=像面需前移）——请微调后焦后星点最小')

    # ---- 8. 曝光模拟照片（单张 vs 叠加）----
    with st.expander('📷 曝光模拟照片（单张 vs 叠加）'):
        st.caption('像素级噪声模拟：每颗星真实追迹 RMS → 高斯 PSF；噪声 = 天光 + 读出 + 暗电流。'
                   '叠加 N 张噪声降 √N、极限星等 +2.5log√N——暗星从噪声中浮现')
        x1, x2, x3 = st.columns(3)
        with x1:
            _ex_t = st.number_input('单张曝光 (s)', 10.0, 1200.0, 120.0, step=10.0, key='astro_ext')
            _ex_n = st.number_input('叠加张数', 1, 300, 20, step=1, key='astro_exn')
        with x2:
            _ex_sky = st.selectbox('天光', list(_SKY_MAG.keys()), key='astro_exsky')
            _ex_qe = st.number_input('QE', 0.2, 0.95, 0.6, step=0.05, key='astro_exqe')
        with x3:
            _ex_zoom = st.radio('显示区域', ['整幅', '中心 3×'], horizontal=True, key='astro_exzoom')
            _ex_seed = st.number_input('随机种子', 1, 999, 3, key='astro_exseed')
        _ex_tgt = st.radio('天空目标', ['纯星空', 'M42 猎户座星云（合成）'],
                           horizontal=True, key='astro_extgt')
        _ex_key = (('exposure', _ex_t, _ex_n, _ex_sky, _ex_qe, _ex_zoom, _ex_seed,
                    _ex_tgt) + _skey)
        with st.spinner('模拟曝光渲染中（约 2-4s）...'):
            _exfig = _cache_fig(_ex_key, lambda: render_exposure_stars(
                specs, epd=epd, fields=fields_ui, wavelengths=wavs_ui,
                t_sec=float(_ex_t), n_stack=int(_ex_n),
                sky_mag=_SKY_MAG[_ex_sky], qe=float(_ex_qe),
                n_stars=80, seed=int(_ex_seed),
                zoom=3 if _ex_zoom == '中心 3×' else 1,
                target='m42' if _ex_tgt == 'M42 猎户座星云（合成）' else None))
        if _exfig is not None:
            _dl_png(_exfig, '下载曝光模拟', 'exposure.png')
            st.pyplot(_exfig)
        else:
            st.warning('曝光模拟失败（结构可能无效）')

_wl_txt = ' / '.join(f'{w:.4f}' for w, _ in wavs_ui)
_fld_txt = ' / '.join(f'{y:.2f}°' for y, _ in fields_ui)
_total = sum(float(s['t']) for s in specs[:-1]
             if np.isfinite(s['t']) and s['t'] != float('inf'))
st.divider()
_pw_idx = int(st.session_state.get('primary_wl', 0)) + 1
_od_txt = '∞' if not np.isfinite(specs[0]['t']) else f'{specs[0]["t"]:.0f}'
_fno_txt = ('—' if _has_mirror else (f"{epd / m['efl']:.2f}" if np.isfinite(m['efl']) and m['efl'] > 0 else '—'))
st.caption(f'📊 {epd:.1f} mm | F/# {_fno_txt} | 波长 {_wl_txt} µm | 视场 {_fld_txt} | 主波长 λ{_pw_idx} | '
           f'视场类型 {st.session_state.get("field_type", "angle")} | 物距 {_od_txt} | {len(specs)} 面 | 总长 {_total:.1f} mm')
st.caption(f'镜头设计工作台 {CFG.VERSION} | 自包含数据: data/（镜片库 + 内置玻璃表，无外部依赖）')
