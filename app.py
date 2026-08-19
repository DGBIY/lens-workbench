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
            with st.expander('📊 运行状态'):
                _run, _log, _pid = run_control.is_running()
                st.caption(f'状态: {"⏳ 运行中 (PID " + str(_pid) + ")" if _run else "⏸ 空闲"}')

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
mc1.metric('近轴 EFFL (mm)', f"{m['efl']:.1f}" if np.isfinite(m['efl']) else '—')
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
tab_work, tab_cmp, tab_ga = st.tabs(['🛠 设计', '📊 多精英对比', '📈 GA 收敛 + 运行'])

with tab_work:
    # ---- LDE 全宽 ----
    st.subheader('📋 Lens Data Editor')
    cur_df = specs_to_df(st.session_state.specs)
    edited = st.data_editor(
        cur_df, num_rows='dynamic', key='lde_table', hide_index=True,
        use_container_width=True,
        disabled={'Surf': True, '备注': True,
                  'Radius': [0, len(cur_df) - 1],
                  'Thick': [0, len(cur_df) - 1],
                  'Glass': [0, len(cur_df) - 1],
                  'ND': [0, len(cur_df) - 1],
                  'VD': [0, len(cur_df) - 1]},
        column_config={
            'Surf': st.column_config.NumberColumn('Surf', width='small'),
            'Radius': st.column_config.NumberColumn('Radius', width='medium', format='%.3f'),
            'Thick': st.column_config.NumberColumn('Thick', width='medium', format='%.3f'),
            'Glass': st.column_config.TextColumn('Glass', width='small'),
            'ND': st.column_config.NumberColumn('ND', width='small', format='%.5f'),
            'VD': st.column_config.NumberColumn('VD', width='small', format='%.1f'),
            'Semi-Dia': st.column_config.NumberColumn('Semi-Dia', width='small', format='%.2f'),
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
        view = st.segmented_control('分析视图', ['2D Layout', '3D', 'Spot', '像差分析'],
                                    default='2D Layout', selection_mode='single')
    else:
        view = st.radio('分析视图', ['2D Layout', '3D', 'Spot', '像差分析'], horizontal=True)

    def _dl_png(fig, label, fname):
        """分析图下载 PNG"""
        import io as _io
        _buf = _io.BytesIO()
        fig.savefig(_buf, format='png', dpi=100)
        _buf.seek(0)
        st.download_button(f'⬇ {label}', _buf, file_name=fname, mime='image/png')

    def _cache_fig(key, maker):
        """按结构+参数缓存图：切换视图/重复交互不重算，结构变化自动失效"""
        if st.session_state.get('fig_cache_key') != key:
            st.session_state.fig_cache_key = key
            st.session_state.fig_cache_val = maker()
        return st.session_state.fig_cache_val

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
    st.subheader('GA 收敛曲线 + 任务日志')
    st.caption('运行中的 GA 每 log_every 代实时写入 ga_history.csv；本区每 5 秒自动刷新。')

    @st.fragment(run_every=5)
    def _ga_curve():
        run, logp, pid = run_control.is_running()
        st.caption(f'运行状态: {"⏳ 运行中 (PID " + str(pid) + ")" if run else "⏸ 空闲"}')
        hists = run_control.list_history_csv()
        if hists:
            sel = st.selectbox('收敛曲线（results/…/ga_history.csv）', hists)
            res = run_control.read_history_csv(sel)
            if res:
                g, b = res
                st.line_chart(pd.DataFrame({'gen': g, 'best_mfe': b}).set_index('gen'),
                              height=320)
                st.caption(f'共 {len(g)} 个数据点 | best {b[-1]:.4f}（第 {g[-1]:.0f} 代）')
            else:
                st.info('该 CSV 暂无数据')
        else:
            st.info('暂无 ga_history.csv——先在侧边栏"🚀 运行"页跑一次 GA（未配置 GA 环境时不可用）')
        log_path = logp if logp else st.session_state.get('panel_log')
        if log_path and os.path.exists(log_path):
            with st.expander('任务日志尾部'):
                st.code(run_control.tail_log(log_path, 100), language='text')

    _ga_curve()

_wl_txt = ' / '.join(f'{w:.4f}' for w, _ in wavs_ui)
_fld_txt = ' / '.join(f'{y:.2f}°' for y, _ in fields_ui)
_total = sum(float(s['t']) for s in specs[:-1]
             if np.isfinite(s['t']) and s['t'] != float('inf'))
st.divider()
_pw_idx = int(st.session_state.get('primary_wl', 0)) + 1
_od_txt = '∞' if not np.isfinite(specs[0]['t']) else f'{specs[0]["t"]:.0f}'
_fno_txt = f"{epd / m['efl']:.2f}" if np.isfinite(m['efl']) and m['efl'] > 0 else '—'
st.caption(f'📊 {epd:.1f} mm | F/# {_fno_txt} | 波长 {_wl_txt} µm | 视场 {_fld_txt} | 主波长 λ{_pw_idx} | '
           f'视场类型 {st.session_state.get("field_type", "angle")} | 物距 {_od_txt} | {len(specs)} 面 | 总长 {_total:.1f} mm')
st.caption(f'镜头设计工作台 {CFG.VERSION} | 自包含数据: data/（镜片库 + 内置玻璃表，无外部依赖）')
