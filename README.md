# 🔭 镜头设计工作台（Lens Design Workbench）

基于 **Streamlit + optiland** 的天文摄影镜头自动设计工作台——遗传算法搜索、
评价函数精评、星场验证、2D 布局，完全自包含（不依赖 Zemax）。

## ✨ 功能

| 模块 | 说明 |
|---|---|
| 🧬 **GA 引擎** | 两种引擎：内置 optiland GA（锦标赛/交叉/变异/超时淘汰）+ 近轴 GPU 子进程（KNN 突变/重启/去重，大种群 50000×60 ≈ 2 分钟） |
| 🎯 **评价函数** | 12 种操作数 MFE：EFFL / AXCL / FCGT / FCGS / RSCE / DIST / WFE / TOTR / BFL / MTF / THF / RELIL |
| 🌠 **星场验证** | 三模式：随机星空 / 网格 / **真实星图**（内置猎户座 M42 星表 11 亮星）+ **CSV 星表导入**（name,ra,dec,mag，超视场自动剔除）；F/d/C 三波长点列图 + 像差放大 + 离焦对比 |
| 📐 **2D 布局图** | Zemax 风格镜头侧视图 + 近轴光线追踪（多视场主光线/边缘光线，F#/EFL 标注） |
| 📋 **LDE 编辑** | 类 Zemax 表格：添加/删除行、玻璃自动带出 AGF nd/vd、光阑可移动 |
| 📉 **像差分析** | 场曲/畸变/RayFan/RMS-vs-场/EE/MTF/轴向色差等 |
| ⚡ **局部优化** | Nelder-Mead：空气间隔+后焦，MFE 分解 + 收敛曲线 |
| 📤 **导出** | .zmx / LDE 文本 / 分析图 PNG |

## 🚀 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt   # optiland 为本地 wheel（见 新/references/optiland_whl/）

# 2. 启动工作台
streamlit run app.py

# 3.（可选）启用近轴 GPU 引擎：设置环境变量指向完整项目
#    GA_PROJECT_ROOT=C:\path\to\GA\完整项目
#    完整项目自带独立 python_env（torch 2.9 + CUDA），工作台通过子进程调用
```

## 🏗️ 架构：两项目 + 子进程隔离

```
窗口/（本仓库：Streamlit 工作台 + optiland 精评，纯 Python）
完整项目/（近轴 GPU 引擎：独立 python_env + torch，大种群搜索）

    run_ga_remote() ── subprocess.Popen → ga_workbench.py（完整项目）
         │ 进度：轮询 history csv（每 1s）
         │ 超时：kill 子进程；正常结束清理临时文件
         ▼
  ⚠️ 子进程隔离是【必要设计】：
     窗口与完整项目都有 config.py（模块名冲突）——
     同一进程内混用两套模块会解析到错误的 config（实测 AttributeError）。
     所有跨项目调用必须走子进程（Popen）。
```

其他关键点：
- 内置 optiland GA（`run_ga`）在主进程跑；评估用**进程池**（v0.26.1 起：
  optiland 对病态结构会死循环，线程无法终止 → 进程池超时 terminate + 重建）
- spawn 进程池要求评估入口为**脚本文件**（app.py 正常；`python -c` 不可用）
- 近轴引擎规格：TARGET_EFL=200 / F5 / BACK_FOCUS=55（完整项目 config.py 统一配置）

## 📁 项目结构

```
窗口/
├── app.py                 # Streamlit 主程序
├── config.py              # 集中配置（GA_PROJECT_ROOT 等）
├── start_workbench.ps1 / 启动工作台.bat
├── core/
│   ├── ga_engine.py       # GA 引擎：内置 optiland GA + run_ga_remote（子进程）+ 挂起防护
│   ├── merit.py           # 评价函数：12 操作数 MFE
│   ├── starfield.py       # 星场渲染：随机/网格/真实星图（M42）/CSV 导入
│   ├── layout.py          # 2D 布局图（v0.26 新增）
│   ├── lens_io.py         # LDE/规格/精英解析 + optiland 系统构建
│   ├── analysis.py        # 像差分析
│   ├── spot_rms.py        # 点列图 + RMS
│   ├── layout2d.py        # 2D Layout（旧版）
│   ├── layout3d.py        # 3D Layout
│   ├── eval.py            # 指标评估（EFFL/AXCL/RSCE）
│   ├── astro_tools.py     # 天文工具（SNR/星等/减焦镜/热漂移）
│   ├── templates.py       # 样板模板
│   ├── compare.py         # 多精英对比
│   ├── zmx_export.py      # .zmx 导出
│   └── run_control.py     # 运行控制（可选）
└── data/
    ├── pybl/              # 镜片库数据
    └── samples/           # 精英文件（ga_elite.txt 格式）
```

## 🧪 测试（一键回归）

```bash
# 完整项目侧统一入口（9 项，独立子进程隔离 config 冲突）：
完整项目\python_env\python.exe 完整项目\scripts\run_tests.py
#   [--fast] 跳过 W3 挂起回归（最慢项）
# 覆盖：语法×2 / merit 自检 / 内置 GA smoke / W3 挂起回归 / UI 冒烟
#      （真实星图+CSV+星场+布局渲染）/ spawn 冒烟 / 子进程+直连双路径快测
```

## ⚙️ 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `GA_PROJECT_ROOT` | 无 | 完整项目根（启用近轴 GPU 引擎） |
| `WORKBENCH_DATA` | `./data` | 数据目录（镜片库/样例） |

## 📝 说明

- 光学计算基于 [optiland](https://github.com/sunglass/optiland)，不依赖 Zemax
- 玻璃数据为 CDGM/SCHOTT 等 AGF 实测值（81 种），生产前请与供应商确认
- 版本历史：v0.16（自包含 UI）→ v0.24（optiland GA + 评价函数）→ v0.25（近轴 GPU 引擎接入）→ v0.26（真实星图/2D 布局/多目标/精评回填）→ v0.26.1（挂起修复/进程池）
