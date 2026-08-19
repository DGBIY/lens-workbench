# 🔭 镜头设计工作台（Lens Design Workbench）

基于 **Streamlit + optiland** 的交互式镜头设计工具——LDE 自由编辑（白纸作画）、多波长分析、
像差/MTF/3D 可视化、局部优化、玻璃搜索，**完全自包含，不需要 Zemax**。

## ✨ 功能

| 模块 | 说明 |
|---|---|
| 📋 **LDE 编辑** | 类 Zemax 表格：直接添加/删除行、玻璃名自动带出 AGF nd/vd、光阑位置可移动（白纸作画） |
| 🔭 **2D / 3D Layout** | 多波长 F/d/C 光线 + 焦点标记；3D 旋转体 + 光线 |
| ⭕ **Spot Diagram** | optiland 真实追迹，RMS 表 + 艾里斑圈（1.22λF#） |
| 📉 **像差分析（11 种）** | 场曲/畸变/RayFan/RMS-vs-场/EE/畸变网格/透过焦×2/RMS 波前/**MTF**/轴向色差 |
| ⚡ **局部优化** | Nelder-Mead：空气间隔+后焦，可选厚度/曲率变量，MFE 分解预览 + 收敛曲线 |
| 🔎 **玻璃搜索** | 按 nd/vd 找最接近玻璃（内置 81 种 AGF 实测表） |
| 📤 **导出** | .zmx / LDE 文本 / 分析图 PNG 下载 |
| 🎛 **集成中心** | 侧边栏分类导航（数据/系统/运行/设计）+ 状态栏（F#/波长/物距/主波长） |

## 🚀 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 启动
streamlit run app.py
```

打开 http://localhost:8501 即可。默认加载内置样例结构（精英55），也可在
`data/samples/` 放入自己的精英文件（`ga_elite.txt` 格式：每行 6 组
`T{类型}R{行号}` + 5 个空气间隔）。

## 📁 项目结构

```
窗口/
├── app.py                 # 主程序（Streamlit）
├── config.py              # 集中配置（路径/版本）
├── requirements.txt
├── core/
│   ├── lens_io.py         # LDE/规格/精英解析
│   ├── layout2d.py        # 2D Layout
│   ├── layout3d.py        # 3D Layout
│   ├── spot_rms.py        # 点列图 + RMS + 艾里斑
│   ├── analysis.py        # 11 种像差分析 + MTF + 优化器
│   ├── eval.py            # 指标评估（EFFL/AXCL/RSCE）
│   ├── run_control.py     # GA 运行控制（可选，见下）
│   ├── compare.py         # 多精英对比
│   ├── _bridge.py         # 自包含桥接（光学核心）
│   ├── _library.py        # 镜片库加载（data/pybl）
│   └── _glass_table.py    # 内置玻璃表（81 种 AGF）
└── data/
    ├── pybl/              # 镜片库数据（pybl1-6）
    └── samples/           # 精英文件（可选，可放自己的）
```

## 🔌 可选：GA 运行控制

侧边栏"🚀 运行"页的 GA 任务控制是**可选功能**，需要完整的 GA 项目环境
（Python 环境 + Zemax COM 脚本）。启用方式：

```bash
# 设置环境变量指向 GA 项目根（含 results/ 和 scripts/）
set GA_PROJECT_ROOT=C:\path\to\GA\完整项目
```

未设置时该页显示提示，其余功能不受影响。

## ⚙️ 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `WORKBENCH_DATA` | `./data` | 数据目录（镜片库/样例） |
| `GA_PROJECT_ROOT` | 无 | GA 项目根（启用运行控制） |

## 🧪 自检

```bash
python core/lens_io.py     # LDE/规格往返 + RSCE 一致性
python core/eval.py        # 指标一致性
python core/analysis.py    # 11 种分析 + 玻璃搜索
```

## 📝 说明

- 所有光学计算基于 [optiland](https://github.com/sunglass/optiland)（光线追迹引擎），
  不依赖 Zemax；与 Zemax 的一致性已在开发中验证（EFFL/AXCL/RSCE）。
- 玻璃数据为 CDGM/SCHOTT 等厂商 AGF 实测值（81 种），仅用于设计参考，
  生产前请与供应商确认最终参数。
