# -*- coding: utf-8 -*-
"""config.py — 集中配置（分享/GitHub 友好）

所有可配置路径集中于此：
- DATA_DIR       数据目录（默认本目录下 data/；可用环境变量 WORKBENCH_DATA 覆盖）
- PYBL_DIR       镜片库（pybl1-6.txt）
- SAMPLES_DIR    样例/用户精英文件（可选，缺省时用内置默认结构）
- GA_PROJECT_ROOT 完整项目根（GA 运行控制用；未设置时该功能禁用，见 README）
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get('WORKBENCH_DATA', os.path.join(HERE, 'data'))
PYBL_DIR = os.path.join(DATA_DIR, 'pybl')
SAMPLES_DIR = os.path.join(DATA_DIR, 'samples')

GA_PROJECT_ROOT = os.environ.get('GA_PROJECT_ROOT') or None

VERSION = 'v0.14'

# ---- 兼容常量（供核心模块 __main__ 自检使用；与完整项目 config 同义）----
ENPD = 58.0      # GA 库系统入瞳直径（mm）
BACK_FOCUS = 55.0
