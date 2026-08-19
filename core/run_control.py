# -*- coding: utf-8 -*-
"""core/run_control.py — GA 运行控制（网页启动流水线/精修/解码等）
Popen 调完整项目 scripts（python_env），.running 锁防 GPU 并发，
日志落盘 results/panel_*.log，收敛曲线读 ga_history.csv（GA 实时追加）。
"""
import os
import re
import subprocess
import time

import sys
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import config as CFG

GA_ROOT = CFG.GA_PROJECT_ROOT
if GA_ROOT:
    PYTHON = os.path.join(GA_ROOT, 'python_env', 'python.exe')
    SCRIPTS_DIR = os.path.join(GA_ROOT, 'scripts')
    RESULTS_DIR = os.path.join(GA_ROOT, 'results')
else:
    PYTHON = SCRIPTS_DIR = RESULTS_DIR = ''
LOCK = os.path.join(RESULTS_DIR or '.', '.panel_running')

# 运行模式定义：args = [(key, type, default, flag/None)]
#   flag='--xxx' → 关键字参数；None → 位置参数
SCRIPTS = {
    'pipeline_funnel': {
        'label': '三层漏斗流水线（GPU GA→精筛→解码）',
        'script': 'pipeline_funnel.py',
        'args': [('quick', 'int', 0, '--quick', 'N>0 快速模式（代数=50），0=完整', 0, 300)],
        'note': 'GA 运行时收敛曲线实时写入 results/run_*/ga_history.csv'},
    'run_full_pipeline': {
        'label': '完整流水线（GPU 全套：验证→GA→解码）',
        'script': 'run_full_pipeline.py',
        'args': [],
        'note': '无参数，按 config.py 全量运行'},
    'decode': {
        'label': '解码精英 → 设计报告',
        'script': 'decode_design.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', None),
                 ('idx', 'int', 55, None, '', 1, 300)],
        'note': ''},
    'refine_hj': {
        'label': '精修 Hooke-Jeeves（空气间隔）',
        'script': 'refine_hj.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', None),
                 ('idx', 'int', 55, None, '', 1, 300),
                 ('engine', 'select', 'optiland', '--engine', '', ['optiland', 'paraxial']),
                 ('max_evals', 'int', 150, '--max-evals', '', 10, 2000)],
        'note': ''},
    'refine_grad': {
        'label': '精修 梯度法',
        'script': 'refine_grad.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', None),
                 ('idx', 'int', 55, None, '', 1, 300),
                 ('maxiter', 'int', 120, '--maxiter', '', 10, 500)],
        'note': ''},
    'gen_zmx_zemax': {
        'label': '生成 Zemax .zmx（需 OpticStudio 运行）',
        'script': 'gen_elite_zmx.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', '--fname'),
                 ('idx', 'int', 55, '--idx', '', 1, 300)],
        'note': '⚠️ 会启动 OpticStudio COM，请勿与其他 Zemax 任务并发'},
    'verify_zemax': {
        'label': 'Zemax 终审（需 OpticStudio 运行）',
        'script': 'verify_zemax.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', None),
                 ('top', 'int', 5, '--top', '', 1, 20)],
        'note': '⚠️ 会启动 OpticStudio COM，请勿与其他 Zemax 任务并发'},
    'visualize': {
        'label': '生成可视化 PNG（结构图/点列图/场曲）',
        'script': 'visualize_design.py',
        'args': [('fname', 'text', 'run_20260818_213729/ga_elite.txt', None),
                 ('idx', 'int', 55, None, '', 1, 300)],
        'note': ''},
}

if not GA_ROOT:
    SCRIPTS = {}   # 未配置 GA 环境（分享版默认禁用，见 README）


def _pid_alive(pid):
    try:
        r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'],
                           capture_output=True, text=True, timeout=10)
        return str(pid) in r.stdout
    except Exception:
        return False


def is_running():
    """返回 (running, log_path, pid)"""
    if not GA_ROOT:
        return False, None, None
    if not os.path.exists(LOCK):
        return False, None, None
    try:
        with open(LOCK, encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        pid = int(lines[0])
        log = lines[1] if len(lines) > 1 else None
    except Exception:
        os.remove(LOCK)
        return False, None, None
    if _pid_alive(pid):
        return True, log, pid
    os.remove(LOCK)
    return False, None, None


def launch(mode, params):
    """启动任务。返回 (ok, message, log_path)"""
    if not GA_ROOT:
        return False, '未配置 GA 环境——设置环境变量 GA_PROJECT_ROOT 后启用（见 README）', ''
    running, log, pid = is_running()
    if running:
        return False, f'已有任务运行中（PID {pid}）——请等待完成或手动清理 results/.panel_running', log
    spec = SCRIPTS[mode]
    script_path = os.path.join(SCRIPTS_DIR, spec['script'])
    if not os.path.exists(script_path):
        return False, f'脚本不存在: {script_path}', None
    argv = [PYTHON, '-u', script_path]
    for arg in spec['args']:
        key, typ, default, flag = arg[:4]
        val = params.get(key, default)
        if typ == 'int':
            s = str(int(val))
        else:
            s = str(val).strip()
        if flag:
            argv.append(flag)
            if typ != 'flag':
                argv.append(s)
        else:
            argv.append(s)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(RESULTS_DIR, f'panel_{stamp}.log')
    try:
        with open(LOCK, 'w', encoding='utf-8') as f:
            f.write('0\n' + log_path)
        proc = subprocess.Popen(argv, cwd=GA_ROOT,
                                stdout=open(log_path, 'w', encoding='utf-8'),
                                stderr=subprocess.STDOUT, shell=False)
        with open(LOCK, 'w', encoding='utf-8') as f:
            f.write(f'{proc.pid}\n{log_path}')
        return True, f'已启动 {spec["label"]}（PID {proc.pid}）', log_path
    except Exception as e:
        if os.path.exists(LOCK):
            os.remove(LOCK)
        return False, f'启动失败: {e}', None


def stop():
    """停止当前任务（杀进程 + 清锁）"""
    if not GA_ROOT:
        return False, '未配置 GA 环境'
    running, log, pid = is_running()
    if not running:
        return False, '当前无运行任务'
    try:
        subprocess.run(['taskkill', '/PID', str(pid), '/F'],
                       capture_output=True, text=True, timeout=10)
    except Exception:
        pass
    if os.path.exists(LOCK):
        os.remove(LOCK)
    return True, f'已终止 PID {pid}'


def tail_log(log_path, n=60, retries=3):
    if not log_path or not os.path.exists(log_path):
        return ''
    last = ''
    for _ in range(retries):
        try:
            with open(log_path, encoding='utf-8', errors='replace') as f:
                lines = f.read().splitlines()
            last = '\n'.join(lines[-n:])
            if last:
                return last
        except Exception:
            pass
        time.sleep(0.5)
    return last


def list_history_csv():
    """results/run_*/ga_history.csv 列表（按时间倒序）"""
    out = []
    if not os.path.isdir(RESULTS_DIR):
        return out
    for d in sorted(os.listdir(RESULTS_DIR), reverse=True):
        p = os.path.join(RESULTS_DIR, d)
        if os.path.isdir(p):
            hp = os.path.join(p, 'ga_history.csv')
            if os.path.exists(hp):
                out.append(os.path.join(d, 'ga_history.csv'))
    return out


def read_history_csv(rel_path):
    """ga_history.csv → (gens, bests)；格式: gen,best_mfe"""
    if not GA_ROOT:
        return None
    import numpy as np
    p = os.path.join(RESULTS_DIR, rel_path)
    if not os.path.exists(p):
        return None
    gens, bests = [], []
    try:
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('gen'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    gens.append(float(parts[0]))
                    bests.append(float(parts[1]))
    except Exception:
        return None
    if not gens:
        return None
    return gens, bests


if __name__ == '__main__':
    print('is_running:', is_running())
    print('history csv:', list_history_csv())
    if list_history_csv():
        g, b = read_history_csv(list_history_csv()[0])
        print(f'最近 history: {len(g)} 点, gen {g[0]:.0f}->{g[-1]:.0f}, best {b[0]:.4f}->{b[-1]:.4f}')
    print('RUN_CONTROL: OK')
