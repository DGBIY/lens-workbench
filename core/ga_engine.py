# -*- coding: utf-8 -*-
"""core/ga_engine.py — optiland 版 GA（v0.24）

遗传算法：从镜片库组合搜索镜头结构，评估全部用 optiland（零 Zemax）。
- 基因：n_groups 组 (库类型, 行号) + n-1 空气间隔（组数可配置，推荐 6）
- 评估：core.merit 评价函数（可配置操作数/权重）
- 种子：样板库模板生成初始个体（GA + 天文模板联动）
- 日志：ga_history.csv（gen,best_mfe）兼容现有收敛曲线图表
- Zemax 联动保留：UI 引擎开关切换（'zemax' 走 run_control）
"""
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from core._library import get_default_library
from core.lens_io import elite_to_specs, library_rows
from core.merit import compute_operands, merit_value
from core.templates import _match_group, TEMPLATES

_lib = get_default_library()

# 位置 i（组序）最小口径：前组大（光阑区），向后递减
POS_DIAM_MIN = [42, 34, 30, 26, 24, 24, 24, 24, 24, 24]


def _random_lens(rng, min_diam, lt_hint=None):
    """随机匹配一片镜片（口径约束；lt_hint 为 None 时随机单/双胶合）"""
    if lt_hint is None:
        lt_hint = rng.choice([5, 6] if rng.random() < 0.6 else [1, 2, 3, 4])
    lts = (lt_hint,)
    cands = []
    for lt in lts:
        lo, hi, _ = library_rows(lt)
        for row in range(lo, hi + 1):
            L = _lib.get_lens(lt, row)
            if L is None:
                continue
            d = min(L[0]['diam'], L[1]['diam']) if isinstance(L, tuple) else L['diam']
            if d >= min_diam:
                cands.append((lt, row))
    if not cands:
        return None
    return cands[rng.randrange(len(cands))]


def random_individual(rng, n_groups):
    pairs = []
    for i in range(n_groups):
        m = _random_lens(rng, POS_DIAM_MIN[i])
        if m is None:
            return None
        pairs.append(m)
    airs = [round(rng.uniform(4.0, 45.0), 1) for _ in range(n_groups - 1)]
    return {'pairs': pairs, 'airs': airs}


def seed_from_template(key, rng, n_groups):
    """样板模板 → 个体（复用模板槽位匹配；组数不符则截断/补随机）"""
    tpl = TEMPLATES.get(key)
    if tpl is None or tpl.get('reflective'):
        return None
    pairs = []
    for grp in tpl['groups'][:n_groups]:
        m = _match_group(grp, rng)
        if m is None:
            return None
        pairs.append(m)
    while len(pairs) < n_groups:
        m = _random_lens(rng, POS_DIAM_MIN[len(pairs)])
        if m is None:
            return None
        pairs.append(m)
    airs = [round(a, 1) for a in tpl['airs'][:n_groups - 1]]
    while len(airs) < n_groups - 1:
        airs.append(round(rng.uniform(4.0, 45.0), 1))
    return {'pairs': pairs, 'airs': airs}


def fitness(ind, epd, fields, wavs, merit, back_focus=55.0):
    specs = elite_to_specs(ind['pairs'], ind['airs'], back_focus)
    if specs is None:
        return 1e9
    ops = compute_operands(specs, epd, fields, wavs)
    if not ops:
        return 1e9
    return merit_value(ops, merit)


# 部分随机结构会让 optiland 追迹挂起（引擎内部循环）——每次评估新建线程池 + 超时
# 超时后 shutdown(wait=False) 丢弃挂起线程，避免阻塞后续评估
def _safe_fitness(ind, epd, fields, wavs, merit, back_focus=55.0, timeout=8.0):
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fitness, ind, epd, fields, wavs, merit, back_focus)
    try:
        return fut.result(timeout=timeout)
    except _Timeout:
        ex.shutdown(wait=False)
        return 1e9  # 超时个体判劣淘汰


def crossover(a, b, rng):
    n = len(a['pairs'])
    k = rng.randint(1, n - 1)
    # 空气间隔随机取父母一方（避免均值产生"半间隔"诱发追迹数值问题）
    airs = [rng.choice((x, y)) for x, y in zip(a['airs'], b['airs'])]
    return {'pairs': a['pairs'][:k] + b['pairs'][k:], 'airs': airs}


def mutate(ind, rng, mr=0.3):
    out = {'pairs': list(ind['pairs']), 'airs': list(ind['airs'])}
    if rng.random() < mr:
        i = rng.randrange(len(out['pairs']))
        m = _random_lens(rng, POS_DIAM_MIN[i], out['pairs'][i][0])
        if m:
            out['pairs'][i] = m
    if out['airs'] and rng.random() < mr:
        i = rng.randrange(len(out['airs']))
        out['airs'][i] = min(50.0, max(3.0, out['airs'][i] * rng.uniform(0.8, 1.2)))
    return out


def run_ga(n_groups=6, merit=None, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8),
           wavs=None, pop=24, gens=20, seed_templates=None, back_focus=55.0,
           log_path=None, seed=42, tournament_k=8, progress=None):
    """运行 optiland GA → (best_ind, best_specs, history)
    progress: 可选回调 progress(gen, best) 用于 UI 进度显示
    """
    if wavs is None:
        wavs = [(0.48613, 1.0), (0.58756, 1.0), (0.65627, 1.0)]
    if merit is None:
        from core.merit import merit_from_preset
        merit = merit_from_preset('深空 APO 推荐')
    rng = random.Random(seed)
    n_groups = int(max(4, min(10, n_groups)))

    # 初始种群：模板种子 + 随机
    pop_inds = []
    for key in (seed_templates or []):
        for _ in range(2):
            s = seed_from_template(key, rng, n_groups)
            if s:
                pop_inds.append(s)
    while len(pop_inds) < pop:
        s = random_individual(rng, n_groups)
        if s:
            pop_inds.append(s)
    pop_inds = pop_inds[:pop]

    hist = []
    best_ind, best_f = None, 1e18
    for g in range(gens):
        scored = [(_safe_fitness(ind, epd, fields, wavs, merit, back_focus), ind)
                  for ind in pop_inds]
        scored.sort(key=lambda x: x[0])
        cur_best, cur_f = scored[0][1], scored[0][0]
        if cur_f < best_f:
            best_f, best_ind = cur_f, cur_best
        hist.append((g + 1, best_f))
        if log_path:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f'{g + 1},{best_f:.6f}\n')
        if progress:
            progress(g + 1, best_f)
        # 繁殖：精英 2 + 锦标赛选择交叉变异
        nxt = [scored[0][1], scored[1][1]]
        while len(nxt) < pop:
            t = sorted(rng.sample(scored, min(tournament_k, len(scored))),
                       key=lambda x: x[0])
            a, b = t[0][1], t[1][1]
            child = crossover(a, b, rng)
            child = mutate(child, rng)
            nxt.append(child)
        pop_inds = nxt[:pop]
    best_specs = elite_to_specs(best_ind['pairs'], best_ind['airs'], back_focus)
    return best_ind, best_specs, hist


def run_ga_remote(engine='cpu', pop=2000, gens=200, seed=42, target_efl=200.0,
                  back_focus=55.0, log_every=50, timeout=3600, progress=None):
    """子进程调用完整项目近轴 GA（ga_workbench.py，固定 6 组基因）
    engine: 'cpu' 向量化 / 'gpu' CUDA（需 GA_PROJECT_ROOT + python_env + N 卡）
    复用成熟引擎（ga_fast：KNN 突变/重启/爬坡/去重）→ (best_genes(17,), history)
    子进程隔离：避免 config 模块名冲突；挂起可 kill；进度轮询 ga_history.csv
    """
    import config as _cfg
    root = _cfg.GA_PROJECT_ROOT
    if not root:
        raise RuntimeError('未配置 GA_PROJECT_ROOT（完整项目）')
    script = os.path.join(root, 'scripts', 'ga_workbench.py')
    py = os.path.join(root, 'python_env', 'python.exe')
    if not (os.path.exists(script) and os.path.exists(py)):
        raise RuntimeError('缺少 ga_workbench.py 或 python_env')
    hist_path = os.path.join(root, 'scripts', f'_wb_hist_{seed}.csv')
    best_out = os.path.join(root, 'scripts', f'_wb_best_{seed}.json')
    for p in (hist_path, best_out):
        if os.path.exists(p):
            os.remove(p)
    argv = [py, '-u', script, '--pop', str(pop), '--gens', str(gens),
            '--seed', str(seed), '--engine', engine,
            '--target-efl', str(target_efl), '--back-focus', str(back_focus),
            '--log-every', str(log_every),
            '--history-out', hist_path, '--best-out', best_out]
    proc = subprocess.Popen(argv, cwd=root, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    t0 = time.time()
    last_n = 0
    while proc.poll() is None:
        if os.path.exists(hist_path):
            rows = [l.strip().split(',') for l in open(hist_path, encoding='utf-8')
                    if l.strip() and not l.startswith('gen')]
            if len(rows) > last_n:
                last_n = len(rows)
                if progress:
                    progress(int(float(rows[-1][0])), float(rows[-1][1]))
        if timeout and time.time() - t0 > timeout:
            proc.kill()
            raise TimeoutError(f'GA 超时（>{timeout}s）')
        time.sleep(1.0)
    if proc.returncode != 0 or not os.path.exists(best_out):
        raise RuntimeError(f'GA 子进程失败（exit={proc.returncode}）')
    with open(best_out, encoding='utf-8') as f:
        d = json.load(f)
    genes = np.array(d['types'] + d['rows'] + d['airs'], dtype=float)
    hist = []
    if os.path.exists(hist_path):
        rows = [l.strip().split(',') for l in open(hist_path, encoding='utf-8')
                if l.strip() and not l.startswith('gen')]
        hist = [(int(float(r[0])), float(r[1])) for r in rows]
    return genes, hist


if __name__ == '__main__':
    from core.merit import merit_from_preset
    merit = merit_from_preset('深空 APO 推荐', target_efl=200.0)
    bi, bs, hist = run_ga(n_groups=6, merit=merit, pop=8, gens=5,
                          seed_templates=['tessar'], seed=7)
    print('hist:', [(g, round(f, 3)) for g, f in hist])
    assert len(hist) == 5
    assert all(hist[i][1] <= hist[i - 1][1] + 1e-9 for i in range(1, len(hist))), 'best 未单调'
    assert bs is not None and len(bs) >= 4
    from core.eval import evaluate_specs
    m = evaluate_specs(bs, epd=40.0, fields=(0.0, 2.0, 4.06, 5.8))
    print('best: EFFL %.1f RSCE %.0fum | 组数 %d' % (m['efl'], m['rsce_um'], len(bi['pairs'])))
    print('GA_ENGINE: OK')
