"""LoRA Cosine —— 参数漂移分析（不需要 GPU，纯读权重）。

train1.py 是按任务顺序累积训练的，所以 train1_continual_lora/<前缀>/ 里的
LoRA 权重是"相对基座的累计增量 ΔW"（不是本任务单独的增量）。

本脚本做三件事：
  1. 任意两个阶段的累计 ΔW 之间的余弦（n×n 矩阵）——看后一阶段是继续沿同一
     方向走，还是拐弯/回退了；
  2. 相邻阶段：cos(ΔW_k, ΔW_{k-1})、相对幅度变化、以及"本阶段新增的更新方向
     与已累计方向"的余弦（判断这一阶段是在"加强"还是在"改写"）；
  3. 逐层的版本（层 × 阶段），看漂移集中在哪几层。

注意：不能直接对 lora_A / lora_B 拼起来的向量算余弦 —— A、B 有缩放不变性
（A→sA、B→B/s 给出同一个 ΔW），必须先还原成 ΔW = B@A*(alpha/r)。见 _common.lora_delta。

用法：
  python3 analysis/lora_cosine.py
  python3 analysis/lora_cosine.py --stages AGNEWS AGNEWS+QQP AGNEWS+QQP+TLDR
  python3 analysis/lora_cosine.py --adapter-root ./train1_continual_lora --out-dir ./analysis_results/lora
"""

import argparse
import os

from _common import (
    ADAPTER_ROOT,
    cosine_from_products,
    inner_products,
    layers_of,
    lora_modules,
    module_layer,
    resolve_args,
    save_heatmap,
    save_json,
    save_matrix_csv,
)

# 分析哪几个模块（本项目 LoRA 只打在注意力的四个投影上）
MODULE_ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def parse_args():
    parser = argparse.ArgumentParser(description='LoRA 权重增量（参数漂移）分析')
    parser.add_argument('--adapter-root', default=ADAPTER_ROOT)
    parser.add_argument('--stages', nargs='*', default=None,
                        help='要比较的 adapter 目录名（按训练顺序给）；不填=自动发现全部')
    parser.add_argument('--out-dir', default=None)
    return parser.parse_args()


def discover_stages(root):
    """找出所有 adapter 目录，按"训练阶段"排序：先按 '+' 个数，再按目录修改时间。"""
    found = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'adapter_config.json')):
            found.append((name.count('+'), os.path.getmtime(path), name))
    found.sort()
    return [name for _, _, name in found]


def module_norms(modules, scaling):
    """{模块类型: 该类型所有层的 ΔW 的 L2 范数}（float64 累加）。"""
    norms = {}
    for module in MODULE_ORDER:
        names = [name for name in modules if module_layer(name)[1] == module]
        if not names:
            norms[module] = 0.0
            continue
        _, norm_a, _ = inner_products(modules, scaling, modules, scaling, names=names)
        norms[module] = norm_a ** 0.5
    return norms


def main():
    args = resolve_args(parse_args(), 'lora_cosine')

    stages = args.stages if args.stages else discover_stages(args.adapter_root)
    if not stages:
        raise SystemExit(f'{args.adapter_root} 下没有找到 adapter')
    print(f'阶段（{len(stages)} 个）：{stages}')

    # ----------------------------------------
    # 读权重（只留 A、B 的小矩阵，需要时现算 ΔW）
    # ----------------------------------------
    weight = {}     # stage -> (modules, scaling)
    norms = {}      # stage -> ||ΔW||（总体）
    print()
    for stage in stages:
        path = os.path.join(args.adapter_root, stage)
        modules, scaling = lora_modules(path)
        weight[stage] = (modules, scaling)
        _, norm_sq, _ = inner_products(modules, scaling, modules, scaling)
        norms[stage] = norm_sq ** 0.5
        per_module = module_norms(modules, scaling)
        print(f'  {stage:28s} 模块数={len(modules):3d}  ||ΔW||={norms[stage]:8.2f}  '
              + ' '.join(f'{m}={per_module[m]:.1f}' for m in MODULE_ORDER))

    # ----------------------------------------
    # 1) 两两余弦矩阵
    # ----------------------------------------
    n = len(stages)
    products = {}
    matrix = []
    for a in stages:
        row = []
        for b in stages:
            if (b, a) in products:
                dot, na, nb = products[(b, a)]
            else:
                dot, na, nb = inner_products(*weight[a], *weight[b])
                products[(a, b)] = (dot, na, nb)
            row.append(cosine_from_products(dot, na, nb))
        matrix.append(row)

    print()
    print('累计 ΔW 的两两余弦（1=同方向，0=正交，-1=反方向）')
    print(' ' * 28 + ''.join(f'{i + 1:>8d}' for i in range(n)))
    for i, stage in enumerate(stages):
        print(f'  {i + 1}. {stage:24s}' + ''.join(f'{matrix[i][j]:8.3f}' for j in range(n)))

    save_matrix_csv(os.path.join(args.out_dir, 'stage_cosine.csv'), matrix,
                    [f'{i + 1}.{s}' for i, s in enumerate(stages)],
                    [f'{i + 1}.{s}' for i, s in enumerate(stages)])
    save_heatmap(matrix, [f'{i + 1}.{s}' for i, s in enumerate(stages)],
                 [f'{i + 1}.{s}' for i, s in enumerate(stages)],
                 os.path.join(args.out_dir, 'stage_cosine.png'),
                 'LoRA cumulative delta: pairwise cosine', vmin=-1, vmax=1, cmap='coolwarm')

    # ----------------------------------------
    # 2) 相邻阶段：加强还是改写
    # ----------------------------------------
    drift_rows = []
    for i in range(1, n):
        prev, curr = stages[i - 1], stages[i]
        dot, norm_prev_sq, norm_curr_sq = products[(prev, curr)]
        # δW_k = ΔW_k - ΔW_{k-1}，下面这些都能从三个标量推出来，不用真的构造向量：
        #   ⟨δW_k, ΔW_{k-1}⟩ = ⟨ΔW_k,ΔW_{k-1}⟩ - ‖ΔW_{k-1}‖²
        #   ⟨δW_k, ΔW_k⟩     = ‖ΔW_k‖² - ⟨ΔW_k,ΔW_{k-1}⟩
        #   ‖δW_k‖²          = ‖ΔW_k‖² + ‖ΔW_{k-1}‖² - 2⟨ΔW_k,ΔW_{k-1}⟩
        cos_to_prev = cosine_from_products(dot, norm_prev_sq, norm_curr_sq)
        norm_update_sq = norm_curr_sq + norm_prev_sq - 2 * dot
        norm_update = max(norm_update_sq, 0.0) ** 0.5
        cos_update_to_prev = ((dot - norm_prev_sq)
                              / (norm_update * (norm_prev_sq ** 0.5) + 1e-24))
        cos_update_to_curr = ((norm_curr_sq - dot)
                              / (norm_update * (norm_curr_sq ** 0.5) + 1e-24))
        row = {
            'stage': curr,
            'prev': prev,
            'cos_to_prev': cos_to_prev,
            # 本任务新产生的更新 δW_k 与"已有累计方向"的关系：
            # > 0 同向加强，< 0 在抵消（这是解释 continual learning 最有用的一个量）
            'cos_update_to_prev': cos_update_to_prev,
            # 同上，但和"更新完之后的方向"比：≈1 表示这一段基本只是延长旧方向
            'cos_update_to_curr': cos_update_to_curr,
            'norm_prev': norm_prev_sq ** 0.5,
            'norm_curr': norm_curr_sq ** 0.5,
            'relative_norm_change': (norm_curr_sq ** 0.5) / (norm_prev_sq ** 0.5 + 1e-12),
            'norm_update': norm_update,
            'norm_update_over_prev': norm_update / (norm_prev_sq ** 0.5 + 1e-12),
            # 抵消比例：新更新里有多大比例是在跟已有方向"对着干"
            'cancel_ratio': max(0.0, 1.0 - cos_update_to_prev),
        }
        drift_rows.append(row)
        print()
        print(f'  {prev}  ->  {curr}')
        print(f'     cos(ΔW_k, ΔW_k-1)         = {row["cos_to_prev"]:+.3f}   （累计方向的整体旋转）')
        print(f'     cos(δW_k, ΔW_k-1)         = {row["cos_update_to_prev"]:+.3f}   '
              f'（>0 强化旧方向 / <0 抵消旧方向）← 重点看这个')
        print(f'     cos(δW_k, ΔW_k)           = {row["cos_update_to_curr"]:+.3f}')
        print(f'     ||δW_k|| / ||ΔW_k-1||     = {row["norm_update_over_prev"]:.3f}')

    if drift_rows:
        save_json(os.path.join(args.out_dir, 'stage_drift.json'),
                  {'stages': stages, 'deltas': drift_rows})
        with open(os.path.join(args.out_dir, 'stage_drift.csv'), 'w',
                  encoding='utf-8-sig') as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=list(drift_rows[0].keys()))
            writer.writeheader()
            writer.writerows(drift_rows)

    # ----------------------------------------
    # 3) 逐层：相邻阶段每层的余弦 + 每层"新更新 vs 旧方向"
    #    （哪个层在抵消、哪个层在加强，看这张表）
    # ----------------------------------------
    layers = layers_of(weight[stages[0]][0])
    layer_matrix = []
    layer_update_matrix = []
    for i in range(1, n):
        modules_prev, scaling_prev = weight[stages[i - 1]]
        modules_curr, scaling_curr = weight[stages[i]]
        row, row_update = [], []
        for layer in layers:
            names = [name for name in modules_prev
                     if module_layer(name)[0] == layer and name in modules_curr]
            if not names:
                row.append(float('nan'))
                row_update.append(float('nan'))
                continue
            dot, na, nb = inner_products(modules_prev, scaling_prev,
                                         modules_curr, scaling_curr, names=names)
            row.append(cosine_from_products(dot, na, nb))
            update_sq = max(na + nb - 2 * dot, 0.0)
            row_update.append((dot - na) / ((update_sq ** 0.5) * (na ** 0.5) + 1e-24))
        layer_matrix.append(row)
        layer_update_matrix.append(row_update)

    save_matrix_csv(os.path.join(args.out_dir, 'layer_cosine.csv'), layer_matrix,
                    [f'{stages[i - 1]} -> {stages[i]}' for i in range(1, n)],
                    [f'layer {layer}' for layer in layers])
    save_matrix_csv(os.path.join(args.out_dir, 'layer_update_cosine.csv'), layer_update_matrix,
                    [f'{stages[i - 1]} -> {stages[i]}' for i in range(1, n)],
                    [f'layer {layer}' for layer in layers])
    if layer_matrix:
        save_heatmap(layer_matrix, [f'{stages[i - 1]}→{stages[i]}' for i in range(1, n)],
                     [f'L{layer}' for layer in layers],
                     os.path.join(args.out_dir, 'layer_cosine.png'),
                     'Per-layer cosine(dW_k, dW_k-1) [cumulative delta]',
                     vmin=-1, vmax=1, cmap='coolwarm')
        save_heatmap(layer_update_matrix, [f'{stages[i - 1]}→{stages[i]}' for i in range(1, n)],
                     [f'L{layer}' for layer in layers],
                     os.path.join(args.out_dir, 'layer_update_cosine.png'),
                     'Per-layer cos(dW_k - dW_k-1, dW_k-1)  [reinforce > 0 > cancel]',
                     vmin=-1, vmax=1, cmap='coolwarm')

    # ----------------------------------------
    # 4) 每个任务自己造成的更新 δW 之间的余弦
    #
    #   δW_k = ΔW_k - ΔW_{k-1}   ← 第 k 个任务"这一件事"造成的 LoRA 更新
    #
    #   它和 Gradient Cosine 正好配对：
    #     Gradient Cosine    理论上前进方向（梯度）
    #     δW Cosine          实际走出来的方向（权重）
    #   两者对比就能看到"想更新成什么样"和"实际更新成什么样"的差距。
    #
    #   同样不需要构造 δW：四个累计内积一组合就行
    #     ⟨δW_i, δW_j⟩ = ⟨ΔW_i,ΔW_j⟩ - ⟨ΔW_i,ΔW_{j-1}⟩
    #                    - ⟨ΔW_{i-1},ΔW_j⟩ + ⟨ΔW_{i-1},ΔW_{j-1}⟩
    #   （base 视为零向量，所以第一段的 δW_1 = ΔW_1）
    # ----------------------------------------
    chain = [None] + stages          # None = base（ΔW = 0）

    def prod(a, b):
        """⟨ΔW_a, ΔW_b⟩；base 与任何阶段的内积都是 0。"""
        if a is None or b is None:
            return 0.0
        if (b, a) in products:
            return products[(b, a)][0]
        return products[(a, b)][0]

    def norm_sq(a):
        if a is None:
            return 0.0
        return products[(a, a)][1]

    def build_delta_matrix(product_fn):
        """由"累计量两两内积"函数推出 δW 的两两内积/余弦矩阵。"""
        size = len(chain)
        values = [[0.0] * size for _ in range(size)]
        norms = [0.0] * size
        for i in range(1, size):
            for j in range(1, size):
                values[i][j] = (product_fn(chain[i], chain[j])
                                - product_fn(chain[i - 1], chain[j])
                                - product_fn(chain[i], chain[j - 1])
                                + product_fn(chain[i - 1], chain[j - 1]))
        for i in range(1, size):
            norms[i] = (norm_sq(chain[i])
                        - 2 * product_fn(chain[i], chain[i - 1])
                        + norm_sq(chain[i - 1]))

        out = []
        for i in range(1, size):
            row = []
            for j in range(1, size):
                denom = (max(norms[i], 0.0) ** 0.5) * (max(norms[j], 0.0) ** 0.5)
                row.append(values[i][j] / (denom + 1e-24))
            out.append(row)
        return out, norms[1:]

    def task_label(stage):
        """这个阶段新训的任务（目录名最后一段）。"""
        return stage.split('+')[-1]

    task_labels = [task_label(stage) for stage in stages]
    delta_matrix, delta_norms = build_delta_matrix(prod)

    print()
    print('每个任务"自己造成的"更新 δW 之间的余弦（= Actual ΔW Cosine）')
    print(f'  {"":10s}' + ''.join(f'{t:>10s}' for t in task_labels)
          + f'{"‖δW‖":>10s}')
    for i, task in enumerate(task_labels):
        print(f'  {task:10s}' + ''.join(f'{delta_matrix[i][j]:10.3f}' for j in range(n))
              + f'{delta_norms[i] ** 0.5:10.2f}')

    save_matrix_csv(os.path.join(args.out_dir, 'task_delta_cosine.csv'), delta_matrix,
                    task_labels, task_labels)
    save_heatmap(delta_matrix, task_labels, task_labels,
                 os.path.join(args.out_dir, 'task_delta_cosine.png'),
                 'Actual per-task LoRA update cosine  cos(δW_i, δW_j)',
                 vmin=-1, vmax=1, cmap='coolwarm')

    # 逐层版本：哪几层的任务更新互相打架
    layer_delta = {}
    for layer in layers:
        names = [name for name in weight[stages[0]][0]
                 if module_layer(name)[0] == layer]
        if not names:
            continue

        def layer_prod(a, b, names=names):
            if a is None or b is None:
                return 0.0
            modules_a, scaling_a = weight[a]
            modules_b, scaling_b = weight[b]
            usable = [name for name in names if name in modules_a and name in modules_b]
            if not usable:
                return 0.0
            dot, _, _ = inner_products(modules_a, scaling_a, modules_b, scaling_b,
                                       names=usable)
            return dot

        def layer_norm_sq(a, names=names):
            if a is None:
                return 0.0
            modules_a, scaling_a = weight[a]
            usable = [name for name in names if name in modules_a]
            _, na, _ = inner_products(modules_a, scaling_a, modules_a, scaling_a,
                                      names=usable)
            return na

        size = len(chain)
        values = [[0.0] * size for _ in range(size)]
        norms = [0.0] * size
        for i in range(1, size):
            for j in range(1, size):
                values[i][j] = (layer_prod(chain[i], chain[j])
                                - layer_prod(chain[i - 1], chain[j])
                                - layer_prod(chain[i], chain[j - 1])
                                + layer_prod(chain[i - 1], chain[j - 1]))
        for i in range(1, size):
            norms[i] = (layer_norm_sq(chain[i])
                        - 2 * layer_prod(chain[i], chain[i - 1])
                        + layer_norm_sq(chain[i - 1]))

        rows = []
        for i in range(1, size):
            rows.append([
                values[i][j] / (((max(norms[i], 0.0) ** 0.5)
                                 * (max(norms[j], 0.0) ** 0.5)) + 1e-24)
                for j in range(1, size)
            ])
        layer_delta[layer] = rows

    if layer_delta:
        # 每个任务对做成一行，列是层，方便看"哪几层在打架"
        pair_labels, pair_rows = [], []
        for i in range(n):
            for j in range(i + 1, n):
                pair_labels.append(f'{task_labels[i]} vs {task_labels[j]}')
                pair_rows.append([layer_delta[layer][i][j] for layer in layers
                                  if layer in layer_delta])
        save_matrix_csv(os.path.join(args.out_dir, 'layer_task_delta_cosine.csv'), pair_rows,
                        pair_labels, [f'layer {layer}' for layer in layers
                                      if layer in layer_delta])
        save_heatmap(pair_rows, pair_labels,
                     [f'L{layer}' for layer in layers if layer in layer_delta],
                     os.path.join(args.out_dir, 'layer_task_delta_cosine.png'),
                     'Per-layer cos(δW_i, δW_j)  [task-vs-task update conflict]',
                     vmin=-1, vmax=1, cmap='coolwarm')

    save_json(os.path.join(args.out_dir, 'summary.json'), {
        'adapter_root': args.adapter_root,
        'stages': stages,
        'task_labels': task_labels,
        'stage_cosine': matrix,
        'stage_drift': drift_rows,
        'task_delta_cosine': delta_matrix,
        'task_delta_norm': [norm ** 0.5 for norm in delta_norms],
        'layer_task_delta_cosine': layer_delta,
        'layer_cosine': layer_matrix,
        'layer_update_cosine': layer_update_matrix,
        'layers': layers,
    })
    print()
    print(f'结果目录：{args.out_dir}')


if __name__ == '__main__':
    main()
