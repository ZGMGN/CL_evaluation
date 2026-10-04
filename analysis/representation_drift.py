"""Representation Drift —— 训练前后内部表示漂移。

做法：同一批 val 样本，分别过"训练前"和"训练后"两个模型，取若干层的 hidden
state 做 masked mean pooling，然后逐样本配对比较：

    cos(h_before, h_after)      表示方向变了多少（1 = 完全没变）
    1 - cos                     余弦距离，越大漂移越大
    ||Δh|| / ||h_before||       相对幅度变化
    CKA(before, after)          两组表示的整体几何相似度（不要求配对）

为什么用配对余弦：样本是同一批，逐样本比较能直接看出"这个样本的表示被推走多远"；
CKA 则回答"表示空间的整体结构有没有被重排"。

显存：一次只加载一个模型（另一个的中间结果缓存在 .pt 里），24G 卡安全。
两个 pass 的结果都会缓存，重跑时加 --reuse 可跳过前向。

用法：
  python3 analysis/representation_drift.py --before base --after AGNEWS
  python3 analysis/representation_drift.py --before AGNEWS+QQP --after AGNEWS+QQP+SST2 \
      --tasks AGNEWS QQP SST2 --layers 8 16 24 32
"""

import argparse
import csv
import os

import torch

from _common import (
    DATA_ROOT,
    LAYERS,
    MODEL_PATH,
    ADAPTER_ROOT,
    add_common_args,
    linear_cka,
    load_model,
    load_tokenizer,
    pad_batch,
    paired_cosine,
    prompt_of,
    read_examples,
    release,
    resolve_args,
    run_over_batches,
    save_heatmap,
    save_json,
    save_matrix_csv,
    to_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description='表示漂移（representation drift）分析')
    add_common_args(parser, with_layers=True)
    parser.add_argument('--before', default='base',
                        help='训练前的阶段：adapter 目录名，或 base（基座模型）')
    parser.add_argument('--after', required=True,
                        help='训练后的阶段：adapter 目录名')
    parser.add_argument('--reuse', action='store_true',
                        help='复用上次缓存的 hidden state，不重新前向')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='判定"基本没动"的阈值，同时用在余弦和 CKA 上（默认 0.9）')
    return parser.parse_args()


def extract(model, tokenizer, examples, layers, args):
    """对一批样本取指定层的 masked mean pooling。返回 {layer: [n, H]}。"""
    device = next(model.parameters()).device
    collected = {layer: [] for layer in layers}

    def run(batch_examples):
        batch_texts = [prompt_of(example) for example in batch_examples]
        batch = to_device(pad_batch(tokenizer, batch_texts, max_length=args.max_length), device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)

        mask = batch['attention_mask'].unsqueeze(-1).float()
        counts = mask.sum(dim=1).clamp(min=1)
        for layer in layers:
            hidden = outputs.hidden_states[layer].float()
            pooled = (hidden * mask).sum(dim=1) / counts
            collected[layer].append(pooled.cpu())
        return [None] * len(batch_examples)

    run_over_batches(examples, run, batch_size=args.batch_size, label='repr')
    torch.cuda.empty_cache()
    return {layer: torch.cat(chunks).half() for layer, chunks in collected.items()}


def run_pass(stage, args, tokenizer):
    """跑一个模型的 pass，结果缓存成 .pt。stage='base' 表示不叠 adapter。"""
    cache = os.path.join(args.out_dir, f'repr_{stage.replace("+", "_")}.pt')
    if args.reuse and os.path.exists(cache):
        print(f'  复用缓存 {cache}')
        return torch.load(cache, weights_only=False)

    adapter = None if stage == 'base' else stage
    print('  加载模型：' + (f'{stage}（纯基座，不叠任何 adapter）' if adapter is None
                          else f'{stage}（基座 + LoRA adapter）'))
    model = load_model(args.model_path, adapter=adapter, adapter_root=args.adapter_root)

    result = {}
    for task in args.tasks:
        examples = read_examples(args.data_root, task, args.split, args.max_samples)
        if not examples:
            continue
        print(f'    [{task}] {len(examples)} 条')
        result[task] = extract(model, tokenizer, examples, args.layers, args)

    release(model)
    torch.save(result, cache)
    print(f'  缓存到 {cache}')
    return result


def linear_map_r2(a, b, ridge=1e-3, seed=0):
    """判断"漂移是不是一个整体线性变换"：半拟半测的 R²。

    做法：在一半样本上拟合线性映射 a→b（岭回归），在另一半上算 R²。

      R² ≈ 1  → 训练前后的表示由一个线性映射连着，只是被整体"搬"走了
                （对应 CKA≈1、余弦下降那种"旋转/缩放"情形）
      R² 低   → 不是整体变换，而是局部/非线性的重排

    用核形式求解（n ≪ d 时比直接解 d×d 的方程快几个数量级）：
        A = Xᵀ (XXᵀ + λI)⁻¹ Y
    """
    a = a.double()
    b = b.double()
    n = a.shape[0]
    if n < 32:      # 样本太少时 R² 没意义，宁可给 nan
        return float('nan')
    torch.manual_seed(seed)
    index = torch.randperm(n)
    fit, test = index[: n // 2], index[n // 2:]

    x = a[fit] - a[fit].mean(dim=0, keepdim=True)
    y = b[fit] - b[fit].mean(dim=0, keepdim=True)
    lam = ridge * float(torch.trace(x @ x.t())) / max(len(fit), 1)
    kernel = x @ x.t() + lam * torch.eye(len(fit), dtype=x.dtype)
    alpha = torch.linalg.solve(kernel, y)

    xt = a[test] - a[fit].mean(dim=0, keepdim=True)
    pred = (xt @ x.t()) @ alpha + b[fit].mean(dim=0, keepdim=True)
    target = b[test]
    resid = ((target - pred) ** 2).sum()
    total = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum()
    return float(1 - resid / (total + 1e-12))


def quadrant(cos, cka, r2=float('nan'), threshold=0.9):
    """把 (余弦, CKA) 两个数合起来给一个判定。

    余弦回答"同一条样本的表示变没变"；CKA 回答"整个表示空间的结构变没变"。
    两个一起看才有意义：
      - 都高            → 基本没动
      - 余弦低 + CKA 高  → 被整体旋转/缩放（保距变换），几何结构还在
      - 余弦低 + CKA 低 + 线性 R² 高 → 一般线性映射：能被线性解释，但拉伸了几何
      - 余弦低 + CKA 低 + 线性 R² 低 → 非线性重排，这才是真正的表示漂移
      - 余弦高 + CKA 低  → 方向基本没变，但相对幅度/夹角变了（尺度重排）
    """
    if cos != cos or cka != cka:
        return 'n/a'
    if cos >= threshold and cka >= threshold:
        return '基本没动'
    if cos >= threshold:
        return '方向没变但尺度/几何变了（幅度重排）'
    if cka >= threshold:
        return '整体保距变换（旋转/缩放，几何保持）'
    if r2 == r2 and r2 >= threshold:
        return '线性但非保距（一般线性映射）'
    if r2 == r2:
        return '非线性重排（真漂移）'
    return '重排（样本不足，R² 不可用）'


def main():
    args = resolve_args(parse_args(), 'representation_drift')
    tokenizer = load_tokenizer(args.model_path)

    # 报告写到 <out>/<before>__<after>/ 子目录，中间结果 .pt 还是放在 <out>/ 里共用
    # —— 不然连着跑多段时后一段会把前一段的 csv/png 覆盖掉。
    report_dir = os.path.join(args.out_dir, f'{args.before}__{args.after}')
    os.makedirs(report_dir, exist_ok=True)

    print(f'数据：{args.split}，每任务最多 {args.max_samples} 条；层：{args.layers}')
    print(f'对比：{args.before}  ->  {args.after}')
    print(f'报告：{report_dir}')
    print()
    print('pass 1/2（训练前）')
    before = run_pass(args.before, args, tokenizer)
    print()
    print('pass 2/2（训练后）')
    after = run_pass(args.after, args, tokenizer)

    # ----------------------------------------
    # 对比
    # ----------------------------------------
    tasks = [task for task in args.tasks if task in before and task in after]
    layers = args.layers

    cos_map = {layer: [] for layer in layers}
    l2_map = {layer: [] for layer in layers}
    cka_map = {layer: [] for layer in layers}
    r2_map = {layer: [] for layer in layers}
    detail = {}

    for task in tasks:
        detail[task] = {}
        for layer in layers:
            a = before[task][layer].float()
            b = after[task][layer].float()
            cos = paired_cosine(a, b)
            l2 = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp(min=1e-6)
            cka = linear_cka(a, b)
            r2 = linear_map_r2(a, b, seed=args.seed)
            detail[task][layer] = {
                'cosine_mean': float(cos.mean()),
                'cosine_std': float(cos.std()) if len(cos) > 1 else 0.0,
                'cosine_dist_mean': float(1 - cos.mean()),
                'relative_l2_mean': float(l2.mean()),
                'cka': cka,
                # 漂移能不能被一个线性映射解释（1 = 完全可以）
                'linear_map_r2': r2,
                'verdict': quadrant(float(cos.mean()), cka, r2, args.threshold),
                'n': int(a.shape[0]),
            }
            cos_map[layer].append(float(cos.mean()))
            l2_map[layer].append(float(l2.mean()))
            cka_map[layer].append(cka)
            r2_map[layer].append(r2)

    # 表格：行=层，列=任务
    for name, matrix in [('cosine', cos_map), ('rel_l2', l2_map), ('cka', cka_map),
                         ('linear_r2', r2_map)]:
        rows = [matrix[layer] for layer in layers]
        save_matrix_csv(os.path.join(report_dir, f'{name}.csv'), rows,
                        [f'layer {layer}' for layer in layers], tasks, fmt='{:.4f}')

    save_heatmap([cos_map[layer] for layer in layers],
                 [f'L{layer}' for layer in layers], tasks,
                 os.path.join(report_dir, 'cosine.png'),
                 f'Representation cosine  {args.before} -> {args.after}',
                 vmin=0.0, vmax=1.0, cmap='viridis')
    save_heatmap([l2_map[layer] for layer in layers],
                 [f'L{layer}' for layer in layers], tasks,
                 os.path.join(report_dir, 'rel_l2.png'),
                 f'Relative L2 of representation  {args.before} -> {args.after}',
                 cmap='magma')
    save_heatmap([cka_map[layer] for layer in layers],
                 [f'L{layer}' for layer in layers], tasks,
                 os.path.join(report_dir, 'cka.png'),
                 f'Linear CKA  {args.before} -> {args.after}',
                 vmin=0.0, vmax=1.0, cmap='viridis')
    save_heatmap([r2_map[layer] for layer in layers],
                 [f'L{layer}' for layer in layers], tasks,
                 os.path.join(report_dir, 'linear_r2.png'),
                 f'R2 of a linear map  (1 = drift is a global linear transform)  '
                 f'{args.before} -> {args.after}',
                 vmin=0.0, vmax=1.0, cmap='viridis')

    print()
    print('表示余弦（1 = 完全没动）')
    print(' ' * 10 + ''.join(f'{t:>10s}' for t in tasks))
    for layer in layers:
        print(f'  L{layer:<7d}' + ''.join(f'{v:10.4f}' for v in cos_map[layer]))

    # ----------------------------------------
    # 余弦 vs CKA：两个问题一起看
    # ----------------------------------------
    verdict_rows = []
    for layer in layers:
        for j, task in enumerate(tasks):
            info = detail[task][layer]
            verdict_rows.append([
                task, f'L{layer}',
                f'{info["cosine_mean"]:.4f}', f'{info["cka"]:.4f}',
                f'{info["linear_map_r2"]:.4f}', info['verdict'],
            ])

    with open(os.path.join(report_dir, 'interpretation.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['task', 'layer', 'cosine', 'cka', 'linear_map_r2', 'verdict'])
        writer.writerows(verdict_rows)
    print(f'  写出 {os.path.join(report_dir, "interpretation.csv")}')

    print()
    print('余弦（单样本有没有变） vs CKA（整体几何有没有变） vs 线性映射 R²')
    print(f'  {"任务":8s} {"层":5s} {"余弦":>8s} {"CKA":>8s} {"线性R²":>8s}   判定')
    for task, layer, cos, cka, r2, verdict in verdict_rows:
        print(f'  {task:8s} {layer:5s} {cos:>8s} {cka:>8s} {r2:>8s}   {verdict}')

    save_json(os.path.join(report_dir, 'summary.json'), {
        'before': args.before,
        'after': args.after,
        'split': args.split,
        'max_samples': args.max_samples,
        'layers': layers,
        'tasks': tasks,
        'per_task_per_layer': detail,
    })
    print()
    print(f'结果目录：{args.out_dir}')


if __name__ == '__main__':
    main()
