"""Functional Drift —— 训练前后"输出分布"漂移。

定义：对同一批样本、同一个位置，比较模型给出的下一个 token 分布

    D_func = KL(P_M1(y|x) || P_M2(y|x))          （不对称）
    D_sym  = KL(M1||M2) + KL(M2||M1)             （对称，默认报这个）
    D_JS   = Jensen-Shannon（有界，0~ln2，跨任务可比性更好）

位置怎么取：默认只看"回答的第一个 token"（logits 在提问最后一个位置上），
这对分类任务就是模型要吐标签的那个决策；加 --positions K 会再看回答的前 K 个
token（K>1 会显著增加显存/磁盘占用，因为每条样本要存 K 个 128k 维分布）。

顺带记录两个很有用的诊断量：
  - 分布的熵：塌缩（比如总答 "0"）时熵会明显下降；
  - top-1 一致率 + 金标准首 token 的概率/排名：直接反映"答案有没有变"。

用法：
  python3 analysis/functional_drift.py --before base --after AGNEWS
  python3 analysis/functional_drift.py --before TLDR --after TLDR+AGNEWS+QQP \
      --tasks TLDR AGNEWS --split test --max-samples 200
"""

import argparse
import os

import torch

from _common import (
    add_common_args,
    entropy,
    gold_answer,
    js_divergence,
    load_model,
    load_tokenizer,
    prompt_of,
    read_examples,
    release,
    resolve_args,
    save_heatmap,
    save_json,
    save_matrix_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description='功能漂移（输出分布 KL）分析')
    add_common_args(parser, default_max_samples=200)
    parser.add_argument('--before', default='base')
    parser.add_argument('--after', required=True)
    parser.add_argument('--positions', type=int, default=1,
                        help='统计回答的前几个 token 位置（默认 1）')
    parser.add_argument('--save-probs', action='store_true',
                        help='把完整分布也存下来（很大，默认不存）')
    parser.add_argument('--reuse', action='store_true',
                        help='复用上次缓存的分布')
    return parser.parse_args()


def compute_distributions(model, tokenizer, examples, args):
    """返回 probs [n, K, V] (fp16, CPU) 和金标准前 K 个 token id [n, K]。

    第 k 个位置的做法：喂 prompt + 前 k 个答案 token，取 prompt 末尾那个位置的
    logits —— 那正是模型在生成第 k 个答案 token 时看到的分布。
    """
    device = next(model.parameters()).device
    probs, golds, skipped = [], [], []
    prompts = [prompt_of(example) for example in examples]
    answers = [gold_answer(example) for example in examples]

    model.config.use_cache = False

    items = list(range(len(examples)))
    batch_size = max(1, args.batch_size)
    index = 0
    while index < len(items):
        chunk = items[index:index + batch_size]
        try:
            input_ids, attention = [], []
            lengths, gold_ids_batch = [], []
            for i in chunk:
                prompt_ids = tokenizer(prompts[i], add_special_tokens=False,
                                       truncation=True, max_length=args.max_length)['input_ids']
                gold_ids = tokenizer(answers[i], add_special_tokens=False,
                                     truncation=True, max_length=args.positions)['input_ids']
                if not prompt_ids:
                    continue
                # 看回答的前 K 个位置：喂 prompt + 前 K-1 个答案 token，
                # 取倒数第 K 个位置往后的 logits
                take = max(0, args.positions - 1)
                ids = prompt_ids + gold_ids[:take]
                if len(ids) > args.max_length:
                    ids = ids[:args.max_length]
                if len(ids) < len(prompt_ids) + take:
                    # 提问太长被截断，这个样本取不满 K 个位置，跳过
                    skipped.append(i)
                    continue
                input_ids.append(ids)
                attention.append([1] * len(ids))
                lengths.append((len(ids), len(prompt_ids)))
                gold_ids_batch.append(gold_ids)

            if not input_ids:
                index += len(chunk)
                continue

            width = max(len(ids) for ids in input_ids)
            pad_id = tokenizer.pad_token_id
            padded = [[pad_id] * (width - len(ids)) + ids for ids in input_ids]
            mask = [[0] * (width - len(ids)) + att for ids, att in zip(input_ids, attention)]
            batch_tensors = {
                'input_ids': torch.tensor(padded, dtype=torch.long, device=device),
                'attention_mask': torch.tensor(mask, dtype=torch.long, device=device),
            }

            with torch.no_grad():
                logits = model(**batch_tensors).logits  # [B, L, V]

            for row, (length, prompt_len) in enumerate(lengths):
                start = prompt_len - 1
                end = min(length, start + args.positions)
                window = logits[row, start:end].float()
                if window.shape[0] < args.positions:
                    window = torch.nn.functional.pad(
                        window, (0, 0, 0, args.positions - window.shape[0]))
                probs.append(torch.softmax(window, dim=-1).half().cpu())
                gid = gold_ids_batch[row]
                gid = (gid + [tokenizer.pad_token_id] * args.positions)[:args.positions]
                golds.append(torch.tensor(gid, dtype=torch.long))

            index += len(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f'    [func] 显存不够，batch 降到 {batch_size} 重试')

    if skipped:
        print(f'    跳过 {len(skipped)} 条（提问被 max_length 截断，取不满 {args.positions} 个位置）')
    if not probs:
        raise SystemExit('没有任何样本可用，检查 --max-length / --positions')
    return torch.stack(probs), torch.stack(golds)


def run_pass(stage, args, tokenizer):
    cache = os.path.join(args.out_dir, f'probs_{stage.replace("+", "_")}.pt')
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
        probs, golds = compute_distributions(model, tokenizer, examples, args)
        result[task] = {'probs': probs, 'golds': golds}

    release(model)
    if args.save_probs:
        torch.save(result, cache)
        print(f'  缓存到 {cache}')
    return result


def main():
    args = resolve_args(parse_args(), 'functional_drift')
    tokenizer = load_tokenizer(args.model_path)

    # 报告写到 <out>/<before>__<after>/ 子目录，概率缓存 .pt 还是放在 <out>/ 里共用，
    # 这样连着跑多段时不会互相覆盖。
    report_dir = os.path.join(args.out_dir, f'{args.before}__{args.after}')
    os.makedirs(report_dir, exist_ok=True)

    per_sample = args.positions * tokenizer.vocab_size * 2 / 1024 ** 2
    print(f'数据：{args.split}，每任务最多 {args.max_samples} 条，位置数 K={args.positions}')
    print(f'（每条样本 {per_sample:.2f} MB，{len(args.tasks)} 个任务预计 '
          f'{per_sample * args.max_samples * len(args.tasks) / 1024:.2f} GB 内存）')
    print(f'对比：{args.before}  ->  {args.after}')
    print(f'报告：{report_dir}')

    print()
    print('pass 1/2（训练前）')
    before = run_pass(args.before, args, tokenizer)
    print()
    print('pass 2/2（训练后）')
    after = run_pass(args.after, args, tokenizer)

    tasks = [task for task in args.tasks if task in before and task in after]
    sym_kl, kl_fwd, kl_bwd, jsd, ent_b, ent_a, top1, gold_prob = ({} for _ in range(8))
    detail = {}

    for task in tasks:
        p = before[task]['probs'].float()      # [n, K, V]
        q = after[task]['probs'].float()
        gold = before[task]['golds']           # [n, K]
        n, k, vocab = p.shape

        p_flat = p.reshape(-1, vocab)
        q_flat = q.reshape(-1, vocab)
        gold_flat = gold.reshape(-1)

        kl_pq = (p_flat * (p_flat.clamp(min=1e-12).log() - q_flat.clamp(min=1e-12).log())).sum(-1)
        kl_qp = (q_flat * (q_flat.clamp(min=1e-12).log() - p_flat.clamp(min=1e-12).log())).sum(-1)

        sym_kl[task] = float((kl_pq + kl_qp).mean())
        kl_fwd[task] = float(kl_pq.mean())
        kl_bwd[task] = float(kl_qp.mean())
        jsd[task] = float(js_divergence(p_flat, q_flat).mean())
        ent_b[task] = float(entropy(p_flat).mean())
        ent_a[task] = float(entropy(q_flat).mean())
        top1[task] = float((p_flat.argmax(-1) == q_flat.argmax(-1)).float().mean())

        # 金标准首 token 的概率与排名。
        # 注意 p_flat 是按"样本优先"展开的（第 i 条样本的第 k 个位置在 i*k + j），
        # 所以"每个样本的第 0 个位置"是 p_flat[0::k]，不能直接取前 n 行。
        first_rows = p_flat[0::k]
        first_rows_after = q_flat[0::k]
        first_gold = gold[:, 0]
        pb = first_rows.gather(1, first_gold.unsqueeze(1)).squeeze(1)
        pa = first_rows_after.gather(1, first_gold.unsqueeze(1)).squeeze(1)
        rank_b = (first_rows > pb.unsqueeze(1)).sum(-1).float() + 1
        rank_a = (first_rows_after > pa.unsqueeze(1)).sum(-1).float() + 1
        gold_prob[task] = {
            'prob_before': float(pb.mean()),
            'prob_after': float(pa.mean()),
            'rank_before': float(rank_b.mean()),
            'rank_after': float(rank_a.mean()),
        }

        detail[task] = {
            'n': n,
            'positions': k,
            'symmetric_kl': sym_kl[task],
            'kl_before_to_after': kl_fwd[task],
            'kl_after_to_before': kl_bwd[task],
            'js_divergence': jsd[task],
            'entropy_before': ent_b[task],
            'entropy_after': ent_a[task],
            'top1_agreement': top1[task],
            **gold_prob[task],
        }

    # ----------------------------------------
    # 出表
    # ----------------------------------------
    metric_names = ['symmetric_kl', 'js_divergence', 'entropy_before', 'entropy_after',
                    'top1_agreement', 'prob_before', 'prob_after']
    rows, labels = [], []
    for name in metric_names:
        if name.startswith('entropy'):
            rows.append([detail[t][name] for t in tasks])
            labels.append(name)
        elif name.startswith('prob'):
            rows.append([detail[t][name] for t in tasks])
            labels.append('gold_prob_' + name.split('_')[1])
        else:
            rows.append([detail[t][name] for t in tasks])
            labels.append(name)

    save_matrix_csv(os.path.join(report_dir, 'metrics.csv'), rows, labels, tasks, fmt='{:.4f}')
    save_heatmap([detail[t]['symmetric_kl'] for t in tasks],
                 ['symmetric_kl'], tasks,
                 os.path.join(report_dir, 'symmetric_kl.png'),
                 f'Symmetric KL  {args.before} -> {args.after}')
    save_heatmap([[detail[t]['entropy_before'], detail[t]['entropy_after']] for t in tasks],
                 tasks, ['before', 'after'],
                 os.path.join(report_dir, 'entropy.png'),
                 f'Output entropy  {args.before} -> {args.after}')

    print()
    print(f'{"任务":8s} {"对称KL":>10s} {"JS":>10s} {"熵(before)":>12s} {"熵(after)":>11s} '
          f'{"top1一致":>9s} {"gold概率(b→a)":>18s}')
    for task in tasks:
        d = detail[task]
        print(f'{task:8s} {d["symmetric_kl"]:10.4f} {d["js_divergence"]:10.4f} '
              f'{d["entropy_before"]:12.4f} {d["entropy_after"]:11.4f} '
              f'{d["top1_agreement"]:9.4f} '
              f'{d["prob_before"]:8.4f}->{d["prob_after"]:.4f}')

    save_json(os.path.join(report_dir, 'summary.json'), {
        'before': args.before,
        'after': args.after,
        'split': args.split,
        'max_samples': args.max_samples,
        'positions': args.positions,
        'tasks': tasks,
        'detail': detail,
    })
    print()
    print(f'结果目录：{args.out_dir}')


if __name__ == '__main__':
    main()
