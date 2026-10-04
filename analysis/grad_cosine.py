"""Gradient Cosine —— 任务之间的梯度冲突分析。

做法：在某个训练阶段上，用每个任务的一小批样本分别算一次 LoRA 参数的梯度，
再比较这些梯度向量的余弦：

    cos(g_i, g_j) > 0   两个任务的更新方向一致（互相帮助）
                ≈ 0     基本不干扰
                < 0     互相冲突，一起训会互相拖（PCGrad 那篇的设定）

关键点：同一阶段、同一批样本、同一个 loss（只对回答段算，和 train1.py 一致），
所以差异只来自"任务"本身。梯度是"当前权重处的局部方向"，换个阶段算结果会变，
所以脚本支持一次跑多个阶段（--stages），看冲突是怎么随训练演化的。

损失与归一化：每个 batch 直接用模型自己的 loss —— labels 里的 -100 已经被排除，
所以它本身就是"该 batch 内有效 token 的平均"；各 batch 的梯度累加后**除以 batch 个数**。
不要再除一次 batch_size：那既不是"样本平均梯度"，又会让样本数不足的尾批被额外放大。
想把"每个样本等权"（长回答不被 token 数放大）用 --loss-weight sample。
顺带一提：余弦对梯度整体的正数缩放不敏感，所以这两种口径主要影响打印出来的 ||g||
（以及尾批权重），对余弦矩阵本身影响很小（实测 ~1e-4 量级）。

显存开销：8B bf16 权重约 16 GB，其余靠梯度检查点压住（默认开）。
batch 4 + 512 token 一般够用，不够会自动折半重试。

用法：
  python3 analysis/grad_cosine.py --adapter AGNEWS+QQP --tasks AGNEWS QQP SST2 TLDR
  python3 analysis/grad_cosine.py --stages base AGNEWS AGNEWS+QQP AGNEWS+QQP+SST2 \
      --tasks AGNEWS QQP SST2 TLDR --repeats 3
"""

import argparse
import os

import torch

from _common import (
    ADAPTER_ROOT,
    cosine_sim,
    DATA_ROOT,
    MODEL_PATH,
    TASKS,
    add_common_args,
    freeze_dropout,
    load_model,
    load_tokenizer,
    pad_batch,
    prompt_lengths,
    prompt_of,
    read_examples,
    release,
    resolve_args,
    save_heatmap,
    save_json,
    save_matrix_csv,
    to_device,
)

MODULE_ORDER = ['q_proj', 'k_proj', 'v_proj', 'o_proj']


def parse_args():
    parser = argparse.ArgumentParser(description='任务梯度冲突（gradient cosine）分析')
    add_common_args(parser, default_max_samples=16)
    # 和其它三个指标保持一致：默认在 val2 上算。
    # val2 既不参与训练、也不参与 checkpoint 选择（train1.py 用 train 训练、val1 选模型），
    # 是干净的留出集；想看在训练数据上的梯度冲突可以用 --split train。
    parser.set_defaults(batch_size=2, max_length=512, split='val2')
    parser.add_argument('--adapter', default=None,
                        help='只分析这一个阶段（adapter 目录名；base = 未训练的基座）')
    parser.add_argument('--stages', nargs='*', default=None,
                        help='分析多个阶段，按训练顺序给；不填则用 --adapter')
    parser.add_argument('--repeats', type=int, default=3,
                        help='每个任务重复采样几批（结果取平均，默认 3）')
    parser.add_argument('--loss-weight', default='token', choices=['token', 'sample'],
                        help='损失口径。token（默认）= 和 train1.py 一样的口径，'
                             'batch 内所有有效 token 平均；sample = 每个样本先按自己的 '
                             '有效 token 平均、再对样本平均（回答长的样本不会被放大，'
                             '但显存更吃、且不是训练时的目标）')
    parser.add_argument('--no-grad-checkpoint', dest='grad_checkpoint',
                        action='store_false',
                        help='关掉梯度检查点（更吃显存，但快）')
    return parser.parse_args()


def build_lora_for_base(model_path, seed=0):
    """base 阶段没有 LoRA，临时挂一个随机初始化的（只看"初始冲突"）。

    为什么非要挂一个：LoRA 训练里基座参数是冻结的，纯基座的 requires_grad 全是
    False，backward 之后拿不到任何梯度，根本上不了"梯度余弦"这张表。而其它阶段都是
    LoRA 形态，为了同构只能临时挂一个。

    两个细节：
      - 先 manual_seed：保证同一个 seed 下随机 LoRA 完全可复现，
        不受"base 排在第几个阶段"影响；
      - lora_B 的默认初始化是 0，而 B=0 时 dL/dA 恒等于 0（A 完全没梯度），
        所以这里给 B 加了一个小扰动，让 A、B 两边都有梯度。
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)

    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                               device_map='auto')
    base.config.use_cache = False
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                        bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(base, config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.normal_(0, 0.01)
    return model


def lora_params(model):
    """{参数名: 参数}，只取需要梯度的 LoRA 参数。"""
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def per_sample_loss(model, batch):
    """每个样本先对自己的有效 token 求平均，再对 batch 内的样本求平均。

    和 HF 默认的 `out.loss` 的区别：默认口径是把 batch 内所有有效 token 放在一起
    平均，回答长的样本会因为 token 多而被放大。MEDDIA 的回答长度从 2 到 472 token
    不等，两种口径差别很大。
    """
    import torch.nn.functional as F

    labels = batch['labels']
    inputs = {key: value for key, value in batch.items() if key != 'labels'}
    logits = model(**inputs).logits[:, :-1, :].float()      # 和 HF 一样 upcast 到 fp32
    shift_labels = labels[:, 1:]
    per_token = F.cross_entropy(logits.transpose(1, 2), shift_labels,
                                ignore_index=-100, reduction='none')
    valid = (shift_labels != -100).sum(dim=1).clamp(min=1)
    return (per_token.sum(dim=1) / valid).mean()


def task_gradients(model, tokenizer, examples, args, params):
    """一个任务的平均梯度：整体向量 + 按模块拆开的向量。

    归一化口径：每个 batch 用模型自己的 loss（labels 里 -100 已被排除，等于
    "该 batch 内有效 token 的平均"），把各 batch 的梯度加起来后**除以 batch 个数**。
    不能再除 batch_size —— 那样会变成 (1/B)·(1/N)·ΣL，既不是"样本平均梯度"，
    又会让样本数不足的尾批被额外放大。
    """
    device = next(model.parameters()).device
    texts = [example['train_text'] for example in examples]
    prompts = [prompt_of(example) for example in examples]
    prompt_lens = prompt_lengths(tokenizer, prompts, args.max_length)

    tags = []
    for text, plen in zip(texts, prompt_lens):
        ids = tokenizer(text, add_special_tokens=False, truncation=True,
                        max_length=args.max_length)['input_ids']
        keep = min(plen, len(ids))
        tags.append([-100] * keep + list(ids[keep:]))

    losses = []
    index, current = 0, max(1, args.batch_size)
    batches = 0
    model.zero_grad(set_to_none=True)

    while index < len(texts):
        try:
            batch = to_device(
                pad_batch(tokenizer, texts[index:index + current],
                          labels=tags[index:index + current],
                          max_length=args.max_length),
                device,
            )
            # 只对回答段算 loss（提问段在 tags 里是 -100），和 train1.py 一致
            if args.loss_weight == 'sample':
                loss = per_sample_loss(model, batch)
            else:
                loss = model(**batch).loss
            loss.backward()
            losses.append(float(loss.detach()))
            batches += 1
            index += current
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f'    [grad] 显存不够，batch 降到 {current} 重试')

    if batches:
        for param in params.values():
            if param.grad is not None:
                param.grad.div_(batches)

    overall, by_module = [], {module: [] for module in MODULE_ORDER}
    for name, param in params.items():
        if param.grad is None:
            continue
        flat = param.grad.detach().reshape(-1).float().cpu()
        overall.append(flat)
        for module in MODULE_ORDER:
            if f'.{module}.' in name:
                by_module[module].append(flat)
                break

    model.zero_grad(set_to_none=True)
    return {
        'overall': torch.cat(overall) if overall else torch.zeros(1),
        'by_module': {module: (torch.cat(pieces) if pieces else torch.zeros(1))
                      for module, pieces in by_module.items()},
        'loss': sum(losses) / max(len(losses), 1),
    }


def cosine_matrix(vectors):
    n = len(vectors)
    return [[cosine_sim(vectors[i], vectors[j]) for j in range(n)] for i in range(n)]


def analyse_stage(stage, args, tokenizer):
    """跑一个阶段：返回该阶段下任务两两的梯度余弦（含 repeat 的均值/标准差）。"""
    print()
    print('=' * 70)
    print(f'阶段：{stage}' + ('（base = 随机初始化 LoRA，只看初始冲突）' if stage == 'base' else ''))
    print('=' * 70)

    if stage == 'base':
        model = build_lora_for_base(args.model_path, args.seed)
    else:
        model = load_model(args.model_path, adapter=stage,
                           adapter_root=args.adapter_root, trainable=True)

    # 梯度检查点 + 关掉 dropout：省显存，同时保证前向确定
    model.train()
    freeze_dropout(model)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    params = lora_params(model)
    print(f'  可训练 LoRA 参数：{sum(p.numel() for p in params.values()):,}')

    data = {}
    for task in args.tasks:
        examples = read_examples(args.data_root, task, args.split)
        if not examples:
            continue
        data[task] = examples

    task_names = list(data)
    repeat_matrices, repeat_module_matrices, norms, losses = [], [], {t: [] for t in task_names}, {}

    for repeat in range(args.repeats):
        vectors, module_vectors = {}, {}
        for task in task_names:
            examples = data[task]
            # 每轮换一批样本：从不同的偏移开始截取
            offset = (repeat * args.max_samples) % max(len(examples) - args.max_samples + 1, 1)
            chunk = examples[offset:offset + args.max_samples]
            result = task_gradients(model, tokenizer, chunk, args, params)
            vectors[task] = result['overall']
            module_vectors[task] = result['by_module']
            norms[task].append(float(result['overall'].norm()))
            losses.setdefault(task, []).append(result['loss'])
        repeat_matrices.append(cosine_matrix([vectors[t] for t in task_names]))
        repeat_module_matrices.append({
            module: cosine_matrix([module_vectors[t][module] for t in task_names])
            for module in MODULE_ORDER
        })
        print(f'  repeat {repeat + 1}/{args.repeats} 完成：'
              + '  '.join(f'{t}||g||={norms[t][-1]:.3f}' for t in task_names))

    n = len(task_names)
    mean_matrix = [[sum(m[i][j] for m in repeat_matrices) / len(repeat_matrices)
                    for j in range(n)] for i in range(n)]
    std_matrix = [[float(torch.tensor([m[i][j] for m in repeat_matrices]).std())
                   for j in range(n)] for i in range(n)]
    module_mean = {
        module: [[sum(m[i][j] for m in [r[module] for r in repeat_module_matrices])
                  / len(repeat_matrices) for j in range(n)] for i in range(n)]
        for module in MODULE_ORDER
    }

    print()
    print('  任务梯度余弦（repeat 平均）')
    print(' ' * 10 + ''.join(f'{t:>10s}' for t in task_names))
    for i, task in enumerate(task_names):
        print(f'  {task:8s}' + ''.join(f'{mean_matrix[i][j]:10.3f}' for j in range(n)))

    release(model)

    return {
        'stage': stage,
        'tasks': task_names,
        'cosine_mean': mean_matrix,
        'cosine_std': std_matrix,
        'cosine_by_module': module_mean,
        'grad_norm_mean': {t: sum(norms[t]) / len(norms[t]) for t in task_names},
        'loss_mean': {t: sum(losses[t]) / len(losses[t]) for t in task_names},
    }


def main():
    args = resolve_args(parse_args(), 'grad_cosine')

    stages = args.stages if args.stages else ([args.adapter] if args.adapter else [])
    stages = [stage for stage in stages if stage]

    if not stages:
        available = sorted(
            name for name in os.listdir(args.adapter_root)
            if os.path.exists(os.path.join(args.adapter_root, name, 'adapter_config.json'))
        ) if os.path.isdir(args.adapter_root) else []
        raise SystemExit(
            '请指定要分析的阶段：\n'
            '  --adapter AGNEWS            单个阶段\n'
            '  --stages AGNEWS AGNEWS+QQP  多个阶段（按训练顺序）\n'
            f'可用的 adapter：{available}\n'
            '注意：--stages base 用"随机初始化的 LoRA"代替真正的基座 —— 基座参数在 LoRA\n'
            '训练里是冻结的、拿不到梯度，只能临时挂一个；它只适合当"初始冲突"的参考，\n'
            '不要和训练后的阶段直接比较。'
        )

    tokenizer = load_tokenizer(args.model_path)
    all_results = []
    for stage in stages:
        if stage != 'base':
            path = os.path.join(args.adapter_root, stage)
            if not os.path.exists(os.path.join(path, 'adapter_config.json')):
                raise SystemExit(f'不是有效的 adapter 目录：{path}')
        all_results.append(analyse_stage(stage, args, tokenizer))

    save_json(os.path.join(args.out_dir, 'grad_cosine.json'), {
        'split': args.split,
        'max_samples': args.max_samples,
        'repeats': args.repeats,
        'max_length': args.max_length,
        'stages': all_results,
    })

    for result in all_results:
        label = result['stage'].replace('+', '_')
        save_matrix_csv(os.path.join(args.out_dir, f'cosine_{label}.csv'),
                        result['cosine_mean'], result['tasks'], result['tasks'])
        save_heatmap(result['cosine_mean'], result['tasks'], result['tasks'],
                     os.path.join(args.out_dir, f'cosine_{label}.png'),
                     f'Task gradient cosine @ {result["stage"]}',
                     vmin=-1, vmax=1, cmap='coolwarm')

    print()
    print(f'结果目录：{args.out_dir}')


if __name__ == '__main__':
    main()
