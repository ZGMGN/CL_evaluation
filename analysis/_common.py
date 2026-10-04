"""四个分析脚本共用的工具：数据/模型加载、OOM 自适应批处理、相似度指标、出表出图。

目录约定和 train1.py / evaluate1.py 保持一致：
  datas_clean/<任务>/{train,val1,val2,test}.jsonl
  train1_continual_lora/<任务前缀>/adapter_model.safetensors
"""

import csv
import json
import os

import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

MODEL_PATH = './models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master'
ADAPTER_ROOT = './train1_continual_lora'
DATA_ROOT = './datas_clean'

EOT = '<|eot_id|>'

# 和 prepare_activation.py 保持一致，方便和已有相似度结果对照
LAYERS = [8, 12, 16, 20, 24, 28, 32]

TASKS = [
    'AGNEWS', 'AMAZON', 'DBPEDIA', 'IMDB', 'MEDDIA',
    'QQP', 'SST2', 'TLDR', 'WMT', 'YELP',
]


# ============================================================
# 命令行
# ============================================================

def add_common_args(parser, with_layers=False, default_max_samples=200):
    parser.add_argument('--model-path', default=MODEL_PATH,
                        help='基座模型目录')
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help='清洗后的数据目录')
    parser.add_argument('--adapter-root', default=ADAPTER_ROOT,
                        help='LoRA adapter 根目录（train1.py 的输出）')
    parser.add_argument('--tasks', nargs='*', default=None,
                        help='要分析的任务，空格分隔；不填=全部 10 个')
    parser.add_argument('--split', default='val2',
                        choices=['train', 'val1', 'val2', 'test'],
                        help='用哪个切分的数据。默认 val2 —— 它既不参与训练也不参与 '
                             'checkpoint 选择（train1.py 用 train 训练、val1 选模型），'
                             '是干净的留出集；和 prepare_activation.py 也一致')
    parser.add_argument('--max-samples', type=int, default=default_max_samples,
                        help='每个任务最多取多少条')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批大小（显存不够会自动折半重试）')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='分词最大长度')
    parser.add_argument('--out-dir', default=None,
                        help='结果目录；不填则用 ./analysis_results/<脚本名>/')
    parser.add_argument('--seed', type=int, default=0)
    if with_layers:
        parser.add_argument('--layers', type=int, nargs='*', default=LAYERS,
                            help=f'取哪几层的 hidden state，默认 {LAYERS}')
    return parser


def resolve_args(args, script_name):
    """补上默认值：任务列表、输出目录、随机种子。"""
    if hasattr(args, 'tasks'):
        args.tasks = args.tasks if args.tasks else list(TASKS)
    args.out_dir = getattr(args, 'out_dir', None) or os.path.join('analysis_results', script_name)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(getattr(args, 'seed', 0))
    return args


# ============================================================
# 数据
# ============================================================

def read_examples(data_root, task, split, max_samples=None):
    path = os.path.join(data_root, task, f'{split}.jsonl')
    if not os.path.exists(path):
        print(f'  [{task}] 跳过：找不到 {path}')
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def prompt_of(example):
    """提问部分（自带 <|begin_of_text|>，末尾停在 assistant 头）。"""
    return example['activation_text']


def gold_answer(example):
    """标准答案 = train_text 里 activation_text 之后、<|eot_id|> 之前的部分。"""
    return example['train_text'][len(example['activation_text']):].replace(EOT, '').strip()


# ============================================================
# 模型
# ============================================================

def load_tokenizer(model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 左填充：最后一个位置永远对应真实末尾，取 logits[:, -1] 就是"下一个 token"
    tokenizer.padding_side = 'left'
    tokenizer.clean_up_tokenization_spaces = False
    return tokenizer


def load_model(model_path, adapter=None, adapter_root=ADAPTER_ROOT,
               trainable=False, dtype=torch.bfloat16):
    """加载基座；adapter 不为空时叠加对应的 LoRA。

    trainable=True 会以可训练方式加载（梯度分析需要），否则是纯推理。
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map='auto',
    )
    model.config.use_cache = False

    if adapter:
        from peft import PeftModel

        path = os.path.join(adapter_root, adapter)
        if not os.path.exists(os.path.join(path, 'adapter_config.json')):
            raise SystemExit(f'不是有效的 adapter 目录：{path}')
        model = PeftModel.from_pretrained(model, path, is_trainable=trainable)

    model.eval()
    return model


def freeze_dropout(model):
    """把 dropout 概率设成 0，这样 train() 模式下前向也完全确定。

    梯度分析需要 train() 才能开梯度检查点省显存，但 dropout 会让梯度带噪声，
    所以这里两件事一起做。
    """
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def release(model):
    del model
    torch.cuda.empty_cache()


# ============================================================
# 分词 / batch
# ============================================================

def pad_batch(tokenizer, texts, labels=None, max_length=1024):
    """把一个 batch 的文本做成左填充的 input_ids/attention_mask/(labels)。

    labels 给的是每条样本自己的 tag 列表（和文本等长），padding 位置填 -100。
    """
    encoded = [
        tokenizer(text, add_special_tokens=False, truncation=True,
                  max_length=max_length)['input_ids']
        for text in texts
    ]
    width = max(len(ids) for ids in encoded)
    pad_id = tokenizer.pad_token_id

    input_ids, attention, padded_labels = [], [], []
    for i, ids in enumerate(encoded):
        pad = width - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention.append([0] * pad + [1] * len(ids))
        if labels is not None:
            # 文本可能被截断，tag 也要跟着切
            tag = list(labels[i])[:len(ids)]
            tag = tag + [-100] * (len(ids) - len(tag))
            padded_labels.append([-100] * pad + tag)

    batch = {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention, dtype=torch.long),
    }
    if labels is not None:
        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def prompt_lengths(tokenizer, texts, max_length=1024):
    """每条样本的提问段 token 数（截断后）。"""
    return [
        len(tokenizer(text, add_special_tokens=False, truncation=True,
                      max_length=max_length)['input_ids'])
        for text in texts
    ]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def run_over_batches(items, fn, batch_size=4, label=''):
    """按 batch 调用 fn(batch)，显存不够就折半重试。返回拼接后的结果列表。"""
    results = []
    index = 0
    current = max(1, batch_size)

    while index < len(items):
        batch = items[index:index + current]
        try:
            results.extend(fn(batch))
            index += len(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            print(f'    [{label}] 显存不够，batch 降到 {current} 重试')

    return results


# ============================================================
# 指标
# ============================================================

def cosine_sim(a, b, eps=1e-8):
    """两个一维向量的余弦相似度。

    用 float64 算：梯度向量有 1300 万个数，float32 求和的累积误差足以让
    自身余弦算成 1.003（实测），float64 就没有这个问题。
    """
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))


def paired_cosine(a, b, eps=1e-8):
    """逐行配对的余弦相似度；a/b 形状 [n, d]。"""
    a = a.float()
    b = b.float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=eps)


def linear_cka(x, y):
    """线性 CKA（不依赖样本配对，衡量两组表示的整体几何是否相似）。"""
    x = x.float()
    y = y.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = torch.norm(x.t() @ y) ** 2
    xtx = torch.norm(x.t() @ x)
    yty = torch.norm(y.t() @ y)
    return float(xty / (xtx * yty + 1e-12))


def symmetric_kl(p, q, eps=1e-12):
    """对称 KL = KL(p||q) + KL(q||p)；输入是行归一化的概率 [n, V]。"""
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return kl_pq + kl_qp


def js_divergence(p, q, eps=1e-12):
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.log() - b.log())).sum(dim=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy(p, eps=1e-12):
    p = p.float() + eps
    p = p / p.sum(dim=-1, keepdim=True)
    return -(p * p.log()).sum(dim=-1)


# ============================================================
# 落盘 / 出图
# ============================================================

def save_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'  写出 {path}')


def save_matrix_csv(path, matrix, row_labels, col_labels, fmt='{:.6f}'):
    """把矩阵存成 csv，表头和行名都带上，方便直接进 Excel。"""
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([''] + list(col_labels))
        for label, row in zip(row_labels, matrix):
            cells = ['' if v != v else fmt.format(v) for v in row]
            writer.writerow([label] + cells)
    print(f'  写出 {path}')


def save_heatmap(matrix, row_labels, col_labels, path, title,
                 vmin=None, vmax=None, cmap='viridis', fmt='{:.2f}'):
    """画矩阵热力图，格子里标数值。没装 matplotlib 就跳过。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  跳过画图：没装 matplotlib')
        return

    if matrix and not isinstance(matrix[0], (list, tuple)):
        matrix = [list(matrix)]      # 一维的当成只有一行
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    fig, ax = plt.subplots(
        figsize=(max(5.5, 1.1 * cols + 2.5), max(4.0, 0.55 * rows + 2.2))
    )
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax.set_xticks(range(cols), col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(rows), row_labels, fontsize=9)
    ax.set_title(title, fontsize=11)

    for i in range(rows):
        for j in range(cols):
            value = matrix[i][j]
            text = 'n/a' if value != value else fmt.format(value)
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  写出 {path}')


# ============================================================
# LoRA
# ============================================================

def lora_config(adapter_dir):
    with open(os.path.join(adapter_dir, 'adapter_config.json'), encoding='utf-8') as f:
        return json.load(f)


def lora_modules(adapter_dir):
    """读 LoRA 权重，返回 ({模块名: (A, B)}, scaling)。

    不直接还原 ΔW：一个 8B 模型的全部 ΔW 拼起来有 13 亿个数（5 GB+），
    逐模块算才不会把内存撑爆。ΔW = B @ A * scaling 在需要时再算。

    为什么要还原 ΔW 而不是直接比 A、B：A/B 有缩放不变性
    （A→sA、B→B/s 给出同一个 ΔW），直接对混在一起的 A、B 算余弦是错的。
    """
    from safetensors.torch import load_file

    config = lora_config(adapter_dir)
    scaling = config['lora_alpha'] / config['r']

    state = load_file(os.path.join(adapter_dir, 'adapter_model.safetensors'))
    modules = {}
    for key in state:
        if not key.endswith('lora_A.weight'):
            continue
        prefix = key[:-len('lora_A.weight')]
        # 去掉 peft 包在外面的前缀，留下 model.layers.3.self_attn.q_proj.
        name = prefix.replace('base_model.model.', '', 1)
        b_key = prefix + 'lora_B.weight'
        if b_key in state:
            modules[name] = (state[key].float(), state[b_key].float())
    return modules, scaling


def delta_w(factored, scaling):
    """由 (A, B) 还原该模块的等效权重增量 ΔW = B @ A * scaling。"""
    a, b = factored
    return (b @ a) * scaling


def module_inner(factored_a, factored_b):
    """两个模块 ΔW 的 Frobenius 内积 ⟨B_a A_a, B_b A_b⟩（不含 scaling）。

    直接构造 ΔW 太贵：q_proj/o_proj 每个模块是 4096×4096（16.7M 个数），
    32 层加起来一个阶段就有 13 亿个数（5 GB+）。用迹的恒等式绕开：

        ⟨B1 A1, B2 A2⟩ = tr(A1ᵀ B1ᵀ B2 A2) = Σ_ij (B1ᵀB2)_ij · (A2 A1ᵀ)_ji

    左右两个矩阵都只有 r×r（16×16），实测与"先构造 ΔW 再算内积"的结果一致
    （相对误差 ~1e-6，就是 float32 的舍入）。
    """
    a1, b1 = factored_a
    a2, b2 = factored_b
    left = b1.t().to(torch.float32) @ b2.to(torch.float32)
    right = a2.to(torch.float32) @ a1.t().to(torch.float32)
    return float(torch.sum(left * right.t(), dtype=torch.float64))


def inner_products(modules_a, scaling_a, modules_b, scaling_b, names=None):
    """逐模块做内积，float64 累加，返回 (dot, ||a||², ||b||²)。

    标量一律用 float64 累加 —— 13 亿项的 float32 求和会积出百分之几十的误差
    （实测能把余弦算成 1.387）。
    """
    dot = norm_a = norm_b = 0.0
    for name in (names if names is not None else modules_a.keys()):
        if name not in modules_b:
            continue
        dot += scaling_a * scaling_b * module_inner(modules_a[name], modules_b[name])
        norm_a += scaling_a ** 2 * module_inner(modules_a[name], modules_a[name])
        norm_b += scaling_b ** 2 * module_inner(modules_b[name], modules_b[name])
    return dot, norm_a, norm_b


def cosine_from_products(dot, norm_a, norm_b, eps=1e-24):
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5) + eps)


def names_of_layer(modules, layer):
    """取某一层的模块名列表（用于逐层比较）。"""
    return [name for name in modules if module_layer(name)[0] == layer]


def layers_of(modules):
    return sorted({module_layer(name)[0] for name in modules if module_layer(name)[0] is not None})


def module_layer(name):
    """从 'model.layers.12.self_attn.q_proj.' 里取出 (12, 'q_proj')。"""
    parts = name.split('.')
    layer, module = None, 'unknown'
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                layer = int(parts[i + 1])
            except ValueError:
                layer = None
    for candidate in ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                      'gate_proj', 'up_proj', 'down_proj'):
        if candidate in parts:
            module = candidate
    return layer, module
