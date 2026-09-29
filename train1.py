import argparse
import json
import os
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ============================================================
# 1. 固定配置
#    需要按次调整的项都做成了命令行参数，见 parse_args()
# ============================================================

# 本地模型目录是 models（复数），不是 model
MODEL_PATH = "./models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master"

# 清洗后的数据（去掉了 system 段里的日期噪声），由 prepare_data/data_clear.py 生成
# 若要用未清洗的数据，把 "./datas_clean" 改成 "./datas"
DATA_ROOT = "./datas_clean"

# adapter 输出根目录；evaluate1.py 默认也从这里找 adapter
OUTPUT_ROOT = "./train1_continual_lora"

# 默认的训练顺序（持续学习：按列表顺序逐个任务往下训练）
DEFAULT_TASKS = [
    'AGNEWS',
    'AMAZON',
    'DBPEDIA',
    'IMDB',
    'MEDDIA',
    'QQP',
    'SST2',
    'TLDR',
    'WMT',
    'YELP',
]

LORA_KWARGS = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
    ],
    bias='none',
    task_type='CAUSAL_LM',
)


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='在多任务上按顺序持续训练 LoRA adapter（continual learning）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 train1.py                                          # 按默认顺序训练全部 10 个任务\n'
            '  python3 train1.py --tasks AGNEWS QQP SST2 TLDR             # 只训练这 4 个，按给定顺序\n'
            '  python3 train1.py --tasks SST2 --max-samples 64 --max-steps 4   # 冒烟测试\n'
            '  python3 train1.py --tasks SST2 YELP --output-dir ./cl_run1      # 换输出目录\n'
            '  python3 train1.py --tasks SST2 --epochs 1 --batch-size 1        # 省显存'
        ),
    )
    parser.add_argument(
        '--tasks', nargs='*', default=None,
        help='要训练的任务名，空格分隔，按给定顺序依次训练；不填则用全部 10 个',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help=f'adapter 输出根目录；不填则用 {OUTPUT_ROOT}。'
             '每个任务写到 <output-dir>/<任务前缀>/，如 <output-dir>/SST2+YELP/',
    )
    parser.add_argument(
        '--epochs', type=float, default=5,
        help='每个任务训练多少轮（默认 5）',
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='per_device_train_batch_size（默认 4；4090D 24G 跑 8B+LoRA，OOM 就降到 1）',
    )
    parser.add_argument(
        '--grad-accum', type=int, default=4,
        help='梯度累积步数（默认 4，等效 batch = batch-size * grad-accum）',
    )
    parser.add_argument(
        '--lr', type=float, default=2e-4,
        help='学习率（默认 2e-4）',
    )
    parser.add_argument(
        '--max-length', type=int, default=1024,
        help='分词时的最大长度（默认 1024）',
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='每个任务只用前 N 条训练（调试用）',
    )
    parser.add_argument(
        '--max-steps', type=int, default=-1,
        help='每个任务最多训练多少步（调试用，-1 表示不限）',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子（默认 42）',
    )
    parser.add_argument(
        '--no-resume', action='store_true',
        help='不复用已有 adapter，从头训练（默认会自动复用最长的已训练任务前缀）',
    )
    parser.add_argument(
        '--no-grad-checkpoint', dest='grad_checkpoint', action='store_false',
        help='关闭梯度检查点（默认开启；开着省显存但慢一些，长文本任务建议保持开启）',
    )
    parser.add_argument(
        '--no-mask-prompt', dest='mask_prompt', action='store_false',
        help='把提问段也算进 loss（默认不算：只对回答段计算 loss，这是 SFT 的标准做法）',
    )
    parser.add_argument(
        '--no-load-best', dest='load_best', action='store_false',
        help='训练结束时加载最好的 checkpoint（默认加载，指标是 val1 上的 eval_loss）',
    )
    return parser.parse_args()


# ============================================================
# 3. tokenizer 与模型
# ============================================================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    # Llama-3.1 没有 pad token，训练时必须补一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 训练用右填充，label 才能和 token 对齐（评测那边是左填充，各管各的）
    tokenizer.padding_side = 'right'

    return tokenizer


def load_model(resume_dir=None, grad_checkpoint=True):
    """加载模型。

    resume_dir 为空 -> 基座 + 新建 LoRA；
    不为空          -> 基座 + 从该目录加载已有 LoRA（is_trainable=True，可继续训练）。

    注意：PeftModel.from_pretrained 的第二个参数是目录，第一个参数必须是"裸的"
    transformers 模型。如果把 PeftModel 再套一层，参数会被冻住，训练时报
    "element 0 of tensors does not require grad"。
    """
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,     # transformers 5.x 用 dtype（torch_dtype 已废弃）
        device_map='auto',
    )
    base.config.use_cache = False

    if resume_dir:
        model = PeftModel.from_pretrained(base, resume_dir, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(**LORA_KWARGS))

    if grad_checkpoint:
        # 1024 长度的序列在 24G 卡上很容易 OOM，开梯度检查点能省不少显存
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model


# ============================================================
# 4. 断点复用：输出目录名 = 任务前缀
# ============================================================

def prefix_name(tasks):
    """['AGNEWS', 'SST2'] -> 'AGNEWS+SST2'。

    目录名本身就记录了"训练到哪几个任务为止"，因此可以直接拿来复用：
    跑 [A,B,C] 会产出 A/ 和 A+B/ 和 A+B+C/；
    下次跑 [A,B,D] 时，A+B/ 已经存在，就能跳过 A、B 直接训 D。
    """
    return '+'.join(tasks)


def manifest_path(save_dir):
    return os.path.join(save_dir, 'run.json')


def write_manifest(save_dir, tasks, args):
    """把这次训练用的任务列表与超参记下来，方便以后核对。"""
    with open(manifest_path(save_dir), 'w', encoding='utf-8') as f:
        json.dump(
            {
                'tasks': tasks,
                'data_root': DATA_ROOT,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'grad_accum': args.grad_accum,
                'lr': args.lr,
                'max_length': args.max_length,
                'seed': args.seed,
                'mask_prompt': args.mask_prompt,
                'load_best': args.load_best,
                'max_samples': args.max_samples,
                'created_at': datetime.now().isoformat(timespec='seconds'),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def find_resume_point(tasks, output_root):
    """找出最长的"已训练完的任务前缀"。

    返回 (已完成任务数, 对应目录)；找不到就返回 (0, None)。
    """
    for k in range(len(tasks), 0, -1):
        save_dir = os.path.join(output_root, prefix_name(tasks[:k]))
        if os.path.exists(os.path.join(save_dir, 'adapter_config.json')):
            return k, save_dir
    return 0, None


def check_manifest(save_dir, args):
    """复用前对比关键超参，不一致就提醒（只是提醒，不阻止）。"""
    path = manifest_path(save_dir)
    if not os.path.exists(path):
        return

    with open(path, encoding='utf-8') as f:
        old = json.load(f)

    now = {
        'data_root': DATA_ROOT,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grad_accum': args.grad_accum,
        'lr': args.lr,
        'max_length': args.max_length,
        'seed': args.seed,
        'mask_prompt': args.mask_prompt,
        'load_best': args.load_best,
    }

    diff = [f'{k}: 已有={old.get(k)}，本次={v}' for k, v in now.items() if old.get(k) != v]
    if diff:
        print('  警告：要复用的 adapter 是用不同超参训练的：')
        for d in diff:
            print(f'    - {d}')
        print('  如果是有意的，忽略即可；否则加 --no-resume 从头训练。')


# ============================================================
# 5. 数据分词
# ============================================================

def build_tokenized_dataset(path, tokenizer, max_length, max_samples=None,
                            mask_prompt=True):
    """读 jsonl -> 分词 -> 生成 labels。

    数据的设计是 train_text = activation_text + 回答 + <|eot_id|>，
    并且已验证 activation_text 的 token 正好是 train_text 的前缀，
    所以可以直接用"提问的 token 数"精确切出回答段的起点。

    mask_prompt=True 时，labels 里提问段全部设成 -100，只有回答段参与 loss。
    """
    dataset = load_dataset('json', data_files=path, split='train')
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize_function(examples):
        full = tokenizer(
            examples['train_text'],
            # train_text 已经自带 <|begin_of_text|>，
            # 不关掉自动补 BOS 会变成两个 BOS（训练与评测必须一致）
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        labels = [list(ids) for ids in full['input_ids']]

        if mask_prompt:
            prompt = tokenizer(
                examples['activation_text'],
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            for row, prompt_ids in zip(labels, prompt['input_ids']):
                for i in range(min(len(prompt_ids), len(row))):
                    row[i] = -100

        full['labels'] = labels
        return full

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )


class CompletionOnlyCollator:
    """把 labels 补齐到与 input_ids 同宽，padding 位置设成 -100。

    提问段是否参与 loss 由 dataset 里 labels 的 -100 决定。
    DataCollatorForLanguageModeling 会用整段 input_ids 覆盖掉 labels，
    所以这里不能复用它。

    要求 tokenizer.padding_side == 'right'，label 才和 token 对齐。
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f['labels'] for f in features]
        inputs = [{k: v for k, v in f.items() if k != 'labels'} for f in features]

        batch = self.tokenizer.pad(inputs, padding=True, return_tensors='pt')

        width = batch['input_ids'].shape[1]
        padded = torch.full((len(labels), width), -100, dtype=torch.long)
        for row, label in enumerate(labels):
            padded[row, :len(label)] = torch.tensor(label, dtype=torch.long)

        batch['labels'] = padded
        return batch


# ============================================================
# 6. 单个任务训练
# ============================================================

def train_one(task_name, done_tasks, total, tokenizer, model, args, output_root):
    """在单个任务上训练，adapter 存到 <output_root>/<已完成任务拼接>/。"""

    save_dir = os.path.join(output_root, prefix_name(done_tasks))

    print()
    print("=" * 70)
    print(f"Task {len(done_tasks)}/{total}: {task_name}")
    print(f"保存到: {save_dir}")
    print("=" * 70)

    data_dir = os.path.join(DATA_ROOT, task_name)
    train_path = os.path.join(data_dir, 'train.jsonl')
    val_path = os.path.join(data_dir, 'val1.jsonl')

    if not os.path.exists(train_path):
        raise SystemExit(
            f'找不到训练数据：{train_path}\n'
            f'持续学习序列不能跳过任务——否则后一个任务的 adapter 目录名\n'
            f'（按任务前缀命名）会把没训过的任务也算进去，而且会被后续复用继承。\n'
            f'请先生成该任务的数据，或者把它从 --tasks 里去掉。'
        )

    tokenized_dataset = build_tokenized_dataset(
        train_path, tokenizer, args.max_length, args.max_samples,
        mask_prompt=args.mask_prompt,
    )

    # 用 val1 做验证；文件不存在就不启用评估
    eval_dataset = None
    if os.path.exists(val_path):
        eval_dataset = build_tokenized_dataset(
            val_path, tokenizer, args.max_length, args.max_samples,
            mask_prompt=args.mask_prompt,
        )

    data_collator = CompletionOnlyCollator(tokenizer)

    has_eval = eval_dataset is not None

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        eval_strategy='epoch' if has_eval else 'no',
        save_strategy='epoch',
        save_total_limit=3,
        report_to='none',
        metric_for_best_model='eval_loss' if has_eval else None,
        greater_is_better=False,
        # 训练结束后加载 val1 上 eval_loss 最好的 checkpoint。
        # 注意：必须要有验证集，且 save_strategy 必须与 eval_strategy 一致。
        load_best_model_at_end=has_eval and args.load_best,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # PeftModel.save_pretrained 保存的是 LoRA adapter，
    # 内容是"累积训练到本任务为止"的，不是本任务独立的 adapter
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    write_manifest(save_dir, done_tasks, args)


# ============================================================
# 7. 入口
# ============================================================

def main():

    args = parse_args()

    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    unknown = [t for t in tasks if t not in DEFAULT_TASKS]
    if unknown:
        raise SystemExit(f'未知任务名: {unknown}\n可用任务: {" ".join(DEFAULT_TASKS)}')

    output_root = args.output_dir if args.output_dir else OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)

    print(f'训练顺序  : {tasks}')
    print(f'输出根目录: {output_root}')
    print(f'超参      : epochs={args.epochs} batch_size={args.batch_size} '
          f'grad_accum={args.grad_accum} 等效batch={args.batch_size * args.grad_accum} lr={args.lr}')
    print(f'loss 范围 : ' + ('只对回答段（提问段设成 -100）' if args.mask_prompt
                             else '整条序列（提问段也算 loss）'))
    if args.max_steps > 0:
        print(f'调试模式  : 每个任务最多 {args.max_steps} 步')
    if args.max_samples:
        print(f'调试模式  : 每个任务只用前 {args.max_samples} 条')

    # ----------------------------------------
    # 找可复用的"已训练任务前缀"
    # ----------------------------------------

    done, resume_dir = (0, None)
    if not args.no_resume:
        done, resume_dir = find_resume_point(tasks, output_root)

    if done >= len(tasks):
        print()
        print(f'全部 {len(tasks)} 个任务的 adapter 都已存在：{prefix_name(tasks)}')
        print(f'位置：{resume_dir}')
        print('如需重训，加 --no-resume')
        return

    if done:
        print()
        print(f'复用已有 adapter：{prefix_name(tasks[:done])}')
        print(f'  跳过已训练完的 {done} 个任务：{tasks[:done]}')
        print(f'  接着训练：{tasks[done:]}')
    elif not args.no_resume:
        print()
        print('没有找到可复用的 adapter，从头训练')

    # ----------------------------------------
    # 加载模型（有可复用 adapter 就叠上去）
    # ----------------------------------------

    if resume_dir:
        check_manifest(resume_dir, args)
        print(f'从已有 adapter 继续：{resume_dir}')

    # ----------------------------------------
    # 开训前先确认数据齐全，避免白等一次 8B 模型加载
    # ----------------------------------------

    missing = [
        name for name in tasks[done:]
        if not os.path.exists(os.path.join(DATA_ROOT, name, 'train.jsonl'))
    ]
    if missing:
        raise SystemExit(
            f'以下任务找不到训练数据：{missing}\n'
            f'路径形如 {DATA_ROOT}/<任务名>/train.jsonl\n'
            f'持续学习序列不能跳过任务——否则 adapter 目录名（按任务前缀命名）\n'
            f'会把没训过的任务也算进去，而且会被后续复用继承。\n'
            f'请先生成数据，或者把这些任务从 --tasks 里去掉。'
        )

    tokenizer = load_tokenizer()
    model = load_model(resume_dir=resume_dir, grad_checkpoint=args.grad_checkpoint)

    # ----------------------------------------
    # 逐任务训练
    # ----------------------------------------

    for step, task_name in enumerate(tasks, start=1):
        if step <= done:
            continue
        train_one(task_name, tasks[:step], len(tasks), tokenizer, model, args, output_root)


if __name__ == '__main__':
    main()
