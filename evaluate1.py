import argparse
import json
import os
import re
from collections import Counter

import torch

# 必须在 import datasets / transformers 之前设置才生效
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

# ============================================================
# 1. 固定配置
#    需要按次调整的项都做成了命令行参数，见 parse_args()
# ============================================================

MODEL_PATH = "./models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master"

# 与 train1.py / prepare_activation.py 保持一致，读清洗后的数据
DATA_ROOT = "./datas_clean"

# train1.py 保存 LoRA adapter 的目录
ADAPTER_ROOT = "./train1_continual_lora"

# 结果的根目录；实际会写到 <OUTPUT_ROOT>/<adapter名 或 base>/
OUTPUT_ROOT = "./evaluate1_results"

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

MAX_LENGTH = 1024

# 每条样本最多生成多少新 token。
# 注意：模型不会只吐一个裸标签，而是 "……as 1 (positive sentiment)." 这种完整句子，
# 数字通常出现在第 10~15 个 token。分类任务也要留够长度，否则会被截断成 0 分。
MAX_NEW_TOKENS = {
    'default': 32,
    'QQP':64,
    'MEDDIA': 160,
    'TLDR': 80,
    'WMT': 80,
}

# 这些任务的答案是一个标签，比对时只取第一个数字
LABEL_TASKS = {
    'AGNEWS', 'AMAZON', 'DBPEDIA', 'IMDB', 'QQP', 'SST2', 'YELP',
}

# 生成类任务（主指标不是 accuracy）
#   WMT          -> corpus BLEU
#   TLDR / MEDDIA -> ROUGE-1/2/L
GENERATIVE_TASKS = {'TLDR', 'MEDDIA'}

EOT = '<|eot_id|>'


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='评估 Llama-3.1-8B-Instruct（可叠加 LoRA adapter）在各任务 test 集上的表现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 evaluate1.py                                    # 基座模型，全部任务\n'
            '  python3 evaluate1.py --adapter SST2+YELP                # 加载指定 adapter\n'
            '  python3 evaluate1.py --tasks SST2 YELP --max-samples 50 # 快速试跑\n'
            '  python3 evaluate1.py --all-adapters                     # 依次评估所有 adapter'
        ),
    )
    parser.add_argument(
        '--adapter', default=None,
        help='train1_continual_lora 下的 adapter 目录名，如 SST2+YELP'
             '（train1.py 按已完成任务的前缀命名）；不填则只评估基座模型',
    )
    parser.add_argument(
        '--all-adapters', action='store_true',
        help=f'依次评估 {ADAPTER_ROOT} 下的所有 adapter（持续学习矩阵）',
    )
    parser.add_argument(
        '--tasks', nargs='*', default=None,
        help='要评估的任务名，空格分隔；不填则评估全部任务',
    )
    parser.add_argument(
        '--max-samples', type=int, default=None,
        help='每个任务只评估前 N 条（调试用）',
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='生成时的批大小（默认 8）',
    )
    parser.add_argument(
        '--max-new-tokens', type=int, default=None,
        help='覆盖默认生成长度，所有任务统一用这个值',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='自定义输出目录；不填则写到 <OUTPUT_ROOT>/<adapter名 或 base>/。'
             '批量评估多个 adapter 时会在这个目录下按 adapter 名建子目录',
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

    # Llama-3.1 没有 pad token，批量生成时需要
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 批量生成必须左填充，否则生成的内容会接在 padding 后面
    tokenizer.padding_side = 'left'

    # BPE 分词器不需要这个后处理，开着会删掉标点前的空格
    tokenizer.clean_up_tokenization_spaces = False

    return tokenizer


def load_model(adapter_name=None):
    """加载基座模型；adapter_name 不为空时再叠加对应的 LoRA adapter。"""

    adapter_path = None
    if adapter_name:
        adapter_path = os.path.join(ADAPTER_ROOT, adapter_name)
        # 先检查再加载，避免白等一次 8B 模型的加载
        if not os.path.exists(os.path.join(adapter_path, 'adapter_config.json')):
            raise SystemExit(
                f'不是有效的 adapter 目录：{adapter_path}\n'
                f'（缺少 adapter_config.json —— 训练中途失败会留下这种空目录）'
            )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,     # transformers 5.x 用 dtype（torch_dtype 已废弃）
        device_map='auto',
    )

    if adapter_name:
        print(f'加载 LoRA adapter: {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print('使用基座模型评估')

    model.eval()
    return model


# ============================================================
# 3. 工具函数
# ============================================================

def gold_answer(example):
    """标准答案 = train_text 里 activation_text 之后、<|eot_id|> 之前的部分。"""
    tail = example['train_text'][len(example['activation_text']):]
    return tail.replace(EOT, '').strip()


# 每个分类任务的合法标签。用来过滤"抽到的数字"——
# 否则模型回答里的年份、产品型号（2016、T120）会被当成标签。
VALID_LABELS = {
    'AGNEWS':  {'0', '1', '2', '3'},
    'AMAZON':  {'1', '2', '3', '4', '5'},
    'DBPEDIA': {str(i) for i in range(14)},
    'IMDB':    {'0', '1'},
    'QQP':     {'0', '1'},
    'SST2':    {'0', '1'},
    'YELP':    {'1', '2', '3', '4', '5'},
}


# 有些任务模型不会直接吐标签数字，而是用自然语言回答。
# 例如 QQP 的 prompt 里带着 "QUESTION 1 / QUESTION 2"，直接取第一个数字
# 会抓到 "1"，把大量样本判错。这里按任务给出语义关键词。
BINARY_SEMANTIC = {
    'QQP': (
        # 命中任意一个 -> 0（不等价）
        [
            'not equivalent', 'not the same', 'different meaning',
            'do not have the same', "don't have the same",
            'not asking the same', 'different question',
            'not identical', 'not similar',
        ],
        # 命中任意一个 -> 1（等价）
        [
            'equivalent', 'same meaning', 'the same meaning',
            'asking the same', 'same question', 'identical',
        ],
    ),
}


def extract_label(text, task_name=None):
    """从生成结果里抽出标签。

    优先级：
      1. 开头就是数字      —— 模型直接吐了裸标签，最可靠
      2. "label/answer/output N" 句式 —— 模型"先解释、后下结论"时用这个
      3. 任务特定的语义判定（现在是 QQP 的等价 / 不等价）
      4. 兜底：第一个落在合法取值范围内的数字

    第 1、2、4 步都会用 VALID_LABELS 校验，避免抓到年份、型号等噪声。
    """
    text = text.strip()
    valid = VALID_LABELS.get(task_name)

    def acceptable(value):
        return valid is None or value in valid

    match = re.match(r'^(-?\d+)\b', text)
    if match and acceptable(match.group(0)):
        return match.group(0)

    match = re.search(
        r'(?:label|answer|output|classification|category)\s*(?:is|:|=)?\s*(-?\d+)',
        text,
        re.I,
    )
    if match and acceptable(match.group(1)):
        return match.group(1)

    if task_name in BINARY_SEMANTIC:
        negative, positive = BINARY_SEMANTIC[task_name]
        low = text.lower()
        # 先判否定，因为 "not equivalent" 里也含 "equivalent"
        if any(word in low for word in negative):
            return '0'
        if any(word in low for word in positive):
            return '1'

    # 兜底：扫一遍所有数字，取第一个合法的
    for match in re.finditer(r'-?\d+', text):
        if acceptable(match.group(0)):
            return match.group(0)

    return text


def token_f1(pred, gold):
    """轻量 token 级 F1，仅作为辅助指标，不作为正式主指标。"""
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_scores(preds, golds):
    """计算生成任务的 ROUGE-1/2/L F1。"""
    if rouge_scorer is None:
        raise RuntimeError(
            "缺少 rouge-score，请先安装：pip install rouge-score"
        )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, gold in zip(preds, golds):
        score = scorer.score(gold.strip(), pred.strip())
        for key in totals:
            totals[key] += score[key].fmeasure

    n = max(len(preds), 1)
    return {key: value / n for key, value in totals.items()}


def bleu_score(preds, golds):
    """使用 sacreBLEU 计算 corpus BLEU。

    返回 (分数, 实际参与计算的样本数)。
    sacreBLEU 遇到空 hypothesis / 空 reference 会报错，所以先过滤掉。
    """
    if sacrebleu is None:
        raise RuntimeError(
            "缺少 sacrebleu，请先安装：pip install sacrebleu"
        )

    pairs = [
        (p.strip(), g.strip())
        for p, g in zip(preds, golds)
        if p.strip() and g.strip()
    ]
    if not pairs:
        return 0.0, 0

    hypotheses, references = zip(*pairs)
    result = sacrebleu.corpus_bleu(list(hypotheses), [list(references)])
    return result.score / 100.0, len(pairs)


def print_samples(golds, preds, n=3):
    """生成类任务抽几条出来人工看看。"""
    print('  抽样（gold -> pred）:')
    for gold, pred in list(zip(golds, preds))[:n]:
        print(f'    {gold[:100]!r}')
        print(f'      -> {pred.strip()[:100]!r}')


# ============================================================
# 4. 单个任务评估
# ============================================================

def evaluate_task(task_name, tokenizer, model, output_dir,
                  batch_size=8, max_samples=None, max_new_tokens=None):
    """在某个任务的 test 集上评估，返回汇总用的 dict。"""

    path = os.path.join(DATA_ROOT, task_name, 'test.jsonl')
    if not os.path.exists(path):
        print(f'[{task_name}] 跳过：找不到 {path}')
        return None

    dataset = load_dataset('json', data_files=path, split='train')
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = dataset['activation_text']
    golds = [gold_answer(example) for example in dataset]
    is_label_task = task_name in LABEL_TASKS

    # 检查输入是否超过 MAX_LENGTH，避免正式实验时不知道有多少样本被截断。
    # 批量分词一次算完，不要逐条循环调用 tokenizer
    # 注意：dataset['col'] 是 datasets 的惰性 Column，不是 list，
    # 直接传给 tokenizer 会报 ValueError，要先 list() 转一下。
    input_lengths = [
        len(ids)
        for ids in tokenizer(
            list(prompts),
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    ]

    truncated_count = sum(length > MAX_LENGTH for length in input_lengths)
    max_input_length = max(input_lengths) if input_lengths else 0
    avg_input_length = (
        sum(input_lengths) / len(input_lengths)
        if input_lengths else 0.0
    )

    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS.get(task_name, MAX_NEW_TOKENS['default'])

    print(f'[{task_name}] {len(prompts)} 条，max_new_tokens={max_new_tokens}')
    print(
        f'  输入长度：avg={avg_input_length:.1f}, '
        f'max={max_input_length}, '
        f'> {MAX_LENGTH} tokens={truncated_count}'
    )
    if truncated_count > 0:
        print(
            f'  WARNING: 有 {truncated_count} 条样本超过 MAX_LENGTH={MAX_LENGTH}，'
            '评估时会被截断。'
        )

    preds = []
    for start in range(0, len(prompts), batch_size):

        batch = prompts[start:start + batch_size]
        # activation_text 已经自带 <|begin_of_text|>，
        # 不关掉自动补 BOS 会变成两个 BOS（必须与 train1.py 保持一致）
        inputs = tokenizer(
            batch,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                       # 贪心解码，保证可复现
                pad_token_id=tokenizer.pad_token_id,
            )

        # generate 返回的是"输入 + 新生成"，只取新生成的部分
        new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
        preds.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

        done = min(start + batch_size, len(prompts))
        print(f'\r  {done}/{len(prompts)}', end='')
    print()

    # ----------------------------------------
    # 先落盘：生成结果很贵，指标算失败也不能丢
    # ----------------------------------------

    out_path = os.path.join(output_dir, f'{task_name}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as fout:
        for gold, pred, example in zip(golds, preds, dataset):
            fout.write(json.dumps({
                'gold': gold,
                'pred': pred.strip(),
                'activation_text': example['activation_text'],
            }, ensure_ascii=False) + '\n')

    # ----------------------------------------
    # 打分
    # ----------------------------------------

    extra = {}
    score = None

    if is_label_task:
        # gold 是干净的单标签（如 "1"），不需要抽取；只对预测结果做抽取
        hits = sum(
            extract_label(p, task_name) == g.strip()
            for p, g in zip(preds, golds)
        )
        metric, score = 'accuracy', hits / len(preds)

    elif task_name == 'WMT':
        # 翻译任务：主指标用 corpus BLEU
        metric = 'BLEU'
        try:
            score, used = bleu_score(preds, golds)
            if used < len(preds):
                print(f'  注意：{len(preds) - used} 条因预测或参考答案为空被排除')
        except Exception as exc:
            print(f'  跳过 BLEU：{exc}')

        extra['exact_match'] = sum(
            p.strip() == g.strip()
            for p, g in zip(preds, golds)
        ) / max(len(preds), 1)
        extra['token_f1'] = (
            sum(token_f1(p, g) for p, g in zip(preds, golds))
            / max(len(preds), 1)
        )

        if score is not None:
            print(f'  BLEU = {score:.4f}')
        print(f'  exact_match = {extra["exact_match"]:.4f}')
        print(f'  token-F1 = {extra["token_f1"]:.4f}')
        print_samples(golds, preds)

    elif task_name in GENERATIVE_TASKS:
        # 摘要 / 医疗对话：主指标用 ROUGE-1/2/L
        metric = 'ROUGE-L'
        try:
            rouge = rouge_scores(preds, golds)
            extra.update(rouge)
            score = rouge['rougeL']
            print(f'  ROUGE-1 = {rouge["rouge1"]:.4f}')
            print(f'  ROUGE-2 = {rouge["rouge2"]:.4f}')
            print(f'  ROUGE-L = {rouge["rougeL"]:.4f}')
        except Exception as exc:
            print(f'  跳过 ROUGE：{exc}')

        # token-F1 仅作辅助指标，方便排查结果
        extra['token_f1'] = (
            sum(token_f1(p, g) for p, g in zip(preds, golds))
            / max(len(preds), 1)
        )
        print(f'  token-F1 = {extra["token_f1"]:.4f}')
        print_samples(golds, preds)

    else:
        raise AssertionError(
            f'{task_name} 未归类：既不在 LABEL_TASKS，也不在 GENERATIVE_TASKS'
        )

    if score is None:
        print(f'  {metric} = 跳过（指标库缺失或计算失败）')
    else:
        print(f'  {metric} = {score:.4f}')

    # ----------------------------------------
    # 汇总
    # ----------------------------------------

    result = {
        'task': task_name,
        'n': len(preds),
        'metric': metric,
        'score': score,
        'input_length_avg': avg_input_length,
        'input_length_max': max_input_length,
        'input_truncated_count': truncated_count,
    }
    result.update(extra)
    return result


# ============================================================
# 5. 跑一个评估对象（基座或某个 adapter）的全部任务
# ============================================================

def run_one(adapter_name, tasks, args, output_dir):

    tag = adapter_name if adapter_name else 'base'

    print()
    print('#' * 70)
    print(f'# 评估对象: {tag}')
    print(f'# 输出目录: {output_dir}')
    print('#' * 70)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_model(adapter_name)

    results = []
    for task_name in tasks:
        print()
        print('=' * 70)
        result = evaluate_task(
            task_name,
            tokenizer=tokenizer,
            model=model,
            output_dir=output_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
        if result is not None:
            results.append(result)

    # ----------------------------------------
    # 汇总：先写文件再打印，打印出问题也不该丢汇总
    # ----------------------------------------

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'adapter': adapter_name,
                'config': {
                    'model_path': MODEL_PATH,
                    'data_root': DATA_ROOT,
                    'tasks': tasks,
                    'max_samples': args.max_samples,
                    'batch_size': args.batch_size,
                    'max_new_tokens': args.max_new_tokens,
                    'max_input_length': MAX_LENGTH,
                },
                'results': results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print('=' * 70)
    print(f'汇总（{tag}）')
    print('=' * 70)

    for result in results:
        # score 可能是 None（指标库缺失或计算失败）
        score_text = '跳过' if result['score'] is None else f'{result["score"]:.4f}'
        line = (f'  {result["task"]:8s} n={result["n"]:5d}  '
                f'{result["metric"]:12s} {score_text}')
        if 'token_f1' in result:
            line += f'   token-F1={result["token_f1"]:.4f}'
        print(line)

    print()
    print(f'预测明细: {output_dir}/<任务名>.jsonl')
    print(f'汇总结果: {summary_path}')

    # 释放显存，方便接着评估下一个 adapter
    del model
    torch.cuda.empty_cache()


# ============================================================
# 6. 入口
# ============================================================

def main():

    args = parse_args()

    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    unknown = [t for t in tasks if t not in DEFAULT_TASKS]
    if unknown:
        raise SystemExit(f'未知任务名: {unknown}\n可用任务: {" ".join(DEFAULT_TASKS)}')

    # 决定要评估哪些对象
    if args.all_adapters:
        if not os.path.isdir(ADAPTER_ROOT):
            raise SystemExit(f'--all-adapters: 目录不存在 {ADAPTER_ROOT}')
        # 只认带 adapter_config.json 的目录；训练中途失败的残留目录会被忽略
        adapters, skipped = [], []
        for name in sorted(os.listdir(ADAPTER_ROOT)):
            path = os.path.join(ADAPTER_ROOT, name)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, 'adapter_config.json')):
                adapters.append(name)
            else:
                skipped.append(name)
        if skipped:
            print(f'跳过 {len(skipped)} 个没有 adapter_config.json 的目录: {skipped}')
        if not adapters:
            raise SystemExit(f'--all-adapters: {ADAPTER_ROOT} 下没有找到 adapter')
        print(f'找到 {len(adapters)} 个 adapter: {adapters}')
    else:
        adapters = [args.adapter]

    for adapter_name in adapters:

        tag = adapter_name if adapter_name else 'base'

        if args.output_dir:
            # 指定了输出目录：单个对象直接写进去；
            # 批量评估时在下面按 adapter 名建子目录，避免互相覆盖
            if len(adapters) == 1:
                output_dir = args.output_dir
            else:
                output_dir = os.path.join(args.output_dir, tag)
        else:
            output_dir = os.path.join(OUTPUT_ROOT, tag)

        run_one(adapter_name, tasks, args, output_dir)


if __name__ == '__main__':
    main()
