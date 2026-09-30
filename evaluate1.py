import argparse
import csv
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
# 持续学习矩阵与热力图写到 <OUTPUT_ROOT>/matrix/<任务前缀>/
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


def effective_max_new_tokens(task_name, override=None):
    """生成长度：命令行 --max-new-tokens 优先，否则按任务查表。

    evaluate_task 和缓存校验都走这里，避免两处逻辑各写一遍后跑偏。
    """
    return override if override else MAX_NEW_TOKENS.get(task_name, MAX_NEW_TOKENS['default'])


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='评估 Llama-3.1-8B-Instruct（可叠加 LoRA adapter）在各任务 test 集上的表现',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            '示例:\n'
            '  python3 evaluate1.py --adapter SST2+YELP                # 加载指定 adapter\n'
            '  python3 evaluate1.py                                    # 默认顺序的持续学习矩阵\n'
            '  python3 evaluate1.py --tasks AGNEWS QQP SST2 TLDR       # 按这个训练顺序评测，出 4x4 热力图\n'
            '  python3 evaluate1.py --max-samples 50                   # 快速试跑\n'
            '  python3 evaluate1.py --include-base                     # 矩阵里多加一行基座模型'
        ),
    )
    parser.add_argument(
        '--tasks', nargs='*', default=None,
        help='持续学习的任务顺序（与 train1.py 的 --tasks 一致）。adapter 按任务前缀自动查找'
             '（如 AGNEWS、AGNEWS+QQP …），每个 adapter 都会在全部任务上评估，'
             '最后生成 n*n 矩阵与热力图；不填则用全部 10 个任务',
    )
    parser.add_argument(
        '--adapter', default=None,
        help='只评估单个 adapter（adapter 根目录下的名字，如 SST2+YELP）；'
             '这种模式下不生成矩阵和热力图',
    )
    parser.add_argument(
        '--all-adapters', action='store_true',
        help=f'忽略任务顺序，直接评估 {ADAPTER_ROOT} 下的所有 adapter（按目录名排序）',
    )
    parser.add_argument(
        '--adapter-root', default=None,
        help=f'adapter 所在的根目录；不填则用 {ADAPTER_ROOT}（与 train1.py 的输出目录保持一致）',
    )
    parser.add_argument(
        '--include-base', action='store_true',
        help='矩阵里额外加一行"不叠任何 adapter 的基座模型"（多花一轮全任务评估的时间）',
    )
    parser.add_argument(
        '--no-heatmap', action='store_true',
        help='只算分数并写 matrix.json / matrix.csv，不画热力图',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='忽略已有结果，全部重算。默认会复用 <结果目录>/<对象>/summary.json 里'
             '已经算过的任务（配置对不上、明细文件不在、或加了 --force 才重算）',
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
        help='自定义结果根目录；不填则写到 <OUTPUT_ROOT>/<adapter名 或 base>/。'
             '评估多个 adapter 时会在这个目录下按 adapter 名建子目录，'
             '矩阵和热力图写到 <结果根目录>/matrix/<任务前缀>/',
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


def load_model(adapter_name=None, adapter_root=ADAPTER_ROOT):
    """加载基座模型；adapter_name 不为空时再叠加对应的 LoRA adapter。"""

    adapter_path = None
    if adapter_name:
        adapter_path = os.path.join(adapter_root, adapter_name)
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

    max_new_tokens = effective_max_new_tokens(task_name, max_new_tokens)

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
        'max_new_tokens': max_new_tokens,
        'input_length_avg': avg_input_length,
        'input_length_max': max_input_length,
        'input_truncated_count': truncated_count,
    }
    result.update(extra)
    return result


# ============================================================
# 5. 结果复用：已经算过的任务不重算
# ============================================================

def data_fingerprint(task_name):
    """test.jsonl 的大小 + 修改时间；数据换过就不该复用旧结果。"""
    path = os.path.join(DATA_ROOT, task_name, 'test.jsonl')
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {'path': path, 'size': stat.st_size, 'mtime': int(stat.st_mtime)}


def load_cached_results(output_dir, adapter_name, tasks, args):
    """从 <output_dir>/summary.json 里挑出可以直接复用的任务结果。

    任何一条对不上就放弃复用（宁可贵一点重算，也不要拿错数）：
      - adapter 名字（None = 基座）和这次要评的对象一致
      - model_path / data_root / max_samples / --max-new-tokens / MAX_LENGTH 一致
      - 这个任务在旧 summary.json 里有结果，且 <任务名>.jsonl 明细还在
      - 旧记录里留了生成长度或数据指纹时，也要和现在一致

    老版本写的 summary.json 没有指纹字段，这种情况跳过对应的校验（保证老结果也能复用）。
    """
    path = os.path.join(output_dir, 'summary.json')
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding='utf-8') as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if old.get('adapter') != adapter_name:
        return {}

    config = old.get('config') or {}
    expected = {
        'model_path': MODEL_PATH,
        'data_root': DATA_ROOT,
        'max_samples': args.max_samples,
        'max_new_tokens': args.max_new_tokens,
        'max_input_length': MAX_LENGTH,
    }
    for key, value in expected.items():
        if key in config and config[key] != value:
            return {}

    fingerprints = old.get('data_fingerprint') or {}
    cached = {}

    for result in old.get('results') or []:
        task = result.get('task')
        if task not in tasks or task in cached:
            continue

        # 生成长度变了，结果就不可比
        if 'max_new_tokens' in result:
            if result['max_new_tokens'] != effective_max_new_tokens(task, args.max_new_tokens):
                continue

        # 明细文件被删了就没法复核，重算
        if not os.path.exists(os.path.join(output_dir, f'{task}.jsonl')):
            continue

        if task in fingerprints:
            if fingerprints[task] != data_fingerprint(task):
                continue

        result = dict(result)
        result['from_cache'] = True
        cached[task] = result

    return cached


# ============================================================
# 6. 跑一个评估对象（基座或某个 adapter）的全部任务
# ============================================================

def write_summary(output_dir, adapter_name, tasks, args, results_by_task, reused_tasks):
    """把当前进度写成 summary.json。

    每算完一个任务就写一次，所以中途断电/打断也能保留已完成的任务，
    下次直接从剩下的接着跑。
    """
    payload = {
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
        'reused_tasks': sorted(set(reused_tasks)),
        'data_fingerprint': {task: data_fingerprint(task) for task in tasks},
        'results': [results_by_task[t] for t in tasks if t in results_by_task],
    }

    path = os.path.join(output_dir, 'summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_one(adapter_name, tasks, args, output_dir, adapter_root=ADAPTER_ROOT):
    """评估一个对象（基座或某个 adapter）在全部任务上的表现。

    已算过的任务默认直接复用（见 load_cached_results），只补算缺的那些；
    全都命中缓存时连模型都不加载。

    返回 results 列表（每个任务一个 dict），供外层拼持续学习矩阵。
    """

    tag = adapter_name if adapter_name else 'base'

    print()
    print('#' * 70)
    print(f'# 评估对象: {tag}')
    print(f'# 输出目录: {output_dir}')
    print('#' * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------
    # 先挑出能直接复用的任务
    # ----------------------------------------

    cached = {} if args.force else load_cached_results(
        output_dir, adapter_name, tasks, args,
    )
    todo = [task for task in tasks if task not in cached]

    if args.force:
        print('  --force：忽略已有结果，全部重算')
    elif cached:
        print(f'  复用已有结果：{len(cached)}/{len(tasks)} 个任务'
              f' {[t for t in tasks if t in cached]}')
    if todo:
        print(f'  本次要评估  ：{todo}')
    else:
        print('  全部任务都能复用，跳过模型加载')

    # ----------------------------------------
    # 只算缺的那些任务
    # ----------------------------------------

    results_by_task = dict(cached)

    if todo:
        tokenizer = load_tokenizer()
        model = load_model(adapter_name, adapter_root=adapter_root)

        for task_name in todo:
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
                results_by_task[task_name] = result
            # 每算完一个任务就落盘：中途断了，这一行已算完的任务下次能接着复用
            write_summary(output_dir, adapter_name, tasks, args,
                          results_by_task, cached)

        # 释放显存，方便接着评估下一个 adapter
        del model
        torch.cuda.empty_cache()

    # 按请求顺序拼回来
    results = [results_by_task[task] for task in tasks if task in results_by_task]

    # ----------------------------------------
    # 汇总：先写文件再打印，打印出问题也不该丢汇总
    # ----------------------------------------

    summary_path = write_summary(output_dir, adapter_name, tasks, args,
                                 results_by_task, cached)

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

    return results


# ============================================================
# 7. 持续学习矩阵（adapter × 任务）与热力图
# ============================================================

# 热力图列标签上的指标缩写
METRIC_ABBR = {
    'accuracy': 'acc',
    'BLEU': 'BLEU',
    'ROUGE-L': 'R-L',
}


def prefix_name(tasks):
    """['AGNEWS', 'QQP'] -> 'AGNEWS+QQP'，与 train1.py 的 adapter 目录命名一致。"""
    return '+'.join(tasks)


def find_sequence_targets(tasks, adapter_root):
    """按 train1.py 的前缀命名规则，找出序列上每个任务对应的 adapter。

    跑过 [A, B, C] 会留下 A/、A+B/、A+B+C/ 三个 adapter，
    所以第 k 行就是"训练到第 k 个任务为止"的模型。

    返回 (targets, missing)：
      targets 每项是 (tag, adapter_name, row_label)，missing 是缺 adapter 的前缀名。
    """
    targets, missing = [], []
    for k, task in enumerate(tasks, start=1):
        prefix = prefix_name(tasks[:k])
        path = os.path.join(adapter_root, prefix)
        if os.path.exists(os.path.join(path, 'adapter_config.json')):
            targets.append((prefix, prefix, f'{k}.{task}'))
        else:
            missing.append(prefix)
    return targets, missing


def row_average(values):
    """一行的平均分；跳过没算出来的格子（nan），全空返回 nan。"""
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else float('nan')


def print_matrix(tasks, rows, scores):
    """把矩阵打成文本表格，方便直接看 log。"""
    averages = [row_average(line) for line in scores]
    print(' ' * 20 + ''.join(f'{task:>10s}' for task in tasks) + f'{"avg":>10s}')
    for i, row in enumerate(rows):
        label = row['label']
        if len(label) > 18:          # adapter 目录名可能很长，别把表格撑歪
            label = label[:15] + '...'
        cells = ''.join(
            '       n/a' if value != value else f'{value:10.4f}'
            for value in scores[i]
        )
        avg = averages[i]
        avg_text = '       n/a' if avg != avg else f'{avg:10.4f}'
        print(f'  {label:<18s}' + cells + avg_text)


def save_matrix(matrix_dir, tasks, rows, scores, metrics):
    """矩阵落盘：matrix.json（完整明细）+ matrix.csv（丢进 Excel 看）。"""
    os.makedirs(matrix_dir, exist_ok=True)

    def clean(value):
        # json 不认 nan，统一写成 null
        return None if value != value else value

    payload = {
        'task_sequence': tasks,
        'columns': tasks,
        'column_metrics': metrics,
        'rows': [
            {
                'row_label': row['label'],
                'tag': row['tag'],
                'adapter': row['adapter'],
                'output_dir': row['output_dir'],
                'scores': [clean(value) for value in scores[i]],
                'avg': clean(row_average(scores[i])),
                'results': row['results'],
            }
            for i, row in enumerate(rows)
        ],
    }

    json_path = os.path.join(matrix_dir, 'matrix.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(matrix_dir, 'matrix.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['row', 'adapter'] + tasks + ['avg'])
        for i, row in enumerate(rows):
            cells = ['' if value != value else f'{value:.6f}' for value in scores[i]]
            avg = row_average(scores[i])
            writer.writerow(
                [row['label'], row['adapter'] or 'base'] + cells
                + ['' if avg != avg else f'{avg:.6f}']
            )

    return json_path, csv_path


def plot_matrix(scores, row_labels, col_labels, out_path, title):
    """画 n×m 热力图（行=训练阶段，列=评测任务）。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  跳过热力图：没装 matplotlib（pip install matplotlib）')
        return False

    n_rows = len(scores)
    n_cols = len(scores[0]) if scores else 0

    fig, ax = plt.subplots(
        figsize=(max(7.0, 1.2 * n_cols + 2.5), max(4.5, 0.62 * n_rows + 2.2))
    )

    image = ax.imshow(scores, cmap='viridis', vmin=0.0, vmax=1.0, aspect='auto')

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label('main metric score')

    ax.set_xticks(range(n_cols), col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_rows), row_labels, fontsize=9)
    ax.set_xlabel('evaluated task')
    ax.set_ylabel('adapter (trained through)')
    ax.set_title(title, fontsize=11)

    # 格子之间画细白线，10x10 也能看清
    ax.set_xticks([i - 0.5 for i in range(1, n_cols)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(1, n_rows)], minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', length=0)

    for i in range(n_rows):
        for j in range(n_cols):
            value = scores[i][j]
            if value != value:
                ax.text(j, i, 'n/a', ha='center', va='center',
                        fontsize=8, color='white')
                continue
            ax.text(
                j, i, f'{value:.2f}',
                ha='center', va='center', fontsize=8,
                color='white' if value < 0.62 else 'black',
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return True


# ============================================================
# 8. 入口
# ============================================================

def main():

    args = parse_args()

    tasks = args.tasks if args.tasks else DEFAULT_TASKS
    unknown = [t for t in tasks if t not in DEFAULT_TASKS]
    if unknown:
        raise SystemExit(f'未知任务名: {unknown}\n可用任务: {" ".join(DEFAULT_TASKS)}')

    if args.adapter and args.all_adapters:
        raise SystemExit('--adapter 和 --all-adapters 只能选一个')

    adapter_root = args.adapter_root if args.adapter_root else ADAPTER_ROOT
    results_root = args.output_dir if args.output_dir else OUTPUT_ROOT

    # ----------------------------------------
    # 决定要评估哪些对象 —— 就是矩阵的每一行
    # ----------------------------------------

    matrix_mode = True

    if args.adapter:
        # 单 adapter：保持原来的行为，不做矩阵
        matrix_mode = False
        targets = [(args.adapter, args.adapter, args.adapter)]

    elif args.all_adapters:
        if not os.path.isdir(adapter_root):
            raise SystemExit(f'--all-adapters: 目录不存在 {adapter_root}')
        # 只认带 adapter_config.json 的目录；训练中途失败的残留目录会被忽略
        names, skipped = [], []
        for name in sorted(os.listdir(adapter_root)):
            path = os.path.join(adapter_root, name)
            if not os.path.isdir(path):
                continue
            if os.path.exists(os.path.join(path, 'adapter_config.json')):
                names.append(name)
            else:
                skipped.append(name)
        if skipped:
            print(f'跳过 {len(skipped)} 个没有 adapter_config.json 的目录: {skipped}')
        if not names:
            raise SystemExit(f'--all-adapters: {adapter_root} 下没有找到 adapter')
        targets = [(name, name, name) for name in names]

    else:
        # 默认：按任务顺序，把 train1.py 训出来的 adapter 全找出来
        found, missing = find_sequence_targets(tasks, adapter_root)
        if missing:
            print(f'注意：序列是 {len(tasks)} 个任务，只找到 {len(found)} 个 adapter；'
                  f'缺 {missing}')
            print('      缺的行不会出现在矩阵里，缺的列记成 n/a。')
        if not found:
            raise SystemExit(
                f'在 {adapter_root} 下没找到这条序列的任何 adapter'
                f'（形如 {prefix_name(tasks[:1])}/、{prefix_name(tasks[:2])}/ …）。\n'
                '请先用 train1.py 训练，或用 --adapter-root 指定 adapter 所在目录。'
            )
        targets = found

    if matrix_mode and args.include_base:
        # tag 同时也是结果子目录名，基座那行统一叫 base
        targets = [('base', None, 'base')] + targets

    # ----------------------------------------
    # 打印这次的配置
    # ----------------------------------------

    print()
    print(f'任务顺序    : {tasks}')
    print(f'adapter 目录: {adapter_root}')
    print(f'结果根目录  : {results_root}')
    print(f'评测对象    : {[label for _, _, label in targets]}')
    if matrix_mode:
        print('矩阵规模    : %d 行 x %d 列（行 = 训练阶段，列 = 评测任务）'
              % (len(targets), len(tasks)))
    if args.force:
        print('结果复用    : 关闭（--force：全部重算）')
    if args.max_samples:
        print(f'调试模式    : 每个任务只用前 {args.max_samples} 条')

    # ----------------------------------------
    # 逐个对象跑完所有任务
    # ----------------------------------------

    rows = []
    for tag, adapter_name, label in targets:

        if args.output_dir:
            # 指定了结果根目录：单个对象直接写进去；
            # 多个对象时按 adapter 名建子目录，避免互相覆盖
            output_dir = (args.output_dir if len(targets) == 1
                          else os.path.join(args.output_dir, tag))
        else:
            output_dir = os.path.join(OUTPUT_ROOT, tag)

        results = run_one(
            adapter_name, tasks, args, output_dir,
            adapter_root=adapter_root,
        )
        rows.append({
            'tag': tag,
            'adapter': adapter_name,
            'label': label,
            'output_dir': output_dir,
            'results': results,
        })

    if not matrix_mode:
        return

    # ----------------------------------------
    # 拼矩阵：行 = 训练阶段，列 = 任务，格子 = 该任务的主指标
    # ----------------------------------------

    scores, metrics = [], {}
    for row in rows:
        by_task = {result['task']: result for result in row['results']}
        line = []
        for task in tasks:
            result = by_task.get(task)
            if result is None:
                # 这个任务的 test 集没找到（evaluate_task 返回 None）
                line.append(float('nan'))
                continue
            metrics.setdefault(task, result['metric'])
            line.append(float('nan') if result['score'] is None
                        else float(result['score']))
        scores.append(line)

    print()
    print('=' * 70)
    print('持续学习矩阵（行 = 训练阶段，列 = 评测任务，格子 = 主指标）')
    print('=' * 70)
    print('  指标：' + '  '.join(f'{task}={metrics.get(task, "?")}' for task in tasks))
    print()
    print_matrix(tasks, rows, scores)

    # 哪几格是复用旧结果、哪几格是这次新算的
    print()
    print('复用情况（缓存命中 / 本次新算）')
    for row in rows:
        total = len(row['results'])
        n_cached = sum(1 for result in row['results'] if result.get('from_cache'))
        print(f'  {row["label"]:20s} {n_cached}/{total} 复用，'
              f'{total - n_cached} 个新算')

    # ----------------------------------------
    # 落盘 + 热力图
    # ----------------------------------------

    matrix_dir = os.path.join(results_root, 'matrix', prefix_name(tasks))
    json_path, csv_path = save_matrix(matrix_dir, tasks, rows, scores, metrics)

    print()
    print(f'矩阵数据: {json_path}')
    print(f'          {csv_path}')

    if args.no_heatmap:
        print('热力图  : 已跳过（--no-heatmap）')
        return

    col_labels = [
        '%s\n[%s]' % (task, METRIC_ABBR.get(metrics.get(task), metrics.get(task) or '-'))
        for task in tasks
    ]
    heatmap_path = os.path.join(matrix_dir, 'heatmap.png')
    if plot_matrix(
        scores,
        [row['label'] for row in rows],
        col_labels,
        heatmap_path,
        'Continual learning matrix (cell = main metric)',
    ):
        print(f'热力图  : {heatmap_path}')


if __name__ == '__main__':
    main()
