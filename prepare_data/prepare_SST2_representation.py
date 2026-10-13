import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Value
from sklearn.model_selection import train_test_split

# 用脚本自身位置定位路径，从仓库根目录或 prepare_data/ 里运行都一致
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    REPO_ROOT, 'models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, 
trust_remote_code=True)
# ================================================================
# prepare_SST2.py 的「标签偏移」变体（representation 版）。
# 数据来源、采样、均衡、分层切分逻辑与原脚本完全一致，
# 唯一区别是标签编号整体后移：0/1 -> 6/7，
# 使得 QQP(0,1) / AGNEWS(2~5) / SST2(6~7) 三个分类任务不再共享输出 token。
# 产物写到 datas_representation/，原 datas/ 不受影响。
# 想切回原始标签，把 LABEL_OFFSET 改成 0 再跑一次即可。
# 改这里的数字时，evaluate1.py 里的 VALID_LABELS['SST2'] 必须同步改，
# 否则答对的标签会被判成"超出范围"而静默计错。
# ================================================================
OUTDIR = os.path.join(REPO_ROOT, 'datas_representation')
LABEL_OFFSET = 6


def instruction_text():
    """提问指令；标签编号随 LABEL_OFFSET 一起移动。"""
    return (
        "Classify the sentiment of the following sentence.\n"
        f"Output {LABEL_OFFSET} for negative sentiment and "
        f"{LABEL_OFFSET + 1} for positive sentiment.\n\n"
    )


dataset = load_dataset('stanfordnlp/sst2', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                instruction_text()
                + f"SENTENCE: {example['sentence']}\n\n"
                + "Answer with a single digit only. No explanation.\n"
                + "Label:"
            )
        }
    ]
    activation_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return {
        'activation_text': activation_text
    }

def build_train_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                instruction_text()
                + f"SENTENCE: {example['sentence']}\n\n"
                + "Answer with a single digit only. No explanation.\n"
                + "Label:"
            )
        },
        {
            'role': 'assistant',
            'content': str(example['label'] + LABEL_OFFSET)
        }
    ]
    train_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {
        'train_text': train_text
    }

def calculate_token_length(example):
    tokens = tokenizer(
        example['train_text'],
        add_special_tokens = False,
        truncation = False
    )
    return {
        'token_length' : len(tokens['input_ids'])
    }

dataset = dataset.map(build_activation_text)
dataset = dataset.map(build_train_text)
dataset = dataset.map(calculate_token_length)
dataset = dataset.filter(
    lambda x:x['token_length'] <= 1020
)

dataset = dataset.remove_columns('token_length')

label_groups = {}
for label in [0,1]:
    label_groups[label] = dataset.filter( lambda x : x['label'] == label)

balanced_dataset_list = []
for label in [0, 1]:
    groups = label_groups[label].select(range(5000))
    balanced_dataset_list.append(groups)

balanced_dataset = concatenate_datasets(
    balanced_dataset_list
)
balanced_dataset = balanced_dataset.shuffle(seed=42)

labels = balanced_dataset['label']
indices = list(range(10000))

train_idx, temp1_idx = train_test_split(
    indices,
    test_size = 0.2,
    random_state = 42,
    stratify = [labels[i] for i in indices]
)

test_idx, temp2_idx = train_test_split(
    temp1_idx,
    test_size = 0.5,
    random_state = 42,
    stratify = [labels[i] for i in temp1_idx]
)

val1_idx, val2_idx = train_test_split(
    temp2_idx,
    test_size = 0.5,
    random_state = 42,
    stratify = [labels[i] for i in temp2_idx]
)

train_dataset = balanced_dataset.select(train_idx)
test_dataset = balanced_dataset.select(test_idx)
val1_dataset = balanced_dataset.select(val1_idx)
val2_dataset = balanced_dataset.select(val2_idx)

# 原始 label 列也一起偏移，保证文件里记的数字和答案里的数字一致
def shift_label(example):
    return {'label': example['label'] + LABEL_OFFSET}


# label 在 HF 里是 ClassLabel（预设 num_classes=2），偏移后会越界，
# 先转成普通整数再偏移；写出的 JSONL 仍然是 "label": <int>，格式不变
train_dataset = train_dataset.cast_column('label', Value('int64')).map(shift_label)
test_dataset = test_dataset.cast_column('label', Value('int64')).map(shift_label)
val1_dataset = val1_dataset.cast_column('label', Value('int64')).map(shift_label)
val2_dataset = val2_dataset.cast_column('label', Value('int64')).map(shift_label)

print("\n========== Split ==========")
print(f"标签映射：原文 0/1 -> 实际 {LABEL_OFFSET}/{LABEL_OFFSET + 1}")
print("train:", len(train_dataset))
print("val1 :", len(val1_dataset))
print("val2 :", len(val2_dataset))
print("test :", len(test_dataset))

os.makedirs(OUTDIR, exist_ok=True)
train_dataset.to_json(
f"{OUTDIR}/SST2/train.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val1_dataset.to_json(
f"{OUTDIR}/SST2/val1.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val2_dataset.to_json(
f"{OUTDIR}/SST2/val2.jsonl",
orient="records",
lines=True,
force_ascii=False
)
test_dataset.to_json(
f"{OUTDIR}/SST2/test.jsonl",
orient="records",
lines=True,
force_ascii=False
)
