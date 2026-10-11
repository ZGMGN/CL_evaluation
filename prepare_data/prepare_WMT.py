import os
from transformers import AutoTokenizer
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# 用脚本自身位置定位路径，从仓库根目录或 prepare_data/ 里运行都一致
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    REPO_ROOT, 'models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, 
trust_remote_code=True)
OUTDIR = os.path.join(REPO_ROOT, 'datas')
# wmt19 的 zh-en train 有 13 个分片、合计约 3.6 GB，而本脚本只用到前 100000 条。
# 分片是按文件名顺序拼接的，第 0 个分片本身就有 1,998,814 行，
# 所以前 10 万条全部落在第 0 分片里 —— 只下这一个分片，
# 产物与"下载全部 13 个分片再 select(range(100000))"完全一致，省下约 3.4 GB。
# 需要超过 100000 条样本时，把下面两行换回：
#   dataset = load_dataset("wmt/wmt19", 'zh-en', split="train")
from huggingface_hub import hf_hub_download

_WMT_SHARD = hf_hub_download(
    'wmt/wmt19', 'zh-en/train-00000-of-00013.parquet', repo_type='dataset')
dataset = load_dataset('parquet', data_files=_WMT_SHARD, split="train")
#{ "en": "1929 or 1989?", "zh": "1929年还是1989年?" }
dataset = dataset.select(range(100000))
def build_activation_text(example):
    messages = [
        {
            'role':'user',
            'content':(
                'Translate the following Chinese text into English.\n\n'
                f'TEXT: {example['translation']['zh']}'
                )
        }
    ]
    activation_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    return {
        'activation_text' : activation_text
    }

def build_train_text(example):
    messages = [
        {
            'role':'user',
            'content':(
                'Translate the following Chinese text into English.\n\n'
                f'TEXT: {example['translation']['zh']}'
                )
        },
        {
            'role':'assistant',
            'content': (
                example['translation']['en']
            )
        }
    ]
    train_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = False
    )
    return {
        'train_text' : train_text
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

N = min(10000, len(dataset))

dataset = dataset.shuffle(seed=42)

dataset = dataset.select(range(N))

indices = list(range(N))

train_idx, temp_idx = train_test_split(
    indices, test_size = 0.2, random_state = 42
)
val1_idx, temp_idx2 = train_test_split(
    temp_idx, test_size = 0.75, random_state = 42
)
val2_idx, test_idx = train_test_split(
    temp_idx2, test_size = 2/3, random_state = 42
)

train_dataset = dataset.select(train_idx)
val1_dataset = dataset.select(val1_idx)
val2_dataset = dataset.select(val2_idx)
test_dataset = dataset.select(test_idx)

print("\n========== Split ==========")

print("train:", len(train_dataset))
print("val1 :", len(val1_dataset))
print("val2 :", len(val2_dataset))
print("test :", len(test_dataset))

os.makedirs(OUTDIR, exist_ok=True)

train_dataset.to_json(
    f"{OUTDIR}/WMT/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val1_dataset.to_json(
    f"{OUTDIR}/WMT/val1.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val2_dataset.to_json(
    f"{OUTDIR}/WMT/val2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_dataset.to_json(
    f"{OUTDIR}/WMT/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)
