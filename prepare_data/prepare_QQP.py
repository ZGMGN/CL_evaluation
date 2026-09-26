import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('SetFit/qqp', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role':'user',
            'content' :(
                "Determine whether the following two questions have the same meaning.\n"
                "Output 0 if they are not equivalent and 1 if they are equivalent.\n\n"
                f"QUESTION 1: {example['text1']}\n"
                f"QUESTION 2: {example['text2']}"
            )
        }
    ]
    activation_text = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = True
    )
    return {
        'activation_text':activation_text
    }
def build_train_text(example):
    messages = [
        {
            'role':'user',
            'content' : (
                "Determine whether the following two questions have the same meaning.\n"
                "Output 0 if they are not equivalent and 1 if they are equivalent.\n\n"
                f"QUESTION 1: {example['text1']}\n"
                f"QUESTION 2: {example['text2']}"
            )
        },
        {
            'role':'assistant',
            'content':str(example['label'])
        }
    ]
    train_text = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = False
    )
    return {
        'train_text':train_text
    }

def calculate_token_length(example):
    tokens = tokenizer(example['activation_text'],
                        add_special_tokens = False,
                        truncation = False)
    return {
        'token_length': len(tokens['input_ids'])
    }
dataset = dataset.map(build_activation_text)
dataset = dataset.map(build_train_text)
dataset = dataset.map(calculate_token_length)
dataset = dataset.filter( lambda x : x['token_length']<= 2048)
dataset = dataset.remove_columns('token_length')
dataset = dataset.shuffle(seed = 42)
indices_temp = list(range(len(dataset)))
labels = dataset['label']
mid_indices, mid_indices2 = train_test_split(
    indices_temp, test_size = 10000, random_state = 42, stratify = [labels[i] for i in indices_temp]
)
train_idx, temp_idx1 = train_test_split(
    mid_indices2, test_size = 0.2, random_state = 42, stratify = [labels[i] for i in mid_indices2]
)
test_idx, temp_idx2 = train_test_split(
    temp_idx1, test_size = 0.5, random_state = 42, stratify = [labels[i] for i in temp_idx1]
)
val1_idx, val2_idx = train_test_split(
    temp_idx2, test_size = 0.5, random_state = 42, stratify = [labels[i] for i in temp_idx2]
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
    f"{OUTDIR}/QQP/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val1_dataset.to_json(
    f"{OUTDIR}/QQP/val1.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val2_dataset.to_json(
    f"{OUTDIR}/QQP/val2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_dataset.to_json(
    f"{OUTDIR}/QQP/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)