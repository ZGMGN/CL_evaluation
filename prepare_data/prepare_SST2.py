import os
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('stanfordnlp/sst2', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                "Classify the sentiment of the following sentence.\n"
                "Output 0 for negative sentiment and 1 for positive sentiment.\n\n"
                f"SENTENCE: {example['sentence']}"
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
                "Classify the sentiment of the following sentence.\n"
                "Output 0 for negative sentiment and 1 for positive sentiment.\n\n"
                f"SENTENCE: {example['sentence']}"
            )
        },
        {
            'role': 'assistant',
            'content': str(example['label'])
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
        example['activation_text'],
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
    lambda x:x['token_length'] <= 2048
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

print("\n========== Split ==========")
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