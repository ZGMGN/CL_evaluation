import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('stanfordnlp/imdb', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                "Classify the sentiment of the following text.\n"
                "Output 0 for negative sentiment and 1 for positive sentiment.\n\n"
                f"TEXT: {example['text']}"
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
                "Classify the sentiment of the following text.\n"
                "Output 0 for negative sentiment and 1 for positive sentiment.\n\n"
                f"TEXT: {example['text']}"
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
dataset = dataset.shuffle(seed = 42)
label = dataset['label']
indices = list(range(10000))
train_idx, temp_idx1 = train_test_split(
    indices, test_size = 0.2, random_state = 42, stratify = [label[i] for i in indices])
test_idx, temp_idx2 = train_test_split(
    temp_idx1, test_size = 0.5, random_state = 42, stratify = [label[i] for i in temp_idx1]
)
val1_idx, val2_idx = train_test_split(
    temp_idx2, test_size = 0.5, random_state = 42, stratify = [label[i] for i in temp_idx2]
)

train_dataset = dataset.select(train_idx)
test_dataset = dataset.select(test_idx)
val1_dataset = dataset.select(val1_idx)
val2_dataset = dataset.select(val2_idx)

train_dataset.to_json(
    f'{OUTDIR}/IMDB/train.jsonl',
    orient = 'records',
    lines = True,
    force_ascii = False
)
test_dataset.to_json(
    f'{OUTDIR}/IMDB/test.jsonl',
    orient = 'records',
    lines = True,
    force_ascii = False
)
val1_dataset.to_json(
    f'{OUTDIR}/IMDB/val1.jsonl',
    orient = 'records',
    lines = True,
    force_ascii = False
)
val2_dataset.to_json(
    f'{OUTDIR}/IMDB/val2.jsonl',
    orient = 'records',
    lines = True,
    force_ascii = False
)