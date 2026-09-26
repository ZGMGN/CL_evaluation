import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('goosmanlei/amazon_reviews_multi', 'en', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role':'user',
            'content':("Predict the star rating of the following review from 1 to 5.\n\n"
                       f"REVIEW_TITLE:{example['review_title']}\n"
                       f"REVIEW_BODY:{example['review_body']}"
            )
        }
    ]
    activation_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = False
    )
    return {
        'activation_text':activation_text
    }

def build_train_text(example):
    messages = [
        {
            'role':'user',
            'content':("Predict the star rating of the following review from 1 to 5.\n\n"
                       f"REVIEW_TITLE:{example['review_title']}\n"
                       f"REVIEW_BODY:{example['review_body']}"
            )
        },
        {
            'role':'assistant',
            'content':str(example['stars'])
        }
    ]
    train_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = False
    )
    return {
        'train_text':train_text
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


N = min(10000, len(dataset))
dataset = dataset.shuffle(seed=42)
indices = list(range(N))
labels = dataset['stars']

# 第一次拆分：train 80%，temp 20%
train_idx, temp_idx = train_test_split(
    indices, test_size = 0.2, random_state = 42, stratify=[labels[i] for i in indices]
)
temp_labels = [labels[i] for i in temp_idx]

# 第二次拆分 temp → val1(25% of temp) + temp2(75% of temp)
val1_idx, temp_idx2 = train_test_split(
    temp_idx, test_size = 0.75, random_state = 42, stratify=temp_labels
)
temp2_labels = [labels[i] for i in temp_idx2]

# 第三次拆分 temp2 → val2(1/3 of temp2) + test(2/3 of temp2)
val2_idx, test_idx = train_test_split(
    temp_idx2, test_size = 2/3, random_state = 42, stratify=temp2_labels
)

train_dataset = dataset.select(train_idx)
val1_dataset = dataset.select(val1_idx)
val2_dataset = dataset.select(val2_idx)
test_dataset = dataset.select(test_idx)

os.makedirs(OUTDIR, exist_ok=True)
train_dataset.to_json(
f"{OUTDIR}/AMAZON/train.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val1_dataset.to_json(
f"{OUTDIR}/AMAZON/val1.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val2_dataset.to_json(
f"{OUTDIR}/AMAZON/val2.jsonl",
orient="records",
lines=True,
force_ascii=False
)
test_dataset.to_json(
f"{OUTDIR}/AMAZON/test.jsonl",
orient="records",
lines=True,
force_ascii=False
)
