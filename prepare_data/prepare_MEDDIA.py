import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
print(f'原始数据量{len(dataset)}')


INSTRUCTION = (
    "If you are a doctor, please answer the medical questions "
    "based on the patient's description."
)

def build_data(example):
    input_text = example['input']
    output_text = example['output']
    return {
        'instruction':INSTRUCTION,
        'input':input_text,
        'output':output_text
    }
dataset = dataset.map(build_data)

def build_activation_text(example):
    messages = [
            {
            'role' : 'user',
            'content':(
                example['instruction'] 
                + '\n\n' 
                + example['input']
            )
        }
    ]
    activation_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {
        'activation_text':activation_text
    }
dataset = dataset.map(build_activation_text)

def build_train_text(example):
    messages = [
        {
            'role':'user',
            'content':(
                example['instruction']
                + '\n\n'
                + example['input']
            )
        },
        {
            'role':'assistant',
            'content':example['output']
        }
    ]
    train_text = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=False)
    return {
        'train_text' : train_text
    }
dataset = dataset.map(build_train_text)

def calculate_length(example):
    tokens = tokenizer(
        example['activation_text'],
        add_special_tokens=False,
        truncation=False)
    return {
        'token_length':len(tokens['input_ids'])
    }

dataset = dataset.map(calculate_length)

before = len(dataset)

dataset = dataset.filter(lambda x: x['token_length']<= 2048)

dataset = dataset.shuffle(seed=42)

N = min(
    10000, len(dataset)
)

dataset = dataset.select(range(N))

indices = list(range(len(dataset)))

train_idx, temp_idx = train_test_split(
    indices, test_size = 0.2, random_state = 42
)
val1_idx, temp_idx2 = train_test_split(
    temp_idx, test_size = 0.75, random_state = 42
)
val2_idx, test_idx = train_test_split(
    temp_idx2, test_size = 2/3, random_state = 42
)

dataset = dataset.remove_columns('token_length')

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
    f"{OUTDIR}/MEDDIA/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val1_dataset.to_json(
    f"{OUTDIR}/MEDDIA/val1.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val2_dataset.to_json(
    f"{OUTDIR}/MEDDIA/val2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_dataset.to_json(
    f"{OUTDIR}/MEDDIA/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)
