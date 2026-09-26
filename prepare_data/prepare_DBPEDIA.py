import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('fancyzhx/dbpedia_14', split = 'train')
def build_activation_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                "Classify the topic of the following text.\n"
                "Output the corresponding label from 0 to 13.\n"
                "0: Company\n"
                "1: EducationalInstitution\n"
                "2: Artist\n"
                "3: Athlete\n"
                "4: OfficeHolder\n"
                "5: MeanOfTransportation\n"
                "6: Building\n"
                "7: NaturalPlace\n"
                "8: Village\n"
                "9: Animal\n"
                "10: Plant\n"
                "11: Album\n"
                "12: Film\n"
                "13: WrittenWork\n\n"
                f"TITLE: {example['title']}\n"
                f"TEXT: {example['content']}"
            )
        }]
    activation_text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
    return {
            'activation_text':activation_text
        }
def build_train_text(example):
    messages = [
        {
            'role': 'user',
            'content': (
                "Classify the topic of the following text.\n"
                "Output the corresponding label from 0 to 13.\n"
                "0: Company\n"
                "1: EducationalInstitution\n"
                "2: Artist\n"
                "3: Athlete\n"
                "4: OfficeHolder\n"
                "5: MeanOfTransportation\n"
                "6: Building\n"
                "7: NaturalPlace\n"
                "8: Village\n"
                "9: Animal\n"
                "10: Plant\n"
                "11: Album\n"
                "12: Film\n"
                "13: WrittenWork\n\n"
                f"TITLE: {example['title']}\n"
                f"TEXT: {example['content']}"
            )
        },
        {
            'role':'assistant',
            'content':str(example['label'])
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
balanced_dataset_idx = {}
for i in range(14):
    balanced_dataset_idx[i] = dataset.filter(lambda x:x['label'] == i)

balanced_dataset1 = []
for i in range(14):
    balanced_dataset_idx[i] = balanced_dataset_idx[i].shuffle(seed = 42)
    balanced_dataset_idx[i] = balanced_dataset_idx[i].select(range(715))
    balanced_dataset1.append(balanced_dataset_idx[i])
balanced_dataset = concatenate_datasets(balanced_dataset1)
balanced_dataset = balanced_dataset.shuffle(seed = 42)

indices = list(range(len(balanced_dataset)))
labels = balanced_dataset['label']
train_idx, temp_idx1 = train_test_split(
    indices, test_size = 0.2, random_state = 42, stratify = [labels[i] for i in indices]
)
test_idx, temp_idx2 = train_test_split(
    temp_idx1, test_size = 0.5, random_state = 42, stratify = [labels[i] for i in temp_idx1]
)
val1_idx, val2_idx = train_test_split(
    temp_idx2, test_size = 0.5, random_state = 42, stratify = [labels[i] for i in temp_idx2]
)
train_dataset = balanced_dataset.select(train_idx)
val1_dataset = balanced_dataset.select(val1_idx)
val2_dataset = balanced_dataset.select(val2_idx)
test_dataset = balanced_dataset.select(test_idx)

print("\n========== Split ==========")

print("train:", len(train_dataset))
print("val1 :", len(val1_dataset))
print("val2 :", len(val2_dataset))
print("test :", len(test_dataset))

os.makedirs(OUTDIR, exist_ok=True)

train_dataset.to_json(
    f"{OUTDIR}/DBPEDIA/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val1_dataset.to_json(
    f"{OUTDIR}/DBPEDIA/val1.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val2_dataset.to_json(
    f"{OUTDIR}/DBPEDIA/val2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_dataset.to_json(
    f"{OUTDIR}/DBPEDIA/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)