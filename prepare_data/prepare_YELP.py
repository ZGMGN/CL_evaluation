import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from collections import Counter
from sklearn.model_selection import train_test_split
tokenizer = AutoTokenizer.from_pretrained('../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master', 
trust_remote_code=True)
OUTDIR = r'../datas'
dataset = load_dataset('codyburker/yelp_review_sampled', split = 'train')

def build_activation_text(example):
    messages = [{
        'role':'user',
        'content':("Predict the star rating of the following review from 1 to 5.\n\n"
                   f"REVIEW:{example['text']}"
        )
    }]
    activation_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = False
    )
    return {
        'activation_text' : activation_text
    }

def build_train_text(example):
    messages = [{
        'role':'user',
        'content':("Predict the star rating of the following review from 1 to 5.\n\n"
                   f"REVIEW:{example['text']}"
        )
    },
    {
        'role':'assistant',
        'content':str(example['stars'])
    }
    ]
    train_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = False,
        tokenize = False
    )
    return {
        'train_text' : train_text
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
star_groups = {}
for star in [1, 2, 3, 4, 5]:
    star_groups[star] = dataset.filter(lambda x : x['stars'] == star)
star_counts = {s: len(ds) for s,ds in star_groups.items()}
print("各个星级可用样本数量：", star_counts)
min_count = min(star_counts.values())
print(f"最少类别样本数 = {min_count}，每个星级抽取 {min_count} 条，均衡后总样本 {5*min_count}")

balanced_ds_list = []
for star, ds in star_groups.items():
    ds_shuffled = ds.shuffle(seed=42)
    ds_sampled = ds_shuffled.select(range(min_count))
    balanced_ds_list.append(ds_sampled)

balanced_dataset = concatenate_datasets(
    balanced_ds_list
)
balanced_dataset = balanced_dataset.shuffle(seed=42)
N = min(10000, len(balanced_dataset))
balanced_dataset = balanced_dataset.select(range(N))

# ================================================================================

# 接下来分层划分（此时数据集已经1:1，分层拆分保证子集依然1:1）
labels = balanced_dataset["stars"]
indices = list(range(N))

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

train_dataset = balanced_dataset.select(train_idx)
val1_dataset = balanced_dataset.select(val1_idx)
val2_dataset = balanced_dataset.select(val2_idx)
test_dataset = balanced_dataset.select(test_idx)

print("\n========== Split ==========")
print("train:", len(train_dataset))
print("val1 :", len(val1_dataset))
print("val2 :", len(val2_dataset))
print("test :", len(test_dataset))

# 打印各集合星级分布，校验均衡效果
def print_dist(name, ds):
    cnt = Counter(ds["stars"])
    total = sum(cnt.values())
    print(f"\n{name} star distribution(total={total}):")
    for star in sorted(cnt.keys()):
        print(f"  star {star}: {cnt[star]} ({cnt[star]/total:.2%})")

print_dist("train", train_dataset)
print_dist("val1", val1_dataset)
print_dist("val2", val2_dataset)
print_dist("test", test_dataset)

os.makedirs(OUTDIR, exist_ok=True)
train_dataset.to_json(
f"{OUTDIR}/YELP/train.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val1_dataset.to_json(
f"{OUTDIR}/YELP/val1.jsonl",
orient="records",
lines=True,
force_ascii=False
)
val2_dataset.to_json(
f"{OUTDIR}/YELP/val2.jsonl",
orient="records",
lines=True,
force_ascii=False
)
test_dataset.to_json(
f"{OUTDIR}/YELP/test.jsonl",
orient="records",
lines=True,
force_ascii=False
)