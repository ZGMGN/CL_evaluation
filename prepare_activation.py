import os
import torch

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================
# 1. Model
# ============================================================

MODEL_PATH = './models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master'

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# Llama 通常没有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map='auto',
    output_hidden_states=True
)

model.eval()


# ============================================================
# 2. Dataset
# ============================================================

DATA_NAME = [
    'AGNEWS',
    'AMAZON',
    'DBPEDIA',
    'IMDB',
    'MEDDIA',
    'QQP',
    'SST2',
    'TLDR',
    'WMT',
    'YELP'
]


# ============================================================
# 3. Layers
# ============================================================

# 一次 forward 同时提取这些层
LAYERS = [
    8,
    12,
    16,
    20,
    24,
    28,
    32
]


# ============================================================
# 4. Batch size
# ============================================================

# 4090D 建议先从 4 开始
BATCH_SIZE = 4


# ============================================================
# 5. Layer activation extraction
# ============================================================

def get_layer_activation_batch(texts):

    # ----------------------------------------
    # tokenize
    # ----------------------------------------

    inputs = tokenizer(
        texts,
        # activation_text 已经自带 <|begin_of_text|>，
        # 不关掉自动补 BOS 会变成两个 BOS
        add_special_tokens=False,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=1024
    )

    # 放到模型所在设备
    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    # ----------------------------------------
    # forward
    # ----------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs,
            output_hidden_states=True
        )

    # ----------------------------------------
    # attention mask
    # ----------------------------------------

    attention_mask = inputs['attention_mask']

    # [B, L]
    # ↓
    # [B, L, 1]

    mask = attention_mask.unsqueeze(-1)

    layer_vectors = {}

    # ----------------------------------------
    # process every selected layer
    # ----------------------------------------

    for layer in LAYERS:

        # [B, L, H]
        hidden = outputs.hidden_states[layer]

        # FP16 → FP32
        hidden = hidden.float()

        # mask 也转换成 FP32
        mask_float = mask.to(hidden.dtype)

        # ------------------------------------
        # masked mean pooling
        # ------------------------------------

        weighted_sum = (
            hidden * mask_float
        ).sum(dim=1)

        token_count = mask_float.sum(dim=1)

        # [B, H]
        vectors = (
            weighted_sum / token_count
        )

        layer_vectors[layer] = vectors

    return layer_vectors


# ============================================================
# 6. Create output directory
# ============================================================

os.makedirs(
    'activation',
    exist_ok=True
)


# ============================================================
# 7. Process every dataset
# ============================================================

for name in DATA_NAME:

    print()
    print("=" * 70)
    print(f"Processing dataset: {name}")
    print("=" * 70)

    # ----------------------------------------
    # load validation data
    # ----------------------------------------

    path = f'datas_clean/{name}/val2.jsonl'

    dataset = load_dataset(
        'json',
        data_files=path,
        split='train'
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    # ----------------------------------------
    # create list for every layer
    # ----------------------------------------

    all_vectors = {
        layer: []
        for layer in LAYERS
    }

    # ----------------------------------------
    # batch inference
    # ----------------------------------------

    for start in range(
        0,
        len(dataset),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(dataset)
        )

        # 当前 batch 的文本
        texts = dataset[start:end][
            'activation_text'
        ]

        # 一次 forward 得到多个 layer
        layer_vectors = get_layer_activation_batch(
            texts
        )

        # ------------------------------------
        # 保存每个 layer 的结果
        # ------------------------------------

        for layer in LAYERS:

            all_vectors[layer].append(
                layer_vectors[layer].cpu()
            )

        # ------------------------------------
        # progress
        # ------------------------------------

        if (
            end % 50 == 0
            or end == len(dataset)
        ):

            print(
                f"{end}/{len(dataset)}"
            )

    # ========================================================
    # 8. Save every layer
    # ========================================================

    for layer in LAYERS:

        # ----------------------------------------
        # [N, 4096]
        # ----------------------------------------

        vectors = torch.cat(
            all_vectors[layer],
            dim=0
        )

        # 确保 FP32
        vectors = vectors.float()

        # ----------------------------------------
        # dataset vector
        # ----------------------------------------

        dataset_vector = vectors.mean(
            dim=0
        )

        dataset_vector = dataset_vector.float()

        # ----------------------------------------
        # print information
        # ----------------------------------------

        print()
        print(
            f"Layer {layer}:"
        )

        print(
            f"  sample vectors shape: "
            f"{vectors.shape}"
        )

        print(
            f"  sample vectors dtype: "
            f"{vectors.dtype}"
        )

        print(
            f"  dataset vector shape: "
            f"{dataset_vector.shape}"
        )

        print(
            f"  dataset vector dtype: "
            f"{dataset_vector.dtype}"
        )

        # ----------------------------------------
        # save sample vectors
        # ----------------------------------------

        torch.save(
            vectors,
            f'activation/'
            f'{name}_layer{layer}_samples.pt'
        )

        # ----------------------------------------
        # save dataset vector
        # ----------------------------------------

        torch.save(
            dataset_vector,
            f'activation/'
            f'{name}_layer{layer}_dataset.pt'
        )

    print()
    print(
        f"{name} finished."
    )


print()
print("=" * 70)
print("All datasets finished.")
print("=" * 70)
