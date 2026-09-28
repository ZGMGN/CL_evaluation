import json
import os
import random
import re
from collections import defaultdict
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
SEED = 42
OUTDIR = r'../datas'
SUMMARIZATION = r'CarperAI/openai_summarize_tldr'

dataset = load_dataset(SUMMARIZATION, split = 'train')
"""
reference:SUBREDDIT: r/relationships 
TITLE: I (f/22) have to figure out if I want to still know these girls or not and would hate to sound insulting ar of dating roand it effected me more than I thought. It was a horrible time in my life due to living with my mother and finally having the chance to cut her out of my life. I can admit because of it was an emotional wreck and this guy was stable and didn't know how to deal with me. We ended by him avoiding for a month or so after going to a festival with my friends. When I think back I wish he just ended. So after he ended it added my depression I suffered but my friends helped me through it and I got rid of everything from him along with cutting contact.   Now: Its been almost 3 years now and I've gotten better after counselling and mild anti depressants. My mother has been out of my life since then so there's been alot of progress. Being stronger after learning some lessons there been more insight about that time of my life but when I see him or a picture everything comes back. The emotions and memories bring me back down.   His friends (both girls) are on my facebook because we get along well which is hard to find and I know they'll always have his back. But seeing him in a picture or talking to him at a convention having a conversation is tough. Crying confront of my current boyfriend is something I want to avoid.   So I've been thinking that I have to cut contact with these girls because it's time to move on because it's healthier. It's best to avoid him as well. But will they be insulted? Will they accept it? Is there going to be awkwardness? I'm not sure if it's the right to do and could use some outside opinions. TL;DR:
POST: Not sure if this belongs here but it's worth a try.   Backstory: When I (f/22) went through my first real breakup 2 years ago because he needed space after a ye
"""
tokenizer = AutoTokenizer.from_pretrained(r'../models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master')

def build_data(example):
    PATTERN = r"""TITLE:\s*(.*?)\s+POST:\s*(.*?)\s+TL;DR:"""
    match = re.search(PATTERN, example['prompt'], re.DOTALL)
    title = ''
    post = ''
    if match:
        title = match.group(1).strip()
        post = match.group(2).strip()
    return {
        'title':title,
        'post':post
    }
dataset = dataset.map(build_data)

def build_activation_text(example):
    messages = [
        {
            'role':'user',
            'content':('Summarize the following Reddit post concisely.\n\n'
                        f'TITLE: {example['title']}\n\n'
                        f'POST: {example['post']}'
            )
        }
    ]
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
            'role':'user',
            'content':('Summarize the following Reddit post concisely.\n\n'
                        f'TITLE: {example['title']}\n\n'
                        f'POST: {example['post']}'
            )
        },
        {
            'role':'assistant',
            'content':example['label']
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
    f"{OUTDIR}/TLDR/train.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val1_dataset.to_json(
    f"{OUTDIR}/TLDR/val1.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

val2_dataset.to_json(
    f"{OUTDIR}/TLDR/val2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

test_dataset.to_json(
    f"{OUTDIR}/TLDR/test.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)