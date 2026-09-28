import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
MODEL_PATH = "./model/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master"

DATA_ROOT = "./datas"

OUTPUT_ROOT = "./train1_continual_lora"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

TASKS = [

]
MAX_LENGTH = 1024
BATCH_SIZe = 8,
GRADIENT_ACCUMULATION = 2,
NUM_EPOCHS = 5,
LEARNING_RATE = 2e-4

tokenizer = AutoTokenizer(
    MODEL_PATH,
    trust_remote_code = True
)
model = AutoModelForCausalLM(
    MODEL_PATH,
    torch.dtype = torch.bfloat16,
    device_map = 'auto'
)
model.config.use_cache = False

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
    ]
    bias='none',
    task_type='CAUSAL_LM'
)
model = get_peft_model(
    model,
    lora_config
)
model.print_trainable_parameters()

for task_id, task_name in enumerate(TASKS, start=1):
    data_path = os.path.join(
        DATA_ROOT,
        task,name,
        'train.jsonl'
    )
    dataset = load_dataset(
        'json',
        'data_path',
        split = 'train'
    )
    def tokenize_function(examples):
        return tokenizer(
            example['train_text'],
            truncation = True,
            max_length = MAX_LENGTH
        )
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched = True,
        remove_columns=dataset.column_names,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False
    )
    output_dir = os.path.join(
        OUTPUT_ROOT,
        f'task{task_id}_{task_name}'
    )

    training_args = TrainingArguments(

        output_dir=output_dir,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,

        bf16=True,

        logging_steps=10
        
        eval_strategy = 'epoch'

        save_strategy = "epoch",

        save_total_limit = 1,

        report_to = "none",

        metric_for_best_model = 'eval_loss',

        greater_is_better=False,
        
        save_total_limit = 3,
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_dataset,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(
        output_dir
    )
    tokenizer.save_pretrained(
        output_dir
    )