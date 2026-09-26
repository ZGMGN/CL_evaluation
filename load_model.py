from modelscope import snapshot_download

IGNORE_FILES = ['original/consolidated.00.pth']

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='./model',
    ignore_file_pattern=IGNORE_FILES,
)

print(model_dir)
