from modelscope import snapshot_download

# 必须和其他脚本读的位置保持一致：
#   train1.py / evaluate1.py / prepare_activation.py 都用
#   ./models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
IGNORE_FILES = ['original/consolidated.00.pth']

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='./models',
    ignore_file_pattern=IGNORE_FILES,
)

print(model_dir)
