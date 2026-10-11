import os

from modelscope import snapshot_download

# 用脚本自身位置定位，避免依赖当前工作目录
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ModelScope 的落盘规则是
#   {cache_dir}/{type}s/{owner}--{name}/snapshots/{revision}
# 所以 cache_dir 要传"仓库根"，它自己会建出 models/ 这一层。
# 传 './models' 会多套一层变成 models/models/...，下游脚本就找不到权重了。
# 目标位置（train1.py / evaluate1.py / prepare_activation.py 都读这里）：
#   <仓库根>/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master
IGNORE_FILES = ['original/consolidated.00.pth']

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir=REPO_ROOT,
    ignore_file_pattern=IGNORE_FILES,
)

print(model_dir)
