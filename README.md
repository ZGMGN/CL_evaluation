# CL_evaluation

构建多任务"数据集"集合的准备脚本：从 HuggingFace 拉取原始数据 → 用 Llama-3.1 的 chat 模板构造**激活文本 / 训练文本** → 按类别分层抽样 → 切成 `train / val1 / val2 / test`。

生成的数据用于后续的激活值提取与数据集相似度分析。

## 环境要求

- **Python >= 3.12**（`prepare_TLDR.py`、`prepare_WMT.py` 使用了 3.12 的 f-string 语法，3.11 及以下会 `SyntaxError`）
- 依赖安装：

```bash
pip install -r requirements.txt
```

- 显存/磁盘：用 8B 模型提取激活值时 bf16 约需 15 GB 显存；模型权重约 15 GB。

## 网络说明

huggingface.co 直连不通时走镜像，脚本顶部已内置（**必须在 `import datasets` 之前设置才生效**）：

```python
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
```

模型权重用 ModelScope 下载（自动跳过与 safetensors 重复的 `consolidated.00.pth`，省约 16 GB）：

```bash
python load_model.py
```

## 目录结构

```
.
├── load_model.py          # 从 ModelScope 下载 Llama-3.1-8B-Instruct
├── prepare_data/
│   ├── CONFIG.py          # 数据集 id、输出目录、标注/无标注清单
│   └── prepare_*.py       # 10 个数据集的准备脚本
├── datas/                 # 生成的数据（不入库）
│   └── <NAME>/{train,val1,val2,test}.jsonl
├── models/                # 模型权重（不入库）
└── requirements.txt
```

## 使用方法

每个数据集一个独立脚本，逐个运行：

```bash
cd prepare_data
python prepare_MEDDIA.py
python prepare_TLDR.py
# ...
```

结果写到 `../datas/<NAME>/`。

## 数据集一览

| 脚本 | HuggingFace 数据集 | 划分 | 任务 | 类别数 | 各切分样本数 |
|---|---|---|---|---|---|
| `prepare_MEDDIA.py` | `lavita/ChatDoctor-HealthCareMagic-100k` | train | 医疗问答 | — | 8000/500/500/1000 |
| `prepare_TLDR.py` | `CarperAI/openai_summarize_tldr` | train | Reddit 摘要 | — | 8000/500/500/1000 |
| `prepare_WMT.py` | `wmt/wmt19` (zh-en) | train | 中译英 | — | 8000/500/500/1000 |
| `prepare_YELP.py` | `codyburker/yelp_review_sampled` | train | 1~5 星评分 | 5 | 8000/500/500/1000（每类 20%） |
| `prepare_AMAZON.py` | `goosmanlei/amazon_reviews_multi` (en) | train | 1~5 星评分 | 5 | 8000/500/500/1000（约 20%/类） |
| `prepare_SST2.py` | `stanfordnlp/sst2` | train | 情感二分类 | 2 | 8000/500/500/1000（50/50） |
| `prepare_IMDB.py` | `stanfordnlp/imdb` | train | 情感二分类 | 2 | 8000/500/500/1000（50/50） |
| `prepare_AGNEWS.py` | `fancyzhx/ag_news` | train | 新闻主题分类 | 4 | 8000/500/500/1000（每类 20%） |
| `prepare_DBPEDIA.py` | `fancyzhx/dbpedia_14` | train | 百科主题分类 | 14 | 8008/500/501/1001（每类 715，合计 10010） |
| `prepare_QQP.py` | `SetFit/qqp` | train | 问题等价判断 | 2 | 8000/500/500/1000（50/50 均衡） |

切分比例统一为 **train 80% / val1 5% / val2 5% / test 10%**，并对分类任务做分层切分（stratify），保证每个切分的类别比例一致。

## 输出格式

每个切分是一个 JSON Lines 文件，一行一条样本。所有数据集都包含以下两列：

- **`activation_text`**：chat 模板格式化后的**提问部分**（`<|begin_of_text|>` + system 段 + user 段，结尾是 `<|start_header_id|>assistant<|end_header_id|>`）——用于前向传播取激活值。
- **`train_text`**：`activation_text` + 回答 + `<|eot_id|>`——用于监督微调；`train_text` 以 `activation_text` 为前缀。

其余列保留原始字段（如 `text` / `sentence` / `title` / `label` / `stars` 等），便于错误分析与重新组模版。

## 复现说明

- 所有随机性都固定为 `SEED = 42`（抽样 shuffle 与分层切分）。
- **数据不进入仓库**：跑一遍脚本即可复现出相同的切分；原始数据从 HuggingFace 镜像拉取，模型权重用 `load_model.py` 下载。`datas/`、`models/` 已在 `.gitignore` 中忽略。

## 已知注意事项

1. `activation_text` 的 system 段包含 Llama-3.1 模板**默认注入**的 `Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024`（该日期是模板里的固定值，不随系统时间变化）。若要去掉，需传入自定义 `chat_template`。
2. 类别列名不统一：多数数据集用 `label`，`AMAZON` / `YELP` 用 `stars`；`TLDR` 的 `label` **是摘要文本而不是类别**，写通用加载逻辑时需注意。
3. `WMT` 只保留了 `activation_text` / `train_text`，未保留原始中英文。
4. 部分原始数据集自带重复文本（如 SST-2、IMDB、TLDR），因此切分之间可能存在极少量重复样本。
