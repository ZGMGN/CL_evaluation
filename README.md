# CL_evaluation

围绕 Llama-3.1-8B-Instruct 的多任务持续学习（continual learning）评测流水线，覆盖 10 个下游任务。

整条链路分两部分：

- **激活值分析**（第 1–4 步）用于观察**任务之间的相似度结构**
- **训练与评测**（第 5–6 步）用于观察**持续微调后各任务性能的变化**

```
1. 准备数据    prepare_data/prepare_*.py    原始数据 -> chat 模板文本，分层抽样切分   -> datas/
2. 清洗数据    prepare_data/data_clear.py   去掉模板默认注入的日期文本                -> datas_clean/
3. 提取激活值  prepare_activation.py        用 val2 提取 7 个层的激活值               -> activation/
4. 相似度分析  calculate_similarity.py      数据集两两余弦相似度                      -> similarity/
               plot_*.py                   画热力图                                 -> heatmap/  heatmap_single_sample/
5. 持续训练    train1.py                    按任务顺序依次训练 LoRA adapter          -> train1_continual_lora/
6. 评测        evaluate1.py                 在各任务 test 集上算 accuracy/BLEU/ROUGE  -> evaluate1_results/
```

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
├── load_model.py            # 从 ModelScope 下载 Llama-3.1-8B-Instruct
├── requirements.txt
│
├── prepare_data/
│   ├── CONFIG.py            # 数据集 id 清单（目前没有脚本引用，仅作文档）
│   ├── prepare_*.py         # 10 个数据集的准备脚本
│   └── data_clear.py        # 清洗：去掉模板默认注入的日期文本
│
├── prepare_activation.py    # 提取激活值（读 datas_clean/，写 activation/）
├── calculate_similarity.py  # 数据集两两余弦相似度（读 activation/，写 similarity/）
├── plot_heatmap.py          # 相似度热力图（读 similarity/，写 heatmap/）
├── plot_single_sample_heatmap.py   # 单样本热力图（写 heatmap_single_sample/）
│
├── train1.py                # 持续学习训练（按任务顺序累积训练 LoRA）
├── evaluate1.py             # 评测（基座模型 或 指定 adapter）
│
├── datas_clean/             # 清洗后的数据（不入库，data_clear.py 生成）
├── activation/              # 激活值（不入库，体积大）
├── similarity/              # 相似度矩阵（入库）
├── heatmap/                 # 相似度热力图（入库）
├── heatmap_single_sample/   # 单样本热力图（入库）
├── train1_continual_lora/   # 训练好的 LoRA adapter（不入库）
├── evaluate1_results/       # 评测结果（不入库）
└── models/                  # 模型权重（不入库）
```

## 使用方法

### 1. 下载模型

```bash
python load_model.py
```

模型落到 `./models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master`。**下面所有脚本都从这个路径读，`cache_dir` 不要改。**

### 2. 生成数据

每个数据集一个独立脚本，逐个运行：

```bash
cd prepare_data
python prepare_MEDDIA.py
python prepare_TLDR.py
# ... 共 10 个
```

结果写到 `../datas/<NAME>/{train,val1,val2,test}.jsonl`。

### 3. 清洗数据

```bash
cd prepare_data
python data_clear.py
```

去掉 chat 模板默认注入的 `Cutting Knowledge Date / Today Date` 两行，输出到 `../datas_clean/<NAME>/`，目录结构保持不变。

**为什么必须清洗**：这段日期文本在所有数据集里**完全相同**，而激活值是对整段序列做 mean pooling。不清洗的话，这块共享文本会系统性抬高所有任务之间的相似度、掩盖真正的任务差异（实测 layer16 的公共方向占比会从 79% 升到 93%）。

### 4. 提取激活值 + 相似度分析

```bash
python prepare_activation.py     # 约 3-5 分钟，显存约 15 GB
python calculate_similarity.py
python plot_heatmap.py
python plot_single_sample_heatmap.py
```

做法：每个任务的 `val2` 切分（约 500 条）过一遍模型，取 7 个层（8/12/16/20/24/28/32）的 hidden states 做 masked mean pooling；每个数据集取样本向量的均值得到"数据集向量"，再算两两余弦相似度。

`plot_single_sample_heatmap.py` 固定取每个数据集的第 0 条样本，**只有一条样本、噪声很大**，仅作粗略参考。

### 5. 持续训练

```bash
# 按默认顺序训练全部 10 个任务
python train1.py

# 只训练指定的 4 个任务（按给定顺序）
python train1.py --tasks SST2 AGNEWS QQP TLDR

# 冒烟测试：每任务 64 条、只跑 2 步
python train1.py --tasks SST2 --max-samples 64 --max-steps 2
```

**adapter 按"已完成任务的前缀"命名**，目录名本身就记录了跑过哪些任务：

```
train1_continual_lora/
    SST2/                     <- 训完 SST2
    SST2+AGNEWS/              <- 再训到 AGNEWS
    SST2+AGNEWS+QQP+TLDR/     <- 再训到 TLDR
```

每个目录里有 `adapter_model.safetensors` / `adapter_config.json` / `tokenizer*`，以及一份 **`run.json`**，记录了这次的任务列表和全部超参。

**自动复用前缀**：每次启动会从长到短查找"已训完的任务前缀"，命中就跳过前面的任务、从那个 adapter 接着训：

```
$ python train1.py --tasks SST2 AGNEWS QQP TLDR        # 第 1 次
$ python train1.py --tasks SST2 AGNEWS QQP TLDR WMT    # 第 2 次
复用已有 adapter：SST2+AGNEWS+QQP+TLDR
  跳过已训练完的 4 个任务：['SST2', 'AGNEWS', 'QQP', 'TLDR']
  接着训练：['WMT']
```

换任务序列时不用重复训练公共前缀。复用前会自动比对 `run.json` 里的超参，不一致会给出警告。加 `--no-resume` 强制从头训练。

**训练细节**：

- **只对回答段计算 loss**：`train_text` 里 `activation_text` 之后的部分参与 loss，提问段设成 `-100`。数据构造保证 `activation_text` 的 token 正好是 `train_text` 的前缀（已对 9000 条抽样验证）。加 `--no-mask-prompt` 可改回整段计算。
- 默认开**梯度检查点**省显存（`--no-grad-checkpoint` 关闭）。
- 默认在训练结束时加载 **val1 上 `eval_loss` 最好的 checkpoint**（`--no-load-best` 关闭）。

常用参数：`--epochs`（默认 5）、`--batch-size`（默认 4）、`--grad-accum`（默认 4）、`--lr`（默认 2e-4）、`--output-dir`。

### 6. 评测

```bash
# 基座模型，全部任务
python evaluate1.py

# 评测某个 adapter
python evaluate1.py --adapter SST2+AGNEWS+QQP+TLDR

# 只评测 4 个任务，每个任务取前 50 条
python evaluate1.py --tasks SST2 AGNEWS QQP TLDR --max-samples 50

# 依次评测所有 adapter（持续学习矩阵）
python evaluate1.py --all-adapters
```

做法：把 `activation_text` 喂给模型做贪心解码，再和 `train_text` 里的标准答案比对。

| 任务类型 | 主指标 |
|---|---|
| 7 个分类任务 | `accuracy`（从生成文本里抽标签，带取值范围校验，避免抓到年份/型号） |
| `WMT` | `BLEU`（sacreBLEU，corpus 级） |
| `TLDR` / `MEDDIA` | `ROUGE-1 / 2 / L` |

结果写到 `<输出目录>/<任务名>.jsonl`（逐条 gold/pred 明细）和 `summary.json`（各任务指标 + 输入长度统计 + 本次配置）。默认输出目录是 `evaluate1_results/<adapter名 或 base>/`，不同 adapter 不会互相覆盖。

## 依赖关系：改了上游必须重跑下游

```
prepare_*.py -> data_clear.py -> prepare_activation.py -> calculate_similarity.py -> plot_*.py
                                          ^
                                          |
                        train1.py / evaluate1.py 也读 datas_clean/
```

**改了任意一步的脚本，它下游的产物都要重跑**，否则下游用的仍是旧数据——而且**不会报任何错，只会静默给出过期结论**。

两个真实教训：

1. `prepare_activation.py` 早期版本漏了 `add_special_tokens=False`，导致文本开头的 `<|begin_of_text|>` 被重复编码成两个 BOS。因为数据文件里存的是**文本**而不是 token id，这个问题不出现在任何数据文件里，只存在于"文本 → token"的那一瞬间，藏了很久才被发现。修复后 `activation/`、`similarity/`、`heatmap/` 全部重算，好在相似度的**排序结论没变**（Spearman ρ ≥ 0.96），只是早期层的公共方向占比从 93% 降到 79%。
2. `datas_clean/` 是从 `datas/` 派生的，`activation/` 又是从 `datas_clean/` 派生的。所以重跑 `prepare_*.py` 之后**必须重新跑 `data_clear.py`**，否则 `datas_clean/` 还是旧内容。

## 数据集一览

| 脚本 | HuggingFace 数据集 | 划分 | 任务 | 类别数 | 各切分样本数 |
|---|---|---|---|---|---|
| `prepare_MEDDIA.py` | `lavita/ChatDoctor-HealthCareMagic-100k` | train | 医疗问答 | — | 8000/500/500/1000 |
| `prepare_TLDR.py` | `CarperAI/openai_summarize_tldr` | train | Reddit 摘要 | — | 8000/500/500/1000 |
| `prepare_WMT.py` | `wmt/wmt19` (zh-en) | train | 中译英 | — | 8000/500/500/1000 |
| `prepare_YELP.py` | `codyburker/yelp_review_sampled` | train | 1~5 星评分 | 5 | 8000/500/500/1000（每类 20%）；手工删掉 1 条 `#NAME?` 脏数据后为 7999/500/499/999 |
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
- **数据不进入仓库**：跑一遍脚本即可复现出相同的切分；原始数据从 HuggingFace 镜像拉取，模型权重用 `load_model.py` 下载。
- 已在 `.gitignore` 中忽略：`datas/`、`datas_clean/`、`activation/`、`models/`、`hf_cache/`、`evaluate1_results/`、`train1_continual_lora/` 以及所有 `*.jsonl`。
- **入库的是分析产物**：`similarity/`（相似度矩阵）、`heatmap/`、`heatmap_single_sample/`。

## 已知注意事项

**关于数据本身**

1. `prepare_*.py` 产出的 `activation_text` 的 system 段包含 Llama-3.1 模板**默认注入**的 `Cutting Knowledge Date: December 2023 / Today Date: 26 Jul 2024`（模板里的固定值，不随系统时间变化）。`data_clear.py` 就是用来去掉这两行的；下游一律用 `datas_clean/`。
2. 类别列名不统一：多数数据集用 `label`，`AMAZON` / `YELP` 用 `stars`；`TLDR` 的 `label` **是摘要文本而不是类别**，写通用加载逻辑时需注意。
3. `WMT` 保留了原始 `translation`（`{en, zh}`）字段，其余数据集也保留了各自原始字段（`text` / `sentence` / `title` 等），便于错误分析。
4. 部分原始数据集自带重复文本（SST-2、IMDB、TLDR、WMT、YELP），切分之间可能存在极少量重复样本，全库精确重复约 41 行。**YELP 里有一条 `#NAME?` 的脏数据**（原始数据集的 Excel 导出错误），被贴了 2/3/4 三个不同星级，当前数据里已手工删除 3 行，所以 YELP 是 `7999/500/499/999` 而不是表格里的 `8000/500/500/1000`。`prepare_YELP.py` 里尚未加过滤，重跑脚本会复现。WMT 的重复对同理（`prepare_WMT.py` 未去重）。
5. `AGNEWS` 的原始文本有大量 HTML 实体残留（`&amp;` → ` amp;`、`&#39;` → ` #39;`、`&lt;` 等），来自上游数据集本身，占比约 25%。这是已知的 AG News 特性，未做清理。

**关于训练与评测**

6. **分类任务的回答只占整条序列的 0.6%–3.7% 的 token**（如 IMDB：329 个 token 里只有 2 个是回答）。所以 `train1.py` 默认只对回答段计算 loss——不掩码的话 96% 以上的梯度都花在复现提问上。详见 `--no-mask-prompt` 的说明。
7. `evaluate1.py` 是在**生成文本**上做匹配的。基座模型（未经 SFT）不会老实输出单个标签，而是"先解释、最后给数字"，所以抽取逻辑做了多重兜底：① 开头就是数字 → ② `Output: N` 句式 → ③ 任务特定语义（QQP 的 equivalent / not equivalent）→ ④ 第一个落在合法取值范围内的数字。同时会把"生成长度不够、还没写出数字"的情况暴露出来（`MAX_NEW_TOKENS` 太小会把准确率压成 0，这也是为什么 QQP 单独设成 64）。
8. **`MEDDIA` 用 ROUGE 作主指标要谨慎**：它是开放式医疗问答，参考答案是一整段医生回复，措辞差异极大，即使医学上正确 ROUGE 也会偏低。建议只作参考，`TLDR` 用 ROUGE 才是标准做法。
9. `WMT` 的 BLEU 计算前会过滤掉空预测/空参考（sacreBLEU 对空串会报错），被排除的条数会打印出来，建议在报告里注明 BLEU 的实际样本数。
10. `rouge-score` / `sacrebleu` / `peft` 缺失时，指标计算不会让整个评测崩溃——明细会先落盘，`summary.json` 里对应指标为 `null`。但正式实验前请确认 `pip install -r requirements.txt` 装齐。
