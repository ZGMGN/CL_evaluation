# 分析指标清单（做了什么 / 怎么算 / 为什么 / 怎么比较）

对应 `analysis/` 下的四个脚本。所有指标默认都在 **val2** 上算（`lora_cosine.py` 只读权重不碰数据），
因为 `train1.py` 用 `train.jsonl` 训练、`val1.jsonl` 选 checkpoint，val2 是干净留出集。

---

## 0. 总览

| 脚本 | 指标 | 回答的问题 | 需要 GPU | 主要产出 |
|---|---|---|---|---|
| `lora_cosine.py` | **Actual ΔW Cosine**（含一系列派生量） | 权重实际往哪动了？ | 否（CPU，约 9 秒） | `stage_cosine.csv` `stage_drift.csv` `task_delta_cosine.csv` + 逐层版 |
| `grad_cosine.py` | **Gradient Cosine** | 理论上应该往哪动？| 是（每阶段一次加载） | `grad_cosine.json` `cosine_<stage>.csv/png` |
| `representation_drift.py` | **Representation Drift** | 内部表示被推走多少？ | 是（前后各一次加载） | `cosine/cka/linear_r2/rel_l2.csv/png` `interpretation.csv` |
| `functional_drift.py` | **Functional Drift** | 输出分布（行为）变了多少？ | 是（前后各一次加载） | `metrics.csv` `symmetric_kl.png` `entropy.png` |

### 因果链

```
Gradient Cosine        理论上"应该"往哪个方向更新
        ↓  （对比：想走的 vs 实际走的）
Actual ΔW Cosine       训练实际走出来的方向（δW 两两余弦）
        ↓
Representation Drift   内部表示因此改变了多少
        ↓
Functional Drift       最终行为（输出分布）因此改变了多少
```

---

## 1. LoRA 参数漂移（`analysis/lora_cosine.py`）

### 基本量：ΔW 与 δW

LoRA 的等效权重增量是

```
ΔW_k = B_k A_k · (α / r)        ← 训练到第 k 个任务为止的【累计】增量
δW_k = ΔW_k − ΔW_{k-1}          ← 第 k 个任务【自己】造成的增量（base 记作 ΔW_0 = 0）
```

**为什么不能直接比 A、B**：A、B 有缩放不变性（`A→sA, B→B/s` 给出同一个 ΔW），
把 A、B 拼起来算余弦会把尺度差异当成方向差异。必须先还原成 ΔW。

**为什么不把 ΔW 拼成大向量**：一个阶段全部 ΔW 有 **13 亿个数（5.4 GB）**，
而且 float32 累加 13 亿项的误差足以让"自己跟自己"的余弦算成 **1.387**（实测）。
所以实现上用的是迹恒等式，只算 r×r 的小矩阵：

```
⟨B₁A₁, B₂A₂⟩ = tr(A₁ᵀ B₁ᵀ B₂ A₂) = Σ_ij (B₁ᵀB₂)_ij · (A₂A₁ᵀ)_ji
```

（14 行就够，比显式构造快约 130 倍，内存从 5.4 GB 降到几百 MB；与显式构造的结果
相对误差 ~1e-6，就是 float32 的舍入。）

### 1.1 累计 ΔW 两两余弦 → `stage_cosine.csv`

`cos(ΔW_i, ΔW_j)`，任意两个阶段之间。对角线恒为 1，矩阵对称。

- **目的**：看整条训练轨迹是不是沿着一个方向走，还是每来一个任务就拐一次弯。
- **怎么比较**：离对角线越远越小 = 拐得越远；相邻两个阶段之间掉得特别快，说明那个任务
  带来了大转弯。
- 实测（AGNEWS → QQP → SST2 → TLDR）：
  `AGNEWS vs +QQP 0.643`、`vs +QQP+SST2 0.494`、`vs +QQP+SST2+TLDR 0.416`。

### 1.2 相邻阶段：强化还是抵消 → `stage_drift.csv`

每列的含义：

| 列 | 定义 | 含义 |
|---|---|---|
| `cos_to_prev` | `cos(ΔW_k, ΔW_{k-1})` | 两个累计量之间的夹角（会被惯性稀释） |
| **`cos_update_to_prev`** | **`cos(δW_k, ΔW_{k-1})`** | **>0 强化旧方向 / <0 抵消旧方向** |
| `cos_update_to_curr` | `cos(δW_k, ΔW_k)` | ≈1 说明这一段基本只是延长旧方向 |
| `norm_prev` / `norm_curr` | `‖ΔW‖` | 累计增量的大小 |
| `relative_norm_change` | `‖ΔW_k‖ / ‖ΔW_{k-1}‖` | 相对幅度变化 |
| `norm_update` / `norm_update_over_prev` | `‖δW_k‖` 及其比值 | 这一段改动的幅度 |
| `cancel_ratio` | `max(0, 1 − cos_update_to_prev)` | 粗略的"抵消比例" |

全部从三个标量推出，不需要构造 `δW`：

```
⟨δW_k, ΔW_{k-1}⟩ = ⟨ΔW_k,ΔW_{k-1}⟩ − ‖ΔW_{k-1}‖²
⟨δW_k, ΔW_k⟩     = ‖ΔW_k‖² − ⟨ΔW_k,ΔW_{k-1}⟩
‖δW_k‖²          = ‖ΔW_k‖² + ‖ΔW_{k-1}‖² − 2⟨ΔW_k,ΔW_{k-1}⟩
```

- **目的**：回答"当前任务产生的新更新，是继续强化过去的方向，还是在抵消它"。
- **为什么比 `cos(ΔW_k, ΔW_{k-1})` 好**：累计量存在惯性。实测两者给出**相反**的印象：

  ```
  阶段              cos(ΔW_k,ΔW_{k-1})   cos(δW_k,ΔW_{k-1})
  AGNEWS → +QQP             +0.643              +0.001
  +QQP   → +SST2            +0.772              -0.034
  +SST2  → +TLDR            +0.830              -0.148
                          ↑ 看着越来越"一致"    ↑ 实际越来越在"抵消"
  ```

- **怎么比较**：> 0 表示同向加强（多任务互帮）；≈ 0 表示开辟新方向；< 0 表示这一任务
  在拆前面任务的台。绝对值越大越明显。

### 1.3 逐层版本 → `layer_cosine.csv` / `layer_update_cosine.csv`

行 = 相邻阶段，列 = 层（默认 32 层全部）。前者是 `cos(ΔW_k,ΔW_{k-1})`，后者是
**每层的 `cos(δW_k, ΔW_{k-1})`**。

- **目的**：找出"哪几层在加强、哪几层在抵消"。
- 实测：`AGNEWS→+QQP` 16/32 层为正（近似正交）、`+QQP→+SST2` 5/32 为正、
  `+SST2→+TLDR` **0/32 层为正**（每层都在抵消）；反复出现在最负位置的是 L31、L9、L8。

### 1.4 任务 × 任务的 δW 余弦 → `task_delta_cosine.csv`（= Actual ΔW Cosine）

把每个任务**自己造成的**更新拿出来两两比较：`cos(δW_i, δW_j)`。

```
⟨δW_i, δW_j⟩ = ⟨ΔW_i,ΔW_j⟩ − ⟨ΔW_i,ΔW_{j-1}⟩ − ⟨ΔW_{i-1},ΔW_j⟩ + ⟨ΔW_{i-1},ΔW_{j-1}⟩
```

- **目的**：和 Gradient Cosine 配对 —— 后者是"理论上应该往哪走"，这个是"实际走出来了哪"
  （"理论方向 vs 实际方向"的完整链条见第 5 节）。
- **怎么比较**：> 0 两个任务的更新同向（互相加强）；≈ 0 各干各的；< 0 互相抵消。
  再结合 `‖δW_i‖`（每个任务更新的幅度）判断冲突来自方向还是幅度。
- 实测：

  ```
            AGNEWS    QQP     SST2    TLDR    ‖δW‖
  AGNEWS     1.000   0.001  -0.025  -0.064   19.25
  QQP        0.001   1.000  -0.024  -0.091   22.94
  SST2      -0.025  -0.024   1.000  -0.095   24.03
  TLDR      -0.064  -0.091  -0.095   1.000   23.31
  ```

  AGNEWS 与 QQP 几乎完全正交（+0.001）；含 SST2/TLDR 的组合全为负；四个任务的 `‖δW‖`
  差不多（19~24），说明**冲突来自方向而不是幅度**（不是"某一步更新特别大把前面冲掉"）。
- 逐层版 `layer_task_delta_cosine.csv/.png`（行=任务对，列=层）：TLDR 相关的三对在
  29~30/32 层上都是负的。

### 注意事项

- 这些都是**累加**关系：目录名里的 `A+B/` 是"训过 A、B 后的总增量"，不是 B 单独的。
  单独量由相邻累计量相减得到。
- 无损、可重复，纯 CPU，跑一遍 9 秒，可以随便重算。

---

## 2. 任务梯度冲突（`analysis/grad_cosine.py`）

### 怎么算

1. 加载「基座 + 某阶段的 LoRA」，`is_trainable=True`，开梯度检查点、把 dropout 关掉
   （保证前向确定）；
2. 每个任务取 `--max-samples` 条 val2 样本（默认 16），按 `--batch-size` 分批；
   输入是 `train_text`，**labels 里提问段设成 -100**（和 `train1.py` 的训练目标一致）；
3. 每个 batch 反向传播，把 LoRA 参数的梯度累加，最后**除以 batch 个数**
   （不除 batch_size —— `out.loss` 本身已经是"batch 内有效 token 的平均"，
   再除一次既不是样本平均，又会放大尾批）；
4. 得到该任务的梯度向量 `g_i`，算 `cos(g_i, g_j)`；重复 `--repeats` 轮取均值，
   同时报标准差。

`--loss-weight` 两种口径：

- `token`（默认）：batch 内所有有效 token 等权，**和训练目标一致**（回答长的样本被放大）；
- `sample`：每个样本先按自己的有效 token 平均、再对样本平均，**每个样本等权**。
  实测 `sample` 口径换 batch size 梯度方向不变（cos=0.999998），`token` 会漂（0.9923）。

### 目的与比较

| cos | 含义 |
|---|---|
| 明显 > 0 | 两个任务更新方向一致，一起训互相帮助 |
| ≈ 0 | 正交，互不干扰 |
| < 0 | 冲突，一起训互相拖（PCGrad 的动机） |

输出 `cosine_mean` / `cosine_std` / `cosine_by_module`（q/k/v/o 四个投影分别算）/ `grad_norm_mean`。

### 注意事项

- **梯度是局部量**：同一对任务在 base / A / A+B 阶段算出来的值会变，所以用 `--stages` 一起看演化。
- **样本量**：默认 16 条 × 3 轮。实测在已经训过的任务上（AGNEWS adapter 看 AGNEWS 题）
  梯度范数会趋近 0（loss 已经很小），两轮之间能差一个数量级 —— 这是**信噪比**问题不是 bug。
  正式跑建议 `--max-samples 32 --repeats 3`，并只看 `cosine_std` 小的格子。
- **`--stages base`**：基座参数在 LoRA 训练里是冻结的、拿不到梯度，所以脚本临时挂一个
  随机初始化的 LoRA（固定 seed，可复现；lora_B 默认是 0 会导致 dL/dA≡0，所以给了小扰动）。
  它只能当"初始冲突"参考，别和训练后的阶段直接比。要"训练前"的参考请用序列里第一个 adapter。

---

## 3. 表示漂移（`analysis/representation_drift.py`）

### 怎么算

同一批 val2 样本分别过"训练前"和"训练后"的模型（一次只加载一个，另一个的中间结果缓存到
`repr_*.pt`），取若干层 hidden state 做 **masked mean pooling**（和 `prepare_activation.py`
一致），然后算四个量：

| 量 | 定义 | 回答的问题 |
|---|---|---|
| 配对余弦 | `cos(h_before_i, h_after_i)` 逐样本 | **这一条样本**的表示变了没有？ |
| 相对 L2 | `‖h_after − h_before‖ / ‖h_before‖` | 幅度变了多少（余弦看不出的） |
| 线性 CKA | 两个 Gram 矩阵的相似度 | **整个表示空间的几何结构**变了没有？ |
| 线性映射 R² | 半拟半测（一半样本拟合线性映射、另一半算 R²） | 这个变化能不能被**一个线性变换**解释？ |

### 余弦和 CKA 不是一回事（重点）

它们回答不同问题，要合起来读。脚本会自动给判定（阈值 `--threshold`，默认 0.9）：

| 余弦 | CKA | 线性 R² | 判定 | 含义 |
|---|---|---|---|---|
| 高 | 高 | — | 基本没动 | 没怎么变 |
| 低 | **高** | — | 整体保距变换（旋转/缩放） | 单条表示都转了，但内部几何还在 |
| 低 | 低 | 高 | 线性但非保距 | 能被线性映射解释，但拉伸了几何 |
| 低 | 低 | 低 | 非线性重排（**真漂移**） | 结构本身被重写 |
| 高 | 低 | — | 幅度重排 | 方向没变，相对尺度/夹角变了 |

用合成数据（n=200, d=64，`b` 由 `a` 按已知方式变换）验证过：

```
情形              余弦      CKA     线性R²    判定
恒等            1.0000   1.0000   1.0000   基本没动
整体旋转         -0.0734   1.0000   1.0000   整体保距变换（旋转/缩放，几何保持）
线性缩放 0.5*R   -0.0457   1.0000   1.0000   整体保距变换（旋转/缩放，几何保持）
一般线性映射      0.0281   0.7407   1.0000   线性但非保距（一般线性映射）
旋转+10% 噪声    -0.0550   0.9925   0.9726   整体保距变换（旋转/缩放，几何保持）
非线性（平方）    -0.0107   0.2559  -1.6659   非线性重排（真漂移）
完全无关（随机）  -0.0044   0.2390  -1.7848   非线性重排（真漂移）
只改一半维度      0.5036   0.6376  -0.3628   非线性重排（真漂移）
```

第二行是最该注意的：**余弦 −0.07（单看会以为"完全变了"），CKA 却精确等于 1.0**，
因为 CKA 对正交变换不变（`XAAᵀXᵀ = XXᵀ`）。只看余弦会写错结论，只看 CKA 又会漏掉真重排。

实测（base → AGNEWS adapter，200 条 val2）：

```
任务      层     余弦      CKA    线性R²   判定
AGNEWS  L16   0.9631   0.8850   0.5567   方向没变但尺度/几何变了（幅度重排）
SST2    L16   0.9861   0.9994   0.7923   基本没动
AGNEWS  L32   0.8672   0.9497   0.4786   整体保距变换（旋转/缩放，几何保持）
SST2    L32   0.8556   0.9900   0.5419   整体保距变换（旋转/缩放，几何保持）
```

两点观察：① 训练在 AGNEWS 上，对"AGNEWS 题目"的表示几何改动更大（同层 CKA 比 SST2 低）；
② R² 只有 0.45~0.79 —— 漂移**不是**一个干净的整体线性变换，只能部分被线性解释。

### 注意事项

- R² 必须半拟半测：n < d 时最小二乘在训练集上必然完美拟合。样本 < 32 条时不给 R²（nan）。
- CKA 用线性 CKA，对正交变换和各向同性缩放不变。
- 默认 `--layers 8 12 16 20 24 28 32`（和 `prepare_activation.py` 对齐，可和相似度结果对照）。

---

## 4. 功能漂移（`analysis/functional_drift.py`）

### 怎么算

在同一批 val2 样本、同一个位置上比较下一个 token 的分布：

- 位置：默认只看**回答的第一个 token**（分类任务里就是要吐标签的那个决策），
  logits 取提问最后一个位置；`--positions K` 可以看前 K 个（喂 `prompt + 前 K−1 个答案 token`，
  取末尾 K 个位置的 logits）；
- `P_before`、`P_after` 都是 128k 维 softmax（存 fp16，每条样本 `K×128256×2` 字节；
  脚本启动时会预估内存）。

### 指标

| 字段 | 定义 | 用途 |
|---|---|---|
| `symmetric_kl` | `KL(before‖after) + KL(after‖before)` | 你公式里的对称 KL，主指标 |
| `kl_before_to_after` / `kl_after_to_before` | 单向 KL | 看变化的方向性 |
| `js_divergence` | Jensen-Shannon（0~ln2） | 有界，跨任务可比性更好 |
| `entropy_before/after` | 分布熵 | **塌缩会在这里露馅**（比如无脑输出 `0` 时熵骤降） |
| `top1_agreement` | 前后 top-1 token 一致率 | 最直观的"答案变了没" |
| `prob_before/after`、`rank_before/after` | 金标准首 token 的概率与排名 | 直接反映"正确答案的概率被推高还是打掉" |

### 怎么比较

- 绝对量级：128k 词表上均匀分布的熵是 `ln(128256) ≈ 11.76`；实测模型在分类题上熵约
  1.8~3.6（非常尖）。所以熵掉到 1 以下基本就是"塌缩成常数输出"。
- 跨任务比 `symmetric_kl` 要小心（不同任务的分布本身尖锐程度不同），**建议以 JS 和
  `top1_agreement` 为主做跨任务比较**，KL 用来看量级。
- 实测（base → AGNEWS，16 条）：

  ```
  任务      对称KL      JS     熵(before)  熵(after)  top1一致  gold概率(b→a)
  AGNEWS   10.1183   0.2628     1.7734     1.9914     0.5000   0.0036→0.2498
  SST2      9.3629   0.2492     3.6096     3.5621     0.5000   0.0019→0.2365
  TLDR      1.2430   0.0706     3.4432     3.6988     0.8125   0.0000→0.0000
  ```

  AGNEWS 训练把 AGNEWS 题目的金标准 token 概率推高了 70 倍（0.0036→0.2498），
  顺带也改了 SST2 的分布（对称 KL 9.36），而 TLDR 几乎没动（JS 0.07、top1 一致 81%）。

### 注意事项

- `--positions K > 1` 会成倍增加内存（每条样本 K 个 128k 维分布），默认 1。
- 提问超过 `--max-length` 被截断、取不满 K 个位置的样本会被跳过，脚本会打印跳过数量。
- 每个模型一次加载，另一个的分布缓存在内存里；加 `--save-probs` 才会落盘。

---

## 5. 怎么把它们串起来

### 四个层次的对照表

| 层次 | 脚本 | 我们手上的最强信号（TLDR 打头序列） |
|---|---|---|
| 理论方向 | `grad_cosine.py` | cos(AGNEWS, TLDR) = −0.028：梯度层面本来就不同向 |
| 实际方向 | `lora_cosine.py` | `cos(δW_TLDR, ΔW_prev) = −0.148`，且 **0/32 层为正**；任务两两 δW 余弦全为负 |
| 表示变化 | `representation_drift.py` | 深层 L32 配对余弦 0.86，CKA 仍 0.95（偏"保距变换"），R² 只有 0.48 |
| 行为变化 | `functional_drift.py` | 塌缩型任务上熵骤降、对称 KL 很大、top1 一致率低 |

### 论文里的报告顺序建议

1. 先报 **Gradient Cosine**：说明"任务之间的梯度本来就多正交/轻微冲突"（图中一个矩阵）；
2. 再报 **Actual ΔW Cosine**：说明"训练实际把方向走成了什么"，并强调它和梯度余弦的差异
   （想走的方向 ≠ 实际走的方向）；配合 `task_delta_cosine` 的任务×任务矩阵；
3. 报 **`cos(δW_k, ΔW_{k-1})` 的逐层结果**：哪几层在抵消（我们的例子里 TLDR 阶段 32/32 层全负）；
4. 报 **Representation Drift**：三种指标 + 判定标签，说明"表示是被整体旋转还是被重排"；
5. 最后报 **Functional Drift**：把行为层面的后果补上（熵、top1 一致率、金标准概率）。

### 组合判读的几个典型组合

| 组合 | 结论 |
|---|---|
| 梯度余弦 < 0 + δW 余弦 < 0 + 该任务 ROUGE/acc 掉 | 典型的**任务冲突导致的遗忘** |
| 梯度余弦 ≈ 0 + δW 余弦 ≈ 0 + 性能不掉 | 两个任务**互不干扰**（可并行训练） |
| δW 余弦 < 0 + 表示 CKA≈1 + 行为 top1 一致率高 | 权重换了方向，但表示结构没坏、行为也没变 → **安全的参数漂移** |
| δW 余弦 < 0 + 表示线性 R² 低 + 熵骤降 | 权重抵消 + 表示被重排 + 输出塌缩 → **真正的格式/能力遗忘** |

---

## 6. 常用命令

```bash
# 1) 权重漂移（CPU，9 秒）
python3 analysis/lora_cosine.py --stages AGNEWS AGNEWS+QQP AGNEWS+QQP+SST2 AGNEWS+QQP+SST2+TLDR

# 2) 梯度冲突（GPU，每阶段一次加载）
python3 analysis/grad_cosine.py --stages AGNEWS AGNEWS+QQP \
    --tasks AGNEWS QQP SST2 TLDR --max-samples 32 --repeats 3 --loss-weight token

# 3) 表示漂移（GPU，前后各一次加载）
python3 analysis/representation_drift.py --before base --after AGNEWS \
    --tasks AGNEWS SST2 TLDR --max-samples 200 --layers 8 16 24 32

# 4) 功能漂移（GPU，前后各一次加载）
python3 analysis/functional_drift.py --before base --after AGNEWS \
    --tasks AGNEWS SST2 TLDR --max-samples 200 --split val2
```

公共参数：`--tasks`、`--split`（默认 val2）、`--max-samples`、`--batch-size`（OOM 自动折半）、
`--max-length`、`--out-dir`、`--seed`。

---

## 7. 踩过的坑（写代码时要注意的）

1. **float32 累加**：13 亿项的求和会把"自己跟自己"的余弦算成 1.387；1300 万项的梯度向量
   会算成 1.003。所有内积/范数一律 float64 累加。
2. **大向量拼接**：把全部 ΔW 拼成一个向量是 5.4 GB/阶段，四个阶段直接把进程 OOM 杀掉
   （exit 137）。改成逐模块 + 迹恒等式后峰值只有几百 MB。
3. **LoRA 的缩放不变性**：不能直接比 A、B，必须先还原 ΔW。
4. **HF 的 loss 已经平均过**：`out.loss` 在 `num_items_in_batch=None` 时是 batch 内有效
   token 的均值，不要再除 batch_size。
5. **梯度在已训任务上趋近 0**：信噪比很低，要加样本量并看 `cosine_std`。
6. **CKA 对正交变换不变**：CKA≈1 不代表"没变"，要配余弦和线性 R² 一起看。
7. **R² 必须半拟半测**：n < d 时训练集上的 R² 恒等于 1，没有信息。
8. **训练时的截断方向**：`train1.py` 用默认的右侧截断，MEDDIA 这种"回答在末尾"的任务
   一旦超过 `--max-length` 会砍掉回答，所以 MEDDIA 现在贴着上限（982/1024），不要调小。
