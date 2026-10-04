#!/usr/bin/env bash
# 一条命令跑完某个任务序列的全部指标分析。
#
# 用法:
#   bash analysis/run_sequence.sh                                  # 默认 AGNEWS QQP SST2 TLDR MEDDIA
#   bash analysis/run_sequence.sh AGNEWS QQP SST2 TLDR              # 指定序列
#   SKIP_EVAL=1 bash analysis/run_sequence.sh ...                   # 跳过评测矩阵那一步
#   SKIP_GRAD=1 SKIP_REPR=1 SKIP_FUNC=1 ...                         # 只跑其中某几步
#
# 结果写到 analysis_results/<序列>/ 下，四个子目录分别是 lora / grad / repr / func，
# 完整日志在该目录的 run.log。
#
# 记忆点：
#   - 只有 lora_cosine 是纯 CPU（十几秒），其余三步要用 GPU；
#   - repr / func 逐段跑同一个 out-dir 并带 --reuse，所以每个阶段只会前向一次；
#   - 之前算过的评测结果会被 evaluate1.py 自动复用，所以第一步通常只要几秒。

set -u

SEQ=("$@")
if [ ${#SEQ[@]} -eq 0 ]; then
    SEQ=(AGNEWS QQP SST2 TLDR MEDDIA)
fi

cd "$(dirname "$0")/.." || exit 1

TAG=$(IFS=+; echo "${SEQ[*]}")
OUT="analysis_results/$TAG"
mkdir -p "$OUT"
LOG="$OUT/run.log"

# 前缀链：AGNEWS / AGNEWS+QQP / AGNEWS+QQP+SST2 / ...
STAGES=()
prefix=""
for task in "${SEQ[@]}"; do
    if [ -z "$prefix" ]; then prefix="$task"; else prefix="$prefix+$task"; fi
    STAGES+=("$prefix")
done

echo "任务序列 : ${SEQ[*]}"
echo "训练阶段 : ${STAGES[*]}"
echo "输出目录 : $OUT"
echo "日志     : $LOG"
echo "开始时间 : $(date '+%F %T')"

run_step () {
    echo
    echo "########## $* ##########"
    "$@"
}

{
    # ----------------------------------------------------------
    # 0) 评测矩阵：已算过的单元格会自动复用，通常几秒
    # ----------------------------------------------------------
    if [ -z "${SKIP_EVAL:-}" ]; then
        run_step python3 -u evaluate1.py --tasks "${SEQ[@]}" --include-base
    fi

    # ----------------------------------------------------------
    # 1) 权重漂移（CPU，十几秒）
    # ----------------------------------------------------------
    run_step python3 -u analysis/lora_cosine.py \
        --stages "${STAGES[@]}" --out-dir "$OUT/lora"

    # ----------------------------------------------------------
    # 2) 梯度冲突（GPU；这是最慢的一步，可以调小 --max-samples）
    # ----------------------------------------------------------
    if [ -z "${SKIP_GRAD:-}" ]; then
        run_step python3 -u analysis/grad_cosine.py \
            --stages base "${STAGES[@]}" --tasks "${SEQ[@]}" \
            --max-samples "${GRAD_SAMPLES:-32}" --repeats "${GRAD_REPEATS:-3}" \
            --batch-size 2 --out-dir "$OUT/grad"
    fi

    # ----------------------------------------------------------
    # 3) 表示漂移：逐段跑，共用 out-dir + --reuse，每阶段只前向一次
    # ----------------------------------------------------------
    if [ -z "${SKIP_REPR:-}" ]; then
        prev=base
        for stage in "${STAGES[@]}"; do
            run_step python3 -u analysis/representation_drift.py \
                --before "$prev" --after "$stage" --tasks "${SEQ[@]}" \
                --max-samples "${DRIFT_SAMPLES:-200}" --batch-size 8 \
                --out-dir "$OUT/repr" --reuse
            prev="$stage"
        done
    fi

    # ----------------------------------------------------------
    # 4) 功能漂移：同样逐段 + 缓存（--save-probs 才能被下次 --reuse 命中）
    # ----------------------------------------------------------
    if [ -z "${SKIP_FUNC:-}" ]; then
        prev=base
        for stage in "${STAGES[@]}"; do
            run_step python3 -u analysis/functional_drift.py \
                --before "$prev" --after "$stage" --tasks "${SEQ[@]}" \
                --max-samples "${DRIFT_SAMPLES:-200}" --batch-size 8 \
                --positions 1 --split val2 --out-dir "$OUT/func" \
                --reuse --save-probs
            prev="$stage"
        done
    fi

    echo
    echo "全部完成：$OUT"
    echo "结束时间：$(date '+%F %T')"
} 2>&1 | tee "$LOG"
