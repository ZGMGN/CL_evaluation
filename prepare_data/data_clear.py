import glob
import json
import os
import re

# 用脚本自身位置定位路径，避免依赖当前工作目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "datas")
OUTPUT_DIR = os.path.join(BASE_DIR, "datas_clean")

# 一次删掉整个日期块（含尾随空行），避免残留多余换行
DATE_BLOCK = re.compile(
    r"Cutting Knowledge Date:.*?Today Date:[^\n]*\n\n",
    re.DOTALL,
)


def remove_llama_date(text):
    """删除 Llama-3.1 模板默认注入的日期两行。

    处理前：
        <|start_header_id|>system<|end_header_id|>\n\n
        Cutting Knowledge Date: December 2023\n
        Today Date: 26 Jul 2024\n\n
        <|eot_id|>
    处理后：
        <|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>
    """
    if text is None:
        return text
    return DATE_BLOCK.sub("", text)


def clean_file(input_path, output_path):
    """逐行清洗 JSONL：只改 activation_text / train_text，其余字段原样保留。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = changed = broken_prefix = 0

    with open(input_path, encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            example = json.loads(line)
            total += 1

            row_changed = False
            for key in ("activation_text", "train_text"):
                if key not in example:
                    continue
                cleaned = remove_llama_date(example[key])
                if cleaned != example[key]:
                    row_changed = True
                example[key] = cleaned

            if row_changed:
                changed += 1

            # 清洗后仍应保持 train_text 以 activation_text 为前缀
            if (
                "activation_text" in example
                and "train_text" in example
                and not example["train_text"].startswith(example["activation_text"])
            ):
                broken_prefix += 1

            # 与原文件保持一致：紧凑分隔符 + 不转义非 ASCII
            fout.write(
                json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    return total, changed, broken_prefix


def main():
    names = sorted(
        d
        for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    )

    grand_total = grand_changed = grand_broken = 0

    for name in names:

        # 下钻一层：datas/<数据集>/<切分>.jsonl
        paths = sorted(glob.glob(os.path.join(INPUT_DIR, name, "*.jsonl")))
        if not paths:
            print(f"[{name}] 跳过（没有 .jsonl 文件）")
            continue

        print(f"[{name}]")

        for input_path in paths:
            split = os.path.splitext(os.path.basename(input_path))[0]

            # 保留目录层级，避免不同数据集的同名切分互相覆盖
            output_path = os.path.join(OUTPUT_DIR, name, f"{split}.jsonl")

            total, changed, broken = clean_file(input_path, output_path)
            grand_total += total
            grand_changed += changed
            grand_broken += broken

            warn = f"  ← 前缀异常 {broken} 行" if broken else ""
            print(f"  {split:6s} {total:6d} 行，改动 {changed:6d} 行{warn}")

    print()
    print("=" * 50)
    print(f"合计 {grand_total} 行，改动 {grand_changed} 行")
    print(f"前缀关系异常：{grand_broken} 行")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
