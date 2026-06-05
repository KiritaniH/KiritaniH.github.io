#!/usr/bin/env python3
"""Task 2: 解码策略对比实验

比较不同解码配置（temperature, top_p, top_k）对输出稳定性和多样性的影响

实验设计：
1. 稳定型任务（数学计算、事实问答）
2. 开放型任务（创意写作、标语）
3. 每种配置重复 3 次
4. 记录唯一输出数、平均长度、延迟
"""

import json
import time
import csv
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 解码配置
DECODE_CONFIGS = [
    {"name": "A_greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0, "note": "greedy"},
    {"name": "B_low_temp", "temperature": 0.2, "top_p": 1.0, "top_k": 20, "note": "conservative"},
    {"name": "C_mid_temp", "temperature": 0.7, "top_p": 1.0, "top_k": 20, "note": "medium randomness"},
    {"name": "D_high_temp", "temperature": 1.0, "top_p": 0.9, "top_k": 50, "note": "open-ended"},
]

# 测试 prompts
PROMPTS = {
    "stable_math": "23 + 67 = ",
    "stable_fact": "中国的首都是",
    "open_creative": "写一个关于 AI 的简短标语（10 字以内）",
    "open_story": "写一个科幻故事的开头（50 字以内）",
}

REPEAT_TIMES = 3  # 每种配置重复次数
MAX_NEW_TOKENS = 64


@dataclass
class Record:
    prompt_name: str
    prompt: str
    config_name: str
    temperature: float
    top_p: float
    top_k: int
    run_id: int
    text: str
    latency_s: float
    num_tokens: int
    unique_hash: str


def load_model_and_tokenizer():
    """加载模型和分词器"""
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print("模型加载完成")
    return model, tokenizer


def generate_with_config(model, tokenizer, prompt: str, config: Dict, max_tokens: int = MAX_NEW_TOKENS) -> Dict:
    """使用指定配置生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    t0 = time.perf_counter()

    with torch.no_grad():
        if config["temperature"] == 0.0:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=config["temperature"],
                top_p=config["top_p"] if config["top_p"] < 1.0 else None,
                top_k=config["top_k"] if config["top_k"] > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

    dt = time.perf_counter() - t0
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    input_len = inputs["input_ids"].shape[1]
    output_len = outputs.shape[1] - input_len

    return {
        "text": generated_text,
        "latency_s": dt,
        "num_tokens": output_len,
    }


def text_hash(text: str) -> str:
    """生成文本的简单哈希用于比较唯一性"""
    # 归一化：去除首尾空格，转小写
    normalized = text.strip().lower()
    return f"{len(normalized)}_{hash(normalized) % 100000}"


def run_experiment(model, tokenizer) -> List[Record]:
    """运行完整的解码对比实验"""
    records = []

    for prompt_name, prompt in PROMPTS.items():
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt_name}")
        print(f"内容：{prompt}")
        print(f"{'='*50}")

        for config in DECODE_CONFIGS:
            print(f"\n  配置：{config['name']} (temp={config['temperature']}, top_p={config['top_p']}, top_k={config['top_k']})")

            generated_texts = []

            for run_id in range(REPEAT_TIMES):
                result = generate_with_config(model, tokenizer, prompt, config)
                text_only = result["text"][len(prompt):].strip()  # 只保留生成的部分

                record = Record(
                    prompt_name=prompt_name,
                    prompt=prompt,
                    config_name=config["name"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                    run_id=run_id,
                    text=text_only,
                    latency_s=result["latency_s"],
                    num_tokens=result["num_tokens"],
                    unique_hash=text_hash(text_only),
                )
                records.append(record)
                generated_texts.append(text_only)

                print(f"    Run {run_id}: {text_only[:50]}..." if len(text_only) > 50 else f"    Run {run_id}: {text_only}")

            # 计算唯一性
            unique_count = len(set(text_hash(t) for t in generated_texts))
            print(f"    唯一输出数：{unique_count}/{REPEAT_TIMES}")

    return records


def analyze_results(records: List[Record]) -> Dict[str, Any]:
    """分析实验结果"""
    analysis = {}

    # 按配置和 prompt 分组
    grouped = {}
    for r in records:
        key = f"{r.prompt_name}_{r.config_name}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for key, recs in grouped.items():
        unique_hashes = set(r.unique_hash for r in recs)
        latencies = [r.latency_s for r in recs]
        token_counts = [r.num_tokens for r in recs]

        analysis[key] = {
            "unique_outputs": len(unique_hashes),
            "avg_latency_s": statistics.mean(latencies),
            "p50_latency_s": sorted(latencies)[len(latencies)//2],
            "avg_tokens": statistics.mean(token_counts),
            "stability_score": 1.0 - (len(unique_hashes) - 1) / (REPEAT_TIMES - 1) if REPEAT_TIMES > 1 else 1.0,
        }

    return analysis


def save_results(records: List[Record], analysis: Dict[str, Any]):
    """保存结果到 CSV"""
    # 保存原始记录
    csv_path = RESULTS_DIR / "decoding_compare.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=Record.__dataclass_fields__.keys())
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"\n原始记录已保存到：{csv_path}")

    # 保存分析结果
    json_path = RESULTS_DIR / "decoding_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"分析结果已保存到：{json_path}")


def print_summary(analysis: Dict[str, Any]):
    """打印摘要"""
    print("\n" + "="*70)
    print("实验结果摘要")
    print("="*70)

    print("\n稳定性评分 (1.0=完全稳定，0.0=完全不同):")
    for key, metrics in analysis.items():
        print(f"  {key}: {metrics['stability_score']:.2f} (唯一输出：{metrics['unique_outputs']}/{REPEAT_TIMES})")

    print("\n平均延迟:")
    for key, metrics in analysis.items():
        print(f"  {key}: {metrics['avg_latency_s']:.3f}s")

    print("\n" + "="*70)


def main():
    print("="*70)
    print("Task 2: 解码策略对比实验")
    print("="*70)
    print(f"模型：{MODEL_PATH}")
    print(f"配置数：{len(DECODE_CONFIGS)}")
    print(f"Prompt 数：{len(PROMPTS)}")
    print(f"重复次数：{REPEAT_TIMES}")
    print(f"总实验次数：{len(DECODE_CONFIGS) * len(PROMPTS) * REPEAT_TIMES}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer()

    # 运行实验
    records = run_experiment(model, tokenizer)

    # 分析结果
    analysis = analyze_results(records)

    # 保存结果
    save_results(records, analysis)

    # 打印摘要
    print_summary(analysis)

    # 清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n实验完成！请查看 results/ 目录中的结果文件")


if __name__ == "__main__":
    main()
