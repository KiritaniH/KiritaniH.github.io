#!/usr/bin/env python3
"""Task 3: KV Cache / 共享前缀实验

实验设计：
1. 构造一个长共享前缀（如课程描述、系统 prompt）
2. 使用相同前缀 + 不同问题进行多次请求
3. 比较共享前缀请求 vs 不同前缀请求的延迟

注意：transformers 的 KV Cache 是自动管理的，
本实验通过测量连续请求的延迟变化来间接观察缓存效应。
"""

import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 长共享前缀（约 500-1000 tokens）
SHARED_PREFIX = """
你是一个教学助手。以下是课程材料的详细内容：

第一章：人工智能基础

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支，
它致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括视觉感知、
语音识别、决策和语言翻译等。

1.1 机器学习的定义

机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习，
而无需显式编程。通过分析大量数据，机器学习算法可以识别模式、
做出预测并改进决策。

主要类型包括：
- 监督学习：从标记的训练数据中学习，用于分类和回归任务
- 无监督学习：从无标记数据中发现隐藏模式，用于聚类和降维
- 强化学习：通过与环境交互学习最优策略，用于游戏和机器人控制

1.2 深度学习简介

深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次表示。
它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。

关键概念：
- 神经网络：由多层相互连接的节点（神经元）组成的计算模型
- 反向传播：通过计算损失函数梯度来更新网络权重的算法
- 卷积神经网络（CNN）：专门用于处理网格状数据（如图像）的网络结构
- 循环神经网络（RNN）：能够处理序列数据的网络，具有记忆能力

1.3 自然语言处理

自然语言处理（NLP）是 AI 的一个重要应用领域，涉及计算机与人类语言之间的交互。
主要任务包括文本分类、情感分析、机器翻译和问答系统等。

近年来，基于 Transformer 架构的大型语言模型（LLM）在 NLP 领域取得了巨大成功，
能够生成流畅的文本、回答问题和执行各种语言任务。
"""

# 问题列表
QUESTIONS = [
    "用一句话总结这门课程的主题。",
    "这门课程适合什么样的学习者？",
    "如果要设计一个实践作业，第一步应该是什么？",
    "机器学习有哪几种主要类型？",
    "深度学习和传统机器学习有什么区别？",
]


@dataclass
class KVCacheRecord:
    test_type: str  # "shared_prefix" or "different_prefix"
    question_id: int
    question: str
    latency_s: float
    num_tokens: int
    tokens_per_sec: float


def load_model_and_tokenizer():
    """加载模型"""
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


def generate_with_prompt(model, tokenizer, full_prompt: str, max_tokens: int = 64) -> Dict:
    """生成文本并返回指标"""
    inputs = tokenizer(full_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # greedy 保证一致性
            pad_token_id=tokenizer.eos_token_id,
        )

    dt = time.perf_counter() - t0
    output_len = outputs.shape[1] - input_len

    return {
        "latency_s": dt,
        "num_tokens": output_len,
        "tokens_per_sec": output_len / dt if dt > 0 else 0,
    }


def run_kv_cache_experiment(model, tokenizer) -> List[KVCacheRecord]:
    """运行 KV Cache 对比实验"""
    records = []

    print("\n" + "="*60)
    print("实验 A: 共享前缀请求")
    print("="*60)
    print(f"前缀长度：{len(SHARED_PREFIX)} 字符")

    # 共享前缀 + 不同问题（模拟缓存友好场景）
    for i, question in enumerate(QUESTIONS):
        full_prompt = SHARED_PREFIX + "\n\n问题：" + question

        result = generate_with_prompt(model, tokenizer, full_prompt)

        record = KVCacheRecord(
            test_type="shared_prefix",
            question_id=i,
            question=question,
            latency_s=result["latency_s"],
            num_tokens=result["num_tokens"],
            tokens_per_sec=result["tokens_per_sec"],
        )
        records.append(record)

        print(f"  问题{i+1}: 延迟={result['latency_s']:.3f}s, "
              f"{result['tokens_per_sec']:.1f} tokens/s")

        # 注意：transformers 不保留 KV Cache 跨请求，
        # 所以这里只是演示，真正受益需要 vLLM 或手动管理

    print("\n" + "="*60)
    print("实验 B: 不同前缀请求")
    print("="*60)

    # 不同前缀 + 不同问题（无缓存收益）
    different_prefixes = [
        "你是一个数学助手。",
        "你是一个编程助手。",
        "你是一个写作助手。",
        "你是一个科学顾问。",
        "你是一个历史专家。",
    ]

    for i, (prefix, question) in enumerate(zip(different_prefixes, QUESTIONS)):
        full_prompt = prefix + "\n\n问题：" + question

        result = generate_with_prompt(model, tokenizer, full_prompt)

        record = KVCacheRecord(
            test_type="different_prefix",
            question_id=i,
            question=question,
            latency_s=result["latency_s"],
            num_tokens=result["num_tokens"],
            tokens_per_sec=result["tokens_per_sec"],
        )
        records.append(record)

        print(f"  问题{i+1}: 延迟={result['latency_s']:.3f}s, "
              f"{result['tokens_per_sec']:.1f} tokens/s")

    return records


def analyze_kv_cache(records: List[KVCacheRecord]) -> Dict:
    """分析 KV Cache 实验结果"""
    shared = [r for r in records if r.test_type == "shared_prefix"]
    different = [r for r in records if r.test_type == "different_prefix"]

    return {
        "shared_prefix": {
            "avg_latency_s": sum(r.latency_s for r in shared) / len(shared),
            "avg_tokens_per_sec": sum(r.tokens_per_sec for r in shared) / len(shared),
            "total_requests": len(shared),
        },
        "different_prefix": {
            "avg_latency_s": sum(r.latency_s for r in different) / len(different),
            "avg_tokens_per_sec": sum(r.tokens_per_sec for r in different) / len(different),
            "total_requests": len(different),
        },
    }


def save_results(records: List[KVCacheRecord], analysis: Dict):
    """保存结果"""
    # CSV
    csv_path = RESULTS_DIR / "prefix_cache_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KVCacheRecord.__dataclass_fields__.keys())
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"\n原始记录已保存到：{csv_path}")

    # JSON
    json_path = RESULTS_DIR / "prefix_cache_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"分析结果已保存到：{json_path}")


def print_summary(analysis: Dict):
    """打印摘要"""
    print("\n" + "="*60)
    print("KV Cache 实验结果摘要")
    print("="*60)

    shared = analysis["shared_prefix"]
    diff = analysis["different_prefix"]

    print(f"\n共享前缀组:")
    print(f"  平均延迟：{shared['avg_latency_s']:.3f}s")
    print(f"  平均速度：{shared['avg_tokens_per_sec']:.1f} tokens/s")

    print(f"\n不同前缀组:")
    print(f"  平均延迟：{diff['avg_latency_s']:.3f}s")
    print(f"  平均速度：{diff['avg_tokens_per_sec']:.1f} tokens/s")

    print("\n注意:")
    print("  transformers 的 KV Cache 在单次生成内自动管理，")
    print("  但跨请求不保留缓存。要观察真正的缓存收益，")
    print("  需要使用 vLLM 等服务框架或手动实现 KV Cache。")
    print("="*60)


def main():
    print("="*60)
    print("Task 3: KV Cache / 共享前缀实验")
    print("="*60)

    model, tokenizer = load_model_and_tokenizer()

    records = run_kv_cache_experiment(model, tokenizer)

    analysis = analyze_kv_cache(records)

    save_results(records, analysis)

    print_summary(analysis)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n实验完成！")


if __name__ == "__main__":
    main()
