#!/usr/bin/env python3
"""Task 4: 吞吐量/延迟/并发基准测试

实验 A: 并发度扫描 (1, 2, 4, 8)
实验 B: 工作负载对比（短答案 vs 推理型）

注意：由于 transformers 不支持真正的并发请求，
本实验使用多线程模拟并发负载。
"""

import json
import time
import csv
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 实验参数
PROMPT_SHORT = "1+1=?"
PROMPT_REASONING = "请解释量子力学的基本原理，并用通俗的语言说明它与经典物理的区别。"

CONCURRENCY_LEVELS = [1, 2, 4, 8]
N_REQUESTS = 16
MAX_TOKENS_SHORT = 32
MAX_TOKENS_LONG = 128


@dataclass
class BenchmarkRecord:
    concurrency: int
    request_id: int
    latency_s: float
    num_tokens: int
    prompt_type: str


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


def generate_once(model, tokenizer, prompt: str, max_tokens: int, request_id: int,
                  concurrency: int, prompt_type: str) -> BenchmarkRecord:
    """单次生成请求"""
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    dt = time.perf_counter() - t0
    output_len = outputs.shape[1] - input_len

    return BenchmarkRecord(
        concurrency=concurrency,
        request_id=request_id,
        latency_s=dt,
        num_tokens=output_len,
        prompt_type=prompt_type,
    )


def run_concurrency_benchmark(model, tokenizer, concurrency: int,
                              prompt: str, max_tokens: int,
                              prompt_type: str, n_requests: int) -> List[BenchmarkRecord]:
    """运行特定并发度的基准测试"""
    records = []

    def run_request(i):
        return generate_once(model, tokenizer, prompt, max_tokens, i,
                            concurrency, prompt_type)

    wall_t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(run_request, i) for i in range(n_requests)]
        for fut in as_completed(futures):
            records.append(fut.result())

    wall_time = time.perf_counter() - wall_t0

    return records, wall_time


def analyze_benchmark(records: List[BenchmarkRecord], wall_time: float) -> Dict[str, Any]:
    """分析基准测试结果"""
    latencies = [r.latency_s for r in records]
    tokens = [r.num_tokens for r in records]

    latencies_sorted = sorted(latencies)

    def percentile(p):
        if not latencies_sorted:
            return 0
        k = int(round((len(latencies_sorted) - 1) * p / 100))
        return latencies_sorted[k]

    total_tokens = sum(tokens)

    return {
        "wall_time_s": wall_time,
        "p50_latency_s": percentile(50),
        "p95_latency_s": percentile(95),
        "p99_latency_s": percentile(99),
        "mean_latency_s": statistics.mean(latencies),
        "std_latency_s": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / wall_time if wall_time > 0 else 0,
        "num_requests": len(records),
    }


def run_full_benchmark(model, tokenizer):
    """运行完整的基准测试"""
    all_results = []

    print("\n" + "="*60)
    print("实验 A: 并发度扫描")
    print("="*60)

    concurrency_results = {}

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n并发度：{concurrency}")
        records, wall_time = run_concurrency_benchmark(
            model, tokenizer,
            concurrency=concurrency,
            prompt=PROMPT_SHORT,
            max_tokens=MAX_TOKENS_SHORT,
            prompt_type="short",
            n_requests=N_REQUESTS,
        )

        analysis = analyze_benchmark(records, wall_time)
        concurrency_results[concurrency] = analysis

        print(f"  Wall 时间：{wall_time:.2f}s")
        print(f"  P50 延迟：{analysis['p50_latency_s']:.3f}s")
        print(f"  P95 延迟：{analysis['p95_latency_s']:.3f}s")
        print(f"  吞吐量：{analysis['tokens_per_sec']:.1f} tokens/s")

        all_results.extend(records)

    print("\n" + "="*60)
    print("实验 B: 工作负载对比")
    print("="*60)

    workload_results = {}

    for prompt_type, prompt, max_tokens in [
        ("short", PROMPT_SHORT, MAX_TOKENS_SHORT),
        ("reasoning", PROMPT_REASONING, MAX_TOKENS_LONG),
    ]:
        print(f"\n负载类型：{prompt_type}")
        print(f"  Prompt: {prompt[:50]}...")

        records, wall_time = run_concurrency_benchmark(
            model, tokenizer,
            concurrency=4,
            prompt=prompt,
            max_tokens=max_tokens,
            prompt_type=prompt_type,
            n_requests=N_REQUESTS,
        )

        analysis = analyze_benchmark(records, wall_time)
        workload_results[prompt_type] = analysis

        print(f"  Wall 时间：{wall_time:.2f}s")
        print(f"  P50 延迟：{analysis['p50_latency_s']:.3f}s")
        print(f"  吞吐量：{analysis['tokens_per_sec']:.1f} tokens/s")

        all_results.extend(records)

    return all_results, concurrency_results, workload_results


def save_results(records: List[BenchmarkRecord],
                 concurrency_results: Dict, workload_results: Dict):
    """保存结果"""
    # CSV
    csv_path = RESULTS_DIR / "concurrency_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BenchmarkRecord.__dataclass_fields__.keys())
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"\n原始记录已保存到：{csv_path}")

    # JSON
    json_path = RESULTS_DIR / "benchmark_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "concurrency": concurrency_results,
            "workload": workload_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"分析结果已保存到：{json_path}")


def print_summary(concurrency_results: Dict, workload_results: Dict):
    """打印摘要"""
    print("\n" + "="*70)
    print("基准测试结果摘要")
    print("="*70)

    print("\n并发度对性能的影响:")
    print(f"{'并发度':<10} {'Wall 时间 (s)':<15} {'P50 延迟 (s)':<15} {'P95 延迟 (s)':<15} {'吞吐量 (tok/s)':<15}")
    print("-" * 70)
    for conc, metrics in concurrency_results.items():
        print(f"{conc:<10} {metrics['wall_time_s']:<15.2f} {metrics['p50_latency_s']:<15.3f} "
              f"{metrics['p95_latency_s']:<15.3f} {metrics['tokens_per_sec']:<15.1f}")

    print("\n工作负载对比 (并发度=4):")
    for wtype, metrics in workload_results.items():
        print(f"\n{wtype}:")
        print(f"  P50 延迟：{metrics['p50_latency_s']:.3f}s")
        print(f"  P95 延迟：{metrics['p95_latency_s']:.3f}s")
        print(f"  吞吐量：{metrics['tokens_per_sec']:.1f} tokens/s")

    print("\n" + "="*70)


def main():
    print("="*70)
    print("Task 4: 吞吐量/延迟/并发基准测试")
    print("="*70)
    print(f"请求总数：{N_REQUESTS}")
    print(f"并发级别：{CONCURRENCY_LEVELS}")

    model, tokenizer = load_model_and_tokenizer()

    records, concurrency_results, workload_results = run_full_benchmark(model, tokenizer)

    save_results(records, concurrency_results, workload_results)

    print_summary(concurrency_results, workload_results)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n实验完成！")


if __name__ == "__main__":
    main()
