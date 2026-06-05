#!/usr/bin/env python3
"""Task 2/5: 手写解码循环

目的：真正理解自回归生成的内部机制
- logits → probabilities → sampled tokens
- 为什么需要逐个 token 生成
- 为什么没有缓存的重复计算代价很高

对比：
1. manual_decode (手写循环)
2. transformers.generate() (内置生成)
3. vLLM API (服务化，跳过如果 vLLM 不可用)
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MAX_NEW_TOKENS = 50


@dataclass
class DecodeConfig:
    max_new_tokens: int = MAX_NEW_TOKENS
    strategy: str = "greedy"  # "greedy" or "sample"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    stop_on_eos: bool = True


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-K 过滤：只保留概率最高的 k 个 token"""
    if top_k <= 0:
        return logits
    values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
    cutoff = values[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-P (Nucleus) 过滤：保留累积概率达到 p 的最小 token 集合"""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建掩码：累积概率超过阈值的位置设为 -inf
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    filtered_sorted = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # 恢复原始顺序
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted)

    return filtered_logits


def select_next_token(logits: torch.Tensor, cfg: DecodeConfig) -> torch.Tensor:
    """根据配置选择下一个 token"""
    if cfg.strategy == "greedy":
        # Greedy: 直接选概率最高的
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Sampling: 应用温度和过滤
    logits = logits / cfg.temperature
    logits = top_k_filter(logits, cfg.top_k)
    logits = top_p_filter(logits, cfg.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def manual_decode(model, tokenizer, prompt: str, cfg: DecodeConfig,
                  device: str = "cuda") -> dict:
    """
    手写解码循环 - 展示自回归生成的内部机制

    核心循环：
    1. 将当前序列输入模型
    2. 获取最后一个位置的 logits
    3. 根据策略选择下一个 token
    4. 将新 token 附加到序列
    5. 重复直到达到最大长度或 EOS
    """
    # 编码输入
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = input_ids.clone()
    new_ids = []

    t0 = time.perf_counter()

    with torch.no_grad():
        for step in range(cfg.max_new_tokens):
            # 关键：每次都将整个序列输入模型
            # 这就是为什么需要 KV Cache - 否则历史 token 被重复计算
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]  # 只取最后一个位置的 logits

            next_token = select_next_token(logits, cfg)
            token_id = next_token.item()
            new_ids.append(token_id)

            # 将新 token 附加到序列
            generated = torch.cat([generated, next_token], dim=-1)

            if cfg.stop_on_eos and token_id == tokenizer.eos_token_id:
                break

    dt = time.perf_counter() - t0

    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "num_new_tokens": len(new_ids),
        "latency_s": dt,
        "tokens_per_sec": len(new_ids) / dt if dt > 0 else 0,
        "method": "manual_decode",
    }


def transformers_generate(model, tokenizer, prompt: str, cfg: DecodeConfig,
                          device: str = "cuda") -> dict:
    """使用 transformers 内置 generate 方法"""
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    t0 = time.perf_counter()

    with torch.no_grad():
        if cfg.strategy == "greedy":
            outputs = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            outputs = model.generate(
                input_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_k=cfg.top_k if cfg.top_k > 0 else None,
                top_p=cfg.top_p if cfg.top_p < 1.0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

    dt = time.perf_counter() - t0

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_len = input_ids.shape[1]
    new_tokens = outputs[0, input_len:].tolist()
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "full_text": full_text,
        "generated_text": generated_text,
        "num_new_tokens": len(new_tokens),
        "latency_s": dt,
        "tokens_per_sec": len(new_tokens) / dt if dt > 0 else 0,
        "method": "transformers_generate",
    }


def compare_methods(model, tokenizer, prompt: str, cfg: DecodeConfig):
    """比较三种方法"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nPrompt: {prompt}")
    print(f"配置：strategy={cfg.strategy}, temp={cfg.temperature}")
    print("="*60)

    results = {}

    # 方法 1: 手写解码
    print("\n[1] Manual Decode (手写循环)...")
    result1 = manual_decode(model, tokenizer, prompt, cfg, device)
    results["manual"] = result1
    print(f"  生成：{result1['generated_text'][:50]}...")
    print(f"  速度：{result1['tokens_per_sec']:.2f} tokens/s")
    print(f"  延迟：{result1['latency_s']:.3f}s")

    # 方法 2: transformers generate
    print("\n[2] Transformers Generate (内置)...")
    result2 = transformers_generate(model, tokenizer, prompt, cfg, device)
    results["transformers"] = result2
    print(f"  生成：{result2['generated_text'][:50]}...")
    print(f"  速度：{result2['tokens_per_sec']:.2f} tokens/s")
    print(f"  延迟：{result2['latency_s']:.3f}s")

    # 计算速度比
    speedup = result2['tokens_per_sec'] / result1['tokens_per_sec'] if result1['tokens_per_sec'] > 0 else 0
    print(f"\n  速度提升：{speedup:.2f}x (transformers 更快)")

    return results


def main():
    print("="*60)
    print("Task 5: 三种生成路径对比")
    print("="*60)

    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print("模型加载完成")

    # 测试 prompt
    test_prompt = "人工智能是"
    cfg = DecodeConfig(strategy="greedy", max_new_tokens=30)

    results = compare_methods(model, tokenizer, test_prompt, cfg)

    # 保存结果
    output_path = RESULTS_DIR / "generation_methods_compare.json"

    # 移除大字段
    save_results = {}
    for method, data in results.items():
        save_results[method] = {
            k: v for k, v in data.items() if k != "full_text"
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到：{output_path}")

    # 清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("关键观察:")
    print("  1. manual_decode 最透明，便于教学理解")
    print("  2. transformers.generate 经过优化，速度更快")
    print("  3. vLLM 在服务场景提供最佳吞吐量（需要服务支持）")
    print("="*60)


if __name__ == "__main__":
    main()
