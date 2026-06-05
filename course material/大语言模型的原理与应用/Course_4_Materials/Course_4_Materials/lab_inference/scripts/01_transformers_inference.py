#!/usr/bin/env python3
"""Task 1: 使用 transformers 完成首次推理请求

功能：
1. 加载本地模型
2. 发送推理请求
3. 打印延迟、usage 和文本
4. 保存结果到 results/task1_sanity_check.json

注意：由于 vLLM 在 Windows 上无法运行，我们使用 transformers 完成核心实验
"""

import json
import time
from pathlib import Path
import torch

# 配置
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"

# 输出目录
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_model_and_tokenizer():
    """加载模型和分词器"""
    print("正在加载模型和分词器...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        revision="master",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        revision="master",
    )

    print(f"模型加载完成")
    print(f"  - 设备：{'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    return model, tokenizer


def generate_text(model, tokenizer, prompt, temperature=0.0, max_new_tokens=128):
    """生成文本"""
    print(f"\n正在生成文本...")

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    t0 = time.perf_counter()

    # 生成
    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

    dt = time.perf_counter() - t0

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 计算新生成的 token 数
    input_len = inputs["input_ids"].shape[1]
    output_len = outputs.shape[1] - input_len

    return {
        "prompt": prompt,
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": output_len,
            "total_tokens": input_len + output_len,
        },
        "latency_s": dt,
    }


def main():
    print("=" * 50)
    print("Task 1: Transformers 首次推理请求")
    print("=" * 50)

    # 步骤 1: 加载模型
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"\n模型加载失败：{e}")
        print(f"\n请检查模型路径：{MODEL_PATH}")
        return

    # 步骤 2: 发送推理请求
    test_prompt = "你好，请用一句话介绍你自己。"
    print(f"\n测试 prompt: {test_prompt}")

    try:
        result = generate_text(model, tokenizer, test_prompt)
    except Exception as e:
        print(f"\n推理失败：{e}")
        import traceback
        traceback.print_exc()
        return

    # 步骤 3: 打印结果
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    print(f"生成文本：{result['text']}")
    print(f"\nUsage: {json.dumps(result['usage'], indent=2, ensure_ascii=False)}")
    print(f"延迟：{result['latency_s']:.3f} 秒")
    tokens_per_sec = result['usage']['completion_tokens'] / result['latency_s']
    print(f"生成速度：{tokens_per_sec:.2f} tokens/s")

    # 步骤 4: 保存结果
    output_path = RESULTS_DIR / "task1_sanity_check.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到：{output_path}")
    print("=" * 50)

    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
