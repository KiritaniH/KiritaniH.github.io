#!/usr/bin/env python3
"""Task 1: vLLM 客户端 - 完成首次推理请求

功能：
1. 请求 /v1/models 获取模型列表
2. 自动使用第一个模型 ID
3. 发送 chat/completions 请求
4. 打印延迟、usage 和文本
5. 保存结果到 results/task1_sanity_check.json
"""

import json
import time
from pathlib import Path
import requests

# 配置
BASE_URL = "http://localhost:8000"
API_KEY = "token-abc123"
MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 输出目录
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fetch_models():
    """获取可用模型列表"""
    print("正在获取模型列表...")
    r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    print(f"返回数据：{json.dumps(data, indent=2, ensure_ascii=False)}")

    models = data.get("data", [])
    if not models:
        raise ValueError("没有可用的模型")

    # 返回第一个模型 ID
    model_id = models[0].get("id")
    print(f"\n使用模型：{model_id}")
    return model_id


def chat_once(prompt, model, temperature=0.0, max_tokens=128):
    """发送单次聊天请求"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(f"\n正在发送请求...")
    t0 = time.perf_counter()

    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()

    dt = time.perf_counter() - t0
    data = r.json()

    # 提取结果
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "prompt": prompt,
        "text": content,
        "usage": usage,
        "latency_s": dt,
        "raw": data,
    }


def main():
    print("=" * 50)
    print("Task 1: vLLM 首次推理请求")
    print("=" * 50)

    # 步骤 1: 获取模型列表
    try:
        model_id = fetch_models()
    except requests.exceptions.ConnectionError:
        print("\n错误：无法连接到 vLLM 服务")
        print(f"请确保服务已在 {BASE_URL} 启动")
        print("\n启动命令示例:")
        print("  vllm serve /path/to/model --host 0.0.0.0 --port 8000 --api-key token-abc123")
        return
    except Exception as e:
        print(f"\n获取模型失败：{e}")
        return

    # 步骤 2: 发送推理请求
    test_prompt = "你好，请用一句话介绍你自己。"
    print(f"\n测试 prompt: {test_prompt}")

    try:
        result = chat_once(test_prompt, model_id)
    except Exception as e:
        print(f"\n推理请求失败：{e}")
        return

    # 步骤 3: 打印结果
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    print(f"生成文本：{result['text']}")
    print(f"\nUsage: {json.dumps(result['usage'], indent=2, ensure_ascii=False)}")
    print(f"延迟：{result['latency_s']:.3f} 秒")

    # 步骤 4: 保存结果
    output_path = RESULTS_DIR / "task1_sanity_check.json"

    # 移除 raw 中的大字段以便阅读
    save_result = result.copy()
    save_result["raw"] = result["raw"]  # 保留完整数据

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到：{output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
