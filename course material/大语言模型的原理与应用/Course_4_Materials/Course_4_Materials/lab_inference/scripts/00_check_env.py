#!/usr/bin/env python3
"""检查实验环境：Python 版本、依赖包、GPU、模型路径"""

import sys
import subprocess
from pathlib import Path


def check_python():
    print(f"Python 版本：{sys.version}")
    print(f"Python 可执行文件：{sys.executable}")


def check_packages():
    packages = ["torch", "transformers", "vllm", "requests", "psutil"]
    print("\n依赖包检查:")
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {pkg}: {version}")
        except ImportError:
            print(f"  ✗ {pkg}: 未安装")


def check_gpu():
    print("\nGPU 检查:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 可用")
            print(f"  GPU 数量：{torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ✗ CUDA 不可用，将使用 CPU")
    except ImportError:
        print("  ✗ torch 未安装")


def check_model_path():
    # 请根据实际情况修改模型路径
    model_paths = [
        Path("/home/models/Qwen3-0.6B"),
        Path("/user/lenovo/.cache/modelscope/hub/models"),
        Path("~/models/Qwen3-0.6B").expanduser(),
    ]

    print("\n模型路径检查:")
    for path in model_paths:
        if path.exists():
            print(f"  ✓ 找到模型：{path}")
            # 列出模型文件
            files = list(path.iterdir())[:10]
            for f in files:
                print(f"    - {f.name}")
            return path
    print("  ✗ 未找到常见模型路径，请手动指定")
    return None


def check_vllm_server():
    print("\nvLLM 服务检查:")
    try:
        import requests
        r = requests.get("http://localhost:8000/v1/models", timeout=3)
        if r.status_code == 200:
            print("  ✓ vLLM 服务已启动")
            return True
        else:
            print(f"  ? vLLM 响应异常：{r.status_code}")
    except requests.exceptions.ConnectionError:
        print("  ✗ vLLM 服务未运行，请先启动服务")
    except Exception as e:
        print(f"  ? 检查失败：{e}")
    return False


if __name__ == "__main__":
    print("=" * 50)
    print("Lab Inference 环境检查")
    print("=" * 50)

    check_python()
    check_packages()
    check_gpu()
    check_model_path()
    check_vllm_server()

    print("\n" + "=" * 50)
