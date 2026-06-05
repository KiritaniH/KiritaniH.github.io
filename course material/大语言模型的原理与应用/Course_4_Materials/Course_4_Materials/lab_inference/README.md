# Lab Inference: vLLM Decoding / KV Cache / Serving

本实验旨在理解从 logits 到 vLLM 服务的完整推理流程。

## 项目结构

```
lab_inference/
├── README.md              # 本文件
├── scripts/               # 核心脚本
│   ├── 00_check_env.py    # 环境检查
│   ├── 01_transformers_inference.py  # Task 1: 基础推理
│   ├── 02_manual_decode.py # Task 5: 手写解码循环
│   ├── 03_decoding_compare.py # Task 2: 解码策略对比
│   ├── 04_prefix_cache_test.py # Task 3: KV Cache 实验
│   ├── 05_benchmark.py    # Task 4: 并发基准测试
│   └── 01_call_vllm.py    # vLLM 客户端（需要 vLLM 服务时使用）
├── results/               # 实验输出
│   ├── task1_sanity_check.json
│   ├── decoding_compare.csv
│   ├── prefix_cache_results.csv
│   └── concurrency_results.csv
└── exp_log.md             # 实验日志（必须提交）
```

## 快速开始

### Windows 注意事项

vLLM 在 Windows 上无法运行（uvloop 不支持 Windows）。本实验使用 `transformers` 完成核心实验目标。

### 1. 环境检查

```bash
python scripts/00_check_env.py
```

### 2. 运行 Task 1（基础推理）

```bash
python scripts/01_transformers_inference.py
```

### 3. 运行 Task 2（解码策略对比）

```bash
python scripts/03_decoding_compare.py
```

### 4. 运行 Task 3（KV Cache 实验）

```bash
python scripts/04_prefix_cache_test.py
```

### 5. 运行 Task 4（并发基准测试）

```bash
python scripts/05_benchmark.py
```

### 6. 运行 Task 5（三种生成路径对比）

```bash
python scripts/02_manual_decode.py
```

## 任务清单

### 必做任务

- [ ] Task 1: 基础推理（`01_transformers_inference.py`）
- [ ] Task 2: 解码策略对比（`03_decoding_compare.py`）
- [ ] Task 4: 并发基准测试（`05_benchmark.py`）

### 选做任务

- [ ] Task 3: KV Cache 实验（`04_prefix_cache_test.py`）
- [ ] Task 5: 三种生成路径对比（`02_manual_decode.py`）

## 实验目标

完成本实验后，你应该能够：

1. 解释 logits、probabilities 和 sampled tokens 之间的关系
2. 理解 greedy decoding、sampling、top-k、top-p 的行为差异
3. 解释为什么自回归推理是逐 token 生成的过程
4. 解释 KV Cache 的作用及为什么它能减少重复计算
5. 设计并运行对比实验，记录实验日志
6. 比较不同生成路径的优缺点

## 提交内容

1. `scripts/` 下的核心脚本
2. `results/` 下的实验输出（CSV/JSON）
3. `exp_log.md` 实验日志
4. 简短结论

## 模型信息

- **模型**: Qwen3-0.6B
- **模型路径**: `C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B`
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)
