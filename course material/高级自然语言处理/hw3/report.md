## 高级自然语言处理 第三次作业

#### 12312515 王洛源

---

### Q1：

（1）

| 阶段                 | 目标        | 获得能力                                       | 挑战                        | 代表模型                     |
| ------------------ | --------- | ------------------------------------------ | ------------------------- | ------------------------ |
| Pretraining        | 学习语言和世界知识 | 掌握自然语言结构和基本知识，模拟一些简单的任务                    | 数据噪声，缺少大量高质量语料，幻觉问题，算力较昂贵 | GPT-3, Gopher, BLOOM     |
| Instruction Tuning | 理解并遵循人类指令 | 展现推理过程，遵循指令，推广到复杂情境                        | 需要大规模且多样的数据，需要提高代码和推理能力   | InstructGPT, FLAN-T5,    |
| Alignment Tuning   | 对齐人类价值与偏好 | 在遵守人类基本价值观前提下完成任务，拒绝有伤风化或难以完成的指令，公平正式的表达方式 | 成本较高，有过度对齐风险              | ChatGPT, Claude, Sparrow |

（2）

**Emergent Abilities**指的是：当模型规模（参数量或训练数据量）增加到一定临界点时，  模型突然表现出一些在小模型上完全不存在、无法通过线性外推预测的新能力。例如数学推理，逻辑链式思考，上下文学习能力等。这是一种量变引起质变的表现，也是模型表示能力与任务复杂度之间的交互效应。

（3）

我在作业文件给出的github库里下载了需要的gsm8k和math的数据集，并且在示例代码的基础上进行了修改，然后分别使用Qwen2.5-0.5B进行测试，原始代码见压缩包内文件，结果如下，从上到下分别是gsm8k使用k=0，k=5；math使用k=0，k=5的结果：

![](C:\Users\lenovo\Desktop\advnlp\hw3\gsm8k_k=0.png)

![](C:\Users\lenovo\Desktop\advnlp\hw3\gsm8k_k=5.png)

![](C:\Users\lenovo\Desktop\advnlp\hw3\math_k=0.png)

![](C:\Users\lenovo\Desktop\advnlp\hw3\math_k=5.png)

最终结果以表格形式呈现如下：

| 正确率           | GSM8K | MATH  |
|:-------------:| ----- | ----- |
| zero-shot CoT | 0.09  | 0     |
| few-shot CoT  | 0.06  | 0.014 |

注意到当k=5时，模型在GSM8K数据集上的正确率反而下降了，这可能是因为模型规模很小，few-shot由于上下文变长、示例分布与测试题差异等因素，反而略微降低了表现导致的。

---

### Q2：

（1）

**Diversity Filtering：** LLMs 会生成大量内容，但其中许多是高度相似、冗余或模板化的。在此基础上，Diversity Filtering 的目标是减少重复，增加语义覆盖范围，提升数据集的多样性与代表性。例如ROUGE-L可以用于将 newly generated instructions 与 seed instruction 做比对，ROUGE-L 越高表示越重复，Self-Instruct 设置 ROUGE-L 小于一定的阈值才保留。

**Quality Filtering：** 在LLM 生成的质量参差不齐的数据中，通过评分机制筛选“自然流畅、有信息量”的样本。例如Best-of-K对同一个指令生成 K 个候选回答，随后用 reward 模型选择最佳回答。

**Correctness Filtering：** 一些特定的专业性强的任务（如数学、逻辑、代码等）生成的内容可能“看起来像回答”，却是错的。因此Correctness Filtering 保证只保留 事实上也正确或者逻辑上合法的样本。例如STaR的基础逻辑是如果多个 reasoning paths 中只有部分有正确答案，那么只保留正确的 reasoning chain。

（2）我们实现了一个简单的 Reward-based Filtering pipeline：使用 Qwen2.5-0.5B生成 k=5个候选回复，随后定义了一个简单的启发式奖励函数，其中：

- length_score：鼓励内容适中、不太短的回答。

- keyword_score：鼓励回答覆盖预期关键术语。

- completeness_score：避免回答中出现话没说完，句子不完整的情况。

- repetition_penalty：鼓励重复性较低的回答。

- fluency_score：鼓励语言更流畅自然的回答。

并且定义reward为这五部分的加权平均：

```
 return (
        0.10 * length_score +
        0.25 * keyword_score +
        0.15 * repetition_score +
        0.35 * completeness +
        0.15 * fluency
    )
```

对每条指令，保留 reward 最高的候选作为过滤后样本，部分运行结果如图所示：

![](C:\Users\lenovo\Desktop\advnlp\hw3\RewardbasedFilteringResult1.png)

![](C:\Users\lenovo\Desktop\advnlp\hw3\RewardbasedFilteringResult2.png)

完整代码见压缩包内文件。
