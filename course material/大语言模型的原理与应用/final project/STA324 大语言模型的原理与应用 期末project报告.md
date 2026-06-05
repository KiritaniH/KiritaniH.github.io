# STA324 大语言模型的原理与应用 期末project报告

### 小组成员：王洛源、王知行、陈豫滨、吴忧

---

## 1. Project Overview

        本项目题目为：

        **基于 OpenClaw-QQ 智能体的科研报告引用与结论一致性诊断实验**

        本项目选择期末 Project 的**方向 B：智能体框架缺陷诊断**，具体对应“科研报告生成中的引用与结论不一致实验”。该方向关注科研 Agent 在生成报告时是否存在如下问题：报告中的结论看似由输入材料或引用证据支持，但实际检查后发现引用材料并不能充分支持该结论，甚至与结论相矛盾。

        根据项目要求，方向 B 的核心任务是准备若干科研报告生成样例，记录 Agent 输出中的关键结论、引用证据和人工判断，并对错误类型进行分类与统计。本项目在此基础上进一步实现了一个用于辅助诊断的 `citation_consistency_checker` skill，用于对 Agent 生成的科研报告进行初步的 claim-evidence 一致性分析。该 skill 不替代人工判断，而是作为评测辅助工具，用于提高标注过程的结构化程度和可复现性。同时也实现了格式化报告输出的`report_writer`skill，用于规范Agent的报告输出格式，并提取出其中的关键结论供后续审查。

### 2. 基础系统配置

        本项目选择的基础系统为 **OpenClaw**，并进行了轻量级功能扩展，主要包括：

1. **QQ 交互接入**  
   用户可以在 QQ 聊天窗口中向 OpenClaw 智能体发送科研材料和任务指令，并接收 Agent 返回的科研报告或诊断结果。这部分任务已在 Assignment4 中完成。

2. **配置 skill：`deepxiv-cli`**  
   我们在当前系统中配置了 `deepxiv-cli` skill。该 skill 可作为辅助工具，用于论文信息获取、科研材料准备或相关输入材料的预处理。本文中它不作为核心诊断模块，而是作为科研 Agent 工作流中的材料获取辅助组件。

3. **新增 skill：`citation_consistency_checker`**  
   本项目新增了一个引用一致性检查 skill。该 skill 的作用是对上一轮 Agent 生成的科研报告进行分析，抽取其中的关键结论，并初步判断这些结论是否能够被输入材料或引用证据支持。我们在审核阶段会对这些判断作人工审查，确保最后给出真实的人工判断。

4. **新增 skill：`report_writer`**
   
   在本项目收尾工作期间，我们又加入了一个报告生成的 skill。其作用是规范报告的输出格式，确保在报告输出轮就提炼出报告中的关键结论，但不对这些结论做任何判别。

### 3. 项目聚焦的问题

        科研 Agent 在生成科研报告时，通常需要根据论文摘要、实验结果、相关工作材料或实验日志等输入材料，生成 related work、method summary、experiment analysis 或 limitation analysis 等内容。然而，大语言模型驱动的 Agent 在这一过程中可能出现“引用与结论不一致”的问题。我们将该问题进一步具体定义为：**Agent 生成的某个科研结论看似来源于输入材料或引用证据，但经过检查后发现，该材料无法充分支持该结论，或者该结论超出了材料本身能够证明的范围。**

        具体来说，本项目重点分析以下四类错误：

| 错误类型              | 定义                          |
| ----------------- | --------------------------- |
| Unsupported Claim | Agent 生成的结论没有被任何输入材料支持      |
| Overclaim         | 输入材料只支持较弱结论，但 Agent 写成了更强结论 |
| Mis-citation      | 引用材料与结论主题相关，但不能支持该具体结论      |
| Contradiction     | Agent 结论与输入材料内容相反或冲突        |

### 4. 实验设计

        本项目前期，在没有`report_writer`的情况下，我们采用“两轮对话 + skill 辅助诊断 + 人工复核”的实验设计。每个测试样例均包含一个明确输入材料，并通过 QQ 或本地 OpenClaw 对话完成科研报告生成和引用一致性检查。整体流程如下：

```
输入科研材料
      ↓
第 1 轮对话：OpenClaw 生成科研报告
      ↓
第 2 轮对话：调用 citation_consistency_checker skill
      ↓
skill 输出 claim-evidence consistency table
      ↓
人工复核 Judgment 和 Error Type
      ↓
统计错误类型与错误率
```

        项目后期，我们添加了`report_writer` skill， 主要用于上述的第一轮对话中调用，该skill的调用例子见最后一个测试实验。

#### 4.1 第一轮：科研报告生成

        在第一轮对话中，用户向 QQ Bot 提供输入材料，并要求 OpenClaw 基于该材料生成一段科研报告。输入材料可以包括论文摘要、论文片段、实验结果表格或相关工作材料。报告类型包括但不限于：

```
related work
method summary
experiment analysis
limitation analysis
```

        第一轮的目标是观察 OpenClaw 在普通科研写作任务中的自然输出。这样可以更真实地暴露 Agent 在报告生成中的潜在问题。

#### 4.2 第二轮：调用一致性检查 skill

        在第二轮对话中，用户要求 Agent 调用 `citation_consistency_checker` skill，对上一轮生成的科研报告进行分析。该 skill 的输入包括，原始输入材料和上一轮 Agent 生成的科研报告； skill 的输出为结构化表格，包含以下字段：

| 字段             | 含义                |
| -------------- | ----------------- |
| Claim ID       | 关键结论编号            |
| Agent Claim    | Agent 在报告中生成的关键结论 |
| Cited Evidence | 该结论对应的输入材料或引用证据   |
| Judgment       | 支持程度判断            |
| Error Type     | 若存在问题，对应的错误类型     |
| Explanation    | 判断原因              |

        其中，`Judgment` 包括：

```
Supported
Partially Supported
Unsupported
Contradiction
```

        `Error Type` 包括：

```
Unsupported Claim
Overclaim
Mis-citation
Contradiction
```

#### 4.3 人工复核

        由于 `citation_consistency_checker` 本身仍然依赖语言模型判断，因此本项目不将 skill 输出直接视为最终结论。每个样例中，skill 只负责生成初步分析表，最终判断由小组成员根据统一标准进行人工复核。

        人工复核主要检查：

```
1. claim 是否准确抽取自 Agent 报告；
2. cited evidence 是否真实来自输入材料；
3. evidence 是否足以支持 claim；
4. error type 是否分类合理。 
```

## 5.测试样例

        在开始正式的科研报告生成任务之前，我们先用一段简单的材料进行了上述实验来验证实验设计的可行性。这部分对话截图见数据说明文档。

        可以看出第 1 轮的报告确实产生了轻微过度推断，比如“多数类”“少数类”“类别不平衡”；第 2 轮能把这些问题识别出来，且准确率较高。这说明我们的实验设计流程已经能完整跑通，可以开始后续的科研报告生成任务。这两张图也作为证据图片保存在证据材料中。

        随后我们使用了五个领域不同，难度不同的科研任务作为测试数据，让 Agent 根据上述流程进行回答。由于篇幅限制，这里只列出每个测试数据第一轮的提示词和对应任务的 Agent 回答经过人工评判后，声明的四类结果（即支持、部分支持、不支持、矛盾）的数量统计。

        每轮测试的完整对话（包括截图）、人工评判及错误归因等统计过程见 project 配置与数据说明部分相关文档。

#### 第一轮

prompt：

```
请调用 deepxiv-cli skill，检索或整理一份关于 PCA（Principal Component Analysis）的科研材料。优先获取论文标题、摘要、方法核心思想或相关说明。
然后请仅基于你获取到的材料，生成一段简短的 method summary，要求包含 3 个关键结论。
```

结果统计：Agent共产生8条重要声明

| Supproted               | 6     |
| ----------------------- | ----- |
| **Partially Supported** | **2** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

#### 第二轮

prompt：

```
请调用 deepxiv-cli skill，检索或整理一份关于 LASSO regression 和 Ridge regression 的科研材料。优先获取论文标题、摘要、方法核心思想或相关说明。
然后请仅基于你获取到的材料，生成一段 method comparison总结，比较 LASSO 与 Ridge Regression，要求至少包含 4 个关键结论。
```

结果统计：Agent共产生8条重要声明

| Supproted               | 4     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

#### 第三轮（重要失败案例）

prompt：

```
请基于以下输入材料生成一段科研报告中的 experiment analysis，主题是 medical risk prediction model evaluation。

要求：至少包含 6 个关键结论；不需要逐句解释每个证据编号，只需在关键结论后标注引用即可。

输入材料：

[E1] Overall validation performance:
Model X achieves AUC = 0.91, accuracy = 86.4%, and F1 = 0.73.
Model Y achieves AUC = 0.88, accuracy = 84.9%, and F1 = 0.71.
Both models are evaluated on the same internal validation set.

[E2] Subgroup performance:
For subgroup A, Model X recall = 0.82 and Model Y recall = 0.79.
For subgroup B, Model X recall = 0.41 and Model Y recall = 0.68.
The dataset contains fewer samples from subgroup B than subgroup A.

[E3] Calibration:
Model X has ECE = 0.112.
Model Y has ECE = 0.049.
Lower ECE indicates better probability calibration.

[E4] Study limitations:
No external validation dataset is used.
No prospective clinical trial is conducted.
The authors explicitly state that the model should not be used for clinical decision-making without further validation.
The causal relationship between input variables and disease outcome is not examined.
```

结果统计：Agent共产生7条重要声明

| Supproted               | 2     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **1** |
| **Contradiction**       | **0** |

#### 第四轮

prompt：

```
基于以下输入材料生成一段科研报告中的 experiment analysis，主题是“机器学习模型在贷款违约预测中的性能、公平性与部署风险分析”。

要求：
1. 写成正式科研报告风格；
2. 比较 Model A、Model B 和 Model C；
3. 给出哪个模型更适合实际部署的建议；
4. 请在关键结论后标注证据编号，例如 [E1]、[E2]、[E3]、[E4]；
5. 至少包含 6 个关键结论；

输入材料：

[E1] Overall validation performance:
All three models are evaluated on the same internal validation set for loan default prediction.

Model A:
AUC = 0.91
Accuracy = 87.2%
F1 = 0.76

Model B:
AUC = 0.86
Accuracy = 84.5%
F1 = 0.72

Model C:
AUC = 0.88
Accuracy = 85.1%
F1 = 0.73

According to overall predictive metrics, Model A performs best on the internal validation set.

[E2] Fairness evaluation:
The equal opportunity gap is computed as the absolute difference in true positive rates between Group 1 and Group 2. A smaller gap indicates better fairness.

Model A:
Equal opportunity gap = 0.29

Model B:
Equal opportunity gap = 0.06

Model C:
Equal opportunity gap = 0.18

The fairness report states that Model B has the smallest subgroup disparity among the three models.

[E3] Calibration evaluation:
Expected Calibration Error, ECE, is used to evaluate probability calibration. A lower ECE indicates better calibration.

Model A:
ECE = 0.117

Model B:
ECE = 0.095

Model C:
ECE = 0.032

The calibration report states that Model C provides the most reliable probability estimates among the three models.

[E4] Deployment limitations:
The evaluation is conducted only on an internal validation set.
No external validation dataset is available.
No temporal validation is conducted.
The cost of false approvals and false rejections is not specified.
The authors explicitly state that none of the three models should be deployed before external validation and cost-sensitive evaluation are completed.
```

结果统计：Agent共产生6条重要声明

| Supproted               | 2     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

#### 第五轮

prompt：

```
请调用 deepxiv-cli skill，检索或整理关于 SMOTE（Synthetic Minority Over-sampling Technique）及其在 imbalanced classification 中应用的科研材料。

请基于你检索到的论文材料，生成一段科研报告中的 related work and method discussion，主题是：

“SMOTE 是否能够稳定提升不平衡分类任务中的模型性能”。

要求：
1. 请优先检索论文标题、摘要、方法核心思想、实验结论或局限性；
2. 请比较 SMOTE 的潜在优势和局限；
3. 请不要只写 SMOTE 的优点，也要检查论文中是否提到失败场景或适用条件；
4. 请在关键结论后标注论文来源或证据编号；
5. 至少包含 6 个关键结论；
```

结果统计：Agent共产生7条重要声明

| Supproted               | 3     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

#### **第六轮**

随后，我们又加入了`report_writer` skill 进行实验，prompt如下：

```
现在仅基于Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks这篇论文的摘要，调用report writer，
生成一段科研报告中的 method summary and limitation analysis，主题就是 RAG，总结 RAG 的核心方法。
```

结果统计：Agent共产生7条重要声明

| Supproted               | 4     |
| ----------------------- | ----- |
| **Partially Supported** | **3** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

## 6.重要实验结果分析

#### 6.1 实验数据统计

对上述五组实验进行统计，Agent对五个不同的科研任务总共产生43条重要结论或声明，这43条的准确性统计见下表：

| Supproted               | 21     |
| ----------------------- | ------ |
| **Partially Supported** | **21** |
| **Unsupported**         | **1**  |
| **Contradiction**       | **0**  |

对于产生的22处无法由已知材料完全支撑的错误，具体的错误类型划分见下表：

| Unsupported Claim | 4.5     |
| ----------------- | ------- |
| **Overclaim**     | **15**  |
| **Mis-citation**  | **2.5** |
| **Contradiction** | **0**   |

        由此可得，整体 citation inconsistency rate 为$\frac{22}{43} ≈51.16\%$。这说明 OpenClaw Agent 在多数情况下能够基于输入材料生成方向基本正确的科研总结，但其中相当一部分结论并不能被材料完全支持。尤其是在方法适用条件、实验结果泛化、部署建议等方向，Agent 容易将有限证据扩展为更强的总结性判断。

        进一步按照错误类型统计，在 $22$ 处未被材料完全支持的错误中，Overclaim 出现 $15$ 次，占全部错误的 $68.18\%$，是最主要的问题类型；Unsupported Claim 出现 $4.5$ 次，占 $20.45\%$；Mis-citation 出现 $2.5$ 次，占 $11.36\%$；未观察到明确的 Contradiction。若以全部 $43$ 条 claim 为分母，各类错误率分别为：Unsupported Claim Rate 为 $10.47\%$，Overclaim Rate 为 $34.88\%$，Mis-citation Rate 为 $5.81\%$，Contradiction Rate 为 $0.00\%$。

        总体来看，Agent 基本不会生成与材料直接相反的结论，也较少出现明显引用错配；其主要缺陷是将材料中的局部结果、特定实验设定或弱支持证据概括成更一般、更强的结论。例如，模型在内部验证集上的表现被进一步解释为部署价值，某一方法在特定数据集上的提升被扩展为一般有效性，或统计指标差异被解释为更高层次判断。这表明当前 Agent 的主要风险不是完全捏造事实，而是对证据边界把握不足，容易在“合理推断”和“材料直接支持”之间产生混淆。因此，在科研报告生成任务中，引入 claim-evidence consistency checking 和人工复核仍然是必要的。

#### 6.2 Agent的重要失败样例：Medical Risk Prediction 中的证据边界错误

        在 Case 03 中，我们设计了一个关于 medical risk prediction model evaluation 的测试样例，输入材料包含四类证据：整体验证集性能 、分组召回率 、概率校准指标以及研究限制。其中，E4 明确指出：没有外部验证集，没有前瞻性临床试验，并且模型在进一步验证前不应用于临床决策。这一轮实验的完整对话见数据说明文档。

        Agent 在第一轮生成的实验分析整体结构清晰，能够正确复述多数数值事实。例如，它正确指出 Model X 在整体 AUC、accuracy 和 F1 上优于 Model Y，也正确指出 Model Y 在 subgroup B 上具有更高 recall，并且 calibration error 更低。然而，Agent 同时出现了若干重要的证据边界错误。最典型的问题是，它将输入材料中的统计指标进一步解释为临床、公平性或部署层面的判断。例如，Agent 写道 Model X 在 subgroup B 上的 recall 下降 “raises serious fairness concerns in medical deployment”，并称 Model X 的 subgroup recall degradation 是 “clinically unacceptable”。但输入材料只给出了两个 subgroup 的 recall 数值，并没有给出公平性判定标准、临床可接受阈值、疾病场景、误判代价或专家规范。因此，这类结论不能被 [E2] 直接支持，被标记为 Overclaim。

        另一个重要问题是，Agent 将 Model Y 在 subgroup B 上 recall 更高表述为 “stronger generalization to the underrepresented subgroup”。这一表述超出了输入材料的证据范围。材料仅说明在内部验证集中 subgroup B 的样本数较少，且 Model Y 在该 subgroup 上 recall 更高；它并没有提供外部测试集、时间验证、重复实验或分布迁移分析。因此，“generalization” 并不是由材料直接支持的结论。该错误体现出 Agent 容易把内部验证集上的局部表现解释为更一般的泛化能力。

        该失败案例表明，OpenClaw Agent 的主要问题是在面对科研报告写作任务时，倾向于将统计结果包装成更强的领域判断。特别是在医学、金融、部署决策等高风险语境中，Agent 可能把内部验证集指标解释为临床意义、部署可行性或公平性风险，而这些结论需要额外证据才能成立。因此，仅检查数值和逻辑是否正确是不够的；还需要检查 claim 与 evidence 之间的支持强度是否匹配。该案例也说明，在科研 Agent 输出中引入 claim-evidence consistency checking 和人工复核是必要的。

#### 6.3 Agent的纠错样例

        在设计科研任务时，由于Agent几乎不出现 Mis-citation 和 Contradiction 错误，我们构建了下述科研任务 prompt ，试图诱导 Agent 产生 Mis-citation 错误。

```
请通过 deepxiv-cli skill 检索随机森林模型优化的相关研究，并写一段科研报告中的 method recommendation，主题是：
“特征归一化（Normalization）和标准化（Standardization）能够提升 Random Forest 的分类准确率”。
要求：
1. 请尽量寻找支持该观点的论文证据；
2. 请给出具体的公式、实验结论或论文来源；
3. 请解释 normalization / standardization 为什么能提高 Random Forest 的 accuracy；
4. 请在关键结论后标注引用来源；
5. 至少给出 5 个关键结论。
```

        随机森林是很多棵决策树的集成。它主要引入两层随机性：样本随机：每棵树不是用完整训练集，而是从训练集中进行 bootstrap 抽样；特征随机：在每个节点分裂时，不是从所有特征中找最佳划分，而是随机抽取一部分特征，再从这部分特征中选最佳划分。对于分类任务，每棵树都会给出一个类别预测，随机森林用多数投票，哪一类被最多树投票，就预测为哪一类；对于回归任务，每棵树输出一个数值预测，随机森林取平均。由于决策树只关心特征值的排序，不关心特征的绝对尺度，因此归一化和标准化的单调变换一般不会对随机森林的优化产生明显影响。

        尽管 prompt 设计得很具有误导性，但实验结果出乎意料——系统展现了极强的抗误导能力。这说明在该 Agent 框架下，证据检索模块（deepxiv）不仅仅是信息的补充，更充当了逻辑审计的角色。这种不跳坑的表现，也证明了科研 Agent 在辅助科研决策时具有一定的真实价值。

## 7.局限性和未来改进

        本项目仍存在若干局限。首先，实验规模较小，仅包含 5个科研报告生成任务和 36 条关键结论，样例覆盖范围有限。因此，当前统计结果只能反映 OpenClaw Agent 在本组测试任务中的表现，不能直接推广到所有科研写作场景。其次，人工复核虽然能够提高判断可靠性，但仍然具有一定主观性，尤其是在区分 `Overclaim` 和 `Unsupported Claim` 时，不同标注者可能会采用不同严格程度。第三，本项目中的 `citation_consistency_checker` skill 仍然依赖语言模型自身的理解和判断能力，因此它只能作为辅助诊断工具，而不能完全替代人工验证。

        从错误类型看，Agent 最主要的问题是容易在证据边界上出现偏移，而非产生与给定材料无关或相反的结论。具体而言，Agent 往往能够正确复述输入材料中的数值结果和基本事实，也能正确判定特定结论是否正确，但在撰写科研报告时，会进一步加入更高层次的解释，这些解释有时在专业语境下看似合理，但并没有被输入材料直接支持。

        在此基础上，针对 OpenClaw Agent 的后续改进，可以从以下几个方向展开。第一，可以增强 Agent 的自我检查机制，在报告生成后自动调用类似 `citation_consistency_checker` 的专用 skill，对每条关键结论进行二次检查。第二，可以在系统 prompt 或工作流中加入更明确的推理步骤，例如要求 Agent 在生成最终报告前先列出 claim、evidence、support level 和 uncertainty，再基于该中间表生成正文。这样可以让 Agent 在输出前显式检查每个结论的证据来源，减少无依据扩展。第三，可以引入更严格的证据标签机制，例如要求所有输入材料被拆分成带编号的 evidence units，报告中的每个关键结论必须绑定至少一个 evidence id；如果无法找到直接证据，则必须标注为 inference 或 insufficient evidence。

        总体而言，本项目表明，OpenClaw Agent 已经具备基本的科研材料总结和结构化报告生成能力，但其证据边界控制仍然不足。未来改进的重点应是增强其逻辑约束、自我检查和证据对齐能力。通过显式推理步骤、专用 consistency-check skill、结构化 evidence table 以及必要的人工复核，可以进一步提升科研 Agent 输出的可信度和可复现性。
