# Project配置与数据说明：

### 1.配置文件说明

项目基于openclaw框架，配置了`deepxiv-cli` skill并实现了 skill：`citation_consistency_checker`。其中后者的SKILL.md文件在压缩包内，使用方式为移动到~/.openclaw/workspace/skills/citation_consistency_checker/SKILL.md，然后重启 openclaw gateway。重启后可以使用`openclaw skills list`检验skill是否配置成功。

---

### 1.5.实验可行性测试过程

在开始正式的科研报告生成任务之前，我们先用一段简单的材料进行了上述实验来验证实验设计的可行性，下面是第一轮对话中我们给Agent提供的材料和Agent的回答：

![](D:\学习材料\大三下\LLM\final%20project\test1.png)

下面是第二轮对话中Agent的结果：

![](D:\学习材料\大三下\LLM\final%20project\result1.png)

---

### 2.测试数据展示

项目使用的五个测试数据内容和Agent两轮的回复，以及我们最后的人工评判结果如下：

**第一组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test2_prompt.png)

<img src="file:///D:/学习材料/大三下/LLM/final%20project/test2_answer.png" title="" alt="" width="1010">

<img src="file:///D:/学习材料/大三下/LLM/final%20project/result2_1.png" title="" alt="" width="1012">

<img src="file:///D:/学习材料/大三下/LLM/final%20project/result2_2.png" title="" alt="" width="1011">

人工评判：C1.1的声明缺少了正交投影、重构误差、固定维数、中心化数据等限定条件，因此修正为 Partially Supported（Overclaim）。其他评判不做修改。

本轮总结：

| Supported               | 6     |
| ----------------------- | ----- |
| **Partially Supported** | **2** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 0     |
| ----------------- | ----- |
| **Overclaim**     | **2** |
| **Mis-citation**  | **0** |
| **Contradiction** | **0** |

**第二组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test3_1.png)

![](D:\学习材料\大三下\LLM\final%20project\test3_2.png)

第二轮对话由于qq电脑端的显示bug，没有电脑端截图，附件中有手机端截图，下面是将Agent回复转换为md格式后的结果：

| ID  | Agent Claim                                                                | Cited Evidence                                                                                                                                     | Judgment  | Error     | Explanation                                                                                   |
| --- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- | --------------------------------------------------------------------------------------------- |
| C1  | LASSO 通过 L1 菱形约束域实现内置变量选择（系数可精确为零），这是 "几何意外"。Ridge 的 L2 球形约束只能让系数趋近零，永不为零。 | 1509.09169 Sec.44: lasso=diamond constraint, corners hit axes, geometric accident; ridge=spherical, elements close to zero but not zero.           | Supported | —         | 原文直接覆盖                                                                                        |
| C2  | LASSO 的 L1 惩罚在零点不可微，需坐标下降等专门算法求解；Ridge 因 L2 惩罚处处光滑，有显式闭式解。                 | 1509.09169 Sec.43: non-differentiable at zero, more intricate than ridge. Sec.45: coordinate descent/gradient ascent. Sec.50.1: ridge closed form. | Partially | Overclaim | LASSO 非可微→需专门算法被证实。但 Agent 暗示 Ridge=可直接计算，原文未达此断言。                                            |
| C3  | 共线性下：LASSO 倾向于从共线性变量组中只挑一个纳入模型并压零其余；Ridge 则将共线性变量作为一组联合收缩。                 | 1509.09169 Sec.50.4: lasso picks one, forces others out, spoils the party; ridge paths nicely grouped per block.                                   | Partially | Overclaim | LASSO side directly supported. Ridge "joint shrinkage" only from visual grouping description. |
| C4  | 预测性能取决于真实参数结构：密集下 Ridge 优于 LASSO；稀疏下 LASSO 优于 Ridge。基于 1000 次仿真（真实基因表达数据）。 | 1509.09169 Sec.50.5: dense favors ridge, sparse favors lasso; 1000 iterations with breastCancerVDX data.                                           | Supported | —         | 原文逐句覆盖                                                                                        |
| C5  | 稀疏度 10% 时，p 超过 500 后 Ridge 预测性能反超 LASSO。原因：LASSO 高维下变量选择不稳定、不可复现。          | 1509.09169 Sec.50.5: p>500 roles reverse; lasso dimension reduction becomes unstable and less reproducible.                                        | Supported | —         | 原文逐句覆盖                                                                                        |
| C6  | LASSO 的自由度可无偏估计为非零系数个数；X 满秩时该估计量相合。                                        | 1509.09169 Sec.47 Thm 6.3: df = E(nonzero), consistent when rank(X)=p.                                                                             | Supported | —         | 原文逐句覆盖                                                                                        |
| C7  | LASSO 解在高维/共线性下可能不唯一（但预测值 X*beta 始终唯一）；Ridge 解始终唯一。                        | 1509.09169 Sec.42: lasso non-unique, non-strict convex. Ridge uniqueness not explicitly stated.                                                    | Partially | Overclaim | LASSO 证实。Ridge 唯一性是 Agent 推论，原文未显式声明。                                                         |
| C8  | 贝叶斯：Ridge = 高斯先验 MAP；LASSO = 拉普拉斯先验 MAP。拉普拉斯先在零点和尾部质量更多，解释了零估计和偏大估计并存。     | 1509.09169 Sec.48: ridge=Gaussian, lasso=Laplace; more mass near zero and tails; zero or large estimates.                                          | Supported | —         | 原文完整覆盖Summary                                                                                 |

人工评判：C1前半句基本成立；后半句“永不为零”太绝对。更严谨说法应是：Ridge 通常会连续收缩系数，通常不会像 LASSO 那样产生精确稀疏解。“永不为零”在数学上过强。修改为Partially Supported（Overclaim）。其他评判不做修改。

本轮总结：

| Supported               | 4     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 0     |
| ----------------- | ----- |
| **Overclaim**     | **4** |
| **Mis-citation**  | **0** |
| **Contradiction** | **0** |

**第三组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test4_prompt.png)

![](D:\学习材料\大三下\LLM\final%20project\test4_answer.png)

![](D:\学习材料\大三下\LLM\final%20project\result4.png)

人工评判：KC4中，ECE 更低支持 “calibration better”，但 “clinically interpretable and reliable risk estimates” 引入了临床解释可靠性这一说法，材料没有临床标准，因此修改为Unsupported Claim。KC5同理，修改为Unsupported Claim。KC7中，E1反而支持模型 X 更优，材料也未说明应该优先考虑哪个维度，因此评判结果修改为Unsupported，错误类型为Mis-citation与Unsupported Claim各占比一半。

本轮总结：

| Supported               | 2     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **1** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 2.5     |
| ----------------- | ------- |
| **Overclaim**     | **2**   |
| **Mis-citation**  | **0.5** |
| **Contradiction** | **0**   |

**第四组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test5_prompt.png)

![](D:\学习材料\大三下\LLM\final%20project\test5_answer1.png)

![](D:\学习材料\大三下\LLM\final%20project\test5_answer2.png)

![](D:\学习材料\大三下\LLM\final%20project\result5.png)

人工评判：本轮Agent的自我评判比较准确，不作修改。

本轮总结：

| Supported               | 2     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 1     |
| ----------------- | ----- |
| **Overclaim**     | **3** |
| **Mis-citation**  | **0** |
| **Contradiction** | **0** |

**第五组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test6_1.png)

![](D:\学习材料\大三下\LLM\final%20project\test6_2.png)

![](D:\学习材料\大三下\LLM\final%20project\test6_3.png)

![](D:\学习材料\大三下\LLM\final%20project\result6_1.png)

![](D:\学习材料\大三下\LLM\final%20project\result6_2.png)

人工评判：Claim5中，P2 支持 outlier/noise 机制；P7 只支持效果不普遍，不能支持离群点导致性能损害的机制。因此修改为Mis-citation。其他结论不变。

本轮总结：

| Supported               | 3     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 0     |
| ----------------- | ----- |
| **Overclaim**     | **3** |
| **Mis-citation**  | **1** |
| **Contradiction** | **0** |

**第六组测试：**

![](D:\学习材料\大三下\LLM\final%20project\test7_1.png)

![](D:\学习材料\大三下\LLM\final%20project\test7_2.png)

![](D:\学习材料\大三下\LLM\final%20project\test7_3.png)

![](D:\学习材料\大三下\LLM\final%20project\test7_4.png)

人工评判：Claim5中，P2 支持 outlier/noise 机制；P7 只支持效果不普遍，不能支持离群点导致性能损害的机制。因此修改为Mis-citation。其他结论不变。

本轮总结：

| Supported               | 3     |
| ----------------------- | ----- |
| **Partially Supported** | **4** |
| **Unsupported**         | **0** |
| **Contradiction**       | **0** |

错误类型统计：

| Unsupported Claim | 0     |
| ----------------- | ----- |
| **Overclaim**     | **3** |
| **Mis-citation**  | **1** |
| **Contradiction** | **0** |

---

### 3.纠错案例图片展示

![](D:\学习材料\大三下\LLM\final%20project\6.3.png)
