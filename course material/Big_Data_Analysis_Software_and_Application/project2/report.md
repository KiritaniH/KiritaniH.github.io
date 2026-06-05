# Project2 Report

## 12312515 王洛源

---

## Q1

#### （1）

我阅读了相关材料，总结提炼了三条可能有效的数据选择方法如下：

1. **质量筛选**    综述中提到很多 instruction 选择方法已经转向自动质量评分、错误回答过滤、强模型打分，在本项目中，所需数据库 DataMind-12K 本身是 agent trajectory 数据，真正影响 SFT 效果的，最重要的是样本是否正确、完整、可执行。对数据科学和数学建模任务，例如本项目要完成的生成式数据代理系统的训练而言，来自数据的错误步骤、错代码、空泛解释会直接污染模型。这一选择方法可以通过先正则，再调用模型二次筛选来完成。

2. **复杂度/难度筛选**   对于生成式数据代理系统而言，体系的核心是多步推理、分析流程和代码执行链。因此训练数据需要有一定的难度和复杂度。如果只保留非常简单的问答，模型会学不到 agent 的关键能力。综述材料也明确把 difficulty / complexity 视为 instruction 数据选择的重要方向。这一选择方法可以通过优先保留多步骤任务，或带公式推导、统计分析等的数据，降低一问一答的简单样本比例来实现。

3. **多样性选择**   阅读材料提到对于类似本项目的任务而言，训练数据需要更宽的任务覆盖和形式覆盖，模型才更能泛化到真实用户问题。这一选择方法可以通过按任务类别或者输出形式分层采样实现。

#### （2）

我在课程服务器上上传了数据，配置好了后面要用的千问模型。然后抽取了部分数据进行观测，设置了（1）问中方法1和2的正则条件，然后进行多样性选择，结果如下：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-14-17-47-22-image.png)

可以看到数据集的质量较高，质量分和复杂度分都比较高，因此正则方法并没有过滤太多数据。另外这里也可以调用模型来进一步筛选数据，但在本项目中，我担心后面要用的0.8B模型能力不足，因此没有调用。

#### （3）

由于服务器仅提供 CPU 资源，我采用了基于 LoRA 的参数高效微调。以下是我的程序运行结果和生成的ckpt：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-15-00-32-38-image.png)

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-15-00-54-02-image.png)

随后在lab13中，我根据老师提供的代码框架进行了训练，下图是训练开始的截图：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-21-17-16-02-image.png)

#### （4）

下图是我的网页和测试demo：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-17-01-24-32-image.png)

---

## Q2

#### （1）本小问内容使用的LLM为deepseek，对话内容如下：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-18-01-16-49-image.png" title="" alt="" width="767">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-18-11-47-23-image.png" title="" alt="" width="769">

随后LLM提出了这样一个思路，我认为对于初创公司来说比较可行，且赛道上已有竞争对手较少：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-05-18-12-27-58-image.png" title="" alt="" width="655">

基于此思路的商业计划书和路演ppt见Q2_1相关文件

#### （2）

架构设计文档见相关文件，下面是一个总体架构图：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-05-18-15-33-24-image.png)
