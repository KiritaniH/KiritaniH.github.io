# Project1 Report

### 12312515 王洛源

---

## Q1

本小问全程在服务器上进行，相关代码文件及图片结果保存在压缩包Q1内。由于服务器与本地路径不同，直接运行代码可能会报错，如需复现需要调整代码读取文件部分。

#### （1）

本项目中，我选择在课程提供的 Jupyter 服务器上部署 OpenClaw，而不是安装在个人电脑本地。这样做主要有两个原因：第一，课程服务器已经提供了基于 Linux 的开发环境，并预装了 Python 和 JupyterLab，便于后续直接开展数据分析任务；第二，将 OpenClaw 部署在服务器端可以将本次项目的工作目录与个人电脑环境分离，更适合在受控环境下处理课程数据。

在安装 OpenClaw 之前，我首先对服务器环境进行了检查，包括操作系统、Python/Jupyter 是否可用，以及 Node.js 是否已经安装。检查结果表明，该服务器为 Ubuntu 环境，Python 与 JupyterLab 已经配置完成，但最初并未安装 Node.js 和 npm。因此，我先补充安装了 OpenClaw 所需的运行环境，包括 Node.js 与 npm，并在 `/data/project1_env` 下建立了独立的项目目录，用于存放配置文件、日志、临时文件和后续分析工作区。在完成这些准备后，我成功安装了 OpenClaw，并完成了初步的 provider 配置。

###### 1. 整体架构

OpenClaw 可以理解为一个将大语言模型与本地执行环境连接起来的智能代理框架，其整体上可以分为以下四个部分：

1. 用户交互层：负责接收用户输入的任务指令，通常通过终端或 TUI 界面进行交互。
2. 模型提供层（LLM provider）：负责连接外部大语言模型服务。在本项目中，后续配置为我自行购买的千问相关 API。
3. 技能/工具层（skills/tools）：负责执行具体任务，例如代码生成、命令执行、文件操作和数据处理等。
4. 本地执行环境：即课程提供的 Linux/Jupyter 服务器，用于实际执行命令、读取文件并生成分析结果。

##### 2. 工作机制

OpenClaw 的基本工作流程如下：

- 用户输入任务要求；
- OpenClaw 将任务发送给已经配置好的大语言模型；
- 模型根据任务内容决定是直接回答，还是调用相应的技能/工具；
- 被调用的工具在本地环境中执行；
- 执行结果再返回给模型，并最终呈现给用户。

因此，OpenClaw 不只是一个像平常我们使用的deepseek，豆包等LLM对话系统，更像是一个能够与本地分析环境协同工作的任务执行代理。

##### 3. 与数据分析相关的三个能力

在本次项目中，OpenClaw 中与数据分析最相关的能力主要包括以下三类：

1. **文件与工作区交互能力**  
   它可以读取文件、检查目录结构并组织输出结果，这对于管理数据集、分析脚本和结果文件非常有帮助。
2. **命令执行能力**  
   它可以辅助生成并执行 shell 命令或分析相关命令，用于检查依赖、运行程序和管理分析流程。
3. **数据分析代码辅助能力**  
   它可以帮助生成或修改数据分析代码，例如数据读取、描述性统计和结果整理等，这与本项目后续基于 Spark 的分析任务直接相关。

总体而言，OpenClaw 提供了一种将大语言模型与本地数据分析环境结合起来的工作方式，在服务器环境中尤其适合用于组织和辅助完成数据分析任务。

下面是我在服务器上成功配置openclaw，并与之进行第一次对话的截图

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-24-17-42-49-20a0468759298dc01cb99be6ab102118.png" title="" alt="" width="929">

#### （2）

首先我上传了数据文件到服务器并且解压，然后将这一小问的任务进行拆分，交给openclaw完成。下面是我们的对话：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-31-04-56-59-image.png" title="" alt="" width="884">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-31-05-07-01-image.png" title="" alt="" width="887">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-31-05-13-45-image.png" title="" alt="" width="886">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-31-05-19-00-image.png" title="" alt="" width="885">

随后我对openclaw的代码进行了初步检查，均可以运行并且能给出预期的结果，输出文件格式也正确，总结也比较合理。

##### （3）

下面是openclaw总结出的几条结论：可以看得出这些结论符合我们日常认知，从数据上看也能得到支撑。

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-04-03-22-14-12-image.png" title="" alt="" width="346">

##### （4）

我在完成第（3）小问的任务时，前一天先让openclaw生成了五条可能的结论，后续我计划第二天再从中选择三条，完成后面的数据处理工作。但我第二天重新进入服务器准备让openclaw在前一天的工作基础上继续完成任务时，发现它忘记了前一天的工作内容，也搞错了工作目录：

<img title="" src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-04-04-00-46-34-image.png" alt="" width="567"><img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-04-04-00-46-56-image.png" title="" alt="" width="375">

于是我在先重新完成（3）之后，认为可能是因为数据没有保存导致的，于是我详细询问了openclaw哪些内容会被它记录，哪些不会，结果如下：

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-04-04-00-49-04-image.png)<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-04-04-00-48-48-image.png" title="" alt="" width="527">

通过这个小插曲，我对openclaw的工作方式有了更深入的了解。

我认为openclaw在大数据分析任务中，相对于其他AI产品的明显优势在于由于openclaw有较高的本地权限，能很方便快捷的完成复杂任务并执行自我纠错。不过这也带来了数据安全等问题。

---

## Q2

本小问同样全程在服务器上进行，相关代码文件及图片结果保存在压缩包Q2内。

##### （1）

整合出的ipynb文件为survival_analysis.ipynb，相关生成的图表和分析结果文件名为result.csv、survival_analysis_cn.png、survival_curve_cn.png

##### （2）

报告文件为report.md，导出后的文件为同名的pdf文件。

##### （3）

本小问中，我直接利用了前两问的原始数据建立了MySQL数据库，其中schema如下：

```
Database: diabetes_health
Table: health_indicators

Columns:
id (INT, primary key)
Diabetes_012 (DECIMAL(2,1))
HighBP (TINYINT)
HighChol (TINYINT)
CholCheck (TINYINT)
BMI (DECIMAL(5,1))
Smoker (TINYINT)
Stroke (TINYINT)
HeartDiseaseorAttack (TINYINT)
PhysActivity (TINYINT)
Fruits (TINYINT)
Veggies (TINYINT)
HvyAlcoholConsump (TINYINT)
AnyHealthcare (TINYINT)
NoDocbcCost (TINYINT)
GenHlth (TINYINT)
MentHlth (TINYINT)
PhysHlth (TINYINT)
DiffWalk (TINYINT)
Sex (TINYINT)
Age (TINYINT)
Education (TINYINT)
Income (TINYINT)
```

然后我通过与千问对话，尝试基于这个数据库让他通过写SQL代码来完成不同的任务，下面是我观察到的两个千问的失败案例：

**Case1：**

```
Q：按 BMI 从高到低列出前 10 个 Diabetes_012=2 的样本

LLM WA：
SELECT 
    id,
    Diabetes_012,
    BMI,
    Age,
    Sex,
    HighBP,
    HighChol,
    Smoker,
    PhysActivity,
    GenHlth
FROM health_indicators
WHERE Diabetes_012 = 2
ORDER BY BMI DESC
LIMIT 10;

My answer：
SELECT *
FROM health_indicators
WHERE Diabetes_012 = 2
ORDER BY BMI DESC
LIMIT 10;
```

**Case2：**

```
Q：糖尿病患者的平均BMI是多少？

LLM WA：
SELECT 
    ROUND(AVG(BMI), 2) AS Avg_BMI_Diabetic,
    COUNT(*) AS Sample_Size,
    MIN(BMI) AS Min_BMI,
    MAX(BMI) AS Max_BMI,
    ROUND(STDDEV(BMI), 2) AS BMI_StdDev
FROM health_indicators
WHERE Diabetes_012 = 2
  AND BMI IS NOT NULL;

My SELECT
    ROUND(AVG(BMI), 2) AS Avg_BMI_Diabetic,
    COUNT(*) AS Sample_Size,
    MIN(BMI) AS Min_BMI,
    MAX(BMI) AS Max_BMI,
    ROUND(STDDEV(BMI), 2) AS BMI_StdDev
FROM health_indicators
WHERE Diabetes_012 IN (1, 2)
  AND BMI IS NOT NULL; 
```

很遗憾的是现在LLM发展的确实很快，我经过多次试验，没有发现千问给出的代码无法运行或者返回错误结果的现象。但是上面这两个例子仍然体现了LLM的局限性。在Case1中，我预期的结果是返回这 10 个样本的完整记录信息，也就是表中的全部字段。但 LLM 实际生成的 SQL 只选择了部分字段，而没有返回样本的全部字段信息。虽然这条 SQL 可以正确运行，并且筛选和排序逻辑也是正确的，但它并没有完整满足查询任务。我认为这个失败案例说明 LLM 在 text-to-SQL 场景下，可能会根据自己的判断擅自简化输出内容，只选择它认为重要的列。这种错误不会导致 SQL 报错，但会导致结果与用户真实需求不一致。Case2也同理，这里的糖尿病患者的定义不够明确，既可以包括前期也可以不包括，但是LLM自动在这一歧义处帮用户做了选择。在大型的任务中，这种prompt不够严谨的现象是难免的，而LLM缺少纠错或者追问机制，只会按照自己的理解执行任务，这样虽然可以执行成功，但未必满足用户真实需求，这一现象又难以发现，容易产生不良后果。

##### (4)

我创建的网页为[https://kiritanih.github.io/](https://kiritanih.github.io/)，打开页面后，点击链接可以查看第（2）问的报告内容
