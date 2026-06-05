# STA323 Assignment1

#### 12312515 王洛源

---

## Q1

#### （1）代码文件见Q1.py相应部分（已用注释标明），运行结果如下：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-14-11-18-43-image.png" title="" alt="" width="817">

#### （2）使用git bash读取原始csv文件并找到出现频率最高的5个stockcode的指令和截图如下：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-14-11-52-44-image.png" title="" alt="" width="259">

#### （3）代码文件见Q1.py相应部分（已用注释标明），结果可视化图片见q1_obs1_country_revenue.png和q1_obs2_hourly_orders.png

在obs1中，我们可以看到英国（United Kingdom）的总销售额明显高于其他国家，这可能是因为数据统计范围的限制，说明本数据集相关的市场主体为英国，明确了主要目标客户；而在obs2中，可以看到白天尤其是中午（11-15时）的销售额最高，这也符合人们的作息规律，可能有助于广告投放或其他营销策略的制定

---

## Q2

根据题目要求，我们首先对给出的数据按照要求进行处理，最终合并成一个清洗好的数据文件Q2_final.csv，所用代码文件为q2_prepare.py

#### （1）代码文件为Q2_1.py，运行结果如下，统计结果文件保存为Q2_1_result.csv

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-14-16-43-48-image.png" title="" alt="" width="892">

#### （3）所用shell脚本为q2_3.sh，由于清洗规则比较复杂，我使用python代码进行并行处理，相关代码为process_chunk.py，运行结果如下，输出文件为SRR12326775_1_Light_Bulk_final.csv

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-14-17-05-11-image.png" title="" alt="" width="567">

#### （4）代码文件为Q2_4.py，运行结果如下：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-14-18-39-24-image.png" title="" alt="" width="536">

---

## Q3

#### （1）

对于这个问题我使用了三种不同的prompt。

第一版prompt基本上就是复制了题干，内容如下：

```
我是一家初创公司的CTO，计划设计一款国际化的基于AI能力的类似大众点评的APP，请你帮我设计这个产品的系统架构，并解释各部分作用
```

LLM给出的回答如下：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-15-12-23-21-image.png" title="" alt="" width="766">

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-15-12-12-56-image.png" title="" alt="" width="765">

第二版prompt，我将重点放在了“国际化”上，内容如下：

```
我是一家初创公司的CTO，我们公司计划推出一款对标大众点评，但更加国际化，面向全球的生活助手软件。现在请你帮我设计这个产品的系统架构，尤其要注明哪些架构是专门为了国际化（如符合不同国家的法律法规，翻译问题等）而特殊设计的，并解释各部分作用
```

然后LLM给出了这样的架构图：

<img src="file:///C:/Users/lenovo/AppData/Roaming/marktext/images/2026-03-15-12-22-40-image.png" title="" alt="" width="367">

第三版prompt，我将重点放在了对大众点评主要功能的拆解上，内容如下：

```
我是一家初创公司的CTO，计划设计一款国际化的基于AI能力的类似大众点评的APP，具备包括但不限于餐厅，娱乐场所，酒店，电影院等消费场所的搜索，评价，团购优惠等服务，现在请你帮我设计这个产品的系统架构，并解释各部分作用
```

LLM提供了一个五个核心层次组成的系统架构，包括数据采集层， 数据处理与存储层， AI核心引擎层， 业务应用层和用户交互层。整体结果和第一版prompt差不多。

根据以上三轮对话，我发现LLM本身在不借助提示的情况下，对题干要求中的大众点评功能和基于AI能力的理解比较到位，但对于全球化这一关键点考虑不够周到。例如，第一、三版prompt中，LLM都只部分考虑了翻译系统和数据隐私的实现方式，但第二版的LLM考虑了用户的定位，归属地判断，多币种支付，甚至包括高德地图/谷歌地图的展示也考虑了进去。不过第二版的LLM相当于只针对国际化这一问题提出了解决方案，将其整合到第一版的架构中，应当就是一套比较完善的体系了。

#### （2）

**PostgreSQL** ：关系型数据库，强调可靠性、数据完整性、SQL 能力和可扩展性；以 ACID、一致性、可扩展能力和扩展生态见长；还支持 json/jsonb、全文检索、地理扩展 PostGIS；成熟、开源、生态强。

**MongoDB**：官方提供内建的 geospatial search、全文搜索、vector search；支持 ACID 事务；支持分片，适合大规模水平扩展；更偏文档模型与灵活对象存储；虽然功能很多，但对于强关系、强事务、复杂 SQL 型主业务不够稳定。

**Snowflake**：架构强调存储与计算分离，擅长分析型查询、弹性计算、数据共享和统一分析；官方也支持 hybrid tables / Unistore，在向事务场景扩展；产品收费，定位更偏分析而非经典应用主库，更适合分析平台/数仓。

从DB-Engines排名来看，三者都属于主流数据库。而对于题干给出的特定问题，在线主业务必须先保证数据一致性、事务可靠性、复杂关系查询能力、成熟索引与 SQL 能力以及后期扩展能力。总体来看PostgreSQL更适合作为本产品的主业务数据库。

我在第（1）问第一版prompt的对话中使用下面的prompt询问LLM对于数据库选择的建议，LLM做出了和我相同的选择。

```
根据我们的产品特点，分析PostgreSQL , MongoDB , Snowflake这三个数据库对于我们产品适配的优缺点，并选出哪个更适合做我们的主业务数据库
```

![](C:\Users\lenovo\AppData\Roaming\marktext\images\2026-03-15-12-57-16-image.png)
