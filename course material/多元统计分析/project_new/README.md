# Project New 使用说明

本文件夹是 MA304 Project 2 的新版完整实现，主题为“证券投资策略与客户交易画像精准匹配研究”。该版本已根据 `docs` 中 Project 1/2 要求和根目录《计划修改.docx》更新：策略样本扩展到 20 个，模拟客户扩展到 25 个，并采用“风格相似 + 收益风险优选”的两阶段匹配逻辑。

## 1. 目录结构

```text
project_new/
├─ project_plan_latest.md          # 最新项目计划
├─ README.md                       # 本说明文档
├─ report_draft.md                 # 中文初步报告
├─ src/
│  ├─ config.py                    # 路径、策略池、特征列表配置
│  ├─ utils.py                     # 通用计算函数
│  ├─ build_dataset.py             # 数据读取、特征工程、模拟客户生成
│  └─ run_analysis.py              # PCA、聚类、判别、匹配、检验、可视化
└─ outputs/
   ├─ figures/                     # 最终图表
   ├─ tables/                      # 最终表格
   └─ logs/                        # 预留日志目录
```

## 2. 数据来源

原始数据仍从仓库根目录的 `data/raw/` 读取：

- `量化策略绩效-1.xlsx`
- `量化策略绩效-2.xlsx`
- `模拟账户A的记录.xlsx`
- `模拟账户B的记录.xlsx`
- `模拟账户C的记录.xlsx`

本项目使用 20 个策略、3 个真实客户和 25 个模拟客户，总计 48 个分析实体。策略端风险收益特征优先使用原始策略总表中的官方 2025 累计收益率、最大回撤和近 6 年年化收益率；客户端由于缺少期初持仓成本和完整资产曲线，收益风险使用可配对买卖记录的 FIFO 近似结果。

按《计划修改.docx》思路，本项目实际纳入了新增策略主题；但由于原始 workbook 中未发现可与“红利ETF精选策略”稳定一一对应的独立交易页，本次以数据完整且名称可核对的 `双创50增强` 替代，并已在项目计划与报告中明确标注。

## 3. 核心特征

最终使用 18 个特征，分为三组：

- 交易行为特征：`log_trade_freq`, `avg_hold_days`, `turnover_rate`, `log_avg_trade_amount`
- 持仓结构特征：`buy_sell_ratio`, `trade_regularity`, `num_stocks`, `concentration_top3`, `concentration_top1`, `avg_position_pct`, `position_peak_ratio`, `stock_turnover`
- 风险收益特征：`avg_return_pct`, `win_rate`, `profit_loss_ratio`, `max_drawdown`, `return_volatility`, `sharpe_approx`

其中 `trade_freq` 和 `avg_trade_amount` 已按计划做对数变换；`position_peak_ratio` 用来缓解 `avg_position_pct` 与 `max_position_pct` 的强相关冗余。

## 4. 方法流程

完整流程如下：

```text
读取 Excel 原始数据
→ 提取策略和真实客户特征
→ 生成 25 个模拟客户
→ 标准化 18 维特征矩阵
→ PCA 降维解释
→ K-means / 层次聚类 / GMM 聚类对比
→ LDA 判别投影
→ 客户-策略两阶段匹配
→ Kruskal-Wallis 聚类差异检验
→ 输出表格、图表和报告
```

匹配分数采用：

```text
style_score = 0.45 * behavior_similarity + 0.55 * holding_similarity
final_score = 0.70 * style_score + 0.30 * performance_score
```

`performance_score` 根据策略是否优于客户的收益、Sharpe、回撤和波动表现加减分。

## 5. 如何运行

在仓库根目录执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:LOKY_MAX_CPU_COUNT='8'
python -X utf8 .\project_new\src\run_analysis.py
```

运行成功后会重新生成 `outputs/tables/` 和 `outputs/figures/` 下的全部结果。

## 6. 关键结果

最终运行结果：

- 最优聚类数：`K = 3`
- K-means 轮廓系数：`0.4203`
- K-means vs 层次聚类 ARI：`0.8438`
- K-means vs GMM ARI：`0.9699`
- 真实客户 Top1 推荐收益改善比例：`100%`
- 真实客户 Top1 推荐回撤改善比例：`100%`

真实客户 Top1 推荐：

| 客户 | Top1策略 | 综合得分 | 策略收益率 | 客户收益率 | 收益改善 | 策略最大回撤 | 客户最大回撤 | 回撤改善 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 客户A | 双创50增强 | 0.4810 | 1.1131 | 0.0116 | 1.1015 | -0.2103 | -0.4683 | 0.2580 |
| 客户B | 医疗ETF增强 | 0.4341 | 0.2091 | -0.0024 | 0.2115 | -0.1908 | -0.9918 | 0.8010 |
| 客户C | 中证2000增强 | 0.5620 | 0.6757 | 0.0051 | 0.6706 | -0.1532 | -0.4904 | 0.3372 |

## 7. 输出文件

主要表格：

- `outputs/tables/full_feature_dataset.csv`
- `outputs/tables/full_dataset_with_clusters.csv`
- `outputs/tables/real_client_top5_recommendations.csv`
- `outputs/tables/real_client_top1_summary.csv`
- `outputs/tables/matching_ranking_full.csv`
- `outputs/tables/k_selection_metrics.csv`
- `outputs/tables/kruskal_cluster_tests.csv`
- `outputs/tables/pca_explained_variance.csv`

主要图表：

- `01_feature_correlation.png`
- `02_pca_summary.png`
- `03_pca_loadings.png`
- `04_k_selection.png`
- `05_dendrogram.png`
- `06_cluster_agreement.png`
- `07_lda_projection.png`
- `08_matching_heatmap.png`
- `09_real_client_top5.png`
- `12_real_client_improvements.png`
- `13_strategy_risk_return.png`
- `14_real_client_matching_heatmap.png`

## 8. 可提交性判断

当前结果可以作为课程项目的初步提交版本：方法链条完整，代码可复现，图表和表格齐全，且三个真实客户的 Top1 推荐均满足“收益优于客户、回撤优于客户”的项目设想。若课程评分更强调“方法完整性、可解释性、可视化和复现性”，这一版已经适合直接提交；若老师特别强调“真实业务外推”和“严格金融绩效口径”，则建议在口头展示或最终定稿时主动说明限制。

但报告中应保留两个限制说明：

- 客户数据缺少期初持仓成本和完整资产曲线，因此客户收益风险指标是基于可配对买卖记录的近似估计。
- 模拟客户用于扩充样本和展示方法可行性，不应被表述为真实客户分布的充分代表。
