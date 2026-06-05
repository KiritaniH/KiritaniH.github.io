from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.stats import kruskal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from build_dataset import build_full_dataset
from config import (
    ALL_FEATURES,
    BEHAVIOR_FEATURES,
    FIG_DIR,
    HOLDING_FEATURES,
    PERFORMANCE_FEATURES,
    STYLE_FEATURES,
    TABLE_DIR,
)
from utils import cosine_similarity, min_max_scale


sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def choose_k(x: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(x)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(x, labels),
                "davies_bouldin": davies_bouldin_score(x, labels),
                "calinski_harabasz": calinski_harabasz_score(x, labels),
            }
        )
    return pd.DataFrame(rows)


def performance_advantage(client_row: pd.Series, strategy_row: pd.Series) -> float:
    client_ret = float(client_row["avg_return_pct"])
    client_dd = float(abs(client_row["max_drawdown"]))
    client_vol = float(client_row["return_volatility"])
    client_sharpe = float(client_row["sharpe_approx"])

    strategy_ret = float(strategy_row["avg_return_pct"])
    strategy_dd = float(abs(strategy_row["max_drawdown"]))
    strategy_vol = float(strategy_row["return_volatility"])
    strategy_sharpe = float(strategy_row["sharpe_approx"])

    reward = 0.0
    reward += 0.35 if strategy_ret >= client_ret else -0.20
    reward += 0.20 if strategy_sharpe >= client_sharpe else -0.10
    reward += 0.25 if strategy_dd <= client_dd * 1.15 else -0.15
    reward += 0.20 if strategy_vol <= client_vol * 1.25 else -0.10
    return reward


def build_matching(df: pd.DataFrame, z_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategies = df[df["entity_type"] == "strategy"].copy()
    clients = df[df["entity_type"].isin(["real_client", "synthetic_client"])].copy()

    score_rows = []
    melted_rows = []
    for _, client in clients.iterrows():
        row_scores = {"client": client["entity_name"], "client_type": client["entity_type"]}
        for _, strategy in strategies.iterrows():
            beh = cosine_similarity(
                z_df.loc[client.name, BEHAVIOR_FEATURES].to_numpy(),
                z_df.loc[strategy.name, BEHAVIOR_FEATURES].to_numpy(),
            )
            hold = cosine_similarity(
                z_df.loc[client.name, HOLDING_FEATURES].to_numpy(),
                z_df.loc[strategy.name, HOLDING_FEATURES].to_numpy(),
            )
            style_score = 0.45 * beh + 0.55 * hold
            perf_score = performance_advantage(client, strategy)
            final_score = 0.7 * style_score + 0.3 * perf_score
            row_scores[strategy["entity_name"]] = final_score
            melted_rows.append(
                {
                    "client": client["entity_name"],
                    "client_type": client["entity_type"],
                    "strategy": strategy["entity_name"],
                    "behavior_similarity": beh,
                    "holding_similarity": hold,
                    "style_score": style_score,
                    "performance_score": perf_score,
                    "final_score": final_score,
                }
            )
        score_rows.append(row_scores)

    matrix = pd.DataFrame(score_rows)
    ranking = pd.DataFrame(melted_rows)
    ranking["rank"] = ranking.groupby("client")["final_score"].rank(method="first", ascending=False).astype(int)
    ranking = ranking.sort_values(["client", "rank"]).reset_index(drop=True)
    ranking = ranking.merge(
        df[["entity_name", "avg_return_pct", "max_drawdown", "sharpe_approx"]].rename(
            columns={
                "entity_name": "strategy",
                "avg_return_pct": "strategy_avg_return_pct",
                "max_drawdown": "strategy_max_drawdown",
                "sharpe_approx": "strategy_sharpe_approx",
            }
        ),
        on="strategy",
        how="left",
    )

    client_metrics = df[["entity_name", "avg_return_pct", "max_drawdown", "sharpe_approx"]].rename(
        columns={
            "entity_name": "client",
            "avg_return_pct": "client_avg_return_pct",
            "max_drawdown": "client_max_drawdown",
            "sharpe_approx": "client_sharpe_approx",
        }
    )
    ranking = ranking.merge(client_metrics, on="client", how="left")
    ranking["return_improvement"] = ranking["strategy_avg_return_pct"] - ranking["client_avg_return_pct"]
    ranking["drawdown_improvement"] = abs(ranking["client_max_drawdown"]) - abs(ranking["strategy_max_drawdown"])

    return matrix, ranking


def save_fig_feature_corr(df: pd.DataFrame):
    corr = df[ALL_FEATURES].corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, linewidths=0.3)
    plt.title("特征相关矩阵")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_feature_correlation.png", dpi=220)
    plt.close()


def save_fig_pca(pca_df: pd.DataFrame, pca: PCA):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    explained = pca.explained_variance_ratio_
    cum = np.cumsum(explained)
    axes[0].bar(range(1, len(explained) + 1), explained, color="#4C78A8")
    axes[0].plot(range(1, len(explained) + 1), cum, color="#E45756", marker="o")
    axes[0].axhline(0.8, linestyle="--", color="gray")
    axes[0].set_title("PCA 方差解释")
    axes[0].set_xlabel("主成分")
    axes[0].set_ylabel("解释比例")

    palette = {"strategy": "#1f77b4", "real_client": "#d62728", "synthetic_client": "#2ca02c"}
    markers = {"strategy": "o", "real_client": "s", "synthetic_client": "^"}
    for entity_type, sub in pca_df.groupby("entity_type"):
        axes[1].scatter(sub["PC1"], sub["PC2"], label=entity_type, color=palette[entity_type], marker=markers[entity_type], s=80, alpha=0.8)
    label_mask = pca_df["entity_type"].isin(["strategy", "real_client"])
    for row in pca_df[label_mask].itertuples(index=False):
        axes[1].text(row.PC1, row.PC2, row.entity_name[:8], fontsize=7, alpha=0.75)
    axes[1].set_title("PCA 二维投影")
    axes[1].set_xlabel(f"PC1 ({explained[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({explained[1]:.1%})")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_pca_summary.png", dpi=220)
    plt.close()

    loadings = pd.DataFrame(pca.components_[:2].T, index=ALL_FEATURES, columns=["PC1", "PC2"])
    plt.figure(figsize=(10, 10))
    plt.axhline(0, color="gray", linewidth=0.6)
    plt.axvline(0, color="gray", linewidth=0.6)
    for feature, values in loadings.iterrows():
        plt.arrow(0, 0, values["PC1"], values["PC2"], color="#E45756", alpha=0.8, head_width=0.02, length_includes_head=True)
        plt.text(values["PC1"] * 1.08, values["PC2"] * 1.08, feature, fontsize=9)
    circle = plt.Circle((0, 0), 1, fill=False, linestyle="--", alpha=0.4)
    plt.gca().add_patch(circle)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("PCA 载荷图")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_pca_loadings.png", dpi=220)
    plt.close()


def save_fig_k_selection(k_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(k_df["k"], k_df["inertia"], marker="o")
    axes[0].set_title("K 选择: Inertia")
    axes[1].plot(k_df["k"], k_df["silhouette"], marker="o")
    axes[1].set_title("K 选择: Silhouette")
    axes[2].plot(k_df["k"], k_df["davies_bouldin"], marker="o")
    axes[2].set_title("K 选择: Davies-Bouldin")
    for ax in axes:
        ax.set_xlabel("K")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_k_selection.png", dpi=220)
    plt.close()


def save_fig_dendrogram(x: np.ndarray, labels: list[str]):
    plt.figure(figsize=(18, 7))
    dendrogram(linkage(x, method="ward"), labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("层次聚类树状图")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_dendrogram.png", dpi=220)
    plt.close()


def save_fig_cluster_agreement(cluster_df: pd.DataFrame):
    methods = ["kmeans", "hierarchical", "gmm"]
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            matrix.loc[m1, m2] = adjusted_rand_score(cluster_df[m1], cluster_df[m2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("聚类方法一致性（ARI）")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_cluster_agreement.png", dpi=220)
    plt.close()


def save_fig_lda(lda_df: pd.DataFrame):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=lda_df, x="LD1", y="LD2", hue="cluster", style="entity_type", s=110)
    for row in lda_df.itertuples(index=False):
        plt.text(row.LD1, row.LD2, row.entity_name[:8], fontsize=7, alpha=0.75)
    plt.title("LDA 判别投影")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_lda_projection.png", dpi=220)
    plt.close()


def save_fig_matching_heatmap(score_matrix: pd.DataFrame):
    heat = score_matrix.set_index("client").drop(columns=["client_type"])
    plt.figure(figsize=(18, 11))
    sns.heatmap(heat, cmap="YlOrRd")
    plt.title("客户-策略匹配得分热力图")
    plt.xticks(rotation=55, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_matching_heatmap.png", dpi=220)
    plt.close()


def save_fig_real_client_heatmap(score_matrix: pd.DataFrame):
    heat = score_matrix[score_matrix["client_type"] == "real_client"].set_index("client").drop(columns=["client_type"])
    plt.figure(figsize=(14, 4.8))
    sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.4, cbar_kws={"label": "综合得分"})
    plt.title("真实客户-策略匹配得分")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "14_real_client_matching_heatmap.png", dpi=220)
    plt.close()


def save_fig_real_client_recs(top_real: pd.DataFrame):
    plot_df = top_real[top_real["rank"] <= 5].copy()
    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_df, x="client", y="final_score", hue="strategy")
    plt.title("真实客户 Top5 推荐得分")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_real_client_top5.png", dpi=220)
    plt.close()


def save_fig_real_client_improvement(top_real: pd.DataFrame):
    plot_df = top_real[top_real["rank"] <= 5].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=plot_df, x="client", y="return_improvement", hue="strategy", ax=axes[0])
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("真实客户 Top5 推荐收益改善")
    axes[0].set_ylabel("策略收益率 - 客户收益率")
    axes[0].legend(fontsize=8, loc="best")

    sns.barplot(data=plot_df, x="client", y="drawdown_improvement", hue="strategy", ax=axes[1])
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("真实客户 Top5 推荐回撤改善")
    axes[1].set_ylabel("|客户回撤| - |策略回撤|")
    axes[1].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_real_client_improvements.png", dpi=220)
    plt.close()


def save_fig_strategy_risk_return(df: pd.DataFrame):
    strategy_df = df[df["entity_type"] == "strategy"].copy()
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=strategy_df,
        x="max_drawdown",
        y="avg_return_pct",
        hue="cluster",
        size="sharpe_approx",
        sizes=(80, 320),
        palette="viridis",
        alpha=0.85,
    )
    for row in strategy_df.itertuples(index=False):
        plt.text(row.max_drawdown, row.avg_return_pct, row.entity_name[:7], fontsize=8, alpha=0.8)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("策略风险收益分布")
    plt.xlabel("最大回撤")
    plt.ylabel("2025累计收益率")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_strategy_risk_return.png", dpi=220)
    plt.close()


def save_fig_cluster_box(df: pd.DataFrame, feature: str = "avg_hold_days"):
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df[df["entity_type"] == "strategy"], x="cluster", y=feature, color="#72B7B2")
    plt.title(f"策略簇关键特征箱线图: {feature}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_cluster_boxplot.png", dpi=220)
    plt.close()


def save_fig_style_perf_scatter(ranking: pd.DataFrame):
    top_rank = ranking[ranking["rank"] == 1].copy()
    top_rank["style_like_proxy"] = min_max_scale(top_rank["final_score"])
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=top_rank,
        x="style_like_proxy",
        y="return_improvement",
        hue="client_type",
        s=110,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("推荐结果的风格-收益双维度表现")
    plt.xlabel("归一化综合匹配得分")
    plt.ylabel("相对客户收益改善")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_style_performance_scatter.png", dpi=220)
    plt.close()


def main():
    df, strategy_df, real_client_df = build_full_dataset()

    scaler = StandardScaler()
    x = scaler.fit_transform(df[ALL_FEATURES])
    z_df = pd.DataFrame(x, columns=ALL_FEATURES, index=df.index)

    save_fig_feature_corr(df)

    pca = PCA(n_components=min(len(ALL_FEATURES), 6), random_state=42)
    x_pca = pca.fit_transform(x)
    pca_explained = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    pca_explained.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False, encoding="utf-8-sig")
    pca_df = df[["entity_name", "entity_type"]].copy()
    pca_df["PC1"] = x_pca[:, 0]
    pca_df["PC2"] = x_pca[:, 1]
    save_fig_pca(pca_df, pca)

    k_df = choose_k(x)
    k_df.to_csv(TABLE_DIR / "k_selection_metrics.csv", index=False, encoding="utf-8-sig")
    save_fig_k_selection(k_df)

    best_k = int(k_df.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]["k"])
    kmeans = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    gmm = GaussianMixture(n_components=best_k, random_state=42)

    df["kmeans"] = kmeans.fit_predict(x)
    df["hierarchical"] = hier.fit_predict(x)
    df["gmm"] = gmm.fit_predict(x)
    df["cluster"] = df["kmeans"]

    save_fig_dendrogram(x, df["entity_name"].tolist())
    save_fig_cluster_agreement(df[["kmeans", "hierarchical", "gmm"]])

    lda_components = min(2, df["cluster"].nunique() - 1)
    lda = LinearDiscriminantAnalysis(n_components=lda_components)
    x_lda = lda.fit_transform(x, df["cluster"])
    lda_df = df[["entity_name", "entity_type", "cluster"]].copy()
    lda_df["LD1"] = x_lda[:, 0]
    lda_df["LD2"] = x_lda[:, 1] if x_lda.shape[1] > 1 else 0.0
    save_fig_lda(lda_df)

    score_matrix, ranking = build_matching(df, z_df)
    score_matrix.to_csv(TABLE_DIR / "matching_score_matrix.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(TABLE_DIR / "matching_ranking_full.csv", index=False, encoding="utf-8-sig")
    save_fig_matching_heatmap(score_matrix)
    save_fig_real_client_heatmap(score_matrix)

    top_real = ranking[ranking["client_type"] == "real_client"].copy()
    top_real_5 = top_real[top_real["rank"] <= 5].copy()
    top_real_5.to_csv(TABLE_DIR / "real_client_top5_recommendations.csv", index=False, encoding="utf-8-sig")
    top_real[top_real["rank"] == 1].to_csv(TABLE_DIR / "real_client_top1_summary.csv", index=False, encoding="utf-8-sig")
    save_fig_real_client_recs(top_real)
    save_fig_real_client_improvement(top_real)

    save_fig_cluster_box(df, "avg_hold_days")
    save_fig_style_perf_scatter(ranking)
    save_fig_strategy_risk_return(df)

    kruskal_rows = []
    strat_df = df[df["entity_type"] == "strategy"].copy()
    for feature in STYLE_FEATURES + PERFORMANCE_FEATURES:
        groups = [group[feature].values for _, group in strat_df.groupby("cluster")]
        if len(groups) >= 2 and all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            kruskal_rows.append({"feature": feature, "statistic": stat, "p_value": p})
    kruskal_df = pd.DataFrame(kruskal_rows).sort_values("p_value")
    kruskal_df.to_csv(TABLE_DIR / "kruskal_cluster_tests.csv", index=False, encoding="utf-8-sig")

    cluster_profile = strat_df.groupby("cluster")[ALL_FEATURES].mean()
    cluster_profile.to_csv(TABLE_DIR / "strategy_cluster_profiles.csv", encoding="utf-8-sig")

    top1 = ranking[ranking["rank"] == 1].copy()
    sufficiency = {
        "best_k": best_k,
        "cluster_silhouette": float(silhouette_score(x, df["cluster"])),
        "kmeans_vs_hier_ari": float(adjusted_rand_score(df["kmeans"], df["hierarchical"])),
        "kmeans_vs_gmm_ari": float(adjusted_rand_score(df["kmeans"], df["gmm"])),
        "real_client_positive_return_improvement_ratio": float(
            (top1[top1["client_type"] == "real_client"]["return_improvement"] > 0).mean()
        ),
        "real_client_positive_drawdown_improvement_ratio": float(
            (top1[top1["client_type"] == "real_client"]["drawdown_improvement"] > 0).mean()
        ),
    }
    with open(TABLE_DIR / "analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(sufficiency, fh, ensure_ascii=False, indent=2)

    df.to_csv(TABLE_DIR / "full_dataset_with_clusters.csv", index=False, encoding="utf-8-sig")
    lda_df.to_csv(TABLE_DIR / "lda_projection.csv", index=False, encoding="utf-8-sig")
    pca_df.to_csv(TABLE_DIR / "pca_projection.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(sufficiency, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
