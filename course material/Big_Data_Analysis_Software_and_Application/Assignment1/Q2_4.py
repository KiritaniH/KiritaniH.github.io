import pandas as pd

df = pd.read_csv("Q2_final.csv")

df["cdr3_len"] = df["cdr3_aa"].str.len()

top3_v = (
    df["v_call"]
    .value_counts()
    .head(3)
    .index
    .tolist()
)

df_top3 = df[df["v_call"].isin(top3_v)].copy()

pair_counts = (
    df_top3.groupby(["v_call", "j_call"])
    .size()
    .reset_index(name="pair_count")
)

idx = pair_counts.groupby("v_call")["pair_count"].idxmax()
top_pairs = pair_counts.loc[idx].copy()

avg_len = (
    df_top3.groupby(["v_call", "j_call"])["cdr3_len"]
    .mean()
    .reset_index(name="avg_cdr3_aa_length")
)

result = pd.merge(top_pairs, avg_len, on=["v_call", "j_call"], how="left")

result = result.sort_values(by="pair_count", ascending=False)

print(result)
result.to_csv("Q2_4_result.csv", index=False)