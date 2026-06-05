import pandas as pd

df = pd.read_csv("Q2_final.csv")

df["seq_len"] = df["sequence"].str.len()

group_counts = (
    df.groupby(["v_call", "j_call"])
      .size()
      .reset_index(name="occurrence_count")
)

idx = df.groupby(["v_call", "j_call"])["seq_len"].idxmax()

longest_seq = (
    df.loc[idx, ["v_call", "j_call", "sequence", "seq_len"]]
      .rename(columns={
          "sequence": "longest_sequence",
          "seq_len": "longest_sequence_length"
      })
)

result = pd.merge(group_counts, longest_seq, on=["v_call", "j_call"], how="inner")

result = result.sort_values(by="occurrence_count", ascending=False)

print(result.head())

result.to_csv("Q2_1_result.csv", index=False)
