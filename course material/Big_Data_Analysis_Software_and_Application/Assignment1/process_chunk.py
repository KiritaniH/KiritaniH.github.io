import sys
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python process_chunk.py input_chunk.csv output_clean.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

required_columns = ["sequence", "v_call", "j_call", "cdr3_aa"]
valid_cdr3_pattern = r"^[ACDEFGHIKLMNPQRSTVWY]+$"

df = pd.read_csv(input_file)

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Missing columns in {input_file}: {missing_cols}")
    sys.exit(1)

df = df[required_columns].copy()

for col in required_columns:
    df[col] = df[col].astype(str)

df = df.replace(["nan", "None", "NULL", "NA", ""], pd.NA)

df = df[df["sequence"].notna()]
df = df[df["v_call"].notna()]
df = df[df["j_call"].notna()]
df = df[df["cdr3_aa"].notna()]

df["sequence"] = df["sequence"].str.strip()
df["v_call"] = df["v_call"].str.strip()
df["j_call"] = df["j_call"].str.strip()
df["cdr3_aa"] = df["cdr3_aa"].str.strip()

df = df[
    (df["sequence"] != "") &
    (df["v_call"] != "") &
    (df["j_call"] != "") &
    (df["cdr3_aa"] != "")
]

df = df[df["cdr3_aa"].str.len().between(8, 30)]

df = df[df["cdr3_aa"].str.match(valid_cdr3_pattern, na=False)]

df.to_csv(output_file, index=False)
print(f"Processed {input_file} -> {output_file}, rows={len(df)}")