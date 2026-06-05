import os
import glob
import pandas as pd

input_folder = "q2_raw"
output_folder = "q2_clean" 
final_output_file = "Q2_final.csv" 

os.makedirs(output_folder, exist_ok=True)

required_columns = ["sequence", "v_call", "j_call", "cdr3_aa"]
valid_cdr3_pattern = r"^[ACDEFGHIKLMNPQRSTVWY]+$"

all_clean_dfs = []

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

if not csv_files:
    print(f"No CSV files found in folder: {input_folder}")
    exit()

print(f"Found {len(csv_files)} CSV files.")

for file_path in csv_files:
    file_name = os.path.basename(file_path)
    print(f"\nProcessing: {file_name}")

    try:
        df = pd.read_csv(file_path, skiprows=1)

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Skip {file_name}: missing columns {missing_cols}")
            continue

        df = df[required_columns].copy()

        for col in required_columns:
            df[col] = df[col].astype(str)
        df = df.replace(["nan", "None", "NULL", "NA", ""], pd.NA)
        df = df[df["v_call"].notna()]
        df = df[df["cdr3_aa"].notna()]
        df = df[df["sequence"].notna()]
        df = df[df["j_call"].notna()]

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

        clean_file_name = file_name.replace(".csv", "_clean.csv")
        clean_file_path = os.path.join(output_folder, clean_file_name)
        df.to_csv(clean_file_path, index=False)

        print(f"Valid rows in {file_name}: {len(df)}")
        print(f"Saved cleaned file to: {clean_file_path}")

        all_clean_dfs.append(df)

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

if all_clean_dfs:
    final_df = pd.concat(all_clean_dfs, ignore_index=True)
    final_df.to_csv(final_output_file, index=False)

    print("\nAll files processed successfully.")
    print(f"Total valid rows in final dataset: {len(final_df)}")
    print(f"Final merged file saved to: {final_output_file}")
    print("\nFirst 5 rows of final dataset:")
    print(final_df.head())
else:
    print("\nNo valid data found. Final file was not created.")