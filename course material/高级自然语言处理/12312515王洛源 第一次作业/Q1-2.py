import json
from tqdm import tqdm
import os
input_path = "industry_instruction_semantic_cluster_dedup_科技_科学研究_valid_train.jsonl"
text_path = "train_text.txt"
bpe_codes_path = "bpe_codes.txt"
tokenized_path = "tokenized_text.txt"
vocab_size = 10000

def extract_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Extracting text"):
            try:
                data = json.loads(line)
                texts = []
                
                # 提取 instruction
                if "instruction" in data and isinstance(data["instruction"],str):
                    texts.append(data["instruction"].replace("\n"," "))
                # 提取 conversations
                if "conversations" in data and isinstance(data["conversations"], list):
                    for item in data["conversations"]:
                        if isinstance(item,dict) and "value" in item and isinstance(item["value"],str):
                            texts.append(item["value"].replace("\n"," "))
                if texts:
                    fout.write(" ".join(texts) + "\n")
            except json.JSONDecodeError:
                continue

extract_text(input_path, text_path)

print(f"training text has been saved as {text_path}")

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

with open(text_path, 'r', encoding='utf-8') as fin, \
     open(bpe_codes_path, 'w', encoding='utf-8') as fout:
    learn_bpe(fin, fout, num_symbols=vocab_size)

print(f"BPE model training has finished，relevant information has been saved as {bpe_codes_path}")

with open(bpe_codes_path, 'r', encoding='utf-8') as codes:
    bpe = BPE(codes)

with open(text_path, 'r', encoding='utf-8') as fin, \
     open(tokenized_path, 'w', encoding='utf-8') as fout:
    for line in tqdm(fin, desc="Applying BPE"):
        tokenized_line = bpe.process_line(line.strip())
        fout.write(tokenized_line + "\n")

print(f"BPE result has been saved as {tokenized_path}")
